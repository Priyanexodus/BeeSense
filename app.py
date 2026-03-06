import json
import torch
import numpy as np
import librosa
import joblib
import tempfile
import os
import boto3
from model import HybridBeeModel

# Global variables for model loading (outside handler for reuse)
MODEL = None
SCALER = None
LABEL_ENCODER = None
DEVICE = None
S3_CLIENT = None

def load_model():
    """Load model once and reuse across invocations"""
    global MODEL, SCALER, LABEL_ENCODER, DEVICE, S3_CLIENT
    
    if MODEL is None:
        DEVICE = torch.device("cpu")  # Lambda uses CPU
        
        # Load preprocessors
        SCALER = joblib.load('artifacts/telemetry_scaler.joblib')
        LABEL_ENCODER = joblib.load('artifacts/label_encoder.joblib')
        
        # Load model
        num_classes = len(LABEL_ENCODER.classes_)
        MODEL = HybridBeeModel(num_classes).to(DEVICE)
        MODEL.load_state_dict(torch.load('artifacts/bee_health_model.pth', map_location=DEVICE))
        MODEL.eval()
        
        # Initialize S3 client
        S3_CLIENT = boto3.client('s3')
    
    return MODEL, SCALER, LABEL_ENCODER, DEVICE, S3_CLIENT

def download_audio_from_s3(s3_uri):
    """
    Download audio file from S3
    
    Args:
        s3_uri: S3 URI like 's3://bucket-name/path/to/file.wav'
                or dict like {'bucket': 'bucket-name', 'key': 'path/to/file.wav'}
    
    Returns:
        Path to downloaded temporary file
    """
    _, _, _, _, s3_client = load_model()
    
    # Parse S3 URI
    if isinstance(s3_uri, str):
        if s3_uri.startswith('s3://'):
            s3_uri = s3_uri[5:]
        parts = s3_uri.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        bucket = s3_uri['bucket']
        key = s3_uri['key']
    
    # Download to temp file
    tmp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    s3_client.download_file(bucket, key, tmp_path)
    
    return tmp_path

def preprocess_audio(audio_path):
    """Convert audio file to spectrogram tensor"""
    CONF = {
        'SAMPLE_RATE': 22050,
        'DURATION': 5,
        'N_MELS': 64,
        'HOP_LEN': 512,
        'N_FFT': 2048,
        'FIXED_WIDTH': 216
    }
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=CONF['SAMPLE_RATE'], duration=CONF['DURATION'])
    
    # Pad/Crop to 5 seconds
    target_len = CONF['SAMPLE_RATE'] * CONF['DURATION']
    y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
    
    # Create Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        n_mels=CONF['N_MELS'], 
        hop_length=CONF['HOP_LEN'], 
        n_fft=CONF['N_FFT']
    )
    
    # Normalize
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = log_mel - np.mean(log_mel)
    
    # Fix width to 216
    if log_mel.shape[1] < CONF['FIXED_WIDTH']:
        log_mel = np.pad(log_mel, ((0, 0), (0, CONF['FIXED_WIDTH'] - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :CONF['FIXED_WIDTH']]
    
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def lambda_handler(event, context):
    """
    Lambda handler for bee health prediction
    
    Expected input:
    {
        "s3_uri": "s3://bucket-name/path/to/audio.wav",
        OR
        "s3_path": {
            "bucket": "bucket-name",
            "key": "path/to/audio.wav"
        },
        "telemetry": {
            "hive_temp": 29.84,
            "hive_humidity": 43.53,
            "hive_pressure": 1007.27,
            "weather_temp": 24.15,
            "weather_humidity": 68,
            "weather_pressure": 1013,
            "rain": 0,
            "time": 0.708
        }
    }
    """
    tmp_audio_path = None
    
    try:
        # Load model
        model, scaler, label_encoder, device, _ = load_model()
        
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        # Download audio from S3
        s3_uri = body.get('s3_uri') or body.get('s3_path')
        if not s3_uri:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing s3_uri or s3_path in request'
                })
            }
        
        tmp_audio_path = download_audio_from_s3(s3_uri)
        
        # Preprocess audio
        audio_tensor = preprocess_audio(tmp_audio_path).to(device)
        
        # Prepare telemetry
        tele = body['telemetry']
        tele_array = np.array([[
            tele['hive_temp'],
            tele['hive_humidity'],
            tele['hive_pressure'],
            tele['weather_temp'],
            tele['weather_humidity'],
            tele['weather_pressure'],
            tele['rain'],
            tele['time']
        ]])
        tele_scaled = scaler.transform(tele_array)
        tele_tensor = torch.tensor(tele_scaled, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(audio_tensor, tele_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Convert to label
        pred_label = label_encoder.inverse_transform([predicted_class.item()])[0]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': str(pred_label),
                'confidence': float(confidence.item()),
                'all_probabilities': {
                    str(label): float(prob)
                    for label, prob in zip(label_encoder.classes_, probabilities[0].cpu().numpy())
                }
            })
        }
    
    except Exception as e:
        import traceback
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        }
    
    finally:
        # Cleanup temp file
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)