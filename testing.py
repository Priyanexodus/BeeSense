import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import librosa

class HybridBeeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridBeeModel, self).__init__()
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.audio_flat_size = 32 * 16 * 54

        # Expecting 3 features here
        self.telemetry_fc = nn.Sequential(nn.Linear(8, 16), nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(self.audio_flat_size + 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, audio, telemetry):
        x1 = self.audio_cnn(audio)
        x2 = self.telemetry_fc(telemetry)
        return self.classifier(torch.cat((x1, x2), dim=1))

class BeeInference:
    def __init__(self, artifacts_dir):
        # 1. Load Preprocessors
        self.CONF = {
            'SAMPLE_RATE': 22050,
            'DURATION': 5,
            'N_MELS': 64,
            'HOP_LEN': 512,
            'N_FFT': 2048,
            'FIXED_WIDTH': 216
        }
        self.scaler = joblib.load(f'{artifacts_dir}//telemetry_scaler.joblib')
        self.le = joblib.load(f'{artifacts_dir}//label_encoder.joblib')

        # 2. Setup Model
        num_classes = len(self.le.classes_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridBeeModel(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(f'{artifacts_dir}//bee_health_model.pth', map_location=self.device))
        self.model.eval() # Set to evaluation mode

    def preprocess_audio(self, audio_path):
        """Converts raw .wav to a normalized Log-Mel Spectrogram."""
        y, sr = librosa.load(audio_path, sr=self.CONF['SAMPLE_RATE'], duration=self.CONF['DURATION'])

        # Pad/Crop to 5 seconds
        target_len = self.CONF['SAMPLE_RATE'] * self.CONF['DURATION']
        y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]

        # Create Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.CONF['N_MELS'],
                                             hop_length=self.CONF['HOP_LEN'], n_fft=self.CONF['N_FFT'])

        # Normalize (Matches your notebook logic)
        log_mel = librosa.power_to_db(mel, ref=np.max) - np.mean(librosa.power_to_db(mel, ref=np.max))

        # Pad Time-Width to 216
        if log_mel.shape[1] < self.CONF['FIXED_WIDTH']:
            log_mel = np.pad(log_mel, ((0, 0), (0, self.CONF['FIXED_WIDTH'] - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :self.CONF['FIXED_WIDTH']]

        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def predict(self, data_dict):

        audio_path = data_dict["audio_path"]
        hive_temp = data_dict['hive temp']
        hive_humidity = data_dict['hive humidity']
        hive_pressure = data_dict['hive pressure']
        weather_temp = data_dict['weather temp']
        weather_humidity = data_dict['weather humidity']
        weather_pressure = data_dict['weather pressure']
        rain = data_dict["rain"]
        time = data_dict["time"]

        audio_tensor = self.preprocess_audio(audio_path)

        # Scale telemetry using the SAVED scaler
        tele_raw = np.array([[hive_temp, hive_humidity, hive_pressure ,weather_temp, weather_humidity, weather_pressure, rain, time]])
        tele_scaled = self.scaler.transform(tele_raw)
        tele_tensor = torch.tensor(tele_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(audio_tensor, tele_tensor)
            pred_id = torch.argmax(output, dim=1).item()

        # Convert numeric ID back to original label
        return self.le.inverse_transform([pred_id])[0]

test_data = {
    "audio_path" : "2022-06-05--17-41-01_2__segment0.wav",
    'hive temp':  29.84,
    'hive humidity': 43.53,
    'hive pressure':  1007.27,
    'weather temp': 24.15,
    'weather humidity':  68,
    'weather pressure':  1013,
    'rain':  0,
    'time': 0.708
}
engine = BeeInference("./artifacts")
print(engine.predict(test_data))