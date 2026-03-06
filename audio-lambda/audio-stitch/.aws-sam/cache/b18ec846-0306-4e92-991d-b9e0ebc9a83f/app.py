import json
import boto3
import os
import struct
import io
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

CHUNKS_TABLE = os.environ['CHUNKS_TABLE']
BEESENSE_BUCKET = os.environ['BEESENSE_BUCKET']
RAW_AUDIO_PREFIX = os.environ['RAW_AUDIO_PREFIX']
PROCESSED_AUDIO_PREFIX = os.environ['PROCESSED_AUDIO_PREFIX']

table = dynamodb.Table(CHUNKS_TABLE)

def lambda_handler(event, context):
    """
    Stitches audio chunks into a complete WAV file
    Can be triggered by:
    1. DynamoDB stream when session is marked as ready
    2. Manual invocation with session_id
    """
    try:
        print(f"Received event: {json.dumps(event)}")
        
        # Determine session_id from event source
        if 'Records' in event:
            # DynamoDB Stream trigger
            session_ids = extract_session_ids_from_stream(event)
            for session_id in session_ids:
                process_session(session_id)
        else:
            # Direct invocation
            session_id = event.get('session_id')
            if session_id:
                process_session(session_id)
            else:
                raise ValueError("No session_id provided")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Processing complete'})
        }
        
    except Exception as e:
        print(f"Error in audio stitcher: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def extract_session_ids_from_stream(event):
    """
    Extract session IDs from DynamoDB stream events that are ready to stitch
    """
    session_ids = set()
    
    for record in event['Records']:
        if record['eventName'] in ['INSERT', 'MODIFY']:
            new_image = record['dynamodb'].get('NewImage', {})
            status = new_image.get('status', {}).get('S', '')
            
            if status == 'ready_to_stitch':
                session_id = new_image.get('session_id', {}).get('S')
                if session_id:
                    session_ids.add(session_id)
    
    return list(session_ids)

def process_session(session_id):
    """
    Process a complete session: fetch chunks, stitch, create WAV
    """
    print(f"Processing session: {session_id}")
    
    # Query all chunks for this session
    response = table.query(
        KeyConditionExpression='session_id = :sid',
        ExpressionAttributeValues={
            ':sid': session_id
        }
    )
    
    chunks = response['Items']
    if not chunks:
        print(f"No chunks found for session {session_id}")
        return
    
    # Sort chunks by index
    chunks.sort(key=lambda x: x['chunk_index'])
    
    # Verify we have all chunks
    total_chunks = chunks[0]['total_chunks']
    if len(chunks) != total_chunks:
        print(f"Missing chunks for session {session_id}: {len(chunks)}/{total_chunks}")
        return
    
    # Get audio parameters from first chunk
    sample_rate = int(chunks[0].get('sample_rate', 16000))
    bits_per_sample = int(chunks[0].get('bits_per_sample', 16))
    channels = int(chunks[0].get('channels', 1))
    device_id = chunks[0].get('device_id', 'unknown')
    timestamp = chunks[0].get('timestamp', 0)
    
    print(f"Audio parameters: {sample_rate}Hz, {bits_per_sample}bit, {channels}ch")
    
    # Download and concatenate all audio chunks
    audio_data = bytearray()
    
    for chunk in chunks:
        try:
            s3_key = chunk['s3_key']
            print(f"Downloading chunk {chunk['chunk_index']}: {s3_key}")
            
            response = s3_client.get_object(
                Bucket=BEESENSE_BUCKET,
                Key=s3_key
            )
            
            chunk_data = response['Body'].read()
            audio_data.extend(chunk_data)
            
        except Exception as e:
            print(f"Error downloading chunk {chunk['chunk_index']}: {str(e)}")
            raise
    
    print(f"Total audio data size: {len(audio_data)} bytes")
    
    # Create WAV file with proper header
    wav_bytes = create_wav_file(
        audio_data,
        sample_rate,
        bits_per_sample,
        channels
    )
    
    # Generate output filename
    output_key = f"{PROCESSED_AUDIO_PREFIX}{device_id}/{session_id}.wav"
    
    # Upload WAV file to S3
    s3_client.put_object(
        Bucket=BEESENSE_BUCKET,
        Key=output_key,
        Body=wav_bytes,
        ContentType='audio/wav',
        Metadata={
            'session_id': session_id,
            'device_id': device_id,
            'sample_rate': str(sample_rate),
            'bits_per_sample': str(bits_per_sample),
            'channels': str(channels),
            'total_chunks': str(total_chunks),
            'timestamp': str(timestamp)
        }
    )
    
    print(f"WAV file created: s3://{BEESENSE_BUCKET}/{output_key}")
    
    # Update session status in DynamoDB
    table.update_item(
        Key={
            'session_id': session_id,
            'chunk_index': 0
        },
        UpdateExpression='SET #status = :status, wav_s3_key = :key, processed_at = :time',
        ExpressionAttributeNames={
            '#status': 'status'
        },
        ExpressionAttributeValues={
            ':status': 'completed',
            ':key': output_key,
            ':time': Decimal(str(time.time()))
        }
    )
    
    print(f"Session {session_id} processing completed")
    
    # Optionally: Clean up raw chunks from S3 after successful stitching
    # cleanup_raw_chunks(session_id, chunks)

def create_wav_file(audio_data, sample_rate, bits_per_sample, channels):
    """
    Create a proper WAV file with RIFF header
    
    WAV file structure:
    - RIFF header (12 bytes)
    - fmt chunk (24 bytes)
    - data chunk header (8 bytes)
    - audio data
    """
    
    # Calculate parameters
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(audio_data)
    file_size = 36 + data_size  # Total size minus 8 bytes for RIFF header
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    
    # RIFF header
    wav_buffer.write(b'RIFF')
    wav_buffer.write(struct.pack('<I', file_size))  # File size
    wav_buffer.write(b'WAVE')
    
    # fmt chunk
    wav_buffer.write(b'fmt ')
    wav_buffer.write(struct.pack('<I', 16))  # fmt chunk size (16 for PCM)
    wav_buffer.write(struct.pack('<H', 1))   # Audio format (1 = PCM)
    wav_buffer.write(struct.pack('<H', channels))  # Number of channels
    wav_buffer.write(struct.pack('<I', sample_rate))  # Sample rate
    wav_buffer.write(struct.pack('<I', byte_rate))  # Byte rate
    wav_buffer.write(struct.pack('<H', block_align))  # Block align
    wav_buffer.write(struct.pack('<H', bits_per_sample))  # Bits per sample
    
    # data chunk
    wav_buffer.write(b'data')
    wav_buffer.write(struct.pack('<I', data_size))  # Data size
    wav_buffer.write(audio_data)  # Audio data
    
    # Get the complete WAV file
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()
    
    print(f"WAV file created: {len(wav_bytes)} bytes "
          f"(Header: 44 bytes, Data: {data_size} bytes)")
    
    return wav_bytes

def cleanup_raw_chunks(session_id, chunks):
    """
    Optional: Delete raw chunk files after successful stitching
    """
    print(f"Cleaning up raw chunks for session {session_id}")
    
    for chunk in chunks:
        try:
            s3_client.delete_object(
                Bucket=BEESENSE_BUCKET,
                Key=chunk['s3_key']
            )
        except Exception as e:
            print(f"Error deleting chunk {chunk['s3_key']}: {str(e)}")
    
    print(f"Cleanup completed for session {session_id}")