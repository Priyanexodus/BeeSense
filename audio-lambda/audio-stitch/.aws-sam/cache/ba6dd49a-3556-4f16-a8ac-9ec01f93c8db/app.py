import json
import boto3
import os
import base64
import time
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

CHUNKS_TABLE = os.environ['CHUNKS_TABLE']
BEESENSE_BUCKET = os.environ['BEESENSE_BUCKET']
RAW_AUDIO_PREFIX = os.environ['RAW_AUDIO_PREFIX']

table = dynamodb.Table(CHUNKS_TABLE)

def lambda_handler(event, context):
    """
    Receives audio chunks from ESP32 via IoT Core
    Expected payload structure:
    {
        "session_id": "device_id_timestamp",
        "chunk_index": 0,
        "total_chunks": 100,
        "audio_data": "base64_encoded_audio_samples",
        "sample_rate": 16000,
        "bits_per_sample": 16,
        "channels": 1,
        "device_id": "esp32_001",
        "timestamp": 1234567890
    }
    """
    try:
        print(f"Received event: {json.dumps(event)}")
        
        # Extract chunk data
        session_id = event['session_id']
        chunk_index = int(event['chunk_index'])
        total_chunks = int(event['total_chunks'])
        audio_data_base64 = event['audio_data']
        sample_rate = event.get('sample_rate', 16000)
        bits_per_sample = event.get('bits_per_sample', 16)
        channels = event.get('channels', 1)
        device_id = event.get('device_id', 'unknown')
        timestamp = event.get('timestamp', int(time.time()))
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data_base64)
        chunk_size = len(audio_bytes)
        
        # Store raw chunk in S3 for reliability
        chunk_key = f"{RAW_AUDIO_PREFIX}{session_id}/chunk_{chunk_index:04d}.raw"
        s3_client.put_object(
            Bucket=BEESENSE_BUCKET,
            Key=chunk_key,
            Body=audio_bytes,
            ContentType='application/octet-stream'
        )
        
        # Store chunk metadata in DynamoDB
        ttl = int(time.time()) + 3600  # Expire after 1 hour
        
        table.put_item(
            Item={
                'session_id': session_id,
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                's3_key': chunk_key,
                'chunk_size': chunk_size,
                'sample_rate': sample_rate,
                'bits_per_sample': bits_per_sample,
                'channels': channels,
                'device_id': device_id,
                'timestamp': timestamp,
                'status': 'received',
                'received_at': Decimal(str(time.time())),
                'ttl': ttl
            }
        )
        
        print(f"Successfully stored chunk {chunk_index}/{total_chunks} for session {session_id}")
        
        # Check if this is the last chunk
        if chunk_index == total_chunks - 1:
            print(f"Last chunk received for session {session_id}, checking completion...")
            check_session_completion(session_id, total_chunks)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Chunk received successfully',
                'session_id': session_id,
                'chunk_index': chunk_index,
                's3_key': chunk_key
            })
        }
        
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def check_session_completion(session_id, expected_chunks):
    """
    Check if all chunks for a session have been received
    """
    try:
        response = table.query(
            KeyConditionExpression='session_id = :sid',
            ExpressionAttributeValues={
                ':sid': session_id
            }
        )
        
        received_chunks = len(response['Items'])
        
        print(f"Session {session_id}: {received_chunks}/{expected_chunks} chunks received")
        
        if received_chunks == expected_chunks:
            # Mark session as complete
            table.update_item(
                Key={
                    'session_id': session_id,
                    'chunk_index': 0  # Use first chunk as session metadata record
                },
                UpdateExpression='SET #status = :status',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':status': 'ready_to_stitch'
                }
            )
            print(f"Session {session_id} marked as ready to stitch")
            
    except Exception as e:
        print(f"Error checking session completion: {str(e)}")