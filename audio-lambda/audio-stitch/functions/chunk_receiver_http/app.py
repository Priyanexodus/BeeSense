import json
import boto3
import os
import base64
import time
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

CHUNKS_TABLE = os.environ['CHUNKS_TABLE']
BEESENSE_BUCKET = os.environ['BEESENSE_BUCKET']
RAW_AUDIO_PREFIX = os.environ['RAW_AUDIO_PREFIX']

table = dynamodb.Table(CHUNKS_TABLE)

def lambda_handler(event, context):
    """
    HTTP endpoint for receiving audio chunks
    """
    try:
        # Parse the request
        path = event.get('path', '')
        http_method = event.get('httpMethod', '')
        
        if path == '/audio/chunk' and http_method == 'POST':
            return handle_chunk_upload(event)
        elif path == '/audio/complete' and http_method == 'POST':
            return handle_session_completion(event)
        else:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Not found'})
            }
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }

def handle_chunk_upload(event):
    """
    Handle audio chunk upload
    """
    body = json.loads(event['body'])
    
    session_id = body['session_id']
    chunk_index = int(body['chunk_index'])
    total_chunks = int(body['total_chunks'])
    audio_data_base64 = body['audio_data']
    sample_rate = body.get('sample_rate', 16000)
    bits_per_sample = body.get('bits_per_sample', 16)
    channels = body.get('channels', 1)
    device_id = body.get('device_id', 'unknown')
    timestamp = body.get('timestamp', int(time.time()))
    
    # Decode base64 audio data
    audio_bytes = base64.b64decode(audio_data_base64)
    chunk_size = len(audio_bytes)
    
    # Store raw chunk in S3
    chunk_key = f"{RAW_AUDIO_PREFIX}{session_id}/chunk_{chunk_index:04d}.raw"
    s3_client.put_object(
        Bucket=BEESENSE_BUCKET,
        Key=chunk_key,
        Body=audio_bytes,
        ContentType='application/octet-stream'
    )
    
    # Store chunk metadata in DynamoDB
    ttl = int(time.time()) + 3600
    
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
    
    print(f"Chunk {chunk_index}/{total_chunks} stored for session {session_id}")
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'message': 'Chunk received successfully',
            'session_id': session_id,
            'chunk_index': chunk_index,
            's3_key': chunk_key
        })
    }

def handle_session_completion(event):
    """
    Mark session as complete and trigger stitching
    """
    body = json.loads(event['body'])
    session_id = body['session_id']
    
    # Query all chunks for this session
    response = table.query(
        KeyConditionExpression='session_id = :sid',
        ExpressionAttributeValues={
            ':sid': session_id
        }
    )
    
    chunks = response['Items']
    if not chunks:
        return {
            'statusCode': 404,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Session not found'
            })
        }
    
    total_chunks = chunks[0]['total_chunks']
    received_chunks = len(chunks)
    
    if received_chunks != total_chunks:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Missing chunks: {received_chunks}/{total_chunks} received'
            })
        }
    
    # Trigger audio stitcher
    lambda_client.invoke(
        FunctionName=os.environ.get('STITCHER_FUNCTION', 'beesense-audio-stitcher'),
        InvocationType='Event',
        Payload=json.dumps({
            'session_id': session_id,
            'trigger': 'manual_completion'
        })
    )
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'message': 'Session marked as complete, stitching initiated',
            'session_id': session_id,
            'total_chunks': total_chunks
        })
    }