import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

CHUNKS_TABLE = os.environ['CHUNKS_TABLE']

table = dynamodb.Table(CHUNKS_TABLE)

def lambda_handler(event, context):
    """
    Handle session completion signal from ESP32
    This is triggered when ESP32 sends a completion message via IoT Core
    
    Expected payload:
    {
        "session_id": "device_id_timestamp",
        "device_id": "esp32_001",
        "total_chunks_sent": 100
    }
    """
    try:
        print(f"Received completion event: {json.dumps(event)}")
        
        session_id = event['session_id']
        device_id = event.get('device_id', 'unknown')
        expected_chunks = event.get('total_chunks_sent')
        
        print(f"Processing completion for session {session_id}")
        
        # Query all chunks for this session
        response = table.query(
            KeyConditionExpression='session_id = :sid',
            ExpressionAttributeValues={
                ':sid': session_id
            }
        )
        
        chunks = response['Items']
        received_chunks = len(chunks)
        
        print(f"Session {session_id}: {received_chunks} chunks found in database")
        
        if not chunks:
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'error': 'No chunks found for session',
                    'session_id': session_id
                })
            }
        
        # Get expected chunks from first chunk metadata
        if expected_chunks is None:
            expected_chunks = chunks[0].get('total_chunks')
        
        # Check if all chunks are present
        if received_chunks == expected_chunks:
            print(f"All chunks received ({received_chunks}/{expected_chunks})")
            
            # Mark session as ready to stitch
            table.update_item(
                Key={
                    'session_id': session_id,
                    'chunk_index': 0
                },
                UpdateExpression='SET #status = :status',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':status': 'ready_to_stitch'
                }
            )
            
            # Immediately invoke stitcher function
            stitcher_function = os.environ.get('STITCHER_FUNCTION', 'beesense-audio-stitcher')
            
            lambda_client.invoke(
                FunctionName=stitcher_function,
                InvocationType='Event',  # Async invocation
                Payload=json.dumps({
                    'session_id': session_id,
                    'trigger': 'completion_signal'
                })
            )
            
            print(f"Stitcher function invoked for session {session_id}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Session complete, stitching initiated',
                    'session_id': session_id,
                    'chunks_received': received_chunks
                })
            }
        else:
            # Not all chunks received yet
            print(f"Session incomplete: {received_chunks}/{expected_chunks} chunks")
            
            return {
                'statusCode': 202,
                'body': json.dumps({
                    'message': 'Session not yet complete',
                    'session_id': session_id,
                    'chunks_received': received_chunks,
                    'chunks_expected': expected_chunks,
                    'chunks_missing': expected_chunks - received_chunks
                })
            }
            
    except Exception as e:
        print(f"Error processing completion: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }