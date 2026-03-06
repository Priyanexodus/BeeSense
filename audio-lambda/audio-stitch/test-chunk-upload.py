#!/usr/bin/env python3
"""
Test script for BeeSense audio chunk upload
Simulates ESP32 sending audio chunks
"""

import json
import base64
import time
import requests
import numpy as np

# Configuration
API_ENDPOINT = "https://your-api-id.execute-api.region.amazonaws.com/prod"
DEVICE_ID = "esp32_test_001"
SAMPLE_RATE = 16000
BITS_PER_SAMPLE = 16
CHANNELS = 1
CHUNK_SIZE = 4096  # bytes
RECORDING_DURATION = 5  # seconds

def generate_test_audio(duration, sample_rate):
    """
    Generate test audio signal (sine wave)
    """
    samples = int(duration * sample_rate)
    frequency = 440  # Hz (A note)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

def send_chunk(session_id, chunk_index, total_chunks, audio_data):
    """
    Send a single audio chunk to the API
    """
    # Encode to base64
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    payload = {
        "session_id": session_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "audio_data": audio_base64,
        "sample_rate": SAMPLE_RATE,
        "bits_per_sample": BITS_PER_SAMPLE,
        "channels": CHANNELS,
        "device_id": DEVICE_ID,
        "timestamp": int(time.time())
    }
    
    response = requests.post(
        f"{API_ENDPOINT}/audio/chunk",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response

def send_completion(session_id, total_chunks):
    """
    Send completion signal
    """
    payload = {
        "session_id": session_id,
        "device_id": DEVICE_ID,
        "total_chunks_sent": total_chunks
    }
    
    response = requests.post(
        f"{API_ENDPOINT}/audio/complete",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response

def main():
    """
    Main test function
    """
    print("=" * 60)
    print("BeeSense Audio Chunk Upload Test")
    print("=" * 60)
    
    # Generate test audio
    print(f"\n📊 Generating {RECORDING_DURATION}s test audio...")
    audio_data = generate_test_audio(RECORDING_DURATION, SAMPLE_RATE)
    total_bytes = len(audio_data)
    print(f"✓ Generated {total_bytes} bytes of audio data")
    
    # Calculate chunks
    total_chunks = (total_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE
    session_id = f"{DEVICE_ID}_{int(time.time())}"
    
    print(f"\n📦 Session: {session_id}")
    print(f"📦 Total chunks: {total_chunks}")
    print(f"📦 Chunk size: {CHUNK_SIZE} bytes")
    
    # Send chunks
    print(f"\n📤 Sending chunks...")
    success_count = 0
    
    for i in range(total_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, total_bytes)
        chunk_data = audio_data[start_idx:end_idx]
        
        try:
            response = send_chunk(session_id, i, total_chunks, chunk_data)
            
            if response.status_code == 200:
                success_count += 1
                print(f"✓ Chunk {i + 1}/{total_chunks} sent successfully")
            else:
                print(f"✗ Chunk {i + 1}/{total_chunks} failed: {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"✗ Chunk {i + 1}/{total_chunks} error: {str(e)}")
        
        # Small delay between chunks
        time.sleep(0.1)
    
    # Send completion signal
    print(f"\n📬 Sending completion signal...")
    try:
        response = send_completion(session_id, total_chunks)
        
        if response.status_code == 200:
            print(f"✓ Completion signal sent successfully")
            result = response.json()
            print(f"  Response: {json.dumps(result, indent=2)}")
        else:
            print(f"✗ Completion signal failed: {response.status_code}")
            print(f"  Response: {response.text}")
    
    except Exception as e:
        print(f"✗ Completion signal error: {str(e)}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Session ID: {session_id}")
    print(f"  Chunks sent: {success_count}/{total_chunks}")
    print(f"  Success rate: {success_count/total_chunks*100:.1f}%")
    print(f"\n💡 Check S3 bucket 'beesense' for the processed WAV file:")
    print(f"  processed-audio/{DEVICE_ID}/{session_id}.wav")
    print("=" * 60)

if __name__ == "__main__":
    main()