#!/usr/bin/env python3
"""
Comprehensive test for both speech input and audio output functionality
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import time

def generate_simple_test_audio(duration_seconds=2, sample_rate=16000):
    """Generate simple test audio that should be easy to transcribe"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)

    # Generate a simple tone pattern that might be recognized as speech
    # Use a lower frequency that's more speech-like
    fundamental = 100  # Lower frequency

    # Simple sine wave with some variation
    audio = 0.3 * np.sin(2 * np.pi * fundamental * t)

    # Add some amplitude variation to simulate speech
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2Hz envelope
    audio = audio * envelope

    # Add minimal noise
    noise = np.random.normal(0, 0.01, audio.shape)
    audio = audio + noise

    # Ensure proper amplitude range
    audio = np.clip(audio, -0.5, 0.5)

    # Convert to float32
    return audio.astype(np.float32)

async def test_complete_pipeline():
    """Test the complete speech-to-speech pipeline"""
    uri = "ws://localhost:8000/ws"
    
    print("🧪 Testing Complete Speech-to-Speech Pipeline")
    print("=" * 50)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
            # Wait for connection message
            conn_msg = await websocket.recv()
            conn_data = json.loads(conn_msg)
            print(f"🔗 Connection: {conn_data.get('message', 'Connected')}")
            
            # Generate simple test audio
            print("🎤 Generating simple test audio...")
            speech_audio = generate_simple_test_audio(duration_seconds=2)
            audio_bytes = speech_audio.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send audio with streaming mode enabled
            message = {
                "type": "audio_chunk",
                "audio_data": audio_b64,
                "mode": "streaming",
                "streaming": True,
                "chunk_id": "pipeline_test_001",
                "timestamp": time.time()
            }
            
            print(f"📤 Sending speech audio ({len(speech_audio)} samples, {len(speech_audio)/16000:.1f}s)")
            await websocket.send(json.dumps(message))
            
            # Collect responses
            responses = []
            streaming_words_received = []
            streaming_audio_received = []
            text_responses = []
            audio_responses = []
            
            print("\n📥 Listening for responses...")
            
            try:
                for i in range(50):  # Listen for up to 50 messages
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)  # Increased timeout
                    data = json.loads(response)
                    responses.append(data)
                    
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'streaming_words':
                        text = data.get('text', '')
                        sequence = data.get('sequence', 0)
                        streaming_words_received.append((sequence, text))
                        print(f"📝 Streaming words #{sequence}: '{text}'")
                        
                    elif msg_type == 'streaming_audio':
                        chunk_index = data.get('chunk_index', 0)
                        audio_data_len = len(data.get('audio_data', ''))
                        is_final = data.get('is_final', False)
                        streaming_audio_received.append((chunk_index, audio_data_len, is_final))
                        print(f"🔊 Streaming audio #{chunk_index}: {audio_data_len} bytes (final: {is_final})")
                        
                    elif msg_type == 'response':
                        text = data.get('text', '')
                        text_responses.append(text)
                        print(f"💬 Text response: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                        
                    elif msg_type == 'audio_response':
                        audio_data_len = len(data.get('audio_data', ''))
                        chunk_id = data.get('chunk_id', 'unknown')
                        audio_responses.append((chunk_id, audio_data_len))
                        print(f"🎵 Audio response: {audio_data_len} bytes (chunk: {chunk_id})")
                        
                    elif msg_type == 'streaming_complete':
                        full_response = data.get('full_response', '')
                        total_words = data.get('total_words_sent', 0)
                        print(f"✅ Streaming complete: {total_words} words sent")
                        print(f"📄 Full response: '{full_response[:200]}{'...' if len(full_response) > 200 else ''}'")
                        break
                        
                    elif msg_type == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        print(f"❌ Error: {error_msg}")
                        break
                        
                    elif msg_type in ['connection', 'info']:
                        print(f"ℹ️  {msg_type}: {data.get('message', '')}")
                        
                    else:
                        print(f"❓ Unknown message type: {msg_type}")
                    
            except asyncio.TimeoutError:
                print("⏰ No more responses (timeout)")
            
            # Analyze results
            print("\n" + "=" * 50)
            print("📊 PIPELINE TEST RESULTS")
            print("=" * 50)
            
            print(f"📈 Total responses received: {len(responses)}")
            print(f"📝 Streaming words received: {len(streaming_words_received)}")
            print(f"🔊 Streaming audio chunks received: {len(streaming_audio_received)}")
            print(f"💬 Text responses received: {len(text_responses)}")
            print(f"🎵 Audio responses received: {len(audio_responses)}")
            
            # Check speech input functionality
            speech_input_working = len(streaming_words_received) > 0 or len(text_responses) > 0
            print(f"\n🎤 SPEECH INPUT: {'✅ WORKING' if speech_input_working else '❌ NOT WORKING'}")
            
            if streaming_words_received:
                print("   - Streaming words detected (real-time transcription)")
                total_words = ' '.join([word for _, word in sorted(streaming_words_received)])
                print(f"   - Transcribed text: '{total_words[:150]}{'...' if len(total_words) > 150 else ''}'")
            
            if text_responses:
                print("   - Text responses received")
            
            # Check audio output functionality  
            audio_output_working = len(streaming_audio_received) > 0 or len(audio_responses) > 0
            print(f"\n🔊 AUDIO OUTPUT: {'✅ WORKING' if audio_output_working else '❌ NOT WORKING'}")
            
            if streaming_audio_received:
                print("   - Streaming audio chunks detected (real-time TTS)")
                total_audio_bytes = sum([size for _, size, _ in streaming_audio_received])
                print(f"   - Total audio data: {total_audio_bytes} bytes")
                final_chunks = [chunk for chunk in streaming_audio_received if chunk[2]]  # is_final=True
                print(f"   - Final audio chunks: {len(final_chunks)}")
            
            if audio_responses:
                print("   - Audio responses received")
                total_audio_bytes = sum([size for _, size in audio_responses])
                print(f"   - Total audio data: {total_audio_bytes} bytes")
            
            # Overall assessment
            print(f"\n🎯 OVERALL PIPELINE: {'✅ WORKING' if speech_input_working and audio_output_working else '⚠️  PARTIAL' if speech_input_working or audio_output_working else '❌ NOT WORKING'}")
            
            if speech_input_working and audio_output_working:
                print("🎉 SUCCESS: Complete speech-to-speech pipeline is functional!")
            elif speech_input_working:
                print("⚠️  Speech input works, but audio output has issues")
            elif audio_output_working:
                print("⚠️  Audio output works, but speech input has issues")
            else:
                print("❌ Both speech input and audio output need attention")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the server is running on localhost:8000")

if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())
