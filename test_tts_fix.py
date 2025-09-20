#!/usr/bin/env python3
"""
Simple test script to verify TTS functionality is working after the fix
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import time

def generate_test_audio(duration_seconds=2, sample_rate=16000, frequency=440):
    """Generate a simple sine wave audio for testing"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    # Generate a sine wave that sounds like speech (mix of frequencies)
    audio = 0.3 * (np.sin(frequency * 2 * np.pi * t) +
                   0.5 * np.sin(frequency * 1.5 * 2 * np.pi * t) +
                   0.3 * np.sin(frequency * 0.8 * 2 * np.pi * t))

    # Add some noise to make it more speech-like
    noise = np.random.normal(0, 0.05, audio.shape)
    audio = audio + noise

    # Convert to int16 format
    audio = (audio * 32767).astype(np.int16)
    return audio

async def test_tts_conversation():
    """Test the TTS functionality by sending audio data through WebSocket"""
    uri = "ws://localhost:8000/ws"

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")

            # Generate test audio that simulates speech
            print("Generating test audio...")
            test_audio = generate_test_audio(duration_seconds=3, frequency=300)  # Lower frequency for speech-like sound
            audio_bytes = test_audio.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            # Send audio chunk to trigger speech processing
            test_message = {
                "type": "audio_chunk",
                "audio_data": audio_b64,
                "mode": "conversation",
                "streaming": False,
                "prompt": "",
                "chunk_id": 1,
                "timestamp": time.time()
            }

            print("Sending test audio chunk to trigger speech-to-speech processing...")
            await websocket.send(json.dumps(test_message))
            
            # Listen for responses - wait longer for processing
            response_count = 0
            audio_received = False
            text_response_received = False

            while response_count < 15:  # Increased limit
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=20.0)  # Longer timeout
                    data = json.loads(response)

                    print(f"Received response type: {data.get('type', 'unknown')}")

                    if data.get('type') == 'audio_response':
                        print(f"✅ TTS Audio received! Chunk ID: {data.get('chunk_id')}")
                        print(f"   Audio data length: {len(data.get('audio_data', ''))}")
                        print(f"   Sample rate: {data.get('metadata', {}).get('sample_rate', 'unknown')}")
                        audio_received = True

                    elif data.get('type') == 'streaming_audio':
                        print(f"✅ Streaming TTS Audio received! Chunk index: {data.get('chunk_index')}")
                        print(f"   Audio data length: {len(data.get('audio_data', ''))}")
                        print(f"   Is final: {data.get('is_final', False)}")
                        audio_received = True

                    elif data.get('type') == 'response':
                        print(f"📝 Text response: {data.get('text', '')[:100]}...")
                        text_response_received = True

                    elif data.get('type') == 'error':
                        print(f"❌ Error received: {data.get('message', 'Unknown error')}")

                    elif data.get('type') == 'info':
                        print(f"ℹ️  Info: {data.get('message', '')}")

                    elif data.get('type') == 'connection':
                        print(f"🔗 Connection: {data.get('message', '')}")

                    response_count += 1

                    # Break if we got both text and audio response
                    if text_response_received and audio_received:
                        print("✅ Received both text and audio responses!")
                        break

                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for response")
                    break
                except Exception as e:
                    print(f"❌ Error receiving response: {e}")
                    break
            
            if audio_received and text_response_received:
                print("\n🎉 SUCCESS: Complete speech-to-speech pipeline working!")
                print("✅ Speech input processed and TTS audio generated successfully.")
                print("✅ The TTS fix is working correctly.")
            elif text_response_received:
                print("\n⚠️  PARTIAL SUCCESS: Speech processing working, but no TTS audio received.")
                print("✅ Speech input and text generation working.")
                print("❓ TTS audio may have been generated but not received in test.")
            else:
                print("\n❌ FAILURE: No meaningful response received.")
                print("❌ Check server logs for more details.")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the server is running on localhost:8000")

if __name__ == "__main__":
    print("Testing TTS functionality after fix...")
    asyncio.run(test_tts_conversation())
