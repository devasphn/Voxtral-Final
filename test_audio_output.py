#!/usr/bin/env python3

import asyncio
import websockets
import json
import base64
import numpy as np

async def test_audio_output():
    """Test if audio output is working by connecting and waiting longer for responses"""
    uri = 'ws://localhost:8000/ws'
    
    print("🧪 Testing Audio Output Pipeline")
    print("=" * 50)
    
    # Create test audio data
    audio_data = np.random.uniform(-0.3, 0.3, 32000).astype(np.float32)
    audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection message
            conn_msg = await websocket.recv()
            conn_data = json.loads(conn_msg)
            print(f"✅ Connected: {conn_data.get('status', 'Unknown')}")
            
            # Send test audio
            message = {
                'type': 'audio_chunk',
                'audio_data': audio_b64,
                'chunk_id': 'audio_output_test',
                'streaming': True
            }
            
            print(f"📤 Sending audio chunk...")
            await websocket.send(json.dumps(message))
            
            # Listen for responses with longer timeout
            responses = []
            audio_chunks = []
            
            try:
                for i in range(30):  # Listen for up to 30 messages
                    print(f"⏳ Waiting for response {i+1}/30...")
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)  # Much longer timeout
                    data = json.loads(response)
                    responses.append(data)
                    
                    msg_type = data.get('type', 'unknown')
                    print(f"📥 Response {i+1}: {msg_type}")
                    
                    if msg_type == 'streaming_words':
                        text = data.get('text', '')
                        print(f"   📝 Words: '{text}'")
                        
                    elif msg_type == 'streaming_audio':
                        chunk_index = data.get('chunk_index', 0)
                        audio_data_len = len(data.get('audio_data', ''))
                        is_final = data.get('is_final', False)
                        audio_chunks.append((chunk_index, audio_data_len, is_final))
                        print(f"   🔊 Audio: {audio_data_len} bytes (chunk {chunk_index}, final: {is_final})")
                        
                    elif msg_type == 'streaming_complete':
                        print(f"   ✅ Streaming complete")
                        break
                        
                    elif msg_type == 'error':
                        error_msg = data.get('error', 'Unknown error')
                        print(f"   ❌ Error: {error_msg}")
                        break
                        
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for more responses")
            
            print("\n" + "=" * 50)
            print("📊 AUDIO OUTPUT TEST RESULTS")
            print("=" * 50)
            print(f"📈 Total responses: {len(responses)}")
            print(f"🔊 Audio chunks received: {len(audio_chunks)}")
            
            if audio_chunks:
                print("✅ AUDIO OUTPUT: WORKING")
                for i, (chunk_idx, size, is_final) in enumerate(audio_chunks):
                    print(f"   Chunk {chunk_idx}: {size} bytes (final: {is_final})")
            else:
                print("❌ AUDIO OUTPUT: NOT WORKING")
                print("   No audio chunks received")
            
            # Check if we got any streaming words
            word_responses = [r for r in responses if r.get('type') == 'streaming_words']
            if word_responses:
                print("✅ SPEECH RECOGNITION: WORKING")
                for r in word_responses:
                    print(f"   Words: '{r.get('text', '')}'")
            else:
                print("❌ SPEECH RECOGNITION: NOT WORKING")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_audio_output())
