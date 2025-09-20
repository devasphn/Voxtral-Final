#!/usr/bin/env python3

import asyncio
import websockets
import json
import base64
import numpy as np

async def test_conversation_mode():
    """Test if conversation mode is working correctly"""
    uri = 'ws://localhost:8000/ws'
    
    print("🧪 Testing Conversation Mode")
    print("=" * 50)
    
    # Create test audio data (simulating speech: "Hello, how are you?")
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
                'chunk_id': 'conversation_test',
                'streaming': True
            }
            
            print(f"📤 Sending audio chunk for conversation test...")
            await websocket.send(json.dumps(message))
            
            # Listen for responses
            responses = []
            words_received = []
            
            try:
                for i in range(10):  # Listen for up to 10 messages
                    print(f"⏳ Waiting for response {i+1}/10...")
                    response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    msg_type = data.get('type', 'unknown')
                    print(f"📥 Response {i+1}: {msg_type}")
                    
                    if msg_type == 'streaming_words':
                        text = data.get('text', '')
                        words_received.append(text)
                        print(f"   📝 Words: '{text}'")
                        
                    elif msg_type == 'streaming_complete':
                        full_response = data.get('full_response', '')
                        print(f"   ✅ Complete response: '{full_response}'")
                        break
                        
                    elif msg_type == 'error':
                        error_msg = data.get('error', 'Unknown error')
                        print(f"   ❌ Error: {error_msg}")
                        break
                        
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for more responses")
            
            print("\n" + "=" * 50)
            print("📊 CONVERSATION MODE TEST RESULTS")
            print("=" * 50)
            
            # Analyze the response
            full_text = ' '.join(words_received)
            print(f"📈 Total responses: {len(responses)}")
            print(f"📝 Full response: '{full_text}'")
            
            # Check if it's echoing (transcription mode) or responding (conversation mode)
            if not full_text:
                print("❌ CONVERSATION MODE: NOT WORKING")
                print("   No response received")
            elif "hello" in full_text.lower() and ("how are you" in full_text.lower() or "hear me" in full_text.lower()):
                print("❌ CONVERSATION MODE: NOT WORKING")
                print("   System is echoing input (transcription mode)")
                print(f"   Response: '{full_text}'")
            else:
                print("✅ CONVERSATION MODE: WORKING")
                print("   System is generating conversational responses")
                print(f"   Response: '{full_text}'")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_conversation_mode())
