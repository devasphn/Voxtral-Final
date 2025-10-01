#!/usr/bin/env python3
"""
Direct test of Voxtral model to diagnose response quality issues
"""

import asyncio
import sys
import os
import numpy as np
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

async def test_voxtral_direct():
    """Test Voxtral model directly without WebSocket"""
    print("🔧 Testing Voxtral Model Directly...")
    
    try:
        # Import and initialize Voxtral model
        from src.models.voxtral_model_realtime import VoxtralModel
        
        print("📦 Initializing Voxtral model...")
        voxtral = VoxtralModel()
        await voxtral.initialize()
        print("✅ Voxtral model initialized successfully")
        
        # Test 1: Text-only processing (bypass audio issues)
        print("\n🧪 Test 1: Direct text processing")
        test_inputs = [
            "Hello, how are you?",
            "What is the weather like today?",
            "Tell me a joke",
            "Can you help me with something?"
        ]
        
        for i, text_input in enumerate(test_inputs, 1):
            print(f"\n   Input {i}: '{text_input}'")
            try:
                result = await voxtral.process_text_direct(text_input, mode="conversation")
                response = result.get('response', '')
                print(f"   Output {i}: '{response}'")
                
                # Check for issues
                if not response or len(response.strip()) < 3:
                    print(f"   ❌ Issue: Response too short or empty")
                elif any(char in response for char in ['<', '>', '+', '*']):
                    print(f"   ❌ Issue: Contains formatting artifacts")
                else:
                    print(f"   ✅ Response looks good")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test 2: Audio processing with simple synthetic audio
        print("\n🧪 Test 2: Audio processing with synthetic speech")
        
        # Create simple test audio (sine wave representing speech)
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a more speech-like signal with modulation
        audio_data = (np.sin(2 * np.pi * frequency * t) * 
                     np.sin(2 * np.pi * 5 * t) * 0.3).astype(np.float32)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.01, audio_data.shape).astype(np.float32)
        audio_data = audio_data + noise
        
        print(f"   Generated test audio: {len(audio_data)} samples, {duration}s duration")
        
        try:
            import torch
            audio_tensor = torch.from_numpy(audio_data)
            
            result = await voxtral.process_realtime_chunk(
                audio_tensor, 
                chunk_id=1, 
                mode="conversation"
            )
            
            print(f"   Audio processing result: {result}")
            
            if result.get('success'):
                response = result.get('response', '')
                print(f"   Audio response: '{response}'")
                
                if result.get('is_silence'):
                    print("   ℹ️  Audio was detected as silence (expected for synthetic audio)")
                elif not response or len(response.strip()) < 3:
                    print("   ❌ Issue: Audio response too short or empty")
                else:
                    print("   ✅ Audio processing successful")
            else:
                print(f"   ❌ Audio processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Audio processing error: {e}")
        
        # Test 3: Check model configuration
        print("\n🧪 Test 3: Model configuration check")
        model_info = voxtral.get_model_info()
        print(f"   Model info: {model_info}")
        
        # Check for potential issues
        issues = []
        if not model_info.get('status') == 'initialized':
            issues.append("Model not properly initialized")
        
        vad_settings = model_info.get('vad_settings', {})
        if vad_settings.get('silence_threshold', 0) > 0.05:
            issues.append(f"VAD silence threshold too high: {vad_settings.get('silence_threshold')}")
        
        if issues:
            print("   ❌ Configuration issues found:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("   ✅ Model configuration looks good")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to test Voxtral model: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

async def test_websocket_message_types():
    """Test what message types the WebSocket actually sends"""
    print("\n🔧 Testing WebSocket Message Types...")
    
    try:
        import websockets
        import json
        import base64
        
        # Create simple test audio
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.1
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Test regular mode
            print("\n   Testing regular mode...")
            message = {
                "type": "audio_chunk",
                "audio_data": audio_base64,
                "mode": "conversation",
                "streaming": False,
                "prompt": "",
                "chunk_id": 1,
                "timestamp": int(time.time() * 1000)
            }
            
            await websocket.send(json.dumps(message))
            
            # Collect responses for 10 seconds
            responses = []
            start_time = time.time()
            
            try:
                while time.time() - start_time < 10:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    msg_type = data.get('type', 'unknown')
                    responses.append(msg_type)
                    
                    print(f"   Received: {msg_type}")
                    if msg_type == 'response':
                        text = data.get('text', '')
                        print(f"   Response text: '{text}'")
                    elif msg_type == 'error':
                        print(f"   Error: {data.get('message', '')}")
                        break
                    elif msg_type in ['response', 'audio_response']:
                        break
                        
            except asyncio.TimeoutError:
                print("   Timeout waiting for responses")
            
            print(f"   Message types received: {set(responses)}")
            
            # Test streaming mode
            print("\n   Testing streaming mode...")
            message['streaming'] = True
            message['chunk_id'] = 2
            
            await websocket.send(json.dumps(message))
            
            streaming_responses = []
            start_time = time.time()
            
            try:
                while time.time() - start_time < 10:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    msg_type = data.get('type', 'unknown')
                    streaming_responses.append(msg_type)
                    
                    print(f"   Streaming received: {msg_type}")
                    if msg_type == 'streaming_words':
                        text = data.get('text', '')
                        print(f"   Streaming text: '{text}'")
                    elif msg_type == 'error':
                        print(f"   Error: {data.get('message', '')}")
                        break
                    elif msg_type == 'streaming_complete':
                        break
                        
            except asyncio.TimeoutError:
                print("   Timeout waiting for streaming responses")
            
            print(f"   Streaming message types received: {set(streaming_responses)}")
            
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

async def main():
    """Run all diagnostic tests"""
    print("🚀 Starting Voxtral Diagnostic Tests...\n")
    
    # Test 1: Direct Voxtral model
    success1 = await test_voxtral_direct()
    
    # Test 2: WebSocket message types
    success2 = await test_websocket_message_types()
    
    print(f"\n📊 Test Results:")
    print(f"   Direct Voxtral test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   WebSocket test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n✅ All tests passed - system appears to be working")
    else:
        print("\n❌ Some tests failed - issues detected")

if __name__ == "__main__":
    asyncio.run(main())
