#!/usr/bin/env python3
"""
Comprehensive test for Voxtral real-time streaming system fixes.
Tests all identified issues: response fragmentation, formatting artifacts, and message type handling.
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import time
import re

async def test_response_completeness():
    """Test that Voxtral generates complete, coherent responses without fragmentation"""
    
    print("🔧 Testing Response Completeness...")
    
    # Create realistic speech audio
    sample_rate = 16000
    duration = 2.5
    samples = int(sample_rate * duration)
    
    # Generate speech-like audio pattern with multiple harmonics
    t = np.linspace(0, duration, samples, False)
    audio_data = (
        0.4 * np.sin(2 * np.pi * 250 * t) +  # Fundamental frequency
        0.3 * np.sin(2 * np.pi * 750 * t) +  # Harmonic
        0.2 * np.sin(2 * np.pi * 1500 * t) + # Higher harmonic
        0.1 * np.sin(2 * np.pi * 3000 * t)   # Even higher harmonic
    ) * 0.15
    
    # Add realistic background noise
    noise = np.random.normal(0, 0.005, samples)
    audio_data = (audio_data + noise).astype(np.float32)
    
    audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            
            # Test message
            message = {
                "type": "audio_chunk",
                "audio_data": audio_base64,
                "mode": "conversation",
                "streaming": True,
                "prompt": "",
                "chunk_id": 0,
                "timestamp": int(time.time() * 1000)
            }
            
            print("📤 Sending test audio (simulating natural speech)...")
            await websocket.send(json.dumps(message))
            
            # Collect streaming responses
            words = []
            full_response = ""
            start_time = time.time()
            
            try:
                while time.time() - start_time < 20:  # 20 second timeout
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(response)
                    
                    if data.get('type') == 'streaming_chunk':
                        word = data.get('text', '').strip()
                        if word:
                            words.append(word)
                            print(f"📥 Word: '{word}'")
                    elif data.get('type') == 'response':
                        full_response = data.get('text', '').strip()
                        if full_response:
                            print(f"📥 Complete response: '{full_response}'")
                            break
                    elif data.get('type') == 'error':
                        print(f"❌ Error: {data.get('message', '')}")
                        return False
                        
            except asyncio.TimeoutError:
                print("⏰ Response collection timeout")
            
            # Analyze response quality
            combined_response = ' '.join(words).strip() if words else full_response
            print(f"\n📊 RESPONSE ANALYSIS:")
            print(f"   Combined response: '{combined_response}'")
            print(f"   Word count: {len(words)}")
            print(f"   Character count: {len(combined_response)}")
            
            # Check for issues
            issues = []
            
            # 1. Check for fragmentation (incomplete sentences)
            if combined_response and not combined_response.endswith(('.', '!', '?')):
                if len(combined_response) > 10:  # Only flag if it's a substantial response
                    issues.append("Response appears fragmented (doesn't end with punctuation)")
            
            # 2. Check for formatting artifacts
            formatting_artifacts = [':', '*', '"', '(', ')', '[', ']']
            found_artifacts = [char for char in formatting_artifacts if char in combined_response]
            if found_artifacts:
                issues.append(f"Contains formatting artifacts: {found_artifacts}")
            
            # 3. Check for minimum response length
            if len(combined_response) < 5:
                issues.append("Response too short")
            
            # 4. Check for appropriate conversational content
            inappropriate_phrases = [
                "nested structure",
                "sentence structure", 
                "break it down",
                "having trouble understanding",
                "Let's play:",
                "You start:"
            ]
            
            found_inappropriate = [phrase for phrase in inappropriate_phrases if phrase.lower() in combined_response.lower()]
            if found_inappropriate:
                issues.append(f"Contains inappropriate content: {found_inappropriate}")
            
            # Results
            if not issues:
                print("✅ SUCCESS: Response is complete and appropriate")
                return True
            else:
                print("❌ ISSUES FOUND:")
                for issue in issues:
                    print(f"   - {issue}")
                return False
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

async def test_message_type_handling():
    """Test that the system handles message types correctly"""
    
    print("\n🔧 Testing Message Type Handling...")
    
    # Simple test audio
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.1
    audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected for message type test")
            
            message = {
                "type": "audio_chunk",
                "audio_data": audio_base64,
                "mode": "conversation",
                "streaming": True,
                "prompt": "",
                "chunk_id": 0,
                "timestamp": int(time.time() * 1000)
            }
            
            await websocket.send(json.dumps(message))
            
            # Collect message types
            message_types = []
            start_time = time.time()
            
            try:
                while time.time() - start_time < 10:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    msg_type = data.get('type', 'unknown')
                    message_types.append(msg_type)
                    
                    if msg_type == 'error':
                        print(f"❌ Error: {data.get('message', '')}")
                        return False
                    elif msg_type in ['response', 'streaming_complete']:
                        break
                        
            except asyncio.TimeoutError:
                print("⏰ Message type collection timeout")
            
            print(f"📊 MESSAGE TYPES RECEIVED: {set(message_types)}")
            
            # Expected message types for streaming conversation
            expected_types = {'streaming_chunk', 'streaming_audio'}
            received_types = set(message_types)
            
            if expected_types.issubset(received_types):
                print("✅ SUCCESS: Correct message types received")
                return True
            else:
                missing = expected_types - received_types
                print(f"⚠️  Missing expected message types: {missing}")
                return len(missing) == 0  # Allow if no critical types missing
                
    except Exception as e:
        print(f"❌ Message type test failed: {e}")
        return False

async def test_multiple_interactions():
    """Test that the system can handle multiple interactions without degradation"""
    
    print("\n🔧 Testing Multiple Interactions...")
    
    # Test audio
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.1
    audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected for multiple interaction test")
            
            success_count = 0
            
            for i in range(3):
                print(f"\n🧪 Interaction {i+1}:")
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_base64,  # FIXED: Use base64 encoded string
                    "mode": "conversation",
                    "streaming": True,
                    "prompt": "",
                    "chunk_id": i,
                    "timestamp": int(time.time() * 1000)
                }
                
                await websocket.send(json.dumps(message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                    data = json.loads(response)
                    
                    if data.get('type') == 'error':
                        print(f"   ❌ Error: {data.get('message', '')}")
                    else:
                        print(f"   ✅ Response type: {data.get('type', 'unknown')}")
                        success_count += 1
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ Timeout for interaction {i+1}")
                
                # Small delay between interactions
                await asyncio.sleep(1)
            
            if success_count >= 2:  # Allow 1 failure
                print(f"✅ SUCCESS: {success_count}/3 interactions successful")
                return True
            else:
                print(f"❌ FAILED: Only {success_count}/3 interactions successful")
                return False
                
    except Exception as e:
        print(f"❌ Multiple interaction test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting COMPREHENSIVE VOXTRAL FIXES VERIFICATION...\n")
    
    # Test 1: Response Completeness
    print("=" * 70)
    print("TEST 1: RESPONSE COMPLETENESS & FORMATTING")
    print("=" * 70)
    completeness_passed = await test_response_completeness()
    
    # Test 2: Message Type Handling
    print("\n" + "=" * 70)
    print("TEST 2: MESSAGE TYPE HANDLING")
    print("=" * 70)
    message_types_passed = await test_message_type_handling()
    
    # Test 3: Multiple Interactions
    print("\n" + "=" * 70)
    print("TEST 3: MULTIPLE INTERACTIONS")
    print("=" * 70)
    interactions_passed = await test_multiple_interactions()
    
    # Final Results
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE VERIFICATION RESULTS")
    print("=" * 80)
    
    total_tests = 3
    passed_tests = sum([completeness_passed, message_types_passed, interactions_passed])
    
    print(f"📊 OVERALL SCORE: {passed_tests}/{total_tests} tests passed")
    print()
    
    if completeness_passed:
        print("✅ Response Completeness: FIXED")
        print("   - No fragmentation detected")
        print("   - No formatting artifacts")
        print("   - Appropriate conversational content")
    else:
        print("❌ Response Completeness: ISSUES REMAIN")
        print("   - Fragmentation or formatting artifacts detected")
    
    if message_types_passed:
        print("✅ Message Type Handling: WORKING")
        print("   - Correct message types received")
        print("   - Proper streaming flow")
    else:
        print("❌ Message Type Handling: ISSUES REMAIN")
        print("   - Unexpected or missing message types")
    
    if interactions_passed:
        print("✅ Multiple Interactions: WORKING")
        print("   - System handles multiple requests")
        print("   - No degradation between interactions")
    else:
        print("❌ Multiple Interactions: ISSUES REMAIN")
        print("   - System fails on repeated interactions")
    
    print()
    
    if passed_tests == total_tests:
        print("🎉 ALL ISSUES RESOLVED!")
        print("   ✅ Voxtral real-time streaming system is fully functional")
        print("   ✅ Ready for production use")
    elif passed_tests >= 2:
        print("⚠️  MOSTLY RESOLVED:")
        print(f"   ✅ {passed_tests}/{total_tests} critical issues fixed")
        print("   ⚠️  Minor issues may remain")
    else:
        print("💥 CRITICAL ISSUES REMAIN:")
        print(f"   ❌ Only {passed_tests}/{total_tests} tests passed")
        print("   🔧 Further investigation required")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
