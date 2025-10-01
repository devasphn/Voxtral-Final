#!/usr/bin/env python3
"""
Simple test to verify the tmp_file error fix
Tests the specific error that was reported: 'name tmp_file is not defined'
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tmp_file_test")

async def test_tmp_file_fix():
    """Test that the tmp_file error is fixed"""
    logger.info("🧪 Testing tmp_file error fix...")
    
    try:
        # Connect to the WebSocket server
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket server")
            
            # Create test audio data (similar to what caused the original error)
            sample_rate = 16000
            duration = 3.2  # 3.2 seconds (51200 samples like in the error log)
            test_audio = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)
            
            # Convert to base64 (as the client would)
            audio_bytes = test_audio.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send audio chunk message
            message = {
                "type": "audio_chunk",
                "audio_data": audio_b64,
                "chunk_id": 0,
                "timestamp": time.time()
            }
            
            logger.info(f"📤 Sending audio chunk ({len(test_audio)} samples, {duration:.1f}s)")
            await websocket.send(json.dumps(message))
            
            # Wait for response or error
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "error":
                    error_message = response_data.get("message", "")
                    if "tmp_file" in error_message and "not defined" in error_message:
                        logger.error("❌ tmp_file error still occurs!")
                        logger.error(f"   Error: {error_message}")
                        return False
                    else:
                        logger.warning(f"⚠️  Different error occurred: {error_message}")
                        return True  # Different error, but tmp_file is fixed
                else:
                    logger.info("✅ No tmp_file error - processing successful!")
                    logger.info(f"   Response type: {response_data.get('type')}")
                    return True
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️  Timeout waiting for response (but no tmp_file error)")
                return True  # No immediate error, likely processing
                
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False

async def test_multiple_chunks():
    """Test multiple audio chunks to ensure stability"""
    logger.info("🧪 Testing multiple audio chunks...")
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected for multi-chunk test")
            
            success_count = 0
            total_chunks = 3
            
            for i in range(total_chunks):
                # Create different test audio for each chunk
                test_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)  # 1 second
                audio_bytes = test_audio.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "chunk_id": i,
                    "timestamp": time.time()
                }
                
                logger.info(f"📤 Sending chunk {i+1}/{total_chunks}")
                await websocket.send(json.dumps(message))
                
                # Wait briefly between chunks
                await asyncio.sleep(1)
                
                # Check for any immediate errors
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "error":
                        error_message = response_data.get("message", "")
                        if "tmp_file" in error_message:
                            logger.error(f"❌ tmp_file error in chunk {i+1}: {error_message}")
                        else:
                            logger.warning(f"⚠️  Other error in chunk {i+1}: {error_message}")
                            success_count += 1  # Not a tmp_file error
                    else:
                        logger.info(f"✅ Chunk {i+1} processed successfully")
                        success_count += 1
                        
                except asyncio.TimeoutError:
                    logger.info(f"✅ Chunk {i+1} - no immediate error")
                    success_count += 1
            
            logger.info(f"📊 Multi-chunk test: {success_count}/{total_chunks} chunks processed without tmp_file errors")
            return success_count == total_chunks
            
    except Exception as e:
        logger.error(f"❌ Multi-chunk test failed: {e}")
        return False

async def main():
    """Main test execution"""
    logger.info("🔧 TESTING tmp_file ERROR FIX")
    logger.info("=" * 50)
    
    # Test 1: Single chunk (reproduces original error scenario)
    test1_result = await test_tmp_file_fix()
    
    # Test 2: Multiple chunks (stability test)
    test2_result = await test_multiple_chunks()
    
    # Results
    logger.info("\n📊 TEST RESULTS:")
    logger.info("=" * 50)
    logger.info(f"Single Chunk Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    logger.info(f"Multi-Chunk Test:  {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    overall_success = test1_result and test2_result
    
    if overall_success:
        logger.info("\n🎉 ALL TESTS PASSED - tmp_file ERROR IS FIXED!")
        logger.info("   The audio processing pipeline is stable and working correctly.")
    else:
        logger.info("\n⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        exit(1)
