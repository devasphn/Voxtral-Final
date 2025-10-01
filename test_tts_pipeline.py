#!/usr/bin/env python3
"""
Test TTS Pipeline - Verify the complete TTS audio generation and streaming_complete handling
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSPipelineTest:
    def __init__(self):
        self.ws_url = "ws://localhost:8000/ws"
        self.websocket = None
        self.test_results = {}
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            logger.info("Connecting to WebSocket server...")
            self.websocket = await websockets.connect(self.ws_url)
            
            # Wait for connection message
            response = await self.websocket.recv()
            data = json.loads(response)
            logger.info(f"Connected: {data}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def test_streaming_complete_handling(self):
        """Test that streaming_complete messages trigger TTS generation"""
        logger.info("\n🧪 Testing streaming_complete message handling...")
        
        try:
            # Create a fake audio chunk (silence)
            sample_rate = 16000
            duration_seconds = 2.0
            samples = int(sample_rate * duration_seconds)
            
            # Generate some fake speech-like audio (sine wave with noise)
            t = np.linspace(0, duration_seconds, samples)
            frequency = 440  # A4 note
            audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)
            audio_data += 0.05 * np.random.normal(0, 1, samples)  # Add noise
            audio_data = audio_data.astype(np.float32)
            
            # Convert to base64
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            # Send audio chunk message
            message = {
                "type": "audio_chunk",
                "audio_data": audio_b64,
                "chunk_id": 0,
                "sample_rate": sample_rate,
                "timestamp": time.time(),
                "mode": "streaming",  # Use streaming mode to trigger streaming_complete
                "streaming": True     # Also set the streaming flag explicitly
            }
            
            logger.info("Sending audio chunk for streaming processing...")
            await self.websocket.send(json.dumps(message))
            
            # Listen for responses
            streaming_complete_received = False
            audio_response_received = False
            response_count = 0
            max_responses = 10
            
            while response_count < max_responses:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    response_count += 1
                    
                    logger.info(f"Received message type: {data.get('type')}")
                    
                    if data.get('type') == 'streaming_complete':
                        streaming_complete_received = True
                        logger.info(f"✅ streaming_complete received: {data.get('full_response', '')}")
                        
                    elif data.get('type') == 'audio_response':
                        audio_response_received = True
                        audio_data = data.get('audio_data', '')
                        logger.info(f"✅ audio_response received: {len(audio_data)} chars of base64 audio")
                        
                        # Verify audio data
                        if audio_data:
                            try:
                                audio_bytes = base64.b64decode(audio_data)
                                logger.info(f"✅ Audio decoded successfully: {len(audio_bytes)} bytes")
                            except Exception as e:
                                logger.error(f"❌ Failed to decode audio: {e}")
                        
                    elif data.get('type') == 'error':
                        logger.error(f"❌ Error received: {data.get('message')}")
                        
                    # Break if we got both messages
                    if streaming_complete_received and audio_response_received:
                        logger.info("✅ Both streaming_complete and audio_response received!")
                        break
                        
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for response")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    break
            
            self.test_results['streaming_complete_handling'] = {
                'streaming_complete_received': streaming_complete_received,
                'audio_response_received': audio_response_received,
                'success': streaming_complete_received and audio_response_received
            }
            
            if streaming_complete_received and audio_response_received:
                logger.info("✅ streaming_complete handling test PASSED")
            else:
                logger.error("❌ streaming_complete handling test FAILED")
                logger.error(f"   streaming_complete: {streaming_complete_received}")
                logger.error(f"   audio_response: {audio_response_received}")
                
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            self.test_results['streaming_complete_handling'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_direct_tts_generation(self):
        """Test direct TTS generation request"""
        logger.info("\n🧪 Testing direct TTS generation...")
        
        try:
            # Send direct TTS generation request
            message = {
                "type": "generate_tts",
                "text": "Hello, this is a test of the TTS system with Indian female voice.",
                "chunk_id": "test_tts_001",
                "voice": "hf_alpha",
                "speed": 1.0,
                "timestamp": time.time()
            }
            
            logger.info("Sending direct TTS generation request...")
            await self.websocket.send(json.dumps(message))
            
            # Wait for audio response
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                data = json.loads(response)
                
                if data.get('type') == 'audio_response':
                    audio_data = data.get('audio_data', '')
                    logger.info(f"✅ Direct TTS audio response received: {len(audio_data)} chars")
                    
                    # Verify audio data
                    if audio_data:
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            logger.info(f"✅ Audio decoded successfully: {len(audio_bytes)} bytes")
                            
                            # Check if it's a valid WAV file
                            if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
                                logger.info("✅ Valid WAV file format detected")
                            else:
                                logger.warning("⚠️ Audio data doesn't appear to be WAV format")
                                
                        except Exception as e:
                            logger.error(f"❌ Failed to decode audio: {e}")
                    
                    self.test_results['direct_tts_generation'] = {
                        'success': True,
                        'audio_size': len(audio_data)
                    }
                    logger.info("✅ Direct TTS generation test PASSED")
                    
                elif data.get('type') == 'error':
                    logger.error(f"❌ TTS generation error: {data.get('message')}")
                    self.test_results['direct_tts_generation'] = {
                        'success': False,
                        'error': data.get('message')
                    }
                else:
                    logger.error(f"❌ Unexpected response type: {data.get('type')}")
                    self.test_results['direct_tts_generation'] = {
                        'success': False,
                        'error': f"Unexpected response type: {data.get('type')}"
                    }
                    
            except asyncio.TimeoutError:
                logger.error("❌ Timeout waiting for TTS response")
                self.test_results['direct_tts_generation'] = {
                    'success': False,
                    'error': 'Timeout'
                }
                
        except Exception as e:
            logger.error(f"❌ Direct TTS test failed: {e}")
            self.test_results['direct_tts_generation'] = {
                'success': False,
                'error': str(e)
            }
    
    async def run_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting TTS Pipeline Tests")
        
        if not await self.connect():
            logger.error("❌ Failed to connect to server")
            return False
        
        try:
            await self.test_direct_tts_generation()
            await self.test_streaming_complete_handling()
            
            # Print summary
            logger.info("\n📊 Test Results Summary:")
            for test_name, result in self.test_results.items():
                status = "✅ PASS" if result.get('success') else "❌ FAIL"
                logger.info(f"  {test_name}: {status}")
                if not result.get('success') and 'error' in result:
                    logger.info(f"    Error: {result['error']}")
            
            return all(result.get('success', False) for result in self.test_results.values())
            
        finally:
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main test function"""
    test = TTSPipelineTest()
    success = await test.run_tests()
    
    if success:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)
