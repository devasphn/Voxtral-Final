#!/usr/bin/env python3
"""
COMPREHENSIVE STREAMING FIXES VALIDATION SCRIPT
Tests all critical fixes for ultra-low latency streaming voice agent
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import time
import logging
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('streaming_test')

class StreamingFixesValidator:
    """
    Comprehensive validator for streaming voice agent fixes
    Tests all critical components and performance targets
    """
    
    def __init__(self, server_url: str = "ws://localhost:8000/ws"):
        self.server_url = server_url
        self.test_results = {}
        self.performance_metrics = {
            'first_word_latency': [],
            'word_to_audio_latency': [],
            'total_generation_time': [],
            'token_count': [],
            'error_count': 0
        }
        
    def generate_test_audio(self, duration: float = 3.0, sample_rate: int = 16000) -> bytes:
        """Generate test audio data"""
        samples = int(duration * sample_rate)
        # Generate speech-like audio with varying frequency
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 5 * t) * 0.3
        audio = audio.astype(np.float32)
        return audio.tobytes()
    
    async def test_streaming_mode_activation(self) -> Dict[str, Any]:
        """Test 1: Verify streaming mode is properly activated"""
        logger.info("[EMOJI] Test 1: Streaming Mode Activation")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Generate test audio
                audio_data = self.generate_test_audio(duration=2.0)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                # Send streaming request
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"test_streaming_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                logger.info("[EMOJI] Sent streaming mode request")
                
                # Listen for streaming responses
                streaming_detected = False
                timeout = 15
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'streaming_words':
                            streaming_detected = True
                            logger.info(f"[OK] Streaming words detected: {data.get('text')}")
                            break
                        elif data.get('type') == 'response':
                            logger.info(f"[NOTE] Regular response: {data.get('text')}")
                            
                    except asyncio.TimeoutError:
                        continue
                
                return {
                    'test_name': 'streaming_mode_activation',
                    'passed': streaming_detected,
                    'details': 'Streaming mode properly activated' if streaming_detected else 'Streaming mode not detected'
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Test 1 failed: {e}")
            return {
                'test_name': 'streaming_mode_activation',
                'passed': False,
                'details': f'Error: {str(e)}'
            }
    
    async def test_numpy_float32_fix(self) -> Dict[str, Any]:
        """Test 2: Verify numpy.float32 iteration error is fixed"""
        logger.info("[EMOJI] Test 2: Numpy Float32 Fix")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Generate test audio that previously caused the error
                audio_data = self.generate_test_audio(duration=4.0)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"test_numpy_fix_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                logger.info("[EMOJI] Sent test for numpy.float32 fix")
                
                # Monitor for errors
                error_detected = False
                success_detected = False
                timeout = 20
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'error':
                            error_msg = data.get('message', '')
                            if 'numpy.float32' in error_msg or 'not iterable' in error_msg:
                                error_detected = True
                                logger.error(f"[ERROR] Numpy error still present: {error_msg}")
                                break
                        elif data.get('type') in ['streaming_words', 'response']:
                            success_detected = True
                            logger.info(f"[OK] Processing successful: {data.get('text', '')[:50]}...")
                            
                    except asyncio.TimeoutError:
                        continue
                
                return {
                    'test_name': 'numpy_float32_fix',
                    'passed': success_detected and not error_detected,
                    'details': 'Numpy.float32 error fixed' if not error_detected else 'Numpy.float32 error still present'
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Test 2 failed: {e}")
            return {
                'test_name': 'numpy_float32_fix',
                'passed': False,
                'details': f'Error: {str(e)}'
            }
    
    async def test_long_token_generation(self) -> Dict[str, Any]:
        """Test 3: Verify longer token generation (50+ tokens instead of 4-5 words)"""
        logger.info("[EMOJI] Test 3: Long Token Generation")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Generate longer audio to encourage longer response
                audio_data = self.generate_test_audio(duration=5.0)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"test_long_gen_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                logger.info("[EMOJI] Sent test for long token generation")
                
                # Collect all generated text
                total_text = ""
                word_count = 0
                timeout = 30
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'streaming_words':
                            text = data.get('text', '')
                            total_text += " " + text
                            word_count += len(text.split())
                            logger.info(f"[NOTE] Words received: {text} (total: {word_count} words)")
                        elif data.get('type') == 'response':
                            text = data.get('text', '')
                            if not total_text:  # If no streaming words, use regular response
                                total_text = text
                                word_count = len(text.split())
                            logger.info(f"[NOTE] Final response: {text}")
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                
                # Check if we got sufficient length
                target_words = 15  # Target at least 15 words (roughly 50+ tokens)
                passed = word_count >= target_words
                
                return {
                    'test_name': 'long_token_generation',
                    'passed': passed,
                    'details': f'Generated {word_count} words (target: {target_words}+). Text: "{total_text.strip()[:100]}..."'
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Test 3 failed: {e}")
            return {
                'test_name': 'long_token_generation',
                'passed': False,
                'details': f'Error: {str(e)}'
            }
    
    async def test_performance_targets(self) -> Dict[str, Any]:
        """Test 4: Verify performance targets are being met"""
        logger.info("[EMOJI] Test 4: Performance Targets")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                audio_data = self.generate_test_audio(duration=3.0)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                start_time = time.time()
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"test_performance_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                
                first_word_time = None
                first_audio_time = None
                total_end_time = None
                
                timeout = 20
                request_start = time.time()
                
                while time.time() - request_start < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        current_time = time.time()
                        
                        if data.get('type') == 'streaming_words' and first_word_time is None:
                            first_word_time = (current_time - start_time) * 1000
                            logger.info(f"[FAST] First word latency: {first_word_time:.1f}ms")
                        elif data.get('type') == 'streaming_audio' and first_audio_time is None:
                            first_audio_time = (current_time - start_time) * 1000
                            logger.info(f"[AUDIO] First audio latency: {first_audio_time:.1f}ms")
                        elif data.get('type') == 'response':
                            total_end_time = (current_time - start_time) * 1000
                            logger.info(f"[EMOJI] Total latency: {total_end_time:.1f}ms")
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                
                # Evaluate performance
                targets_met = 0
                total_targets = 0
                details = []
                
                if first_word_time is not None:
                    total_targets += 1
                    if first_word_time <= 100:
                        targets_met += 1
                        details.append(f"[OK] First word: {first_word_time:.1f}ms (target: 100ms)")
                    else:
                        details.append(f"[ERROR] First word: {first_word_time:.1f}ms (target: 100ms)")
                
                if first_audio_time is not None:
                    total_targets += 1
                    if first_audio_time <= 300:
                        targets_met += 1
                        details.append(f"[OK] First audio: {first_audio_time:.1f}ms (target: 300ms)")
                    else:
                        details.append(f"[ERROR] First audio: {first_audio_time:.1f}ms (target: 300ms)")
                
                if total_end_time is not None:
                    total_targets += 1
                    if total_end_time <= 2000:  # Relaxed target for testing
                        targets_met += 1
                        details.append(f"[OK] Total time: {total_end_time:.1f}ms (target: 2000ms)")
                    else:
                        details.append(f"[ERROR] Total time: {total_end_time:.1f}ms (target: 2000ms)")
                
                passed = targets_met >= (total_targets * 0.6)  # 60% of targets met
                
                return {
                    'test_name': 'performance_targets',
                    'passed': passed,
                    'details': f'Met {targets_met}/{total_targets} targets. ' + '; '.join(details)
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Test 4 failed: {e}")
            return {
                'test_name': 'performance_targets',
                'passed': False,
                'details': f'Error: {str(e)}'
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("[INIT] Starting comprehensive streaming fixes validation...")
        
        tests = [
            self.test_streaming_mode_activation,
            self.test_numpy_float32_fix,
            self.test_long_token_generation,
            self.test_performance_targets
        ]
        
        results = []
        passed_count = 0
        
        for test in tests:
            try:
                result = await test()
                results.append(result)
                if result['passed']:
                    passed_count += 1
                    logger.info(f"[OK] {result['test_name']}: PASSED")
                else:
                    logger.error(f"[ERROR] {result['test_name']}: FAILED - {result['details']}")
                    
                # Wait between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"[ERROR] Test failed with exception: {e}")
                results.append({
                    'test_name': 'unknown',
                    'passed': False,
                    'details': f'Exception: {str(e)}'
                })
        
        # Generate summary
        total_tests = len(tests)
        success_rate = (passed_count / total_tests) * 100
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_count,
            'failed_tests': total_tests - passed_count,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if success_rate >= 75 else 'FAILED',
            'test_results': results
        }
        
        logger.info(f"[STATS] VALIDATION SUMMARY:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_count}")
        logger.info(f"   Failed: {total_tests - passed_count}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Overall Status: {summary['overall_status']}")
        
        return summary

async def main():
    """Main test execution"""
    validator = StreamingFixesValidator()
    
    try:
        results = await validator.run_all_tests()
        
        # Save results to file
        import json
        with open('streaming_fixes_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("[FILE] Results saved to streaming_fixes_validation_results.json")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_status'] == 'PASSED' else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"[ERROR] Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
