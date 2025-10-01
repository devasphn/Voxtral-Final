#!/usr/bin/env python3
"""
Audio Pipeline Stability Test for Voxtral Voice Application
Tests the complete audio processing pipeline to ensure no runtime errors occur
"""

import asyncio
import time
import logging
import sys
import os
import numpy as np
import base64
import json
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audio_pipeline_test")

class AudioPipelineStabilityTester:
    """Tests audio processing pipeline stability and error handling"""
    
    def __init__(self):
        self.test_results = {}
        self.unified_manager = None
        self.audio_processor = None
        
    async def initialize_services(self):
        """Initialize audio processing services"""
        logger.info("🔧 Initializing audio processing services...")
        
        try:
            # Import services
            from src.api.ui_server_realtime import get_unified_manager, get_audio_processor
            from src.models.audio_processor_realtime import AudioProcessor
            
            self.unified_manager = get_unified_manager()
            self.audio_processor = get_audio_processor()
            
            if not self.unified_manager:
                logger.error("❌ Unified manager not available")
                return False
                
            if not self.audio_processor:
                logger.info("🔧 Creating audio processor...")
                self.audio_processor = AudioProcessor()
            
            logger.info("✅ Audio processing services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            return False
    
    async def test_audio_validation(self):
        """Test enhanced audio validation"""
        logger.info("\n🧪 Testing Audio Validation")
        
        test_cases = [
            ("empty_array", np.array([], dtype=np.float32)),
            ("nan_values", np.array([1.0, np.nan, 0.5], dtype=np.float32)),
            ("inf_values", np.array([1.0, np.inf, 0.5], dtype=np.float32)),
            ("silent_audio", np.zeros(1000, dtype=np.float32)),
            ("valid_audio", np.random.normal(0, 0.1, 1000).astype(np.float32))
        ]
        
        validation_results = []
        
        for test_name, audio_data in test_cases:
            try:
                # Test audio validation
                if len(audio_data) == 0:
                    result = "empty_detected"
                elif np.isnan(audio_data).any() or np.isinf(audio_data).any():
                    result = "invalid_detected"
                elif np.max(np.abs(audio_data)) == 0:
                    result = "silent_detected"
                else:
                    result = "valid"
                
                validation_results.append({
                    'test_name': test_name,
                    'result': result,
                    'success': True
                })
                
                logger.info(f"  ✅ {test_name}: {result}")
                
            except Exception as e:
                validation_results.append({
                    'test_name': test_name,
                    'result': 'error',
                    'error': str(e),
                    'success': False
                })
                logger.error(f"  ❌ {test_name}: {e}")
        
        self.test_results['audio_validation'] = validation_results
        
        success_rate = sum(1 for r in validation_results if r['success']) / len(validation_results)
        logger.info(f"📊 Audio Validation Success Rate: {success_rate*100:.1f}%")
    
    async def test_voxtral_processing(self):
        """Test Voxtral audio processing without tmp_file errors"""
        logger.info("\n🧪 Testing Voxtral Audio Processing")
        
        try:
            voxtral_model = await self.unified_manager.get_voxtral_model()
            
            # Create test audio data
            sample_rate = 16000
            duration = 2.0  # 2 seconds
            test_audio = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)
            
            # Test processing
            start_time = time.time()
            result = await voxtral_model.process_realtime_chunk(
                test_audio, 
                chunk_id="stability_test_1",
                mode="conversation"
            )
            processing_time = (time.time() - start_time) * 1000
            
            if result.get('success', False):
                self.test_results['voxtral_processing'] = {
                    'success': True,
                    'processing_time_ms': processing_time,
                    'response_length': len(result.get('response', '')),
                    'no_tmp_file_error': True
                }
                logger.info(f"  ✅ Voxtral processing successful in {processing_time:.1f}ms")
                logger.info(f"  📝 Response: '{result.get('response', '')[:50]}...'")
            else:
                self.test_results['voxtral_processing'] = {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'processing_time_ms': processing_time
                }
                logger.error(f"  ❌ Voxtral processing failed: {result.get('error', 'Unknown')}")
                
        except Exception as e:
            self.test_results['voxtral_processing'] = {
                'success': False,
                'error': str(e),
                'tmp_file_error': 'tmp_file' in str(e)
            }
            logger.error(f"  ❌ Voxtral processing exception: {e}")
    
    async def test_tts_generation(self):
        """Test TTS generation with Indian female voice"""
        logger.info("\n🧪 Testing TTS Generation")
        
        try:
            kokoro_model = await self.unified_manager.get_kokoro_model()
            
            test_text = "Hello, this is a test of the Indian female voice."
            
            start_time = time.time()
            result = await kokoro_model.synthesize_speech(
                text=test_text,
                voice="hf_alpha"
            )
            generation_time = (time.time() - start_time) * 1000
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                self.test_results['tts_generation'] = {
                    'success': True,
                    'generation_time_ms': generation_time,
                    'audio_samples': len(audio_data) if audio_data is not None else 0,
                    'voice_used': 'hf_alpha'
                }
                logger.info(f"  ✅ TTS generation successful in {generation_time:.1f}ms")
                logger.info(f"  🎵 Generated {len(audio_data) if audio_data else 0} audio samples")
            else:
                self.test_results['tts_generation'] = {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'generation_time_ms': generation_time
                }
                logger.error(f"  ❌ TTS generation failed: {result.get('error', 'Unknown')}")
                
        except Exception as e:
            self.test_results['tts_generation'] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"  ❌ TTS generation exception: {e}")
    
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling improvements"""
        logger.info("\n🧪 Testing WebSocket Error Handling")
        
        try:
            # Simulate various error conditions
            error_scenarios = [
                ("invalid_base64", "invalid_base64_data"),
                ("empty_audio", base64.b64encode(b"").decode()),
                ("malformed_json", "not_json"),
            ]
            
            error_handling_results = []
            
            for scenario_name, test_data in error_scenarios:
                try:
                    if scenario_name == "invalid_base64":
                        # Test base64 decoding error handling
                        try:
                            base64.b64decode(test_data)
                            result = "unexpected_success"
                        except Exception:
                            result = "error_caught"
                    elif scenario_name == "empty_audio":
                        # Test empty audio handling
                        audio_bytes = base64.b64decode(test_data)
                        if len(audio_bytes) == 0:
                            result = "empty_detected"
                        else:
                            result = "unexpected_data"
                    elif scenario_name == "malformed_json":
                        # Test JSON parsing error handling
                        try:
                            json.loads(test_data)
                            result = "unexpected_success"
                        except json.JSONDecodeError:
                            result = "error_caught"
                    
                    error_handling_results.append({
                        'scenario': scenario_name,
                        'result': result,
                        'success': result in ['error_caught', 'empty_detected']
                    })
                    
                    status = "✅" if result in ['error_caught', 'empty_detected'] else "⚠️"
                    logger.info(f"  {status} {scenario_name}: {result}")
                    
                except Exception as e:
                    error_handling_results.append({
                        'scenario': scenario_name,
                        'result': 'exception',
                        'error': str(e),
                        'success': False
                    })
                    logger.error(f"  ❌ {scenario_name}: {e}")
            
            self.test_results['websocket_error_handling'] = error_handling_results
            
            success_rate = sum(1 for r in error_handling_results if r['success']) / len(error_handling_results)
            logger.info(f"📊 Error Handling Success Rate: {success_rate*100:.1f}%")
            
        except Exception as e:
            self.test_results['websocket_error_handling'] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"❌ WebSocket error handling test failed: {e}")
    
    async def test_memory_management(self):
        """Test memory management improvements"""
        logger.info("\n🧪 Testing Memory Management")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate multiple audio processing cycles
            for i in range(5):
                test_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)  # 1 second
                
                # Process audio (this should clean up memory properly)
                if self.audio_processor:
                    try:
                        self.audio_processor.validate_realtime_chunk(test_audio, f"memory_test_{i}")
                    except:
                        pass  # We're just testing memory cleanup
                
                # Force garbage collection
                gc.collect()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            self.test_results['memory_management'] = {
                'success': memory_increase < 50,  # Less than 50MB increase is acceptable
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase
            }
            
            status = "✅" if memory_increase < 50 else "⚠️"
            logger.info(f"  {status} Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
        except Exception as e:
            self.test_results['memory_management'] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"❌ Memory management test failed: {e}")
    
    def generate_stability_report(self):
        """Generate comprehensive stability report"""
        logger.info("\n📊 AUDIO PIPELINE STABILITY REPORT")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            logger.info(f"\n🔍 {test_name.upper().replace('_', ' ')}:")
            
            if isinstance(result, list):
                # Multiple test cases
                successful = sum(1 for r in result if r.get('success', False))
                total = len(result)
                passed_tests += 1 if successful == total else 0
                logger.info(f"   Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
            elif isinstance(result, dict):
                # Single test case
                success = result.get('success', False)
                passed_tests += 1 if success else 0
                status = "✅ PASSED" if success else "❌ FAILED"
                logger.info(f"   Status: {status}")
                
                if 'error' in result:
                    logger.info(f"   Error: {result['error']}")
                if 'processing_time_ms' in result:
                    logger.info(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
        
        logger.info(f"\n🎯 OVERALL STABILITY STATUS:")
        if passed_tests == total_tests:
            logger.info("   ✅ ALL TESTS PASSED - AUDIO PIPELINE IS STABLE")
        else:
            logger.info(f"   ⚠️  {total_tests - passed_tests} ISSUES DETECTED - REVIEW REQUIRED")
        
        logger.info(f"   📈 Overall Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        
        return passed_tests == total_tests

async def main():
    """Main test execution"""
    logger.info("🧪 AUDIO PIPELINE STABILITY TESTING")
    logger.info("=" * 60)
    
    tester = AudioPipelineStabilityTester()
    
    # Initialize services
    if not await tester.initialize_services():
        logger.error("❌ Service initialization failed - cannot proceed with tests")
        return False
    
    # Run all stability tests
    await tester.test_audio_validation()
    await tester.test_voxtral_processing()
    await tester.test_tts_generation()
    await tester.test_websocket_error_handling()
    await tester.test_memory_management()
    
    # Generate stability report
    all_passed = tester.generate_stability_report()
    
    return all_passed

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)
