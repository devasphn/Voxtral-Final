#!/usr/bin/env python3
"""
Performance Optimization Test Script for Voxtral TTS System
Tests VAD responsiveness and Voxtral model latency improvements
"""

import asyncio
import time
import numpy as np
import logging
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.audio_processor_realtime import AudioProcessor
from src.models.voxtral_model_realtime import voxtral_model
from src.models.speech_to_speech_pipeline import speech_to_speech_pipeline
from src.utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("performance_test")

class PerformanceOptimizationTest:
    """Test suite for performance optimizations"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.test_results = {}
        
    def generate_test_audio(self, duration_s: float = 1.0, speech_type: str = "normal") -> np.ndarray:
        """Generate synthetic audio for testing"""
        sample_rate = 16000
        samples = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, samples)
        
        if speech_type == "normal":
            # Simulate normal speech with modulated sine waves
            audio = (np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * 0.05 +
                    np.sin(2 * np.pi * 400 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)) * 0.03)
        elif speech_type == "quiet":
            # Simulate quiet speech
            audio = (np.sin(2 * np.pi * 150 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t)) * 0.02)
        elif speech_type == "loud":
            # Simulate loud speech
            audio = (np.sin(2 * np.pi * 250 * t) * (1 + 0.7 * np.sin(2 * np.pi * 6 * t)) * 0.08)
        else:  # silence
            audio = np.random.normal(0, 0.001, samples)
            
        return audio.astype(np.float32)
    
    def test_vad_responsiveness(self):
        """Test VAD responsiveness with optimized settings"""
        logger.info("🧪 Testing VAD Responsiveness...")
        
        test_cases = [
            ("normal_speech", "normal", 1.0),
            ("quiet_speech", "quiet", 1.0),
            ("loud_speech", "loud", 1.0),
            ("short_speech", "normal", 0.3),  # Test minimum duration
            ("silence", "silence", 1.0)
        ]
        
        vad_results = {}
        
        for test_name, speech_type, duration in test_cases:
            audio = self.generate_test_audio(duration, speech_type)
            
            start_time = time.time()
            vad_result = self.audio_processor.detect_voice_activity(audio, chunk_id=test_name)
            processing_time = (time.time() - start_time) * 1000
            
            vad_results[test_name] = {
                'has_voice': vad_result['has_voice'],
                'confidence': vad_result['confidence'],
                'processing_time_ms': processing_time,
                'rms_energy': vad_result['rms_energy'],
                'spectral_centroid': vad_result['spectral_centroid']
            }
            
            logger.info(f"   {test_name}: {'✅' if vad_result['has_voice'] else '❌'} "
                       f"(conf: {vad_result['confidence']:.2f}, {processing_time:.1f}ms)")
        
        self.test_results['vad'] = vad_results
        return vad_results
    
    def test_vad_timing(self):
        """Test VAD timing with new optimized durations"""
        logger.info("🧪 Testing VAD Timing Optimization...")
        
        # Reset VAD state
        self.audio_processor.reset_vad_state()
        
        # Test minimum voice duration (should be 200ms now)
        short_speech = self.generate_test_audio(0.25, "normal")  # 250ms speech
        vad_short = self.audio_processor.detect_voice_activity(short_speech, chunk_id="timing_short")
        
        # Test minimum silence duration (should be 800ms now)
        silence = self.generate_test_audio(0.9, "silence")  # 900ms silence
        vad_silence = self.audio_processor.detect_voice_activity(silence, chunk_id="timing_silence")
        
        timing_results = {
            'short_speech_detected': vad_short['has_voice'],
            'short_speech_duration_ms': vad_short.get('voice_duration_ms', 0),
            'silence_duration_ms': vad_silence.get('silence_duration_ms', 0),
            'min_voice_setting': self.audio_processor.min_voice_duration_ms,
            'min_silence_setting': self.audio_processor.min_silence_duration_ms
        }
        
        logger.info(f"   Min voice duration: {timing_results['min_voice_setting']}ms (optimized from 400ms)")
        logger.info(f"   Min silence duration: {timing_results['min_silence_setting']}ms (optimized from 1200ms)")
        logger.info(f"   Short speech (250ms): {'✅' if timing_results['short_speech_detected'] else '❌'}")
        
        self.test_results['vad_timing'] = timing_results
        return timing_results
    
    async def test_voxtral_performance(self):
        """Test Voxtral model performance with optimizations"""
        logger.info("🧪 Testing Voxtral Performance Optimization...")
        
        if not voxtral_model.is_initialized:
            logger.info("🔄 Initializing Voxtral model...")
            await voxtral_model.initialize()
        
        test_texts = [
            "Hello, how are you?",
            "What's the weather like today?",
            "Can you help me with something?",
            "Tell me a short joke.",
            "What time is it?"
        ]
        
        performance_results = []
        
        for i, text in enumerate(test_texts):
            # Generate test audio for the text
            audio = self.generate_test_audio(2.0, "normal")
            audio_tensor = self.audio_processor.preprocess_realtime_chunk(audio, chunk_id=f"perf_test_{i}")
            
            start_time = time.time()
            try:
                response = await voxtral_model.process_realtime_chunk(
                    audio_tensor, 
                    chunk_id=f"perf_test_{i}",
                    prompt=f"Respond briefly to: {text}"
                )
                processing_time = (time.time() - start_time) * 1000
                
                result = {
                    'test_input': text,
                    'processing_time_ms': processing_time,
                    'response_length': len(response) if response else 0,
                    'meets_target': processing_time <= 100,  # 100ms target
                    'success': True
                }
                
                logger.info(f"   Test {i+1}: {processing_time:.1f}ms {'✅' if result['meets_target'] else '⚠️'} "
                           f"(target: 100ms)")
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                result = {
                    'test_input': text,
                    'processing_time_ms': processing_time,
                    'error': str(e),
                    'meets_target': False,
                    'success': False
                }
                logger.error(f"   Test {i+1}: FAILED after {processing_time:.1f}ms - {e}")
            
            performance_results.append(result)
        
        # Calculate statistics
        successful_tests = [r for r in performance_results if r['success']]
        if successful_tests:
            avg_time = sum(r['processing_time_ms'] for r in successful_tests) / len(successful_tests)
            target_met_count = sum(1 for r in successful_tests if r['meets_target'])
            target_met_percentage = (target_met_count / len(successful_tests)) * 100
            
            stats = {
                'total_tests': len(performance_results),
                'successful_tests': len(successful_tests),
                'average_processing_time_ms': avg_time,
                'target_met_count': target_met_count,
                'target_met_percentage': target_met_percentage,
                'max_tokens_setting': 50  # Our optimized setting
            }
            
            logger.info(f"   📊 Average processing time: {avg_time:.1f}ms")
            logger.info(f"   📊 Target met: {target_met_count}/{len(successful_tests)} ({target_met_percentage:.1f}%)")
            logger.info(f"   📊 Max tokens optimized to: {stats['max_tokens_setting']} (was 200)")
            
        else:
            stats = {'error': 'All tests failed'}
        
        self.test_results['voxtral_performance'] = {
            'individual_results': performance_results,
            'statistics': stats
        }
        
        return performance_results
    
    async def test_end_to_end_pipeline(self):
        """Test complete speech-to-speech pipeline performance"""
        logger.info("🧪 Testing End-to-End Pipeline Performance...")
        
        if not speech_to_speech_pipeline.is_initialized:
            logger.info("🔄 Initializing Speech-to-Speech pipeline...")
            await speech_to_speech_pipeline.initialize()
        
        # Generate test audio
        test_audio = self.generate_test_audio(2.0, "normal")
        
        start_time = time.time()
        try:
            result = await speech_to_speech_pipeline.process_conversation_turn(
                test_audio,
                conversation_id="e2e_test",
                voice_preference="af_heart",
                speed_preference=1.0
            )
            
            total_time = (time.time() - start_time) * 1000
            
            pipeline_result = {
                'total_latency_ms': total_time,
                'meets_target': total_time <= 300,  # 300ms target
                'success': result['success'],
                'stage_timings': result.get('stage_timings', {}),
                'transcription_length': len(result.get('transcription', '')),
                'response_length': len(result.get('response_text', ''))
            }
            
            logger.info(f"   📊 Total pipeline latency: {total_time:.1f}ms {'✅' if pipeline_result['meets_target'] else '⚠️'} "
                       f"(target: 300ms)")
            
            if 'stage_timings' in result:
                timings = result['stage_timings']
                logger.info(f"   📊 STT: {timings.get('stt_ms', 0):.1f}ms")
                logger.info(f"   📊 LLM: {timings.get('llm_ms', 0):.1f}ms")
                logger.info(f"   📊 TTS: {timings.get('tts_ms', 0):.1f}ms")
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            pipeline_result = {
                'total_latency_ms': total_time,
                'meets_target': False,
                'success': False,
                'error': str(e)
            }
            logger.error(f"   ❌ Pipeline test failed after {total_time:.1f}ms: {e}")
        
        self.test_results['pipeline'] = pipeline_result
        return pipeline_result
    
    async def run_all_tests(self):
        """Run all performance optimization tests"""
        logger.info("🚀 Starting Performance Optimization Test Suite...")
        logger.info("=" * 60)
        
        # Test 1: VAD Responsiveness
        self.test_vad_responsiveness()
        logger.info("")
        
        # Test 2: VAD Timing
        self.test_vad_timing()
        logger.info("")
        
        # Test 3: Voxtral Performance
        await self.test_voxtral_performance()
        logger.info("")
        
        # Test 4: End-to-End Pipeline
        await self.test_end_to_end_pipeline()
        logger.info("")
        
        # Summary
        self.print_summary()
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        logger.info("📋 PERFORMANCE OPTIMIZATION TEST SUMMARY")
        logger.info("=" * 60)
        
        # VAD Summary
        if 'vad' in self.test_results:
            vad_results = self.test_results['vad']
            speech_detected = sum(1 for r in vad_results.values() if r['has_voice'])
            logger.info(f"🎤 VAD Tests: {speech_detected}/4 speech samples detected correctly")
        
        # Voxtral Summary
        if 'voxtral_performance' in self.test_results:
            stats = self.test_results['voxtral_performance']['statistics']
            if 'average_processing_time_ms' in stats:
                logger.info(f"⚡ Voxtral Performance: {stats['average_processing_time_ms']:.1f}ms average "
                           f"({stats['target_met_percentage']:.1f}% meet 100ms target)")
        
        # Pipeline Summary
        if 'pipeline' in self.test_results:
            pipeline = self.test_results['pipeline']
            if pipeline['success']:
                logger.info(f"🔄 Pipeline Performance: {pipeline['total_latency_ms']:.1f}ms total "
                           f"({'✅' if pipeline['meets_target'] else '⚠️'} 300ms target)")
        
        logger.info("=" * 60)

async def main():
    """Main test execution"""
    test_suite = PerformanceOptimizationTest()
    results = await test_suite.run_all_tests()
    
    # Save results to file
    import json
    with open('performance_test_results.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("📁 Test results saved to performance_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())
