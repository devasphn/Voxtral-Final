#!/usr/bin/env python3
"""
VAD Optimization Test Script - Standalone test for VAD improvements
Tests only the VAD system without requiring Voxtral/Kokoro dependencies
"""

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vad_test")

class VADOptimizationTest:
    """Test suite for VAD optimizations only"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        
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
        elif speech_type == "partial":
            # Simulate partial speech (what was causing issues)
            audio = np.concatenate([
                np.sin(2 * np.pi * 200 * t[:samples//3]) * 0.05,  # "hello"
                np.random.normal(0, 0.001, samples//3),           # brief pause
                np.sin(2 * np.pi * 180 * t[:samples//3]) * 0.04   # "how is it going"
            ])
        else:  # silence
            audio = np.random.normal(0, 0.001, samples)
            
        return audio.astype(np.float32)
    
    def test_vad_optimization_comparison(self):
        """Compare old vs new VAD settings"""
        logger.info("🧪 Testing VAD Optimization - Before vs After Comparison")
        logger.info("=" * 60)
        
        # Test cases that were problematic before
        test_cases = [
            ("normal_speech", "normal", 1.0),
            ("quiet_speech", "quiet", 1.0),
            ("partial_speech", "partial", 1.0),  # This was the main issue
            ("short_speech", "normal", 0.25),    # 250ms speech
            ("very_short", "normal", 0.15),      # 150ms speech
            ("silence", "silence", 1.0)
        ]
        
        logger.info("Current Optimized Settings:")
        logger.info(f"   Min voice duration: {self.audio_processor.min_voice_duration_ms}ms (was 400ms)")
        logger.info(f"   Min silence duration: {self.audio_processor.min_silence_duration_ms}ms (was 1200ms)")
        logger.info(f"   VAD threshold: {self.audio_processor.vad_threshold} (was 0.015)")
        logger.info(f"   Energy threshold: {self.audio_processor.energy_threshold} (was 3e-6)")
        logger.info("")
        
        results = {}
        
        for test_name, speech_type, duration in test_cases:
            logger.info(f"Testing {test_name} ({speech_type}, {duration}s):")
            
            # Reset VAD state for each test
            self.audio_processor.reset_vad_state()
            
            audio = self.generate_test_audio(duration, speech_type)
            
            start_time = time.time()
            vad_result = self.audio_processor.detect_voice_activity(audio, chunk_id=test_name)
            processing_time = (time.time() - start_time) * 1000
            
            # Test validation (which includes VAD)
            validation_start = time.time()
            is_valid = self.audio_processor.validate_realtime_chunk(audio, chunk_id=test_name)
            validation_time = (time.time() - validation_start) * 1000
            
            results[test_name] = {
                'has_voice': vad_result['has_voice'],
                'confidence': vad_result['confidence'],
                'rms_energy': vad_result['rms_energy'],
                'spectral_centroid': vad_result['spectral_centroid'],
                'processing_time_ms': processing_time,
                'validation_passed': is_valid,
                'validation_time_ms': validation_time
            }
            
            # Determine expected result
            expected_voice = speech_type != "silence"
            result_icon = "✅" if (vad_result['has_voice'] == expected_voice) else "❌"
            
            logger.info(f"   VAD Result: {result_icon} Voice={vad_result['has_voice']} "
                       f"(conf: {vad_result['confidence']:.2f}, RMS: {vad_result['rms_energy']:.4f})")
            logger.info(f"   Validation: {'✅' if is_valid else '❌'} "
                       f"(VAD: {processing_time:.1f}ms, Total: {validation_time:.1f}ms)")
            logger.info("")
        
        return results
    
    def test_timing_improvements(self):
        """Test the timing improvements specifically"""
        logger.info("🧪 Testing Timing Improvements")
        logger.info("=" * 60)
        
        # Simulate the original problem: "hello, how is it going?" being cut to "hello"
        logger.info("Simulating original problem: partial speech capture")
        
        # Reset VAD state
        self.audio_processor.reset_vad_state()
        
        # Create a sequence that simulates natural speech with brief pauses
        sample_rate = 16000
        
        # "hello" - 300ms
        hello_audio = self.generate_test_audio(0.3, "normal")
        
        # brief pause - 100ms
        pause_audio = self.generate_test_audio(0.1, "silence")
        
        # "how is it going" - 800ms
        continuation_audio = self.generate_test_audio(0.8, "normal")
        
        # Test each part separately to see VAD behavior
        logger.info("Testing speech segments:")
        
        # Test "hello" part
        vad_hello = self.audio_processor.detect_voice_activity(hello_audio, chunk_id="hello")
        logger.info(f"   'hello' (300ms): Voice={vad_hello['has_voice']} "
                   f"(conf: {vad_hello['confidence']:.2f})")
        
        # Test pause
        vad_pause = self.audio_processor.detect_voice_activity(pause_audio, chunk_id="pause")
        logger.info(f"   pause (100ms): Voice={vad_pause['has_voice']} "
                   f"(conf: {vad_pause['confidence']:.2f})")
        
        # Test continuation
        vad_continuation = self.audio_processor.detect_voice_activity(continuation_audio, chunk_id="continuation")
        logger.info(f"   'how is it going' (800ms): Voice={vad_continuation['has_voice']} "
                   f"(conf: {vad_continuation['confidence']:.2f})")
        
        # Test combined audio
        combined_audio = np.concatenate([hello_audio, pause_audio, continuation_audio])
        self.audio_processor.reset_vad_state()
        vad_combined = self.audio_processor.detect_voice_activity(combined_audio, chunk_id="combined")
        
        logger.info(f"   Combined (1.2s total): Voice={vad_combined['has_voice']} "
                   f"(conf: {vad_combined['confidence']:.2f})")
        
        # Check if the optimization helps with continuity
        logger.info("")
        logger.info("Optimization Impact:")
        logger.info(f"   With 200ms min voice duration: Short segments more likely to be detected")
        logger.info(f"   With 800ms silence requirement: Faster processing after speech ends")
        logger.info(f"   More permissive logic: Better handling of natural speech patterns")
        
        return {
            'hello_detected': vad_hello['has_voice'],
            'pause_detected': vad_pause['has_voice'],
            'continuation_detected': vad_continuation['has_voice'],
            'combined_detected': vad_combined['has_voice']
        }
    
    def test_sensitivity_levels(self):
        """Test different VAD sensitivity levels"""
        logger.info("🧪 Testing VAD Sensitivity Levels")
        logger.info("=" * 60)
        
        # Test all sensitivity levels
        sensitivity_levels = ["low", "medium", "high"]
        test_audio = self.generate_test_audio(1.0, "quiet")  # Use quiet speech for sensitivity test
        
        results = {}
        
        for sensitivity in sensitivity_levels:
            logger.info(f"Testing {sensitivity.upper()} sensitivity:")
            
            # Adjust sensitivity
            self.audio_processor.adjust_vad_sensitivity(sensitivity)
            self.audio_processor.reset_vad_state()
            
            # Test VAD
            vad_result = self.audio_processor.detect_voice_activity(test_audio, chunk_id=f"sens_{sensitivity}")
            
            results[sensitivity] = {
                'has_voice': vad_result['has_voice'],
                'confidence': vad_result['confidence'],
                'settings': {
                    'vad_threshold': self.audio_processor.vad_threshold,
                    'min_voice_duration_ms': self.audio_processor.min_voice_duration_ms,
                    'min_silence_duration_ms': self.audio_processor.min_silence_duration_ms
                }
            }
            
            logger.info(f"   Voice detected: {'✅' if vad_result['has_voice'] else '❌'} "
                       f"(conf: {vad_result['confidence']:.2f})")
            logger.info(f"   Settings: threshold={self.audio_processor.vad_threshold}, "
                       f"min_voice={self.audio_processor.min_voice_duration_ms}ms, "
                       f"min_silence={self.audio_processor.min_silence_duration_ms}ms")
            logger.info("")
        
        # Reset to medium (default optimized)
        self.audio_processor.adjust_vad_sensitivity("medium")
        
        return results
    
    def run_all_tests(self):
        """Run all VAD optimization tests"""
        logger.info("🚀 Starting VAD Optimization Test Suite")
        logger.info("=" * 80)
        
        results = {}
        
        # Test 1: Optimization comparison
        results['optimization'] = self.test_vad_optimization_comparison()
        
        # Test 2: Timing improvements
        results['timing'] = self.test_timing_improvements()
        
        # Test 3: Sensitivity levels
        results['sensitivity'] = self.test_sensitivity_levels()
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print test summary"""
        logger.info("📋 VAD OPTIMIZATION TEST SUMMARY")
        logger.info("=" * 80)
        
        # Optimization results
        if 'optimization' in results:
            opt_results = results['optimization']
            speech_tests = [k for k in opt_results.keys() if k != 'silence']
            speech_detected = sum(1 for k in speech_tests if opt_results[k]['has_voice'])
            silence_correct = not opt_results.get('silence', {}).get('has_voice', True)
            
            logger.info(f"🎤 Speech Detection: {speech_detected}/{len(speech_tests)} speech samples detected")
            logger.info(f"🔇 Silence Detection: {'✅' if silence_correct else '❌'} silence correctly identified")
        
        # Timing results
        if 'timing' in results:
            timing = results['timing']
            logger.info(f"🕐 Timing Test: Combined speech {'✅' if timing['combined_detected'] else '❌'} detected")
        
        # Performance improvements
        logger.info("")
        logger.info("🚀 PERFORMANCE IMPROVEMENTS:")
        logger.info(f"   ⚡ Min voice duration: 400ms → 200ms (50% faster trigger)")
        logger.info(f"   ⚡ Min silence duration: 1200ms → 800ms (33% faster processing)")
        logger.info(f"   ⚡ More sensitive thresholds for better speech capture")
        logger.info(f"   ⚡ Simplified logic for more permissive detection")
        
        logger.info("")
        logger.info("✅ VAD optimization testing completed!")
        logger.info("=" * 80)

def main():
    """Main test execution"""
    test_suite = VADOptimizationTest()
    results = test_suite.run_all_tests()
    
    # Save results
    import json
    with open('vad_optimization_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("📁 VAD test results saved to vad_optimization_results.json")

if __name__ == "__main__":
    main()
