#!/usr/bin/env python3
"""
Ultra-Low Latency Performance Test for Voxtral-Final Speech-to-Speech System
Tests all optimizations implemented for achieving ultra-low latency targets
"""

import asyncio
import time
import logging
import numpy as np
import sys
import os
from typing import Dict, List, Any

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultra_low_latency_test")

class UltraLowLatencyTester:
    """Comprehensive tester for ultra-low latency optimizations"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_targets = {
            'voxtral_processing_ms': 100,    # Target: 100ms (down from 688-1193ms)
            'kokoro_tts_ms': 50,             # Target: 50ms (maintain current performance)
            'total_end_to_end_ms': 300,      # Target: 300ms total pipeline
            'vad_processing_ms': 10,         # Target: 10ms VAD processing
            'audio_preprocessing_ms': 20,    # Target: 20ms audio preprocessing
        }
        
    async def run_comprehensive_test(self):
        """Run all ultra-low latency tests"""
        logger.info("🚀 Starting Ultra-Low Latency Comprehensive Test Suite")
        logger.info("=" * 80)
        
        try:
            # Test 1: Model Initialization Performance
            await self.test_model_initialization()
            
            # Test 2: Voxtral Model Performance
            await self.test_voxtral_performance()
            
            # Test 3: Audio Processing Performance
            await self.test_audio_processing_performance()
            
            # Test 4: End-to-End Pipeline Performance
            await self.test_end_to_end_pipeline()
            
            # Test 5: UI Simplification Validation
            await self.test_ui_simplification()
            
            # Generate comprehensive report
            self.generate_performance_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise
    
    async def test_model_initialization(self):
        """Test ultra-low latency model initialization optimizations"""
        logger.info("🧪 Testing Model Initialization Performance...")
        
        try:
            from src.models.unified_model_manager import UnifiedModelManager
            
            start_time = time.time()
            
            # Test unified model manager initialization
            model_manager = UnifiedModelManager()
            init_success = await model_manager.initialize()
            
            init_time = (time.time() - start_time) * 1000
            
            self.test_results['model_initialization'] = {
                'success': init_success,
                'initialization_time_ms': init_time,
                'target_met': init_time < 10000,  # 10 second target for initialization
                'optimizations_detected': {
                    'fast_initialization': getattr(model_manager, 'fast_initialization', False),
                    'skip_detailed_logging': getattr(model_manager, 'skip_detailed_logging', False)
                }
            }
            
            logger.info(f"✅ Model initialization completed in {init_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Model initialization test failed: {e}")
            self.test_results['model_initialization'] = {'success': False, 'error': str(e)}
    
    async def test_voxtral_performance(self):
        """Test Voxtral model ultra-low latency optimizations"""
        logger.info("🧪 Testing Voxtral Model Performance...")
        
        try:
            from src.models.voxtral_model_realtime import voxtral_model
            from src.models.audio_processor_realtime import AudioProcessor
            
            # Ensure model is initialized
            if not voxtral_model.is_initialized:
                await voxtral_model.initialize()
            
            # Create test audio (voice-like signal)
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
            test_audio = test_audio.astype(np.float32) * 0.1
            
            # Preprocess audio
            audio_processor = AudioProcessor()
            audio_tensor = audio_processor.preprocess_realtime_chunk(test_audio)
            
            # Test multiple runs for consistent performance
            processing_times = []
            for i in range(5):
                start_time = time.time()
                
                result = await voxtral_model.process_realtime_chunk(
                    audio_tensor,
                    chunk_id=f"test_{i}",
                    mode="conversation"
                )
                
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                
                logger.info(f"   Run {i+1}: {processing_time:.1f}ms - {'✅' if processing_time < self.performance_targets['voxtral_processing_ms'] else '❌'}")
            
            avg_processing_time = np.mean(processing_times)
            min_processing_time = np.min(processing_times)
            max_processing_time = np.max(processing_times)
            
            self.test_results['voxtral_performance'] = {
                'success': True,
                'avg_processing_time_ms': avg_processing_time,
                'min_processing_time_ms': min_processing_time,
                'max_processing_time_ms': max_processing_time,
                'target_met': avg_processing_time < self.performance_targets['voxtral_processing_ms'],
                'improvement_factor': 688 / avg_processing_time,  # Based on original 688ms baseline
                'optimizations_detected': {
                    'torch_compile': getattr(voxtral_model, 'use_torch_compile', False),
                    'float16_dtype': voxtral_model.torch_dtype == 'torch.float16',
                    'kv_cache_optimization': getattr(voxtral_model, 'use_kv_cache_optimization', False)
                }
            }
            
            logger.info(f"✅ Voxtral average processing time: {avg_processing_time:.1f}ms (target: {self.performance_targets['voxtral_processing_ms']}ms)")
            
        except Exception as e:
            logger.error(f"❌ Voxtral performance test failed: {e}")
            self.test_results['voxtral_performance'] = {'success': False, 'error': str(e)}
    
    async def test_audio_processing_performance(self):
        """Test audio processing and VAD performance"""
        logger.info("🧪 Testing Audio Processing Performance...")
        
        try:
            from src.models.audio_processor_realtime import AudioProcessor
            
            audio_processor = AudioProcessor()
            
            # Create test audio
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.1
            
            # Test VAD performance
            vad_times = []
            preprocessing_times = []
            
            for i in range(10):
                # Test VAD
                vad_start = time.time()
                vad_result = audio_processor.detect_voice_activity(test_audio, chunk_id=i)
                vad_time = (time.time() - vad_start) * 1000
                vad_times.append(vad_time)
                
                # Test preprocessing
                prep_start = time.time()
                audio_tensor = audio_processor.preprocess_realtime_chunk(test_audio, chunk_id=i)
                prep_time = (time.time() - prep_start) * 1000
                preprocessing_times.append(prep_time)
            
            avg_vad_time = np.mean(vad_times)
            avg_prep_time = np.mean(preprocessing_times)
            
            self.test_results['audio_processing'] = {
                'success': True,
                'avg_vad_time_ms': avg_vad_time,
                'avg_preprocessing_time_ms': avg_prep_time,
                'vad_target_met': avg_vad_time < self.performance_targets['vad_processing_ms'],
                'preprocessing_target_met': avg_prep_time < self.performance_targets['audio_preprocessing_ms'],
                'optimizations_detected': {
                    'optimized_vad_thresholds': audio_processor.min_voice_duration_ms == 200,
                    'reduced_silence_duration': audio_processor.min_silence_duration_ms == 800
                }
            }
            
            logger.info(f"✅ VAD processing: {avg_vad_time:.1f}ms, Preprocessing: {avg_prep_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Audio processing test failed: {e}")
            self.test_results['audio_processing'] = {'success': False, 'error': str(e)}
    
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline performance"""
        logger.info("🧪 Testing End-to-End Pipeline Performance...")
        
        try:
            from src.models.speech_to_speech_pipeline import SpeechToSpeechPipeline
            
            # Initialize pipeline
            pipeline = SpeechToSpeechPipeline()
            await pipeline.initialize()
            
            # Create test audio
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
            test_audio = test_audio.astype(np.float32) * 0.1
            
            # Test multiple end-to-end runs
            e2e_times = []
            for i in range(3):
                start_time = time.time()
                
                result = await pipeline.process_conversation_turn(
                    test_audio,
                    conversation_id=f"e2e_test_{i}",
                    voice_preference="af_heart",
                    speed_preference=1.0
                )
                
                e2e_time = (time.time() - start_time) * 1000
                e2e_times.append(e2e_time)
                
                logger.info(f"   E2E Run {i+1}: {e2e_time:.1f}ms - {'✅' if e2e_time < self.performance_targets['total_end_to_end_ms'] else '❌'}")
            
            avg_e2e_time = np.mean(e2e_times)
            
            self.test_results['end_to_end_pipeline'] = {
                'success': True,
                'avg_e2e_time_ms': avg_e2e_time,
                'target_met': avg_e2e_time < self.performance_targets['total_end_to_end_ms'],
                'optimizations_detected': {
                    'emotional_tts_disabled': not pipeline.enable_emotional_tts,
                    'reduced_context_size': pipeline.conversation_context.maxlen == 5,
                    'reduced_history_size': pipeline.pipeline_history.maxlen == 20
                }
            }
            
            logger.info(f"✅ End-to-end pipeline: {avg_e2e_time:.1f}ms (target: {self.performance_targets['total_end_to_end_ms']}ms)")
            
        except Exception as e:
            logger.error(f"❌ End-to-end pipeline test failed: {e}")
            self.test_results['end_to_end_pipeline'] = {'success': False, 'error': str(e)}
    
    async def test_ui_simplification(self):
        """Test UI simplification (speech-to-speech only mode)"""
        logger.info("🧪 Testing UI Simplification...")
        
        try:
            # Read the UI file to verify simplifications
            ui_file_path = "src/api/ui_server_realtime.py"
            
            with open(ui_file_path, 'r') as f:
                ui_content = f.read()
            
            # Check for removal of text mode
            text_mode_removed = 'value="transcribe"' not in ui_content
            speech_mode_default = "currentMode = 'speech_to_speech'" in ui_content
            mode_selection_simplified = 'Speech-to-Speech Conversation' in ui_content
            
            self.test_results['ui_simplification'] = {
                'success': True,
                'text_mode_removed': text_mode_removed,
                'speech_mode_default': speech_mode_default,
                'mode_selection_simplified': mode_selection_simplified,
                'all_optimizations_applied': text_mode_removed and speech_mode_default and mode_selection_simplified
            }
            
            logger.info(f"✅ UI Simplification: Text mode removed: {text_mode_removed}, Default speech mode: {speech_mode_default}")
            
        except Exception as e:
            logger.error(f"❌ UI simplification test failed: {e}")
            self.test_results['ui_simplification'] = {'success': False, 'error': str(e)}
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("=" * 80)
        logger.info("📊 ULTRA-LOW LATENCY PERFORMANCE REPORT")
        logger.info("=" * 80)
        
        # Overall success rate
        successful_tests = sum(1 for test in self.test_results.values() if test.get('success', False))
        total_tests = len(self.test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"Overall Test Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        logger.info("")
        
        # Performance targets analysis
        if 'voxtral_performance' in self.test_results and self.test_results['voxtral_performance']['success']:
            voxtral_data = self.test_results['voxtral_performance']
            logger.info(f"🎯 Voxtral Performance:")
            logger.info(f"   Current: {voxtral_data['avg_processing_time_ms']:.1f}ms")
            logger.info(f"   Target:  {self.performance_targets['voxtral_processing_ms']}ms")
            logger.info(f"   Status:  {'✅ TARGET MET' if voxtral_data['target_met'] else '❌ TARGET MISSED'}")
            logger.info(f"   Improvement: {voxtral_data.get('improvement_factor', 1):.1f}x faster than baseline")
            logger.info("")
        
        if 'end_to_end_pipeline' in self.test_results and self.test_results['end_to_end_pipeline']['success']:
            e2e_data = self.test_results['end_to_end_pipeline']
            logger.info(f"🎯 End-to-End Performance:")
            logger.info(f"   Current: {e2e_data['avg_e2e_time_ms']:.1f}ms")
            logger.info(f"   Target:  {self.performance_targets['total_end_to_end_ms']}ms")
            logger.info(f"   Status:  {'✅ TARGET MET' if e2e_data['target_met'] else '❌ TARGET MISSED'}")
            logger.info("")
        
        # Optimization summary
        logger.info("🔧 Applied Optimizations Summary:")
        for test_name, test_data in self.test_results.items():
            if test_data.get('success') and 'optimizations_detected' in test_data:
                logger.info(f"   {test_name}:")
                for opt_name, opt_status in test_data['optimizations_detected'].items():
                    logger.info(f"     - {opt_name}: {'✅' if opt_status else '❌'}")
        
        logger.info("=" * 80)
        logger.info("🎉 Ultra-Low Latency Test Suite Complete!")
        logger.info("=" * 80)

async def main():
    """Main test execution"""
    tester = UltraLowLatencyTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
