#!/usr/bin/env python3
"""
Model Warm-up Script for Ultra-Low Latency
Pre-loads and warms up all models to eliminate cold starts
"""

import asyncio
import time
import logging
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.unified_model_manager import UnifiedModelManager
from src.models.audio_processor_realtime import AudioProcessor
from src.utils.performance_monitor import PerformanceMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
warmup_logger = logging.getLogger("model_warmup")

class ModelWarmup:
    """Warm up all models for ultra-low latency performance"""
    
    def __init__(self):
        self.unified_manager = None
        self.audio_processor = None
        self.performance_monitor = None
        
    async def warmup_all_models(self):
        """Warm up all models and components"""
        warmup_logger.info("🔥 Starting model warm-up for ultra-low latency...")
        total_start = time.time()
        
        try:
            # Step 1: Initialize unified model manager
            warmup_logger.info("📦 Initializing unified model manager...")
            self.unified_manager = UnifiedModelManager()
            await self.unified_manager.initialize()
            
            # Step 2: Initialize audio processor
            warmup_logger.info("🎤 Initializing audio processor...")
            self.audio_processor = AudioProcessor()
            await self.audio_processor.initialize()
            
            # Step 3: Initialize performance monitor
            warmup_logger.info("📊 Initializing performance monitor...")
            self.performance_monitor = PerformanceMonitor()
            
            # Step 4: Warm up Voxtral model
            await self._warmup_voxtral()
            
            # Step 5: Warm up Kokoro TTS model
            await self._warmup_kokoro()
            
            # Step 6: Test end-to-end pipeline
            await self._test_pipeline()
            
            total_time = time.time() - total_start
            warmup_logger.info(f"✅ Model warm-up completed in {total_time:.2f}s")
            warmup_logger.info("🚀 System ready for ultra-low latency operation!")
            
            return True
            
        except Exception as e:
            warmup_logger.error(f"❌ Model warm-up failed: {e}")
            return False
    
    async def _warmup_voxtral(self):
        """Warm up Voxtral model with dummy audio"""
        warmup_logger.info("🎯 Warming up Voxtral model...")
        
        try:
            # Generate dummy audio (1 second of silence)
            sample_rate = 16000
            dummy_audio = np.zeros(sample_rate, dtype=np.float32)
            
            # Process dummy audio to warm up the model
            voxtral_model = self.unified_manager.voxtral_model
            if voxtral_model:
                start_time = time.time()
                
                # Warm up with a few dummy requests
                for i in range(3):
                    result = await voxtral_model.process_realtime_chunk(
                        dummy_audio, 
                        chunk_id=f"warmup_{i}",
                        mode="conversation"
                    )
                    warmup_logger.info(f"   Warmup {i+1}/3: {result.get('processing_time_ms', 0):.1f}ms")
                
                warmup_time = time.time() - start_time
                warmup_logger.info(f"✅ Voxtral warmed up in {warmup_time:.2f}s")
            else:
                warmup_logger.warning("⚠️ Voxtral model not available for warmup")
                
        except Exception as e:
            warmup_logger.error(f"❌ Voxtral warmup failed: {e}")
    
    async def _warmup_kokoro(self):
        """Warm up Kokoro TTS model with dummy text"""
        warmup_logger.info("🔊 Warming up Kokoro TTS model...")
        
        try:
            kokoro_model = await self.unified_manager.get_kokoro_model()
            if kokoro_model:
                start_time = time.time()
                
                # Warm up with a few dummy TTS requests
                test_texts = [
                    "Hello, this is a test.",
                    "The system is warming up.",
                    "Ready for conversation."
                ]
                
                for i, text in enumerate(test_texts):
                    result = await kokoro_model.synthesize_speech(
                        text=text,
                        voice="hf_alpha"
                    )
                    if result.get('success'):
                        warmup_logger.info(f"   Warmup {i+1}/3: {result.get('generation_time_ms', 0):.1f}ms")
                    else:
                        warmup_logger.warning(f"   Warmup {i+1}/3: Failed")
                
                warmup_time = time.time() - start_time
                warmup_logger.info(f"✅ Kokoro TTS warmed up in {warmup_time:.2f}s")
            else:
                warmup_logger.warning("⚠️ Kokoro TTS model not available for warmup")
                
        except Exception as e:
            warmup_logger.error(f"❌ Kokoro TTS warmup failed: {e}")
    
    async def _test_pipeline(self):
        """Test the complete pipeline end-to-end"""
        warmup_logger.info("🔄 Testing end-to-end pipeline...")
        
        try:
            # Generate test audio
            sample_rate = 16000
            duration = 2.0  # 2 seconds
            test_audio = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)
            
            # Test complete pipeline
            start_time = time.time()
            
            # 1. Audio preprocessing
            audio_tensor = self.audio_processor.preprocess_realtime_chunk(test_audio, "pipeline_test")
            
            # 2. Voxtral processing
            voxtral_result = await self.unified_manager.voxtral_model.process_realtime_chunk(
                audio_tensor,
                chunk_id="pipeline_test",
                mode="conversation"
            )
            
            # 3. TTS generation (if we got a response)
            if voxtral_result.get('success') and voxtral_result.get('response'):
                kokoro_model = await self.unified_manager.get_kokoro_model()
                tts_result = await kokoro_model.synthesize_speech(
                    text="Pipeline test successful.",
                    voice="hf_alpha"
                )
                
                if tts_result.get('success'):
                    pipeline_time = time.time() - start_time
                    warmup_logger.info(f"✅ End-to-end pipeline test: {pipeline_time*1000:.1f}ms")
                else:
                    warmup_logger.warning("⚠️ TTS generation failed in pipeline test")
            else:
                warmup_logger.info("ℹ️ Pipeline test completed (no speech detected in test audio)")
                
        except Exception as e:
            warmup_logger.error(f"❌ Pipeline test failed: {e}")

async def main():
    """Main warmup function"""
    warmup = ModelWarmup()
    success = await warmup.warmup_all_models()
    
    if success:
        print("🚀 Model warmup completed successfully!")
        return 0
    else:
        print("❌ Model warmup failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
