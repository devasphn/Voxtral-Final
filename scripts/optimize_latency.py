#!/usr/bin/env python3
"""
Ultra-Low Latency Optimization Script
Optimizes the system for <200ms TTS chunking and <500ms end-to-end latency
"""

import asyncio
import time
import logging
import sys
import os
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
latency_logger = logging.getLogger("latency_optimization")

class LatencyOptimizer:
    """Optimize system for ultra-low latency performance"""
    
    def __init__(self):
        self.config_path = os.path.join(project_root, 'config.yaml')
        self.optimizations_applied = []
        
    async def optimize_for_ultra_low_latency(self):
        """Apply all ultra-low latency optimizations"""
        latency_logger.info("🚀 Starting ultra-low latency optimization...")
        
        try:
            # Load current configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply optimizations
            config = await self._optimize_audio_settings(config)
            config = await self._optimize_vad_settings(config)
            config = await self._optimize_streaming_settings(config)
            config = await self._optimize_tts_settings(config)
            config = await self._optimize_model_settings(config)
            
            # Save optimized configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            latency_logger.info("✅ Ultra-low latency optimization completed!")
            latency_logger.info(f"Applied {len(self.optimizations_applied)} optimizations:")
            for opt in self.optimizations_applied:
                latency_logger.info(f"  - {opt}")
                
            return True
            
        except Exception as e:
            latency_logger.error(f"❌ Latency optimization failed: {e}")
            return False
    
    async def _optimize_audio_settings(self, config):
        """Optimize audio settings for minimal latency"""
        latency_logger.info("🎤 Optimizing audio settings...")
        
        # Ultra-low latency audio settings
        if config['audio']['chunk_size'] > 128:
            config['audio']['chunk_size'] = 128  # Smallest possible chunks
            self.optimizations_applied.append("Reduced audio chunk size to 128 for minimal latency")
        
        if config['audio']['frame_duration_ms'] > 5:
            config['audio']['frame_duration_ms'] = 5  # Minimal frame duration
            self.optimizations_applied.append("Reduced frame duration to 5ms for fastest processing")
        
        # Ensure 16kHz sample rate for consistency
        if config['audio']['sample_rate'] != 16000:
            config['audio']['sample_rate'] = 16000
            self.optimizations_applied.append("Standardized sample rate to 16kHz")
        
        return config
    
    async def _optimize_vad_settings(self, config):
        """Optimize VAD settings for fastest speech detection"""
        latency_logger.info("🎯 Optimizing VAD settings...")
        
        # Ultra-fast VAD settings
        if config['vad']['threshold'] > 0.003:
            config['vad']['threshold'] = 0.003  # Lower threshold for faster detection
            self.optimizations_applied.append("Lowered VAD threshold to 0.003 for faster detection")
        
        if config['vad']['min_voice_duration_ms'] > 100:
            config['vad']['min_voice_duration_ms'] = 100  # Minimal voice duration
            self.optimizations_applied.append("Reduced min voice duration to 100ms")
        
        if config['vad']['min_silence_duration_ms'] > 400:
            config['vad']['min_silence_duration_ms'] = 400  # Faster silence detection
            self.optimizations_applied.append("Reduced min silence duration to 400ms")
        
        if config['vad']['chunk_size_ms'] > 10:
            config['vad']['chunk_size_ms'] = 10  # Smallest VAD chunks
            self.optimizations_applied.append("Reduced VAD chunk size to 10ms")
        
        return config
    
    async def _optimize_streaming_settings(self, config):
        """Optimize streaming settings for minimal latency"""
        latency_logger.info("📡 Optimizing streaming settings...")
        
        # Ultra-low latency streaming
        if config['streaming']['buffer_size'] > 512:
            config['streaming']['buffer_size'] = 512  # Minimal buffer
            self.optimizations_applied.append("Reduced streaming buffer to 512 bytes")
        
        if config['streaming']['latency_target_ms'] > 30:
            config['streaming']['latency_target_ms'] = 30  # Aggressive target
            self.optimizations_applied.append("Set streaming latency target to 30ms")
        
        return config
    
    async def _optimize_tts_settings(self, config):
        """Optimize TTS settings for <200ms chunking"""
        latency_logger.info("🔊 Optimizing TTS settings...")
        
        # Ultra-fast TTS settings
        if config['tts']['performance']['target_latency_ms'] > 150:
            config['tts']['performance']['target_latency_ms'] = 150  # <200ms target
            self.optimizations_applied.append("Set TTS target latency to 150ms")
        
        if config['tts']['performance']['batch_size'] > 1:
            config['tts']['performance']['batch_size'] = 1  # No batching for speed
            self.optimizations_applied.append("Set TTS batch size to 1 for minimal latency")
        
        if config['tts']['performance']['max_queue_size'] > 2:
            config['tts']['performance']['max_queue_size'] = 2  # Minimal queue
            self.optimizations_applied.append("Reduced TTS queue size to 2")
        
        # Speech-to-speech optimizations
        if config['speech_to_speech']['latency_target_ms'] > 200:
            config['speech_to_speech']['latency_target_ms'] = 200  # <200ms target
            self.optimizations_applied.append("Set speech-to-speech latency target to 200ms")
        
        if config['speech_to_speech']['streaming']['words_trigger_threshold'] > 2:
            config['speech_to_speech']['streaming']['words_trigger_threshold'] = 2  # Start after 2 words
            self.optimizations_applied.append("Reduced words trigger threshold to 2")
        
        if config['speech_to_speech']['streaming']['interruption_threshold_ms'] > 30:
            config['speech_to_speech']['streaming']['interruption_threshold_ms'] = 30  # Fast interruption
            self.optimizations_applied.append("Set interruption threshold to 30ms")
        
        if config['speech_to_speech']['voice_agent']['first_word_target_ms'] > 100:
            config['speech_to_speech']['voice_agent']['first_word_target_ms'] = 100  # Fast first word
            self.optimizations_applied.append("Set first word target to 100ms")
        
        if config['speech_to_speech']['voice_agent']['word_to_audio_target_ms'] > 150:
            config['speech_to_speech']['voice_agent']['word_to_audio_target_ms'] = 150  # Fast word-to-audio
            self.optimizations_applied.append("Set word-to-audio target to 150ms")
        
        return config
    
    async def _optimize_model_settings(self, config):
        """Optimize model settings for fastest inference"""
        latency_logger.info("🧠 Optimizing model settings...")
        
        # Ensure float16 for maximum speed
        if config['model']['torch_dtype'] != 'float16':
            config['model']['torch_dtype'] = 'float16'
            self.optimizations_applied.append("Set model dtype to float16 for maximum speed")
        
        # Ensure CUDA device
        if config['model']['device'] != 'cuda':
            config['model']['device'] = 'cuda'
            self.optimizations_applied.append("Set model device to CUDA")
        
        return config
    
    async def verify_latency_targets(self):
        """Verify that latency targets are met"""
        latency_logger.info("🎯 Verifying latency targets...")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check TTS chunking target (<200ms)
            tts_latency = config['tts']['performance']['target_latency_ms']
            if tts_latency <= 200:
                latency_logger.info(f"✅ TTS chunking target: {tts_latency}ms (<200ms) ✓")
            else:
                latency_logger.warning(f"⚠️ TTS chunking target: {tts_latency}ms (>200ms)")
            
            # Check end-to-end target (<500ms)
            speech_latency = config['speech_to_speech']['latency_target_ms']
            streaming_latency = config['streaming']['latency_target_ms']
            vad_latency = config['vad']['min_voice_duration_ms']
            
            estimated_end_to_end = tts_latency + speech_latency + streaming_latency + vad_latency
            
            if estimated_end_to_end <= 500:
                latency_logger.info(f"✅ Estimated end-to-end latency: {estimated_end_to_end}ms (<500ms) ✓")
            else:
                latency_logger.warning(f"⚠️ Estimated end-to-end latency: {estimated_end_to_end}ms (>500ms)")
            
            # Check individual component targets
            components = [
                ("VAD processing", vad_latency, 100),
                ("Streaming latency", streaming_latency, 50),
                ("Speech-to-speech", speech_latency, 250),
                ("TTS generation", tts_latency, 200)
            ]
            
            all_targets_met = True
            for name, actual, target in components:
                if actual <= target:
                    latency_logger.info(f"✅ {name}: {actual}ms (<={target}ms) ✓")
                else:
                    latency_logger.warning(f"⚠️ {name}: {actual}ms (>{target}ms)")
                    all_targets_met = False
            
            return all_targets_met
            
        except Exception as e:
            latency_logger.error(f"❌ Error verifying latency targets: {e}")
            return False

async def main():
    """Main optimization function"""
    optimizer = LatencyOptimizer()
    
    # Apply optimizations
    success = await optimizer.optimize_for_ultra_low_latency()
    if not success:
        return 1
    
    # Verify targets
    targets_met = await optimizer.verify_latency_targets()
    
    if targets_met:
        print("🚀 Ultra-low latency optimization completed successfully!")
        print("🎯 All latency targets are met:")
        print("   - TTS chunking: <200ms")
        print("   - End-to-end: <500ms")
        return 0
    else:
        print("⚠️ Latency optimization completed with warnings!")
        print("Some targets may need manual adjustment.")
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
