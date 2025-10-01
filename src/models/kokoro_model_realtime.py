"""
Kokoro TTS Model for Real-time Speech Synthesis
Production-ready implementation following Voxtral patterns
"""
import asyncio
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union
from threading import Lock
from collections import deque
import soundfile as sf
import io
import librosa  # CRITICAL FIX: Add librosa for real-time resampling

from src.utils.config import config

# Setup logging
tts_logger = logging.getLogger("kokoro_tts")

class KokoroTTSModel:
    """
    Production-ready Kokoro TTS model wrapper for real-time speech synthesis
    Follows the same patterns as VoxtralModel for consistency
    """
    
    def __init__(self):
        self.pipeline = None
        self.model_lock = Lock()
        self.is_initialized = False
        
        # Real-time TTS optimization
        self.recent_generations = deque(maxlen=10)  # Track recent generations
        self.generation_history = deque(maxlen=100)  # Performance tracking
        
        # Performance optimization settings
        self.device = config.model.device
        self.torch_dtype = getattr(torch, config.model.torch_dtype)

        # TTS-specific settings
        self.sample_rate = config.tts.sample_rate
        self.voice = config.tts.voice
        self.speed = config.tts.speed
        self.lang_code = config.tts.lang_code

        # ULTRA-LOW LATENCY: Optimized for sub-500ms response time
        self.chunk_size = 1024   # ULTRA-LOW: Smaller chunks for faster processing
        self.max_text_length = 200  # ULTRA-LOW: Shorter text for faster synthesis
        self.streaming_chunk_size = 512  # ULTRA-LOW: Minimal chunks for immediate response
        self.enable_streaming_optimizations = True
        self.prefill_audio_buffer = False  # ULTRA-LOW: No buffering for immediate response
        self.use_fast_synthesis = True    # ULTRA-LOW: Maximum speed synthesis
        self.min_chunk_size_for_split = 1024  # ULTRA-LOW: Lower threshold for faster splitting

        # CRITICAL FIX: Target audio chunk duration for ultra-low latency
        self.target_chunk_duration_ms = 300  # Target 300ms chunks (was 1680ms)
        self.max_chunk_duration_ms = 500     # Maximum 500ms chunks
        
        # CRITICAL FIX: Audio pipeline standardization
        self.target_sample_rate = 16000  # Standardized sample rate for entire pipeline
        self.native_sample_rate = self.sample_rate  # Original Kokoro sample rate

        tts_logger.info(f"[AUDIO] KokoroTTSModel initialized with device: {self.device}")
        tts_logger.info(f"   [MIC] Voice: {self.voice}, Speed: {self.speed}, Lang: {self.lang_code}")
        tts_logger.info(f"   [SAMPLE] Native: {self.native_sample_rate}Hz, Target: {self.target_sample_rate}Hz")

        # OPTIMIZED: Use Hindi language code for Indian accent
        if self.lang_code == "a":
            self.lang_code = "h"  # Change to Hindi for Indian accent
            tts_logger.info(f"   [OPTIMIZED] Language code updated to: {self.lang_code} (Hindi for Indian accent)")

    def _resample_audio(self, audio_data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """
        CRITICAL FIX: Real-time audio resampling for sample rate standardization
        Converts audio from source sample rate to target sample rate
        """
        if source_sr == target_sr:
            return audio_data

        try:
            # Use librosa for high-quality resampling
            resampled_audio = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=source_sr,
                target_sr=target_sr,
                res_type='kaiser_fast'  # Fast, high-quality resampling
            )

            tts_logger.debug(f"[RESAMPLE] Audio resampled from {source_sr}Hz to {target_sr}Hz")
            tts_logger.debug(f"   [SHAPE] Original: {audio_data.shape}, Resampled: {resampled_audio.shape}")

            return resampled_audio.astype(np.float32)

        except Exception as e:
            tts_logger.error(f"[ERROR] Audio resampling failed: {e}")
            # Fallback: return original audio (may cause issues but prevents crash)
            return audio_data

    def _process_emotional_expressions(self, text: str) -> str:
        """
        Process emotional expressions in text to make them more natural for TTS
        Converts *laugh*, *sigh*, etc. to more speakable text
        """
        import re

        # Define emotional expression mappings
        emotion_mappings = {
            r'\*laugh\*': 'haha',
            r'\*chuckle\*': 'hehe',
            r'\*giggle\*': 'hehe',
            r'\*sigh\*': 'hmm',
            r'\*gasp\*': 'oh',
            r'\*whisper\*': '',  # Remove whisper markers
            r'\*shout\*': '',    # Remove shout markers
            r'\*excited\*': '',  # Remove excited markers
            r'\*sad\*': '',      # Remove sad markers
            r'\*happy\*': '',    # Remove happy markers
            r'\*surprised\*': 'oh',
            r'\*confused\*': 'um',
            r'\*thinking\*': 'hmm',
            r'\*pause\*': '...',
            r'\*breathe\*': '',
            r'\*inhale\*': '',
            r'\*exhale\*': '',
        }

        # Apply emotion mappings
        processed_text = text
        for pattern, replacement in emotion_mappings.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

        # Clean up extra spaces
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        if processed_text != text:
            tts_logger.debug(f"[EMOTION] Processed emotional expressions: '{text}' -> '{processed_text}'")

        return processed_text
    
    async def initialize(self) -> bool:
        """Initialize the Kokoro TTS model with production-ready settings"""
        if self.is_initialized:
            tts_logger.info("[AUDIO] Kokoro TTS model already initialized")
            return True

        start_time = time.time()
        tts_logger.info("[INIT] Initializing Kokoro TTS model for real-time synthesis...")

        try:
            # Check and download model files if needed
            from src.utils.kokoro_model_manager import kokoro_model_manager

            tts_logger.info("[SEARCH] Checking Kokoro model files...")
            status = kokoro_model_manager.get_model_status()

            if status['integrity_percentage'] < 100:
                tts_logger.info(f"[INPUT] Model files incomplete ({status['integrity_percentage']:.1f}%), downloading...")
                download_success = kokoro_model_manager.download_model_files()
                if not download_success:
                    tts_logger.error("[ERROR] Failed to download Kokoro model files")
                    return False
                tts_logger.info("[OK] Model files downloaded successfully")
            else:
                tts_logger.info("[OK] All model files verified and ready")

            # Import Kokoro pipeline
            from kokoro import KPipeline

            tts_logger.info(f"[INPUT] Loading Kokoro pipeline with language code: {self.lang_code}")

            # Initialize pipeline with language code
            self.pipeline = KPipeline(lang_code=self.lang_code)

            # STARTUP-OPTIMIZED: Minimal pre-warming for faster startup
            if self.enable_streaming_optimizations:
                tts_logger.info("[INIT] Quick pre-warming Kokoro pipeline...")

                # STARTUP-OPTIMIZED: Single short warm-up for faster startup
                try:
                    warm_generator = self.pipeline("Hi", voice=self.voice, speed=self.speed)
                    for i, (gs, ps, audio) in enumerate(warm_generator):
                        if i >= 0:  # Just process first chunk
                            break
                except Exception as e:
                    tts_logger.debug(f"Warm-up warning: {e}")

                tts_logger.info("[OK] Kokoro pipeline pre-warmed quickly")

            # Test the pipeline with a short sample
            test_text = "Kokoro TTS initialization test."
            tts_logger.info("[EMOJI] Testing Kokoro pipeline with sample text...")

            test_generator = self.pipeline(test_text, voice=self.voice, speed=self.speed)
            test_audio = None

            for i, (gs, ps, audio) in enumerate(test_generator):
                test_audio = audio
                break  # Just test the first chunk

            if test_audio is not None:
                tts_logger.info(f"[OK] Kokoro pipeline test successful - generated {len(test_audio)} samples")
            else:
                raise RuntimeError("Pipeline test failed - no audio generated")

            self.is_initialized = True
            init_time = time.time() - start_time
            tts_logger.info(f"[SUCCESS] Kokoro TTS model fully initialized in {init_time:.2f}s and ready for synthesis!")
            return True

        except ImportError as e:
            tts_logger.error(f"[ERROR] Failed to import Kokoro: {e}")
            tts_logger.error("[IDEA] Please install Kokoro: pip install kokoro>=0.9.4")
            return False
        except Exception as e:
            tts_logger.error(f"[ERROR] Failed to initialize Kokoro TTS model: {e}")
            import traceback
            tts_logger.error(f"[ERROR] Full error traceback: {traceback.format_exc()}")
            return False
    
    async def synthesize_speech(self, text: str, voice: Optional[str] = None, 
                               speed: Optional[float] = None, chunk_id: Optional[str] = None) -> Dict[str, Any]:
        """
        PRODUCTION-READY speech synthesis for real-time applications
        
        Args:
            text: Text to synthesize
            voice: Voice to use (optional, defaults to configured voice)
            speed: Speech speed (optional, defaults to configured speed)
            chunk_id: Unique identifier for this synthesis request
            
        Returns:
            Dict containing audio data, metadata, and performance metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Kokoro TTS model not initialized. Call initialize() first.")
        
        synthesis_start_time = time.time()
        chunk_id = chunk_id or f"tts_{int(time.time() * 1000)}"
        
        # Use provided parameters or defaults
        voice = voice or self.voice
        speed = speed or self.speed
        
        try:
            tts_logger.debug(f"[AUDIO] Synthesizing speech for chunk {chunk_id}: '{text[:50]}...'")
            
            # Validate and preprocess text
            if not text or not text.strip():
                tts_logger.warning(f"[WARN] Empty text provided for chunk {chunk_id}")
                return {
                    'audio_data': np.array([]),
                    'sample_rate': self.sample_rate,
                    'synthesis_time_ms': (time.time() - synthesis_start_time) * 1000,
                    'chunk_id': chunk_id,
                    'text_length': 0,
                    'success': True,
                    'is_empty': True
                }

            # ENHANCED: Process emotional expressions for better TTS output
            text = self._process_emotional_expressions(text)

            # Truncate text if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                tts_logger.warning(f"[WARN] Text truncated to {self.max_text_length} characters for chunk {chunk_id}")
            
            # Generate speech using Kokoro pipeline
            generator = self.pipeline(text, voice=voice, speed=speed)
            
            # Collect all audio chunks
            audio_chunks = []
            total_samples = 0
            
            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(audio)
                    total_samples += len(audio)
                    # ULTRA-OPTIMIZED: Minimal logging for maximum speed - only log every 10th chunk
                    if i % 10 == 0:
                        tts_logger.debug(f"   [EMOJI] Generated chunk {i}: {len(audio)} samples")

                    # ULTRA-LOW LATENCY: Dynamic chunk limiting based on target duration
                    current_duration_ms = (total_samples / self.target_sample_rate) * 1000
                    if current_duration_ms >= self.target_chunk_duration_ms:
                        tts_logger.debug(f"   [CHUNK] Reached target duration: {current_duration_ms:.1f}ms")
                        break

                    # ULTRA-LOW LATENCY: Hard limit to prevent oversized chunks
                    if current_duration_ms >= self.max_chunk_duration_ms:
                        tts_logger.debug(f"   [CHUNK] Hit maximum duration limit: {current_duration_ms:.1f}ms")
                        break
            
            # Concatenate all audio chunks
            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)
                tts_logger.debug(f"   [EMOJI] Concatenated {len(audio_chunks)} chunks into {len(final_audio)} samples")

                # CRITICAL FIX: Resample audio to standardized 16kHz if needed
                if self.native_sample_rate != self.target_sample_rate:
                    final_audio = self._resample_audio(
                        final_audio,
                        self.native_sample_rate,
                        self.target_sample_rate
                    )
                    tts_logger.debug(f"   [RESAMPLE] Audio resampled to {self.target_sample_rate}Hz")
            else:
                final_audio = np.array([])
                tts_logger.warning(f"[WARN] No audio generated for chunk {chunk_id}")
            
            synthesis_time = (time.time() - synthesis_start_time) * 1000
            audio_duration_s = len(final_audio) / self.target_sample_rate if len(final_audio) > 0 else 0
            
            # Track performance metrics
            performance_stats = {
                'synthesis_time_ms': synthesis_time,
                'audio_duration_s': audio_duration_s,
                'text_length': len(text),
                'audio_samples': len(final_audio),
                'real_time_factor': audio_duration_s / (synthesis_time / 1000) if synthesis_time > 0 else 0
            }
            
            self.generation_history.append(performance_stats)
            
            tts_logger.info(f"[OK] Synthesized speech for chunk {chunk_id} in {synthesis_time:.1f}ms "
                           f"({audio_duration_s:.2f}s audio, RTF: {performance_stats['real_time_factor']:.2f})")
            
            return {
                'audio_data': final_audio,
                'sample_rate': self.target_sample_rate,  # CRITICAL FIX: Return standardized sample rate
                'synthesis_time_ms': synthesis_time,
                'chunk_id': chunk_id,
                'text_length': len(text),
                'audio_duration_s': audio_duration_s,
                'success': True,
                'is_empty': False,
                'voice_used': voice,
                'speed_used': speed,
                'performance_stats': performance_stats,
                'native_sample_rate': self.native_sample_rate,
                'target_sample_rate': self.target_sample_rate
            }
            
        except Exception as e:
            synthesis_time = (time.time() - synthesis_start_time) * 1000
            tts_logger.error(f"[ERROR] Error synthesizing speech for chunk {chunk_id}: {e}")
            
            # Return error response with timing info
            error_msg = "Could not synthesize speech"
            if "CUDA out of memory" in str(e):
                error_msg = "GPU memory error during TTS"
            elif "timeout" in str(e).lower():
                error_msg = "TTS synthesis timeout"
            
            return {
                'audio_data': np.array([]),
                'sample_rate': self.sample_rate,
                'synthesis_time_ms': synthesis_time,
                'chunk_id': chunk_id,
                'text_length': len(text) if text else 0,
                'success': False,
                'error': str(e),
                'error_message': error_msg,
                'is_empty': True
            }
    
    async def synthesize_speech_streaming(self, text: str, voice: Optional[str] = None,
                                        speed: Optional[float] = None, chunk_id: Optional[str] = None):
        """
        ULTRA-LOW LATENCY: Streaming speech synthesis for real-time applications
        Yields audio chunks as they are generated instead of waiting for completion
        """
        if not self.is_initialized:
            raise RuntimeError("Kokoro TTS model not initialized")

        voice = voice or self.voice
        speed = speed or self.speed
        chunk_id = chunk_id or f"stream_{int(time.time() * 1000)}"
        synthesis_start_time = time.time()

        try:
            tts_logger.debug(f"[AUDIO] Starting streaming synthesis for chunk {chunk_id}: '{text[:50]}...'")

            # Validate and preprocess text
            if not text or not text.strip():
                tts_logger.warning(f"[WARN] Empty text provided for streaming chunk {chunk_id}")
                return

            # ENHANCED: Process emotional expressions for better TTS output
            text = self._process_emotional_expressions(text)

            # Truncate text if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                tts_logger.warning(f"[WARN] Text truncated to {self.max_text_length} characters for streaming chunk {chunk_id}")

            # ULTRA-LOW LATENCY: Generate speech using optimized streaming pipeline
            generator = self.pipeline(text, voice=voice, speed=speed)

            chunk_count = 0
            total_audio_samples = 0

            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None and len(audio) > 0:
                    total_audio_samples += len(audio)

                    # OPTIMIZED FOR WORD-BY-WORD: Send complete audio for each word without sub-chunking
                    # This reduces the number of streaming_audio messages per word

                    # Convert to proper WAV format with headers
                    if hasattr(audio, 'astype'):
                        audio_pcm = (audio * 32767).astype(np.int16)
                    else:
                        # Handle tensor case
                        audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
                        audio_pcm = (audio_np * 32767).astype(np.int16)

                    # Create WAV format with proper headers
                    audio_bytes = self._create_wav_bytes(audio_pcm)
                    chunk_count += 1

                    yield {
                        'audio_chunk': audio_bytes,
                        'chunk_index': chunk_count,
                        'is_final': False,
                        'sample_rate': self.sample_rate,
                        'synthesis_time_ms': (time.time() - synthesis_start_time) * 1000
                    }

                    # Minimal delay for smooth streaming
                    await asyncio.sleep(0.001)  # 1ms delay

                    # Log every chunk for word-by-word tracking
                    tts_logger.debug(f"[AUDIO] Word chunk {chunk_count}: {len(audio)} samples -> {len(audio_bytes)} bytes")

            synthesis_time = (time.time() - synthesis_start_time) * 1000
            tts_logger.info(f"[OK] Streaming synthesis completed in {synthesis_time:.1f}ms ({chunk_count} chunks)")

            # Send final chunk marker
            yield {
                'audio_chunk': None,
                'chunk_index': chunk_count,
                'is_final': True,
                'synthesis_time_ms': synthesis_time,
                'total_chunks': chunk_count
            }

        except Exception as e:
            tts_logger.error(f"[ERROR] Streaming synthesis failed for chunk {chunk_id}: {e}")
            # Send error marker
            yield {
                'audio_chunk': None,
                'chunk_index': 0,
                'is_final': True,
                'error': str(e),
                'synthesis_time_ms': (time.time() - synthesis_start_time) * 1000
            }

    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        # Common Kokoro voices - this could be expanded based on the model
        return [
            'af_heart', 'af_bella', 'af_sarah', 'af_nicole', 'af_sky',
            'am_adam', 'am_michael', 'am_edward', 'am_lewis', 'am_william'
        ]
    
    def set_voice_parameters(self, voice: Optional[str] = None, speed: Optional[float] = None):
        """Update voice parameters for future synthesis"""
        if voice is not None:
            self.voice = voice
            tts_logger.info(f"[MIC] Voice updated to: {voice}")
        
        if speed is not None:
            self.speed = max(0.5, min(2.0, speed))  # Clamp speed between 0.5 and 2.0
            tts_logger.info(f"[FAST] Speed updated to: {self.speed}")

    def _create_wav_bytes(self, audio_pcm: np.ndarray) -> bytes:
        """
        Create WAV format bytes with proper headers for browser compatibility

        Args:
            audio_pcm: PCM audio data as int16 numpy array

        Returns:
            bytes: Complete WAV file as bytes
        """
        try:
            # CRITICAL FIX: WAV file parameters with standardized sample rate
            sample_rate = self.target_sample_rate  # Use standardized 16kHz
            num_channels = 1  # Mono
            bits_per_sample = 16
            byte_rate = sample_rate * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8

            tts_logger.debug(f"[WAV] Creating header with sample_rate={sample_rate}Hz")

            # Convert PCM data to bytes
            pcm_bytes = audio_pcm.tobytes()
            data_size = len(pcm_bytes)
            file_size = 36 + data_size

            # Create WAV header
            wav_header = bytearray(44)

            # RIFF header
            wav_header[0:4] = b'RIFF'
            wav_header[4:8] = file_size.to_bytes(4, 'little')
            wav_header[8:12] = b'WAVE'

            # fmt chunk
            wav_header[12:16] = b'fmt '
            wav_header[16:20] = (16).to_bytes(4, 'little')  # Chunk size
            wav_header[20:22] = (1).to_bytes(2, 'little')   # Audio format (PCM)
            wav_header[22:24] = num_channels.to_bytes(2, 'little')
            wav_header[24:28] = sample_rate.to_bytes(4, 'little')
            wav_header[28:32] = byte_rate.to_bytes(4, 'little')
            wav_header[32:34] = block_align.to_bytes(2, 'little')
            wav_header[34:36] = bits_per_sample.to_bytes(2, 'little')

            # data chunk
            wav_header[36:40] = b'data'
            wav_header[40:44] = data_size.to_bytes(4, 'little')

            # Combine header and data
            wav_bytes = bytes(wav_header) + pcm_bytes

            tts_logger.debug(f"[AUDIO] Created WAV: {len(wav_bytes)} bytes ({data_size} audio + 44 header)")
            return wav_bytes

        except Exception as e:
            tts_logger.error(f"[ERROR] WAV creation failed: {e}")
            # Fallback: return raw PCM bytes
            return audio_pcm.tobytes()

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information and performance statistics"""
        base_info = {
            "model_name": "Kokoro-82M",
            "model_type": "text_to_speech",
            "is_initialized": self.is_initialized,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "current_voice": self.voice,
            "current_speed": self.speed,
            "language_code": self.lang_code,
            "available_voices": self.get_available_voices()
        }
        
        if self.is_initialized and self.generation_history:
            # Calculate performance statistics
            recent_stats = list(self.generation_history)[-10:]  # Last 10 generations
            
            if recent_stats:
                avg_synthesis_time = sum(s['synthesis_time_ms'] for s in recent_stats) / len(recent_stats)
                avg_rtf = sum(s['real_time_factor'] for s in recent_stats) / len(recent_stats)
                total_generations = len(self.generation_history)
                
                base_info.update({
                    "tts_stats": {
                        "total_generations": total_generations,
                        "avg_synthesis_time_ms": round(avg_synthesis_time, 1),
                        "avg_real_time_factor": round(avg_rtf, 2),
                        "recent_generations_in_memory": len(recent_stats),
                        "performance_history_size": len(self.generation_history)
                    }
                })
        
        return base_info

# Global model instance for real-time TTS
kokoro_model = KokoroTTSModel()

# Main execution block for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_kokoro():
        """Test the Kokoro TTS model"""
        try:
            await kokoro_model.initialize()
            
            test_text = "Hello, this is a test of the Kokoro text-to-speech system. It should generate natural sounding speech."
            result = await kokoro_model.synthesize_speech(test_text, chunk_id="test_001")
            
            if result['success'] and len(result['audio_data']) > 0:
                # Save test audio
                sf.write('test_kokoro_output.wav', result['audio_data'], result['sample_rate'])
                print(f"[OK] Test successful! Audio saved to test_kokoro_output.wav")
                print(f"   Synthesis time: {result['synthesis_time_ms']:.1f}ms")
                print(f"   Audio duration: {result['audio_duration_s']:.2f}s")
            else:
                print(f"[ERROR] Test failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"[ERROR] Test error: {e}")
    
    asyncio.run(test_kokoro())
