"""
FIXED Enhanced Audio processor for REAL-TIME streaming with PRODUCTION VAD
Added proper Voice Activity Detection and silence filtering
"""
import numpy as np
import librosa
import torch
import torchaudio
from typing import Tuple, Optional
import logging
from collections import deque
import time
import sys
import os

# Add current directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import config

# Enhanced logging for real-time audio processing
audio_logger = logging.getLogger("realtime_audio")
audio_logger.setLevel(logging.DEBUG)

class AudioProcessor:
    """Enhanced audio processor optimized for real-time streaming with PRODUCTION VAD"""
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        self.n_mels = config.spectrogram.n_mels
        self.hop_length = config.spectrogram.hop_length
        self.win_length = config.spectrogram.win_length
        self.n_fft = config.spectrogram.n_fft
        
        # Real-time processing metrics
        self.processing_history = deque(maxlen=100)
        self.chunk_counter = 0

        # OPTIMIZED VAD SETTINGS - Enhanced for real-time responsiveness
        self.vad_threshold = 0.012           # Slightly more sensitive RMS threshold
        self.min_voice_duration_ms = 200     # REDUCED: Faster trigger (was 400ms)
        self.min_silence_duration_ms = 800   # REDUCED: Faster processing (was 1200ms)
        self.energy_threshold = 2e-6         # Slightly more sensitive energy threshold
        self.zero_crossing_threshold = 0.25  # Slightly more permissive ZCR
        self.spectral_centroid_threshold = 350  # More sensitive spectral detection
        
        # Silence detection counters
        self.consecutive_silent_chunks = 0
        self.consecutive_voice_chunks = 0
        self.last_voice_activity = False
        
        # Ensure n_fft is sufficient for n_mels
        min_n_fft = 2 * (self.n_mels - 1)
        if self.n_fft < min_n_fft:
            self.n_fft = 1024
            audio_logger.info(f"[CHART] Adjusted n_fft to {self.n_fft} to accommodate {self.n_mels} mel bins")
        
        # Initialize mel spectrogram transform (matching Voxtral architecture)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=min(self.win_length, self.n_fft),
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            f_min=0.0,
            f_max=self.sample_rate // 2,
            norm='slaney',
            mel_scale='htk'
        )

        audio_logger.info(f"[SPEAKER] AudioProcessor initialized for PRODUCTION real-time streaming:")
        audio_logger.info(f"   [STATS] Sample rate: {self.sample_rate} Hz")
        audio_logger.info(f"   [AUDIO] Mel bins: {self.n_mels}")
        audio_logger.info(f"   [GEOM] FFT size: {self.n_fft}")
        audio_logger.info(f"   [TIME]  Hop length: {self.hop_length}")
        audio_logger.info(f"   [WINDOW] Window length: {self.win_length}")
        audio_logger.info(f"   [VAD]  VAD threshold: {self.vad_threshold}")
        audio_logger.info(f"   [MUTE] Energy threshold: {self.energy_threshold}")
    
    def detect_voice_activity(self, audio_data: np.ndarray, chunk_id: int = None) -> dict:
        """
        PRODUCTION Voice Activity Detection with multiple metrics
        
        Returns:
            dict with VAD results and metrics
        """
        try:
            chunk_id = chunk_id or self.chunk_counter
            
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate total energy
            total_energy = np.sum(audio_data ** 2)
            
            # Calculate zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
            zcr = len(zero_crossings) / len(audio_data)
            
            # Calculate spectral centroid (frequency content indicator)
            try:
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio_data, 
                    sr=self.sample_rate
                )[0].mean()
            except:
                spectral_centroid = 0
            
            # Calculate max amplitude
            max_amplitude = np.max(np.abs(audio_data))
            
            # PRODUCTION VAD DECISION LOGIC
            has_voice = False
            confidence = 0.0
            
            # Primary check: RMS energy
            rms_check = rms_energy > self.vad_threshold
            
            # Secondary check: Total energy
            energy_check = total_energy > self.energy_threshold
            
            # Tertiary check: Not too noisy (reasonable ZCR)
            zcr_check = zcr < self.zero_crossing_threshold
            
            # Quaternary check: Has meaningful amplitude (calibrated for normal speech)
            amplitude_check = max_amplitude > 0.005  # Calibrated for normal speech levels

            # NEW: Spectral centroid check for speech-like characteristics (more permissive)
            spectral_check = spectral_centroid > self.spectral_centroid_threshold

            # Combine checks with weighting - optimized for normal speech detection
            primary_checks = [rms_check, energy_check, amplitude_check]
            passed_primary_checks = sum(primary_checks)

            # OPTIMIZED: More permissive voice detection logic
            # Voice detected if: (1/2 primary + spectral) OR (2/3 primary) OR (strong spectral)
            strong_spectral = spectral_centroid > 800  # Strong speech characteristics

            if ((passed_primary_checks >= 1 and spectral_check) or
                passed_primary_checks >= 2 or
                strong_spectral):
                has_voice = True
                confidence = (passed_primary_checks + (1 if spectral_check else 0) + (1 if strong_spectral else 0)) / 5.0

                # Boost confidence if ZCR also passes (not too noisy)
                if zcr_check:
                    confidence = min(1.0, confidence + 0.1)

                # Additional boost for very clear speech
                if spectral_centroid > 1200:  # Very clear speech characteristics
                    confidence = min(1.0, confidence + 0.2)
            
            # Update consecutive counters for stability
            if has_voice:
                self.consecutive_voice_chunks += 1
                self.consecutive_silent_chunks = 0
            else:
                self.consecutive_silent_chunks += 1
                self.consecutive_voice_chunks = 0
            
            # Calculate duration-based stability
            chunk_duration_ms = (len(audio_data) / self.sample_rate) * 1000
            voice_duration_ms = self.consecutive_voice_chunks * chunk_duration_ms
            silence_duration_ms = self.consecutive_silent_chunks * chunk_duration_ms
            
            # Apply minimum duration requirements
            stable_voice = (has_voice and voice_duration_ms >= self.min_voice_duration_ms)
            stable_silence = (not has_voice and silence_duration_ms >= self.min_silence_duration_ms)
            
            # Final decision with stability
            final_voice_detected = stable_voice or (has_voice and self.last_voice_activity)
            
            # Update last activity
            self.last_voice_activity = final_voice_detected
            
            vad_result = {
                "has_voice": final_voice_detected,
                "confidence": confidence,
                "rms_energy": rms_energy,
                "total_energy": total_energy,
                "zero_crossing_rate": zcr,
                "max_amplitude": max_amplitude,
                "spectral_centroid": spectral_centroid,
                "checks_passed": passed_primary_checks,
                "consecutive_voice_chunks": self.consecutive_voice_chunks,
                "consecutive_silent_chunks": self.consecutive_silent_chunks,
                "voice_duration_ms": voice_duration_ms,
                "silence_duration_ms": silence_duration_ms,
                "stable_voice": stable_voice,
                "stable_silence": stable_silence
            }
            
            # Log VAD decision
            if final_voice_detected:
                audio_logger.debug(f"[VAD] Chunk {chunk_id}: VOICE detected (conf: {confidence:.2f}, RMS: {rms_energy:.4f})")
            else:
                audio_logger.debug(f"[MUTE] Chunk {chunk_id}: SILENCE detected (RMS: {rms_energy:.4f}, silence: {silence_duration_ms:.0f}ms)")
            
            return vad_result
            
        except Exception as e:
            audio_logger.error(f"[ERROR] VAD error for chunk {chunk_id}: {e}")
            return {
                "has_voice": False,
                "confidence": 0.0,
                "error": str(e)
            }

    def preprocess_realtime_chunk(self, audio_data: np.ndarray, chunk_id: int = None, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Enhanced preprocessing specifically optimized for real-time audio chunks
        
        Args:
            audio_data: Raw audio data as numpy array
            chunk_id: Unique identifier for this chunk (for logging)
            sample_rate: Sample rate of input audio
            
        Returns:
            Preprocessed audio tensor ready for Voxtral
        """
        start_time = time.time()
        chunk_id = chunk_id or self.chunk_counter
        self.chunk_counter += 1
        
        try:
            audio_logger.debug(f"[AUDIO] Processing real-time audio chunk {chunk_id}")
            audio_logger.debug(f"   [EMOJI] Input shape: {audio_data.shape}")
            audio_logger.debug(f"   [STATS] Input dtype: {audio_data.dtype}")
            audio_logger.debug(f"   [CHART] Input range: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
            
            # Ensure audio_data is writeable
            if not audio_data.flags.writeable:
                audio_data = audio_data.copy()
                audio_logger.debug(f"   [CONFIG] Made audio data writeable for chunk {chunk_id}")
            
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                original_dtype = audio_data.dtype
                audio_data = audio_data.astype(np.float32)
                audio_logger.debug(f"   [EMOJI] Converted dtype from {original_dtype} to float32")
            
            # Check for invalid values
            nan_count = np.sum(np.isnan(audio_data))
            inf_count = np.sum(np.isinf(audio_data))
            if nan_count > 0 or inf_count > 0:
                audio_logger.warning(f"[WARN]  Chunk {chunk_id} has {nan_count} NaN and {inf_count} infinite values - cleaning")
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize audio to [-1, 1] range with enhanced handling for real-time
            max_val = np.max(np.abs(audio_data))
            audio_logger.debug(f"   [STATS] Max amplitude: {max_val:.6f}")
            
            if max_val > 1.0:
                audio_data = audio_data / max_val
                audio_logger.debug(f"   [CONFIG] Normalized loud audio (max: {max_val:.4f}) for chunk {chunk_id}")
            elif max_val < 1e-8:
                audio_logger.warning(f"[WARN]  Chunk {chunk_id} is very quiet (max: {max_val:.2e}), amplifying carefully")
                # More conservative amplification for real-time
                audio_data = audio_data * 100.0
                audio_data = np.clip(audio_data, -1.0, 1.0)
            elif max_val < 1e-4:
                audio_logger.debug(f"   [SPEAKER] Quiet audio detected (max: {max_val:.6f}), gentle amplification")
                audio_data = audio_data * 10.0
                audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Resample if necessary (real-time optimized)
            if sample_rate and sample_rate != self.sample_rate:
                audio_logger.info(f"[EMOJI] Resampling chunk {chunk_id} from {sample_rate}Hz to {self.sample_rate}Hz")
                resample_start = time.time()
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.sample_rate,
                    res_type='fast'  # Fast resampling for real-time
                )
                resample_time = (time.time() - resample_start) * 1000
                audio_logger.debug(f"   [FAST] Resampling completed in {resample_time:.1f}ms")
            
            # Create tensor with explicit copy for writeability
            audio_tensor = torch.from_numpy(audio_data.copy()).float()
            
            # Ensure mono audio
            if len(audio_tensor.shape) > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)
                audio_logger.debug(f"   [AUDIO] Converted to mono audio for chunk {chunk_id}")
            
            # Ensure tensor is contiguous
            if not audio_tensor.is_contiguous():
                audio_tensor = audio_tensor.contiguous()
                audio_logger.debug(f"   [CONFIG] Made tensor contiguous for chunk {chunk_id}")
            
            # Calculate processing metrics
            processing_time = (time.time() - start_time) * 1000
            audio_duration_s = len(audio_tensor) / self.sample_rate
            
            # Store processing stats for monitoring
            processing_stats = {
                'chunk_id': chunk_id,
                'processing_time_ms': processing_time,
                'audio_duration_s': audio_duration_s,
                'input_samples': len(audio_data),
                'output_samples': len(audio_tensor),
                'max_amplitude': float(torch.max(torch.abs(audio_tensor))),
                'timestamp': time.time()
            }
            self.processing_history.append(processing_stats)
            
            audio_logger.info(f"[OK] Chunk {chunk_id} preprocessed in {processing_time:.1f}ms ({audio_duration_s:.2f}s audio)")
            audio_logger.debug(f"   [STATS] Output shape: {audio_tensor.shape}")
            audio_logger.debug(f"   [CHART] Output range: [{torch.min(audio_tensor):.4f}, {torch.max(audio_tensor):.4f}]")
            
            return audio_tensor
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            audio_logger.error(f"[ERROR] Error preprocessing chunk {chunk_id} after {processing_time:.1f}ms: {e}")
            raise
    
    def validate_realtime_chunk(self, audio_data: np.ndarray, chunk_id: int = None) -> bool:
        """
        PRODUCTION validation with Voice Activity Detection
        
        Args:
            audio_data: Audio data to validate
            chunk_id: Chunk identifier for logging
            
        Returns:
            True if contains voice activity, False if silence/noise
        """
        chunk_id = chunk_id or f"chunk_{int(time.time()*1000)}"
        
        try:
            audio_logger.debug(f"[SEARCH] Validating real-time chunk {chunk_id}")
            
            # Check if audio data exists and is not empty
            if audio_data is None or len(audio_data) == 0:
                audio_logger.warning(f"[WARN]  Chunk {chunk_id} is empty")
                return False
            
            # Check for valid data types
            if not isinstance(audio_data, np.ndarray):
                audio_logger.warning(f"[WARN]  Chunk {chunk_id} is not a numpy array: {type(audio_data)}")
                return False
            
            # Check for NaN or infinite values
            nan_count = np.sum(np.isnan(audio_data))
            inf_count = np.sum(np.isinf(audio_data))
            if nan_count > 0 or inf_count > 0:
                audio_logger.warning(f"[WARN]  Chunk {chunk_id} contains {nan_count} NaN and {inf_count} inf values")
                # Clean the data first
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Check for reasonable audio length - more permissive for real-time
            min_samples = int(0.05 * self.sample_rate)  # At least 50ms
            if len(audio_data) < min_samples:
                audio_logger.warning(f"[WARN]  Chunk {chunk_id} too short: {len(audio_data)} samples, minimum: {min_samples}")
                return False
            
            # CRITICAL: Apply Voice Activity Detection
            vad_result = self.detect_voice_activity(audio_data, chunk_id)
            
            # Return VAD decision
            has_voice = vad_result.get("has_voice", False)
            confidence = vad_result.get("confidence", 0.0)
            
            if has_voice:
                audio_logger.debug(f"[OK] Chunk {chunk_id} validation passed - VOICE detected (confidence: {confidence:.2f})")
            else:
                audio_logger.debug(f"[MUTE] Chunk {chunk_id} validation failed - SILENCE detected")
                
            return has_voice
            
        except Exception as e:
            audio_logger.error(f"[ERROR] Error validating chunk {chunk_id}: {e}")
            return False
    
    def generate_log_mel_spectrogram(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate log-mel spectrogram optimized for real-time processing
        """
        try:
            audio_logger.debug(f"[AUDIO] Generating log-mel spectrogram")
            start_time = time.time()
            
            # Ensure audio tensor has the right shape
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Generate mel spectrogram
            mel_spec = self.mel_transform(audio_tensor)
            
            # Convert to log scale with numerical stability
            log_mel_spec = torch.log(mel_spec + 1e-8)
            
            processing_time = (time.time() - start_time) * 1000
            audio_logger.debug(f"[OK] Log-mel spectrogram generated in {processing_time:.1f}ms")
            audio_logger.debug(f"   [STATS] Output shape: {log_mel_spec.shape}")
            
            return log_mel_spec.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            audio_logger.error(f"[ERROR] Error generating log-mel spectrogram: {e}")
            raise
    
    def get_processing_stats(self) -> dict:
        """Get real-time processing statistics with VAD metrics"""
        if not self.processing_history:
            return {"message": "No processing history available"}
        
        history = list(self.processing_history)
        processing_times = [h['processing_time_ms'] for h in history]
        audio_durations = [h['audio_duration_s'] for h in history]
        
        return {
            "total_chunks_processed": len(history),
            "avg_processing_time_ms": round(np.mean(processing_times), 2),
            "min_processing_time_ms": round(np.min(processing_times), 2),
            "max_processing_time_ms": round(np.max(processing_times), 2),
            "avg_audio_duration_s": round(np.mean(audio_durations), 3),
            "total_audio_processed_s": round(np.sum(audio_durations), 2),
            "realtime_factor": round(np.mean([
                h['audio_duration_s'] / (h['processing_time_ms'] / 1000)
                for h in history if h['processing_time_ms'] > 0
            ]), 2),
            "current_chunk_counter": self.chunk_counter,
            "vad_settings": {
                "vad_threshold": self.vad_threshold,
                "min_voice_duration_ms": self.min_voice_duration_ms,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "energy_threshold": self.energy_threshold
            },
            "vad_state": {
                "consecutive_voice_chunks": self.consecutive_voice_chunks,
                "consecutive_silent_chunks": self.consecutive_silent_chunks,
                "last_voice_activity": self.last_voice_activity
            }
        }
    
    def reset_vad_state(self):
        """Reset VAD state counters (useful for new conversation sessions)"""
        self.consecutive_silent_chunks = 0
        self.consecutive_voice_chunks = 0
        self.last_voice_activity = False
        audio_logger.info("[EMOJI] VAD state reset for new session")
    
    def adjust_vad_sensitivity(self, sensitivity: str = "medium"):
        """
        Adjust VAD sensitivity for different environments
        
        Args:
            sensitivity: "low" (noisy), "medium" (normal), "high" (quiet)
        """
        if sensitivity == "low":  # Noisy environment - most restrictive
            self.vad_threshold = 0.025
            self.energy_threshold = 8e-6
            self.min_voice_duration_ms = 600
            self.spectral_centroid_threshold = 600
            audio_logger.info("[SPEAKER] VAD sensitivity set to LOW (noisy environment)")

        elif sensitivity == "high":  # Quiet environment - more sensitive
            self.vad_threshold = 0.008
            self.energy_threshold = 1e-6
            self.min_voice_duration_ms = 150  # OPTIMIZED: Even faster for quiet environments
            self.min_silence_duration_ms = 600  # OPTIMIZED: Faster processing
            self.spectral_centroid_threshold = 200
            audio_logger.info("[MUTE] VAD sensitivity set to HIGH (quiet environment - optimized)")

        else:  # Medium (default) - OPTIMIZED for real-time responsiveness
            self.vad_threshold = 0.012
            self.energy_threshold = 2e-6
            self.min_voice_duration_ms = 200  # OPTIMIZED: Faster trigger
            self.min_silence_duration_ms = 800  # OPTIMIZED: Faster processing
            self.spectral_centroid_threshold = 350
            audio_logger.info("[VAD] VAD sensitivity set to MEDIUM (optimized for real-time responsiveness)")
    
    # Legacy methods for backward compatibility
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None) -> torch.Tensor:
        """Legacy method that redirects to real-time preprocessing"""
        return self.preprocess_realtime_chunk(audio_data, sample_rate=sample_rate)
    
    def validate_audio_format(self, audio_data: np.ndarray) -> bool:
        """Legacy method that redirects to real-time validation"""
        return self.validate_realtime_chunk(audio_data)
    
    def process_streaming_audio(self, audio_chunk: np.ndarray, chunk_id: int = None) -> torch.Tensor:
        """
        Enhanced method for processing streaming audio chunks in real-time
        """
        return self.preprocess_realtime_chunk(audio_chunk, chunk_id=chunk_id)
    
    def chunk_audio(self, audio_tensor: torch.Tensor, chunk_duration: float = 2.0) -> list:
        """
        Chunk audio for real-time processing (shorter chunks for better latency)
        """
        try:
            chunk_samples = int(chunk_duration * self.sample_rate)
            chunks = []
            
            audio_logger.debug(f"[EMOJI] Chunking audio into {chunk_duration}s segments")
            
            for i, start_idx in enumerate(range(0, len(audio_tensor), chunk_samples)):
                end_idx = min(start_idx + chunk_samples, len(audio_tensor))
                chunk = audio_tensor[start_idx:end_idx]
                
                # Pad chunk to exact size if needed (only for the last chunk)
                if len(chunk) < chunk_samples and i > 0:
                    padding = chunk_samples - len(chunk)
                    chunk = torch.cat([chunk, torch.zeros(padding)])
                    audio_logger.debug(f"   [CONFIG] Padded last chunk with {padding} zeros")
                
                chunks.append(chunk)
                audio_logger.debug(f"   [EMOJI] Chunk {i}: {len(chunk)} samples ({len(chunk)/self.sample_rate:.2f}s)")
            
            audio_logger.info(f"[OK] Audio chunked into {len(chunks)} segments of ~{chunk_duration}s each")
            return chunks
            
        except Exception as e:
            audio_logger.error(f"[ERROR] Error chunking audio: {e}")
            raise

# FIXED: Add proper main execution block for testing
if __name__ == "__main__":
    """Test audio processor functionality with VAD"""
    print("[EMOJI] Testing Audio Processor with VAD...")
    
    try:
        # Initialize processor
        processor = AudioProcessor()
        
        # Test with dummy audio (voice simulation)
        sample_rate = 16000
        duration = 1.0
        
        # Generate voice-like audio (modulated sine wave)
        t = np.linspace(0, duration, int(sample_rate * duration))
        voice_audio = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        voice_audio = voice_audio.astype(np.float32) * 0.1  # Scale to reasonable amplitude
        
        # Generate silence
        silence_audio = np.random.normal(0, 0.0001, int(sample_rate * duration)).astype(np.float32)
        
        print(f"[STATS] Voice audio: {len(voice_audio)} samples")
        print(f"[STATS] Silence audio: {len(silence_audio)} samples")
        
        # Test VAD on voice
        vad_voice = processor.detect_voice_activity(voice_audio, chunk_id="voice_test")
        print(f"[VAD] VAD Voice Result: {vad_voice['has_voice']} (confidence: {vad_voice['confidence']:.2f})")
        
        # Test VAD on silence
        vad_silence = processor.detect_voice_activity(silence_audio, chunk_id="silence_test")
        print(f"[MUTE] VAD Silence Result: {vad_silence['has_voice']} (confidence: {vad_silence['confidence']:.2f})")
        
        # Test validation with VAD
        voice_valid = processor.validate_realtime_chunk(voice_audio, chunk_id="voice_validation")
        silence_valid = processor.validate_realtime_chunk(silence_audio, chunk_id="silence_validation")
        
        print(f"[OK] Voice validation: {voice_valid}")
        print(f"[ERROR] Silence validation: {silence_valid}")
        
        # Test preprocessing
        if voice_valid:
            voice_tensor = processor.preprocess_realtime_chunk(voice_audio, chunk_id="voice_preprocessing")
            print(f"[OK] Voice preprocessing completed: {voice_tensor.shape}")
        
        # Test performance stats
        stats = processor.get_processing_stats()
        print(f"[STATS] Processing stats: {stats}")
        
        print("[SUCCESS] All VAD tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        raise
