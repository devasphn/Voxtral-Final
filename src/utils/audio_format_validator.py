"""
Audio Format Validator for Voxtral Streaming Voice Agent
Comprehensive validation and debugging tools for audio format issues
"""

import numpy as np
import logging
import struct
from typing import Dict, Any, Optional, Tuple, Union
from io import BytesIO

# Setup logging
audio_validator_logger = logging.getLogger("audio_format_validator")
audio_validator_logger.setLevel(logging.INFO)

class AudioFormatValidator:
    """
    Comprehensive audio format validation and debugging utility
    Helps diagnose and fix audio format issues in the streaming pipeline
    """
    
    def __init__(self):
        self.supported_formats = ['wav', 'pcm', 'float32', 'int16']
        self.supported_sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
        
    def validate_wav_headers(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Validate WAV file headers and extract metadata
        
        Args:
            audio_bytes: Raw audio bytes to validate
            
        Returns:
            Dict containing validation results and metadata
        """
        result = {
            'is_valid': False,
            'format': 'unknown',
            'sample_rate': 0,
            'channels': 0,
            'bit_depth': 0,
            'duration_ms': 0,
            'file_size': len(audio_bytes),
            'errors': [],
            'warnings': []
        }
        
        try:
            if len(audio_bytes) < 44:
                result['errors'].append(f"File too small: {len(audio_bytes)} bytes (minimum 44 for WAV)")
                return result
            
            # Check RIFF header
            riff_header = audio_bytes[:4]
            if riff_header != b'RIFF':
                result['errors'].append(f"Invalid RIFF header: {riff_header}")
                return result
            
            # Check WAVE format
            wave_header = audio_bytes[8:12]
            if wave_header != b'WAVE':
                result['errors'].append(f"Invalid WAVE header: {wave_header}")
                return result
            
            # Parse format chunk
            fmt_chunk = audio_bytes[12:16]
            if fmt_chunk != b'fmt ':
                result['errors'].append(f"Invalid fmt chunk: {fmt_chunk}")
                return result
            
            # Extract audio format information
            fmt_size = struct.unpack('<I', audio_bytes[16:20])[0]
            audio_format = struct.unpack('<H', audio_bytes[20:22])[0]
            channels = struct.unpack('<H', audio_bytes[22:24])[0]
            sample_rate = struct.unpack('<I', audio_bytes[24:28])[0]
            byte_rate = struct.unpack('<I', audio_bytes[28:32])[0]
            block_align = struct.unpack('<H', audio_bytes[32:34])[0]
            bits_per_sample = struct.unpack('<H', audio_bytes[34:36])[0]
            
            # Validate format
            if audio_format != 1:  # PCM format
                result['warnings'].append(f"Non-PCM format: {audio_format}")
            
            if sample_rate not in self.supported_sample_rates:
                result['warnings'].append(f"Unusual sample rate: {sample_rate}Hz")
            
            if channels not in [1, 2]:
                result['warnings'].append(f"Unusual channel count: {channels}")
            
            if bits_per_sample not in [8, 16, 24, 32]:
                result['warnings'].append(f"Unusual bit depth: {bits_per_sample}")
            
            # Find data chunk
            data_offset = 36
            while data_offset < len(audio_bytes) - 8:
                chunk_id = audio_bytes[data_offset:data_offset+4]
                chunk_size = struct.unpack('<I', audio_bytes[data_offset+4:data_offset+8])[0]
                
                if chunk_id == b'data':
                    # Calculate duration
                    data_size = chunk_size
                    duration_seconds = data_size / (sample_rate * channels * (bits_per_sample // 8))
                    
                    result.update({
                        'is_valid': True,
                        'format': 'wav',
                        'sample_rate': sample_rate,
                        'channels': channels,
                        'bit_depth': bits_per_sample,
                        'duration_ms': duration_seconds * 1000,
                        'data_size': data_size,
                        'data_offset': data_offset + 8
                    })
                    break
                
                data_offset += 8 + chunk_size
            
            if not result['is_valid']:
                result['errors'].append("No data chunk found")
            
        except Exception as e:
            result['errors'].append(f"Parsing error: {str(e)}")
        
        return result
    
    def validate_audio_chunk(self, audio_data: Union[bytes, np.ndarray], 
                           expected_format: str = 'wav',
                           expected_sample_rate: int = 24000) -> Dict[str, Any]:
        """
        Validate audio chunk format and content
        
        Args:
            audio_data: Audio data to validate
            expected_format: Expected audio format
            expected_sample_rate: Expected sample rate
            
        Returns:
            Dict containing validation results
        """
        result = {
            'is_valid': False,
            'format_matches': False,
            'sample_rate_matches': False,
            'has_audio_content': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            if isinstance(audio_data, bytes):
                if expected_format == 'wav':
                    wav_result = self.validate_wav_headers(audio_data)
                    result['is_valid'] = wav_result['is_valid']
                    result['format_matches'] = wav_result['is_valid']
                    result['sample_rate_matches'] = wav_result.get('sample_rate') == expected_sample_rate
                    result['has_audio_content'] = wav_result.get('duration_ms', 0) > 0
                    result['issues'].extend(wav_result.get('errors', []))
                    result['issues'].extend(wav_result.get('warnings', []))
                    
                    if not result['sample_rate_matches']:
                        result['recommendations'].append(f"Convert sample rate from {wav_result.get('sample_rate')}Hz to {expected_sample_rate}Hz")
                
            elif isinstance(audio_data, np.ndarray):
                # Validate numpy array
                if len(audio_data) == 0:
                    result['issues'].append("Empty audio array")
                else:
                    result['has_audio_content'] = True
                    
                    # Check for clipping
                    if audio_data.dtype == np.float32:
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 1.0:
                            result['issues'].append(f"Audio clipping detected: max value {max_val}")
                            result['recommendations'].append("Normalize audio to [-1, 1] range")
                        elif max_val < 0.1:
                            result['issues'].append(f"Audio very quiet: max value {max_val}")
                            result['recommendations'].append("Check audio gain/volume")
                    
                    # Check for silence
                    if np.all(audio_data == 0):
                        result['issues'].append("Audio contains only silence")
                    
                    result['is_valid'] = len(result['issues']) == 0
            
        except Exception as e:
            result['issues'].append(f"Validation error: {str(e)}")
        
        return result
    
    def diagnose_ultrasonic_noise(self, audio_data: Union[bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Diagnose potential causes of ultrasonic noise in audio
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dict containing diagnosis results
        """
        diagnosis = {
            'likely_causes': [],
            'severity': 'unknown',
            'recommendations': []
        }
        
        try:
            # Convert to numpy array for analysis
            if isinstance(audio_data, bytes):
                # Try to parse as WAV first
                wav_result = self.validate_wav_headers(audio_data)
                if wav_result['is_valid']:
                    # Extract audio samples from WAV
                    data_offset = wav_result.get('data_offset', 44)
                    bit_depth = wav_result.get('bit_depth', 16)
                    
                    if bit_depth == 16:
                        samples = np.frombuffer(audio_data[data_offset:], dtype=np.int16).astype(np.float32) / 32767.0
                    else:
                        samples = np.frombuffer(audio_data[data_offset:], dtype=np.float32)
                else:
                    # Try as raw PCM data
                    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                samples = audio_data.astype(np.float32)
            
            if len(samples) == 0:
                diagnosis['likely_causes'].append("Empty audio data")
                return diagnosis
            
            # Check for common ultrasonic noise causes
            max_val = np.max(np.abs(samples))
            
            # 1. Severe clipping
            if max_val > 1.0:
                diagnosis['likely_causes'].append(f"Audio clipping (max: {max_val:.3f})")
                diagnosis['severity'] = 'high'
                diagnosis['recommendations'].append("Normalize audio to prevent clipping")
            
            # 2. Bit depth mismatch
            if np.any(samples > 1.0) or np.any(samples < -1.0):
                diagnosis['likely_causes'].append("Possible bit depth mismatch or scaling issue")
                diagnosis['severity'] = 'high'
                diagnosis['recommendations'].append("Check audio data type conversion")
            
            # 3. Sample rate mismatch
            # Check for high-frequency content that might indicate sample rate issues
            if len(samples) > 1024:
                fft = np.fft.fft(samples[:1024])
                freqs = np.fft.fftfreq(1024, 1/24000)  # Assuming 24kHz
                high_freq_power = np.sum(np.abs(fft[freqs > 12000]))  # Above Nyquist for lower sample rates
                total_power = np.sum(np.abs(fft))
                
                if high_freq_power / total_power > 0.1:  # More than 10% high frequency content
                    diagnosis['likely_causes'].append("High frequency content suggests sample rate mismatch")
                    diagnosis['recommendations'].append("Verify sample rate consistency throughout pipeline")
            
            # 4. Format corruption
            if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
                diagnosis['likely_causes'].append("Audio data corruption (NaN/Inf values)")
                diagnosis['severity'] = 'critical'
                diagnosis['recommendations'].append("Check audio generation pipeline for corruption")
            
            # 5. Endianness issues
            if max_val < 0.001 and len(samples) > 100:
                diagnosis['likely_causes'].append("Possible endianness or format interpretation issue")
                diagnosis['recommendations'].append("Check byte order and data type interpretation")
            
            if not diagnosis['likely_causes']:
                diagnosis['severity'] = 'low'
                diagnosis['likely_causes'].append("No obvious issues detected")
            
        except Exception as e:
            diagnosis['likely_causes'].append(f"Analysis error: {str(e)}")
            diagnosis['severity'] = 'unknown'
        
        return diagnosis

# Global instance
audio_format_validator = AudioFormatValidator()
