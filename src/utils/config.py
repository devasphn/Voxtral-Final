"""
Configuration management for Voxtral Real-time Streaming (FIXED)
Updated for Pydantic v2 and pydantic-settings
"""
import yaml
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

# Import pydantic_settings with fallback
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    # Fallback for older pydantic versions or missing pydantic-settings
    try:
        from pydantic import BaseSettings
        SettingsConfigDict = None
        PYDANTIC_SETTINGS_AVAILABLE = True
    except ImportError:
        BaseSettings = BaseModel
        SettingsConfigDict = None
        PYDANTIC_SETTINGS_AVAILABLE = False

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    http_port: int = 8000
    health_port: int = 8005
    tcp_ports: List[int] = [8765, 8766]

class ModelConfig(BaseModel):
    name: str = "mistralai/Voxtral-Mini-3B-2507"
    cache_dir: str = "./model_cache"  # FIXED: Match config.yaml
    device: str = "cuda"
    torch_dtype: str = "float16"  # FIXED: Match config.yaml for optimal performance
    max_memory_per_gpu: str = "6GB"
    # ADDED: CUDA memory optimization settings
    require_auth_token: bool = True
    use_safetensors: bool = True
    enable_memory_optimization: bool = True

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    chunk_size: int = 512  # FIXED: Match config.yaml for lower latency
    format: str = "int16"
    channels: int = 1
    frame_duration_ms: int = 20  # FIXED: Match config.yaml for faster processing

class SpectrogramConfig(BaseModel):
    n_mels: int = 128
    hop_length: int = 160
    win_length: int = 400
    n_fft: int = 1024  # FIXED: Match config.yaml to resolve mel filterbank warning

class StreamingConfig(BaseModel):
    max_connections: int = 100
    buffer_size: int = 2048  # FIXED: Match config.yaml for lower latency
    timeout_seconds: int = 300
    latency_target_ms: int = 100  # FIXED: Match config.yaml aggressive target
    
class PerformanceConfig(BaseModel):
    """Performance monitoring and optimization configuration"""
    enable_monitoring: bool = True
    latency_targets: Dict[str, int] = {
        "voxtral_processing_ms": 100,
        "kokoro_generation_ms": 150,
        "audio_conversion_ms": 50,
        "total_end_to_end_ms": 300
    }
    alert_thresholds: Dict[str, float] = {
        "consecutive_failures": 5,
        "degradation_threshold": 1.5,
        "success_rate_threshold": 0.8
    }
    optimization_level: str = "balanced"  # "performance", "balanced", "memory_efficient"

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "/workspace/logs/voxtral_streaming.log"

class TTSVoicesConfig(BaseModel):
    english: List[str] = ["af_heart", "af_bella", "af_nicole", "af_sarah"]  # Kokoro English voices
    hindi: List[str] = ["hm_omega", "hf_alpha", "hf_beta", "hm_psi"]  # Kokoro Hindi voices

# Orpheus configuration classes removed - using Kokoro TTS only

class TTSPerformanceConfig(BaseModel):
    """TTS performance and optimization settings"""
    batch_size: int = 1  # Direct integration uses batch_size=1
    max_queue_size: int = 16  # FIXED: Match config.yaml for lower latency
    num_workers: int = 2  # FIXED: Match config.yaml to minimize overhead
    target_latency_ms: int = 100  # FIXED: Match config.yaml aggressive target
    memory_optimization: str = "balanced"  # "performance", "balanced", "memory_efficient"

class GPUMemoryConfig(BaseModel):
    """GPU memory management configuration"""
    min_vram_gb: float = 8.0
    recommended_vram_gb: float = 16.0
    memory_fraction: float = 0.9
    cleanup_frequency: str = "after_each_generation"  # "after_each_generation", "periodic"
    enable_monitoring: bool = True

class TTSConfig(BaseModel):
    engine: str = "kokoro"  # Changed to Kokoro TTS only
    default_voice: str = "hm_omega"  # Kokoro Hindi voice (replaces ऋतिका)
    sample_rate: int = 24000
    enabled: bool = True
    # Kokoro TTS settings
    voice: str = "hm_omega"  # Kokoro Hindi voice
    speed: float = 1.0  # Kokoro speech speed
    lang_code: str = "h"  # Kokoro language code (h=Hindi, a=American English, b=British English)
    voices: TTSVoicesConfig = TTSVoicesConfig()
    performance: TTSPerformanceConfig = TTSPerformanceConfig()
    gpu_memory: GPUMemoryConfig = GPUMemoryConfig()

class SpeechToSpeechConfig(BaseModel):
    enabled: bool = True
    latency_target_ms: int = 300  # Updated to match <300ms requirement
    buffer_size: int = 8192
    output_format: str = "wav"
    quality: str = "high"
    emotional_expression: bool = True

class Config(BaseSettings):
    """Main configuration class using BaseSettings for environment variable support"""
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    spectrogram: SpectrogramConfig = SpectrogramConfig()
    streaming: StreamingConfig = StreamingConfig()
    logging: LoggingConfig = LoggingConfig()
    tts: TTSConfig = TTSConfig()
    performance: PerformanceConfig = PerformanceConfig()
    speech_to_speech: SpeechToSpeechConfig = SpeechToSpeechConfig()
    
    # Pydantic v2 settings configuration with fallback
    if PYDANTIC_SETTINGS_AVAILABLE and SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_nested_delimiter="__",
            case_sensitive=False,
            extra="ignore"
        )
    else:
        # Fallback configuration for older pydantic versions
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file with environment variable override support"""
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            # Open with explicit UTF-8 encoding to handle Unicode characters
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return Config(**config_data)
        except UnicodeDecodeError:
            print(f"Warning: Unicode decode error in {config_path}. Using default configuration.")
            return Config()
        except Exception as e:
            print(f"Warning: Error loading config from {config_path}: {e}. Using default configuration.")
            return Config()
    else:
        # Return default config if file doesn't exist
        return Config()

# Global config instance
config = load_config()

# Environment variable overrides (still supported for backward compatibility)
if os.getenv("VOXTRAL_HTTP_PORT"):
    config.server.http_port = int(os.getenv("VOXTRAL_HTTP_PORT"))
    
if os.getenv("VOXTRAL_HEALTH_PORT"):
    config.server.health_port = int(os.getenv("VOXTRAL_HEALTH_PORT"))
    
if os.getenv("VOXTRAL_MODEL_NAME"):
    config.model.name = os.getenv("VOXTRAL_MODEL_NAME")
    
if os.getenv("CUDA_VISIBLE_DEVICES"):
    if os.getenv("CUDA_VISIBLE_DEVICES") == "-1":
        config.model.device = "cpu"
