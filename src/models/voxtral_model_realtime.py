"""
PRODUCTION-READY Voxtral model wrapper for CONVERSATIONAL real-time streaming
FIXED: FlashAttention2 detection, VAD implementation, silence handling
"""
import torch
import asyncio
import time
from typing import Optional, List, Dict, Any
# Import with compatibility layer
try:
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    VOXTRAL_AVAILABLE = True
except ImportError:
    from src.utils.compatibility import FallbackVoxtralModel, FallbackAutoProcessor
    VoxtralForConditionalGeneration = FallbackVoxtralModel
    AutoProcessor = FallbackAutoProcessor
    VOXTRAL_AVAILABLE = False

import logging
from threading import Lock
import base64

# NOTE: mistral_common imports removed - using standard Hugging Face VoxtralProcessor API
# The official VoxtralProcessor uses standard conversation format, not mistral_common classes
import tempfile
import soundfile as sf
import numpy as np
import os
from collections import deque
import sys

# Import streaming performance optimizer
try:
    from src.utils.streaming_performance_optimizer import streaming_optimizer
except ImportError:
    streaming_optimizer = None

# Add current directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import config with fallback
try:
    from src.utils.config import config
except ImportError:
    from src.utils.compatibility import get_config
    config = get_config()

# Enhanced logging for real-time streaming
realtime_logger = logging.getLogger("voxtral_realtime")
realtime_logger.setLevel(logging.DEBUG)

def safe_to_string(value: Any) -> str:
    """
    CRITICAL UTILITY: Ultra-safe conversion of any value to string
    Handles numpy scalars, torch tensors, and all edge cases that cause iteration errors
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    # Handle numpy types (CRITICAL FIX for numpy.float32)
    if hasattr(value, 'item') and hasattr(value, 'ndim'):
        try:
            if value.ndim == 0:  # Numpy scalar
                return str(value.item())
            elif hasattr(value, 'size') and value.size == 1:  # Single element array
                return str(value.item())
            else:
                return str(value)  # Multi-element array
        except (ValueError, RuntimeError, AttributeError):
            return str(value)

    # CRITICAL FIX: Handle numpy.float32 and other numpy scalars without ndim
    if str(type(value)).startswith("<class 'numpy.float") or str(type(value)).startswith("<class 'numpy.int"):
        try:
            return str(float(value) if 'float' in str(type(value)) else int(value))
        except Exception:
            return str(value)

    # Handle torch tensors
    if hasattr(value, 'item') and hasattr(value, 'numel'):
        try:
            if value.numel() == 1:  # Single element tensor
                return str(value.item())
            else:
                return str(value)  # Multi-element tensor
        except (ValueError, RuntimeError, AttributeError):
            return str(value)

    # Handle iterables (but not strings/bytes)
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
        try:
            return ' '.join(str(item) for item in value)
        except (TypeError, ValueError):
            return str(value)

    # Final fallback
    try:
        result = str(value)
        # Clean up problematic string representations
        if result in ['None', 'nan', 'inf', '-inf', 'NaN']:
            return ""
        return result
    except Exception:
        return ""

def safe_string_contains(text: Any, chars) -> bool:
    """
    CRITICAL UTILITY: Ultra-safe string containment check
    Prevents numpy.float32 iteration errors by ensuring proper string conversion
    """
    try:
        # CRITICAL FIX: Multiple layers of protection against numpy.float32 iteration
        # Layer 1: Handle None and empty values
        if text is None:
            return False

        # Layer 2: Force string conversion for ALL non-string types
        if not isinstance(text, str):
            # Special handling for numpy types
            if hasattr(text, 'item') and hasattr(text, 'dtype'):
                # This is a numpy scalar
                text_str = str(text.item())
            elif hasattr(text, '__float__'):
                # This handles numpy.float32, numpy.float64, etc.
                text_str = str(float(text))
            elif hasattr(text, '__int__'):
                # This handles numpy.int32, numpy.int64, etc.
                text_str = str(int(text))
            else:
                # Fallback to safe_to_string
                text_str = safe_to_string(text)
        else:
            text_str = text

        # Layer 3: Final validation
        if not text_str or not isinstance(text_str, str):
            return False

        # Layer 4: Safe character list processing
        if isinstance(chars, str):
            chars = [chars]
        elif not hasattr(chars, '__iter__'):
            chars = [str(chars)]

        # Layer 5: Ultra-safe containment check with explicit error handling
        for char in chars:
            try:
                char_str = safe_to_string(char)
                if char_str and isinstance(char_str, str) and isinstance(text_str, str):
                    if char_str in text_str:
                        return True
            except Exception as char_error:
                realtime_logger.debug(f"Character check error for '{char}': {char_error}")
                continue

        return False

    except Exception as e:
        # Enhanced error logging for debugging
        realtime_logger.warning(f"[CRITICAL] safe_string_contains error: {e}, text type: {type(text)}, text value: {text}, chars type: {type(chars)}")
        return False

class VoxtralModel:
    """PRODUCTION-READY Voxtral model for conversational real-time streaming with VAD"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.audio_processor = None
        self.model_lock = Lock()
        self.is_initialized = False
        
        # Real-time streaming optimization
        self.recent_chunks = deque(maxlen=5)  # Reduced for faster processing
        self.processing_history = deque(maxlen=50)  # Reduced memory usage
        
        # ULTRA-LOW LATENCY Performance optimization settings
        self.device = config.model.device
        # OPTIMIZATION: Use float16 for inference to reduce memory and increase speed
        # FIXED: Proper dtype logic - prioritize float16 for maximum performance
        if config.model.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif config.model.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif config.model.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            # Default to float16 for best performance
            self.torch_dtype = torch.float16

        # CALIBRATED VAD and silence detection settings - Perfectly aligned with AudioProcessor
        self.silence_threshold = 0.015  # Aligned with AudioProcessor threshold for perfect compatibility
        self.min_speech_duration = 0.4  # Aligned with AudioProcessor minimum duration (400ms)
        self.max_silence_duration = 1.2  # Aligned with AudioProcessor silence duration (1200ms)

        # ULTRA-LOW LATENCY Performance optimization flags
        self.use_torch_compile = True   # ENABLED: For maximum performance optimization
        self.flash_attention_available = False
        self.use_kv_cache_optimization = True  # ADDED: Advanced KV cache optimization
        self.use_memory_efficient_attention = True  # ADDED: Memory efficient attention
        self.use_gradient_checkpointing = False  # DISABLED: Not needed for inference
        self.use_scaled_dot_product_attention = True  # ENABLED: PyTorch 2.0+ optimization
        self.enable_gpu_memory_optimization = True  # ENABLED: GPU memory optimizations
        
        realtime_logger.info(f"VoxtralModel initialized for {self.device} with {self.torch_dtype}")
        
    def get_audio_processor(self):
        """Lazy initialization of Audio processor"""
        if self.audio_processor is None:
            from src.models.audio_processor_realtime import AudioProcessor
            self.audio_processor = AudioProcessor()
            realtime_logger.info("Audio processor lazy-loaded into Voxtral model")
        return self.audio_processor
    
    def _check_flash_attention_availability(self):
        """
        FIXED: Properly detect FlashAttention2 availability
        """
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            
            # Test if we can actually use it
            if self.device == "cuda" and torch.cuda.is_available():
                # Try to get GPU compute capability
                gpu_capability = torch.cuda.get_device_capability()
                major, minor = gpu_capability
                
                # FlashAttention2 requires compute capability >= 8.0 for optimal performance
                if major >= 8:
                    self.flash_attention_available = True
                    realtime_logger.info(f"[OK] FlashAttention2 available - GPU compute capability: {major}.{minor}")
                    return "flash_attention_2"
                else:
                    realtime_logger.info(f"[IDEA] FlashAttention2 available but GPU compute capability ({major}.{minor}) < 8.0, using eager attention")
                    return "eager"
            else:
                realtime_logger.info("[IDEA] FlashAttention2 available but CUDA not available, using eager attention")
                return "eager"
                
        except ImportError:
            realtime_logger.info("[IDEA] FlashAttention2 not installed, using eager attention")
            realtime_logger.info("[IDEA] To install: pip install flash-attn --no-build-isolation")
            return "eager"
        except Exception as e:
            realtime_logger.warning(f"[WARN] FlashAttention2 check failed: {e}, using eager attention")
            return "eager"
    
    def _calculate_audio_energy(self, audio_data: np.ndarray) -> float:
        """
        Calculate RMS energy of audio signal for VAD
        """
        try:
            # Calculate RMS (Root Mean Square) energy
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            return float(rms_energy)
        except Exception as e:
            realtime_logger.warning(f"Error calculating audio energy: {e}")
            return 0.0
    
    def _is_speech_detected(self, audio_data: np.ndarray, duration_s: float) -> bool:
        """
        PRODUCTION VAD: Detect if audio contains speech using energy and duration thresholds
        """
        try:
            # Calculate audio energy
            energy = self._calculate_audio_energy(audio_data)
            
            # Apply energy threshold
            if energy < self.silence_threshold:
                realtime_logger.debug(f"[MUTE] Audio energy ({energy:.6f}) below silence threshold ({self.silence_threshold})")
                return False
            
            # Apply minimum duration threshold
            if duration_s < self.min_speech_duration:
                realtime_logger.debug(f"[TIME] Audio duration ({duration_s:.2f}s) below minimum speech duration ({self.min_speech_duration}s)")
                return False
            
            # Additional checks for speech-like characteristics
            # Check for spectral variation (speech has more variation than steady noise)
            spectral_variation = np.std(audio_data)
            if spectral_variation < self.silence_threshold * 0.5:
                realtime_logger.debug(f"[STATS] Low spectral variation ({spectral_variation:.6f}), likely not speech")
                return False
            
            realtime_logger.debug(f"[VAD] Speech detected - Energy: {energy:.6f}, Duration: {duration_s:.2f}s, Variation: {spectral_variation:.6f}")
            return True
            
        except Exception as e:
            realtime_logger.error(f"Error in speech detection: {e}")
            return False
    
    def _check_model_cache(self) -> bool:
        """ADDED: Check if model is already cached to avoid re-downloading"""
        try:
            import os
            from pathlib import Path

            cache_dir = config.model.cache_dir
            model_name = config.model.name

            if not cache_dir or not os.path.exists(cache_dir):
                realtime_logger.info(f"[CACHE] Cache directory {cache_dir} not found")
                return False

            # Check for HuggingFace cache structure
            model_cache_name = model_name.replace('/', '--')
            model_cache_path = os.path.join(cache_dir, f"models--{model_cache_name}")

            if os.path.exists(model_cache_path):
                # Check for snapshots directory
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir)
                               if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        latest_snapshot = os.path.join(snapshots_dir, snapshots[0])
                        # Check for essential model files
                        essential_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
                        files_exist = any(os.path.exists(os.path.join(latest_snapshot, f))
                                        for f in essential_files)
                        if files_exist:
                            realtime_logger.info(f"[CACHE] Found cached model at: {latest_snapshot}")
                            return True

            realtime_logger.info(f"[CACHE] No valid cache found for {model_name}")
            return False

        except Exception as e:
            realtime_logger.warning(f"[CACHE] Cache check failed: {e}")
            return False

    async def initialize(self):
        """Initialize the Voxtral model with FIXED attention implementation handling"""
        try:
            realtime_logger.info("[INIT] Starting Voxtral model initialization for conversational streaming...")
            start_time = time.time()

            # ADDED: Check cache status before initialization
            cache_available = self._check_model_cache()
            if cache_available:
                realtime_logger.info("[CACHE] Using cached model - faster initialization expected")
            else:
                realtime_logger.info("[CACHE] No cache found - will download model")

            # OPTIMIZED: RTX A4500 GPU memory optimization setup
            if self.enable_gpu_memory_optimization and self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Check if we have RTX A4500 (20GB VRAM)
                    gpu_name = torch.cuda.get_device_name(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                    torch.cuda.empty_cache()

                    if "RTX A4500" in gpu_name or total_memory >= 19.0:  # RTX A4500 has ~20GB
                        # RTX A4500 optimizations
                        torch.cuda.set_per_process_memory_fraction(0.90)  # Use 90% of 20GB = 18GB
                        realtime_logger.info(f"[INIT] RTX A4500 detected ({total_memory:.1f}GB) - applying optimal settings")

                        # Set optimal CUDA memory allocator settings for RTX A4500
                        import os
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:16'

                    else:
                        # General GPU optimizations
                        torch.cuda.set_per_process_memory_fraction(0.85)  # More conservative for other GPUs
                        realtime_logger.info(f"[INIT] GPU detected ({gpu_name}, {total_memory:.1f}GB) - applying standard settings")

                    # Enable memory pool for faster allocation
                    if hasattr(torch.cuda, 'memory_pool'):
                        torch.cuda.memory_pool.set_memory_fraction(0.90)

                    realtime_logger.info("[INIT] GPU memory optimization enabled")
                except Exception as e:
                    realtime_logger.warning(f"[WARN] GPU memory optimization failed: {e}")
                    realtime_logger.info("[IDEA] Continuing without GPU memory optimization...")
            elif self.device == "cuda" and not torch.cuda.is_available():
                realtime_logger.warning("[WARN] CUDA device specified but CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load processor with authentication
            realtime_logger.info(f"[INPUT] Loading AutoProcessor from {config.model.name}")

            processor_kwargs = {
                "cache_dir": config.model.cache_dir,
                "trust_remote_code": True
            }

            # FIXED: Add authentication token if available
            import os
            if os.getenv('HF_TOKEN'):
                processor_kwargs["token"] = os.getenv('HF_TOKEN')
                realtime_logger.info("[EMOJI] Using HuggingFace authentication token for processor")

            self.processor = AutoProcessor.from_pretrained(
                config.model.name,
                **processor_kwargs
            )
            realtime_logger.info("[OK] AutoProcessor loaded successfully")
            
            # FIXED: Determine attention implementation with proper detection
            attn_implementation = self._check_flash_attention_availability()
            realtime_logger.info(f"[CONFIG] Using attention implementation: {attn_implementation}")
            
            # Load model with FIXED attention settings
            realtime_logger.info(f"[INPUT] Loading Voxtral model from {config.model.name}")
            
            # FIXED: Updated model loading parameters for compatibility
            model_kwargs = {
                "cache_dir": config.model.cache_dir,
                "dtype": self.torch_dtype,  # FIXED: Use 'dtype' instead of deprecated 'torch_dtype'
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "attn_implementation": attn_implementation,
                # ULTRA-LOW LATENCY: Additional optimization parameters
                "use_safetensors": True,  # Faster loading
                # REMOVED: variant parameter that may cause loading issues
            }

            # FIXED: Add authentication token if available
            import os
            if os.getenv('HF_TOKEN'):
                model_kwargs["token"] = os.getenv('HF_TOKEN')
                realtime_logger.info("[EMOJI] Using HuggingFace authentication token")
            
            try:
                realtime_logger.info(f"[EMOJI] Loading Voxtral model with dtype={self.torch_dtype}, attention={attn_implementation}")
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    config.model.name,
                    **model_kwargs
                )
                realtime_logger.info(f"[OK] Voxtral model loaded successfully with {attn_implementation} attention")

            except Exception as model_load_error:
                realtime_logger.error(f"[ERROR] Initial model loading failed: {model_load_error}")

                # ENHANCED: Multiple fallback strategies for robust loading
                if attn_implementation != "eager":
                    realtime_logger.warning(f"[WARN] Model loading with {attn_implementation} failed: {model_load_error}")
                    realtime_logger.info("[EMOJI] Retrying with eager attention as fallback...")

                    model_kwargs["attn_implementation"] = "eager"
                    try:
                        self.model = VoxtralForConditionalGeneration.from_pretrained(
                            config.model.name,
                            **model_kwargs
                        )
                        realtime_logger.info("[OK] Voxtral model loaded successfully with eager attention fallback")
                    except Exception as eager_error:
                        realtime_logger.error(f"[ERROR] Eager attention fallback also failed: {eager_error}")
                        # Try without safetensors as final fallback
                        realtime_logger.info("[EMOJI] Trying final fallback without safetensors...")
                        model_kwargs["use_safetensors"] = False
                        self.model = VoxtralForConditionalGeneration.from_pretrained(
                            config.model.name,
                            **model_kwargs
                        )
                        realtime_logger.info("[OK] Model loaded successfully without safetensors")

                elif "safetensors" in str(model_load_error).lower() or "num" in str(model_load_error).lower():
                    realtime_logger.warning(f"[WARN] Safetensors loading failed: {model_load_error}")
                    realtime_logger.info("[EMOJI] Retrying without safetensors...")

                    model_kwargs["use_safetensors"] = False
                    self.model = VoxtralForConditionalGeneration.from_pretrained(
                        config.model.name,
                        **model_kwargs
                    )
                    realtime_logger.info("[OK] Model loaded successfully without safetensors")

                else:
                    # Generic fallback - try with minimal parameters
                    realtime_logger.warning(f"[WARN] Generic model loading error: {model_load_error}")
                    realtime_logger.info("[EMOJI] Trying minimal parameter fallback...")

                    minimal_kwargs = {
                        "cache_dir": config.model.cache_dir,
                        "dtype": self.torch_dtype,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "use_safetensors": False,
                        "attn_implementation": "eager"
                    }

                    # Add token if available
                    if os.getenv('HF_TOKEN'):
                        minimal_kwargs["token"] = os.getenv('HF_TOKEN')

                    self.model = VoxtralForConditionalGeneration.from_pretrained(
                        config.model.name,
                        **minimal_kwargs
                    )
                    realtime_logger.info("[OK] Model loaded successfully with minimal parameters")
            
            # Set model to evaluation mode
            self.model.eval()
            realtime_logger.info("[CONFIG] Model set to evaluation mode")
            
            # TEMPORARY FIX: Disable torch.compile to resolve 'function' object issue
            # TODO: Re-enable with proper model wrapper that preserves attributes
            if False and self.use_torch_compile and hasattr(torch, 'compile'):
                try:
                    realtime_logger.info("[FAST] Attempting ULTRA-LOW LATENCY model compilation...")

                    # Set optimal compilation environment
                    import os
                    os.environ['TORCH_COMPILE_DEBUG'] = '0'  # Disable debug for speed
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = './torch_compile_cache'
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:16'

                    # FIXED: Method 1 - Use mode-based compilation (no options conflict)
                    try:
                        self.model = torch.compile(
                            self.model,
                            mode="reduce-overhead",  # Maximum performance mode
                            fullgraph=True,         # Compile entire graph
                            dynamic=False           # Static shapes for speed
                        )
                        realtime_logger.info("[OK] Model compiled with reduce-overhead mode + CUDA optimizations")
                    except Exception as mode_error:
                        realtime_logger.warning(f"[WARN] Mode compilation failed: {mode_error}")

                        # FIXED: Method 2 - Use options-based compilation (no mode conflict)
                        try:
                            if self.device == "cuda" and torch.cuda.is_available():
                                self.model = torch.compile(
                                    self.model,
                                    options={
                                        "triton.cudagraphs": True,
                                        "max_autotune": True,
                                        "epilogue_fusion": True,
                                        "max_autotune_gemm": True,
                                    }
                                )
                                realtime_logger.info("[OK] Model compiled with CUDA-optimized options")
                            else:
                                # CPU fallback
                                self.model = torch.compile(self.model)
                                realtime_logger.info("[OK] Model compiled with basic optimizations")
                        except Exception as options_error:
                            realtime_logger.warning(f"[WARN] Options compilation failed: {options_error}")
                            # Final fallback to basic compilation
                            try:
                                self.model = torch.compile(self.model)
                                realtime_logger.info("[OK] Model compiled with basic optimizations")
                            except Exception as basic_error:
                                realtime_logger.warning(f"[WARN] Basic compilation failed: {basic_error}")
                                realtime_logger.info("[IDEA] Continuing without torch.compile...")

                except Exception as e:
                    realtime_logger.warning(f"[WARN] All torch.compile attempts failed: {e}")
                    try:
                        # Fallback to max-autotune mode
                        realtime_logger.info("[EMOJI] Falling back to max-autotune compilation mode...")
                        self.model = torch.compile(self.model, mode="max-autotune")
                        realtime_logger.info("[OK] Model compiled with max-autotune optimizations")
                    except Exception as e2:
                        realtime_logger.warning(f"[WARN] Max-autotune compilation also failed: {e2}")
                        try:
                            # Final fallback to default mode
                            realtime_logger.info("[EMOJI] Final fallback to default compilation mode...")
                            self.model = torch.compile(self.model, mode="default")
                            realtime_logger.info("[OK] Model compiled with default optimizations")
                        except Exception as e3:
                            realtime_logger.warning(f"[WARN] All compilation modes failed: {e3}")
                            realtime_logger.info("[IDEA] Continuing without torch.compile...")
            else:
                realtime_logger.info("[IDEA] torch.compile disabled or not available")

            # ULTRA-LOW LATENCY: CUDA Graphs and Memory Optimization
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Enable memory optimization
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.95)

                realtime_logger.info("[INIT] CUDA optimizations enabled for maximum performance")

            # ULTRA-LOW LATENCY: Additional model optimizations
            if hasattr(self.model, 'generation_config'):
                # Optimize generation config for speed
                self.model.generation_config.use_cache = True
                self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
                # ULTRA-OPTIMIZED generation parameters for sub-100ms target
                self.model.generation_config.do_sample = False  # Greedy decoding for speed
                self.model.generation_config.temperature = 0.01  # Minimal temperature for speed
                self.model.generation_config.top_p = 0.7  # More focused sampling
                self.model.generation_config.top_k = 15  # Reduced vocabulary for speed
                self.model.generation_config.repetition_penalty = 1.3  # Higher penalty for concise responses
                self.model.generation_config.num_beams = 1  # No beam search for speed
                self.model.generation_config.early_stopping = True  # Stop early when possible
                realtime_logger.info("[OK] Generation config optimized for ultra-low latency")

            # ULTRA-LOW LATENCY: Enable PyTorch 2.0+ optimizations
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.use_scaled_dot_product_attention:
                # This enables Flash Attention or other optimized attention implementations
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                realtime_logger.info("[OK] PyTorch 2.0+ scaled dot product attention optimizations enabled")

            # ULTRA-LOW LATENCY: Set optimal inference settings
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for speed
                    torch.backends.cudnn.allow_tf32 = True
                    realtime_logger.info("[OK] CUDA optimizations enabled for maximum performance")
                except Exception as e:
                    realtime_logger.warning(f"[WARN] CUDA optimizations failed: {e}")
                    realtime_logger.info("[IDEA] Continuing without CUDA optimizations...")
            else:
                realtime_logger.info("[IDEA] CUDA not available, using CPU optimizations")
            
            self.is_initialized = True
            init_time = time.time() - start_time
            realtime_logger.info(f"[SUCCESS] Voxtral model fully initialized in {init_time:.2f}s and ready for conversation!")
            
        except Exception as e:
            realtime_logger.error(f"[ERROR] Failed to initialize Voxtral model: {e}")
            import traceback
            realtime_logger.error(f"[ERROR] Full error traceback: {traceback.format_exc()}")
            raise
    
    async def process_realtime_chunk(self, audio_data: torch.Tensor, chunk_id: int, mode: str = "conversation", prompt: str = "") -> Dict[str, Any]:
        """
        PRODUCTION-READY processing for conversational real-time audio chunks with VAD
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        try:
            chunk_start_time = time.time()
            realtime_logger.debug(f"[AUDIO] Processing conversational chunk {chunk_id} with {len(audio_data)} samples")
            
            # Convert tensor to numpy for VAD analysis
            audio_np = audio_data.detach().cpu().numpy().copy()
            sample_rate = config.audio.sample_rate
            duration_s = len(audio_np) / sample_rate
            
            # CRITICAL: Apply VAD before processing
            if not self._is_speech_detected(audio_np, duration_s):
                realtime_logger.debug(f"[MUTE] Chunk {chunk_id} contains no speech - skipping processing")
                return {
                    'response': '',  # Empty response for silence
                    'processing_time_ms': (time.time() - chunk_start_time) * 1000,
                    'chunk_id': chunk_id,
                    'audio_duration_s': duration_s,
                    'success': True,
                    'is_silence': True
                }
            
            with self.model_lock:
                # Store chunk in recent history
                self.recent_chunks.append({
                    'chunk_id': chunk_id,
                    'timestamp': chunk_start_time,
                    'has_speech': True
                })
                
                # Ensure audio_data is properly formatted
                if not audio_data.data.is_contiguous():
                    audio_data = audio_data.contiguous()
                
                realtime_logger.debug(f"[SPEAKER] Audio stats for chunk {chunk_id}: length={len(audio_np)}, max_val={np.max(np.abs(audio_np)):.4f}")
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    try:
                        # Write audio to temporary file
                        sf.write(tmp_file.name, audio_np, sample_rate)
                        realtime_logger.debug(f"[FLOPPY] Written chunk {chunk_id} to temporary file: {tmp_file.name}")
                        
                        # FIXED: Use standard Hugging Face VoxtralProcessor API
                        # Choose processing mode based on requirements
                        # CRITICAL FIX: Only use audio-only mode for explicit speech-to-speech mode
                        if mode == "speech_to_speech":
                            # Audio-only mode for pure speech-to-speech processing
                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "audio", "path": tmp_file.name}
                                    ]
                                }
                            ]
                            realtime_logger.debug(f"[VAD] Using audio-only mode for speech-to-speech processing")
                        else:
                            # Audio + text mode for conversational AI
                            # ENHANCED: Context-aware system prompt with greeting detection
                            if prompt:
                                conversation_prompt = prompt
                            else:
                                # Smart prompt selection based on audio characteristics
                                # For short audio (likely greetings), use greeting-focused prompt
                                if duration_s <= 3.0:  # Short audio likely to be greetings
                                    conversation_prompt = "You are a helpful AI assistant. The person just spoke to you briefly - they might be greeting you, testing if you can hear them, or asking a quick question. If it sounds like a greeting (hello, hi, can you hear me, etc.), respond with a warm, friendly greeting back. If it's a question, answer it directly and warmly. Keep responses brief and natural."
                                else:
                                    # Longer audio, use standard conversational prompt
                                    conversation_prompt = "You are a helpful AI assistant in a natural voice conversation. Listen carefully to what the person is saying and respond naturally, as if you're having a friendly chat. Keep your responses conversational, concise (1-2 sentences), and engaging. Always acknowledge what the person said appropriately."

                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "audio", "path": tmp_file.name},
                                        {"type": "text", "text": conversation_prompt}
                                    ]
                                }
                            ]
                            realtime_logger.debug(f"[VAD] Using audio+text mode for conversational AI")

                        # Process inputs with correct VoxtralProcessor API
                        inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
                        
                        # Move to device
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(self.device)
                        elif isinstance(inputs, dict):
                            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                                    for k, v in inputs.items()}
                        
                        realtime_logger.debug(f"[INIT] Starting inference for chunk {chunk_id}")
                        inference_start = time.time()

                        # ENHANCED: Generate response with timeout and real-time optimizations
                        try:
                            with torch.no_grad():
                                # Use mixed precision for speed
                                with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu", dtype=self.torch_dtype, enabled=True):
                                    # STREAMING: Check if streaming mode is requested
                                    streaming_mode = mode == "streaming" or prompt.get("streaming", False) if isinstance(prompt, dict) else False

                                    if streaming_mode:
                                        # STREAMING GENERATION: 250 tokens with word-level streaming
                                        outputs = self.model.generate(
                                            **inputs,
                                            max_new_tokens=250,     # STREAMING: Full response length
                                            min_new_tokens=1,
                                            do_sample=True,         # STREAMING: Enable sampling for natural responses
                                            num_beams=1,
                                            temperature=0.3,        # STREAMING: Balanced temperature for quality
                                            top_p=0.9,             # STREAMING: Broader sampling for natural speech
                                            top_k=50,              # STREAMING: Expanded vocabulary
                                            repetition_penalty=1.2,
                                            pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                                            eos_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                                            use_cache=True,
                                            early_stopping=True,
                                            output_scores=False,
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            return_dict_in_generate=False,
                                            synced_gpus=False,
                                        )
                                    else:
                                        # ULTRA-LOW LATENCY: Maximum speed generation parameters for sub-50ms target
                                        outputs = self.model.generate(
                                            **inputs,
                                            max_new_tokens=15,      # OPTIMIZED: Slightly increased for complete greetings
                                            min_new_tokens=3,       # OPTIMIZED: Minimum 3 tokens for meaningful response
                                            do_sample=False,        # ULTRA-OPTIMIZED: Greedy decoding for maximum speed
                                            num_beams=1,           # Keep single beam for speed
                                            temperature=0.01,      # ULTRA-OPTIMIZED: Minimal temperature for fastest generation
                                            top_p=0.7,            # ULTRA-OPTIMIZED: More focused sampling
                                            top_k=10,             # ULTRA-OPTIMIZED: Minimal vocabulary for speed
                                            repetition_penalty=1.4, # ULTRA-OPTIMIZED: Higher penalty for more concise responses
                                            pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                                            eos_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                                            use_cache=True,         # Use KV cache for speed
                                            # ULTRA-LOW LATENCY: Additional optimizations
                                            early_stopping=True,   # Stop as soon as EOS is generated
                                            output_scores=False,   # Don't compute scores for speed
                                            output_attentions=False, # Don't compute attentions for speed
                                            output_hidden_states=False, # Don't compute hidden states for speed
                                            return_dict_in_generate=False, # Return simple tensor for speed
                                            synced_gpus=False,     # Disable GPU synchronization for speed
                                        )

                            inference_time = (time.time() - inference_start) * 1000

                            # Check if we exceeded target time and log warning
                            if inference_time > 100:  # Target is 100ms
                                realtime_logger.warning(f"[WARN] Inference for chunk {chunk_id} took {inference_time:.1f}ms (target: 100ms)")
                            else:
                                realtime_logger.debug(f"[FAST] Inference completed for chunk {chunk_id} in {inference_time:.1f}ms")

                        except Exception as e:
                            inference_time = (time.time() - inference_start) * 1000
                            realtime_logger.error(f"[ERROR] Inference failed for chunk {chunk_id} after {inference_time:.1f}ms: {e}")
                            raise
                        
                        # Decode response
                        if hasattr(inputs, 'input_ids'):
                            input_length = inputs.input_ids.shape[1]
                        elif 'input_ids' in inputs:
                            input_length = inputs['input_ids'].shape[1]
                        else:
                            input_length = 0
                            
                        response = self.processor.batch_decode(
                            outputs[:, input_length:], 
                            skip_special_tokens=True
                        )[0]
                        
                        total_processing_time = (time.time() - chunk_start_time) * 1000
                        
                        # Store performance metrics
                        performance_data = {
                            'chunk_id': chunk_id,
                            'total_time_ms': total_processing_time,
                            'inference_time_ms': inference_time,
                            'audio_length_s': len(audio_np) / sample_rate,
                            'response_length': len(response),
                            'timestamp': chunk_start_time,
                            'has_speech': True
                        }
                        self.processing_history.append(performance_data)
                        
                        # Clean and optimize response
                        cleaned_response = response.strip()
                        
                        # Filter out common noise responses
                        noise_responses = [
                            "I'm not sure what you're asking",
                            "I can't understand",
                            "Could you repeat that",
                            "I didn't catch that",
                            "Yeah, I think it's a good idea"  # This seems to be a common noise response
                        ]
                        
                        # If response is too short or matches noise patterns, treat as silence
                        if len(cleaned_response) < 3 or any(noise in cleaned_response for noise in noise_responses):
                            realtime_logger.debug(f"[MUTE] Filtering out noise response: '{cleaned_response}'")
                            return {
                                'response': '',
                                'processing_time_ms': total_processing_time,
                                'chunk_id': chunk_id,
                                'audio_duration_s': duration_s,
                                'success': True,
                                'is_silence': True,
                                'filtered_response': cleaned_response
                            }
                        
                        if not cleaned_response:
                            cleaned_response = "[Audio processed]"
                        
                        realtime_logger.info(f"[OK] Chunk {chunk_id} processed in {total_processing_time:.1f}ms: '{cleaned_response[:50]}{'...' if len(cleaned_response) > 50 else ''}'")
                        
                        return {
                            'response': cleaned_response,
                            'processing_time_ms': total_processing_time,
                            'inference_time_ms': inference_time,
                            'chunk_id': chunk_id,
                            'audio_duration_s': len(audio_np) / sample_rate,
                            'success': True,
                            'is_silence': False
                        }
                        
                    finally:
                        # Cleanup temporary file
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
                
        except Exception as e:
            processing_time = (time.time() - chunk_start_time) * 1000
            realtime_logger.error(f"[ERROR] Error processing chunk {chunk_id}: {e}")
            
            # Return error response with timing info
            error_msg = "Could not process audio"
            if "CUDA out of memory" in str(e):
                error_msg = "GPU memory error"
            elif "timeout" in str(e).lower():
                error_msg = "Processing timeout"
            
            return {
                'response': error_msg,
                'processing_time_ms': processing_time,
                'chunk_id': chunk_id,
                'success': False,
                'error': str(e),
                'is_silence': False
            }

    async def transcribe_audio(self, audio_data: torch.Tensor) -> str:
        """Unified conversational processing (legacy method)"""
        result = await self.process_realtime_chunk(
            audio_data,
            chunk_id=int(time.time() * 1000),
            mode="conversation"
        )
        return result['response']

    async def understand_audio(self, audio_data: torch.Tensor, question: str = "") -> str:
        """Unified conversational processing (legacy method)"""
        result = await self.process_realtime_chunk(
            audio_data,
            chunk_id=int(time.time() * 1000),
            mode="conversation"
        )
        return result['response']

    async def process_audio_stream(self, audio_data: torch.Tensor, prompt: str = "") -> str:
        """Unified conversational processing (legacy method)"""
        result = await self.process_realtime_chunk(
            audio_data,
            chunk_id=int(time.time() * 1000),
            mode="conversation"
        )
        return result['response']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information with real-time stats"""
        base_info = {
            "status": "initialized" if self.is_initialized else "not_initialized",
            "model_name": config.model.name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "mode": "conversational_optimized",
            "flash_attention_available": self.flash_attention_available,
            "torch_compile_enabled": self.use_torch_compile,
            "vad_settings": {
                "silence_threshold": self.silence_threshold,
                "min_speech_duration": self.min_speech_duration,
                "max_silence_duration": self.max_silence_duration
            }
        }
        
        if self.is_initialized and self.processing_history:
            # Calculate real-time performance stats
            recent_history = list(self.processing_history)[-10:]  # Last 10 chunks
            if recent_history:
                avg_processing_time = np.mean([h['total_time_ms'] for h in recent_history])
                avg_inference_time = np.mean([h['inference_time_ms'] for h in recent_history])
                total_chunks = len(self.processing_history)
                speech_chunks = len([h for h in recent_history if h.get('has_speech', False)])
                
                base_info.update({
                    "realtime_stats": {
                        "total_chunks_processed": total_chunks,
                        "speech_chunks_in_recent_10": speech_chunks,
                        "avg_processing_time_ms": round(avg_processing_time, 1),
                        "avg_inference_time_ms": round(avg_inference_time, 1),
                        "recent_chunks_in_memory": len(self.recent_chunks),
                        "performance_history_size": len(self.processing_history)
                    }
                })
        
        return base_info

    async def process_streaming_chunk(self, audio_data: np.ndarray, prompt: str = None,
                                    chunk_id: str = None, mode: str = "streaming"):
        """
        STREAMING VOICE AGENT: Process audio with token-by-token streaming generation
        Yields tokens as they are generated for immediate TTS processing
        """
        if not self.is_initialized:
            raise RuntimeError("Voxtral model not initialized")

        chunk_id = chunk_id or f"stream_{int(time.time() * 1000)}"
        start_time = time.time()

        # CRITICAL FIX: Ensure prompt is always a string to prevent numpy.float32 iteration error
        if prompt is None:
            prompt = ""
        elif not isinstance(prompt, str):
            # Handle any non-string types that might be passed
            if hasattr(prompt, 'item') and hasattr(prompt, 'dtype'):
                # Handle numpy scalars
                prompt = str(prompt.item())
            elif hasattr(prompt, '__float__'):
                # Handle numpy.float32, etc.
                prompt = str(float(prompt))
            else:
                # Fallback to string conversion
                prompt = str(prompt) if prompt is not None else ""

        # Final safety check
        if not isinstance(prompt, str):
            prompt = ""

        try:
            realtime_logger.info(f"[VAD] Starting streaming processing for chunk {chunk_id}")

            # Audio preprocessing (same as regular processing)
            preprocessing_start = time.time()

            # CRITICAL FIX: Ensure audio_data is properly handled as numpy array
            try:
                # Convert to numpy array if not already
                if not isinstance(audio_data, np.ndarray):
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.cpu().numpy()
                    else:
                        audio_data = np.array(audio_data, dtype=np.float32)

                # Ensure audio is in the correct format
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # CRITICAL FIX: Ensure it's at least 1D and not a scalar
                if audio_data.ndim == 0:
                    audio_data = np.array([audio_data], dtype=np.float32)

                # Ensure we have valid audio data
                if audio_data.size == 0:
                    raise ValueError("Empty audio data")

                # Normalize audio with safe handling
                max_val = np.max(np.abs(audio_data))
                if max_val > 0 and np.isfinite(max_val):
                    audio_data = audio_data / max_val * 0.95

                # Final validation
                if not isinstance(audio_data, np.ndarray):
                    raise ValueError("Audio data is not a numpy array after preprocessing")

            except Exception as audio_error:
                realtime_logger.error(f"[ERROR] Audio preprocessing error: {audio_error}")
                raise RuntimeError(f"Audio preprocessing failed: {audio_error}")

            preprocessing_time = (time.time() - preprocessing_start) * 1000
            realtime_logger.debug(f"[FAST] Audio preprocessing completed in {preprocessing_time:.1f}ms")

            # CRITICAL FIX: Use the same working approach as conversation mode
            # Instead of direct processor call, use apply_chat_template approach
            processor_start = time.time()

            try:
                # Create temporary audio file (same as conversation mode)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Convert audio to proper format for file writing
                    if isinstance(audio_data, torch.Tensor):
                        audio_np = audio_data.cpu().numpy()
                    else:
                        audio_np = audio_data

                    # Ensure audio is in correct format
                    if audio_np.dtype != np.float32:
                        audio_np = audio_np.astype(np.float32)

                    # Write to temporary file
                    sample_rate = 16000
                    sf.write(tmp_file.name, audio_np, sample_rate)
                    realtime_logger.debug(f"[FLOPPY] Written streaming chunk {chunk_id} to temporary file: {tmp_file.name}")

                    # CRITICAL FIX: Calculate duration_s for prompt selection
                    duration_s = len(audio_np) / sample_rate
                    realtime_logger.debug(f"[AUDIO] Calculated audio duration: {duration_s:.2f}s")

                    # ENHANCED: Use intelligent prompting for streaming mode (same as conversation mode)
                    # Smart prompt selection based on audio characteristics
                    if duration_s <= 3.0:  # Short audio likely to be greetings
                        streaming_prompt = "You are a helpful AI assistant. The person just spoke to you briefly - they might be greeting you, testing if you can hear them, or asking a quick question. If it sounds like a greeting (hello, hi, can you hear me, etc.), respond with a warm, friendly greeting back. If it's a question, answer it directly and warmly. Keep responses brief and natural."

                        # CRITICAL FIX: Ensure all conversation content is properly typed
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio", "path": str(tmp_file.name)},
                                    {"type": "text", "text": str(streaming_prompt)}
                                ]
                            }
                        ]
                        realtime_logger.debug(f"[VAD] Using audio+text mode for streaming (greeting-optimized)")
                    else:
                        # Longer audio, use audio-only mode for better performance
                        conversation = [
                            {
                                "role": "user",
                                "content": [{"type": "audio", "path": str(tmp_file.name)}]
                            }
                        ]
                        realtime_logger.debug(f"[VAD] Using audio-only mode for streaming processing")

                    # CRITICAL FIX: Additional safety check before processor call
                    # Ensure all string values in conversation are actually strings
                    try:
                        for message in conversation:
                            if "content" in message:
                                for content_item in message["content"]:
                                    if "text" in content_item:
                                        content_item["text"] = str(content_item["text"])
                                    if "path" in content_item:
                                        content_item["path"] = str(content_item["path"])
                    except Exception as conv_error:
                        realtime_logger.warning(f"[WARN] Conversation sanitization error: {conv_error}")

                    # Process inputs with correct VoxtralProcessor API (working approach)
                    inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")

                    # Move to device
                    if hasattr(inputs, 'to'):
                        inputs = inputs.to(self.device)
                    elif isinstance(inputs, dict):
                        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            except Exception as processor_error:
                realtime_logger.error(f"[ERROR] Processor error: {processor_error}")
                raise RuntimeError(f"Processor failed: {processor_error}")

            processor_time = (time.time() - processor_start) * 1000
            realtime_logger.debug(f"[FAST] Processor completed in {processor_time:.1f}ms")

            # STREAMING INFERENCE: Generate tokens one by one
            inference_start = time.time()
            realtime_logger.debug(f"[DEBUG] Starting inference section")

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # STREAMING GENERATION: Iterative token generation for real-time streaming
                    realtime_logger.debug(f"[DEBUG] Starting streaming generation")

                    # Initialize generation state
                    realtime_logger.debug(f"[DEBUG] Initializing generation state")

                    # CRITICAL FIX: Properly extract input_ids from BatchFeature or dict
                    if hasattr(inputs, 'input_ids'):
                        # BatchFeature object from apply_chat_template
                        input_ids = inputs.input_ids
                        attention_mask = getattr(inputs, 'attention_mask', None)
                        realtime_logger.debug(f"[DEBUG] BatchFeature input_ids type: {type(input_ids)}, shape: {getattr(input_ids, 'shape', 'N/A')}")
                    elif isinstance(inputs, dict):
                        # Dictionary format
                        input_ids = inputs.get('input_ids', inputs.get('audio_values'))
                        attention_mask = inputs.get('attention_mask')
                        realtime_logger.debug(f"[DEBUG] Dict input_ids type: {type(input_ids)}, shape: {getattr(input_ids, 'shape', 'N/A')}")
                    else:
                        # Direct tensor
                        input_ids = inputs
                        attention_mask = None
                        realtime_logger.debug(f"[DEBUG] Direct inputs type: {type(input_ids)}, shape: {getattr(input_ids, 'shape', 'N/A')}")

                    # Validate input_ids
                    if input_ids is None:
                        raise RuntimeError("Failed to extract input_ids from processor output")

                    # Prepare for iterative generation
                    realtime_logger.debug(f"[DEBUG] Preparing for iterative generation")
                    current_input_ids = input_ids
                    generated_tokens = []
                    word_buffer = ""
                    step = 0

                    # Enhanced generation parameters for streaming mode
                    realtime_logger.debug(f"[DEBUG] Creating generation config")
                    try:
                        # CRITICAL FIX: Safe token ID retrieval
                        pad_token_id = None
                        eos_token_id = None

                        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                            if hasattr(self.processor.tokenizer, 'eos_token_id'):
                                eos_token_id = self.processor.tokenizer.eos_token_id
                                pad_token_id = self.processor.tokenizer.eos_token_id
                                realtime_logger.debug(f"[DEBUG] Token IDs - EOS: {eos_token_id}, PAD: {pad_token_id}")

                        # ULTRA-LOW LATENCY: Optimized generation config for <100ms processing
                        generation_config = {
                            'do_sample': False,         # CHANGED: Greedy decoding for maximum speed
                            'temperature': float(0.0), # CHANGED: Greedy decoding
                            'top_p': float(1.0),        # CHANGED: Disable top-p sampling
                            'top_k': int(1),            # CHANGED: Greedy selection
                            'repetition_penalty': float(1.0),  # CHANGED: Disable repetition penalty
                            'length_penalty': float(0.8),      # CHANGED: Encourage shorter responses
                            'no_repeat_ngram_size': int(0),     # CHANGED: Disable n-gram blocking
                            'pad_token_id': int(pad_token_id) if pad_token_id is not None else None,
                            'eos_token_id': int(eos_token_id) if eos_token_id is not None else None,
                            'use_cache': True,
                            'early_stopping': True,       # CHANGED: Enable early stopping
                            'forced_eos_token_id': None,  # Don't force early EOS
                        }
                        realtime_logger.debug(f"[DEBUG] Generation config created successfully")

                    except Exception as config_error:
                        realtime_logger.error(f"[ERROR] Generation config error: {config_error}")
                        raise RuntimeError(f"Generation config failed: {config_error}")

                    # STREAMING LOOP: Generate tokens one by one with enhanced parameters
                    max_tokens = 200  # Increased target for longer responses
                    min_words_before_stop = 10  # Minimum words before allowing EOS
                    words_generated = 0
                    outputs = None  # Initialize outputs variable to prevent UnboundLocalError

                    for step in range(max_tokens):
                        # Generate next token with enhanced error handling
                        try:
                            realtime_logger.debug(f"[DEBUG] Starting generation step {step}")

                            # CRITICAL FIX: Safe generation config processing
                            safe_generation_config = {}
                            for key, value in generation_config.items():
                                try:
                                    # Convert numpy types to Python types
                                    if hasattr(value, 'item'):
                                        safe_generation_config[key] = value.item()
                                    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                                        safe_generation_config[key] = value.item()
                                    else:
                                        safe_generation_config[key] = value
                                except Exception as config_convert_error:
                                    realtime_logger.debug(f"Config conversion error for {key}: {config_convert_error}")
                                    safe_generation_config[key] = value

                            realtime_logger.debug(f"[DEBUG] Safe generation config prepared")

                            with torch.no_grad():
                                outputs = self.model.generate(
                                    current_input_ids,
                                    max_new_tokens=1,
                                    min_new_tokens=1,
                                    **safe_generation_config,
                                    output_scores=False,
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    return_dict_in_generate=False,
                                    synced_gpus=False,
                                )
                            realtime_logger.debug(f"[DEBUG] Generation step {step} completed")

                        except Exception as gen_error:
                            realtime_logger.error(f"[ERROR] Generation error at step {step}: {gen_error}")
                            realtime_logger.error(f"[ERROR] Generation config: {generation_config}")
                            break

                        # Extract new token with robust error handling
                        try:
                            if outputs.dim() > 1:
                                new_token_id = outputs[0, -1].item()
                            else:
                                new_token_id = outputs[-1].item()
                            generated_tokens.append(new_token_id)
                        except Exception as token_error:
                            realtime_logger.error(f"[ERROR] Token extraction error: {token_error}")
                            break

                        # CRITICAL FIX: Ultra-robust token decoding using safe utility function
                        try:
                            # Step 1: Decode token using processor
                            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                                raw_token_text = self.processor.tokenizer.decode([new_token_id], skip_special_tokens=True)
                            else:
                                raw_token_text = f"<{new_token_id}>"

                            # Step 2: Use safe utility function for type conversion
                            token_text = safe_to_string(raw_token_text)

                        except Exception as decode_error:
                            realtime_logger.warning(f"[WARN] Token decode error: {decode_error}, using fallback")
                            token_text = safe_to_string(f"<{new_token_id}>")

                        # CRITICAL FIX: Final safety check - ensure token_text is always a string
                        if not isinstance(token_text, str):
                            token_text = safe_to_string(token_text)
                        if not token_text:
                            token_text = ""

                        # Add to word buffer with ultra-safe string concatenation
                        try:
                            safe_token_text = safe_to_string(token_text)
                            if safe_token_text:
                                word_buffer += safe_token_text
                        except Exception as buffer_error:
                            realtime_logger.debug(f"Buffer concatenation error: {buffer_error}")

                        # Check if we have complete words (2+ words for TTS trigger)
                        # ULTRA-SAFE word detection and punctuation checking
                        words = []
                        has_punctuation = False

                        try:
                            # Safe word splitting
                            if isinstance(word_buffer, str) and word_buffer.strip():
                                words = word_buffer.strip().split()
                        except Exception as split_error:
                            realtime_logger.debug(f"Word splitting error: {split_error}")
                            words = []

                        try:
                            # CRITICAL FIX: ULTIMATE protection against numpy.float32 iteration error
                            # Layer 1: Ensure token_text is a string with comprehensive type checking
                            if not isinstance(token_text, str):
                                # Handle numpy types explicitly
                                if hasattr(token_text, 'item') and hasattr(token_text, 'dtype'):
                                    safe_token_text = str(token_text.item())
                                elif hasattr(token_text, '__float__'):
                                    safe_token_text = str(float(token_text))
                                else:
                                    safe_token_text = safe_to_string(token_text)
                            else:
                                safe_token_text = token_text

                            # Layer 2: Final validation before punctuation check
                            if not isinstance(safe_token_text, str):
                                realtime_logger.warning(f"[CRITICAL] Token text is not string after conversion: {type(safe_token_text)}, value: {safe_token_text}")
                                safe_token_text = str(safe_token_text) if safe_token_text is not None else ""

                            # Layer 3: Ensure we have a valid string before punctuation check
                            if not safe_token_text:
                                safe_token_text = ""

                            punctuation_chars = ['.', '!', '?', '\n', ',', ';', ':', '—', '–', '…']

                            # Layer 4: Additional safety wrapper around the punctuation check
                            try:
                                has_punctuation = safe_string_contains(safe_token_text, punctuation_chars)
                            except Exception as punct_check_error:
                                realtime_logger.error(f"[CRITICAL] Punctuation check failed: {punct_check_error}, token_text type: {type(safe_token_text)}, value: {safe_token_text}")
                                has_punctuation = False

                        except Exception as punct_error:
                            realtime_logger.debug(f"Punctuation check error: {punct_error}")
                            has_punctuation = False

                        if len(words) >= 2 or has_punctuation:
                            # Send words for TTS processing with enhanced logic
                            words_to_send = ' '.join(words[:-1]) if len(words) > 2 else ' '.join(words)
                            if words_to_send.strip():
                                words_generated += len(words_to_send.split())

                                yield {
                                    'type': 'words',
                                    'text': words_to_send.strip(),
                                    'tokens': generated_tokens[-len(words_to_send.split()):] if generated_tokens else [],
                                    'step': step,
                                    'is_complete': False,
                                    'chunk_id': chunk_id,
                                    'timestamp': time.time(),
                                    'word_count': words_generated
                                }

                                # Keep last word in buffer for next iteration
                                word_buffer = words[-1] if len(words) > 2 else ""

                                realtime_logger.debug(f"[TARGET] Sent {len(words_to_send.split())} words: '{words_to_send}'")

                        # Update input for next iteration
                        current_input_ids = outputs

                        # Enhanced EOS token handling - only stop if we have enough content
                        eos_token_id = generation_config.get('eos_token_id')
                        if (new_token_id == eos_token_id and
                            words_generated >= min_words_before_stop and
                            step > 20):  # Minimum 20 tokens before allowing EOS
                            realtime_logger.info(f"[OK] Natural EOS reached after {words_generated} words, {step} tokens")
                            break
                        elif new_token_id == eos_token_id and words_generated < min_words_before_stop:
                            realtime_logger.debug(f"[WARN] Early EOS ignored - only {words_generated} words generated")
                            # Continue generation despite EOS

                        # Adaptive delay based on generation speed
                        await asyncio.sleep(0.001 if step < 50 else 0.0005)

                    # Send any remaining text
                    if word_buffer.strip():
                        yield {
                            'type': 'words',
                            'text': word_buffer.strip(),
                            'tokens': [generated_tokens[-1]] if generated_tokens else [],
                            'step': step,
                            'is_complete': False,
                            'chunk_id': chunk_id,
                            'timestamp': time.time()
                        }

            inference_time = (time.time() - inference_start) * 1000
            total_time = (time.time() - start_time) * 1000

            # Decode final output with robust error handling
            try:
                if hasattr(self.processor, 'tokenizer') and generated_tokens:
                    response_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                elif hasattr(self.processor, 'tokenizer') and 'outputs' in locals() and outputs is not None:
                    response_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    response_text = f"Generated {len(generated_tokens)} tokens" if generated_tokens else "No response generated"
            except Exception as decode_error:
                realtime_logger.warning(f"[WARN] Final decode error: {decode_error}")
                response_text = f"Generated {len(generated_tokens)} tokens" if generated_tokens else "No response generated"

            # Send completion marker
            yield {
                'type': 'complete',
                'chunk_id': chunk_id,
                'response_text': response_text,
                'inference_time_ms': inference_time,
                'total_time_ms': total_time,
                'is_complete': True,
                'timestamp': time.time()
            }

            realtime_logger.info(f"[OK] Streaming generation completed for chunk {chunk_id} in {inference_time:.1f}ms")

        except Exception as e:
            realtime_logger.error(f"[ERROR] Error in streaming processing for chunk {chunk_id}: {e}")
            yield {
                'type': 'error',
                'error': str(e),
                'chunk_id': chunk_id,
                'is_complete': True,
                'timestamp': time.time()
            }

# Global model instance for real-time streaming
voxtral_model = VoxtralModel()

# FIXED: Add proper main execution block for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_model():
        """Test model initialization and basic functionality"""
        print("[EMOJI] Testing Voxtral Conversational Model with VAD...")
        
        try:
            # Initialize model
            await voxtral_model.initialize()
            
            # Test with dummy audio (silence)
            silent_audio = torch.zeros(16000) + 0.001  # Very quiet audio
            result = await voxtral_model.process_realtime_chunk(silent_audio, 1)
            print(f"Silent audio result: {result}")
            
            # Test with dummy audio (louder)
            loud_audio = torch.randn(16000) * 0.1  # Louder audio
            result = await voxtral_model.process_realtime_chunk(loud_audio, 2)
            print(f"Loud audio result: {result}")
            
            print(f"[OK] Test completed successfully")
            print(f"[STATS] Model info: {voxtral_model.get_model_info()}")
            
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_model())
