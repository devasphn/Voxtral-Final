"""
Unified Model Manager for Kokoro TTS Integration
Centralized management of both Voxtral and Kokoro TTS models with shared GPU memory
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple
from threading import Lock
import torch
import gc

# Import model classes
from src.models.voxtral_model_realtime import VoxtralModel
from src.models.kokoro_model_realtime import KokoroTTSModel
from src.utils.gpu_memory_manager import GPUMemoryManager, InsufficientVRAMError

# Setup logging
unified_logger = logging.getLogger("unified_model_manager")
unified_logger.setLevel(logging.INFO)

class ModelInitializationError(Exception):
    """Raised when model initialization fails"""
    pass

class UnifiedModelManager:
    """
    Centralized management of both Voxtral and Kokoro TTS models
    Handles initialization order, memory sharing, and lifecycle management
    """

    def __init__(self):
        self.voxtral_model = None
        self.kokoro_model = None
        self.gpu_memory_manager = None
        self.initialization_lock = Lock()
        self.is_initialized = False

        # Initialization state tracking
        self.voxtral_initialized = False
        self.kokoro_initialized = False
        self.memory_manager_initialized = False

        # ULTRA-LOW LATENCY: Minimal performance tracking to reduce overhead
        self.initialization_times = {}
        self.memory_usage = {}

        # ULTRA-LOW LATENCY: Optimization flags
        self.skip_detailed_logging = True  # Reduce logging overhead
        self.fast_initialization = True   # Skip non-essential initialization steps

        unified_logger.info("UnifiedModelManager created (ultra-low latency mode)")
    
    async def initialize(self) -> bool:
        """
        Initialize both models with optimal memory management
        Returns True if successful, raises ModelInitializationError on failure
        """
        try:
            with self.initialization_lock:
                if self.is_initialized:
                    unified_logger.info("[OK] Models already initialized")
                    return True
                
                unified_logger.info("[INIT] Starting unified model initialization...")
                total_start_time = time.time()
                
                # Step 1: Initialize GPU Memory Manager
                await self._initialize_memory_manager()
                
                # ULTRA-LOW LATENCY: Step 2 - Initialize models in optimal order
                await self._initialize_voxtral_model()
                await self._initialize_kokoro_model()

                # ULTRA-LOW LATENCY: Step 3 - Minimal post-initialization (skip heavy optimizations)
                if not self.fast_initialization:
                    await self._post_initialization_optimization()

                total_time = time.time() - total_start_time
                self.initialization_times["total"] = total_time

                self.is_initialized = True
                unified_logger.info(f"[SUCCESS] ULTRA-LOW LATENCY model initialization completed in {total_time:.2f}s")

                # ULTRA-LOW LATENCY: Skip detailed memory logging unless needed
                if not self.skip_detailed_logging:
                    await self._log_memory_statistics()
                
                return True
                
        except Exception as e:
            unified_logger.error(f"[ERROR] Unified model initialization failed: {e}")
            await self._cleanup_partial_initialization()
            raise ModelInitializationError(f"Failed to initialize models: {e}")
    
    async def _initialize_memory_manager(self):
        """Initialize GPU memory manager and validate requirements"""
        try:
            unified_logger.info("[BRAIN] Initializing GPU Memory Manager...")
            start_time = time.time()
            
            self.gpu_memory_manager = GPUMemoryManager()
            
            # Validate VRAM requirements
            self.gpu_memory_manager.validate_vram_requirements()
            
            # Create shared memory pool
            shared_pool = self.gpu_memory_manager.create_shared_memory_pool()
            
            self.memory_manager_initialized = True
            init_time = time.time() - start_time
            self.initialization_times["memory_manager"] = init_time
            
            unified_logger.info(f"[OK] GPU Memory Manager initialized in {init_time:.2f}s")
            
        except InsufficientVRAMError as e:
            unified_logger.error(f"[ERROR] Insufficient VRAM: {e}")
            raise ModelInitializationError(f"VRAM requirements not met: {e}")
        except Exception as e:
            unified_logger.error(f"[ERROR] Memory manager initialization failed: {e}")
            raise ModelInitializationError(f"Memory manager initialization failed: {e}")
    
    async def _initialize_voxtral_model(self):
        """Initialize Voxtral model first for optimal memory layout"""
        try:
            unified_logger.info("[VAD] Initializing Voxtral model...")
            start_time = time.time()
            
            # Import and create Voxtral model
            self.voxtral_model = VoxtralModel()
            
            # Initialize with memory optimization
            await self.voxtral_model.initialize()
            
            # Track memory usage
            if self.gpu_memory_manager.device == "cuda":
                voxtral_memory = torch.cuda.memory_allocated() / (1024**3)
                self.gpu_memory_manager.track_model_memory("voxtral", voxtral_memory)
                self.memory_usage["voxtral_gb"] = voxtral_memory
            
            self.voxtral_initialized = True
            init_time = time.time() - start_time
            self.initialization_times["voxtral"] = init_time
            
            unified_logger.info(f"[OK] Voxtral model initialized in {init_time:.2f}s")
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Voxtral initialization failed: {e}")
            raise ModelInitializationError(f"Voxtral initialization failed: {e}")
    
    async def _initialize_kokoro_model(self):
        """Initialize Kokoro TTS model"""
        try:
            unified_logger.info("[AUDIO] Initializing Kokoro TTS model...")
            start_time = time.time()

            # Create Kokoro TTS Model
            self.kokoro_model = KokoroTTSModel()

            # Initialize the model
            success = await self.kokoro_model.initialize()

            if not success:
                unified_logger.error("[ERROR] Kokoro TTS model initialization failed")
                raise ModelInitializationError("Kokoro TTS model initialization failed")

            # Track memory usage
            if self.gpu_memory_manager.device == "cuda":
                total_memory = torch.cuda.memory_allocated() / (1024**3)
                tts_memory = total_memory - self.memory_usage.get("voxtral_gb", 0)
                self.gpu_memory_manager.track_model_memory("kokoro", tts_memory)
                self.memory_usage["kokoro_gb"] = tts_memory

            self.kokoro_initialized = True
            init_time = time.time() - start_time
            self.initialization_times["kokoro"] = init_time

            unified_logger.info(f"[OK] Kokoro TTS model initialized in {init_time:.2f}s")

        except Exception as e:
            unified_logger.error(f"[ERROR] Kokoro TTS model initialization failed: {e}")
            raise ModelInitializationError(f"Kokoro TTS model initialization failed: {e}")
    
    async def _post_initialization_optimization(self):
        """Perform post-initialization memory optimization"""
        try:
            unified_logger.info("[FAST] Performing post-initialization optimization...")
            
            # Clean up any unused memory
            self.gpu_memory_manager.cleanup_unused_memory()
            
            # Get memory optimization recommendations
            recommendations = self.gpu_memory_manager.optimize_memory_allocation()
            
            # Apply recommendations if needed
            if recommendations.get("optimization_level") == "memory_efficient":
                unified_logger.info("[FLOPPY] Applying memory-efficient optimizations...")
                # Could implement model-specific optimizations here
            
            # Verify both models are working
            await self._verify_model_functionality()
            
            unified_logger.info("[OK] Post-initialization optimization completed")
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Post-initialization optimization failed: {e}")
            raise ModelInitializationError(f"Post-initialization optimization failed: {e}")
    
    async def _verify_model_functionality(self):
        """Verify both models are functioning correctly"""
        try:
            unified_logger.info("[SEARCH] Verifying model functionality...")
            
            # Test Voxtral model
            if self.voxtral_model and self.voxtral_model.is_initialized:
                model_info = self.voxtral_model.get_model_info()
                if model_info.get("status") != "initialized":
                    raise ModelInitializationError("Voxtral model verification failed")
                unified_logger.info("[OK] Voxtral model verification passed")
            
            # Test Kokoro TTS model
            if self.kokoro_model and self.kokoro_model.is_initialized:
                model_info = self.kokoro_model.get_model_info()
                if not model_info.get("is_initialized"):
                    raise ModelInitializationError("Kokoro TTS model verification failed")

                unified_logger.info("[OK] Kokoro TTS model verification passed")
            
            unified_logger.info("[OK] All model functionality verified")
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Model verification failed: {e}")
            raise ModelInitializationError(f"Model verification failed: {e}")
    
    async def _log_memory_statistics(self):
        """Log comprehensive memory usage statistics"""
        try:
            if self.gpu_memory_manager:
                stats = self.gpu_memory_manager.get_memory_stats()
                
                unified_logger.info("[STATS] Final Memory Statistics:")
                unified_logger.info(f"   Total VRAM: {stats.total_vram_gb:.2f} GB")
                unified_logger.info(f"   Used VRAM: {stats.used_vram_gb:.2f} GB")
                unified_logger.info(f"   Available VRAM: {stats.available_vram_gb:.2f} GB")
                unified_logger.info(f"   Voxtral Memory: {stats.voxtral_memory_gb:.2f} GB")
                unified_logger.info(f"   Kokoro Memory: {stats.kokoro_memory_gb:.2f} GB")
                unified_logger.info(f"   System RAM: {stats.system_ram_gb:.2f} GB")
                unified_logger.info(f"   System RAM Used: {stats.system_ram_used_gb:.2f} GB")
                
                # Calculate efficiency metrics
                total_model_memory = stats.voxtral_memory_gb + stats.kokoro_memory_gb
                memory_efficiency = (total_model_memory / stats.used_vram_gb * 100) if stats.used_vram_gb > 0 else 0
                
                unified_logger.info(f"   Memory Efficiency: {memory_efficiency:.1f}%")
                
        except Exception as e:
            unified_logger.warning(f"[WARN] Failed to log memory statistics: {e}")
    
    async def _cleanup_partial_initialization(self):
        """Cleanup resources from partial initialization"""
        try:
            unified_logger.info("[CLEANUP] Cleaning up partial initialization...")
            
            if self.kokoro_model:
                await self.kokoro_model.cleanup()
                self.kokoro_model = None
                self.kokoro_initialized = False
            
            if self.voxtral_model:
                # Voxtral model doesn't have async cleanup, but we can clear references
                self.voxtral_model = None
                self.voxtral_initialized = False
            
            if self.gpu_memory_manager:
                self.gpu_memory_manager.cleanup_unused_memory()
            
            self.is_initialized = False
            unified_logger.info("[OK] Partial initialization cleanup completed")
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Cleanup failed: {e}")
    
    async def get_voxtral_model(self) -> Optional[VoxtralModel]:
        """Get initialized Voxtral model"""
        if not self.is_initialized or not self.voxtral_initialized:
            raise ModelInitializationError("Voxtral model not initialized")
        return self.voxtral_model
    
    async def get_kokoro_model(self) -> Optional[KokoroTTSModel]:
        """Get initialized Kokoro TTS model"""
        if not self.is_initialized or not self.kokoro_initialized:
            raise ModelInitializationError("Kokoro TTS model not initialized")
        return self.kokoro_model
    
    async def cleanup_gpu_memory(self) -> None:
        """Cleanup GPU memory and run garbage collection"""
        try:
            unified_logger.info("[CLEANUP] Cleaning up GPU memory...")
            
            if self.gpu_memory_manager:
                self.gpu_memory_manager.cleanup_unused_memory()
            
            # Additional cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
            unified_logger.info("[OK] GPU memory cleanup completed")
            
        except Exception as e:
            unified_logger.error(f"[ERROR] GPU memory cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        try:
            if not self.gpu_memory_manager:
                return {"error": "Memory manager not initialized"}
            
            base_stats = self.gpu_memory_manager.get_memory_stats()
            
            return {
                "memory_stats": {
                    "total_vram_gb": base_stats.total_vram_gb,
                    "used_vram_gb": base_stats.used_vram_gb,
                    "available_vram_gb": base_stats.available_vram_gb,
                    "voxtral_memory_gb": base_stats.voxtral_memory_gb,
                    "kokoro_memory_gb": base_stats.kokoro_memory_gb,
                    "system_ram_gb": base_stats.system_ram_gb,
                    "system_ram_used_gb": base_stats.system_ram_used_gb
                },
                "initialization_stats": {
                    "is_initialized": self.is_initialized,
                    "voxtral_initialized": self.voxtral_initialized,
                    "kokoro_initialized": self.kokoro_initialized,
                    "initialization_times": self.initialization_times
                },
                "model_info": {
                    "voxtral_available": self.voxtral_model is not None,
                    "kokoro_available": self.kokoro_model is not None,
                    "device": self.gpu_memory_manager.device if self.gpu_memory_manager else "unknown"
                }
            }
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        try:
            info = {
                "unified_manager": {
                    "is_initialized": self.is_initialized,
                    "voxtral_initialized": self.voxtral_initialized,
                    "kokoro_initialized": self.kokoro_initialized,
                    "memory_manager_initialized": self.memory_manager_initialized
                },
                "initialization_times": self.initialization_times,
                "memory_usage": self.memory_usage
            }
            
            # Add Voxtral model info
            if self.voxtral_model:
                info["voxtral"] = self.voxtral_model.get_model_info()
            
            # Add Kokoro model info
            if self.kokoro_model:
                info["kokoro"] = self.kokoro_model.get_model_info()
            
            # Add memory manager info
            if self.gpu_memory_manager:
                info["memory_manager"] = self.get_memory_stats()
            
            return info
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown and cleanup all resources"""
        try:
            unified_logger.info("[EMOJI] Shutting down Unified Model Manager...")
            
            # Cleanup Kokoro model
            if self.kokoro_model:
                await self.kokoro_model.cleanup()
                self.kokoro_model = None
                self.kokoro_initialized = False
            
            # Cleanup Voxtral model (no async cleanup available)
            if self.voxtral_model:
                self.voxtral_model = None
                self.voxtral_initialized = False
            
            # Final memory cleanup
            if self.gpu_memory_manager:
                self.gpu_memory_manager.cleanup_unused_memory()
                self.gpu_memory_manager = None
                self.memory_manager_initialized = False
            
            # Clear state
            self.is_initialized = False
            self.initialization_times.clear()
            self.memory_usage.clear()
            
            unified_logger.info("[OK] Unified Model Manager shutdown completed")
            
        except Exception as e:
            unified_logger.error(f"[ERROR] Shutdown failed: {e}")

# Global unified model manager instance
unified_model_manager = UnifiedModelManager()