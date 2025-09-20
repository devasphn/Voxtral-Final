"""
GPU Memory Manager for Kokoro TTS Integration
Handles shared GPU memory allocation between Voxtral and Kokoro TTS models
"""

import torch
import gc
import logging
import psutil
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Setup logging
gpu_logger = logging.getLogger("gpu_memory_manager")
gpu_logger.setLevel(logging.INFO)

@dataclass
class MemoryStats:
    """Memory statistics data structure"""
    total_vram_gb: float
    used_vram_gb: float
    voxtral_memory_gb: float
    kokoro_memory_gb: float
    available_vram_gb: float
    system_ram_gb: float
    system_ram_used_gb: float

class InsufficientVRAMError(Exception):
    """Raised when VRAM requirements are not met"""
    pass

class MemoryAllocationError(Exception):
    """Raised when memory allocation fails"""
    pass

class GPUMemoryManager:
    """
    Optimized GPU memory allocation and sharing between models
    Ensures efficient VRAM usage for Voxtral and Kokoro TTS integration
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_pool = None
        self.allocated_memory = {}
        self.voxtral_memory_gb = 0.0
        self.kokoro_memory_gb = 0.0

        # Memory requirements (in GB)
        self.min_vram_gb = 8.0  # RTX 3070/4060 Ti minimum
        self.recommended_vram_gb = 16.0  # RTX A4500 recommended
        self.voxtral_base_memory_gb = 4.5  # Estimated Voxtral memory usage
        self.kokoro_base_memory_gb = 1.5  # Estimated Kokoro TTS memory usage (lighter than Orpheus)
        
        gpu_logger.info(f"GPUMemoryManager initialized for device: {self.device}")
        
    def validate_vram_requirements(self) -> bool:
        """
        Validate that sufficient VRAM is available for both models
        Returns True if requirements are met, raises InsufficientVRAMError otherwise
        """
        try:
            if self.device == "cpu":
                gpu_logger.warning("[WARN] Running on CPU - GPU acceleration not available")
                return True
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_memory_gb = total_memory / (1024**3)
            
            # Check current memory usage
            allocated_memory = torch.cuda.memory_allocated(0)
            allocated_memory_gb = allocated_memory / (1024**3)
            
            available_memory_gb = total_memory_gb - allocated_memory_gb
            
            gpu_logger.info(f"[SEARCH] GPU Memory Analysis:")
            gpu_logger.info(f"   Total VRAM: {total_memory_gb:.2f} GB")
            gpu_logger.info(f"   Currently allocated: {allocated_memory_gb:.2f} GB")
            gpu_logger.info(f"   Available: {available_memory_gb:.2f} GB")
            gpu_logger.info(f"   Required for models: {self.voxtral_base_memory_gb + self.kokoro_base_memory_gb:.2f} GB")
            
            # Check minimum requirements
            if total_memory_gb < self.min_vram_gb:
                raise InsufficientVRAMError(
                    f"Insufficient VRAM: {total_memory_gb:.2f} GB available, "
                    f"minimum {self.min_vram_gb:.2f} GB required"
                )
            
            # Check if we have enough available memory for both models
            required_memory_gb = self.voxtral_base_memory_gb + self.kokoro_base_memory_gb
            if available_memory_gb < required_memory_gb:
                gpu_logger.warning(
                    f"[WARN] Limited available VRAM: {available_memory_gb:.2f} GB available, "
                    f"{required_memory_gb:.2f} GB required. May need memory optimization."
                )
                
                # Try to free up memory
                self.cleanup_unused_memory()
                
                # Re-check after cleanup
                allocated_memory = torch.cuda.memory_allocated(0)
                allocated_memory_gb = allocated_memory / (1024**3)
                available_memory_gb = total_memory_gb - allocated_memory_gb
                
                if available_memory_gb < required_memory_gb:
                    raise InsufficientVRAMError(
                        f"Insufficient available VRAM after cleanup: {available_memory_gb:.2f} GB available, "
                        f"{required_memory_gb:.2f} GB required"
                    )
            
            # Log recommendations
            if total_memory_gb >= self.recommended_vram_gb:
                gpu_logger.info("[OK] Excellent VRAM capacity - optimal performance expected")
            elif total_memory_gb >= self.min_vram_gb:
                gpu_logger.info("[OK] Sufficient VRAM - good performance expected")
            
            return True
            
        except Exception as e:
            gpu_logger.error(f"[ERROR] VRAM validation failed: {e}")
            raise
    
    def create_shared_memory_pool(self) -> Optional[Any]:
        """
        Create a shared memory pool for efficient tensor allocation
        Returns memory pool object or None if not applicable
        """
        try:
            if self.device == "cpu":
                gpu_logger.info("[IDEA] CPU mode - no GPU memory pool needed")
                return None
            
            gpu_logger.info("[SWIM] Creating shared GPU memory pool...")
            
            # Set memory allocation strategy for better sharing
            torch.cuda.empty_cache()
            
            # Configure memory allocation settings
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Reserve 90% of GPU memory for our models
                torch.cuda.set_per_process_memory_fraction(0.9)
                gpu_logger.info("[TARGET] Set GPU memory fraction to 90%")
            
            # Enable memory pooling if available
            if hasattr(torch.cuda, 'memory_pool'):
                self.memory_pool = torch.cuda.memory_pool()
                gpu_logger.info("[OK] GPU memory pool created successfully")
            else:
                gpu_logger.info("[IDEA] Using default CUDA memory allocator")
            
            return self.memory_pool
            
        except Exception as e:
            gpu_logger.error(f"[ERROR] Failed to create memory pool: {e}")
            return None
    
    def cleanup_unused_memory(self) -> None:
        """
        Cleanup unused GPU memory and run garbage collection
        """
        try:
            gpu_logger.info("[CLEANUP] Cleaning up unused GPU memory...")
            
            if self.device == "cuda":
                # Get memory stats before cleanup
                before_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                before_cached = torch.cuda.memory_reserved(0) / (1024**3)
                
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Get memory stats after cleanup
                after_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                after_cached = torch.cuda.memory_reserved(0) / (1024**3)
                
                freed_allocated = before_allocated - after_allocated
                freed_cached = before_cached - after_cached
                
                gpu_logger.info(f"[OK] Memory cleanup completed:")
                gpu_logger.info(f"   Freed allocated memory: {freed_allocated:.2f} GB")
                gpu_logger.info(f"   Freed cached memory: {freed_cached:.2f} GB")
                gpu_logger.info(f"   Current allocated: {after_allocated:.2f} GB")
                gpu_logger.info(f"   Current cached: {after_cached:.2f} GB")
            else:
                # CPU cleanup
                gc.collect()
                gpu_logger.info("[OK] CPU memory garbage collection completed")
                
        except Exception as e:
            gpu_logger.error(f"[ERROR] Memory cleanup failed: {e}")
    
    def track_model_memory(self, model_name: str, memory_gb: float) -> None:
        """
        Track memory usage for a specific model
        """
        self.allocated_memory[model_name] = memory_gb
        
        if model_name.lower() == "voxtral":
            self.voxtral_memory_gb = memory_gb
        elif model_name.lower() == "kokoro":
            self.kokoro_memory_gb = memory_gb
            
        gpu_logger.info(f"[STATS] Tracking {model_name} memory usage: {memory_gb:.2f} GB")
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get comprehensive memory usage statistics
        """
        try:
            if self.device == "cuda":
                # GPU memory stats
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                
                total_vram_gb = total_memory / (1024**3)
                used_vram_gb = allocated_memory / (1024**3)
                available_vram_gb = total_vram_gb - used_vram_gb
            else:
                # CPU mode - no GPU stats
                total_vram_gb = 0.0
                used_vram_gb = 0.0
                available_vram_gb = 0.0
            
            # System RAM stats
            system_memory = psutil.virtual_memory()
            system_ram_gb = system_memory.total / (1024**3)
            system_ram_used_gb = system_memory.used / (1024**3)
            
            return MemoryStats(
                total_vram_gb=total_vram_gb,
                used_vram_gb=used_vram_gb,
                voxtral_memory_gb=self.voxtral_memory_gb,
                kokoro_memory_gb=self.kokoro_memory_gb,
                available_vram_gb=available_vram_gb,
                system_ram_gb=system_ram_gb,
                system_ram_used_gb=system_ram_used_gb
            )
            
        except Exception as e:
            gpu_logger.error(f"[ERROR] Failed to get memory stats: {e}")
            # Return empty stats on error
            return MemoryStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """
        Optimize memory allocation settings based on available VRAM
        Returns optimization recommendations
        """
        try:
            stats = self.get_memory_stats()
            recommendations = {}
            
            gpu_logger.info("[TARGET] Analyzing memory allocation for optimization...")
            
            # Determine optimal settings based on available VRAM
            if stats.total_vram_gb >= self.recommended_vram_gb:
                # High VRAM - use optimal settings
                recommendations.update({
                    "precision": "fp16",
                    "batch_size": 1,
                    "use_cache": True,
                    "memory_efficient": False,
                    "optimization_level": "performance"
                })
                gpu_logger.info("[INIT] High VRAM detected - using performance optimization")
                
            elif stats.total_vram_gb >= self.min_vram_gb:
                # Medium VRAM - balanced settings
                recommendations.update({
                    "precision": "fp16",
                    "batch_size": 1,
                    "use_cache": True,
                    "memory_efficient": True,
                    "optimization_level": "balanced"
                })
                gpu_logger.info("[EMOJI] Medium VRAM detected - using balanced optimization")
                
            else:
                # Low VRAM - memory-efficient settings
                recommendations.update({
                    "precision": "fp32",  # Sometimes fp32 is more memory efficient
                    "batch_size": 1,
                    "use_cache": False,
                    "memory_efficient": True,
                    "optimization_level": "memory_efficient"
                })
                gpu_logger.info("[FLOPPY] Low VRAM detected - using memory-efficient optimization")
            
            # Add memory management recommendations
            recommendations.update({
                "cleanup_frequency": "after_each_generation" if stats.total_vram_gb < self.recommended_vram_gb else "periodic",
                "preload_models": stats.total_vram_gb >= self.recommended_vram_gb,
                "memory_monitoring": True
            })
            
            return recommendations
            
        except Exception as e:
            gpu_logger.error(f"[ERROR] Memory optimization analysis failed: {e}")
            return {"optimization_level": "safe", "memory_efficient": True}
    
    def monitor_memory_usage(self) -> Dict[str, Any]:
        """
        Monitor real-time memory usage and detect issues
        """
        try:
            stats = self.get_memory_stats()
            
            # Calculate usage percentages
            vram_usage_percent = (stats.used_vram_gb / stats.total_vram_gb * 100) if stats.total_vram_gb > 0 else 0
            ram_usage_percent = (stats.system_ram_used_gb / stats.system_ram_gb * 100) if stats.system_ram_gb > 0 else 0
            
            # Detect potential issues
            warnings = []
            if vram_usage_percent > 90:
                warnings.append("VRAM usage critically high (>90%)")
            elif vram_usage_percent > 80:
                warnings.append("VRAM usage high (>80%)")
                
            if ram_usage_percent > 90:
                warnings.append("System RAM usage critically high (>90%)")
            
            # Check for memory leaks (simplified detection)
            expected_model_memory = self.voxtral_memory_gb + self.kokoro_memory_gb
            if stats.used_vram_gb > expected_model_memory * 1.5:  # 50% overhead threshold
                warnings.append("Potential memory leak detected")
            
            monitoring_data = {
                "timestamp": time.time(),
                "vram_usage_percent": vram_usage_percent,
                "ram_usage_percent": ram_usage_percent,
                "warnings": warnings,
                "stats": stats,
                "status": "critical" if any("critically high" in w for w in warnings) else 
                         "warning" if warnings else "healthy"
            }
            
            # Log warnings
            for warning in warnings:
                gpu_logger.warning(f"[WARN] {warning}")
            
            return monitoring_data
            
        except Exception as e:
            gpu_logger.error(f"[ERROR] Memory monitoring failed: {e}")
            return {"status": "error", "error": str(e)}

# Global memory manager instance
gpu_memory_manager = GPUMemoryManager()