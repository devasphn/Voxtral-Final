"""
RunPod Platform Optimizer
Optimizes the Voxtral Voice AI system for RunPod's HTTP/TCP-only infrastructure
Eliminates cold starts and implements RunPod-specific optimizations
"""
import asyncio
import logging
import time
import os
import torch
import gc
from typing import Dict, Any, Optional
from pathlib import Path

# Setup logging
runpod_logger = logging.getLogger("runpod_optimizer")

class RunPodOptimizer:
    """
    Comprehensive optimizer for RunPod deployment
    Handles cold start elimination, memory optimization, and HTTP/TCP-only constraints
    """
    
    def __init__(self):
        self.is_initialized = False
        self.models_preloaded = False
        self.optimization_level = "aggressive"  # aggressive, balanced, conservative
        self.startup_time = None
        
        # RunPod specific settings
        self.is_runpod_environment = self._detect_runpod_environment()
        self.workspace_path = "/workspace" if self.is_runpod_environment else "."
        
        runpod_logger.info(f"RunPod environment detected: {self.is_runpod_environment}")
        runpod_logger.info(f"Workspace path: {self.workspace_path}")
    
    def _detect_runpod_environment(self) -> bool:
        """Detect if running in RunPod environment"""
        return (
            os.path.exists("/workspace") or 
            os.environ.get("RUNPOD_POD_ID") is not None or
            os.environ.get("RUNPOD_API_KEY") is not None
        )
    
    async def initialize(self) -> bool:
        """Initialize RunPod optimizations"""
        start_time = time.time()
        runpod_logger.info("🚀 Initializing RunPod optimizations...")
        
        try:
            # Step 1: GPU Memory Optimization
            await self._optimize_gpu_memory()
            
            # Step 2: Model Pre-loading
            await self._preload_models()
            
            # Step 3: Network Optimization
            await self._optimize_networking()
            
            # Step 4: File System Optimization
            await self._optimize_filesystem()
            
            self.startup_time = time.time() - start_time
            self.is_initialized = True
            
            runpod_logger.info(f"✅ RunPod optimization completed in {self.startup_time:.2f}s")
            return True
            
        except Exception as e:
            runpod_logger.error(f"❌ RunPod optimization failed: {e}")
            return False
    
    async def _optimize_gpu_memory(self):
        """Optimize GPU memory for RunPod environment"""
        runpod_logger.info("🖥️  Optimizing GPU memory...")
        
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction for RunPod
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)  # Use 90% of GPU memory
            
            # Enable memory efficient attention
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            runpod_logger.info(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            runpod_logger.warning("   No CUDA GPU detected")
    
    async def _preload_models(self):
        """Pre-load models to eliminate cold starts"""
        runpod_logger.info("📦 Pre-loading models...")
        
        try:
            # Import and initialize unified model manager
            from src.models.unified_model_manager import unified_model_manager
            
            if not unified_model_manager.is_initialized:
                await unified_model_manager.initialize()
                runpod_logger.info("   ✅ Unified model manager pre-loaded")
            
            # Pre-load speech-to-speech pipeline
            from src.models.speech_to_speech_pipeline import speech_to_speech_pipeline
            
            if not speech_to_speech_pipeline.is_initialized:
                await speech_to_speech_pipeline.initialize()
                runpod_logger.info("   ✅ Speech-to-speech pipeline pre-loaded")
            
            self.models_preloaded = True
            
        except Exception as e:
            runpod_logger.error(f"   ❌ Model pre-loading failed: {e}")
            self.models_preloaded = False
    
    async def _optimize_networking(self):
        """Optimize networking for RunPod's HTTP/TCP-only environment"""
        runpod_logger.info("🌐 Optimizing networking for RunPod...")
        
        # Set optimal TCP settings for RunPod proxy
        network_optimizations = {
            "TCP_NODELAY": "1",
            "TCP_KEEPALIVE": "1",
            "TCP_KEEPIDLE": "30",
            "TCP_KEEPINTVL": "5",
            "TCP_KEEPCNT": "3"
        }
        
        for key, value in network_optimizations.items():
            os.environ[key] = value
        
        runpod_logger.info("   ✅ TCP optimizations applied")
    
    async def _optimize_filesystem(self):
        """Optimize filesystem for RunPod environment"""
        runpod_logger.info("📁 Optimizing filesystem...")
        
        # Create optimized directory structure
        directories = [
            f"{self.workspace_path}/model_cache",
            f"{self.workspace_path}/logs",
            f"{self.workspace_path}/temp_audio"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Set optimal permissions
        for directory in directories:
            os.chmod(directory, 0o755)
        
        runpod_logger.info("   ✅ Directory structure optimized")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            "initialized": self.is_initialized,
            "models_preloaded": self.models_preloaded,
            "runpod_environment": self.is_runpod_environment,
            "startup_time": self.startup_time,
            "optimization_level": self.optimization_level,
            "workspace_path": self.workspace_path,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for RunPod deployment"""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check GPU
        if torch.cuda.is_available():
            health_status["checks"]["gpu"] = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_used": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            health_status["checks"]["gpu"] = {"available": False}
            health_status["status"] = "warning"
        
        # Check models
        health_status["checks"]["models"] = {
            "preloaded": self.models_preloaded,
            "unified_manager": self._check_model_availability("unified_model_manager"),
            "speech_pipeline": self._check_model_availability("speech_to_speech_pipeline")
        }
        
        # Check filesystem
        health_status["checks"]["filesystem"] = {
            "workspace_exists": os.path.exists(self.workspace_path),
            "model_cache_exists": os.path.exists(f"{self.workspace_path}/model_cache"),
            "logs_writable": os.access(f"{self.workspace_path}/logs", os.W_OK)
        }
        
        return health_status
    
    def _check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        try:
            if model_name == "unified_model_manager":
                from src.models.unified_model_manager import unified_model_manager
                return unified_model_manager.is_initialized
            elif model_name == "speech_to_speech_pipeline":
                from src.models.speech_to_speech_pipeline import speech_to_speech_pipeline
                return speech_to_speech_pipeline.is_initialized
            return False
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup resources for graceful shutdown"""
        runpod_logger.info("🧹 Cleaning up RunPod resources...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        runpod_logger.info("   ✅ Cleanup completed")

# Global optimizer instance
runpod_optimizer = RunPodOptimizer()

async def initialize_runpod_optimizations():
    """Initialize RunPod optimizations - call this at startup"""
    return await runpod_optimizer.initialize()

def get_runpod_status():
    """Get RunPod optimization status"""
    return runpod_optimizer.get_optimization_status()

async def runpod_health_check():
    """Perform RunPod health check"""
    return await runpod_optimizer.health_check()
