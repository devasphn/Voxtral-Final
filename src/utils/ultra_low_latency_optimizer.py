"""
Ultra-Low Latency Optimizer for Voxtral-Final System
Comprehensive optimization utilities for achieving sub-300ms total latency
"""

import os
import time
import torch
import psutil
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Setup logging
optimizer_logger = logging.getLogger("ultra_low_latency_optimizer")
optimizer_logger.setLevel(logging.INFO)

class UltraLowLatencyOptimizer:
    """
    Comprehensive optimizer for achieving world-class voice agent performance
    Targets: Voxtral <100ms, Kokoro TTS <150ms, Total <300ms
    """
    
    def __init__(self):
        self.optimization_flags = {
            'torch_compile': True,
            'cuda_graphs': True,
            'flash_attention': True,
            'quantization': True,
            'memory_optimization': True,
            'streaming_tts': True
        }
        self.performance_targets = {
            'voxtral_processing_ms': 100,
            'kokoro_generation_ms': 150,
            'audio_conversion_ms': 50,
            'total_end_to_end_ms': 300
        }
        self.metrics = {
            'latencies': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
    
    def setup_cuda_environment(self):
        """Setup optimal CUDA environment for maximum performance"""
        try:
            # Set optimal CUDA environment variables
            cuda_env = {
                'CUDA_VISIBLE_DEVICES': '0',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'CUDA_LAUNCH_BLOCKING': '0',
                'CUDA_CACHE_DISABLE': '0',
                'CUDA_AUTO_BOOST': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16',
                'TORCH_CUDNN_V8_API_ENABLED': '1',
                'TORCH_COMPILE_DEBUG': '0',
                'OMP_NUM_THREADS': '4',
                'MKL_NUM_THREADS': '4'
            }
            
            for key, value in cuda_env.items():
                os.environ[key] = value
                
            optimizer_logger.info("✅ CUDA environment optimized for maximum performance")
            return True
            
        except Exception as e:
            optimizer_logger.error(f"❌ Failed to setup CUDA environment: {e}")
            return False
    
    def optimize_pytorch_settings(self):
        """Apply PyTorch-specific optimizations"""
        try:
            if torch.cuda.is_available():
                # Enable cuDNN optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Memory optimization
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.95)
                
                # Enable Flash Attention if available
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                
                optimizer_logger.info("✅ PyTorch optimizations applied")
                return True
            else:
                optimizer_logger.warning("⚠️ CUDA not available, skipping GPU optimizations")
                return False
                
        except Exception as e:
            optimizer_logger.error(f"❌ Failed to optimize PyTorch settings: {e}")
            return False
    
    def optimize_model_compilation(self, model, model_name: str = "model"):
        """Apply optimal torch.compile configuration"""
        try:
            if not hasattr(torch, 'compile'):
                optimizer_logger.warning("⚠️ torch.compile not available")
                return model
            
            # Method 1: Mode-based compilation (fastest)
            try:
                compiled_model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    fullgraph=True,
                    dynamic=False
                )
                optimizer_logger.info(f"✅ {model_name} compiled with reduce-overhead mode")
                return compiled_model
                
            except Exception as mode_error:
                optimizer_logger.warning(f"⚠️ Mode compilation failed for {model_name}: {mode_error}")
                
                # Method 2: Options-based compilation
                try:
                    if torch.cuda.is_available():
                        compiled_model = torch.compile(
                            model,
                            options={
                                "triton.cudagraphs": True,
                                "max_autotune": True,
                                "epilogue_fusion": True,
                                "max_autotune_gemm": True,
                            }
                        )
                        optimizer_logger.info(f"✅ {model_name} compiled with CUDA options")
                        return compiled_model
                    else:
                        compiled_model = torch.compile(model)
                        optimizer_logger.info(f"✅ {model_name} compiled with basic optimizations")
                        return compiled_model
                        
                except Exception as options_error:
                    optimizer_logger.warning(f"⚠️ Options compilation failed for {model_name}: {options_error}")
                    return model
                    
        except Exception as e:
            optimizer_logger.error(f"❌ Model compilation failed for {model_name}: {e}")
            return model
    
    @contextmanager
    def measure_latency(self, operation_name: str):
        """Context manager for measuring operation latency"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            self.metrics['latencies'].append({
                'operation': operation_name,
                'latency_ms': latency_ms,
                'timestamp': time.time()
            })
            
            # Check against targets
            target = self.performance_targets.get(f"{operation_name}_ms", 1000)
            if latency_ms > target:
                optimizer_logger.warning(f"⚠️ {operation_name} exceeded target: {latency_ms:.1f}ms > {target}ms")
            else:
                optimizer_logger.info(f"✅ {operation_name} within target: {latency_ms:.1f}ms <= {target}ms")
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        try:
            performance = {
                'timestamp': time.time(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            if torch.cuda.is_available():
                try:
                    import GPUtil
                    gpu = GPUtil.getGPUs()[0]
                    performance.update({
                        'gpu_name': gpu.name,
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_memory_util': gpu.memoryUtil * 100,
                        'gpu_load': gpu.load * 100,
                        'gpu_temperature': gpu.temperature
                    })
                except ImportError:
                    performance.update({
                        'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                        'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                        'gpu_memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2)
                    })
            
            return performance
            
        except Exception as e:
            optimizer_logger.error(f"❌ Failed to get system performance: {e}")
            return {'error': str(e)}
    
    def optimize_audio_processing(self, chunk_size: int = 512, sample_rate: int = 16000) -> Dict[str, int]:
        """Get optimized audio processing parameters"""
        return {
            'chunk_size': min(chunk_size, 512),  # Ensure low latency
            'sample_rate': sample_rate,
            'frame_duration_ms': max(10, min(20, chunk_size * 1000 // sample_rate)),
            'buffer_size': chunk_size * 2,  # Double buffering
            'overlap_samples': chunk_size // 8  # 12.5% overlap
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        recent_latencies = self.metrics['latencies'][-10:] if self.metrics['latencies'] else []
        
        report = {
            'optimization_status': {
                'torch_compile': hasattr(torch, 'compile'),
                'cuda_available': torch.cuda.is_available(),
                'flash_attention': hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
                'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else False
            },
            'performance_targets': self.performance_targets,
            'recent_latencies': recent_latencies,
            'system_performance': self.get_system_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current state"""
        recommendations = []
        
        if not torch.cuda.is_available():
            recommendations.append("⚠️ CUDA not available - consider GPU acceleration")
        
        if not hasattr(torch, 'compile'):
            recommendations.append("⚠️ torch.compile not available - upgrade PyTorch to 2.0+")
        
        if torch.cuda.is_available():
            memory_util = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if memory_util > 0.9:
                recommendations.append("⚠️ High GPU memory usage - consider reducing batch size")
            elif memory_util < 0.5:
                recommendations.append("✅ GPU memory usage optimal")
        
        recent_latencies = self.metrics['latencies'][-5:] if self.metrics['latencies'] else []
        if recent_latencies:
            avg_latency = sum(l['latency_ms'] for l in recent_latencies) / len(recent_latencies)
            if avg_latency > 300:
                recommendations.append("⚠️ Average latency exceeds 300ms target")
            else:
                recommendations.append("✅ Latency within target range")
        
        return recommendations

# Global optimizer instance
ultra_optimizer = UltraLowLatencyOptimizer()
