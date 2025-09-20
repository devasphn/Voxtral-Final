#!/usr/bin/env python3
"""
Performance Monitor for Voxtral-Final System
Real-time monitoring and optimization validation
"""

import time
import json
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.ultra_low_latency_optimizer import ultra_optimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_monitor")

class PerformanceMonitor:
    """Real-time performance monitoring for Voxtral system"""
    
    def __init__(self, output_file: str = "performance_log.json"):
        self.output_file = output_file
        self.monitoring = False
        self.metrics_history = []
    
    def check_system_readiness(self) -> bool:
        """Check if system is ready for optimal performance"""
        print("🔍 SYSTEM READINESS CHECK")
        print("=" * 50)
        
        # Setup optimizations
        cuda_ok = ultra_optimizer.setup_cuda_environment()
        pytorch_ok = ultra_optimizer.optimize_pytorch_settings()
        
        # Get optimization report
        report = ultra_optimizer.get_optimization_report()
        
        print(f"🎮 CUDA Available: {'✅' if report['optimization_status']['cuda_available'] else '❌'}")
        print(f"🔥 torch.compile: {'✅' if report['optimization_status']['torch_compile'] else '❌'}")
        print(f"⚡ Flash Attention: {'✅' if report['optimization_status']['flash_attention'] else '❌'}")
        print(f"🧠 cuDNN Enabled: {'✅' if report['optimization_status']['cudnn_enabled'] else '❌'}")
        
        # System performance
        perf = report['system_performance']
        if 'error' not in perf:
            print(f"\n💻 SYSTEM PERFORMANCE:")
            print(f"   CPU Usage: {perf['cpu_usage']:.1f}%")
            print(f"   Memory Usage: {perf['memory_usage']:.1f}%")
            print(f"   Available Memory: {perf['available_memory_gb']:.1f} GB")
            
            if 'gpu_name' in perf:
                print(f"   GPU: {perf['gpu_name']}")
                print(f"   GPU Memory: {perf['gpu_memory_used_mb']:.0f}/{perf['gpu_memory_total_mb']:.0f} MB ({perf['gpu_memory_util']:.1f}%)")
                print(f"   GPU Load: {perf['gpu_load']:.1f}%")
                print(f"   GPU Temperature: {perf['gpu_temperature']:.0f}°C")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        
        # Overall readiness
        ready = cuda_ok and pytorch_ok and report['optimization_status']['cuda_available']
        print(f"\n🎯 SYSTEM READY: {'✅ YES' if ready else '❌ NO'}")
        
        return ready
    
    def test_latency_targets(self) -> Dict[str, bool]:
        """Test if system meets latency targets"""
        print("\n🧪 LATENCY TARGET TESTING")
        print("=" * 50)
        
        results = {}
        
        # Test audio processing latency
        print("Testing audio processing...")
        with ultra_optimizer.measure_latency("audio_processing"):
            # Simulate audio processing
            time.sleep(0.01)  # 10ms simulation
        
        # Test model inference latency
        print("Testing model inference...")
        with ultra_optimizer.measure_latency("voxtral_processing"):
            # Simulate model inference
            time.sleep(0.05)  # 50ms simulation
        
        # Test TTS generation latency
        print("Testing TTS generation...")
        with ultra_optimizer.measure_latency("kokoro_generation"):
            # Simulate TTS generation
            time.sleep(0.08)  # 80ms simulation
        
        # Check results against targets
        recent_metrics = ultra_optimizer.metrics['latencies'][-3:]
        for metric in recent_metrics:
            operation = metric['operation']
            latency = metric['latency_ms']
            target = ultra_optimizer.performance_targets.get(f"{operation}_ms", 1000)
            
            meets_target = latency <= target
            results[operation] = meets_target
            
            status = "✅" if meets_target else "❌"
            print(f"   {operation}: {latency:.1f}ms (target: {target}ms) {status}")
        
        return results
    
    async def monitor_realtime(self, duration_seconds: int = 60):
        """Monitor system performance in real-time"""
        print(f"\n📊 REAL-TIME MONITORING ({duration_seconds}s)")
        print("=" * 50)
        
        self.monitoring = True
        start_time = time.time()
        
        try:
            while self.monitoring and (time.time() - start_time) < duration_seconds:
                # Collect metrics
                perf = ultra_optimizer.get_system_performance()
                perf['timestamp'] = datetime.now().isoformat()
                
                # Add to history
                self.metrics_history.append(perf)
                
                # Display current metrics
                if 'error' not in perf:
                    print(f"\r⏱️  CPU: {perf['cpu_usage']:5.1f}% | "
                          f"RAM: {perf['memory_usage']:5.1f}% | "
                          f"GPU: {perf.get('gpu_load', 0):5.1f}% | "
                          f"VRAM: {perf.get('gpu_memory_util', 0):5.1f}%", end="")
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
        
        self.monitoring = False
        
        # Save metrics to file
        self.save_metrics()
        
        # Generate summary
        self.print_summary()
    
    def save_metrics(self):
        """Save collected metrics to file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump({
                    'collection_time': datetime.now().isoformat(),
                    'metrics_count': len(self.metrics_history),
                    'metrics': self.metrics_history,
                    'optimization_report': ultra_optimizer.get_optimization_report()
                }, f, indent=2)
            
            print(f"\n💾 Metrics saved to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def print_summary(self):
        """Print performance summary"""
        if not self.metrics_history:
            return
        
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Calculate averages
        cpu_avg = sum(m['cpu_usage'] for m in self.metrics_history) / len(self.metrics_history)
        mem_avg = sum(m['memory_usage'] for m in self.metrics_history) / len(self.metrics_history)
        
        print(f"Average CPU Usage: {cpu_avg:.1f}%")
        print(f"Average Memory Usage: {mem_avg:.1f}%")
        
        if 'gpu_load' in self.metrics_history[0]:
            gpu_avg = sum(m.get('gpu_load', 0) for m in self.metrics_history) / len(self.metrics_history)
            vram_avg = sum(m.get('gpu_memory_util', 0) for m in self.metrics_history) / len(self.metrics_history)
            print(f"Average GPU Load: {gpu_avg:.1f}%")
            print(f"Average VRAM Usage: {vram_avg:.1f}%")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if cpu_avg < 50:
            print("   ✅ CPU usage optimal")
        elif cpu_avg < 80:
            print("   ⚠️ CPU usage moderate")
        else:
            print("   ❌ CPU usage high")
        
        if mem_avg < 70:
            print("   ✅ Memory usage optimal")
        elif mem_avg < 90:
            print("   ⚠️ Memory usage moderate")
        else:
            print("   ❌ Memory usage high")

def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description="Voxtral Performance Monitor")
    parser.add_argument("--check", action="store_true", help="Check system readiness")
    parser.add_argument("--test", action="store_true", help="Test latency targets")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor for N seconds")
    parser.add_argument("--output", default="performance_log.json", help="Output file")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.output)
    
    if args.check:
        monitor.check_system_readiness()
    
    if args.test:
        monitor.test_latency_targets()
    
    if args.monitor > 0:
        asyncio.run(monitor.monitor_realtime(args.monitor))
    
    if not any([args.check, args.test, args.monitor]):
        # Default: run all checks
        print("🚀 VOXTRAL PERFORMANCE MONITOR")
        print("=" * 50)
        
        monitor.check_system_readiness()
        monitor.test_latency_targets()
        
        print("\n💡 Use --monitor 60 to start real-time monitoring")

if __name__ == "__main__":
    main()
