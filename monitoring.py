#!/usr/bin/env python3
"""
Voxtral Performance Monitoring
"""
import psutil
import time
import json
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPUtil not available, GPU monitoring disabled")

class VoxtralMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
    
    def get_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_metrics = {}
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # First GPU
                        gpu_metrics = {
                            'utilization': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'temperature': gpu.temperature
                        }
                except Exception as e:
                    print(f"GPU metrics error: {e}")
                    gpu_metrics = {'error': str(e)}
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - self.start_time,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'gpu': gpu_metrics
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return None
    
    def log_metrics(self):
        """Log current system metrics"""
        metrics = self.get_system_metrics()
        if metrics:
            gpu_util = metrics['gpu'].get('utilization', 0) if metrics['gpu'] else 0
            gpu_mem = metrics['gpu'].get('memory_percent', 0) if metrics['gpu'] else 0
            
            print(f"[{metrics['timestamp']}] "
                  f"CPU: {metrics['cpu_percent']:.1f}% | "
                  f"RAM: {metrics['memory_percent']:.1f}% | "
                  f"GPU: {gpu_util:.1f}% | "
                  f"VRAM: {gpu_mem:.1f}%")
            
            self.metrics_history.append(metrics)
            
            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
    
    def get_alert_status(self):
        """Check for performance alerts"""
        metrics = self.get_system_metrics()
        if not metrics:
            return []
        
        alerts = []
        
        if metrics['cpu_percent'] > 90:
            alerts.append(f"⚠️ High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > 85:
            alerts.append(f"⚠️ High memory usage: {metrics['memory_percent']:.1f}%")
        
        if metrics['gpu']:
            gpu_util = metrics['gpu'].get('utilization', 0)
            if gpu_util > 95:
                alerts.append(f"⚠️ High GPU usage: {gpu_util:.1f}%")
            
            gpu_mem = metrics['gpu'].get('memory_percent', 0)
            if gpu_mem > 90:
                alerts.append(f"⚠️ High VRAM usage: {gpu_mem:.1f}%")
            
            gpu_temp = metrics['gpu'].get('temperature', 0)
            if gpu_temp > 80:
                alerts.append(f"🔥 High GPU temperature: {gpu_temp}°C")
        
        return alerts

def main():
    print("📊 Starting Voxtral Performance Monitor")
    monitor = VoxtralMonitor()
    
    try:
        while True:
            monitor.log_metrics()
            
            # Check for alerts
            alerts = monitor.get_alert_status()
            for alert in alerts:
                print(alert)
            
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\n📊 Monitoring stopped")

if __name__ == "__main__":
    main()