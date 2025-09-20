"""
Performance Monitor for Kokoro TTS Integration
Real-time performance tracking and latency optimization
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
import statistics
import threading

# Setup logging
perf_logger = logging.getLogger("performance_monitor")
perf_logger.setLevel(logging.INFO)

@dataclass
class LatencyBreakdown:
    """Latency breakdown data structure"""
    voxtral_processing_ms: float
    text_generation_ms: float
    kokoro_generation_ms: float
    audio_conversion_ms: float
    total_latency_ms: float
    target_met: bool

@dataclass
class TimingOperation:
    """Individual timing operation"""
    operation_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceDegradationError(Exception):
    """Raised when performance degrades significantly"""
    pass

class LatencyExceededError(Exception):
    """Raised when latency targets are exceeded"""
    pass

class PerformanceMonitor:
    """
    Real-time performance tracking and latency optimization
    Monitors end-to-end latency and provides optimization recommendations
    """
    
    def __init__(self):
        self.active_timings = {}  # operation_id -> TimingOperation
        self.completed_operations = deque(maxlen=1000)  # Keep last 1000 operations
        self.latency_history = deque(maxlen=100)  # Keep last 100 latency measurements
        self.lock = threading.Lock()
        
        # Performance targets (in milliseconds)
        self.targets = {
            "voxtral_processing_ms": 100,
            "kokoro_generation_ms": 150,
            "audio_conversion_ms": 50,
            "total_end_to_end_ms": 300
        }
        
        # Performance statistics
        self.stats = {
            "total_operations": 0,
            "operations_within_target": 0,
            "operations_exceeded_target": 0,
            "average_latency_ms": 0.0,
            "min_latency_ms": float('inf'),
            "max_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "consecutive_failures": 5,  # Alert after 5 consecutive target misses
            "degradation_threshold": 1.5,  # Alert if latency increases by 50%
            "success_rate_threshold": 0.8  # Alert if success rate drops below 80%
        }
        
        self.consecutive_failures = 0
        self.baseline_latency_ms = None
        
        perf_logger.info("PerformanceMonitor initialized")
        perf_logger.info(f"[TARGET] Performance targets: {self.targets}")
    
    def start_timing(self, operation: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start timing an operation
        Returns operation ID for later reference
        """
        operation_id = f"{operation}_{int(time.time() * 1000000)}"  # Microsecond precision
        
        timing_op = TimingOperation(
            operation_id=operation_id,
            operation_name=operation,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_timings[operation_id] = timing_op
        
        perf_logger.debug(f"[TIME] Started timing: {operation} (ID: {operation_id})")
        return operation_id
    
    def end_timing(self, timing_id: str) -> float:
        """
        End timing for an operation
        Returns duration in milliseconds
        """
        end_time = time.time()
        
        with self.lock:
            if timing_id not in self.active_timings:
                perf_logger.warning(f"[WARN] Timing ID not found: {timing_id}")
                return 0.0
            
            timing_op = self.active_timings.pop(timing_id)
            timing_op.end_time = end_time
            timing_op.duration_ms = (end_time - timing_op.start_time) * 1000
            
            # Add to completed operations
            self.completed_operations.append(timing_op)
            
            # Update statistics
            self._update_statistics(timing_op)
        
        perf_logger.debug(f"[TIME] Ended timing: {timing_op.operation_name} - {timing_op.duration_ms:.1f}ms")
        return timing_op.duration_ms
    
    def log_latency_breakdown(self, components: Dict[str, float]) -> None:
        """
        Log latency breakdown by component and check targets
        """
        try:
            # Calculate total latency
            total_latency = sum(components.values())
            
            # Create latency breakdown
            breakdown = LatencyBreakdown(
                voxtral_processing_ms=components.get("voxtral_processing_ms", 0.0),
                text_generation_ms=components.get("text_generation_ms", 0.0),
                kokoro_generation_ms=components.get("kokoro_generation_ms", 0.0),
                audio_conversion_ms=components.get("audio_conversion_ms", 0.0),
                total_latency_ms=total_latency,
                target_met=total_latency <= self.targets["total_end_to_end_ms"]
            )
            
            # Add to history
            with self.lock:
                self.latency_history.append(breakdown)
                self.stats["total_operations"] += 1
                
                if breakdown.target_met:
                    self.stats["operations_within_target"] += 1
                    self.consecutive_failures = 0
                else:
                    self.stats["operations_exceeded_target"] += 1
                    self.consecutive_failures += 1
            
            # Log breakdown
            perf_logger.info(f"[STATS] Latency breakdown (Total: {total_latency:.1f}ms):")
            for component, latency in components.items():
                target = self.targets.get(component, None)
                status = "[OK]" if target and latency <= target else "[WARN]" if target else "ℹ[EMOJI]"
                target_info = f" (target: {target}ms)" if target else ""
                perf_logger.info(f"   {status} {component}: {latency:.1f}ms{target_info}")
            
            # Check for performance issues
            self._check_performance_alerts(breakdown)
            
        except Exception as e:
            perf_logger.error(f"[ERROR] Failed to log latency breakdown: {e}")
    
    def check_performance_targets(self) -> Dict[str, bool]:
        """
        Check if performance targets are being met
        Returns dict of target compliance
        """
        try:
            if not self.latency_history:
                return {"no_data": True}
            
            recent_operations = list(self.latency_history)[-10:]  # Last 10 operations
            
            results = {}
            
            # Check individual component targets
            for target_name, target_value in self.targets.items():
                if target_name == "total_end_to_end_ms":
                    values = [op.total_latency_ms for op in recent_operations]
                elif target_name == "voxtral_processing_ms":
                    values = [op.voxtral_processing_ms for op in recent_operations]
                elif target_name == "kokoro_generation_ms":
                    values = [op.kokoro_generation_ms for op in recent_operations]
                elif target_name == "audio_conversion_ms":
                    values = [op.audio_conversion_ms for op in recent_operations]
                else:
                    continue
                
                if values:
                    avg_value = statistics.mean(values)
                    results[target_name] = avg_value <= target_value
                    results[f"{target_name}_avg"] = avg_value
            
            # Overall success rate
            total_within_target = sum(1 for op in recent_operations if op.target_met)
            success_rate = total_within_target / len(recent_operations) if recent_operations else 0
            results["overall_success_rate"] = success_rate
            results["success_rate_target_met"] = success_rate >= self.alert_thresholds["success_rate_threshold"]
            
            return results
            
        except Exception as e:
            perf_logger.error(f"[ERROR] Failed to check performance targets: {e}")
            return {"error": str(e)}
    
    def _update_statistics(self, timing_op: TimingOperation):
        """Update performance statistics with new timing data"""
        try:
            duration = timing_op.duration_ms
            
            # Update min/max
            self.stats["min_latency_ms"] = min(self.stats["min_latency_ms"], duration)
            self.stats["max_latency_ms"] = max(self.stats["max_latency_ms"], duration)
            
            # Update average (running average)
            total_ops = len(self.completed_operations)
            if total_ops > 0:
                recent_durations = [op.duration_ms for op in list(self.completed_operations)[-100:]]
                self.stats["average_latency_ms"] = statistics.mean(recent_durations)
                
                # Calculate percentiles
                if len(recent_durations) >= 10:
                    sorted_durations = sorted(recent_durations)
                    self.stats["p95_latency_ms"] = sorted_durations[int(len(sorted_durations) * 0.95)]
                    self.stats["p99_latency_ms"] = sorted_durations[int(len(sorted_durations) * 0.99)]
            
        except Exception as e:
            perf_logger.error(f"[ERROR] Failed to update statistics: {e}")
    
    def _check_performance_alerts(self, breakdown: LatencyBreakdown):
        """Check for performance alerts and warnings"""
        try:
            # Check consecutive failures
            if self.consecutive_failures >= self.alert_thresholds["consecutive_failures"]:
                perf_logger.warning(
                    f"[EMOJI] PERFORMANCE ALERT: {self.consecutive_failures} consecutive operations "
                    f"exceeded target latency ({self.targets['total_end_to_end_ms']}ms)"
                )
            
            # Check for performance degradation
            if self.baseline_latency_ms is None and len(self.latency_history) >= 10:
                # Establish baseline from first 10 operations
                baseline_ops = list(self.latency_history)[:10]
                self.baseline_latency_ms = statistics.mean([op.total_latency_ms for op in baseline_ops])
                perf_logger.info(f"[STATS] Baseline latency established: {self.baseline_latency_ms:.1f}ms")
            
            if self.baseline_latency_ms and breakdown.total_latency_ms > 0:
                degradation_ratio = breakdown.total_latency_ms / self.baseline_latency_ms
                if degradation_ratio >= self.alert_thresholds["degradation_threshold"]:
                    perf_logger.warning(
                        f"[EMOJI] PERFORMANCE DEGRADATION: Current latency ({breakdown.total_latency_ms:.1f}ms) "
                        f"is {degradation_ratio:.1f}x baseline ({self.baseline_latency_ms:.1f}ms)"
                    )
            
            # Check individual component targets
            if breakdown.voxtral_processing_ms > self.targets["voxtral_processing_ms"]:
                perf_logger.warning(
                    f"[WARN] Voxtral processing exceeded target: {breakdown.voxtral_processing_ms:.1f}ms "
                    f"(target: {self.targets['voxtral_processing_ms']}ms)"
                )
            
            if breakdown.kokoro_generation_ms > self.targets["kokoro_generation_ms"]:
                perf_logger.warning(
                    f"[WARN] Kokoro generation exceeded target: {breakdown.kokoro_generation_ms:.1f}ms "
                    f"(target: {self.targets['kokoro_generation_ms']}ms)"
                )
            
        except Exception as e:
            perf_logger.error(f"[ERROR] Failed to check performance alerts: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            with self.lock:
                summary = {
                    "statistics": self.stats.copy(),
                    "targets": self.targets.copy(),
                    "recent_performance": self.check_performance_targets(),
                    "active_timings": len(self.active_timings),
                    "completed_operations": len(self.completed_operations),
                    "latency_history_size": len(self.latency_history),
                    "consecutive_failures": self.consecutive_failures,
                    "baseline_latency_ms": self.baseline_latency_ms
                }
                
                # Add recent latency trend
                if len(self.latency_history) >= 5:
                    recent_latencies = [op.total_latency_ms for op in list(self.latency_history)[-5:]]
                    summary["recent_latency_trend"] = {
                        "last_5_operations_ms": recent_latencies,
                        "trend_average_ms": statistics.mean(recent_latencies),
                        "trend_improving": len(recent_latencies) >= 2 and recent_latencies[-1] < recent_latencies[0]
                    }
                
                return summary
                
        except Exception as e:
            perf_logger.error(f"[ERROR] Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations based on current metrics
        """
        recommendations = []
        
        try:
            if not self.latency_history:
                return ["No performance data available yet"]
            
            recent_ops = list(self.latency_history)[-10:]
            
            # Analyze bottlenecks
            avg_voxtral = statistics.mean([op.voxtral_processing_ms for op in recent_ops])
            avg_kokoro = statistics.mean([op.kokoro_generation_ms for op in recent_ops])
            avg_conversion = statistics.mean([op.audio_conversion_ms for op in recent_ops])

            # Voxtral optimization
            if avg_voxtral > self.targets["voxtral_processing_ms"]:
                recommendations.append(
                    f"[VAD] Voxtral processing is slow ({avg_voxtral:.1f}ms avg). "
                    "Consider: reducing audio chunk size, optimizing VAD settings, or using faster GPU."
                )

            # Kokoro optimization
            if avg_kokoro > self.targets["kokoro_generation_ms"]:
                recommendations.append(
                    f"[AUDIO] Kokoro generation is slow ({avg_kokoro:.1f}ms avg). "
                    "Consider: reducing voice complexity, optimizing speed settings, or using faster GPU."
                )
            
            # Audio conversion optimization
            if avg_conversion > self.targets["audio_conversion_ms"]:
                recommendations.append(
                    f"[CONFIG] Audio conversion is slow ({avg_conversion:.1f}ms avg). "
                    "Consider: optimizing SNAC processing, using GPU acceleration, or reducing audio quality."
                )
            
            # Overall performance
            success_rate = self.stats["operations_within_target"] / max(self.stats["total_operations"], 1)
            if success_rate < self.alert_thresholds["success_rate_threshold"]:
                recommendations.append(
                    f"[STATS] Overall success rate is low ({success_rate:.1%}). "
                    "Consider: increasing performance targets, optimizing hardware, or reducing workload."
                )
            
            # Memory optimization
            if self.consecutive_failures > 0:
                recommendations.append(
                    "[BRAIN] Recent performance issues detected. "
                    "Consider: clearing GPU memory, restarting models, or checking system resources."
                )
            
            if not recommendations:
                recommendations.append("[OK] Performance is within targets. No optimizations needed.")
            
            return recommendations
            
        except Exception as e:
            perf_logger.error(f"[ERROR] Failed to generate recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]
    
    def reset_statistics(self):
        """Reset all performance statistics"""
        with self.lock:
            self.completed_operations.clear()
            self.latency_history.clear()
            self.active_timings.clear()
            
            self.stats = {
                "total_operations": 0,
                "operations_within_target": 0,
                "operations_exceeded_target": 0,
                "average_latency_ms": 0.0,
                "min_latency_ms": float('inf'),
                "max_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0
            }
            
            self.consecutive_failures = 0
            self.baseline_latency_ms = None
        
        perf_logger.info("[STATS] Performance statistics reset")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()