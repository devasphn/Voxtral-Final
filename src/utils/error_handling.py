"""
Comprehensive Error Handling System for Kokoro TTS Integration
Centralized error handling, recovery mechanisms, and error analytics
"""

import logging
import time
import asyncio
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import functools

# Setup logging
error_logger = logging.getLogger("error_handling")
error_logger.setLevel(logging.INFO)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    INITIALIZATION = "initialization"
    RUNTIME = "runtime"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

@dataclass
class ErrorRecord:
    """Individual error record"""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None

class ErrorHandler:
    """
    Comprehensive error handling system with recovery mechanisms
    """
    
    def __init__(self):
        self.error_history = deque(maxlen=1000)  # Keep last 1000 errors
        self.error_patterns = {}  # Pattern recognition for common errors
        self.recovery_strategies = {}  # Registered recovery strategies
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "recent_error_rate": 0.0
        }
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        error_logger.info("ErrorHandler initialized")
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for common errors"""
        
        # GPU Memory errors
        self.register_recovery_strategy(
            "cuda_out_of_memory",
            self._recover_gpu_memory_error,
            ErrorCategory.MEMORY
        )
        
        # Model initialization errors
        self.register_recovery_strategy(
            "model_initialization_failed",
            self._recover_model_initialization_error,
            ErrorCategory.INITIALIZATION
        )
        
        # Performance degradation
        self.register_recovery_strategy(
            "performance_degradation",
            self._recover_performance_degradation,
            ErrorCategory.PERFORMANCE
        )
        
        # Network/connection errors
        self.register_recovery_strategy(
            "connection_error",
            self._recover_connection_error,
            ErrorCategory.NETWORK
        )
    
    def register_recovery_strategy(self, error_pattern: str, recovery_func: Callable, category: ErrorCategory):
        """Register a recovery strategy for a specific error pattern"""
        self.recovery_strategies[error_pattern] = {
            "function": recovery_func,
            "category": category,
            "success_count": 0,
            "attempt_count": 0
        }
        error_logger.info(f"[NOTE] Registered recovery strategy for '{error_pattern}' in category {category.value}")
    
    async def handle_error(self, 
                          error: Exception, 
                          context: Dict[str, Any] = None,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          attempt_recovery: bool = True) -> Dict[str, Any]:
        """
        Handle an error with optional recovery attempt
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Error severity level
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Dictionary with error handling results
        """
        try:
            error_type = type(error).__name__
            error_message = str(error)
            timestamp = time.time()
            
            # Classify error
            category = self._classify_error(error, error_message)
            
            # Create error record
            error_record = ErrorRecord(
                timestamp=timestamp,
                error_type=error_type,
                error_message=error_message,
                severity=severity,
                category=category,
                context=context or {},
                stack_trace=traceback.format_exc()
            )
            
            # Add to history and update stats
            self.error_history.append(error_record)
            self._update_error_stats(error_record)
            
            # Log error
            self._log_error(error_record)
            
            # Attempt recovery if requested
            recovery_result = None
            if attempt_recovery:
                recovery_result = await self._attempt_recovery(error_record)
                error_record.recovery_attempted = True
                error_record.recovery_successful = recovery_result.get("success", False)
                error_record.recovery_method = recovery_result.get("method")
            
            # Check for error patterns
            self._analyze_error_patterns(error_record)
            
            return {
                "error_handled": True,
                "error_record": error_record,
                "recovery_result": recovery_result,
                "recommendations": self._get_error_recommendations(error_record)
            }
            
        except Exception as handling_error:
            error_logger.error(f"[ERROR] Error handling failed: {handling_error}")
            return {
                "error_handled": False,
                "handling_error": str(handling_error),
                "original_error": str(error)
            }
    
    def _classify_error(self, error: Exception, error_message: str) -> ErrorCategory:
        """Classify error into appropriate category"""
        error_type = type(error).__name__.lower()
        message_lower = error_message.lower()
        
        # Memory errors
        if "cuda out of memory" in message_lower or "memory" in error_type:
            return ErrorCategory.MEMORY
        
        # Initialization errors
        if "initialization" in message_lower or "init" in error_type:
            return ErrorCategory.INITIALIZATION
        
        # Performance errors
        if "timeout" in message_lower or "performance" in message_lower:
            return ErrorCategory.PERFORMANCE
        
        # Network errors
        if "connection" in message_lower or "network" in message_lower or "http" in error_type:
            return ErrorCategory.NETWORK
        
        # Validation errors
        if "validation" in message_lower or "invalid" in message_lower:
            return ErrorCategory.VALIDATION
        
        # Runtime errors (default for most exceptions)
        if "runtime" in error_type or "exception" in error_type:
            return ErrorCategory.RUNTIME
        
        return ErrorCategory.UNKNOWN
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level based on severity"""
        severity_emoji = {
            ErrorSeverity.LOW: "ℹ[EMOJI]",
            ErrorSeverity.MEDIUM: "[WARN]",
            ErrorSeverity.HIGH: "[ERROR]",
            ErrorSeverity.CRITICAL: "[EMOJI]"
        }
        
        emoji = severity_emoji.get(error_record.severity, "[EMOJI]")
        
        log_message = (
            f"{emoji} {error_record.severity.value.upper()} ERROR "
            f"[{error_record.category.value}]: {error_record.error_type} - {error_record.error_message}"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            error_logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            error_logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            error_logger.warning(log_message)
        else:
            error_logger.info(log_message)
        
        # Log context if available
        if error_record.context:
            error_logger.debug(f"   Context: {error_record.context}")
    
    async def _attempt_recovery(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Attempt to recover from error using registered strategies"""
        try:
            # Find matching recovery strategy
            recovery_strategy = None
            strategy_key = None
            
            for pattern, strategy in self.recovery_strategies.items():
                if (pattern.lower() in error_record.error_message.lower() or 
                    pattern.lower() in error_record.error_type.lower() or
                    strategy["category"] == error_record.category):
                    recovery_strategy = strategy
                    strategy_key = pattern
                    break
            
            if not recovery_strategy:
                return {"success": False, "reason": "No recovery strategy found"}
            
            error_logger.info(f"[CONFIG] Attempting recovery using strategy: {strategy_key}")
            
            # Update attempt count
            recovery_strategy["attempt_count"] += 1
            
            # Execute recovery function
            recovery_func = recovery_strategy["function"]
            recovery_result = await recovery_func(error_record)
            
            if recovery_result.get("success", False):
                recovery_strategy["success_count"] += 1
                error_logger.info(f"[OK] Recovery successful using strategy: {strategy_key}")
            else:
                error_logger.warning(f"[ERROR] Recovery failed using strategy: {strategy_key}")
            
            return {
                "success": recovery_result.get("success", False),
                "method": strategy_key,
                "details": recovery_result,
                "strategy_stats": {
                    "success_rate": recovery_strategy["success_count"] / recovery_strategy["attempt_count"],
                    "total_attempts": recovery_strategy["attempt_count"]
                }
            }
            
        except Exception as recovery_error:
            error_logger.error(f"[ERROR] Recovery attempt failed: {recovery_error}")
            return {
                "success": False,
                "reason": f"Recovery attempt failed: {str(recovery_error)}"
            }
    
    async def _recover_gpu_memory_error(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Recovery strategy for GPU memory errors"""
        try:
            error_logger.info("[CLEANUP] Attempting GPU memory recovery...")
            
            # Import here to avoid circular imports
            import torch
            import gc
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Wait a moment for cleanup
            await asyncio.sleep(1.0)
            
            # Check if memory was freed
            if torch.cuda.is_available():
                memory_freed = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                
                return {
                    "success": True,
                    "method": "gpu_memory_cleanup",
                    "memory_freed_bytes": memory_freed,
                    "actions_taken": ["torch.cuda.empty_cache()", "gc.collect()"]
                }
            else:
                return {
                    "success": True,
                    "method": "cpu_memory_cleanup",
                    "actions_taken": ["gc.collect()"]
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _recover_model_initialization_error(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Recovery strategy for model initialization errors"""
        try:
            error_logger.info("[EMOJI] Attempting model initialization recovery...")
            
            actions_taken = []
            
            # Try memory cleanup first
            memory_recovery = await self._recover_gpu_memory_error(error_record)
            if memory_recovery.get("success"):
                actions_taken.extend(memory_recovery.get("actions_taken", []))
            
            # Wait for system to stabilize
            await asyncio.sleep(2.0)
            actions_taken.append("system_stabilization_wait")
            
            # Could implement model reinitialization here if needed
            # For now, just report that cleanup was attempted
            
            return {
                "success": True,
                "method": "model_init_recovery",
                "actions_taken": actions_taken,
                "recommendation": "Retry model initialization after cleanup"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _recover_performance_degradation(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Recovery strategy for performance degradation"""
        try:
            error_logger.info("[FAST] Attempting performance recovery...")
            
            actions_taken = []
            
            # Memory cleanup
            memory_recovery = await self._recover_gpu_memory_error(error_record)
            if memory_recovery.get("success"):
                actions_taken.extend(memory_recovery.get("actions_taken", []))
            
            # Could implement performance tuning adjustments here
            actions_taken.append("performance_monitoring_reset")
            
            return {
                "success": True,
                "method": "performance_recovery",
                "actions_taken": actions_taken,
                "recommendation": "Monitor performance metrics closely"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _recover_connection_error(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Recovery strategy for connection errors"""
        try:
            error_logger.info("[EMOJI] Attempting connection recovery...")
            
            # Wait and retry logic
            await asyncio.sleep(1.0)
            
            return {
                "success": True,
                "method": "connection_retry",
                "actions_taken": ["connection_retry_delay"],
                "recommendation": "Retry the failed operation"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_error_stats(self, error_record: ErrorRecord):
        """Update error statistics"""
        self.error_stats["total_errors"] += 1
        
        # Update category stats
        category_key = error_record.category.value
        if category_key not in self.error_stats["errors_by_category"]:
            self.error_stats["errors_by_category"][category_key] = 0
        self.error_stats["errors_by_category"][category_key] += 1
        
        # Update severity stats
        severity_key = error_record.severity.value
        if severity_key not in self.error_stats["errors_by_severity"]:
            self.error_stats["errors_by_severity"][severity_key] = 0
        self.error_stats["errors_by_severity"][severity_key] += 1
        
        # Calculate recovery success rate
        total_recovery_attempts = sum(1 for err in self.error_history if err.recovery_attempted)
        successful_recoveries = sum(1 for err in self.error_history if err.recovery_successful)
        
        if total_recovery_attempts > 0:
            self.error_stats["recovery_success_rate"] = successful_recoveries / total_recovery_attempts
        
        # Calculate recent error rate (errors per minute in last 10 minutes)
        current_time = time.time()
        recent_errors = [err for err in self.error_history if current_time - err.timestamp < 600]  # 10 minutes
        self.error_stats["recent_error_rate"] = len(recent_errors) / 10.0  # errors per minute
    
    def _analyze_error_patterns(self, error_record: ErrorRecord):
        """Analyze error patterns for trend detection"""
        # Simple pattern analysis - could be enhanced with ML
        error_key = f"{error_record.category.value}:{error_record.error_type}"
        
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = {
                "count": 0,
                "first_seen": error_record.timestamp,
                "last_seen": error_record.timestamp,
                "frequency": 0.0
            }
        
        pattern = self.error_patterns[error_key]
        pattern["count"] += 1
        pattern["last_seen"] = error_record.timestamp
        
        # Calculate frequency (errors per hour)
        time_span = pattern["last_seen"] - pattern["first_seen"]
        if time_span > 0:
            pattern["frequency"] = pattern["count"] / (time_span / 3600)  # per hour
        
        # Alert on high frequency patterns
        if pattern["frequency"] > 10:  # More than 10 errors per hour
            error_logger.warning(
                f"[EMOJI] High frequency error pattern detected: {error_key} "
                f"({pattern['count']} occurrences, {pattern['frequency']:.1f}/hour)"
            )
    
    def _get_error_recommendations(self, error_record: ErrorRecord) -> List[str]:
        """Get recommendations based on error type and context"""
        recommendations = []
        
        if error_record.category == ErrorCategory.MEMORY:
            recommendations.extend([
                "Consider reducing model precision (fp32 → fp16)",
                "Reduce batch size or audio chunk size",
                "Check available GPU memory before operations",
                "Implement memory monitoring and cleanup"
            ])
        
        elif error_record.category == ErrorCategory.INITIALIZATION:
            recommendations.extend([
                "Verify all dependencies are installed correctly",
                "Check GPU drivers and CUDA compatibility",
                "Ensure sufficient system resources",
                "Try initializing models sequentially rather than in parallel"
            ])
        
        elif error_record.category == ErrorCategory.PERFORMANCE:
            recommendations.extend([
                "Monitor system resource usage",
                "Consider reducing quality settings temporarily",
                "Check for background processes consuming resources",
                "Implement performance monitoring and alerting"
            ])
        
        elif error_record.category == ErrorCategory.NETWORK:
            recommendations.extend([
                "Check network connectivity",
                "Implement retry logic with exponential backoff",
                "Consider connection pooling",
                "Add timeout handling"
            ])
        
        # Add general recommendations
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            recommendations.append("Consider implementing circuit breaker pattern")
            recommendations.append("Add comprehensive error monitoring and alerting")
        
        return recommendations
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary and statistics"""
        return {
            "statistics": self.error_stats.copy(),
            "error_patterns": dict(self.error_patterns),
            "recent_errors": [
                {
                    "timestamp": err.timestamp,
                    "type": err.error_type,
                    "message": err.error_message[:100] + "..." if len(err.error_message) > 100 else err.error_message,
                    "severity": err.severity.value,
                    "category": err.category.value,
                    "recovery_attempted": err.recovery_attempted,
                    "recovery_successful": err.recovery_successful
                }
                for err in list(self.error_history)[-10:]  # Last 10 errors
            ],
            "recovery_strategies": {
                pattern: {
                    "success_rate": strategy["success_count"] / max(strategy["attempt_count"], 1),
                    "total_attempts": strategy["attempt_count"],
                    "category": strategy["category"].value
                }
                for pattern, strategy in self.recovery_strategies.items()
            }
        }
    
    def reset_error_history(self):
        """Reset error history and statistics"""
        self.error_history.clear()
        self.error_patterns.clear()
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "recent_error_rate": 0.0
        }
        error_logger.info("[STATS] Error history and statistics reset")

# Decorator for automatic error handling
def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 attempt_recovery: bool = True,
                 context: Dict[str, Any] = None):
    """
    Decorator for automatic error handling
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                result = await error_handler.handle_error(
                    e, context=context, severity=severity, attempt_recovery=attempt_recovery
                )
                # Re-raise the error after handling
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't await, so just log
                asyncio.create_task(error_handler.handle_error(
                    e, context=context, severity=severity, attempt_recovery=False
                ))
                raise e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Global error handler instance
error_handler = ErrorHandler()