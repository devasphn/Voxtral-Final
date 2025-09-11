"""
Logging configuration for Voxtral Real-time Streaming
Provides centralized logging setup with structured formatting
"""
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging(log_level="INFO", log_file=None, use_json=False):
    """Setup centralized logging configuration"""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if use_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if use_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Default logger setup
logger = setup_logging(
    log_level="INFO",
    log_file="./logs/voxtral_streaming.log",
    use_json=False
)

# Create specialized loggers
audio_logger = logging.getLogger("voxtral.audio")
streaming_logger = logging.getLogger("voxtral.streaming")
model_logger = logging.getLogger("voxtral.model")
api_logger = logging.getLogger("voxtral.api")
health_logger = logging.getLogger("voxtral.health")

# Export main logger
__all__ = ["logger", "setup_logging", "audio_logger", "streaming_logger", "model_logger", "api_logger", "health_logger"]
