"""
FIXED Health check API for monitoring server status
Resolved deprecation warnings and improved error handling
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import time
import psutil
import torch
from typing import Dict, Any
import logging
import sys
import os

# Add current directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import config

# Create separate logger for health check
health_logger = logging.getLogger("health_check")

# Global variable to track model status (avoid circular imports)
_model_status = {"initialized": False, "info": {}}

def update_model_status(status: Dict[str, Any]):
    """Update model status from external sources"""
    global _model_status
    _model_status = status

# FIXED: Use lifespan event handler instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    health_logger.info("Health check server starting...")
    yield
    # Shutdown
    health_logger.info("Health check server shutting down...")

# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Voxtral Streaming Health Check", 
    version="1.0.0",
    docs_url=None,  # Disable docs to avoid conflicts
    redoc_url=None,  # Disable redoc to avoid conflicts
    lifespan=lifespan
)

@app.get("/health")
async def health_check() -> JSONResponse:
    """Basic health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "service": "voxtral-streaming",
        "version": "2.0.0"
    })

@app.get("/status")
async def detailed_status() -> JSONResponse:
    """Detailed system status"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Faster check
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_info = {}
        try:
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.device_count() > 0 else 0,
                    "gpu_memory_cached": torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.device_count() > 0 else 0
                }
            else:
                gpu_info = {"gpu_available": False}
        except Exception as e:
            gpu_info = {"gpu_available": False, "error": str(e)}
        
        # Get live model status from UI server
        model_info = {"status": "unknown"}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/api/status", timeout=2) as response:
                    if response.status == 200:
                        ui_status = await response.json()
                        model_info = ui_status.get("model", {"status": "unknown"})
        except Exception:
            # Fallback to global variable if UI server unavailable
            model_info = _model_status.get("info", {"status": "unknown"})
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1e9, 2),
                "disk_free_gb": round(disk.free / 1e9, 2)
            },
            "gpu": gpu_info,
            "model": model_info,
            "config": {
                "http_port": config.server.http_port,
                "health_port": config.server.health_port,
                "tcp_ports": config.server.tcp_ports,
                "sample_rate": config.audio.sample_rate,
                "latency_target": config.streaming.latency_target_ms
            }
        })
        
    except Exception as e:
        health_logger.error(f"Error in detailed status: {e}")
        return JSONResponse({
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }, status_code=500)

@app.get("/ready")
async def readiness_check() -> JSONResponse:
    """Readiness probe for model initialization"""
    try:
        model_initialized = _model_status.get("initialized", False)
        
        if model_initialized:
            return JSONResponse({
                "ready": True,
                "timestamp": time.time(),
                "model_status": "initialized"
            })
        else:
            return JSONResponse({
                "ready": False,
                "timestamp": time.time(), 
                "model_status": "not_initialized"
            }, status_code=503)
    except Exception as e:
        health_logger.error(f"Error in readiness check: {e}")
        return JSONResponse({
            "ready": False,
            "timestamp": time.time(),
            "error": str(e)
        }, status_code=500)

@app.get("/ping")
async def ping() -> JSONResponse:
    """Simple ping endpoint"""
    return JSONResponse({
        "pong": True,
        "timestamp": time.time()
    })

# FIXED: Add proper main execution block
if __name__ == "__main__":
    try:
        health_logger.info("Starting Health Check Server")
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.health_port,
            log_level="info",
            access_log=False  # Reduce log noise
        )
    except Exception as e:
        health_logger.error(f"Failed to start health check server: {e}")
        raise
