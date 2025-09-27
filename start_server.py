#!/usr/bin/env python3
"""
PRODUCTION: Voxtral Real-time Streaming Server Startup Script
Ensures proper model pre-loading and server initialization for RunPod deployment
"""

import asyncio
import logging
import sys
import os
import time
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/voxtral_streaming.log', mode='a')
    ]
)

logger = logging.getLogger("server_startup")

class VoxtralServerManager:
    """Manages the complete Voxtral server lifecycle"""
    
    def __init__(self):
        self.server_process = None
        self.health_server_process = None
        self.models_initialized = False
        self.shutdown_requested = False
        
    async def initialize_models(self):
        """Pre-load all models before starting the server"""
        logger.info("🚀 Starting Voxtral Real-time Streaming Server...")
        logger.info("📋 Phase 1: Model Initialization")
        
        try:
            # Import and initialize models
            from src.api.ui_server_realtime import initialize_models_at_startup
            
            start_time = time.time()
            await initialize_models_at_startup()
            init_time = time.time() - start_time
            
            logger.info(f"✅ Models initialized successfully in {init_time:.2f}s")
            self.models_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def start_health_server(self):
        """Start the health check server"""
        try:
            logger.info("📋 Phase 2: Starting Health Check Server...")
            
            import uvicorn
            from src.api.health_check import app as health_app
            from src.utils.config import config
            
            # Start health server in background
            health_config = uvicorn.Config(
                health_app,
                host=config.server.host,
                port=config.server.health_port,
                log_level="info",
                access_log=False
            )
            
            health_server = uvicorn.Server(health_config)
            self.health_server_process = asyncio.create_task(health_server.serve())
            
            logger.info(f"✅ Health check server started on port {config.server.health_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health server startup failed: {e}")
            return False
    
    async def start_main_server(self):
        """Start the main WebSocket server"""
        try:
            logger.info("📋 Phase 3: Starting Main WebSocket Server...")
            
            import uvicorn
            from src.api.ui_server_realtime import app
            from src.utils.config import config
            
            # Start main server
            main_config = uvicorn.Config(
                app,
                host=config.server.host,
                port=config.server.http_port,
                log_level="info",
                access_log=True,
                ws_ping_interval=20,
                ws_ping_timeout=10
            )
            
            main_server = uvicorn.Server(main_config)
            self.server_process = asyncio.create_task(main_server.serve())
            
            logger.info(f"✅ Main server started on port {config.server.http_port}")
            logger.info("🎉 Voxtral Real-time Streaming Server is ready!")
            logger.info(f"🌐 WebSocket endpoint: ws://0.0.0.0:{config.server.http_port}/ws")
            logger.info(f"🏥 Health check: http://0.0.0.0:{config.server.health_port}/health")
            logger.info(f"📊 Status endpoint: http://0.0.0.0:{config.server.health_port}/status")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Main server startup failed: {e}")
            return False
    
    async def run(self):
        """Run the complete server startup sequence"""
        try:
            # Phase 1: Initialize models
            if not await self.initialize_models():
                logger.error("❌ Server startup failed - model initialization error")
                return False
            
            # Phase 2: Start health server
            if not await self.start_health_server():
                logger.error("❌ Server startup failed - health server error")
                return False
            
            # Small delay to ensure health server is ready
            await asyncio.sleep(1)
            
            # Phase 3: Start main server
            if not await self.start_main_server():
                logger.error("❌ Server startup failed - main server error")
                return False
            
            # Wait for both servers
            await asyncio.gather(
                self.server_process,
                self.health_server_process,
                return_exceptions=True
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
            await self.shutdown()
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        logger.info("🛑 Shutting down servers...")
        
        if self.server_process:
            self.server_process.cancel()
        if self.health_server_process:
            self.health_server_process.cancel()
        
        logger.info("✅ Shutdown complete")

def setup_signal_handlers(server_manager):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(server_manager.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Create server manager
    server_manager = VoxtralServerManager()
    
    # Setup signal handlers
    setup_signal_handlers(server_manager)
    
    # Run the server
    await server_manager.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)
