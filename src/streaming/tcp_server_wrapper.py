#!/usr/bin/env python3
"""
Wrapper script to ensure TCP server starts correctly with proper signal handling
"""
import sys
import signal
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.streaming.tcp_server import TCPStreamingServer
from src.utils.logging_config import logger

# Setup signal handlers
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    for task in asyncio.all_tasks():
        task.cancel()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def run_tcp_server():
    """Run TCP server with proper initialization and error handling"""
    server = TCPStreamingServer()
    
    try:
        logger.info("🚀 TCP Server Wrapper: Starting server...")
        await server.start_server()
    except Exception as e:
        logger.error(f"❌ TCP Server Wrapper: Server failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    """Main entry point"""
    try:
        # Configure asyncio for production
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        logger.info("🔧 TCP Server Wrapper: Event loop created")
        
        # Run the server
        loop.run_until_complete(run_tcp_server())
        
    except KeyboardInterrupt:
        logger.info("⚠️ TCP Server Wrapper: Interrupted by user")
    except Exception as e:
        logger.error(f"❌ TCP Server Wrapper: Fatal error: {e}")
        sys.exit(1)
    finally:
        loop.close()
        logger.info("✅ TCP Server Wrapper: Event loop closed")

if __name__ == "__main__":
    main()
