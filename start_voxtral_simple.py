#!/usr/bin/env python3
"""
Simple Voxtral Startup Script
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voxtral-startup")

def main():
    """Main startup function"""
    logger.info("🚀 Starting Voxtral Voice AI System")
    
    # Set environment variables
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'TOKENIZERS_PARALLELISM': 'false',
    })
    
    # Add src to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        # Import and start the application
        from api.ui_server_realtime import app
        logger.info("✅ Application imported successfully")
        
        # Start with optimized settings
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,
            limit_concurrency=100,
            timeout_keep_alive=30,
            access_log=True,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("📝 Please check if all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()