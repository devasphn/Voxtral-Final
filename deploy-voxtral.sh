#!/bin/bash
set -e

echo "🚀 Voxtral Voice AI - Production RunPod Deployment"
echo "=================================================="

# Environment Configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/transformers_cache
export TOKENIZERS_PARALLELISM=false

echo "✅ Environment variables configured"

# System Dependencies Installation
echo "📦 Installing system dependencies..."
apt-get update -qq && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    git \
    wget \
    curl \
    sox \
    libsox-fmt-all \
    htop \
    nvtop

echo "✅ System dependencies installed"

# PyTorch Installation (Critical First)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Core ML Dependencies
echo "🧠 Installing core ML dependencies..."
pip install transformers==4.56.0 \
            huggingface-hub==0.34.0 \
            accelerate==0.25.0 \
            tokenizers==0.22.1 \
            safetensors==0.6.2

# Audio Processing
echo "🎵 Installing audio processing libraries..."
pip install librosa==0.10.1 \
            soundfile==0.13.1 \
            'numpy>=1.24.4,<2.0.0' \
            scipy==1.11.4 \
            pydub==0.25.1

# Mistral Common for Voxtral
echo "🗣️ Installing Mistral Common for Voxtral..."
pip install mistral-common[audio]==1.8.1

# Kokoro TTS
echo "🎤 Installing Kokoro TTS..."
pip install kokoro==0.9.4

# Web Framework
echo "🌐 Installing web framework..."
pip install fastapi==0.115.0 \
            'uvicorn[standard]==0.37.0' \
            websockets==15.0.1 \
            pydantic==2.11.9 \
            pydantic-settings==2.1.0 \
            python-multipart==0.0.6 \
            aiofiles==23.2.1

# Utilities
echo "🔧 Installing utilities..."
pip install pyyaml==6.0.3 \
            python-dotenv==1.1.1 \
            psutil==5.9.6 \
            httpx==0.25.0

# Create Directory Structure
echo "📁 Creating directory structure..."
mkdir -p logs temp_audio model_cache audio_buffer

# Fix Sample Rate Configuration
echo "🎛️ Fixing audio sample rate configuration..."
cat > sample_rate_fix.py << 'EOF'
import os
import sys

# Add src directory to path
sys.path.insert(0, '/workspace/Voxtral-Final/src')

try:
    from utils.config import TTSConfig
    print("✅ Found TTS config, updating sample rate to 24000Hz for Kokoro compatibility")
    
    # Create fixed config file
    config_content = '''
class TTSConfig:
    sample_rate: int = 24000  # Kokoro native rate
    chunk_size: int = 1024
    buffer_size: int = 4096
    format: str = "pcm16"
'''
    
    with open('/workspace/Voxtral-Final/src/utils/config.py', 'r') as f:
        original_content = f.read()
    
    # Backup original
    with open('/workspace/Voxtral-Final/src/utils/config.py.bak', 'w') as f:
        f.write(original_content)
    
    # Update sample rate in config
    updated_content = original_content.replace(
        'sample_rate: int = 16000', 
        'sample_rate: int = 24000  # Updated for Kokoro TTS compatibility'
    )
    
    with open('/workspace/Voxtral-Final/src/utils/config.py', 'w') as f:
        f.write(updated_content)
        
    print("✅ Sample rate configuration updated successfully")
    
except Exception as e:
    print(f"⚠️ Could not update config automatically: {e}")
    print("📝 Manual fix required: Update sample_rate to 24000 in src/utils/config.py")
EOF

python sample_rate_fix.py

# Memory Management Optimization
echo "🧠 Optimizing memory management..."
cat > memory_optimization.py << 'EOF'
import os
import gc
import torch

# Optimize CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)
    print(f"✅ CUDA optimized - Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ CUDA not available")

# Set memory management environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("✅ Memory optimization completed")
EOF

python memory_optimization.py

# Create Improved Startup Script
echo "🚀 Creating optimized startup script..."
cat > start_voxtral.py << 'EOF'
#!/usr/bin/env python3
"""
Voxtral Voice AI - Optimized Startup Script
RunPod Production Version
"""
import os
import sys
import asyncio
import logging
import signal
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voxtral-startup")

def setup_environment():
    """Setup optimized environment"""
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'HF_HOME': '/workspace/huggingface_cache',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONPATH': '/workspace/Voxtral-Final/src',
    })
    logger.info("✅ Environment configured")

def cleanup_handler(signum, frame):
    """Graceful shutdown handler"""
    logger.info("🛑 Shutdown signal received, cleaning up...")
    # Add cleanup logic here
    sys.exit(0)

def main():
    """Main startup function"""
    logger.info("🚀 Starting Voxtral Voice AI System")
    
    # Setup environment
    setup_environment()
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Import and start the application
    try:
        sys.path.insert(0, '/workspace/Voxtral-Final/src')
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
EOF

chmod +x start_voxtral.py

# Create Health Check Script
echo "🏥 Creating health check script..."
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
import requests
import json
import sys
import time

def check_health():
    """Check if Voxtral service is healthy"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Voxtral service is healthy")
            return True
        else:
            print(f"⚠️ Service returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Running health check...")
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
EOF

chmod +x health_check.py

# Performance Test Script
echo "⚡ Creating performance test script..."
cat > performance_test.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import websockets
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perf-test")

async def test_websocket_connection():
    """Test WebSocket connection performance"""
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✅ WebSocket connected successfully")
            
            # Send test message
            test_message = {
                "type": "test",
                "data": "performance test"
            }
            
            start_time = time.time()
            await websocket.send(json.dumps(test_message))
            response = await websocket.recv()
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            logger.info(f"✅ WebSocket latency: {latency:.2f}ms")
            
            return latency < 100  # Success if under 100ms
            
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}")
        return False

async def main():
    logger.info("⚡ Starting performance tests...")
    
    # Test WebSocket
    ws_success = await test_websocket_connection()
    
    if ws_success:
        logger.info("✅ All performance tests passed")
    else:
        logger.error("❌ Performance tests failed")

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x performance_test.py

echo "✅ Deployment completed successfully!"
echo ""
echo "🚀 To start Voxtral:"
echo "   python start_voxtral.py"
echo ""
echo "🏥 To check health:"
echo "   python health_check.py"
echo ""
echo "⚡ To run performance tests:"
echo "   python performance_test.py"
echo ""
echo "📊 To monitor resources:"
echo "   htop  # CPU monitoring"
echo "   nvtop # GPU monitoring"
echo ""
echo "🌐 Access your application:"
echo "   https://your-runpod-id-8000.proxy.runpod.net"