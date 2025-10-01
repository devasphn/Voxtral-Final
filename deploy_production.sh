#!/bin/bash

# 🚀 PRODUCTION DEPLOYMENT SCRIPT FOR RUNPOD
# Voxtral + Kokoro TTS Ultra-Low Latency Voice Agent
# Optimized for RunPod infrastructure with <500ms end-to-end latency
#
# PREREQUISITES:
# - Must be run from the /workspace/Voxtral-Final directory
# - Repository must already be cloned and current directory set
# - User must have sudo/root access for system package installation

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory first (before defining LOG_FILE)
mkdir -p "./logs"

# Global variables
LOG_FILE="./logs/deployment.log"

# Logging functions
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
    echo -e "${BLUE}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

error() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

success() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1"
    echo -e "${GREEN}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1
    error "Script failed at line $line_number with exit code $exit_code"
    error "Deployment failed. Check logs at: $LOG_FILE"
    exit $exit_code
}

# Set error trap
trap 'handle_error $LINENO' ERR

# Step 1: Create Python virtual environment
create_virtual_environment() {
    log "🐍 Step 1: Creating Python virtual environment..."
    python3 -m venv venv
    success "Virtual environment created successfully"
}

# Step 2: Activate virtual environment
activate_virtual_environment() {
    log "⚡ Step 2: Activating virtual environment..."
    source venv/bin/activate
    success "Virtual environment activated"
}

# Step 3: Install system dependencies
install_system_packages() {
    log "📦 Step 3: Installing system dependencies..."
    
    # Update package lists
    apt update
    
    # Install essential build tools
    apt install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        tree \
        htop
    
    # Install Python development packages
    apt install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel
    
    # Install audio system packages
    apt install -y \
        libasound2-dev \
        portaudio19-dev \
        libsndfile1-dev \
        libfftw3-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        ffmpeg \
        pulseaudio \
        alsa-utils
    
    # Install ML/AI libraries
    apt install -y \
        libopenblas-dev \
        liblapack-dev \
        libhdf5-dev \
        pkg-config
    
    success "System dependencies installed successfully"
}

# Step 4: Install Python packages
install_python_packages() {
    log "🐍 Step 4: Installing Python packages from requirements.txt..."
    
    # Upgrade pip first
    python -m pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support
    pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    
    # Install remaining requirements
    pip install -r requirements.txt
    
    success "Python packages installed successfully"
}

# Step 5: Pre-download models
predownload_models() {
    log "📥 Step 5: Pre-downloading models..."
    
    # Check if HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        error "HF_TOKEN environment variable is not set!"
        echo "Please set your HuggingFace token:"
        echo "export HF_TOKEN='hf_token'"
        exit 1
    fi
    
    # Create model cache directory
    mkdir -p model_cache
    
    # Download Voxtral model
    log "Downloading Voxtral model..."
    python3 -c "
import os
os.environ['HF_TOKEN'] = '$HF_TOKEN'
try:
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    print('Downloading Voxtral model...')
    processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507', cache_dir='./model_cache')
    model = VoxtralForConditionalGeneration.from_pretrained('mistralai/Voxtral-Mini-3B-2507', cache_dir='./model_cache', torch_dtype='auto', device_map='auto')
    print('✅ Voxtral model cached successfully')
except Exception as e:
    print(f'❌ Voxtral download failed: {e}')
    exit(1)
"
    
    # Download Kokoro TTS model
    log "Downloading Kokoro TTS model..."
    python3 -c "
try:
    import kokoro
    print('✅ Kokoro TTS package available')
except Exception as e:
    print(f'❌ Kokoro TTS import failed: {e}')
    exit(1)
"
    
    success "Models pre-downloaded successfully"
}

# Step 6: Run health checks
run_health_checks() {
    log "🏥 Step 6: Running health checks..."
    
    # Check Python environment
    log "Checking Python environment..."
    python3 -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')
"
    
    # Check critical imports
    log "Checking critical package imports..."
    python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    exit(1)

try:
    import transformers
    print(f'✅ Transformers {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
    exit(1)

try:
    import fastapi
    print(f'✅ FastAPI {fastapi.__version__}')
except ImportError as e:
    print(f'❌ FastAPI import failed: {e}')
    exit(1)

try:
    import kokoro
    print(f'✅ Kokoro TTS available')
except ImportError as e:
    print(f'❌ Kokoro TTS import failed: {e}')
    exit(1)
"
    
    # Test configuration loading
    log "Testing configuration system..."
    python3 -c "
try:
    from src.utils.config import config
    print('✅ Configuration loaded successfully')
    print(f'   Server HTTP port: {config.server.http_port}')
    print(f'   Model name: {config.model.name}')
    print(f'   TTS engine: {config.tts.engine}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"
    
    success "Health checks completed successfully"
}

# Main deployment function
main() {
    log "🚀 Starting PRODUCTION deployment of Voxtral + Kokoro TTS system"
    log "📝 Deployment logs: $LOG_FILE"
    log "📂 Current directory: $(pwd)"
    
    # Verify we're in the correct directory
    if [[ ! -f "requirements.txt" || ! -d "src" ]]; then
        error "❌ Must be run from the Voxtral-Final project directory!"
        error "   Current directory: $(pwd)"
        error "   Expected files: requirements.txt, src/ directory"
        exit 1
    fi
    
    success "✅ Verified project directory structure"
    
    # Execute deployment steps in order
    create_virtual_environment
    activate_virtual_environment
    install_system_packages
    install_python_packages
    predownload_models
    run_health_checks
    
    success "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 System Information:"
    echo "   🌐 UI: http://localhost:8000"
    echo "   🔌 WebSocket: ws://localhost:8765"
    echo "   ❤️ Health: http://localhost:8000/health"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Start the system: python3 -m src.api.ui_server_realtime"
    echo "   2. Access the UI at http://localhost:8000"
    echo "   3. Monitor GPU usage: watch -n 1 nvidia-smi"
    echo ""
    echo "📋 Logs available at: $LOG_FILE"
}

# Run main function
main "$@"
