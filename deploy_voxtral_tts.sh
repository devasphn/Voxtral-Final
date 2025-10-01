#!/bin/bash

# Voxtral + Kokoro TTS Deployment Script for RunPod
# Ultra-Low Latency Voice AI System (<500ms end-to-end)
# Optimized for RunPod HTTP/TCP-only infrastructure

set -e  # Exit on any error

echo "🚀 Starting Voxtral + Kokoro TTS Deployment on RunPod"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    print_warning "Not running on RunPod. Creating workspace directory..."
    mkdir -p /workspace
fi

cd /workspace

# Step 1: System Dependencies
print_status "Installing system dependencies..."
apt update && apt upgrade -y
apt install -y git curl wget build-essential python3-dev python3-pip ffmpeg libsndfile1 portaudio19-dev

# Step 2: Verify GPU
print_status "Verifying GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_success "GPU detected and accessible"
else
    print_error "No GPU detected. This application requires CUDA-compatible GPU."
    exit 1
fi

# Step 3: Environment Variables
print_status "Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    print_warning "HF_TOKEN not set. Some models may require authentication."
    print_warning "Set HF_TOKEN environment variable if needed."
fi

# Step 4: Create directories
print_status "Creating required directories..."
mkdir -p model_cache logs temp_audio
chmod 755 model_cache logs temp_audio

# Step 5: Python Dependencies
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

print_status "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

print_status "Installing project dependencies..."
pip install -r requirements.txt

# Step 6: Verify Installation
print_status "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('No CUDA device detected')
"

# Step 7: Model Pre-loading (Optional but recommended)
print_status "Pre-loading models (this may take several minutes)..."
python3 -c "
import os
if os.environ.get('HF_TOKEN'):
    os.environ['HUGGINGFACE_HUB_TOKEN'] = os.environ['HF_TOKEN']

try:
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    print('Downloading Voxtral model...')
    model = VoxtralForConditionalGeneration.from_pretrained(
        'mistralai/Voxtral-Mini-3B-2507',
        cache_dir='./model_cache',
        torch_dtype='float16',
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(
        'mistralai/Voxtral-Mini-3B-2507',
        cache_dir='./model_cache'
    )
    print('Voxtral model downloaded successfully')
except Exception as e:
    print(f'Warning: Could not pre-load Voxtral model: {e}')

try:
    import kokoro
    print('Kokoro TTS is available')
except Exception as e:
    print(f'Warning: Kokoro TTS not available: {e}')
"

# Step 8: Create startup script
print_status "Creating startup script..."
cat > start_voxtral.sh << 'EOF'
#!/bin/bash

# Voxtral + Kokoro TTS Startup Script
echo "🎤 Starting Voxtral + Kokoro TTS Voice AI System"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start services in background
echo "Starting Health Check Server (port 8005)..."
python3 -m src.api.health_check &
HEALTH_PID=$!

echo "Starting Main UI Server (port 8000)..."
python3 -m src.api.ui_server_realtime &
UI_PID=$!

echo "Starting TCP Server (port 8766)..."
python3 -m src.streaming.tcp_server &
TCP_PID=$!

# Wait a moment for services to start
sleep 5

echo "✅ All services started!"
echo "🌐 Web Interface: https://[POD_ID]-8000.proxy.runpod.net"
echo "🏥 Health Check: https://[POD_ID]-8005.proxy.runpod.net/health"
echo "🔌 TCP Server: Port 8766"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo "Stopping services..."
    kill $HEALTH_PID $UI_PID $TCP_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for any process to exit
wait
EOF

chmod +x start_voxtral.sh

# Step 9: Create health check script
print_status "Creating health check script..."
cat > health_check.sh << 'EOF'
#!/bin/bash

# Quick health check for Voxtral system
echo "🏥 Voxtral System Health Check"
echo "=============================="

# Check if services are running
if pgrep -f "src.api.ui_server_realtime" > /dev/null; then
    echo "✅ UI Server: Running"
else
    echo "❌ UI Server: Not running"
fi

if pgrep -f "src.api.health_check" > /dev/null; then
    echo "✅ Health Check Server: Running"
else
    echo "❌ Health Check Server: Not running"
fi

if pgrep -f "src.streaming.tcp_server" > /dev/null; then
    echo "✅ TCP Server: Running"
else
    echo "❌ TCP Server: Not running"
fi

# Check GPU
echo ""
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits

# Check ports
echo ""
echo "🔌 Port Status:"
netstat -tlnp | grep -E ":(8000|8005|8766)" || echo "No services listening on expected ports"
EOF

chmod +x health_check.sh

# Step 10: Final setup
print_status "Final setup and verification..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Set proper permissions
chmod -R 755 src/
chmod +x *.sh

print_success "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Start the system: ./start_voxtral.sh"
echo "2. Check health: ./health_check.sh"
echo "3. Access web interface: https://[POD_ID]-8000.proxy.runpod.net"
echo ""
echo "🔧 Manual Commands:"
echo "- Start UI: python3 -m src.api.ui_server_realtime"
echo "- Start Health: python3 -m src.api.health_check"
echo "- Start TCP: python3 -m src.streaming.tcp_server"
echo ""
print_success "Ready for ultra-low latency voice conversations! 🎤✨"
