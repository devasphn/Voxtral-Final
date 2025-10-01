#!/bin/bash

# Setup script for Voxtral + Kokoro TTS Development Environment
# This script sets up the development environment without full deployment

set -e

echo "🔧 Setting up Voxtral + Kokoro TTS Development Environment"
echo "========================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python $python_version is compatible"
else
    print_error "Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found. Please install pip3."
    exit 1
fi

# Create virtual environment (optional but recommended)
if [ "$1" = "--venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created and activated"
fi

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Check for CUDA
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
    print_success "CUDA GPU detected"
    CUDA_AVAILABLE=true
else
    print_warning "No CUDA GPU detected. CPU-only mode will be used."
    CUDA_AVAILABLE=false
fi

# Install PyTorch
print_status "Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
print_status "Installing project dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directories..."
mkdir -p model_cache logs temp_audio
chmod 755 model_cache logs temp_audio

# Verify installation
print_status "Verifying installation..."
python3 -c "
import torch
import transformers
import fastapi
import websockets
print('✅ Core dependencies installed successfully')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'FastAPI: {fastapi.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"

# Create development scripts
print_status "Creating development scripts..."

# Create dev start script
cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Voxtral Development Server"

# Start health check in background
python3 -m src.api.health_check &
echo "Health check started on port 8005"

# Start main UI server
echo "Starting UI server on port 8000..."
python3 -m src.api.ui_server_realtime
EOF

chmod +x start_dev.sh

# Create test script
cat > test_imports.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing imports..."

python3 -c "
try:
    from src.models.voxtral_model_realtime import VoxtralModel
    print('✅ VoxtralModel import successful')
except Exception as e:
    print(f'❌ VoxtralModel import failed: {e}')

try:
    from src.models.kokoro_model_realtime import KokoroTTSModel
    print('✅ KokoroTTSModel import successful')
except Exception as e:
    print(f'❌ KokoroTTSModel import failed: {e}')

try:
    from src.api.ui_server_realtime import app
    print('✅ FastAPI app import successful')
except Exception as e:
    print(f'❌ FastAPI app import failed: {e}')

try:
    from src.utils.config import config
    print('✅ Config import successful')
except Exception as e:
    print(f'❌ Config import failed: {e}')
"
EOF

chmod +x test_imports.sh

print_success "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Test imports: ./test_imports.sh"
echo "2. Start development server: ./start_dev.sh"
echo "3. Access UI: http://localhost:8000"
echo ""
echo "💡 Tips:"
echo "- Use --venv flag to create virtual environment: ./setup.sh --venv"
echo "- Check logs in the logs/ directory"
echo "- Models will be cached in model_cache/ directory"
echo ""
if [ "$1" = "--venv" ]; then
    echo "🔄 To activate virtual environment later: source venv/bin/activate"
fi
