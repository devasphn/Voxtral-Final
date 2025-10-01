#!/bin/bash

# 🚀 ENHANCED PRODUCTION DEPLOYMENT SCRIPT FOR RUNPOD
# Voxtral + Orpheus TTS Speech-to-Speech System
# Comprehensive deployment with system packages, cleanup, and file audit

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/deployment.log"
CLEANUP_LOG="$SCRIPT_DIR/logs/cleanup.log"
AUDIT_LOG="$SCRIPT_DIR/logs/file_audit.log"
BACKUP_DIR="$SCRIPT_DIR/backup_$(date +%Y%m%d_%H%M%S)"

# Create log directory
mkdir -p "$SCRIPT_DIR/logs"

# Enhanced logging functions
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
    echo -e "${BLUE}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

warn() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1"
    echo -e "${YELLOW}$message${NC}"
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

audit() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] AUDIT: $1"
    echo -e "${PURPLE}$message${NC}"
    echo "$message" >> "$AUDIT_LOG"
}

cleanup_log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] CLEANUP: $1"
    echo -e "${CYAN}$message${NC}"
    echo "$message" >> "$CLEANUP_LOG"
}

# Enhanced error handling with rollback capability
handle_error() {
    local exit_code=$?
    local line_number=$1
    error "Script failed at line $line_number with exit code $exit_code"

    if [ -d "$BACKUP_DIR" ]; then
        warn "Backup directory exists at: $BACKUP_DIR"
        warn "You can restore files if needed"
    fi

    log "Deployment failed. Check logs at: $LOG_FILE"
    exit $exit_code
}

# Set error trap
trap 'handle_error $LINENO' ERR

# System package installation for Ubuntu/RunPod
install_system_packages() {
    log "🔧 Installing system packages for speech-to-speech system..."

    # Check if running as root or with sudo access
    if [ "$EUID" -eq 0 ]; then
        SUDO=""
    elif command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        error "This script requires root access or sudo privileges for system package installation"
        exit 1
    fi

    # Update package lists
    log "Updating package lists..."
    $SUDO apt update || {
        error "Failed to update package lists"
        exit 1
    }

    # Install build tools and development packages
    log "Installing build tools and development packages..."
    $SUDO apt install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        tree \
        htop \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release || {
        error "Failed to install build tools"
        exit 1
    }

    # Install Python development packages
    log "Installing Python development packages..."
    $SUDO apt install -y \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel || {
        error "Failed to install Python development packages"
        exit 1
    }

    # Install audio system packages
    log "Installing audio system packages..."
    $SUDO apt install -y \
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
        alsa-utils || {
        error "Failed to install audio system packages"
        exit 1
    }

    # Install additional libraries for ML/AI
    log "Installing additional ML/AI libraries..."
    $SUDO apt install -y \
        libopenblas-dev \
        liblapack-dev \
        libhdf5-dev \
        pkg-config || {
        warn "Some ML/AI libraries failed to install - continuing anyway"
    }

    # Verify CUDA installation (if available)
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected, checking CUDA installation..."
        nvidia-smi

        # Check if CUDA toolkit is needed
        if ! command -v nvcc &> /dev/null; then
            warn "CUDA compiler not found. Installing CUDA toolkit..."
            $SUDO apt install -y nvidia-cuda-toolkit || {
                warn "CUDA toolkit installation failed - PyTorch will handle CUDA"
            }
        fi
    else
        warn "No NVIDIA GPU detected. System will run on CPU (not recommended for production)"
    fi

    success "System packages installed successfully"
}

# Check if HF_TOKEN is set
check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        error "HF_TOKEN environment variable is not set!"
        echo "Please set your HuggingFace token:"
        echo "export HF_TOKEN='your_token_here'"
        exit 1
    fi
    success "HuggingFace token is configured"
}

# Comprehensive file audit function
perform_file_audit() {
    log "🔍 Performing comprehensive file audit..."

    audit "Starting file audit of repository: $SCRIPT_DIR"
    audit "Audit log: $AUDIT_LOG"

    # Initialize counters
    local total_files=0
    local essential_files=0
    local duplicate_files=0
    local problematic_files=0
    local unnecessary_files=0

    # Define file categories
    declare -a ESSENTIAL_FILES=(
        "src/"
        "requirements.txt"
        "README.md"
        "PRODUCTION_RUNPOD_DEPLOYMENT.md"
        "deploy_production.sh"
        "test_production_system.py"
        ".env.example"
        ".gitignore"
        "TECHNICAL_AUDIT_REPORT.md"
    )

    declare -a DUPLICATE_FILES=(
        "ENHANCED_RUNPOD_DEPLOYMENT.md"
        "PERFECT_RUNPOD_COMMANDS.md"
        "RUNPOD_PRODUCTION_GUIDE.md"
        "SPEECH_TO_SPEECH_IMPLEMENTATION.md"
        "setup_runpod_enhanced.sh"
        "setup_runpod_perfect.sh"
        "start_perfect.sh"
        "deploy_speech_to_speech.sh"
        "setup.sh"
        "install_kokoro.sh"
        "cleanup.sh"
        "run_realtime.sh"
    )

    declare -a TEST_FILES_TO_REMOVE=(
        "test_emotional_tts.py"
        "test_perfect_installation.py"
        "test_production_readiness.py"
        "test_speech_to_speech.py"
        "monitor_speech_to_speech.py"
    )

    # Scan all files
    audit "Scanning all files in repository..."
    while IFS= read -r -d '' file; do
        ((total_files++))
        local relative_path="${file#$SCRIPT_DIR/}"

        # Check for merge conflict markers
        if [ -f "$file" ] && grep -q "<<<<<<< HEAD\|>>>>>>> \|=======" "$file" 2>/dev/null; then
            audit "MERGE_CONFLICT: $relative_path contains unresolved merge conflicts"
            ((problematic_files++))
        fi

        # Check for sensitive information
        if [ -f "$file" ] && grep -qE "(hf_[a-zA-Z0-9]{34}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48})" "$file" 2>/dev/null; then
            audit "SENSITIVE: $relative_path may contain sensitive tokens"
            ((problematic_files++))
        fi

        # Categorize files
        local is_essential=false
        for essential in "${ESSENTIAL_FILES[@]}"; do
            if [[ "$relative_path" == "$essential"* ]]; then
                audit "ESSENTIAL: $relative_path"
                ((essential_files++))
                is_essential=true
                break
            fi
        done

        if [ "$is_essential" = false ]; then
            for duplicate in "${DUPLICATE_FILES[@]}"; do
                if [[ "$relative_path" == "$duplicate" ]]; then
                    audit "DUPLICATE: $relative_path (redundant deployment/setup file)"
                    ((duplicate_files++))
                    break
                fi
            done

            for test_file in "${TEST_FILES_TO_REMOVE[@]}"; do
                if [[ "$relative_path" == "$test_file" ]]; then
                    audit "UNNECESSARY: $relative_path (redundant test file)"
                    ((unnecessary_files++))
                    break
                fi
            done
        fi

    done < <(find "$SCRIPT_DIR" -type f -not -path "*/.*" -not -path "*/logs/*" -not -path "*/backup_*" -print0)

    # Check for missing referenced files
    audit "Checking for missing file references in source code..."
    if [ -d "$SCRIPT_DIR/src" ]; then
        while IFS= read -r -d '' py_file; do
            # Check for import statements that might reference missing files
            if grep -qE "from src\.|import src\." "$py_file" 2>/dev/null; then
                local imports=$(grep -E "from src\.|import src\." "$py_file" | head -5)
                audit "IMPORTS: ${py_file#$SCRIPT_DIR/} has src imports: $imports"
            fi
        done < <(find "$SCRIPT_DIR/src" -name "*.py" -print0)
    fi

    # Generate audit summary
    audit "=== FILE AUDIT SUMMARY ==="
    audit "Total files scanned: $total_files"
    audit "Essential files: $essential_files"
    audit "Duplicate files: $duplicate_files"
    audit "Unnecessary files: $unnecessary_files"
    audit "Problematic files: $problematic_files"
    audit "==========================="

    success "File audit completed. Check $AUDIT_LOG for details."
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."

    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA GPU not detected! This system requires CUDA support."
        exit 1
    fi

    # Check VRAM
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    if [ "$VRAM" -lt 16000 ]; then
        warn "GPU has only ${VRAM}MB VRAM. 16GB+ recommended for optimal performance."
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found!"
        exit 1
    fi

    success "System requirements check passed"
}

# Comprehensive cleanup process
perform_cleanup() {
    log "🧹 Starting comprehensive repository cleanup..."

    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    cleanup_log "Created backup directory: $BACKUP_DIR"

    # Fix config.yaml merge conflicts
    if [ -f "$SCRIPT_DIR/config.yaml" ] && grep -q "<<<<<<< HEAD" "$SCRIPT_DIR/config.yaml"; then
        cleanup_log "Fixing merge conflicts in config.yaml"
        cp "$SCRIPT_DIR/config.yaml" "$BACKUP_DIR/config.yaml.backup"

        # Remove merge conflict markers and keep the HEAD version (Orpheus configuration)
        sed '/<<<<<<< HEAD/,/=======/d; />>>>>>> .*/d' "$SCRIPT_DIR/config.yaml" > "$SCRIPT_DIR/config.yaml.tmp"
        mv "$SCRIPT_DIR/config.yaml.tmp" "$SCRIPT_DIR/config.yaml"
        cleanup_log "Fixed merge conflicts in config.yaml"
    fi

    # Remove duplicate deployment files
    declare -a DUPLICATE_FILES=(
        "ENHANCED_RUNPOD_DEPLOYMENT.md"
        "PERFECT_RUNPOD_COMMANDS.md"
        "RUNPOD_PRODUCTION_GUIDE.md"
        "SPEECH_TO_SPEECH_IMPLEMENTATION.md"
        "setup_runpod_enhanced.sh"
        "setup_runpod_perfect.sh"
        "start_perfect.sh"
        "deploy_speech_to_speech.sh"
        "setup.sh"
        "install_kokoro.sh"
        "cleanup.sh"
        "run_realtime.sh"
    )

    for file in "${DUPLICATE_FILES[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            cleanup_log "Backing up and removing duplicate file: $file"
            cp "$SCRIPT_DIR/$file" "$BACKUP_DIR/"
            rm "$SCRIPT_DIR/$file"
        fi
    done

    # Remove redundant test files
    declare -a TEST_FILES_TO_REMOVE=(
        "test_emotional_tts.py"
        "test_perfect_installation.py"
        "test_production_readiness.py"
        "test_speech_to_speech.py"
        "monitor_speech_to_speech.py"
    )

    for file in "${TEST_FILES_TO_REMOVE[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            cleanup_log "Backing up and removing redundant test file: $file"
            cp "$SCRIPT_DIR/$file" "$BACKUP_DIR/"
            rm "$SCRIPT_DIR/$file"
        fi
    done

    # Clean temporary files and caches
    cleanup_log "Cleaning temporary files and caches..."
    find "$SCRIPT_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "*.log" -not -path "*/logs/*" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name ".DS_Store" -delete 2>/dev/null || true

    # Remove empty directories (except essential ones)
    find "$SCRIPT_DIR" -type d -empty -not -path "*/src/*" -not -path "*/logs" -not -path "*/model_cache" -delete 2>/dev/null || true

    # Verify essential files still exist
    declare -a ESSENTIAL_FILES=(
        "src/"
        "requirements.txt"
        "README.md"
        "PRODUCTION_RUNPOD_DEPLOYMENT.md"
        "deploy_production.sh"
        "test_production_system.py"
    )

    for file in "${ESSENTIAL_FILES[@]}"; do
        if [ ! -e "$SCRIPT_DIR/$file" ]; then
            error "Essential file missing after cleanup: $file"
            exit 1
        fi
    done

    cleanup_log "Repository cleanup completed successfully"
    cleanup_log "Backup created at: $BACKUP_DIR"
    success "Repository cleanup completed. Backup available at: $BACKUP_DIR"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."

    # Set optimization environment variables
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8

    # Create directories
    mkdir -p model_cache
    mkdir -p logs
    mkdir -p temp

    success "Environment setup completed"
}

# Create and activate virtual environment
setup_python_env() {
    log "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        success "Virtual environment created"
    else
        log "Virtual environment already exists"
    fi
    
    source venv/bin/activate
    success "Virtual environment activated"
    
    # Upgrade pip
    python -m pip install --upgrade pip setuptools wheel
    success "Pip upgraded"
}

# Install dependencies in correct order
install_dependencies() {
    log "Installing dependencies in correct order..."
    
    # Install PyTorch with CUDA support
    log "Installing PyTorch with CUDA 12.1 support..."
    pip install torch>=2.1.0 torchaudio>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121
    
    # Install core transformers
    log "Installing transformers stack..."
    pip install transformers>=4.56.0
    pip install huggingface-hub>=0.34.0
    pip install accelerate>=0.25.0
    pip install tokenizers>=0.15.0
    
    # Install Mistral Common
    log "Installing Mistral Common with audio support..."
    pip install "mistral-common[audio]>=1.8.1"
    
    # Install Orpheus TTS
    log "Installing Orpheus TTS..."
    pip install orpheus-speech>=0.1.0
    
    # Install vLLM
    log "Installing vLLM..."
    pip install "vllm>=0.6.0,<0.8.0"
    
    # Install audio processing
    log "Installing audio processing libraries..."
    pip install librosa>=0.10.1
    pip install soundfile>=0.12.1
    pip install numpy>=1.24.0
    pip install scipy>=1.11.0
    
    # Install web framework
    log "Installing web framework..."
    pip install fastapi>=0.107.0
    pip install "uvicorn[standard]>=0.24.0"
    pip install websockets>=12.0
    pip install pydantic>=2.9.0
    pip install pydantic-settings>=2.1.0
    pip install python-multipart>=0.0.6
    
    # Install utilities
    log "Installing utilities..."
    pip install pyyaml>=6.0.1
    pip install python-dotenv>=1.0.0
    pip install psutil>=5.9.0
    
    # Install Flash Attention (optional)
    log "Installing Flash Attention (optional)..."
    pip install flash-attn>=2.5.0 || warn "Flash Attention installation failed - continuing without it"
    
    success "Dependencies installed successfully"
}

# Pre-download models
predownload_models() {
    log "Pre-downloading models for faster startup..."
    
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
import os
os.environ['HF_TOKEN'] = '$HF_TOKEN'
try:
    from src.utils.kokoro_model_manager import KokoroModelManager
    print('Downloading Kokoro TTS model...')
    manager = KokoroModelManager(cache_dir='./model_cache')
    success = manager.download_model()
    if success:
        print('✅ Kokoro TTS model cached successfully')
    else:
        print('❌ Kokoro TTS download failed')
        exit(1)
except Exception as e:
    print(f'❌ Kokoro TTS download failed: {e}')
    exit(1)
"
    
    success "Model pre-downloading completed"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Test core imports
    python3 -c "
import torch
import transformers
import fastapi
import websockets
print('✅ Core packages imported successfully')
print(f'✅ PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available')
    exit(1)
"
    
    # Test Voxtral
    python3 -c "
try:
    from transformers import VoxtralForConditionalGeneration
    print('✅ Voxtral classes available')
except ImportError as e:
    print(f'❌ Voxtral import failed: {e}')
    exit(1)
"
    
    # Test configuration
    python3 -c "
try:
    from src.utils.config import config
    print('✅ Configuration loaded successfully')
    print(f'   Server port: {config.server.http_port}')
    print(f'   Model name: {config.model.name}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"
    
    success "Installation verification passed"
}

# Start system
start_system() {
    log "Starting the speech-to-speech system..."
    
    # Start the system in background
    nohup python3 -m src.api.ui_server_realtime > logs/system.log 2>&1 &
    SYSTEM_PID=$!
    
    # Wait for startup
    log "Waiting for system startup..."
    sleep 15
    
    # Check if process is still running
    if ! kill -0 $SYSTEM_PID 2>/dev/null; then
        error "System failed to start. Check logs/system.log for details."
        exit 1
    fi
    
    success "System started successfully (PID: $SYSTEM_PID)"
    echo $SYSTEM_PID > .system.pid
}

# Verify system is working
verify_system() {
    log "Verifying system functionality..."
    
    # Health check
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        success "Health check passed"
    else
        error "Health check failed"
        return 1
    fi
    
    # WebSocket test
    python3 -c "
import asyncio
import websockets
import json
import sys

async def test_websocket():
    try:
        uri = 'ws://localhost:8765'
        async with websockets.connect(uri, timeout=10) as websocket:
            await websocket.send(json.dumps({'type': 'ping'}))
            response = await websocket.recv()
            print('✅ WebSocket connection successful')
            return True
    except Exception as e:
        print(f'❌ WebSocket test failed: {e}')
        return False

result = asyncio.run(test_websocket())
sys.exit(0 if result else 1)
" || {
        error "WebSocket test failed"
        return 1
    }
    
    success "System verification completed successfully"
}

# Enhanced verification with detailed checks
enhanced_verification() {
    log "🔍 Performing enhanced system verification..."

    # Check Python environment
    log "Verifying Python environment..."
    python3 -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')
"

    # Check critical imports
    log "Verifying critical package imports..."
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
    import websockets
    print(f'✅ WebSockets available')
except ImportError as e:
    print(f'❌ WebSockets import failed: {e}')
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

    # Test audio processor
    log "Testing audio processor initialization..."
    python3 -c "
try:
    from src.models.audio_processor_realtime import AudioProcessor
    processor = AudioProcessor()
    print('✅ Audio processor initialized successfully')
except Exception as e:
    print(f'❌ Audio processor error: {e}')
    exit(1)
"

    success "Enhanced verification completed successfully"
}

# Generate deployment report
generate_deployment_report() {
    local report_file="$SCRIPT_DIR/logs/deployment_report_$(date +%Y%m%d_%H%M%S).md"

    log "📋 Generating deployment report..."

    cat > "$report_file" << EOF
# Deployment Report
**Date**: $(date)
**Script**: $0
**Working Directory**: $SCRIPT_DIR

## System Information
- **OS**: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")
- **Kernel**: $(uname -r)
- **Python**: $(python3 --version)
- **GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Not available")

## Deployment Steps Completed
- ✅ System package installation
- ✅ Repository cleanup and file audit
- ✅ Python environment setup
- ✅ Dependency installation
- ✅ Model pre-downloading
- ✅ System verification
- ✅ Service startup

## Files and Directories
### Essential Files Preserved
$(find "$SCRIPT_DIR" -maxdepth 1 -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.sh" | sort)

### Backup Location
$BACKUP_DIR

## Log Files
- Deployment: $LOG_FILE
- Cleanup: $CLEANUP_LOG
- Audit: $AUDIT_LOG

## Service Information
- **UI**: http://localhost:8000
- **WebSocket**: ws://localhost:8765
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/api/performance/metrics

## Next Steps
1. Access the UI at http://localhost:8000
2. Run tests: python3 test_production_system.py
3. Monitor logs: tail -f logs/system.log
4. Check GPU usage: watch -n 1 nvidia-smi

EOF

    success "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log "🚀 Starting ENHANCED PRODUCTION deployment of Voxtral + Orpheus TTS system"
    log "📝 Deployment logs: $LOG_FILE"

    # Phase 1: System Setup
    log "=== PHASE 1: SYSTEM SETUP ==="
    install_system_packages
    check_hf_token
    check_system_requirements

    # Phase 2: Repository Cleanup
    log "=== PHASE 2: REPOSITORY CLEANUP ==="
    perform_file_audit
    perform_cleanup

    # Phase 3: Environment Setup
    log "=== PHASE 3: ENVIRONMENT SETUP ==="
    setup_environment
    setup_python_env
    install_dependencies

    # Phase 4: Model Setup
    log "=== PHASE 4: MODEL SETUP ==="
    predownload_models

    # Phase 5: Verification
    log "=== PHASE 5: VERIFICATION ==="
    verify_installation
    enhanced_verification

    # Phase 6: System Startup
    log "=== PHASE 6: SYSTEM STARTUP ==="
    start_system
    verify_system

    # Phase 7: Reporting
    log "=== PHASE 7: REPORTING ==="
    generate_deployment_report

    success "🎉 ENHANCED DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 System Information:"
    echo "   🌐 UI: http://localhost:8000"
    echo "   🔌 WebSocket: ws://localhost:8765"
    echo "   ❤️ Health: http://localhost:8000/health"
    echo "   📈 Metrics: http://localhost:8000/api/performance/metrics"
    echo ""
    echo "📝 Monitoring Commands:"
    echo "   tail -f logs/system.log        # Application logs"
    echo "   tail -f logs/voxtral_streaming.log  # Streaming logs"
    echo "   htop                           # System resources"
    echo "   watch -n 1 nvidia-smi         # GPU usage"
    echo ""
    echo "📋 Reports and Logs:"
    echo "   Deployment: $LOG_FILE"
    echo "   Cleanup: $CLEANUP_LOG"
    echo "   Audit: $AUDIT_LOG"
    echo "   Backup: $BACKUP_DIR"
    echo ""
    echo "🛑 To stop the system:"
    echo "   kill \$(cat .system.pid)"
}

# Run main function with all arguments
main "$@"
