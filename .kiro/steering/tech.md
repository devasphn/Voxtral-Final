# Technology Stack

## Core Technologies

### Python & Frameworks
- **Python**: 3.8+ (3.12 not supported due to model compatibility)
- **FastAPI**: Web framework for API endpoints and WebSocket handling
- **Uvicorn**: ASGI server for production deployment
- **Pydantic v2**: Data validation and settings management with `pydantic-settings`

### AI/ML Stack
- **PyTorch**: 2.4.1+ with CUDA 12.1 support
- **Transformers**: 4.45.2+ (Hugging Face)
- **Mistral Voxtral**: `mistralai/Voxtral-Mini-3B-2507` for speech recognition
- **Kokoro TTS**: High-quality text-to-speech synthesis with 8 optimized voices
- **Audio Processing**: librosa, soundfile, numpy for audio pipeline

### Audio Processing
- **librosa**: Audio analysis and feature extraction
- **torchaudio**: PyTorch audio processing
- **soundfile**: Audio file I/O
- **numpy**: Numerical computations for audio data

### Infrastructure
- **CUDA**: GPU acceleration throughout pipeline
- **WebSockets**: Real-time bidirectional communication
- **RunPod**: Primary cloud deployment platform
- **Docker**: Containerization (PyTorch 2.1.0+ with CUDA 12.1)

## Build & Development Commands

### Setup & Installation
```bash
# Complete deployment (single command)
bash deploy_voxtral_tts.sh

# Manual setup
bash setup.sh
pip install -r requirements.txt
```

### Running Services
```bash
# Start all services
python -m src.api.ui_server_realtime &    # Main UI (port 8000)
python -m src.api.health_check &          # Health check (port 8005)
python -m src.streaming.tcp_server &      # TCP server (port 8766)

# Stop services
pkill -f 'src.api' && pkill -f 'src.streaming'
```

### Testing
```bash
# Run test suite
python -m pytest tests/test_streaming.py -v

# Integration testing
python test_integration.py

# Setup validation
python validate_setup.py
```

### Model Management
```bash
# Pre-cache models (done automatically in setup)
python -c "from transformers import VoxtralForConditionalGeneration, AutoProcessor; ..."

# Clear model cache
rm -rf /workspace/model_cache
```

## Performance Optimization

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
```

### GPU Requirements
- **Minimum**: 8GB VRAM (RTX 3070/4060 Ti)
- **Recommended**: RTX A4500 or better
- **Memory**: 16GB+ system RAM
- **Storage**: 50GB+ for model cache

## Configuration Management

### Primary Config
- `config.yaml`: Main application configuration
- Environment-specific settings via Pydantic BaseSettings
- Model paths and performance tuning parameters

### Key Configuration Classes
- `ServerConfig`: Ports and networking
- `ModelConfig`: AI model settings
- `AudioConfig`: Audio processing parameters
- `TTSConfig`: Text-to-speech configuration