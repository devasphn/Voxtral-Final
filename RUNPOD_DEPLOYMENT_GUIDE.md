# RunPod Deployment Guide - Voxtral Voice Agent

## Ultra-Low Latency Voice AI (<500ms) - Step-by-Step Manual Commands

This guide provides exact commands for deploying the Voxtral Voice Agent on RunPod infrastructure with HTTP/TCP-only networking.

---

## Prerequisites

- RunPod account with GPU pod (RTX 4090 or A100 recommended)
- HuggingFace account with access token
- Basic familiarity with Linux terminal

---

## Step 1: Environment Setup

### 1.1 Connect to RunPod Web Terminal
```bash
# Open RunPod web terminal (provided by RunPod interface)
# Ensure you have a GPU pod running with at least 16GB VRAM
```

### 1.2 Update System
```bash
apt update && apt upgrade -y
```

### 1.3 Install System Dependencies
```bash
apt install -y git curl wget build-essential python3-dev python3-pip ffmpeg libsndfile1
```

### 1.4 Verify GPU Access
```bash
nvidia-smi
```
**Expected Output:** GPU information showing available VRAM

---

## Step 2: Project Setup

### 2.1 Clone Repository
```bash
cd /workspace
git clone https://github.com/devasphn/Voxtral-Final.git
cd Voxtral-Final
```

### 2.2 Set Environment Variables
```bash
export HF_TOKEN="your_huggingface_token_here"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 2.3 Create Required Directories
```bash
mkdir -p model_cache logs temp_audio
chmod 755 model_cache logs temp_audio
```

---

## Step 3: Dependency Installation

### 3.1 Upgrade pip
```bash
python3 -m pip install --upgrade pip
```

### 3.2 Install PyTorch with CUDA Support
```bash
pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3.3 Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 3.4 Verify Installation
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```
**Expected Output:** PyTorch version, CUDA Available: True, GPU name

---

## Step 4: Model Pre-loading

### 4.1 Pre-download Voxtral Model
```bash
python3 -c "
import os
os.environ['HF_TOKEN'] = '$HF_TOKEN'
from transformers import VoxtralForConditionalGeneration, AutoProcessor
print('Downloading Voxtral model...')
processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507', cache_dir='./model_cache')
model = VoxtralForConditionalGeneration.from_pretrained('mistralai/Voxtral-Mini-3B-2507', cache_dir='./model_cache', torch_dtype='auto', device_map='auto')
print('✅ Voxtral model cached successfully')
"
```

### 4.2 Pre-download Kokoro TTS Model
```bash
python3 -c "
import os
import sys
sys.path.append('.')
os.environ['HF_TOKEN'] = '$HF_TOKEN'
try:
    from src.models.kokoro_model_realtime import KokoroTTSModel
    print('Downloading Kokoro TTS model...')
    kokoro = KokoroTTSModel()
    print('✅ Kokoro TTS model cached successfully')
except Exception as e:
    print(f'⚠️ Kokoro model will be downloaded on first use: {e}')
"
```

---

## Step 5: Configuration

### 5.1 Verify Configuration
```bash
cat config.yaml | grep -E "(port|latency|chunk_size|sample_rate)"
```
**Expected Output:** Port 8000, latency targets <500ms, optimized settings

### 5.2 Apply Latency Optimizations
```bash
python3 scripts/optimize_latency.py
```
**Expected Output:** Optimization completed with latency targets met

---

## Step 6: Service Startup

### 6.1 Warm Up Models
```bash
python3 scripts/warmup_models.py
```
**Expected Output:** All models warmed up successfully

### 6.2 Start the Voice Agent Service
```bash
python3 src/api/ui_server_realtime.py
```
**Expected Output:** 
```
INFO - Server starting on 0.0.0.0:8000
INFO - Models initialized successfully
INFO - Ultra-low latency mode enabled
INFO - Ready for voice conversations
```

---

## Step 7: Health Checks

### 7.1 Check Service Status (New Terminal)
```bash
curl -f http://localhost:8000/health
```
**Expected Output:** `{"status": "healthy", "models": "loaded"}`

### 7.2 Check GPU Memory Usage
```bash
nvidia-smi
```
**Expected Output:** GPU memory usage showing loaded models

### 7.3 Check Process Status
```bash
ps aux | grep python3
```
**Expected Output:** Running Python processes including ui_server_realtime.py

### 7.4 Check Logs
```bash
tail -f logs/voxtral_streaming.log
```
**Expected Output:** Recent log entries showing successful initialization

---

## Step 8: Access and Testing

### 8.1 Get RunPod Proxy URL
```bash
echo "Access your voice agent at: https://[your-pod-id]-8000.proxy.runpod.net"
```

### 8.2 Test Pipeline Verification
```bash
python3 scripts/verify_pipeline.py
```
**Expected Output:** All pipeline steps verified successfully

### 8.3 Test Latency Performance
```bash
python3 -c "
import asyncio
import time
import sys
sys.path.append('.')
from scripts.warmup_models import ModelWarmup

async def test():
    warmup = ModelWarmup()
    start = time.time()
    success = await warmup.warmup_all_models()
    end = time.time()
    print(f'Model warmup time: {(end-start)*1000:.1f}ms')
    return success

result = asyncio.run(test())
print('✅ Latency test passed' if result else '❌ Latency test failed')
"
```

---

## Step 9: Usage Instructions

### 9.1 Open Voice Agent
1. Open browser and navigate to: `https://[your-pod-id]-8000.proxy.runpod.net`
2. You should see: "Voxtral Voice Agent - Ultra-Low Latency Voice AI (<500ms)"

### 9.2 Start Conversation
1. Click **"Connect"** button → Status should show "Connected"
2. Click **"Start"** button → Status should show "Listening"
3. Speak into your microphone → Voice agent will respond with audio

### 9.3 Expected Performance
- **First response latency:** <500ms
- **TTS chunking:** <200ms per chunk
- **Voice detection:** <100ms
- **End-to-end conversation:** <500ms

---

## Troubleshooting

### Common Issues and Solutions

**Issue:** CUDA out of memory
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
python3 -c "import torch; torch.cuda.empty_cache()"
```

**Issue:** HuggingFace authentication
```bash
huggingface-cli login --token $HF_TOKEN
```

**Issue:** Port already in use
```bash
lsof -ti:8000 | xargs kill -9
```

**Issue:** Audio not working
```bash
# Check audio system
pulseaudio --check -v
```

---

## Performance Monitoring

### Monitor Real-time Performance
```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Monitor logs
tail -f logs/voxtral_streaming.log | grep -E "(latency|ms|error)"

# Terminal 3: Monitor system resources
htop
```

---

## Stopping the Service

```bash
# Find and stop the service
pkill -f ui_server_realtime.py

# Clean up GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"
```

---

## Success Criteria

✅ **Deployment Successful When:**
- Service starts without errors
- Health check returns healthy status
- Web UI loads at RunPod proxy URL
- Voice conversation works end-to-end
- Latency targets are met (<500ms)
- GPU memory usage is stable
- No audio dropouts or overlaps

---

**🚀 Your ultra-low latency voice agent is now ready for real-time conversations!**
