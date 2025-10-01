# RunPod Manual Deployment Guide - Voxtral Voice AI

## Ultra-Low Latency Voice AI System (<500ms end-to-end)

This guide provides step-by-step manual commands for deploying Voxtral Voice AI on RunPod infrastructure.

---

## Prerequisites

- RunPod account with GPU pod access
- Pod with CUDA 12.1 support
- Minimum 16GB GPU memory recommended
- HTTP/TCP-only networking (no UDP/WebRTC)

---

## Step-by-Step Deployment Commands

### Step 1: Navigate to Workspace and Clone Repository

```bash
cd /workspace
```

```bash
git clone https://github.com/devasphn/Voxtral-Final.git
```

```bash
cd Voxtral-Final
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv voxtral_env
```

```bash
source voxtral_env/bin/activate
```

### Step 3: Upgrade pip and Install System Dependencies

```bash
pip install --upgrade pip
```

```bash
apt-get update
```

```bash
apt-get install -y ffmpeg libsndfile1 portaudio19-dev
```

### Step 4: Install PyTorch with CUDA 12.1 Support

```bash
pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
python verify_requirements.py
```

### Step 7: Create Necessary Directories

```bash
mkdir -p logs model_cache temp_audio
```

```bash
chmod 755 logs model_cache temp_audio
```

### Step 8: Pre-download Models (Eliminate Cold Starts)

```bash
python -c "
from transformers import VoxtralForConditionalGeneration, AutoProcessor
print('Downloading Voxtral model...')
model = VoxtralForConditionalGeneration.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
print('Voxtral model downloaded successfully')
"
```

```bash
python -c "
import kokoro
print('Kokoro TTS ready')
"
```

### Step 9: Test GPU and CUDA

```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

### Step 10: Initialize RunPod Optimizations

```bash
python -c "
import asyncio
from src.utils.runpod_optimizer import initialize_runpod_optimizations
asyncio.run(initialize_runpod_optimizations())
print('RunPod optimizations initialized')
"
```

### Step 11: Test Pipeline Components

```bash
python -c "
import asyncio
from src.pipeline.ultra_low_latency_pipeline import initialize_pipeline
result = asyncio.run(initialize_pipeline())
print(f'Pipeline initialization: {\"Success\" if result else \"Failed\"}')
"
```

### Step 12: Start the Voice AI Server (Simple UI)

```bash
python -m uvicorn src.api.ui_server_simple:app --host 0.0.0.0 --port 8000 --workers 1
```

### Step 13: Start WebSocket Streaming Server (Separate Terminal)

Open a new terminal and run:

```bash
cd /workspace/Voxtral-Final
```

```bash
source voxtral_env/bin/activate
```

```bash
python -c "
import asyncio
from src.streaming.runpod_streaming import start_runpod_streaming_server
asyncio.run(start_runpod_streaming_server(host='0.0.0.0', port=8765))
"
```

---

## Access URLs

After successful deployment, access your voice AI system using:

### Web Interface
```
https://[POD_ID]-8000.proxy.runpod.net
```

### WebSocket Endpoint
```
wss://[POD_ID]-8765.proxy.runpod.net
```

Replace `[POD_ID]` with your actual RunPod pod ID.

---

## Health Check Commands

### Check System Status

```bash
python -c "
import asyncio
from src.utils.runpod_optimizer import runpod_health_check
result = asyncio.run(runpod_health_check())
print('Health Check Results:')
for key, value in result.items():
    print(f'  {key}: {value}')
"
```

### Check Pipeline Performance

```bash
python -c "
from src.pipeline.ultra_low_latency_pipeline import get_pipeline_stats
stats = get_pipeline_stats()
print('Pipeline Performance:')
for key, value in stats.items():
    print(f'  {key}: {value}')
"
```

### Check Latency Optimization

```bash
python -c "
from src.optimization.latency_optimizer import get_latency_report
report = get_latency_report()
print('Latency Optimization Report:')
print(f'  Targets: {report.get(\"targets\", {})}')
print(f'  Optimizations: {report.get(\"optimizations_enabled\", {})}')
"
```

---

## Troubleshooting Commands

### Check GPU Memory Usage

```bash
nvidia-smi
```

### Check Process Status

```bash
ps aux | grep python
```

### Check Port Usage

```bash
netstat -tulpn | grep :8000
netstat -tulpn | grep :8765
```

### View Logs

```bash
tail -f logs/voxtral_streaming.log
```

### Restart Services

```bash
pkill -f uvicorn
pkill -f python
```

Then restart from Step 12.

---

## Performance Targets

- **End-to-End Latency**: <500ms
- **TTS Chunking**: <200ms
- **Audio Preprocessing**: <50ms
- **Voxtral Inference**: <150ms
- **Response Generation**: <50ms
- **Kokoro Synthesis**: <200ms

---

## Configuration Options

### Enable Ultra-Low Latency Mode

Edit `config.yaml`:

```yaml
ui:
  mode: "simple"
  
speech_to_speech:
  latency_target_ms: 200
  streaming:
    mode: "word_level"
    words_trigger_threshold: 2
```

### Adjust Performance Settings

```yaml
performance:
  gpu_memory_fraction: 0.9
  enable_mixed_precision: true
  batch_size: 1
  max_sequence_length: 100
```

---

## Security Notes

- All communication uses HTTPS/WSS through RunPod proxy
- No direct UDP/WebRTC exposure
- Audio data is processed in real-time and not stored
- Models are cached locally for performance

---

## Support

For issues or questions:
1. Check the health check commands above
2. Review logs in `/workspace/Voxtral-Final/logs/`
3. Verify GPU and CUDA availability
4. Ensure all dependencies are correctly installed

---

**Deployment Complete!** 

Your ultra-low latency voice AI system is now running on RunPod with:
- ✅ <500ms end-to-end latency
- ✅ Real-time chunked streaming
- ✅ RunPod HTTP/TCP optimization
- ✅ Simplified UI interface
- ✅ Comprehensive monitoring
