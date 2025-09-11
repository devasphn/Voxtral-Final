# Voxtral Real-time Streaming Server

A high-performance real-time audio streaming implementation using Mistral AI's Voxtral model, optimized for <200ms latency responses. This system provides both WebSocket and TCP streaming interfaces for real-time speech recognition and understanding.

## 🚀 Features

- **Real-time Audio Processing**: Process streaming audio with <200ms latency
- **Multiple Interfaces**: WebSocket (8765) and TCP (8766) streaming servers
- **Web UI**: Modern web interface on port 8000
- **Health Monitoring**: Health check API on port 8005
- **Log-Mel Spectrogram Processing**: Optimized audio preprocessing matching Voxtral's architecture
- **Multiple Modes**: Transcription, understanding, and general audio processing
- **GPU Optimized**: Supports CUDA with memory optimization
- **Production Ready**: Comprehensive logging, error handling, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │    │  WebSocket       │    │  TCP Server     │
│   Port 8000     │    │  Port 8765       │    │  Port 8766      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Voxtral Model      │
                    │  Audio Processor    │
                    │  Log-Mel Spectrogram│
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Health Check       │
                    │  Port 8005          │
                    └─────────────────────┘
```

## 📋 Requirements

### System Requirements
- **GPU**: RTX A4500 (recommended) or any CUDA-compatible GPU with 8GB+ VRAM
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for model cache
- **OS**: Ubuntu 20.04+ or similar Linux distribution

### Python Dependencies
- Python 3.8+
- PyTorch 2.1.0+
- Transformers 4.45.0+
- FastAPI, WebSockets, Librosa, and more (see requirements.txt)

## 🚀 RunPod Deployment

### Step 1: Pod Configuration
When creating your RunPod pod:
- **Template**: PyTorch
- **GPU**: RTX A4500
- **Container Disk**: 50GB minimum
- **HTTP Ports**: 8000, 8005
- **TCP Ports**: 8765, 8766

### Step 2: Deploy Files
Upload all project files to `/workspace` on your RunPod instance.

### Step 3: Setup and Run
```bash
# Make scripts executable
chmod +x setup.sh run.sh

# Run setup (installs dependencies and downloads model)
./setup.sh

# Start all services
./run_realtime.sh
```

### Step 4: Access Services
- **Web UI**: `https://[POD_ID]-8000.proxy.runpod.net`
- **Health Check**: `https://[POD_ID]-8005.proxy.runpod.net/health`
- **WebSocket**: `wss://[POD_ID]-8000.proxy.runpod.net/ws`
- **TCP**: Direct connection to RunPod public IP on assigned port

## 🎯 Usage

### Web Interface
1. Open the web UI in your browser
2. Click "Connect" to establish WebSocket connection
3. Click "Start Recording" to begin audio capture
4. Speak into your microphone
5. Click "Stop Recording" to process audio
6. View transcription/response in real-time

### WebSocket API
```javascript
const ws = new WebSocket('ws://your-server:8765');

// Send audio data
ws.send(JSON.stringify({
    type: 'audio',
    audio_data: base64AudioData,
    mode: 'transcribe', // or 'understand'
    prompt: 'Optional prompt text'
}));

// Receive response
ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log(response.text);
};
```

### TCP API
```python
import socket
import json
import struct

# Connect to TCP server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('your-server', 8766))

# Send message
message = {
    'type': 'audio',
    'audio_data': base64_audio,
    'mode': 'transcribe'
}
data = json.dumps(message).encode('utf-8')
sock.send(struct.pack('!I', len(data)) + data)

# Receive response
length = struct.unpack('!I', sock.recv(4))[0]
response = json.loads(sock.recv(length).decode('utf-8'))
```

## 🔧 Configuration

### config.yaml
```yaml
server:
  host: "0.0.0.0"
  http_port: 8000
  health_port: 8005
  tcp_ports: [8765, 8766]

model:
  name: "mistralai/Voxtral-Mini-3B-2507"
  device: "cuda"
  torch_dtype: "bfloat16"

audio:
  sample_rate: 16000
  chunk_size: 1024
  channels: 1

spectrogram:
  n_mels: 128
  hop_length: 160
  win_length: 400
  n_fft: 400
```

## 📊 Performance Optimization

### Latency Optimization
- **Model Compilation**: Automatic `torch.compile()` when available
- **Mixed Precision**: bfloat16 inference
- **KV Caching**: Enabled for faster generation
- **Greedy Decoding**: For minimal latency
- **Optimized Audio Processing**: Efficient log-mel spectrogram generation

### Memory Optimization
- **Memory Mapping**: Efficient model loading
- **Chunked Processing**: 30-second audio segments
- **Buffer Management**: Optimized audio buffering
- **GPU Memory Management**: Configured memory allocation

## 🔍 Monitoring

### Health Endpoints
- `GET /health` - Basic health check
- `GET /status` - Detailed system status
- `GET /ready` - Model readiness probe

### Logging
- **File Logging**: `/workspace/logs/voxtral_streaming.log`
- **Console Logging**: Real-time console output
- **Structured Logging**: JSON-formatted logs with timestamps

### Metrics
- Processing latency (ms)
- Audio duration (s)
- GPU memory usage
- Connection count
- Error rates

## 🧪 Testing

Run the test suite:
```bash
pip install pytest
python -m pytest tests/test_streaming.py -v
```

### Test Coverage
- Audio preprocessing pipeline
- Log-mel spectrogram generation
- Model initialization and inference
- WebSocket message handling
- TCP server communication
- Configuration management
- Health check endpoints

## 🔧 Troubleshooting

### Common Issues

**Model Download Fails**
```bash
python -c "from transformers import VoxtralForConditionalGeneration; VoxtralForConditionalGeneration.from_pretrained('mistralai/Voxtral-Mini-3B-2507')"
```

**Audio Processing Errors**
```bash
python -c "import librosa, torch, torchaudio; print('Audio libraries OK')"
```

**Port Access Issues**
- Ensure ports are exposed in RunPod configuration
- Bind to `0.0.0.0`, not `127.0.0.1` or `localhost`
- Check firewall settings

**GPU Not Detected**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Performance Issues
- **High Latency**: Reduce `max_new_tokens`, use smaller audio chunks
- **Memory Issues**: Reduce batch size, enable gradient checkpointing
- **Connection Issues**: Check network configuration, increase timeout values

## 📁 Project Structure

```
voxtral_realtime_streaming/
├── src/
│   ├── models/
│   │   ├── voxtral_model.py      # Voxtral model wrapper
│   │   └── audio_processor.py    # Audio preprocessing
│   ├── streaming/
│   │   ├── websocket_server.py   # WebSocket server
│   │   └── tcp_server.py         # TCP server
│   ├── api/
│   │   ├── ui_server.py          # Web UI server
│   │   └── health_check.py       # Health monitoring
│   └── utils/
│       ├── config.py             # Configuration management
│       └── logging_config.py     # Logging setup
├── config/
│   ├── requirements.txt          # Python dependencies
│   └── config.yaml              # Configuration file
├── scripts/
│   ├── setup.sh                 # Setup script
│   └── run.sh                   # Run script
└── tests/
    └── test_streaming.py        # Test suite
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Voxtral model
- **Hugging Face** for the Transformers library
- **RunPod** for GPU infrastructure
- **FastAPI** and **WebSockets** for the API framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `/workspace/logs/`
3. Test individual components
4. Open an issue with detailed error information

---

**Built with ❤️ for real-time AI audio processing**