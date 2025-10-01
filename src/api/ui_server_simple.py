"""
Simplified UI server for Voxtral Voice AI
Essential elements only: Connect, Start, Status
Optimized for ultra-low latency (<500ms end-to-end)
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import asyncio
import time
import json
import base64
import numpy as np
import logging
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import config
from src.utils.logging_config import logger

# Initialize FastAPI app
app = FastAPI(
    title="Voxtral Voice AI - Simplified",
    description="Ultra-low latency voice AI with essential controls only",
    version="3.0.0"
)

# Enhanced logging
ui_logger = logging.getLogger("simple_ui")
ui_logger.setLevel(logging.INFO)

# Global variables for model management
_unified_manager = None
_speech_to_speech_pipeline = None

def get_unified_manager():
    """Get unified model manager instance"""
    global _unified_manager
    if _unified_manager is None:
        from src.models.unified_model_manager import unified_model_manager
        _unified_manager = unified_model_manager
        ui_logger.info("Unified model manager loaded")
    return _unified_manager

def get_speech_to_speech_pipeline():
    """Get speech-to-speech pipeline instance"""
    global _speech_to_speech_pipeline
    if _speech_to_speech_pipeline is None:
        from src.models.speech_to_speech_pipeline import speech_to_speech_pipeline
        _speech_to_speech_pipeline = speech_to_speech_pipeline
        ui_logger.info("Speech-to-Speech pipeline loaded")
    return _speech_to_speech_pipeline

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve simplified voice AI interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voxtral Voice AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            text-align: center;
            max-width: 500px;
            width: 90%;
        }
        h1 {
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .controls {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 30px;
        }
        button {
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            min-height: 60px;
        }
        .connect-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }
        .start-btn {
            background: linear-gradient(45deg, #00b894, #00a085);
            color: white;
        }
        .stop-btn {
            background: linear-gradient(45deg, #e17055, #d63031);
            color: white;
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            border-left: 4px solid #00b894;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-connected { background: #00b894; }
        .status-disconnected { background: #e17055; }
        .status-recording {
            background: #fdcb6e;
            animation: pulse 1s infinite;
            box-shadow: 0 0 10px rgba(253, 203, 110, 0.5);
        }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
            100% { opacity: 1; transform: scale(1); }
        }
        .latency-info {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
        }
        .performance-indicator {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .perf-excellent { background: rgba(0, 184, 148, 0.3); }
        .perf-good { background: rgba(253, 203, 110, 0.3); }
        .perf-poor { background: rgba(225, 112, 85, 0.3); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voxtral Voice AI</h1>
        
        <div class="controls">
            <button id="connectBtn" class="connect-btn" onclick="toggleConnection()">
                Connect
            </button>
            <button id="startBtn" class="start-btn" onclick="toggleRecording()" disabled>
                Start Conversation
            </button>
        </div>
        
        <div class="status">
            <div id="connectionStatus">
                <span class="status-indicator status-disconnected"></span>
                <strong>Status:</strong> Disconnected
            </div>
            <div id="recordingStatus" style="margin-top: 10px;">
                <span class="status-indicator status-disconnected"></span>
                <strong>Recording:</strong> Stopped
            </div>
            <div class="latency-info" id="latencyInfo">
                Target: &lt;500ms end-to-end | &lt;200ms TTS chunking
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioStream = null;
        let isConnected = false;
        let isRecording = false;

        function updateStatus(connection, recording, latency = null) {
            const connStatus = document.getElementById('connectionStatus');
            const recStatus = document.getElementById('recordingStatus');
            const latencyInfo = document.getElementById('latencyInfo');
            
            // Update connection status
            const connIndicator = connStatus.querySelector('.status-indicator');
            if (connection) {
                connIndicator.className = 'status-indicator status-connected';
                connStatus.innerHTML = '<span class="status-indicator status-connected"></span><strong>Status:</strong> Connected';
            } else {
                connIndicator.className = 'status-indicator status-disconnected';
                connStatus.innerHTML = '<span class="status-indicator status-disconnected"></span><strong>Status:</strong> Disconnected';
            }
            
            // Update recording status
            const recIndicator = recStatus.querySelector('.status-indicator');
            if (recording) {
                recIndicator.className = 'status-indicator status-recording';
                recStatus.innerHTML = '<span class="status-indicator status-recording"></span><strong>Recording:</strong> Active';
            } else {
                recIndicator.className = 'status-indicator status-disconnected';
                recStatus.innerHTML = '<span class="status-indicator status-disconnected"></span><strong>Recording:</strong> Stopped';
            }
            
            // Update latency info with performance indicator
            if (latency) {
                let perfClass = 'perf-excellent';
                let perfText = 'Excellent';

                if (latency > 500) {
                    perfClass = 'perf-poor';
                    perfText = 'Needs Optimization';
                } else if (latency > 300) {
                    perfClass = 'perf-good';
                    perfText = 'Good';
                }

                latencyInfo.innerHTML = `Last response: ${latency}ms | Target: &lt;500ms end-to-end <span class="performance-indicator ${perfClass}">${perfText}</span>`;
            }
        }

        function toggleConnection() {
            const connectBtn = document.getElementById('connectBtn');
            const startBtn = document.getElementById('startBtn');
            
            if (!isConnected) {
                // Connect
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    isConnected = true;
                    connectBtn.textContent = 'Disconnect';
                    connectBtn.className = 'stop-btn';
                    startBtn.disabled = false;
                    updateStatus(true, false);
                };
                
                ws.onclose = function() {
                    isConnected = false;
                    connectBtn.textContent = 'Connect';
                    connectBtn.className = 'connect-btn';
                    startBtn.disabled = true;
                    updateStatus(false, false);
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.latency_ms) {
                        updateStatus(true, isRecording, data.latency_ms);
                    }
                };
                
            } else {
                // Disconnect
                if (ws) {
                    ws.close();
                }
                if (isRecording) {
                    stopRecording();
                }
            }
        }

        function toggleRecording() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }

        async function startRecording() {
            try {
                // Request high-quality audio with optimized settings
                const constraints = {
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                };

                audioStream = await navigator.mediaDevices.getUserMedia(constraints);

                // Use optimal MIME type for low latency
                const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm';

                mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType: mimeType,
                    audioBitsPerSecond: 64000  // Optimized bitrate
                });

                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
                        const reader = new FileReader();
                        reader.onload = function() {
                            const audioData = new Uint8Array(reader.result);
                            const base64Audio = btoa(String.fromCharCode.apply(null, audioData));

                            ws.send(JSON.stringify({
                                type: 'audio',
                                audio_data: base64Audio,
                                mode: 'speech_to_speech',
                                timestamp: Date.now()
                            }));
                        };
                        reader.readAsArrayBuffer(event.data);
                    }
                };

                // Start with 50ms chunks for ultra-low latency
                mediaRecorder.start(50);
                isRecording = true;

                const startBtn = document.getElementById('startBtn');
                startBtn.textContent = 'Stop Conversation';
                startBtn.className = 'stop-btn';

                updateStatus(true, true);

            } catch (error) {
                console.error('Error starting recording:', error);
                alert('Could not access microphone. Please check permissions and ensure HTTPS connection.');
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
            }
            
            isRecording = false;
            
            const startBtn = document.getElementById('startBtn');
            startBtn.textContent = 'Start Conversation';
            startBtn.className = 'start-btn';
            
            updateStatus(isConnected, false);
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio processing"""
    await websocket.accept()
    ui_logger.info("WebSocket connection established")
    
    try:
        # Initialize models if needed
        unified_manager = get_unified_manager()
        if not unified_manager.is_initialized:
            await unified_manager.initialize()
        
        pipeline = get_speech_to_speech_pipeline()
        if not pipeline.is_initialized:
            await pipeline.initialize()
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio":
                start_time = time.time()
                
                # Process audio through speech-to-speech pipeline
                audio_data = base64.b64decode(message.get("audio_data", ""))
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Process through pipeline
                result = await pipeline.process_speech_to_speech(audio_array)
                
                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Send response
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "success": result.get("success", False),
                    "text": result.get("text", ""),
                    "audio_data": result.get("audio_data", ""),
                    "latency_ms": latency_ms
                }))
                
    except WebSocketDisconnect:
        ui_logger.info("WebSocket connection closed")
    except Exception as e:
        ui_logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def initialize_models_at_startup():
    """Initialize models at startup for faster response times"""
    ui_logger.info("Initializing models at startup...")
    
    try:
        unified_manager = get_unified_manager()
        await unified_manager.initialize()
        ui_logger.info("Unified model manager initialized")
        
        pipeline = get_speech_to_speech_pipeline()
        await pipeline.initialize()
        ui_logger.info("Speech-to-speech pipeline initialized")
        
    except Exception as e:
        ui_logger.error(f"Error initializing models: {e}")

if __name__ == "__main__":
    ui_logger.info("Starting Voxtral Simplified UI Server")
    
    # Pre-load models before starting server
    import asyncio
    asyncio.run(initialize_models_at_startup())
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.http_port,
        log_level="info"
    )
