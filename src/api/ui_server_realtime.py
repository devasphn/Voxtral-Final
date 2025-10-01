"""
FIXED UI server for CONVERSATIONAL real-time streaming
Improved WebSocket handling and silence detection UI feedback
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import asyncio
import time
import json
import base64
import numpy as np
from pathlib import Path
import logging
import sys
import os
import soundfile as sf  # FIXED: Add missing soundfile import
from io import BytesIO  # FIXED: Add missing BytesIO import

# Add current directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import config
from src.utils.logging_config import logger

# Initialize FastAPI app
app = FastAPI(
    title="Voxtral Conversational Streaming UI",
    description="Web interface for Voxtral CONVERSATIONAL audio streaming with VAD",
    version="2.2.0"
)

# Enhanced logging for real-time streaming
streaming_logger = logging.getLogger("realtime_streaming")
streaming_logger.setLevel(logging.DEBUG)

# Global variables for unified model management
_unified_manager = None
_audio_processor = None
_performance_monitor = None
_speech_to_speech_pipeline = None

# Response deduplication tracking
recent_responses = {}  # client_id -> last_response_text

def get_unified_manager():
    """Get unified model manager instance"""
    global _unified_manager
    if _unified_manager is None:
        from src.models.unified_model_manager import unified_model_manager
        _unified_manager = unified_model_manager
        streaming_logger.info("Unified model manager loaded")
    return _unified_manager

def get_audio_processor():
    """Lazy initialization of Audio processor"""
    global _audio_processor
    if _audio_processor is None:
        from src.models.audio_processor_realtime import AudioProcessor
        _audio_processor = AudioProcessor()
        streaming_logger.info("Audio processor lazy-loaded")
    return _audio_processor

def get_performance_monitor():
    """Get performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        from src.utils.performance_monitor import performance_monitor
        _performance_monitor = performance_monitor
        streaming_logger.info("Performance monitor loaded")
    return _performance_monitor
def get_speech_to_speech_pipeline():
    """Lazy initialization of Speech-to-Speech pipeline"""
    global _speech_to_speech_pipeline
    if _speech_to_speech_pipeline is None:
        from src.models.speech_to_speech_pipeline import speech_to_speech_pipeline
        _speech_to_speech_pipeline = speech_to_speech_pipeline
        streaming_logger.info("Speech-to-Speech pipeline lazy-loaded")
    return _speech_to_speech_pipeline

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve CONVERSATIONAL streaming web interface with VAD feedback"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voxtral Conversational AI with VAD</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
            justify-content: center;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            min-width: 120px;
        }
        .connect-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }
        .stream-btn {
            background: linear-gradient(45deg, #00b894, #00a085);
            color: white;
        }
        .stop-btn {
            background: linear-gradient(45deg, #e17055, #d63031);
            color: white;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #00b894;
        }
        .conversation {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            border-left: 4px solid #74b9ff;
            max-height: 400px;
            overflow-y: auto;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
        }
        .user-message {
            background: rgba(0, 184, 148, 0.3);
            text-align: right;
        }
        .ai-message {
            background: rgba(116, 185, 255, 0.3);
        }
        .silence-message {
            background: rgba(155, 155, 155, 0.3);
            font-style: italic;
            opacity: 0.7;
        }
        .timestamp {
            font-size: 0.8em;
            opacity: 0.7;
        }
        .audio-controls {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
        }
        .connected {
            background: #00b894;
        }
        .disconnected {
            background: #e17055;
        }
        .streaming {
            background: #0984e3;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .vad-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .vad-status {
            padding: 5px 15px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .vad-speech {
            background: #00b894;
            color: white;
        }
        .vad-silence {
            background: #636e72;
            color: white;
        }
        .vad-processing {
            background: #fdcb6e;
            color: #2d3436;
        }
        .realtime-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #e17055;
            border-radius: 50%;
            margin-left: 10px;
        }
        .realtime-indicator.active {
            background: #00b894;
            animation: blink 1s infinite;
        }
        .realtime-indicator.speech {
            background: #fdcb6e;
            animation: pulse-speech 0.5s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        @keyframes pulse-speech {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }
        .volume-meter {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .volume-bar {
            height: 100%;
            background: linear-gradient(45deg, #00b894, #74b9ff);
            width: 0%;
            transition: width 0.1s;
        }
        .volume-bar.speech {
            background: linear-gradient(45deg, #fdcb6e, #e17055);
        }
        select, input {
            padding: 10px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
        }
        .performance-warning {
            background: rgba(241, 196, 15, 0.3);
            border-left: 4px solid #f1c40f;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .vad-stats {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="connection-status disconnected" id="connectionStatus">
        Disconnected
    </div>

    <div class="container">
        <h1>Voxtral Voice Agent</h1>
        <p style="text-align: center; opacity: 0.8;">Ultra-Low Latency Voice AI (&lt;500ms)</p>

        <div class="status" id="status">
            Ready to connect. Click "Connect" to start.
        </div>

        <div class="controls">
            <button id="connectBtn" class="connect-btn" onclick="connect()">Connect</button>
            <button id="streamBtn" class="stream-btn" onclick="startConversation()" disabled>Start</button>
        </div>

        <div class="vad-indicator">
            <strong>Status:</strong>
            <span class="vad-status vad-silence" id="vadStatus">Waiting</span>
        </div>

        <div class="volume-meter">
            <div class="volume-bar" id="volumeBar"></div>
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="latencyMetric">-</div>
                <div class="metric-label">Latency (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="chunksMetric">0</div>
                <div class="metric-label">Processed</div>
            </div>
        </div>
        
        <div class="conversation" id="conversation" style="display: none;">
            <div id="conversationContent">
                <div class="message ai-message">
                    <div><strong>AI:</strong> Ready for voice conversation.</div>
                </div>
            </div>
        </div>
        
        <div id="performanceWarning" class="performance-warning" style="display: none;">
            [WARN] High latency detected. For better performance, try using "Simple Transcription" mode or check your internet connection.
        </div>
    </div>
    
    <script>
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let audioWorkletNode = null;
        let isStreaming = false;
        let wsUrl = '';
        let chunkCounter = 0;
        let streamStartTime = null;
        let latencySum = 0;
        let responseCount = 0;
        let speechChunks = 0;
        let silenceChunks = 0;
        let lastVadUpdate = 0;

        // Enhanced continuous speech buffering variables
        let continuousAudioBuffer = [];
        let speechStartTime = null;
        let lastSpeechTime = null;
        let isSpeechActive = false;
        let silenceStartTime = null;
        let pendingResponse = false;
        let lastResponseText = '';  // For deduplication

        // Streaming mode settings
        let streamingModeEnabled = true;  // Default to streaming mode

        // ULTRA-LOW LATENCY: Audio playback queue management
        let audioQueue = [];
        let isPlayingAudio = false;
        let currentAudio = null;
        const MAX_QUEUE_SIZE = 2;  // CRITICAL: Limit queue size for <500ms latency
        
        // ULTRA-LOW LATENCY: Speech-to-Speech only variables
        let currentMode = 'speech_to_speech';  // FIXED: Default to speech-to-speech mode
        let selectedVoice = 'hf_alpha';  // FIXED: Use Indian female voice as default
        let selectedSpeed = 1.0;
        let currentConversationId = null;
        let speechToSpeechActive = false;

        // ULTRA-LOW LATENCY: Enhanced configuration for continuous speech capture
        const CHUNK_SIZE = 2048;  // OPTIMIZED: Reduced chunk size for lower latency
        const CHUNK_INTERVAL = 50;  // OPTIMIZED: Faster processing interval (was 100ms)
        const SAMPLE_RATE = 16000;
        const LATENCY_WARNING_THRESHOLD = 500;  // OPTIMIZED: Lower threshold for faster warnings
        const SILENCE_THRESHOLD = 0.005;  // OPTIMIZED: Lower threshold for faster detection
        const MIN_SPEECH_DURATION = 200;  // OPTIMIZED: Reduced minimum speech duration (was 500ms)
        const END_OF_SPEECH_SILENCE = 800;  // OPTIMIZED: Faster silence detection (was 1500ms)
        
        function log(message, type = 'info') {
            console.log(`[Voxtral VAD] ${message}`);
        }

        // ULTRA-LOW LATENCY: Enhanced VAD function for continuous speech detection
        function detectSpeechInBuffer(audioData) {
            if (!audioData || audioData.length === 0) return false;

            // OPTIMIZED: Calculate RMS energy with reduced computation
            let sum = 0;
            const step = Math.max(1, Math.floor(audioData.length / 1024)); // Sample every nth element for speed
            for (let i = 0; i < audioData.length; i += step) {
                sum += audioData[i] * audioData[i];
            }
            const rms = Math.sqrt(sum / (audioData.length / step));

            // OPTIMIZED: Calculate max amplitude with sampling
            let maxAmplitude = 0;
            for (let i = 0; i < audioData.length; i += step) {
                const abs = Math.abs(audioData[i]);
                if (abs > maxAmplitude) maxAmplitude = abs;
            }

            // OPTIMIZED: Speech detected with lower thresholds for faster detection
            const hasSpeech = rms > SILENCE_THRESHOLD && maxAmplitude > 0.001;

            return hasSpeech;
        }
        
        // Detect environment and construct WebSocket URL
        function detectEnvironment() {
            const hostname = window.location.hostname;
            const protocol = window.location.protocol;
            const port = window.location.port;

            if (hostname.includes('proxy.runpod.net')) {
                const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
                wsUrl = `${wsProtocol}//${hostname}/ws`;
                document.getElementById('envInfo') && (document.getElementById('envInfo').textContent = 'RunPod Cloud (HTTP Proxy)');
            } else if (hostname === 'localhost' || hostname === '127.0.0.1') {
                const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
                wsUrl = `${wsProtocol}//${hostname}:${port || '8000'}/ws`;
                document.getElementById('envInfo') && (document.getElementById('envInfo').textContent = 'Local Development');
            } else {
                const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
                wsUrl = `${wsProtocol}//${hostname}${port ? ':' + port : ''}/ws`;
                document.getElementById('envInfo') && (document.getElementById('envInfo').textContent = 'Custom Deployment');
            }

            log(`WebSocket URL detected: ${wsUrl}`);
        }
        
        function updateStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.textContent = message;
            
            status.style.borderLeftColor = type === 'error' ? '#e17055' : 
                                           type === 'success' ? '#00b894' : '#74b9ff';
            log(message, type);
        }
        
        function updateConnectionStatus(connected, streaming = false) {
            const status = document.getElementById('connectionStatus');
            const indicator = document.getElementById('realtimeIndicator');
            
            if (streaming) {
                status.textContent = 'Conversing';
                status.className = 'connection-status streaming';
                indicator.classList.add('active');
            } else if (connected) {
                status.textContent = 'Connected';
                status.className = 'connection-status connected';
                indicator.classList.remove('active');
            } else {
                status.textContent = 'Disconnected';
                status.className = 'connection-status disconnected';
                indicator.classList.remove('active');
            }
        }
        
        function updateVadStatus(status, hasSpeech = false) {
            const vadStatus = document.getElementById('vadStatus');
            const indicator = document.getElementById('realtimeIndicator');
            
            if (status === 'speech') {
                vadStatus.textContent = 'Speaking';
                vadStatus.className = 'vad-status vad-speech';
                indicator.classList.add('speech');
                indicator.classList.remove('active');
            } else if (status === 'silence') {
                vadStatus.textContent = 'Silent';
                vadStatus.className = 'vad-status vad-silence';
                indicator.classList.remove('speech');
                if (isStreaming) indicator.classList.add('active');
            } else if (status === 'processing') {
                vadStatus.textContent = 'Processing';
                vadStatus.className = 'vad-status vad-processing';
            } else {
                vadStatus.textContent = 'Waiting';
                vadStatus.className = 'vad-status vad-silence';
                indicator.classList.remove('speech', 'active');
            }
        }
        
        function updateVadStats() {
            document.getElementById('speechChunks').textContent = speechChunks;
            document.getElementById('silenceChunks').textContent = silenceChunks;
            const total = speechChunks + silenceChunks;
            const processingRate = total > 0 ? Math.round((speechChunks / total) * 100) : 0;
            document.getElementById('processingRate').textContent = processingRate;
            document.getElementById('silenceSkipped').textContent = silenceChunks;
        }

        function updateMetrics() {
            // Update latency metrics
            if (responseCount > 0) {
                const avgLatency = Math.round(latencySum / responseCount);
                const latencyElement = document.getElementById('avgLatency');
                if (latencyElement) {
                    latencyElement.textContent = avgLatency;
                }

                const responseCountElement = document.getElementById('responseCount');
                if (responseCountElement) {
                    responseCountElement.textContent = responseCount;
                }
            }
        }

        // Simplified voice settings - use defaults
        function updateMode() {
            currentMode = 'speech_to_speech';
            log('Mode: Ultra-low latency voice conversation');
        }

        function updateVoiceSettings() {
            // Use default settings for simplicity
            selectedVoice = 'hf_alpha';  // Default Hindi female voice
            selectedSpeed = 1.0;         // Default speed
            streamingModeEnabled = true; // Always use streaming mode
            log('Voice settings: Using optimized defaults (Hindi female, normal speed)');

            const selectedOption = streamingSelect.options[streamingSelect.selectedIndex];
            log(`Streaming mode updated: ${selectedOption.text} (${streamingModeEnabled ? 'ENABLED' : 'DISABLED'})`);

            // Update status to reflect mode change
            if (streamingModeEnabled) {
                updateStatus('[INIT] Ultra-low latency streaming mode enabled', 'success');
            } else {
                updateStatus('[EMOJI] Regular conversation mode enabled', 'info');
            }
        }

        // Simplified functions for essential UI operations
        function showProcessingStatus(message) {
            updateStatus(message, 'loading');
        }

        function hideProcessingStatus() {
            updateStatus('Ready', 'success');
        }

        function showAudioPlayback(audioData, sampleRate, voice, speed, duration) {
            const playbackDiv = document.getElementById('audioPlayback');
            const audioElement = document.getElementById('responseAudio');
            const voiceSpan = document.getElementById('voiceUsed');
            const speedSpan = document.getElementById('speedUsed');
            const durationSpan = document.getElementById('audioDuration');

            // Convert base64 audio to blob and create URL
            const audioBytes = Uint8Array.from(atob(audioData), c => c.charCodeAt(0));
            const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);

            audioElement.src = audioUrl;
            voiceSpan.textContent = voice;
            speedSpan.textContent = speed;
            durationSpan.textContent = `${duration.toFixed(1)}s`;

            playbackDiv.style.display = 'block';

            // Auto-play the response
            audioElement.play().catch(e => {
                log('Auto-play failed (user interaction required): ' + e.message);
            });
        }

        function addToSpeechHistory(userText, aiText, conversationId, emotionAnalysis = null) {
            const historyDiv = document.getElementById('speechToSpeechHistory');

            if (userText) {
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                let emotionInfo = '';
                if (emotionAnalysis && emotionAnalysis.user_emotion) {
                    emotionInfo = ` <span style="opacity: 0.7; font-size: 0.8em;">[${emotionAnalysis.user_emotion}]</span>`;
                }
                userMessage.innerHTML = `
                    <div><strong>You:</strong> ${userText}${emotionInfo}</div>
                    <div class="timestamp">${conversationId} - ${new Date().toLocaleTimeString()}</div>
                `;
                historyDiv.appendChild(userMessage);
            }

            if (aiText) {
                const aiMessage = document.createElement('div');
                aiMessage.className = 'message ai-message';
                let emotionInfo = '';
                if (emotionAnalysis && emotionAnalysis.response_emotion) {
                    const score = emotionAnalysis.appropriateness_score || 0;
                    const scoreColor = score >= 0.9 ? '#00b894' : score >= 0.7 ? '#fdcb6e' : '#e17055';
                    emotionInfo = ` <span style="opacity: 0.7; font-size: 0.8em;">[${emotionAnalysis.response_emotion}, score: <span style="color: ${scoreColor}">${(score * 100).toFixed(0)}%</span>]</span>`;
                }
                aiMessage.innerHTML = `
                    <div><strong>AI:</strong> ${aiText}${emotionInfo}</div>
                    <div class="timestamp">Response - ${new Date().toLocaleTimeString()}</div>
                `;
                historyDiv.appendChild(aiMessage);
            }

            // Scroll to bottom
            historyDiv.scrollTop = historyDiv.scrollHeight;
        }

        function showEmotionalAnalysis(analysis) {
            // Create or update emotional analysis display
            let analysisDiv = document.getElementById('emotionalAnalysis');
            if (!analysisDiv) {
                analysisDiv = document.createElement('div');
                analysisDiv.id = 'emotionalAnalysis';
                analysisDiv.style.cssText = `
                    background: rgba(116, 185, 255, 0.1);
                    padding: 10px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    font-size: 0.9em;
                    border-left: 4px solid #74b9ff;
                `;

                const speechConversation = document.getElementById('speechToSpeechConversation');
                const historyDiv = document.getElementById('speechToSpeechHistory');
                speechConversation.insertBefore(analysisDiv, historyDiv);
            }

            const scoreColor = analysis.appropriateness_score >= 0.9 ? '#00b894' :
                              analysis.appropriateness_score >= 0.7 ? '#fdcb6e' : '#e17055';

            analysisDiv.innerHTML = `
                <strong>[MASK] Emotional Analysis:</strong><br>
                <div style="margin-top: 5px;">
                    <span>User: <strong>${analysis.user_emotion}</strong></span> →
                    <span>AI: <strong>${analysis.response_emotion}</strong></span>
                    <span style="color: ${scoreColor}; margin-left: 10px;">
                        (${(analysis.appropriateness_score * 100).toFixed(0)}% appropriate)
                    </span>
                </div>
                <div style="margin-top: 5px; opacity: 0.8; font-style: italic;">
                    ${analysis.emotional_reasoning}
                </div>
                <div style="margin-top: 5px; font-size: 0.8em; opacity: 0.7;">
                    Voice: ${analysis.voice_selected} | Speed: ${analysis.speed_selected.toFixed(1)}x
                </div>
            `;

            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (analysisDiv && analysisDiv.parentNode) {
                    analysisDiv.style.opacity = '0.5';
                }
            }, 10000);
        }

        async function connect() {
            try {
                updateStatus('Connecting to Voxtral conversational AI...', 'loading');
                log('Attempting WebSocket connection...');
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    updateStatus('Connected! Ready to start conversation.', 'success');
                    updateConnectionStatus(true);
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('streamBtn').disabled = false;
                    log('WebSocket connection established');
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                
                ws.onclose = (event) => {
                    updateStatus(`Disconnected from server (Code: ${event.code})`, 'error');
                    updateConnectionStatus(false);
                    updateVadStatus('waiting');
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('streamBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = true;
                    log(`WebSocket connection closed: ${event.code}`);
                };
                
                ws.onerror = (error) => {
                    updateStatus('Connection error - check console for details', 'error');
                    updateConnectionStatus(false);
                    updateVadStatus('waiting');
                    log('WebSocket error occurred');
                    console.error('WebSocket error:', error);
                };
                
            } catch (error) {
                updateStatus('Failed to connect: ' + error.message, 'error');
                log('Connection failed: ' + error.message);
            }
        }
        
        function handleWebSocketMessage(data) {
            log(`Received message type: ${data.type}`);
            
            switch (data.type) {
                case 'connection':
                    updateStatus(data.message, 'success');
                    break;
                    
                case 'response':
                    // Check for response deduplication
                    if (data.text && data.text.trim() !== '' && data.text !== lastResponseText) {
                        displayConversationMessage(data);
                        lastResponseText = data.text;
                        log(`Received unique response: "${data.text.substring(0, 50)}..."`);
                    } else if (data.text === lastResponseText) {
                        log('Duplicate response detected - skipping display');
                    }

                    // Reset pending response flag to allow new speech processing
                    pendingResponse = false;
                    break;
                    
                case 'error':
                    updateStatus('Error: ' + data.message, 'error');
                    break;
                    
                case 'info':
                    updateStatus(data.message, 'loading');
                    break;

                case 'audio_response':
                    // Handle TTS audio response
                    handleAudioResponse(data);
                // Speech-to-Speech message types
                case 'processing':
                    if (speechToSpeechActive) {
                        showProcessingStatus(data.stage, data.message);
                    }
                    break;

                case 'transcription':
                    if (speechToSpeechActive) {
                        showTranscription(data.text, data.conversation_id);
                        log(`Transcription received: "${data.text}"`);
                    }
                    break;

                case 'response_text':
                    if (speechToSpeechActive) {
                        showResponseText(data.text, data.conversation_id);
                        log(`AI response text: "${data.text}"`);
                    }
                    break;

                case 'speech_response':
                    if (speechToSpeechActive) {
                        hideProcessingStatus();
                        showAudioPlayback(
                            data.audio_data,
                            data.sample_rate,
                            data.voice_used,
                            data.speed_used,
                            data.audio_duration_s
                        );
                        log(`Speech response received: ${data.audio_duration_s}s audio`);
                    }
                    break;

                case 'conversation_complete':
                    if (speechToSpeechActive) {
                        hideProcessingStatus();

                        // Add to conversation history with emotional context
                        const transcription = document.getElementById('transcriptionText').textContent;
                        const responseText = document.getElementById('responseText').textContent;

                        if (transcription || responseText) {
                            addToSpeechHistory(transcription, responseText, data.conversation_id, data.emotion_analysis);
                        }

                        // Display emotional analysis if available
                        if (data.emotion_analysis) {
                            showEmotionalAnalysis(data.emotion_analysis);
                        }

                        // Update metrics
                        if (data.total_latency_ms) {
                            latencySum += data.total_latency_ms;
                            responseCount++;
                            updateMetrics();

                            if (data.total_latency_ms > LATENCY_WARNING_THRESHOLD) {
                                document.getElementById('performanceWarning').style.display = 'block';
                            }
                        }

                        log(`Conversation complete: ${data.total_latency_ms}ms (target: ${data.meets_target ? 'met' : 'exceeded'})`);
                        if (data.emotion_analysis) {
                            log(`Emotional context: ${data.emotion_analysis.emotional_reasoning}`);
                        }
                    }
                    break;

                case 'streaming_words':
                    // Handle streaming word-by-word text
                    if (streamingModeEnabled) {
                        handleStreamingWords(data);
                        log(`[INIT] Streaming words: "${data.text}" (sequence: ${data.sequence})`);
                    }
                    break;

                case 'streaming_audio':
                    // Handle streaming audio chunks
                    if (streamingModeEnabled) {
                        handleStreamingAudio(data);
                        log(`[AUDIO] Streaming audio chunk ${data.chunk_index} (final: ${data.is_final})`);
                    }
                    break;

                case 'interruption':
                    // Handle user interruption detection
                    if (streamingModeEnabled) {
                        handleInterruption(data);
                        log(`[EMOJI] User interruption detected: ${data.message}`);
                    }
                    break;

                case 'streaming_complete':
                    // Handle streaming completion - NO TTS for complete response in streaming mode
                    if (streamingModeEnabled) {
                        log(`[STREAMING] Streaming complete - word-by-word TTS already handled: "${data.full_response}"`);
                        // Just update UI status, don't generate additional TTS
                        updateStatus(`[COMPLETE] Response generated (${data.total_words_sent || 'unknown'} words) - Ready for next input`, 'success');

                        // FIXED: Keep speech-to-speech active for continuous conversation
                        if (speechToSpeechActive) {
                            // Reset for next conversation but keep active
                            setTimeout(() => {
                                updateStatus('[READY] Listening for your next message...', 'info');
                                updateVadStatus('listening');
                            }, 2000);  // Brief pause before ready for next input
                        }
                    } else {
                        // Only generate complete TTS if NOT in streaming mode
                        if (data.full_response && data.full_response.trim()) {
                            log(`[STREAMING] Non-streaming mode - generating TTS for full response: "${data.full_response}"`);
                            handleStreamingComplete(data);
                        }
                    }
                    break;

                case 'session_reset_complete':
                    log('[SESSION] Session reset completed successfully');
                    updateStatus('Session reset - ready for new conversation', 'success');
                    break;

                default:
                    log(`Unknown message type: ${data.type}`);
            }
        }

        function handleAudioResponse(data) {
            try {
                log(`[AUDIO] Received TTS audio response for chunk ${data.chunk_id} (${data.audio_data.length} chars)`);

                // ULTRA-LOW LATENCY: Implement queue size limit to prevent backlog
                if (audioQueue.length >= MAX_QUEUE_SIZE) {
                    log(`[AUDIO] Queue full (${audioQueue.length}), dropping oldest chunk`);
                    audioQueue.shift(); // Remove oldest chunk
                }

                // Add to audio queue for sequential playback
                audioQueue.push({
                    chunkId: data.chunk_id,
                    audioData: data.audio_data,
                    metadata: data.metadata || {},
                    voice: data.voice || 'unknown'
                });

                log(`[AUDIO] Added audio to queue. Queue length: ${audioQueue.length}`);

                // Start processing queue if not already playing
                if (!isPlayingAudio) {
                    processAudioQueue();
                }

            } catch (error) {
                log(`[ERROR] Error handling audio response: ${error}`);
                updateStatus('Error processing audio response', 'error');
                console.error('Audio response error:', error);
            }
        }

        async function processAudioQueue() {
            if (isPlayingAudio || audioQueue.length === 0) {
                return;
            }

            isPlayingAudio = true;

            while (audioQueue.length > 0) {
                const audioItem = audioQueue.shift();

                try {
                    log(`[AUDIO] Processing audio chunk ${audioItem.chunkId} from queue`);
                    await playAudioItem(audioItem);
                    log(`[OK] Completed playing audio chunk ${audioItem.chunkId}`);
                } catch (error) {
                    log(`[ERROR] Error playing audio chunk ${audioItem.chunkId}: ${error}`);
                    console.error('Audio playback error:', error);
                }

                // ULTRA-LOW LATENCY: Minimal delay between audio chunks
                await new Promise(resolve => setTimeout(resolve, 10));  // CRITICAL: Reduced to 10ms for ultra-low latency
            }

            isPlayingAudio = false;
            updateStatus('Ready for conversation', 'success');
            log('[AUDIO] Audio queue processing completed');
        }

        function playAudioItem(audioItem) {
            return new Promise((resolve, reject) => {
                try {
                    const { chunkId, audioData, metadata, voice } = audioItem;

                    log(`[AUDIO] Converting base64 audio for chunk ${chunkId} (${audioData.length} chars)`);

                    // FIXED: Convert base64 to proper WAV format with headers
                    const wavBuffer = createWAVFromBase64(audioData);

                    log(`[AUDIO] Created WAV audio buffer: ${wavBuffer.length} bytes`);

                    // Create audio blob with correct WAV MIME type
                    const audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);

                    // Create audio element with enhanced configuration
                    const audio = new Audio();
                    audio.preload = 'auto';
                    audio.volume = 1.0;

                    // Enhanced audio debugging
                    log(`[AUDIO] Audio metadata: ${JSON.stringify(metadata)}`);
                    log(`[AUDIO] Audio blob size: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

                    // Store reference for cleanup
                    currentAudio = audio;

                    // Set up event listeners BEFORE setting src
                    audio.addEventListener('loadstart', () => {
                        log(`[AUDIO] Started loading audio chunk ${chunkId}`);
                    });

                    audio.addEventListener('loadedmetadata', () => {
                        log(`[AUDIO] Audio metadata loaded - Duration: ${audio.duration}s, Sample Rate: ${audio.sampleRate || 'unknown'}Hz`);
                    });

                    audio.addEventListener('canplaythrough', () => {
                        log(`[AUDIO] Audio chunk ${chunkId} ready to play (${metadata.audio_duration_ms || 'unknown'}ms)`);
                        log(`[AUDIO] Browser audio info - Duration: ${audio.duration}s, Buffered: ${audio.buffered.length} ranges`);
                    });

                    audio.addEventListener('play', () => {
                        log(`[AUDIO] Started playing audio chunk ${chunkId} with voice '${voice}'`);
                        log(`[AUDIO] Playback info - Current time: ${audio.currentTime}s, Volume: ${audio.volume}, Playback rate: ${audio.playbackRate}`);
                        updateStatus(`[SPEAKER] Playing AI response (${voice})...`, 'success');
                    });

                    audio.addEventListener('timeupdate', () => {
                        if (audio.currentTime > 0) {
                            log(`[AUDIO] Playing chunk ${chunkId} - Progress: ${audio.currentTime.toFixed(2)}s / ${audio.duration.toFixed(2)}s`);
                        }
                    });

                    audio.addEventListener('ended', () => {
                        log(`[OK] Finished playing audio chunk ${chunkId} - Total duration: ${audio.duration}s`);
                        URL.revokeObjectURL(audioUrl);
                        currentAudio = null;
                        resolve();
                    });

                    audio.addEventListener('error', (e) => {
                        const errorDetails = {
                            code: audio.error?.code,
                            message: audio.error?.message,
                            networkState: audio.networkState,
                            readyState: audio.readyState
                        };
                        log(`[ERROR] Audio playback error for chunk ${chunkId}: ${JSON.stringify(errorDetails)}`);
                        log(`[ERROR] Audio element state - src: ${audio.src.substring(0, 50)}..., duration: ${audio.duration}`);
                        URL.revokeObjectURL(audioUrl);
                        currentAudio = null;
                        reject(new Error(`Audio playback failed: ${JSON.stringify(errorDetails)}`));
                    });

                    audio.addEventListener('abort', () => {
                        log(`[WARN] Audio playback aborted for chunk ${chunkId}`);
                        URL.revokeObjectURL(audioUrl);
                        currentAudio = null;
                        resolve(); // Don't reject on abort, just continue
                    });

                    // Set source and start loading
                    audio.src = audioUrl;

                    // Start playback with retry logic
                    const playWithRetry = async (retries = 3) => {
                        try {
                            await audio.play();
                        } catch (playError) {
                            log(`[WARN] Play attempt failed for chunk ${chunkId}: ${playError.message}`);

                            if (retries > 0 && !playError.message.includes('aborted')) {
                                log(`[EMOJI] Retrying playback for chunk ${chunkId} (${retries} attempts left)`);
                                setTimeout(() => playWithRetry(retries - 1), 200);
                            } else {
                                URL.revokeObjectURL(audioUrl);
                                currentAudio = null;
                                reject(new Error(`Failed to play audio after retries: ${playError.message}`));
                            }
                        }
                    };

                    // Wait for audio to be ready, then play
                    if (audio.readyState >= 3) { // HAVE_FUTURE_DATA
                        playWithRetry();
                    } else {
                        audio.addEventListener('canplay', () => playWithRetry(), { once: true });
                    }

                } catch (error) {
                    log(`[ERROR] Error creating audio for chunk ${audioItem.chunkId}: ${error}`);
                    reject(error);
                }
            });
        }

        // FIXED: Add proper WAV header creation function
        function createWAVFromBase64(base64Data) {
            try {
                // Decode base64 to binary
                const binaryString = atob(base64Data);
                const audioData = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    audioData[i] = binaryString.charCodeAt(i);
                }

                // Check if data already has WAV header (RIFF signature)
                if (audioData.length >= 12 &&
                    audioData[0] === 0x52 && audioData[1] === 0x49 &&
                    audioData[2] === 0x46 && audioData[3] === 0x46) {
                    // Already has WAV header
                    log(`[AUDIO] Audio data already has WAV header`);
                    return audioData;
                }

                // Create WAV header for raw PCM data (16kHz, 16-bit, mono)
                const sampleRate = 16000;
                const numChannels = 1;
                const bitsPerSample = 16;
                const byteRate = sampleRate * numChannels * bitsPerSample / 8;
                const blockAlign = numChannels * bitsPerSample / 8;
                const dataSize = audioData.length;
                const fileSize = 36 + dataSize;

                const wavHeader = new ArrayBuffer(44);
                const view = new DataView(wavHeader);

                // RIFF header
                view.setUint32(0, 0x52494646, false); // "RIFF"
                view.setUint32(4, fileSize, true);    // File size
                view.setUint32(8, 0x57415645, false); // "WAVE"

                // fmt chunk
                view.setUint32(12, 0x666d7420, false); // "fmt "
                view.setUint32(16, 16, true);          // Chunk size
                view.setUint16(20, 1, true);           // Audio format (PCM)
                view.setUint16(22, numChannels, true); // Number of channels
                view.setUint32(24, sampleRate, true);  // Sample rate
                view.setUint32(28, byteRate, true);    // Byte rate
                view.setUint16(32, blockAlign, true);  // Block align
                view.setUint16(34, bitsPerSample, true); // Bits per sample

                // data chunk
                view.setUint32(36, 0x64617461, false); // "data"
                view.setUint32(40, dataSize, true);    // Data size

                // Combine header and data
                const wavFile = new Uint8Array(44 + dataSize);
                wavFile.set(new Uint8Array(wavHeader), 0);
                wavFile.set(audioData, 44);

                log(`[AUDIO] Created WAV file with header: ${wavFile.length} bytes total (${dataSize} bytes audio + 44 bytes header)`);
                return wavFile;

            } catch (error) {
                log(`[ERROR] Error creating WAV from base64: ${error}`);
                // Fallback: return original data
                const binaryString = atob(base64Data);
                const audioData = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    audioData[i] = binaryString.charCodeAt(i);
                }
                return audioData;
            }
        }

        function handleStreamingComplete(data) {
            try {
                log(`[STREAMING] Processing streaming completion for chunk ${data.chunk_id}`);
                log(`[STREAMING] Full response (${data.total_words_sent} words): "${data.full_response}"`);

                // Send request to server to generate TTS for the complete response
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const ttsRequest = {
                        type: "generate_tts",
                        text: data.full_response,
                        chunk_id: data.chunk_id,
                        voice: selectedVoice,
                        speed: selectedSpeed,
                        timestamp: Date.now() / 1000
                    };

                    log(`[TTS] Requesting TTS generation for complete response with voice '${selectedVoice}'`);
                    ws.send(JSON.stringify(ttsRequest));
                } else {
                    log(`[ERROR] WebSocket not available for TTS request`);
                }

                // Update UI to show completion
                updateStatus(`[COMPLETE] Response generated (${data.total_words_sent} words, ${data.voxtral_time_ms}ms)`, 'success');

                // Update metrics
                if (data.voxtral_time_ms) {
                    latencySum += data.voxtral_time_ms;
                    responseCount++;
                    updateMetrics();

                    if (data.voxtral_time_ms > LATENCY_WARNING_THRESHOLD) {
                        document.getElementById('performanceWarning').style.display = 'block';
                    }
                }

            } catch (error) {
                log(`[ERROR] Error handling streaming completion: ${error}`);
                updateStatus('Error processing streaming completion', 'error');
            }
        }

        function displayConversationMessage(data) {
            const conversationDiv = document.getElementById('conversation');
            const contentDiv = document.getElementById('conversationContent');
            conversationDiv.style.display = 'block';
            document.getElementById('vadStats').style.display = 'block';
            
            const timestamp = new Date().toLocaleTimeString();
            
            // Check if this was a silence response (empty or skipped)
            if (!data.text || data.text.trim() === '' || data.skipped_reason === 'no_speech_detected') {
                silenceChunks++;
                updateVadStatus('silence');
                
                // Optionally show silence messages (uncomment if desired)
                /*
                const silenceMessage = document.createElement('div');
                silenceMessage.className = 'message silence-message';
                silenceMessage.innerHTML = `
                    <div><em>[MUTE] Silence detected - no response needed</em></div>
                    <div class="timestamp">${timestamp} (${data.processing_time_ms}ms)</div>
                `;
                contentDiv.appendChild(silenceMessage);
                */
            } else {
                // This was a real speech response
                speechChunks++;
                responseCount++;
                updateVadStatus('processing');
                
                const aiMessage = document.createElement('div');
                aiMessage.className = 'message ai-message';
                aiMessage.innerHTML = `
                    <div><strong>AI:</strong> ${data.text}</div>
                    <div class="timestamp">${timestamp} (${data.processing_time_ms}ms)</div>
                `;
                
                contentDiv.appendChild(aiMessage);
                conversationDiv.scrollTop = conversationDiv.scrollHeight;
                
                // Update metrics
                if (data.processing_time_ms) {
                    latencySum += data.processing_time_ms;
                    const avgLatency = Math.round(latencySum / responseCount);
                    document.getElementById('latencyMetric').textContent = avgLatency;
                    
                    // Show performance warning if latency is high
                    const warningDiv = document.getElementById('performanceWarning');
                    if (data.processing_time_ms > LATENCY_WARNING_THRESHOLD) {
                        warningDiv.style.display = 'block';
                    } else if (avgLatency < LATENCY_WARNING_THRESHOLD) {
                        warningDiv.style.display = 'none';
                    }
                }
                
                setTimeout(() => updateVadStatus('silence'), 2000);
                
                log(`AI Response: "${data.text}" (${data.processing_time_ms}ms)`);
            }
            
            document.getElementById('chunksMetric').textContent = speechChunks;
            updateVadStats();
        }
        
        function updateStreamDuration() {
            if (streamStartTime && isStreaming) {
                const duration = Math.floor((Date.now() - streamStartTime) / 1000);
                const minutes = Math.floor(duration / 60).toString().padStart(2, '0');
                const seconds = (duration % 60).toString().padStart(2, '0');
                document.getElementById('durationMetric').textContent = `${minutes}:${seconds}`;
            }
        }
        
        async function startConversation() {
            try {
                log('Starting conversational audio streaming with VAD...');
                updateStatus('Initializing microphone for conversation...', 'loading');
                
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: SAMPLE_RATE,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
                
                log('Microphone access granted');
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: SAMPLE_RATE
                });
                
                await audioContext.resume();
                log(`Audio context created with sample rate: ${audioContext.sampleRate}`);
                
                // FIXED: Use modern AudioWorklet instead of deprecated ScriptProcessor
                try {
                    // Try to use AudioWorklet (modern approach)
                    await audioContext.audioWorklet.addModule('data:text/javascript;base64,' + btoa(`
                        class AudioProcessor extends AudioWorkletProcessor {
                            process(inputs, outputs, parameters) {
                                const input = inputs[0];
                                if (input.length > 0) {
                                    const inputData = input[0];
                                    this.port.postMessage({
                                        type: 'audiodata',
                                        data: inputData
                                    });
                                }
                                return true;
                            }
                        }
                        registerProcessor('audio-processor', AudioProcessor);
                    `));

                    audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    source.connect(audioWorkletNode);

                    let audioBuffer = [];
                    let lastChunkTime = Date.now();

                    audioWorkletNode.port.onmessage = (event) => {
                        if (event.data.type === 'audiodata') {
                            const inputData = event.data.data;
                            processAudioData(inputData);
                        }
                    };

                    log('[AUDIO] Using modern AudioWorklet for audio processing');

                } catch (workletError) {
                    // Fallback to ScriptProcessor for older browsers
                    log(`[WARN] AudioWorklet not supported, falling back to ScriptProcessor: ${workletError.message}`);

                    audioWorkletNode = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);
                    const source = audioContext.createMediaStreamSource(mediaStream);

                    source.connect(audioWorkletNode);
                    audioWorkletNode.connect(audioContext.destination);

                    let audioBuffer = [];
                    let lastChunkTime = Date.now();

                    audioWorkletNode.onaudioprocess = (event) => {
                        const inputBuffer = event.inputBuffer;
                        const inputData = inputBuffer.getChannelData(0);
                        processAudioData(inputData);
                    };

                    log('[AUDIO] Using fallback ScriptProcessor for audio processing');
                }

                function processAudioData(inputData) {
                    if (!isStreaming || pendingResponse) return;

                    // Update volume meter and VAD indicator
                    updateVolumeMeter(inputData);

                    // Add to continuous buffer
                    continuousAudioBuffer.push(...inputData);

                    // Detect speech in current chunk
                    const hasSpeech = detectSpeechInBuffer(inputData);
                    const now = Date.now();

                    if (hasSpeech) {
                        if (!isSpeechActive) {
                            // Speech started
                            speechStartTime = now;
                            isSpeechActive = true;
                            silenceStartTime = null;
                            log('Speech detected - starting continuous capture');
                            updateVadStatus('speech');
                        }
                        lastSpeechTime = now;
                    } else {
                        if (isSpeechActive && !silenceStartTime) {
                            // Silence started after speech
                            silenceStartTime = now;
                            updateVadStatus('silence');
                        }
                    }

                    // Check if we should process accumulated speech
                    if (isSpeechActive && silenceStartTime &&
                        (now - silenceStartTime >= END_OF_SPEECH_SILENCE) &&
                        (lastSpeechTime - speechStartTime >= MIN_SPEECH_DURATION)) {

                        // Process the complete utterance
                        log(`Processing complete utterance: ${continuousAudioBuffer.length} samples, ${(lastSpeechTime - speechStartTime)}ms duration`);
                        sendCompleteUtterance(new Float32Array(continuousAudioBuffer));

                        // Reset for next utterance
                        continuousAudioBuffer = [];
                        isSpeechActive = false;
                        speechStartTime = null;
                        lastSpeechTime = null;
                        silenceStartTime = null;
                        pendingResponse = true;  // Prevent processing until response received
                    }

                    // ULTRA-LOW LATENCY: Prevent buffer from growing too large (max 5 seconds for lower latency)
                    const maxBufferSize = SAMPLE_RATE * 5;  // OPTIMIZED: Reduced from 30 to 5 seconds
                    if (continuousAudioBuffer.length > maxBufferSize) {
                        continuousAudioBuffer = continuousAudioBuffer.slice(-maxBufferSize);
                        log('Audio buffer trimmed to prevent memory overflow (5s max)');
                    }
                };
                
                isStreaming = true;
                streamStartTime = Date.now();
                
                document.getElementById('streamBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                updateConnectionStatus(true, true);
                updateStatus('[VAD] Conversation active with VAD - speak naturally!', 'success');
                updateVadStatus('silence');
                
                setInterval(updateStreamDuration, 1000);
                
                log('Conversational streaming with VAD started successfully');
                
            } catch (error) {
                updateStatus('Failed to start conversation: ' + error.message, 'error');
                log('Conversation start failed: ' + error.message);
                console.error('Conversation error:', error);
            }
        }
        
        function stopConversation() {
            isStreaming = false;

            log('Stopping conversational streaming...');

            // Stop and clear audio playback
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }

            // Clear audio queue
            audioQueue = [];
            isPlayingAudio = false;
            log('[AUDIO] Audio queue cleared');

            // FIXED: Send session reset to server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: "reset_session",
                    timestamp: Date.now() / 1000
                }));
                log('[SESSION] Session reset requested');
            }
            
            if (audioWorkletNode) {
                audioWorkletNode.disconnect();
                audioWorkletNode = null;
            }
            
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            document.getElementById('streamBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('speechToSpeechBtn').disabled = false;
            updateConnectionStatus(true, false);
            updateStatus('Conversation ended. Ready to start a new conversation.', 'info');
            updateVadStatus('waiting');

            // Reset speech-to-speech mode
            if (speechToSpeechActive) {
                speechToSpeechActive = false;
                currentConversationId = null;

                // Hide speech-to-speech displays
                hideProcessingStatus();
                document.getElementById('currentTranscription').style.display = 'none';
                document.getElementById('currentResponse').style.display = 'none';
                document.getElementById('audioPlayback').style.display = 'none';

                // Show regular conversation area
                document.getElementById('speechToSpeechConversation').style.display = 'none';
                document.getElementById('conversation').style.display = 'block';

                log('Speech-to-Speech mode deactivated');
            }
            
            document.getElementById('volumeBar').style.width = '0%';
            document.getElementById('volumeBar').classList.remove('speech');
            
            log('Conversational streaming stopped');
        }
        
        function updateVolumeMeter(audioData) {
            let sum = 0;
            for (let i = 0; i < audioData.length; i++) {
                sum += audioData[i] * audioData[i];
            }
            const rms = Math.sqrt(sum / audioData.length);
            const volume = Math.min(100, rms * 100 * 10);
            
            const volumeBar = document.getElementById('volumeBar');
            volumeBar.style.width = volume + '%';
            
            // Simple client-side VAD indication based on volume
            const now = Date.now();
            if (volume > 5 && now - lastVadUpdate > 100) { // Throttle updates
                updateVadStatus('speech');
                volumeBar.classList.add('speech');
                lastVadUpdate = now;
                
                // Reset to silence after 500ms of no update
                setTimeout(() => {
                    if (Date.now() - lastVadUpdate >= 400) {
                        updateVadStatus('silence');
                        volumeBar.classList.remove('speech');
                    }
                }, 500);
            }
        }
        
        function sendCompleteUtterance(audioData) {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                log('Cannot send audio - WebSocket not connected');
                return;
            }

            try {
                const base64Audio = arrayBufferToBase64(audioData.buffer);

                const message = {
                    type: 'audio_chunk',
                    audio_data: base64Audio,
                    mode: speechToSpeechActive ? 'speech_to_speech' : (streamingModeEnabled ? 'streaming' : 'conversation'),
                    streaming: streamingModeEnabled,  // Use user-selected streaming mode
                    prompt: '',  // No custom prompts - using hardcoded optimal prompt
                    chunk_id: chunkCounter++,
                    timestamp: Date.now()
                };

                // Add speech-to-speech specific parameters
                if (speechToSpeechActive) {
                    message.conversation_id = currentConversationId;
                    message.voice = selectedVoice === 'auto' ? null : selectedVoice;  // null for auto-selection
                    message.speed = selectedSpeed;
                }
                
                ws.send(JSON.stringify(message));
                log(`Sent audio chunk ${chunkCounter} (${audioData.length} samples)`);
                
            } catch (error) {
                log('Error sending audio chunk: ' + error.message);
                console.error('Audio chunk error:', error);
            }
        }
        
        function arrayBufferToBase64(buffer) {
            const bytes = new Uint8Array(buffer);
            let binary = '';
            const chunkSize = 8192;

            for (let i = 0; i < bytes.length; i += chunkSize) {
                const chunk = bytes.slice(i, i + chunkSize);
                binary += String.fromCharCode.apply(null, chunk);
            }

            return btoa(binary);
        }

        // Streaming mode handlers
        function handleStreamingWords(data) {
            // Display words as they arrive in real-time
            const conversationDiv = document.getElementById('conversation');
            let currentResponseDiv = document.getElementById('current-streaming-response');

            if (!currentResponseDiv) {
                // Create new response container for streaming
                currentResponseDiv = document.createElement('div');
                currentResponseDiv.id = 'current-streaming-response';
                currentResponseDiv.className = 'message ai-message streaming';
                currentResponseDiv.innerHTML = '<strong>AI:</strong> <span class="streaming-text"></span>';
                conversationDiv.appendChild(currentResponseDiv);
            }

            const textSpan = currentResponseDiv.querySelector('.streaming-text');
            textSpan.textContent = data.full_text_so_far;

            // Auto-scroll to bottom
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        function handleStreamingAudio(data) {
            // Handle streaming audio chunks for immediate playback
            try {
                const audioBytes = Uint8Array.from(atob(data.audio_data), c => c.charCodeAt(0));
                const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);

                // Create and play audio immediately
                const audio = new Audio(audioUrl);
                audio.play().catch(e => {
                    log(`Error playing streaming audio: ${e.message}`);
                });

                log(`[AUDIO] Playing streaming audio chunk ${data.chunk_index}`);

                // Clean up URL after playback
                audio.addEventListener('ended', () => {
                    URL.revokeObjectURL(audioUrl);
                });

            } catch (error) {
                log(`Error handling streaming audio: ${error.message}`);
            }
        }

        function handleInterruption(data) {
            // Handle user interruption - stop current audio and clear streaming
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }

            // Clear current streaming response
            const currentResponseDiv = document.getElementById('current-streaming-response');
            if (currentResponseDiv) {
                currentResponseDiv.remove();
            }

            // Show interruption message
            updateStatus('[EMOJI] Interruption detected - ready for new input', 'info');
            log('[EMOJI] User interruption handled - cleared streaming state');
        }

        // Initialize on page load
        window.addEventListener('load', () => {
            detectEnvironment();
            updateStatus('Ready to connect for conversation with VAD');
            log('Conversational application with VAD initialized');
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (isStreaming) {
                stopConversation();
            }
            if (ws) {
                ws.close();
            }
        });

        // Initialize the interface
        updateMode();
        updateVoiceSettings();
        updateStreamingMode();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def api_status():
    """API endpoint for unified model system status"""
    try:
        unified_manager = get_unified_manager()
        performance_monitor = get_performance_monitor()
        
        # Get comprehensive system status
        model_info = unified_manager.get_model_info()
        memory_stats = unified_manager.get_memory_stats()
        performance_summary = performance_monitor.get_performance_summary()
        
        # Determine overall health status
        is_healthy = (
            unified_manager.is_initialized and
            model_info['unified_manager']['voxtral_initialized'] and
            model_info['unified_manager']['kokoro_initialized']
        )
        
        # Get Voxtral model info through unified manager
        try:
            voxtral_model = await unified_manager.get_voxtral_model()
            voxtral_model_info = voxtral_model.get_model_info()
        except Exception as e:
            voxtral_model_info = {"error": str(e), "status": "not_available"}

        # Get speech-to-speech pipeline info if enabled
        speech_to_speech_info = None
        if config.speech_to_speech.enabled:
            try:
                pipeline = get_speech_to_speech_pipeline()
                speech_to_speech_info = pipeline.get_pipeline_info()
            except Exception as e:
                speech_to_speech_info = {"error": str(e), "enabled": False}

        return JSONResponse({
            "status": "healthy" if is_healthy else "initializing",
            "timestamp": time.time(),
            "unified_system": {
                "initialized": unified_manager.is_initialized,
                "voxtral_ready": model_info['unified_manager']['voxtral_initialized'],
                "kokoro_ready": model_info['unified_manager']['kokoro_initialized'],
                "memory_manager_ready": model_info['unified_manager']['memory_manager_initialized']
            },
            "memory_stats": memory_stats.get("memory_stats", {}),
            "performance_stats": {
                "total_operations": performance_summary["statistics"]["total_operations"],
                "average_latency_ms": performance_summary["statistics"]["average_latency_ms"],
                "operations_within_target": performance_summary["statistics"]["operations_within_target"]
            },
            "voxtral_model": voxtral_model_info,
            "speech_to_speech": speech_to_speech_info,
            "config": {
                "sample_rate": config.audio.sample_rate,
                "tcp_ports": config.server.tcp_ports,
                "latency_target": config.streaming.latency_target_ms,
                "mode": "conversational_optimized_with_kokoro_tts",
                "integration_type": "kokoro_tts",
                "speech_to_speech_enabled": config.speech_to_speech.enabled,
                "speech_to_speech_latency_target": config.speech_to_speech.latency_target_ms,
                "mode": "conversational_optimized_with_vad_and_speech_to_speech" if config.speech_to_speech.enabled else "conversational_optimized_with_vad"
            }
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "timestamp": time.time(),
            "error": str(e),
            "integration_type": "kokoro_tts"
        }, status_code=500)

# WebSocket endpoint for CONVERSATIONAL streaming with VAD
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for CONVERSATIONAL audio streaming with VAD"""
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    streaming_logger.info(f"[CONVERSATION] Client connected: {client_id}")
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Connected to Voxtral conversational AI with VAD",
            "server_config": {
                "sample_rate": config.audio.sample_rate,
                "chunk_size": config.audio.chunk_size,
                "latency_target": config.streaming.latency_target_ms,
                "streaming_mode": "conversational_optimized_with_vad",
                "vad_enabled": True
            }
        }))
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=3600.0)  # FIXED: 1 hour timeout for continuous conversation
                message = json.loads(data)
                msg_type = message.get("type")

                streaming_logger.debug(f"[CONVERSATION] Received message type: {msg_type} from {client_id}")

                if msg_type == "audio_chunk":
                    await handle_conversational_audio_chunk(websocket, message, client_id)

                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                    
                elif msg_type == "status":
                    unified_manager = get_unified_manager()
                    model_info = unified_manager.get_model_info()
                    performance_monitor = get_performance_monitor()
                    performance_summary = performance_monitor.get_performance_summary()

                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "model_info": model_info,
                        "performance_summary": performance_summary
                    }))

                elif msg_type == "generate_tts":
                    await handle_tts_generation_request(websocket, message, client_id)

                elif msg_type == "text_input":
                    await handle_text_input_direct(websocket, message, client_id)

                elif msg_type == "reset_session":
                    # FIXED: Handle session reset for clean state between interactions
                    await handle_session_reset(websocket, client_id)

                else:
                    streaming_logger.warning(f"[CONVERSATION] Unknown message type: {msg_type}")
                    
            except asyncio.TimeoutError:
                streaming_logger.info(f"[CONVERSATION] Client {client_id} timeout after 1 hour - sending ping")
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except:
                    break
                    
    except WebSocketDisconnect:
        streaming_logger.info(f"[CONVERSATION] Client disconnected: {client_id}")
        # FIXED: Cleanup client state on disconnect
        if client_id in recent_responses:
            del recent_responses[client_id]
    except ConnectionResetError:
        streaming_logger.info(f"[CONVERSATION] Client connection reset: {client_id}")
        # FIXED: Cleanup client state on connection reset
        if client_id in recent_responses:
            del recent_responses[client_id]
    except Exception as e:
        streaming_logger.error(f"[CONVERSATION] WebSocket error for {client_id}: {e}")
        # FIXED: Cleanup client state on error
        if client_id in recent_responses:
            del recent_responses[client_id]
        # FIXED: Attempt to send error message if connection is still open
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Connection error occurred",
                "timestamp": time.time()
            }))
        except:
            # Connection is already closed, ignore
            pass

async def detect_user_interruption(audio_chunk: np.ndarray, current_state: str = "idle") -> bool:
    """
    INTERRUPTION DETECTION: Detect when user starts speaking during TTS playback
    Uses enhanced VAD with state-aware thresholds
    """
    try:
        # Only check for interruption during speaking state
        if current_state != "speaking":
            return False

        # Enhanced VAD for interruption detection
        energy = np.mean(audio_chunk ** 2)

        # Lower threshold during speaking to catch interruptions quickly
        interruption_threshold = 0.003  # More sensitive than regular VAD

        # Quick spectral check for human speech characteristics
        if len(audio_chunk) >= 512:
            fft = np.fft.rfft(audio_chunk[:512])
            # Focus on human speech frequency range (300-3400 Hz)
            speech_band_start = int(300 * len(fft) / (16000 / 2))
            speech_band_end = int(3400 * len(fft) / (16000 / 2))
            speech_energy = np.mean(np.abs(fft[speech_band_start:speech_band_end]))

            # Interruption detected if both energy and speech characteristics present
            is_interruption = (energy > interruption_threshold) and (speech_energy > 0.01)
        else:
            is_interruption = energy > interruption_threshold

        return is_interruption

    except Exception as e:
        logger.error(f"[ERROR] Interruption detection error: {e}")
        return False

async def handle_session_reset(websocket: WebSocket, client_id: str):
    """Handle session reset to clean state between interactions"""
    try:
        streaming_logger.info(f"[SESSION-RESET] Resetting session for client {client_id}")

        # Clear client response history
        if client_id in recent_responses:
            del recent_responses[client_id]

        # Reset streaming coordinator state
        from src.streaming.streaming_coordinator import streaming_coordinator
        await streaming_coordinator.reset_session()

        # Clear any pending audio processing
        audio_processor = get_audio_processor()
        if hasattr(audio_processor, 'reset_state'):
            audio_processor.reset_state()

        # Send confirmation
        await websocket.send_text(json.dumps({
            "type": "session_reset_complete",
            "message": "Session reset successfully",
            "timestamp": time.time()
        }))

        streaming_logger.info(f"[SESSION-RESET] Session reset completed for client {client_id}")

    except Exception as e:
        streaming_logger.error(f"[SESSION-RESET] Error resetting session for {client_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Session reset failed: {str(e)}"
        }))

async def handle_text_input_direct(websocket: WebSocket, data: dict, client_id: str):
    """Handle direct text input for testing Voxtral model without audio processing"""
    try:
        text_input = data.get("text", "").strip()
        mode = data.get("mode", "conversation")
        streaming = data.get("streaming", True)
        prompt = data.get("prompt", "")

        if not text_input:
            streaming_logger.warning(f"[TEXT-INPUT] Empty text received from {client_id}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Empty text input"
            }))
            return

        streaming_logger.info(f"[TEXT-INPUT] Processing direct text input from {client_id}: '{text_input}'")

        # Get unified manager
        unified_manager = get_unified_manager()
        voxtral_model = unified_manager.voxtral_model

        if not voxtral_model:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Voxtral model not available"
            }))
            return

        # Process text directly with Voxtral model (simplified for testing)
        result = await voxtral_model.process_text_direct(
            text_input,
            mode=mode,
            prompt=prompt,
            streaming=False
        )

        if result and result.get("response"):
            response_text = result["response"]
            streaming_logger.info(f"[TEXT-INPUT] Generated response: '{response_text}'")

            if streaming:
                # Simulate streaming by sending words one by one
                words = response_text.split()
                for word in words:
                    await websocket.send_text(json.dumps({
                        "type": "streaming_chunk",
                        "text": word,
                        "timestamp": time.time()
                    }))
                    await asyncio.sleep(0.1)  # Small delay between words

            # Send final response
            await websocket.send_text(json.dumps({
                "type": "response",
                "text": response_text,
                "timestamp": time.time()
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No response generated"
            }))

    except Exception as e:
        streaming_logger.error(f"[TEXT-INPUT] Error handling text input from {client_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Text input processing error: {str(e)}"
        }))

async def handle_tts_generation_request(websocket: WebSocket, data: dict, client_id: str):
    """Handle TTS generation request for streaming complete responses"""
    try:
        text = data.get("text", "").strip()
        chunk_id = data.get("chunk_id", "unknown")
        voice = data.get("voice", "hf_alpha")
        speed = data.get("speed", 1.0)

        if not text:
            streaming_logger.warning(f"[TTS] Empty text received for TTS generation from {client_id}")
            return

        streaming_logger.info(f"[TTS] Generating speech for chunk {chunk_id}: '{text[:50]}...' with voice '{voice}'")

        # Get unified manager and performance monitor
        unified_manager = get_unified_manager()
        performance_monitor = get_performance_monitor()

        if not unified_manager.is_initialized:
            streaming_logger.error(f"[TTS] Unified model manager not initialized!")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "TTS service not available - model not initialized",
                "chunk_id": chunk_id
            }))
            return

        # Get Kokoro TTS model
        kokoro_model = await unified_manager.get_kokoro_model()

        # Start TTS timing
        tts_timing_id = performance_monitor.start_timing("kokoro_generation", {
            "chunk_id": chunk_id,
            "text_length": len(text),
            "voice": voice
        })

        try:
            # Generate speech using Kokoro TTS model
            result = await kokoro_model.synthesize_speech(
                text=text,
                voice=voice
            )

            if not result.get("success", False):
                raise Exception(f"Kokoro TTS generation failed: {result.get('error', 'Unknown error')}")

            audio_data = result["audio_data"]
            sample_rate = result.get("sample_rate", 24000)

            # End TTS timing
            tts_generation_time = performance_monitor.end_timing(tts_timing_id)

            if audio_data is not None and len(audio_data) > 0:
                # Audio quality validation and normalization
                audio_rms = np.sqrt(np.mean(audio_data**2))
                audio_peak = np.max(np.abs(audio_data))

                streaming_logger.info(f"[AUDIO] Audio quality check - RMS: {audio_rms:.6f}, Peak: {audio_peak:.6f}")

                # Normalize audio if needed
                normalized_audio = audio_data
                if audio_rms < 0.05:  # Too quiet
                    target_rms = 0.2
                    gain = target_rms / (audio_rms + 1e-8)
                    normalized_audio = audio_data * gain
                    streaming_logger.info(f"[SPEAKER] Audio boosted by {gain:.2f}x (was too quiet)")
                elif audio_peak > 0.95:  # Risk of clipping
                    gain = 0.9 / audio_peak
                    normalized_audio = audio_data * gain
                    streaming_logger.info(f"[EMOJI] Audio reduced by {gain:.2f}x (preventing clipping)")

                # Convert to WAV format
                wav_buffer = BytesIO()
                sf.write(wav_buffer, normalized_audio, sample_rate, format='WAV', subtype='PCM_16')
                wav_bytes = wav_buffer.getvalue()
                wav_buffer.close()

                # Validate WAV file creation
                if len(wav_bytes) < 100:
                    raise Exception(f"WAV file too small: {len(wav_bytes)} bytes")

                if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
                    raise Exception("Invalid WAV file headers")

                streaming_logger.info(f"[OK] WAV file created: {len(wav_bytes)} bytes with proper headers")

                # Convert to base64 for transmission
                audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

                # Calculate audio duration
                audio_duration_ms = (len(audio_data) / sample_rate) * 1000

                # Send audio response
                await websocket.send_text(json.dumps({
                    "type": "audio_response",
                    "audio_data": audio_b64,
                    "chunk_id": chunk_id,
                    "voice": voice,
                    "format": "wav",
                    "conversation_ready": True,
                    "metadata": {
                        "audio_duration_ms": audio_duration_ms,
                        "generation_time_ms": tts_generation_time,
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "format": "WAV",
                        "subtype": "PCM_16"
                    }
                }))

                streaming_logger.info(f"[TTS-KOKORO] Audio response generated for chunk {chunk_id} in {tts_generation_time:.1f}ms")

            else:
                raise Exception("No audio data generated")

        except Exception as tts_error:
            performance_monitor.end_timing(tts_timing_id)
            streaming_logger.error(f"[ERROR] TTS generation failed for chunk {chunk_id}: {tts_error}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"TTS generation failed: {str(tts_error)}",
                "chunk_id": chunk_id
            }))

    except Exception as e:
        streaming_logger.error(f"[ERROR] Error handling TTS generation request from {client_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "TTS request processing failed",
            "chunk_id": data.get("chunk_id", "unknown")
        }))

async def handle_conversational_audio_chunk(websocket: WebSocket, data: dict, client_id: str):
    """Process conversational audio chunks with VAD using unified model manager"""
    try:
        chunk_start_time = time.time()
        chunk_id = data.get("chunk_id", 0)

        streaming_logger.info(f"[CONVERSATION] Processing chunk {chunk_id} for {client_id}")

        # Get services from unified manager
        unified_manager = get_unified_manager()
        audio_processor = get_audio_processor()
        performance_monitor = get_performance_monitor()

        # Check if unified manager is initialized
        if not unified_manager.is_initialized:
            streaming_logger.error(f"[CONVERSATION] Unified model manager not initialized!")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Models not properly initialized. Please restart the server."
            }))
            return
        
        # Get models from unified manager
        try:
            voxtral_model = await unified_manager.get_voxtral_model()
            kokoro_model = await unified_manager.get_kokoro_model()
        except Exception as e:
            streaming_logger.error(f"[CONVERSATION] Failed to get models: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to access models: {str(e)}"
            }))
            return
        
        audio_b64 = data.get("audio_data")
        if not audio_b64:
            return
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            # FIXED: Enhanced audio validation
            if len(audio_array) == 0:
                streaming_logger.warning(f"[CONVERSATION] Empty audio array for chunk {chunk_id}")
                return

            if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                streaming_logger.error(f"[CONVERSATION] Invalid audio data (NaN/Inf) for chunk {chunk_id}")
                return

            if np.max(np.abs(audio_array)) == 0:
                streaming_logger.debug(f"[CONVERSATION] Silent audio chunk {chunk_id} - skipping")
                return

        except Exception as e:
            streaming_logger.error(f"[CONVERSATION] Audio decoding error for chunk {chunk_id}: {e}")
            return
        
        if not audio_processor.validate_realtime_chunk(audio_array, chunk_id):
            return
        
        try:
            audio_tensor = audio_processor.preprocess_realtime_chunk(audio_array, chunk_id)
        except Exception as e:
            streaming_logger.error(f"[CONVERSATION] Audio preprocessing error for chunk {chunk_id}: {e}")
            return
        
        # Smart Conversation Mode - unified processing with performance monitoring
        mode = data.get("mode", "conversation")  # FIXED: Respect client-specified mode
        prompt = data.get("prompt", "")  # FIXED: Allow custom prompts
        
        # Start performance timing
        voxtral_timing_id = performance_monitor.start_timing("voxtral_processing", {
            "chunk_id": chunk_id,
            "client_id": client_id,
            "audio_length": len(audio_array)
        })
        
        # Check if streaming mode is requested
        streaming_mode = mode == "streaming" or data.get("streaming", False)

        try:
            if streaming_mode:
                # STREAMING MODE: Process with token-by-token streaming
                streaming_logger.info(f"[VAD] Starting streaming processing for chunk {chunk_id}")

                # Import streaming coordinator
                from src.streaming.streaming_coordinator import streaming_coordinator

                # Start streaming session
                session_id = await streaming_coordinator.start_streaming_session(f"{client_id}_{chunk_id}")

                # Check for interruption first
                current_state = getattr(streaming_coordinator, 'state', 'idle')
                is_interruption = await detect_user_interruption(audio_array, current_state.value if hasattr(current_state, 'value') else str(current_state))

                if is_interruption:
                    streaming_logger.info(f"[EMOJI] User interruption detected for {client_id}")
                    await streaming_coordinator.handle_interruption("user_speech")
                    await websocket.send_text(json.dumps({
                        "type": "interruption",
                        "message": "User interruption detected",
                        "chunk_id": chunk_id,
                        "timestamp": time.time()
                    }))
                    return

                # Process with streaming Voxtral
                voxtral_stream = voxtral_model.process_streaming_chunk(
                    audio_array,
                    prompt=prompt,
                    chunk_id=chunk_id,
                    mode="streaming"
                )

                # Process streaming tokens and coordinate TTS
                full_response = ""
                words_sent_for_tts = []

                async for stream_chunk in streaming_coordinator.process_voxtral_stream(voxtral_stream):
                    if stream_chunk.type == 'words_ready':
                        words_text = stream_chunk.content['text']
                        full_response += " " + words_text
                        words_sent_for_tts.append(words_text)

                        # Enhanced logging for visibility
                        streaming_logger.info(f"[STREAMING] Received words from Voxtral (seq {stream_chunk.content['sequence_number']}): '{words_text}'")
                        streaming_logger.info(f"[STREAMING] Full response so far: '{full_response.strip()}'")

                        # Send words to client immediately
                        await websocket.send_text(json.dumps({
                            "type": "streaming_words",
                            "text": words_text,
                            "full_text_so_far": full_response.strip(),
                            "chunk_id": chunk_id,
                            "sequence": stream_chunk.content['sequence_number'],
                            "timestamp": time.time()
                        }))

                        # FIXED: Queue TTS instead of immediate generation to prevent overlap
                        try:
                            tts_timing_id = performance_monitor.start_timing("kokoro_streaming", {
                                "chunk_id": f"{chunk_id}_tts_{stream_chunk.content['sequence_number']}",
                                "text_length": len(words_text),
                                "voice": "hf_alpha"
                            })

                            # FIXED: Generate complete TTS audio for sequential playback
                            tts_result = await kokoro_model.synthesize_speech(
                                text=words_text,
                                voice="hf_alpha",  # FIXED: Use Indian female voice
                                speed=1.0
                            )

                            if tts_result and 'audio_data' in tts_result:
                                # Send complete audio chunk for sequential playback
                                audio_b64 = base64.b64encode(tts_result['audio_data']).decode('utf-8')
                                await websocket.send_text(json.dumps({
                                    "type": "audio_response",
                                    "audio_data": audio_b64,
                                    "chunk_id": f"{chunk_id}_tts_{stream_chunk.content['sequence_number']}",
                                    "voice": "hf_alpha",
                                    "text_source": words_text,
                                    "sequence": stream_chunk.content['sequence_number'],
                                    "metadata": {
                                        "sample_rate": tts_result.get('sample_rate', 22050),
                                        "duration_ms": tts_result.get('duration_ms', 0)
                                    },
                                    "timestamp": time.time()
                                }))

                                tts_time = performance_monitor.end_timing(tts_timing_id)
                                streaming_logger.info(f"[OK] TTS chunk completed in {tts_time:.1f}ms")

                        except Exception as tts_error:
                            streaming_logger.error(f"[ERROR] TTS streaming error: {tts_error}")

                    elif stream_chunk.type == 'session_complete':
                        # End Voxtral timing
                        voxtral_processing_time = performance_monitor.end_timing(voxtral_timing_id)

                        # Send completion
                        await websocket.send_text(json.dumps({
                            "type": "streaming_complete",
                            "full_response": full_response.strip(),
                            "total_words_sent": len(words_sent_for_tts),
                            "voxtral_time_ms": voxtral_processing_time,
                            "chunk_id": chunk_id,
                            "timestamp": time.time()
                        }))
                        break

                    elif stream_chunk.type == 'error':
                        streaming_logger.error(f"[ERROR] Streaming error: {stream_chunk.content}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Streaming error: {stream_chunk.content.get('error', 'Unknown error')}",
                            "chunk_id": chunk_id
                        }))
                        break

            else:
                # REGULAR MODE: Original processing
                result = await voxtral_model.process_realtime_chunk(
                    audio_tensor,
                    chunk_id,
                    mode=mode,
                    prompt=prompt
                )

                # End Voxtral timing
                voxtral_processing_time = performance_monitor.end_timing(voxtral_timing_id)

                if result['success']:
                    response = result['response']
                    processing_time = result['processing_time_ms']

                    # Check for response deduplication with timestamp-based structure
                    last_response_data = recent_responses.get(client_id, {})
                    last_response = last_response_data.get('response', '') if isinstance(last_response_data, dict) else last_response_data
                    is_duplicate = response and response.strip() and response == last_response

                    if not is_duplicate:
                        # Send text response first
                        await websocket.send_text(json.dumps({
                            "type": "response",
                            "mode": mode,
                            "text": response,
                            "chunk_id": chunk_id,
                            "processing_time_ms": round(processing_time, 1),
                            "audio_duration_ms": len(audio_array) / config.audio.sample_rate * 1000,
                            "timestamp": data.get("timestamp", time.time()),
                            "skipped_reason": result.get('skipped_reason', None),
                            "had_speech": result.get('had_speech', True)
                        }))

                        # Generate TTS audio if we have a meaningful response using Kokoro TTS
                        if response and response.strip():
                            try:
                                # Start TTS timing
                                tts_timing_id = performance_monitor.start_timing("kokoro_generation", {
                                    "chunk_id": chunk_id,
                                    "text_length": len(response),
                                    "voice": "hf_alpha"  # OPTIMIZED: Hindi female voice for Indian accent (was "af_heart")
                                })

                                # Generate speech using Kokoro TTS model
                                result = await kokoro_model.synthesize_speech(
                                    text=response,
                                    voice="hf_alpha"  # OPTIMIZED: Use Hindi female voice for Indian accent
                                )

                                if not result.get("success", False):
                                    raise Exception(f"Kokoro TTS generation failed: {result.get('error', 'Unknown error')}")

                                audio_data = result["audio_data"]
                                sample_rate = result.get("sample_rate", 24000)

                                # End TTS timing
                                tts_generation_time = performance_monitor.end_timing(tts_timing_id)

                                if audio_data is not None and len(audio_data) > 0:
                                    # Audio quality validation and normalization
                                    audio_rms = np.sqrt(np.mean(audio_data**2))
                                    audio_peak = np.max(np.abs(audio_data))

                                    logger.info(f"[AUDIO] Audio quality check - RMS: {audio_rms:.6f}, Peak: {audio_peak:.6f}")

                                    # Normalize audio if too quiet or too loud
                                    normalized_audio = audio_data
                                    if audio_rms < 0.05:  # Too quiet
                                        target_rms = 0.2
                                        gain = target_rms / (audio_rms + 1e-8)
                                        normalized_audio = audio_data * gain
                                        logger.info(f"[SPEAKER] Audio boosted by {gain:.2f}x (was too quiet)")
                                    elif audio_peak > 0.95:  # Risk of clipping
                                        gain = 0.9 / audio_peak
                                        normalized_audio = audio_data * gain
                                        logger.info(f"[EMOJI] Audio reduced by {gain:.2f}x (preventing clipping)")

                                    # Convert numpy array to proper WAV format with headers
                                    # FIXED: soundfile and BytesIO now imported at top of file

                                    # Create WAV file in memory with normalized audio
                                    wav_buffer = BytesIO()
                                    sf.write(wav_buffer, normalized_audio, sample_rate, format='WAV', subtype='PCM_16')
                                    wav_bytes = wav_buffer.getvalue()
                                    wav_buffer.close()

                                    # Validate WAV file creation
                                    if len(wav_bytes) < 100:  # WAV header alone is ~44 bytes
                                        raise Exception(f"WAV file too small: {len(wav_bytes)} bytes")

                                    # Verify WAV headers
                                    if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
                                        raise Exception("Invalid WAV file headers")

                                    logger.info(f"[OK] WAV file created: {len(wav_bytes)} bytes with proper headers")

                                    # Convert to base64 for transmission
                                    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

                                    # Calculate audio duration from actual audio samples
                                    audio_duration_ms = (len(audio_data) / sample_rate) * 1000

                                    # Send audio response
                                    await websocket.send_text(json.dumps({
                                        "type": "audio_response",
                                        "audio_data": audio_b64,
                                        "chunk_id": chunk_id,
                                        "voice": "hf_alpha",  # FIXED: Use Indian female voice
                                        "format": "wav",
                                        "conversation_ready": True,  # FIXED: Signal conversation can continue
                                        "metadata": {
                                            "audio_duration_ms": audio_duration_ms,
                                            "generation_time_ms": tts_generation_time,
                                            "sample_rate": sample_rate,
                                            "channels": 1,
                                            "format": "WAV",
                                            "subtype": "PCM_16"
                                        }
                                    }))

                                    streaming_logger.info(f"[TTS-KOKORO] Audio response generated for chunk {chunk_id} in {tts_generation_time:.1f}ms")

                                    # Log performance breakdown
                                    performance_monitor.log_latency_breakdown({
                                        "voxtral_processing_ms": voxtral_processing_time,
                                        "kokoro_generation_ms": tts_generation_time,
                                        "audio_conversion_ms": 0,  # Already included in generation
                                        "total_end_to_end_ms": voxtral_processing_time + tts_generation_time
                                    })

                                else:
                                    streaming_logger.warning(f"[TTS-DIRECT] Failed to generate audio for chunk {chunk_id}")

                            except Exception as tts_error:
                                streaming_logger.error(f"[TTS-DIRECT] Error generating audio response: {tts_error}")
                                # End timing even on error
                                if 'tts_timing_id' in locals():
                                    performance_monitor.end_timing(tts_timing_id)

                        # Update recent response tracking with timestamp
                        if response and response.strip():
                            recent_responses[client_id] = {
                                'response': response,
                                'timestamp': time.time()
                            }
                            streaming_logger.info(f"[CONVERSATION] Unique response sent for chunk {chunk_id}: '{response[:50]}...'")

                            # FIXED: Clear old responses to prevent false duplicates (older than 30 seconds)
                            current_time = time.time()
                            expired_clients = [cid for cid, data in recent_responses.items()
                                             if isinstance(data, dict) and current_time - data.get('timestamp', 0) > 30]
                            for expired_client in expired_clients:
                                del recent_responses[expired_client]
                        else:
                            streaming_logger.info(f"[CONVERSATION] Silence detected for chunk {chunk_id} - no response needed")
                    else:
                        streaming_logger.info(f"[CONVERSATION] Duplicate response detected for chunk {chunk_id} - skipping")
                else:
                    streaming_logger.warning(f"[CONVERSATION] Processing failed for chunk {chunk_id}")

        except Exception as e:
            streaming_logger.error(f"[CONVERSATION] Voxtral/Streaming processing error for chunk {chunk_id}: {e}")
            # End Voxtral timing on error
            if 'voxtral_timing_id' in locals():
                performance_monitor.end_timing(voxtral_timing_id)

    except Exception as e:
        streaming_logger.error(f"[CONVERSATION] Error handling audio chunk: {e}")

async def initialize_models_at_startup():
    """Initialize all models using unified model manager at application startup"""
    streaming_logger.info("[INIT] Initializing unified model system at startup...")

    try:
        # Initialize unified model manager
        unified_manager = get_unified_manager()
        audio_processor = get_audio_processor()
        performance_monitor = get_performance_monitor()

        if not unified_manager.is_initialized:
            streaming_logger.info("[INPUT] Initializing unified model manager...")
            success = await unified_manager.initialize()
            
            if success:
                streaming_logger.info("[OK] Unified model manager initialized successfully")
                
                # Get model info for logging
                model_info = unified_manager.get_model_info()
                streaming_logger.info(f"[STATS] Voxtral initialized: {model_info['unified_manager']['voxtral_initialized']}")
                streaming_logger.info(f"[STATS] Kokoro TTS initialized: {model_info['unified_manager']['kokoro_initialized']}")

                # Log memory statistics
                memory_stats = unified_manager.get_memory_stats()
                if "memory_stats" in memory_stats:
                    stats = memory_stats["memory_stats"]
                    streaming_logger.info(f"[FLOPPY] GPU Memory: {stats['used_vram_gb']:.2f}GB / {stats['total_vram_gb']:.2f}GB")
                    streaming_logger.info(f"[FLOPPY] Voxtral: {stats['voxtral_memory_gb']:.2f}GB, Kokoro: {stats['kokoro_memory_gb']:.2f}GB")
                
            else:
                raise Exception("Unified model manager initialization failed")
        else:
            streaming_logger.info("[OK] Unified model manager already initialized")

        streaming_logger.info("[SUCCESS] All models ready for conversation with Kokoro TTS integration!")

    except Exception as e:
        streaming_logger.error(f"[ERROR] Failed to initialize unified model system: {e}")
        # Try to get error details from unified manager
        try:
            unified_manager = get_unified_manager()
            error_summary = unified_manager.get_model_info()
            streaming_logger.error(f"[STATS] Model states: {error_summary}")
        except:
            pass
        raise

if __name__ == "__main__":
    streaming_logger.info("Starting Voxtral Conversational Streaming UI Server with VAD")

    # Pre-load models before starting server
    import asyncio
    asyncio.run(initialize_models_at_startup())

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.http_port,
        log_level="info"
    )
