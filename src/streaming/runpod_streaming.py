"""
RunPod-Optimized Real-time Audio Streaming
WebSocket-based solution for HTTP/TCP-only environment
Ultra-low latency streaming with chunked audio processing
"""
import asyncio
import websockets
import json
import base64
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, AsyncGenerator
from collections import deque
import io

# Setup logging
streaming_logger = logging.getLogger("runpod_streaming")

class RunPodAudioStreamer:
    """
    Real-time audio streaming optimized for RunPod's HTTP/TCP-only infrastructure
    Uses WebSocket for bidirectional communication with chunked processing
    """
    
    def __init__(self):
        self.is_initialized = False
        self.active_connections = set()
        self.audio_buffer = deque(maxlen=100)  # Buffer for audio chunks
        self.processing_queue = asyncio.Queue(maxsize=50)
        
        # Performance tracking
        self.latency_history = deque(maxlen=100)
        self.chunk_count = 0
        
        # RunPod optimizations
        self.chunk_size_ms = 100  # 100ms chunks for ultra-low latency
        self.max_chunk_size = 4096  # Maximum chunk size in bytes
        self.compression_enabled = True  # Enable audio compression
        
        streaming_logger.info("RunPod Audio Streamer initialized")
    
    async def initialize(self):
        """Initialize the audio streamer"""
        streaming_logger.info("🎵 Initializing RunPod Audio Streamer...")
        
        try:
            # Initialize audio processing components
            await self._initialize_audio_processor()
            
            # Start background processing task
            asyncio.create_task(self._process_audio_queue())
            
            self.is_initialized = True
            streaming_logger.info("✅ RunPod Audio Streamer ready")
            return True
            
        except Exception as e:
            streaming_logger.error(f"❌ Audio streamer initialization failed: {e}")
            return False
    
    async def _initialize_audio_processor(self):
        """Initialize audio processing components"""
        try:
            from src.models.audio_processor_realtime import AudioProcessor
            self.audio_processor = AudioProcessor()
            await self.audio_processor.initialize()
            streaming_logger.info("   ✅ Audio processor initialized")
        except Exception as e:
            streaming_logger.error(f"   ❌ Audio processor initialization failed: {e}")
            raise
    
    async def handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        streaming_logger.info(f"🔌 New connection: {client_id}")
        
        self.active_connections.add(websocket)
        
        try:
            # Send welcome message with RunPod-specific configuration
            await self._send_welcome_message(websocket)
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            streaming_logger.info(f"🔌 Connection closed: {client_id}")
        except Exception as e:
            streaming_logger.error(f"❌ Connection error for {client_id}: {e}")
        finally:
            self.active_connections.discard(websocket)
    
    async def _send_welcome_message(self, websocket):
        """Send welcome message with configuration"""
        welcome_msg = {
            "type": "welcome",
            "status": "connected",
            "config": {
                "chunk_size_ms": self.chunk_size_ms,
                "max_chunk_size": self.max_chunk_size,
                "compression_enabled": self.compression_enabled,
                "sample_rate": 16000,
                "channels": 1,
                "format": "float32"
            },
            "runpod_optimized": True,
            "latency_target_ms": 500
        }
        
        await websocket.send(json.dumps(welcome_msg))
    
    async def _handle_message(self, websocket, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "audio_chunk":
                await self._handle_audio_chunk(websocket, data)
            elif message_type == "start_stream":
                await self._handle_start_stream(websocket, data)
            elif message_type == "stop_stream":
                await self._handle_stop_stream(websocket, data)
            elif message_type == "ping":
                await self._handle_ping(websocket, data)
            else:
                streaming_logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            streaming_logger.error("Invalid JSON received")
        except Exception as e:
            streaming_logger.error(f"Error handling message: {e}")
    
    async def _handle_audio_chunk(self, websocket, data):
        """Handle incoming audio chunk"""
        start_time = time.time()
        
        try:
            # Decode audio data
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                await self._send_error(websocket, "No audio data provided")
                return
            
            audio_bytes = base64.b64decode(audio_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Add to processing queue
            await self.processing_queue.put({
                "websocket": websocket,
                "audio": audio_array,
                "timestamp": start_time,
                "chunk_id": self.chunk_count
            })
            
            self.chunk_count += 1
            
        except Exception as e:
            streaming_logger.error(f"Error processing audio chunk: {e}")
            await self._send_error(websocket, f"Audio processing error: {str(e)}")
    
    async def _process_audio_queue(self):
        """Background task to process audio queue"""
        while True:
            try:
                # Get audio chunk from queue
                chunk_data = await self.processing_queue.get()
                
                # Process audio through speech-to-speech pipeline
                await self._process_speech_to_speech(chunk_data)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                streaming_logger.error(f"Error in audio processing queue: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _process_speech_to_speech(self, chunk_data):
        """Process audio through speech-to-speech pipeline"""
        start_time = time.time()
        websocket = chunk_data["websocket"]
        audio = chunk_data["audio"]
        chunk_id = chunk_data["chunk_id"]
        
        try:
            # Get speech-to-speech pipeline
            from src.models.speech_to_speech_pipeline import speech_to_speech_pipeline
            
            if not speech_to_speech_pipeline.is_initialized:
                await speech_to_speech_pipeline.initialize()
            
            # Process audio
            result = await speech_to_speech_pipeline.process_speech_to_speech(audio)
            
            # Calculate latency
            processing_time = (time.time() - start_time) * 1000
            self.latency_history.append(processing_time)
            
            # Send response
            response = {
                "type": "speech_response",
                "chunk_id": chunk_id,
                "success": result.get("success", False),
                "text": result.get("text", ""),
                "audio_data": result.get("audio_data", ""),
                "processing_time_ms": processing_time,
                "timestamp": time.time()
            }
            
            if websocket in self.active_connections:
                await websocket.send(json.dumps(response))
            
            streaming_logger.debug(f"Processed chunk {chunk_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            streaming_logger.error(f"Error in speech-to-speech processing: {e}")
            await self._send_error(websocket, f"Processing error: {str(e)}")
    
    async def _handle_start_stream(self, websocket, data):
        """Handle start streaming request"""
        response = {
            "type": "stream_started",
            "status": "ready",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(response))
    
    async def _handle_stop_stream(self, websocket, data):
        """Handle stop streaming request"""
        response = {
            "type": "stream_stopped",
            "status": "stopped",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(response))
    
    async def _handle_ping(self, websocket, data):
        """Handle ping request"""
        response = {
            "type": "pong",
            "timestamp": time.time(),
            "latency_stats": self.get_latency_stats()
        }
        await websocket.send(json.dumps(response))
    
    async def _send_error(self, websocket, error_message):
        """Send error message to client"""
        error_response = {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        }
        
        if websocket in self.active_connections:
            await websocket.send(json.dumps(error_response))
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latency_history:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        
        latencies = list(self.latency_history)
        return {
            "avg": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "count": len(latencies)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get streamer status"""
        return {
            "initialized": self.is_initialized,
            "active_connections": len(self.active_connections),
            "chunks_processed": self.chunk_count,
            "queue_size": self.processing_queue.qsize(),
            "latency_stats": self.get_latency_stats()
        }

# Global streamer instance
runpod_streamer = RunPodAudioStreamer()

async def start_runpod_streaming_server(host="0.0.0.0", port=8765):
    """Start the RunPod streaming server"""
    streaming_logger.info(f"🚀 Starting RunPod streaming server on {host}:{port}")
    
    # Initialize streamer
    await runpod_streamer.initialize()
    
    # Start WebSocket server
    async with websockets.serve(
        runpod_streamer.handle_websocket_connection,
        host,
        port,
        max_size=2**20,  # 1MB max message size
        ping_interval=20,
        ping_timeout=60,
        compression=None  # Disable compression for lower latency
    ):
        streaming_logger.info(f"✅ RunPod streaming server running on ws://{host}:{port}")
        streaming_logger.info(f"🌐 RunPod URL: wss://[POD_ID]-{port}.proxy.runpod.net")
        
        # Keep server running
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(start_runpod_streaming_server())
