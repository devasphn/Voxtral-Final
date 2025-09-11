"""
WebSocket server for real-time audio streaming with Voxtral (FIXED)
Updated to resolve deprecation warnings and RunPod proxy compatibility
"""
import asyncio
import websockets.asyncio.server
import json
import base64
import numpy as np
import logging
import time
from typing import Set, Dict, Any
import traceback

from src.models.voxtral_model_realtime import voxtral_model
from src.models.audio_processor_realtime import AudioProcessor
from src.utils.config import config
from src.utils.logging_config import logger

class WebSocketServer:
    """WebSocket server for real-time Voxtral streaming"""
    
    def __init__(self):
        self.clients: Set = set()
        self.audio_processor = AudioProcessor()
        self.host = config.server.host
        self.port = config.server.tcp_ports[0]  # Use first TCP port (8765)
        
        logger.info(f"WebSocket server configured for {self.host}:{self.port}")
    
    async def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_info} (Total: {len(self.clients)})")
        
        # Send welcome message
        await self.send_message(websocket, {
            "type": "connection",
            "status": "connected",
            "message": "Connected to Voxtral streaming server",
            "server_config": {
                "sample_rate": config.audio.sample_rate,
                "chunk_size": config.audio.chunk_size,
                "latency_target": config.streaming.latency_target_ms
            }
        })
    
    async def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client disconnected: {client_info} (Total: {len(self.clients)})")
    
    async def send_message(self, websocket, message: Dict[str, Any]):
        """Send JSON message to client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Attempted to send message to closed connection")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def handle_audio_data(self, websocket, data: Dict[str, Any]):
        """Process incoming audio data"""
        try:
            start_time = time.time()
            
            # Extract audio data
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                await self.send_message(websocket, {
                    "type": "error",
                    "message": "No audio data provided"
                })
                return
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Validate audio format
            if not self.audio_processor.validate_audio_format(audio_array):
                await self.send_message(websocket, {
                    "type": "error", 
                    "message": "Invalid audio format"
                })
                return
            
            # Preprocess audio
            audio_tensor = self.audio_processor.preprocess_audio(audio_array)
            
            # Get processing mode
            mode = data.get("mode", "transcribe")
            prompt = data.get("prompt", "")
            
            # Process with Voxtral
            if mode == "transcribe":
                response = await voxtral_model.transcribe_audio(audio_tensor)
            elif mode == "understand":
                if not prompt:
                    prompt = "What can you tell me about this audio?"
                response = await voxtral_model.understand_audio(audio_tensor, prompt)
            else:
                # General processing
                response = await voxtral_model.process_audio_stream(audio_tensor, prompt)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Send response
            await self.send_message(websocket, {
                "type": "response",
                "mode": mode,
                "text": response,
                "processing_time_ms": round(processing_time, 1),
                "audio_duration_ms": len(audio_array) / config.audio.sample_rate * 1000
            })
            
            logger.debug(f"Processed audio in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error handling audio data: {e}")
            await self.send_message(websocket, {
                "type": "error",
                "message": f"Processing error: {str(e)}"
            })
    
    async def handle_message(self, websocket, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "audio":
                await self.handle_audio_data(websocket, data)
                
            elif msg_type == "ping":
                await self.send_message(websocket, {
                    "type": "pong", 
                    "timestamp": time.time()
                })
                
            elif msg_type == "status":
                model_info = voxtral_model.get_model_info()
                await self.send_message(websocket, {
                    "type": "status",
                    "model_info": model_info,
                    "connected_clients": len(self.clients)
                })
                
            else:
                await self.send_message(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
                
        except json.JSONDecodeError:
            await self.send_message(websocket, {
                "type": "error",
                "message": "Invalid JSON format"
            })
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_message(websocket, {
                "type": "error", 
                "message": "Server error processing message"
            })
    
    async def handle_client(self, websocket):
        """Handle individual client connection (FIXED - removed path parameter)"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.debug("Client connection closed normally")
        except Exception as e:
            logger.error(f"Client connection error: {e}")
            logger.debug(traceback.format_exc())
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server with RunPod compatibility"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        # Initialize Voxtral model
        if not voxtral_model.is_initialized:
            await voxtral_model.initialize()
        
        # Start WebSocket server with updated API
        async with websockets.asyncio.server.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=config.streaming.buffer_size,
            ping_interval=20,
            ping_timeout=60,
            # Additional headers for RunPod proxy compatibility
            extra_headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        ) as server:
            logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
            logger.info(f"RunPod WebSocket URL: wss://[POD_ID]-{self.port}.proxy.runpod.net/ws")
            
            # Keep server running indefinitely
            await asyncio.Future()  # Run forever

async def main():
    """Main entry point for WebSocket server"""
    server = WebSocketServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("WebSocket server shutdown requested")
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
