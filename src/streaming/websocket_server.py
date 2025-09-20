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

            # Check if speech-to-speech mode is requested
            if mode == "speech_to_speech" and config.speech_to_speech.enabled:
                await self.handle_speech_to_speech(websocket, audio_array, data)
                return

            prompt = data.get("prompt", "")

            # Process with Voxtral (legacy modes)
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

    async def handle_speech_to_speech(self, websocket, audio_array: np.ndarray, data: Dict[str, Any]):
        """Handle speech-to-speech conversation processing"""
        try:
            start_time = time.time()
            conversation_id = data.get("conversation_id", f"ws_{int(time.time() * 1000)}")

            # Voice preferences
            voice_preference = data.get("voice", None)
            speed_preference = data.get("speed", None)

            logger.info(f"[SPEAK] Processing speech-to-speech conversation: {conversation_id}")

            # Send processing status
            await self.send_message(websocket, {
                "type": "processing",
                "conversation_id": conversation_id,
                "stage": "speech_to_text",
                "message": "Converting speech to text..."
            })

            # Process through speech-to-speech pipeline
            result = await speech_to_speech_pipeline.process_conversation_turn(
                audio_array,
                conversation_id=conversation_id,
                voice_preference=voice_preference,
                speed_preference=speed_preference
            )

            if not result['success']:
                await self.send_message(websocket, {
                    "type": "error",
                    "conversation_id": conversation_id,
                    "message": f"Speech-to-speech processing failed: {result.get('error', 'Unknown error')}"
                })
                return

            # Send transcription update
            if result['transcription']:
                await self.send_message(websocket, {
                    "type": "transcription",
                    "conversation_id": conversation_id,
                    "text": result['transcription'],
                    "stage_timing_ms": result['stage_timings'].get('stt_ms', 0)
                })

            # Send response text
            if result['response_text']:
                await self.send_message(websocket, {
                    "type": "response_text",
                    "conversation_id": conversation_id,
                    "text": result['response_text'],
                    "stage_timing_ms": result['stage_timings'].get('llm_ms', 0)
                })

            # Send audio response if available
            if len(result['response_audio']) > 0:
                # Convert audio to base64 for transmission
                audio_bytes = result['response_audio'].astype(np.float32).tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                await self.send_message(websocket, {
                    "type": "speech_response",
                    "conversation_id": conversation_id,
                    "audio_data": audio_b64,
                    "sample_rate": result['sample_rate'],
                    "voice_used": result.get('voice_used', 'unknown'),
                    "speed_used": result.get('speed_used', 1.0),
                    "audio_duration_s": len(result['response_audio']) / result['sample_rate'],
                    "stage_timing_ms": result['stage_timings'].get('tts_ms', 0)
                })

            # Send final summary
            await self.send_message(websocket, {
                "type": "conversation_complete",
                "conversation_id": conversation_id,
                "total_latency_ms": result['total_latency_ms'],
                "meets_target": result['total_latency_ms'] <= config.speech_to_speech.latency_target_ms,
                "stage_timings": result['stage_timings'],
                "is_silence": result.get('is_silence', False)
            })

            logger.info(f"[OK] Speech-to-speech conversation {conversation_id} completed in {result['total_latency_ms']:.1f}ms")

            # Update health check status with latest performance
            try:
                from src.api.health_check import update_speech_to_speech_status
                pipeline_info = speech_to_speech_pipeline.get_pipeline_info()
                update_speech_to_speech_status({
                    "initialized": speech_to_speech_pipeline.is_initialized,
                    "info": pipeline_info
                })
            except Exception as e:
                logger.debug(f"Could not update speech-to-speech health status: {e}")

        except Exception as e:
            logger.error(f"[ERROR] Error in speech-to-speech processing: {e}")
            await self.send_message(websocket, {
                "type": "error",
                "conversation_id": data.get("conversation_id", "unknown"),
                "message": f"Speech-to-speech error: {str(e)}"
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

        # Initialize Speech-to-Speech pipeline if enabled
        if config.speech_to_speech.enabled and not speech_to_speech_pipeline.is_initialized:
            logger.info("[EMOJI] Initializing Speech-to-Speech pipeline...")
            await speech_to_speech_pipeline.initialize()
            logger.info("[OK] Speech-to-Speech pipeline ready for conversational AI")

            # Update health check status
            try:
                from src.api.health_check import update_speech_to_speech_status
                pipeline_info = speech_to_speech_pipeline.get_pipeline_info()
                update_speech_to_speech_status({
                    "initialized": speech_to_speech_pipeline.is_initialized,
                    "info": pipeline_info
                })
                logger.info("[STATS] Speech-to-Speech status updated in health check system")
            except Exception as e:
                logger.warning(f"[WARN] Could not update speech-to-speech health status: {e}")
        
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
