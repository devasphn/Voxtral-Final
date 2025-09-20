"""
FIXED TCP server for real-time audio streaming with PRODUCTION VAD
Fixed import paths and added proper silence detection
"""
import asyncio
import json
import base64
import numpy as np
import struct
import logging
import time
from typing import Dict, Any, Set
import traceback
import sys
import os

# FIXED: Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import config
from src.utils.logging_config import logger

# FIXED: Import with proper error handling and correct paths
try:
    from src.models.voxtral_model_realtime import voxtral_model
    from src.models.audio_processor_realtime import AudioProcessor
    logger.info("Successfully imported models for TCP server")
except ImportError as e:
    logger.error(f"Import error in TCP server: {e}")
    # Fallback imports
    try:
        # Try alternative import paths
        from voxtral_model_realtime import voxtral_model
        from audio_processor_realtime import AudioProcessor
        logger.info("Successfully imported models using fallback paths")
    except ImportError as e2:
        logger.error(f"Fallback import also failed: {e2}")
        voxtral_model = None
        AudioProcessor = None

class TCPStreamingServer:
    """PRODUCTION TCP server for real-time audio streaming with VAD"""
    
    def __init__(self):
        self.clients: Set[asyncio.StreamWriter] = set()
        self.audio_processor = None
        self.host = config.server.host
        self.port = config.server.tcp_ports[1]  # Use second TCP port (8766)
        self.initialized = False
        
        # Production metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.vad_filtered_requests = 0
        
        logger.info(f"TCP server configured for {self.host}:{self.port}")
    
    async def initialize_components(self):
        """Initialize audio processor and model with error handling"""
        try:
            if not self.initialized:
                # Initialize audio processor
                if AudioProcessor:
                    self.audio_processor = AudioProcessor()
                    logger.info("[OK] TCP server audio processor initialized with VAD")
                else:
                    logger.error("[ERROR] AudioProcessor not available - cannot start TCP server")
                    raise RuntimeError("AudioProcessor not available")
                
                # Initialize Voxtral model if available
                if voxtral_model and not voxtral_model.is_initialized:
                    logger.info("[INIT] Initializing Voxtral model for TCP server...")
                    await voxtral_model.initialize()
                    logger.info("[OK] Voxtral model initialized for TCP server")
                elif not voxtral_model:
                    logger.error("[ERROR] Voxtral model not available - cannot start TCP server")
                    raise RuntimeError("Voxtral model not available")
                
                self.initialized = True
                logger.info("[SUCCESS] TCP server components fully initialized")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize TCP server components: {e}")
            raise
    
    async def send_response(self, writer: asyncio.StreamWriter, response_data: Dict[str, Any]):
        """Send response back to TCP client"""
        try:
            # Convert response to JSON
            response_json = json.dumps(response_data)
            response_bytes = response_json.encode('utf-8')
            
            # Send length prefix (4 bytes) + data
            length_prefix = struct.pack('!I', len(response_bytes))
            writer.write(length_prefix + response_bytes)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"[ERROR] Error sending TCP response: {e}")
    
    async def read_message(self, reader: asyncio.StreamReader) -> Dict[str, Any]:
        """Read a message from TCP client"""
        try:
            # Read length prefix (4 bytes)
            length_data = await reader.readexactly(4)
            message_length = struct.unpack('!I', length_data)[0]
            
            # Validate message length
            if message_length > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError(f"Message too large: {message_length} bytes")
            
            # Read the actual message
            message_data = await reader.readexactly(message_length)
            message_json = message_data.decode('utf-8')
            
            return json.loads(message_json)
            
        except asyncio.IncompleteReadError:
            logger.debug("Client disconnected during read")
            raise ConnectionResetError("Client disconnected")
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Invalid JSON from TCP client: {e}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error reading TCP message: {e}")
            raise
    
    async def handle_audio_stream(self, writer: asyncio.StreamWriter, data: Dict[str, Any]):
        """PRODUCTION audio processing with VAD filtering"""
        try:
            start_time = time.time()
            self.total_requests += 1
            
            # Check if components are initialized
            if not self.initialized or not self.audio_processor or not voxtral_model:
                await self.send_response(writer, {
                    "type": "error",
                    "message": "Server components not initialized"
                })
                return
            
            # Extract audio data
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                await self.send_response(writer, {
                    "type": "error",
                    "message": "No audio data provided"
                })
                return
            
            # Decode audio
            try:
                audio_bytes = base64.b64decode(audio_b64)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                logger.debug(f"[STATS] TCP audio decoded: {len(audio_array)} samples")
            except Exception as e:
                logger.error(f"[ERROR] TCP audio decoding error: {e}")
                await self.send_response(writer, {
                    "type": "error",
                    "message": f"Audio decoding error: {str(e)}"
                })
                return
            
            # CRITICAL: Apply VAD validation first
            if not self.audio_processor.validate_realtime_chunk(audio_array, chunk_id=f"tcp_{self.total_requests}"):
                self.vad_filtered_requests += 1
                logger.debug(f"[MUTE] TCP request {self.total_requests}: Filtered by VAD (silent/noise)")
                
                # Send empty response for silence - don't process
                await self.send_response(writer, {
                    "type": "response",
                    "text": "",  # Empty response for silence
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "audio_duration_ms": len(audio_array) / config.audio.sample_rate * 1000,
                    "filtered_by_vad": True,
                    "vad_stats": {
                        "total_requests": self.total_requests,
                        "vad_filtered": self.vad_filtered_requests,
                        "success_rate": (self.successful_requests / self.total_requests) * 100
                    }
                })
                return
            
            # Audio contains speech - proceed with processing
            logger.info(f"[VAD] TCP request {self.total_requests}: Speech detected, processing...")
            
            # Preprocess audio
            try:
                audio_tensor = self.audio_processor.preprocess_realtime_chunk(
                    audio_array, 
                    chunk_id=f"tcp_{self.total_requests}"
                )
            except Exception as e:
                logger.error(f"[ERROR] TCP audio preprocessing error: {e}")
                await self.send_response(writer, {
                    "type": "error",
                    "message": f"Audio preprocessing error: {str(e)}"
                })
                return
            
            # Smart Conversation Mode - unified processing
            mode = "conversation"  # Always use conversation mode
            prompt = ""  # Prompt is hardcoded in the model
            
            # Process with Voxtral
            try:
                result = await voxtral_model.process_realtime_chunk(
                    audio_tensor,
                    chunk_id=f"tcp_{self.total_requests}",
                    mode=mode,
                    prompt=prompt
                )
                
                if result['success'] and not result.get('is_silence', False):
                    response_text = result['response']
                    processing_time = result['processing_time_ms']
                    self.successful_requests += 1
                    
                    logger.info(f"[OK] TCP request {self.total_requests}: Success - '{response_text[:50]}...'")
                else:
                    # Model detected silence or returned empty response
                    response_text = ""
                    processing_time = result.get('processing_time_ms', (time.time() - start_time) * 1000)
                    logger.debug(f"[MUTE] TCP request {self.total_requests}: Model detected silence")
                    
            except Exception as e:
                logger.error(f"[ERROR] TCP Voxtral processing error: {e}")
                response_text = "Processing error"
                processing_time = (time.time() - start_time) * 1000
            
            # Send response
            await self.send_response(writer, {
                "type": "response",
                "mode": mode,
                "text": response_text,
                "processing_time_ms": round(processing_time, 1),
                "audio_duration_ms": len(audio_array) / config.audio.sample_rate * 1000,
                "timestamp": time.time(),
                "server_stats": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "vad_filtered": self.vad_filtered_requests,
                    "success_rate": round((self.successful_requests / self.total_requests) * 100, 1)
                }
            })
            
            logger.debug(f"[STATS] TCP processing completed in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"[ERROR] Error processing TCP audio stream: {e}")
            await self.send_response(writer, {
                "type": "error",
                "message": f"Processing error: {str(e)}"
            })
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle individual TCP client connection"""
        client_addr = writer.get_extra_info('peername')
        logger.info(f"[EMOJI] TCP client connected: {client_addr}")
        
        self.clients.add(writer)
        
        try:
            # Send welcome message
            await self.send_response(writer, {
                "type": "connection",
                "status": "connected", 
                "message": "Connected to Voxtral TCP streaming server with VAD",
                "protocol": "tcp",
                "server_config": {
                    "sample_rate": config.audio.sample_rate,
                    "chunk_size": config.audio.chunk_size,
                    "format": config.audio.format,
                    "vad_enabled": True,
                    "silence_filtering": True
                }
            })
            
            # Handle client messages
            while True:
                try:
                    message = await asyncio.wait_for(
                        self.read_message(reader), 
                        timeout=config.streaming.timeout_seconds
                    )
                    
                    msg_type = message.get("type")
                    
                    if msg_type == "audio":
                        await self.handle_audio_stream(writer, message)
                        
                    elif msg_type == "ping":
                        await self.send_response(writer, {
                            "type": "pong",
                            "timestamp": time.time()
                        })
                        
                    elif msg_type == "status":
                        model_info = voxtral_model.get_model_info() if voxtral_model else {"status": "not_available"}
                        await self.send_response(writer, {
                            "type": "status",
                            "model_info": model_info,
                            "connected_clients": len(self.clients),
                            "server_stats": {
                                "total_requests": self.total_requests,
                                "successful_requests": self.successful_requests,
                                "vad_filtered": self.vad_filtered_requests
                            }
                        })
                        
                    else:
                        await self.send_response(writer, {
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}"
                        })
                        
                except asyncio.TimeoutError:
                    logger.debug(f"[EMOJI] TCP client timeout: {client_addr}")
                    break
                    
        except ConnectionResetError:
            logger.debug(f"[EMOJI] TCP client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"[ERROR] TCP client error {client_addr}: {e}")
            logger.debug(traceback.format_exc())
        finally:
            self.clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
            logger.info(f"[EMOJI] TCP client {client_addr} connection closed")
    
    async def start_server(self):
        """Start the TCP streaming server"""
        logger.info(f"[INIT] Starting TCP server on {self.host}:{self.port}")
        
        # Initialize components
        await self.initialize_components()
        
        try:
            # Start TCP server
            server = await asyncio.start_server(
                self.handle_client,
                self.host,
                self.port,
                reuse_address=True,  # Allow reuse of address
                reuse_port=True      # Allow reuse of port
            )
            
            logger.info(f"[OK] TCP server running on {self.host}:{self.port}")
            logger.info(f"[VAD] VAD-enabled streaming server ready for production")
            
            # Keep server running
            async with server:
                await server.serve_forever()
                
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.error(f"[ERROR] Port {self.port} is already in use. Please run cleanup.sh first.")
                raise
            else:
                logger.error(f"[ERROR] Failed to start TCP server: {e}")
                raise

async def main():
    """Main entry point for TCP server"""
    server = TCPStreamingServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("[EMOJI] TCP server shutdown requested")
    except Exception as e:
        logger.error(f"[ERROR] TCP server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
