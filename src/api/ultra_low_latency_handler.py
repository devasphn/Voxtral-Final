"""
Ultra-Low Latency WebSocket Handler for <200ms Speech-to-Speech
Implements chunked streaming with real-time audio processing
"""

import asyncio
import json
import time
import base64
import logging
from typing import Dict, Any, Optional
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Import optimized components
from src.models.ultra_low_latency_manager import get_ultra_low_latency_manager
from src.audio.chunked_streaming_processor import get_chunked_processor
from src.utils.performance_monitor import get_performance_monitor

# Setup logging
ull_handler_logger = logging.getLogger("ull_websocket_handler")
ull_handler_logger.setLevel(logging.INFO)

class UltraLowLatencyWebSocketHandler:
    """
    Ultra-optimized WebSocket handler for <200ms end-to-end latency
    Features:
    - Chunked audio streaming
    - Real-time VAD and processing
    - Interruption handling
    - Performance monitoring
    - Zero-copy audio operations
    """
    
    def __init__(self):
        self.active_connections = {}
        self.connection_stats = {}
        self.manager = None
        self.processor = None
        self.performance_monitor = None
        
        # Performance targets
        self.latency_targets = {
            "first_word_ms": 100,
            "end_to_end_ms": 200,
            "chunk_processing_ms": 50
        }
        
        ull_handler_logger.info("UltraLowLatencyWebSocketHandler initialized")
    
    async def initialize(self):
        """Initialize the handler with optimized components"""
        try:
            ull_handler_logger.info("[INIT] Initializing ultra-low latency handler...")
            
            # Initialize manager
            self.manager = get_ultra_low_latency_manager()
            if not self.manager.is_initialized:
                success = await self.manager.initialize()
                if not success:
                    raise RuntimeError("Failed to initialize ultra-low latency manager")
            
            # Initialize processor
            self.processor = get_chunked_processor()
            
            # Initialize performance monitor
            self.performance_monitor = get_performance_monitor()
            
            ull_handler_logger.info("[SUCCESS] Ultra-low latency handler initialized")
            return True
            
        except Exception as e:
            ull_handler_logger.error(f"[ERROR] Handler initialization failed: {e}")
            return False
    
    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection with ultra-low latency processing"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        connection_start = time.time()
        
        try:
            ull_handler_logger.info(f"[CONNECT] Ultra-low latency client connected: {client_id}")
            
            # Initialize connection tracking
            self.active_connections[client_id] = {
                "websocket": websocket,
                "connected_at": connection_start,
                "chunks_processed": 0,
                "total_latency_ms": 0.0,
                "interruptions_handled": 0
            }
            
            # Send welcome message
            await self._send_welcome_message(websocket, client_id)
            
            # Main message processing loop
            async for message in websocket:
                await self._process_message(websocket, client_id, message)
                
        except ConnectionClosed:
            ull_handler_logger.info(f"[DISCONNECT] Client disconnected: {client_id}")
        except WebSocketException as e:
            ull_handler_logger.error(f"[ERROR] WebSocket error for {client_id}: {e}")
        except Exception as e:
            ull_handler_logger.error(f"[ERROR] Unexpected error for {client_id}: {e}")
        finally:
            # Cleanup connection
            if client_id in self.active_connections:
                connection_duration = time.time() - connection_start
                stats = self.active_connections[client_id]
                ull_handler_logger.info(f"[STATS] {client_id} session: {connection_duration:.1f}s, "
                                      f"{stats['chunks_processed']} chunks, "
                                      f"avg latency: {stats['total_latency_ms']/max(1, stats['chunks_processed']):.1f}ms")
                del self.active_connections[client_id]
    
    async def _send_welcome_message(self, websocket, client_id: str):
        """Send optimized welcome message"""
        welcome_msg = {
            "type": "welcome",
            "message": "Ultra-Low Latency Voice Agent Ready",
            "client_id": client_id,
            "latency_targets": self.latency_targets,
            "features": [
                "chunked_streaming",
                "real_time_vad",
                "interruption_handling",
                "sub_200ms_latency"
            ],
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(welcome_msg))
        ull_handler_logger.debug(f"[WELCOME] Sent to {client_id}")
    
    async def _process_message(self, websocket, client_id: str, message: str):
        """Process incoming WebSocket message with ultra-low latency"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "audio_chunk":
                await self._handle_audio_chunk(websocket, client_id, data)
            elif message_type == "interruption":
                await self._handle_interruption(websocket, client_id, data)
            elif message_type == "performance_request":
                await self._handle_performance_request(websocket, client_id)
            else:
                ull_handler_logger.warning(f"[WARN] Unknown message type: {message_type} from {client_id}")
                
        except json.JSONDecodeError as e:
            ull_handler_logger.error(f"[ERROR] Invalid JSON from {client_id}: {e}")
            await self._send_error(websocket, "Invalid JSON format")
        except Exception as e:
            ull_handler_logger.error(f"[ERROR] Message processing failed for {client_id}: {e}")
            await self._send_error(websocket, f"Processing error: {str(e)}")
    
    async def _handle_audio_chunk(self, websocket, client_id: str, data: Dict[str, Any]):
        """Handle audio chunk with ultra-low latency processing"""
        chunk_start_time = time.time()
        chunk_id = data.get("chunk_id", f"chunk_{int(time.time() * 1000)}")
        
        try:
            # Extract audio data
            audio_b64 = data.get("audio_data")
            if not audio_b64:
                await self._send_error(websocket, "No audio data provided")
                return
            
            # Decode audio (zero-copy when possible)
            audio_bytes = base64.b64decode(audio_b64)
            
            ull_handler_logger.debug(f"[AUDIO] Processing chunk {chunk_id} from {client_id}: {len(audio_bytes)} bytes")
            
            # Start performance timing
            timing_id = self.performance_monitor.start_timing("ultra_low_latency_processing", {
                "chunk_id": chunk_id,
                "client_id": client_id,
                "audio_size": len(audio_bytes)
            })
            
            # Process audio stream with chunked processor
            first_response_sent = False
            total_responses = 0
            
            async for result in self.processor.process_audio_stream(audio_bytes, chunk_id):
                if result.text_response and result.text_response.strip():
                    # Send text response immediately
                    await self._send_text_response(websocket, client_id, result)
                    
                    # Track first word latency
                    if not first_response_sent:
                        first_word_latency = (time.time() - chunk_start_time) * 1000
                        if first_word_latency <= self.latency_targets["first_word_ms"]:
                            ull_handler_logger.info(f"[PERF] First word in {first_word_latency:.1f}ms ✅")
                        else:
                            ull_handler_logger.warning(f"[PERF] First word in {first_word_latency:.1f}ms ❌ (target: {self.latency_targets['first_word_ms']}ms)")
                        first_response_sent = True
                
                if result.audio_response:
                    # Send audio response immediately
                    await self._send_audio_response(websocket, client_id, result)
                
                total_responses += 1
            
            # End performance timing
            processing_time = self.performance_monitor.end_timing(timing_id)
            
            # Update connection stats
            if client_id in self.active_connections:
                stats = self.active_connections[client_id]
                stats["chunks_processed"] += 1
                stats["total_latency_ms"] += processing_time
            
            # Update manager performance metrics
            self.manager.update_performance_metrics(processing_time, chunk_id)
            
            # Send completion message
            await self._send_completion_message(websocket, client_id, chunk_id, processing_time, total_responses)
            
            ull_handler_logger.debug(f"[COMPLETE] Chunk {chunk_id} processed in {processing_time:.1f}ms with {total_responses} responses")
            
        except Exception as e:
            ull_handler_logger.error(f"[ERROR] Audio chunk processing failed for {client_id}: {e}")
            await self._send_error(websocket, f"Audio processing error: {str(e)}")
    
    async def _send_text_response(self, websocket, client_id: str, result):
        """Send text response with minimal latency"""
        response_msg = {
            "type": "text_response",
            "chunk_id": result.chunk_id,
            "text": result.text_response,
            "processing_time_ms": result.processing_time_ms,
            "latency_breakdown": result.latency_breakdown,
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(response_msg))
    
    async def _send_audio_response(self, websocket, client_id: str, result):
        """Send audio response with minimal latency"""
        if result.audio_response:
            audio_b64 = base64.b64encode(result.audio_response).decode('utf-8')
            
            response_msg = {
                "type": "audio_response",
                "chunk_id": result.chunk_id,
                "audio_data": audio_b64,
                "sample_rate": 24000,  # Kokoro TTS sample rate
                "processing_time_ms": result.processing_time_ms,
                "is_final": result.is_final,
                "timestamp": time.time()
            }
            
            await websocket.send(json.dumps(response_msg))
    
    async def _send_completion_message(self, websocket, client_id: str, chunk_id: str, processing_time: float, response_count: int):
        """Send processing completion message"""
        completion_msg = {
            "type": "processing_complete",
            "chunk_id": chunk_id,
            "processing_time_ms": processing_time,
            "response_count": response_count,
            "latency_target_met": processing_time <= self.latency_targets["end_to_end_ms"],
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(completion_msg))
    
    async def _handle_interruption(self, websocket, client_id: str, data: Dict[str, Any]):
        """Handle user interruption"""
        interruption_type = data.get("interruption_type", "user_speech")
        
        ull_handler_logger.info(f"[INTERRUPT] Handling {interruption_type} from {client_id}")
        
        # Handle interruption in processor
        await self.processor.handle_interruption(interruption_type)
        
        # Update connection stats
        if client_id in self.active_connections:
            self.active_connections[client_id]["interruptions_handled"] += 1
        
        # Send acknowledgment
        interrupt_msg = {
            "type": "interruption_handled",
            "interruption_type": interruption_type,
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(interrupt_msg))
    
    async def _handle_performance_request(self, websocket, client_id: str):
        """Handle performance statistics request"""
        try:
            # Get performance stats from various components
            manager_stats = self.manager.get_performance_metrics()
            processor_stats = self.processor.get_performance_stats()
            connection_stats = self.active_connections.get(client_id, {})
            
            performance_msg = {
                "type": "performance_stats",
                "client_id": client_id,
                "manager_metrics": manager_stats,
                "processor_metrics": processor_stats,
                "connection_stats": connection_stats,
                "latency_targets": self.latency_targets,
                "timestamp": time.time()
            }
            
            await websocket.send(json.dumps(performance_msg))
            
        except Exception as e:
            ull_handler_logger.error(f"[ERROR] Performance request failed for {client_id}: {e}")
            await self._send_error(websocket, f"Performance request error: {str(e)}")
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message to client"""
        error_msg = {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        }
        
        try:
            await websocket.send(json.dumps(error_msg))
        except Exception as e:
            ull_handler_logger.error(f"[ERROR] Failed to send error message: {e}")
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": len(self.connection_stats),
            "latency_targets": self.latency_targets,
            "connection_details": {
                client_id: {
                    "connected_duration": time.time() - stats["connected_at"],
                    "chunks_processed": stats["chunks_processed"],
                    "average_latency": stats["total_latency_ms"] / max(1, stats["chunks_processed"]),
                    "interruptions_handled": stats["interruptions_handled"]
                }
                for client_id, stats in self.active_connections.items()
            }
        }

# Global instance
_ull_handler = None

def get_ull_websocket_handler() -> UltraLowLatencyWebSocketHandler:
    """Get the global ultra-low latency WebSocket handler"""
    global _ull_handler
    if _ull_handler is None:
        _ull_handler = UltraLowLatencyWebSocketHandler()
    return _ull_handler
