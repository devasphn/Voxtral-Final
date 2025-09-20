"""
Streaming Coordinator for Ultra-Low Latency Voice Agent
Manages the streaming pipeline between Voxtral, TTS, and user interruption
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum

# Setup logging
streaming_logger = logging.getLogger("streaming_coordinator")
streaming_logger.setLevel(logging.INFO)

class StreamingState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"

@dataclass
class StreamingChunk:
    """Data structure for streaming chunks"""
    type: str  # 'words', 'audio', 'complete', 'error'
    content: Any
    timestamp: float
    chunk_id: str
    metadata: Dict[str, Any] = None

class StreamingCoordinator:
    """
    Ultra-Low Latency Streaming Coordinator
    Manages real-time streaming between ASR → Voxtral → TTS with interruption support
    """
    
    def __init__(self):
        self.state = StreamingState.IDLE
        self.current_session_id = None
        self.word_buffer = deque(maxlen=100)  # Buffer for streaming words
        self.tts_queue = asyncio.Queue()
        self.interruption_detected = False
        self.active_tts_tasks = set()
        self.performance_metrics = {
            'first_word_latency': [],
            'word_to_audio_latency': [],
            'interruption_response_time': [],
            'total_session_latency': []
        }
        
        # Streaming configuration
        self.config = {
            'words_trigger_threshold': 2,  # Start TTS after 2 words
            'max_word_buffer_size': 50,    # Max words to buffer
            'interruption_timeout_ms': 100, # Max time to detect interruption
            'tts_chunk_size_ms': 200,      # TTS chunk size for streaming
            'max_concurrent_tts': 3        # Max concurrent TTS tasks
        }
        
        # Callbacks for external integration
        self.on_words_ready: Optional[Callable] = None
        self.on_audio_ready: Optional[Callable] = None
        self.on_interruption: Optional[Callable] = None
        self.on_session_complete: Optional[Callable] = None
    
    async def start_streaming_session(self, session_id: str = None) -> str:
        """Start a new streaming session"""
        session_id = session_id or f"session_{int(time.time() * 1000)}"
        self.current_session_id = session_id
        self.state = StreamingState.LISTENING
        self.interruption_detected = False
        self.word_buffer.clear()
        
        # Clear TTS queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        streaming_logger.info(f"[VAD] Started streaming session: {session_id}")
        return session_id
    
    async def process_voxtral_stream(self, voxtral_stream: AsyncGenerator) -> AsyncGenerator[StreamingChunk, None]:
        """
        Process streaming tokens from Voxtral and coordinate TTS generation
        """
        if self.state != StreamingState.LISTENING:
            streaming_logger.warning(f"[WARN] Cannot process stream in state: {self.state}")
            return
        
        self.state = StreamingState.PROCESSING
        session_start_time = time.time()
        first_word_sent = False
        words_sent_count = 0
        
        try:
            async for token_data in voxtral_stream:
                # Enhanced error handling for token data
                if not isinstance(token_data, dict):
                    streaming_logger.warning(f"[WARN] Invalid token data type: {type(token_data)}")
                    continue

                # Check for interruption
                if self.interruption_detected:
                    streaming_logger.info(f"[EMOJI] Stream interrupted for session {self.current_session_id}")
                    yield StreamingChunk(
                        type='interrupted',
                        content={'reason': 'user_interruption'},
                        timestamp=time.time(),
                        chunk_id=self.current_session_id
                    )
                    break

                if token_data.get('type') == 'words':
                    words_text = token_data.get('text', '').strip()
                    # Enhanced text validation
                    if words_text and isinstance(words_text, str):
                        # Add to word buffer with enhanced validation
                        try:
                            self.word_buffer.append({
                                'text': words_text,
                                'timestamp': time.time(),
                                'tokens': token_data.get('tokens', []),
                                'word_count': token_data.get('word_count', 0)
                            })
                        except Exception as buffer_error:
                            streaming_logger.error(f"[ERROR] Word buffer error: {buffer_error}")
                            continue

                        # Dynamic TTS triggering based on content and timing
                        trigger_threshold = self.config['words_trigger_threshold']
                        buffer_size = len(self.word_buffer)

                        # Trigger TTS if we have enough words OR if enough time has passed
                        time_since_last = time.time() - (self.word_buffer[-2]['timestamp'] if len(self.word_buffer) > 1 else session_start_time)
                        should_trigger = (buffer_size >= trigger_threshold or
                                        (buffer_size >= 1 and time_since_last > 0.5))  # 500ms timeout

                        if should_trigger:
                            # Collect words for TTS
                            words_for_tts = []
                            words_to_collect = min(trigger_threshold, buffer_size)

                            for _ in range(words_to_collect):
                                if self.word_buffer:
                                    words_for_tts.append(self.word_buffer.popleft())

                            # Combine words into text with validation
                            try:
                                combined_text = ' '.join(word['text'] for word in words_for_tts if isinstance(word.get('text'), str))
                            except Exception as combine_error:
                                streaming_logger.error(f"[ERROR] Text combination error: {combine_error}")
                                continue
                            
                            # Track first word latency
                            if not first_word_sent:
                                first_word_latency = (time.time() - session_start_time) * 1000
                                self.performance_metrics['first_word_latency'].append(first_word_latency)
                                first_word_sent = True
                                streaming_logger.info(f"[FAST] First words ready in {first_word_latency:.1f}ms: '{combined_text}'")
                            
                            # Send words for TTS
                            chunk = StreamingChunk(
                                type='words_ready',
                                content={
                                    'text': combined_text,
                                    'word_count': len(words_for_tts),
                                    'sequence_number': words_sent_count
                                },
                                timestamp=time.time(),
                                chunk_id=f"{self.current_session_id}_words_{words_sent_count}",
                                metadata={
                                    'session_id': self.current_session_id,
                                    'is_first_chunk': not first_word_sent,
                                    'total_words_sent': words_sent_count + len(words_for_tts)
                                }
                            )
                            
                            words_sent_count += len(words_for_tts)
                            yield chunk
                            
                            # Trigger TTS generation asynchronously
                            if self.on_words_ready:
                                asyncio.create_task(self.on_words_ready(chunk))
                
                elif token_data.get('type') == 'complete':
                    # Send any remaining words
                    if self.word_buffer:
                        remaining_words = []
                        while self.word_buffer:
                            remaining_words.append(self.word_buffer.popleft())
                        
                        if remaining_words:
                            combined_text = ' '.join(word['text'] for word in remaining_words)
                            chunk = StreamingChunk(
                                type='words_ready',
                                content={
                                    'text': combined_text,
                                    'word_count': len(remaining_words),
                                    'sequence_number': words_sent_count,
                                    'is_final': True
                                },
                                timestamp=time.time(),
                                chunk_id=f"{self.current_session_id}_words_final",
                                metadata={
                                    'session_id': self.current_session_id,
                                    'is_final_chunk': True
                                }
                            )
                            yield chunk
                            
                            if self.on_words_ready:
                                asyncio.create_task(self.on_words_ready(chunk))
                    
                    # Send completion
                    total_latency = (time.time() - session_start_time) * 1000
                    self.performance_metrics['total_session_latency'].append(total_latency)
                    
                    yield StreamingChunk(
                        type='session_complete',
                        content={
                            'total_words_sent': words_sent_count,
                            'total_latency_ms': total_latency,
                            'voxtral_data': token_data
                        },
                        timestamp=time.time(),
                        chunk_id=f"{self.current_session_id}_complete"
                    )
                    
                    if self.on_session_complete:
                        asyncio.create_task(self.on_session_complete(self.current_session_id))
                    
                    break
                
                elif token_data.get('type') == 'error':
                    streaming_logger.error(f"[ERROR] Voxtral stream error: {token_data.get('error')}")
                    yield StreamingChunk(
                        type='error',
                        content={'error': token_data.get('error')},
                        timestamp=time.time(),
                        chunk_id=self.current_session_id
                    )
                    break
        
        except Exception as e:
            streaming_logger.error(f"[ERROR] Error processing Voxtral stream: {e}")
            streaming_logger.error(f"[ERROR] Error details: {type(e).__name__}: {str(e)}")

            # Enhanced error reporting
            yield StreamingChunk(
                type='error',
                content={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'session_id': self.current_session_id,
                    'words_processed': words_sent_count
                },
                timestamp=time.time(),
                chunk_id=self.current_session_id
            )
            yield StreamingChunk(
                type='error',
                content={'error': str(e)},
                timestamp=time.time(),
                chunk_id=self.current_session_id
            )
        
        finally:
            self.state = StreamingState.IDLE
    
    async def handle_interruption(self, interruption_source: str = "user_speech"):
        """Handle user interruption - cancel ongoing TTS and reset state"""
        interruption_start = time.time()
        
        streaming_logger.info(f"[EMOJI] Interruption detected from {interruption_source}")
        self.interruption_detected = True
        self.state = StreamingState.INTERRUPTED
        
        # Cancel all active TTS tasks
        for task in list(self.active_tts_tasks):
            if not task.done():
                task.cancel()
                streaming_logger.debug(f"[EMOJI] Cancelled TTS task: {task}")
        
        self.active_tts_tasks.clear()
        
        # Clear buffers
        self.word_buffer.clear()
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Track interruption response time
        response_time = (time.time() - interruption_start) * 1000
        self.performance_metrics['interruption_response_time'].append(response_time)
        
        streaming_logger.info(f"[OK] Interruption handled in {response_time:.1f}ms")
        
        # Callback for external handling
        if self.on_interruption:
            await self.on_interruption(interruption_source, response_time)
        
        # Reset state for new session
        await asyncio.sleep(0.1)  # Brief pause before allowing new session
        self.state = StreamingState.IDLE
        self.interruption_detected = False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[metric_name] = {
                    'count': len(values),
                    'avg_ms': sum(values) / len(values),
                    'min_ms': min(values),
                    'max_ms': max(values),
                    'recent_avg_ms': sum(values[-5:]) / min(len(values), 5)
                }
            else:
                metrics[metric_name] = {'count': 0, 'avg_ms': 0}
        
        metrics['current_state'] = self.state.value
        metrics['active_tts_tasks'] = len(self.active_tts_tasks)
        metrics['word_buffer_size'] = len(self.word_buffer)
        
        return metrics
    
    def register_callbacks(self, 
                          on_words_ready: Callable = None,
                          on_audio_ready: Callable = None, 
                          on_interruption: Callable = None,
                          on_session_complete: Callable = None):
        """Register callbacks for external integration"""
        self.on_words_ready = on_words_ready
        self.on_audio_ready = on_audio_ready
        self.on_interruption = on_interruption
        self.on_session_complete = on_session_complete
        
        streaming_logger.info("[OK] Streaming callbacks registered")

# Global streaming coordinator instance
streaming_coordinator = StreamingCoordinator()
