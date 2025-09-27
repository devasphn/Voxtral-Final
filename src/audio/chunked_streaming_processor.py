"""
Chunked Audio Streaming Processor for Ultra-Low Latency Speech-to-Speech
Implements 512-1024 sample chunked processing with overlap and VAD
"""

import asyncio
import time
import logging
import torch
import numpy as np
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple
from collections import deque
import threading
from dataclasses import dataclass

# Setup logging
chunk_logger = logging.getLogger("chunked_streaming")
chunk_logger.setLevel(logging.INFO)

@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: torch.Tensor
    chunk_id: str
    timestamp: float
    sample_rate: int
    is_voice: bool
    confidence: float
    sequence_number: int

@dataclass
class ProcessingResult:
    """Result of chunk processing"""
    chunk_id: str
    text_response: str
    audio_response: Optional[bytes]
    processing_time_ms: float
    latency_breakdown: Dict[str, float]
    is_final: bool

class ChunkedStreamingProcessor:
    """
    Ultra-low latency chunked audio streaming processor
    Features:
    - 512-1024 sample chunks with 128 sample overlap
    - Real-time VAD with <50ms detection
    - Parallel processing pipeline
    - Interruption handling
    - Zero-copy tensor operations
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap_size = 128  # 8ms overlap at 16kHz
        self.vad_threshold = 0.5
        
        # Processing buffers
        self.audio_buffer = deque(maxlen=10)  # Keep last 10 chunks
        self.processing_queue = asyncio.Queue(maxsize=5)
        self.output_queue = asyncio.Queue(maxsize=10)
        
        # VAD state
        self.vad_history = deque(maxlen=5)  # Keep last 5 VAD decisions
        self.speech_active = False
        self.silence_counter = 0
        self.speech_start_time = None
        
        # Performance tracking
        self.chunk_counter = 0
        self.processing_times = deque(maxlen=100)
        self.vad_times = deque(maxlen=100)
        
        # Threading for parallel processing
        self.processing_lock = threading.Lock()
        self.vad_lock = threading.Lock()
        
        chunk_logger.info(f"ChunkedStreamingProcessor initialized - {chunk_size} samples, {sample_rate}Hz")
    
    async def process_audio_stream(self, audio_data: bytes, chunk_id: str) -> AsyncGenerator[ProcessingResult, None]:
        """
        Process incoming audio stream in chunks
        Yields ProcessingResult objects as they become available
        """
        try:
            # Convert audio bytes to tensor
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_array)
            
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            chunk_logger.debug(f"[STREAM] Processing audio stream: {len(audio_array)} samples")
            
            # Split into overlapping chunks
            chunks = self._create_overlapping_chunks(audio_tensor, chunk_id)
            
            # Process chunks in parallel
            async for result in self._process_chunks_parallel(chunks):
                yield result
                
        except Exception as e:
            chunk_logger.error(f"[ERROR] Audio stream processing failed: {e}")
            yield ProcessingResult(
                chunk_id=chunk_id,
                text_response="",
                audio_response=None,
                processing_time_ms=0.0,
                latency_breakdown={"error": 0.0},
                is_final=True
            )
    
    def _create_overlapping_chunks(self, audio_tensor: torch.Tensor, base_chunk_id: str) -> List[AudioChunk]:
        """Create overlapping audio chunks for smooth processing"""
        chunks = []
        total_samples = len(audio_tensor)
        step_size = self.chunk_size - self.overlap_size
        
        for i in range(0, total_samples - self.overlap_size, step_size):
            end_idx = min(i + self.chunk_size, total_samples)
            chunk_data = audio_tensor[i:end_idx]
            
            # Pad if necessary
            if len(chunk_data) < self.chunk_size:
                padding = torch.zeros(self.chunk_size - len(chunk_data), device=chunk_data.device)
                chunk_data = torch.cat([chunk_data, padding])
            
            chunk = AudioChunk(
                data=chunk_data,
                chunk_id=f"{base_chunk_id}_chunk_{len(chunks)}",
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                is_voice=False,  # Will be determined by VAD
                confidence=0.0,
                sequence_number=len(chunks)
            )
            
            chunks.append(chunk)
        
        chunk_logger.debug(f"[CHUNK] Created {len(chunks)} overlapping chunks from {total_samples} samples")
        return chunks
    
    async def _process_chunks_parallel(self, chunks: List[AudioChunk]) -> AsyncGenerator[ProcessingResult, None]:
        """Process chunks in parallel with VAD and speech processing"""
        
        # Create processing tasks
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._process_single_chunk(chunk))
            tasks.append(task)
        
        # Process chunks as they complete
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                if result:
                    yield result
            except Exception as e:
                chunk_logger.error(f"[ERROR] Chunk processing failed: {e}")
    
    async def _process_single_chunk(self, chunk: AudioChunk) -> Optional[ProcessingResult]:
        """Process a single audio chunk with VAD and speech recognition"""
        start_time = time.time()
        
        try:
            # Step 1: Voice Activity Detection (target: <50ms)
            vad_start = time.time()
            is_voice, confidence = await self._detect_voice_activity(chunk)
            vad_time = (time.time() - vad_start) * 1000
            
            chunk.is_voice = is_voice
            chunk.confidence = confidence
            
            # Update VAD history
            with self.vad_lock:
                self.vad_history.append((is_voice, confidence))
                self._update_speech_state(is_voice)
            
            # Skip processing if no voice detected
            if not is_voice:
                return None
            
            # Step 2: Speech Processing (target: <100ms)
            speech_start = time.time()
            text_response = await self._process_speech(chunk)
            speech_time = (time.time() - speech_start) * 1000
            
            # Step 3: TTS Generation (target: <150ms)
            tts_start = time.time()
            audio_response = await self._generate_speech(text_response, chunk.chunk_id)
            tts_time = (time.time() - tts_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            # Track performance
            self.processing_times.append(total_time)
            self.vad_times.append(vad_time)
            
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                text_response=text_response,
                audio_response=audio_response,
                processing_time_ms=total_time,
                latency_breakdown={
                    "vad_ms": vad_time,
                    "speech_processing_ms": speech_time,
                    "tts_generation_ms": tts_time,
                    "total_ms": total_time
                },
                is_final=chunk.sequence_number == 0  # Mark first chunk as final for now
            )
            
        except Exception as e:
            chunk_logger.error(f"[ERROR] Single chunk processing failed: {e}")
            return None
    
    async def _detect_voice_activity(self, chunk: AudioChunk) -> Tuple[bool, float]:
        """Ultra-fast voice activity detection"""
        try:
            # Simple but fast VAD based on energy and spectral features
            audio_data = chunk.data.cpu().numpy() if chunk.data.is_cuda else chunk.data.numpy()
            
            # Energy-based detection
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Spectral centroid (simple frequency analysis)
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            # Voice detection heuristics
            energy_threshold = 0.01  # Minimum energy for voice
            freq_threshold = (300, 3400)  # Voice frequency range
            
            is_voice = (
                rms_energy > energy_threshold and
                freq_threshold[0] < spectral_centroid < freq_threshold[1]
            )
            
            # Confidence based on energy and frequency characteristics
            confidence = min(1.0, rms_energy * 10) * (1.0 if is_voice else 0.1)
            
            return is_voice, confidence
            
        except Exception as e:
            chunk_logger.error(f"[VAD] Voice detection failed: {e}")
            return False, 0.0
    
    def _update_speech_state(self, is_voice: bool):
        """Update speech state for interruption handling"""
        if is_voice:
            if not self.speech_active:
                self.speech_active = True
                self.speech_start_time = time.time()
                chunk_logger.debug("[VAD] Speech started")
            self.silence_counter = 0
        else:
            self.silence_counter += 1
            # End speech after 3 consecutive silent chunks
            if self.speech_active and self.silence_counter >= 3:
                self.speech_active = False
                speech_duration = time.time() - self.speech_start_time if self.speech_start_time else 0
                chunk_logger.debug(f"[VAD] Speech ended after {speech_duration:.2f}s")
    
    async def _process_speech(self, chunk: AudioChunk) -> str:
        """Process speech chunk and generate text response"""
        try:
            # Import here to avoid circular imports
            from src.models.ultra_low_latency_manager import get_ultra_low_latency_manager
            
            manager = get_ultra_low_latency_manager()
            voxtral_model = await manager.get_voxtral_model()
            
            # Process with optimized settings
            result = await voxtral_model.process_realtime_chunk(
                chunk.data,
                chunk.chunk_id,
                mode="conversation",
                prompt=""
            )
            
            return result.get("response", "")
            
        except Exception as e:
            chunk_logger.error(f"[SPEECH] Speech processing failed: {e}")
            return ""
    
    async def _generate_speech(self, text: str, chunk_id: str) -> Optional[bytes]:
        """Generate speech audio from text"""
        try:
            if not text or not text.strip():
                return None
            
            # Import here to avoid circular imports
            from src.models.ultra_low_latency_manager import get_ultra_low_latency_manager
            
            manager = get_ultra_low_latency_manager()
            kokoro_model = await manager.get_kokoro_model()
            
            # Generate speech with streaming
            result = await kokoro_model.synthesize_speech(
                text=text,
                voice="hm_omega"
            )
            
            if result.get("success"):
                return result.get("audio_data")
            else:
                chunk_logger.error(f"[TTS] Speech generation failed: {result.get('error')}")
                return None
                
        except Exception as e:
            chunk_logger.error(f"[TTS] Speech generation failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.processing_times:
            return {"status": "no_data"}
        
        processing_times = list(self.processing_times)
        vad_times = list(self.vad_times)
        
        return {
            "chunks_processed": len(processing_times),
            "average_processing_ms": np.mean(processing_times),
            "median_processing_ms": np.median(processing_times),
            "p95_processing_ms": np.percentile(processing_times, 95),
            "max_processing_ms": np.max(processing_times),
            "average_vad_ms": np.mean(vad_times) if vad_times else 0,
            "speech_active": self.speech_active,
            "silence_counter": self.silence_counter,
            "target_latency_ms": 200,
            "latency_violations": sum(1 for t in processing_times if t > 200)
        }
    
    async def handle_interruption(self, interruption_type: str = "user_speech"):
        """Handle interruption in the processing pipeline"""
        chunk_logger.info(f"[INTERRUPT] Handling interruption: {interruption_type}")
        
        # Clear processing queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset speech state
        with self.vad_lock:
            self.speech_active = False
            self.silence_counter = 0
            self.speech_start_time = None
        
        chunk_logger.info("[INTERRUPT] Processing pipeline reset for interruption")

# Global instance
_chunked_processor = None

def get_chunked_processor() -> ChunkedStreamingProcessor:
    """Get the global chunked streaming processor"""
    global _chunked_processor
    if _chunked_processor is None:
        _chunked_processor = ChunkedStreamingProcessor()
    return _chunked_processor
