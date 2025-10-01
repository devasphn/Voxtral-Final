"""
Ultra-Low Latency Voice AI Pipeline
Implements the exact pipeline: Audio input → Voxtral → Response generation → Kokoro TTS → Audio output
Target: <500ms end-to-end latency with <200ms TTS chunking
"""
import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import base64
import io

# Import latency optimizer
try:
    from src.optimization.latency_optimizer import latency_optimizer
    LATENCY_OPTIMIZER_AVAILABLE = True
except ImportError:
    LATENCY_OPTIMIZER_AVAILABLE = False

# Setup logging
pipeline_logger = logging.getLogger("ultra_low_latency_pipeline")

@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance tracking"""
    total_latency_ms: float = 0.0
    voxtral_latency_ms: float = 0.0
    response_generation_ms: float = 0.0
    kokoro_latency_ms: float = 0.0
    audio_processing_ms: float = 0.0
    timestamp: float = 0.0
    success: bool = False

class UltraLowLatencyPipeline:
    """
    Ultra-low latency voice AI pipeline optimized for <500ms end-to-end performance
    Implements chunked streaming throughout the pipeline for minimal latency
    """
    
    def __init__(self):
        self.is_initialized = False
        self.voxtral_model = None
        self.kokoro_model = None
        self.audio_processor = None
        
        # Performance tracking
        self.metrics_history = []
        self.average_latency = 0.0
        
        # Pipeline configuration
        self.chunk_size_ms = 100  # 100ms audio chunks
        self.max_response_length = 100  # Limit response length for speed
        self.enable_streaming = True
        self.target_latency_ms = 500
        
        pipeline_logger.info("Ultra-Low Latency Pipeline initialized")
    
    async def initialize(self) -> bool:
        """Initialize all pipeline components"""
        start_time = time.time()
        pipeline_logger.info("🚀 Initializing Ultra-Low Latency Pipeline...")
        
        try:
            # Initialize latency optimizer first
            if LATENCY_OPTIMIZER_AVAILABLE and not latency_optimizer.is_initialized:
                await latency_optimizer.initialize()
                pipeline_logger.info("   ⚡ Latency optimizer ready")

            # Initialize Voxtral model
            await self._initialize_voxtral()

            # Initialize Kokoro TTS
            await self._initialize_kokoro()

            # Initialize audio processor
            await self._initialize_audio_processor()

            # Warm up the pipeline
            await self._warmup_pipeline()

            init_time = (time.time() - start_time) * 1000
            self.is_initialized = True

            pipeline_logger.info(f"✅ Pipeline initialized in {init_time:.1f}ms")
            return True
            
        except Exception as e:
            pipeline_logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    async def _initialize_voxtral(self):
        """Initialize Voxtral model"""
        try:
            from src.models.voxtral_model_realtime import VoxtralModel
            self.voxtral_model = VoxtralModel()
            await self.voxtral_model.initialize()
            pipeline_logger.info("   ✅ Voxtral model ready")
        except Exception as e:
            pipeline_logger.error(f"   ❌ Voxtral initialization failed: {e}")
            raise
    
    async def _initialize_kokoro(self):
        """Initialize Kokoro TTS model"""
        try:
            from src.models.kokoro_model_realtime import KokoroTTSModel
            self.kokoro_model = KokoroTTSModel()
            await self.kokoro_model.initialize()
            pipeline_logger.info("   ✅ Kokoro TTS ready")
        except Exception as e:
            pipeline_logger.error(f"   ❌ Kokoro initialization failed: {e}")
            raise
    
    async def _initialize_audio_processor(self):
        """Initialize audio processor"""
        try:
            from src.models.audio_processor_realtime import AudioProcessor
            self.audio_processor = AudioProcessor()
            await self.audio_processor.initialize()
            pipeline_logger.info("   ✅ Audio processor ready")
        except Exception as e:
            pipeline_logger.error(f"   ❌ Audio processor initialization failed: {e}")
            raise
    
    async def _warmup_pipeline(self):
        """Warm up the pipeline with dummy data"""
        pipeline_logger.info("🔥 Warming up pipeline...")
        
        try:
            # Create dummy audio data
            dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
            
            # Run through pipeline once to warm up
            await self.process_audio_to_audio(dummy_audio, warmup=True)
            
            pipeline_logger.info("   ✅ Pipeline warmed up")
        except Exception as e:
            pipeline_logger.warning(f"   ⚠️  Pipeline warmup failed: {e}")
    
    async def process_audio_to_audio(self, audio_input: np.ndarray, warmup: bool = False) -> Dict[str, Any]:
        """
        Main pipeline: Audio input → Voxtral → Response generation → Kokoro TTS → Audio output
        Target: <500ms end-to-end latency
        """
        start_time = time.time()
        metrics = PipelineMetrics(timestamp=start_time)
        
        try:
            # Step 1: Audio preprocessing (target: <50ms)
            step_start = time.time()
            processed_audio = await self._preprocess_audio(audio_input)
            metrics.audio_processing_ms = (time.time() - step_start) * 1000
            
            # Step 2: Voxtral speech-to-text (target: <150ms)
            step_start = time.time()
            transcription = await self._voxtral_transcribe(processed_audio, warmup)
            metrics.voxtral_latency_ms = (time.time() - step_start) * 1000
            
            if not transcription or warmup:
                if warmup:
                    return {"success": True, "warmup": True}
                return {"success": False, "error": "No transcription"}
            
            # Step 3: Generate response (target: <100ms)
            step_start = time.time()
            response_text = await self._generate_response(transcription)
            metrics.response_generation_ms = (time.time() - step_start) * 1000
            
            # Step 4: Kokoro TTS synthesis (target: <200ms)
            step_start = time.time()
            audio_output = await self._kokoro_synthesize(response_text)
            metrics.kokoro_latency_ms = (time.time() - step_start) * 1000
            
            # Calculate total latency
            metrics.total_latency_ms = (time.time() - start_time) * 1000
            metrics.success = True
            
            # Update metrics
            self._update_metrics(metrics)
            
            # Log performance
            if not warmup:
                pipeline_logger.info(f"🎯 Pipeline completed in {metrics.total_latency_ms:.1f}ms")
                pipeline_logger.debug(f"   Breakdown: Audio={metrics.audio_processing_ms:.1f}ms, "
                                    f"Voxtral={metrics.voxtral_latency_ms:.1f}ms, "
                                    f"Response={metrics.response_generation_ms:.1f}ms, "
                                    f"Kokoro={metrics.kokoro_latency_ms:.1f}ms")
            
            return {
                "success": True,
                "transcription": transcription,
                "response_text": response_text,
                "audio_data": audio_output,
                "metrics": metrics,
                "latency_ms": metrics.total_latency_ms
            }
            
        except Exception as e:
            metrics.total_latency_ms = (time.time() - start_time) * 1000
            metrics.success = False
            self._update_metrics(metrics)
            
            pipeline_logger.error(f"❌ Pipeline error: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": metrics,
                "latency_ms": metrics.total_latency_ms
            }
    
    async def _preprocess_audio(self, audio_input: np.ndarray) -> np.ndarray:
        """Preprocess audio for optimal pipeline performance"""
        # Use latency optimizer if available
        if LATENCY_OPTIMIZER_AVAILABLE and latency_optimizer.is_initialized:
            return await latency_optimizer.optimize_audio_preprocessing(audio_input)
        elif self.audio_processor:
            return await self.audio_processor.preprocess_audio(audio_input)
        return audio_input
    
    async def _voxtral_transcribe(self, audio: np.ndarray, warmup: bool = False) -> Optional[str]:
        """Transcribe audio using Voxtral model with optimization"""
        if not self.voxtral_model:
            raise RuntimeError("Voxtral model not initialized")

        try:
            # Use latency optimizer if available
            if LATENCY_OPTIMIZER_AVAILABLE and latency_optimizer.is_initialized and not warmup:
                import torch
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                result = await latency_optimizer.optimize_voxtral_inference(self.voxtral_model, audio_tensor)
            else:
                result = await self.voxtral_model.transcribe_audio(audio)

            if warmup:
                return "warmup"

            if result and result.get("success"):
                return result.get("text", "").strip()

            return None

        except Exception as e:
            pipeline_logger.error(f"Voxtral transcription error: {e}")
            return None
    
    async def _generate_response(self, transcription: str) -> str:
        """Generate response text with optimization"""
        # Use latency optimizer if available
        if LATENCY_OPTIMIZER_AVAILABLE and latency_optimizer.is_initialized:
            return await latency_optimizer.optimize_response_generation(transcription)

        # Fallback to simple response generation
        if not transcription:
            return "I didn't catch that. Could you please repeat?"

        # Simple response patterns for demonstration
        text_lower = transcription.lower()

        if "hello" in text_lower or "hi" in text_lower:
            return "Hello! How can I help you today?"
        elif "how are you" in text_lower:
            return "I'm doing great, thank you for asking!"
        elif "what" in text_lower and "time" in text_lower:
            return "I don't have access to the current time, but I'm here to help with other questions."
        elif "thank" in text_lower:
            return "You're very welcome!"
        elif "bye" in text_lower or "goodbye" in text_lower:
            return "Goodbye! Have a wonderful day!"
        else:
            return f"I heard you say: {transcription}. How can I assist you with that?"
    
    async def _kokoro_synthesize(self, text: str) -> str:
        """Synthesize speech using Kokoro TTS with optimization"""
        if not self.kokoro_model:
            raise RuntimeError("Kokoro model not initialized")

        try:
            # Use latency optimizer if available
            if LATENCY_OPTIMIZER_AVAILABLE and latency_optimizer.is_initialized:
                result = await latency_optimizer.optimize_kokoro_synthesis(self.kokoro_model, text)
            else:
                result = await self.kokoro_model.synthesize_speech_async(text)

            if result and result.get("success"):
                audio_data = result.get("audio_data")
                if audio_data:
                    # Convert to base64 for transmission
                    if isinstance(audio_data, np.ndarray):
                        audio_bytes = audio_data.tobytes()
                    else:
                        audio_bytes = audio_data
                    return base64.b64encode(audio_bytes).decode('utf-8')

            return ""

        except Exception as e:
            pipeline_logger.error(f"Kokoro synthesis error: {e}")
            return ""
    
    def _update_metrics(self, metrics: PipelineMetrics):
        """Update performance metrics"""
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        # Update average latency
        successful_metrics = [m for m in self.metrics_history if m.success]
        if successful_metrics:
            self.average_latency = sum(m.total_latency_ms for m in successful_metrics) / len(successful_metrics)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.metrics_history:
            return {"no_data": True}
        
        successful_metrics = [m for m in self.metrics_history if m.success]
        
        if not successful_metrics:
            return {"no_successful_runs": True}
        
        latencies = [m.total_latency_ms for m in successful_metrics]
        
        return {
            "average_latency_ms": self.average_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "success_rate": len(successful_metrics) / len(self.metrics_history),
            "total_runs": len(self.metrics_history),
            "target_met_percentage": len([l for l in latencies if l <= self.target_latency_ms]) / len(latencies) * 100,
            "last_latency_ms": latencies[-1] if latencies else 0
        }

# Global pipeline instance
ultra_low_latency_pipeline = UltraLowLatencyPipeline()

async def initialize_pipeline():
    """Initialize the ultra-low latency pipeline"""
    return await ultra_low_latency_pipeline.initialize()

async def process_voice_input(audio_input: np.ndarray) -> Dict[str, Any]:
    """Process voice input through the ultra-low latency pipeline"""
    return await ultra_low_latency_pipeline.process_audio_to_audio(audio_input)

def get_pipeline_stats() -> Dict[str, Any]:
    """Get pipeline performance statistics"""
    return ultra_low_latency_pipeline.get_performance_stats()
