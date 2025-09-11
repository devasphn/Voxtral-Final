"""
Test suite for Voxtral streaming functionality
"""
import asyncio
import pytest
import numpy as np
import torch
import websockets
import json
import base64
from unittest.mock import AsyncMock, MagicMock, patch

# Test configurations
TEST_AUDIO_DURATION = 2.0  # seconds
TEST_SAMPLE_RATE = 16000

@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing"""
    duration = TEST_AUDIO_DURATION
    sample_rate = TEST_SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio

@pytest.fixture  
def audio_base64(sample_audio):
    """Convert sample audio to base64 for testing"""
    audio_bytes = sample_audio.tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')

class TestAudioProcessor:
    """Test cases for AudioProcessor"""
    
    def test_audio_preprocessing(self, sample_audio):
        """Test audio preprocessing functionality"""
        from src.models.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        
        # Test preprocessing
        audio_tensor = processor.preprocess_audio(sample_audio)
        
        assert isinstance(audio_tensor, torch.Tensor)
        assert audio_tensor.dtype == torch.float32
        assert len(audio_tensor.shape) == 1  # Should be mono
        assert torch.max(torch.abs(audio_tensor)) <= 1.0  # Normalized
    
    def test_log_mel_spectrogram_generation(self, sample_audio):
        """Test log-mel spectrogram generation"""
        from src.models.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        audio_tensor = processor.preprocess_audio(sample_audio)
        
        # Generate log-mel spectrogram
        log_mel_spec = processor.generate_log_mel_spectrogram(audio_tensor)
        
        assert isinstance(log_mel_spec, torch.Tensor)
        assert len(log_mel_spec.shape) == 2  # Time x Frequency
        assert log_mel_spec.shape[0] == 128  # n_mels from config
    
    def test_audio_chunking(self, sample_audio):
        """Test audio chunking functionality"""
        from src.models.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        audio_tensor = processor.preprocess_audio(sample_audio)
        
        # Test chunking
        chunks = processor.chunk_audio(audio_tensor, chunk_duration=1.0)
        
        assert len(chunks) == 2  # 2 second audio with 1 second chunks
        assert all(isinstance(chunk, torch.Tensor) for chunk in chunks)
        assert all(len(chunk) == TEST_SAMPLE_RATE for chunk in chunks)  # Each chunk is 1 second
    
    def test_audio_validation(self):
        """Test audio format validation"""
        from src.models.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        
        # Valid audio
        valid_audio = np.random.randn(1000).astype(np.float32)
        assert processor.validate_audio_format(valid_audio) == True
        
        # Empty audio
        empty_audio = np.array([])
        assert processor.validate_audio_format(empty_audio) == False
        
        # Audio with NaN
        nan_audio = np.array([1.0, np.nan, 3.0])
        assert processor.validate_audio_format(nan_audio) == False
        
        # Silent audio (all zeros)
        silent_audio = np.zeros(1000)
        assert processor.validate_audio_format(silent_audio) == False

class TestVoxtralModel:
    """Test cases for VoxtralModel"""
    
    @pytest.mark.asyncio
    @patch('src.models.voxtral_model.VoxtralForConditionalGeneration')
    @patch('src.models.voxtral_model.AutoProcessor')
    async def test_model_initialization(self, mock_processor, mock_model):
        """Test model initialization"""
        from src.models.voxtral_model import VoxtralModel
        
        # Mock the model and processor
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        model = VoxtralModel()
        await model.initialize()
        
        assert model.is_initialized == True
        assert model.processor is not None
        assert model.model is not None
    
    @pytest.mark.asyncio
    async def test_audio_processing_mock(self, sample_audio):
        """Test audio processing with mocked model"""
        from src.models.voxtral_model import VoxtralModel
        
        model = VoxtralModel()
        model.is_initialized = True
        
        # Mock the model components
        model.processor = MagicMock()
        model.model = MagicMock()
        
        # Mock processor methods
        model.processor.apply_chat_template.return_value = MagicMock()
        model.processor.batch_decode.return_value = ["Test transcription"]
        
        # Mock model generate method
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        model.model.generate.return_value = mock_outputs
        
        # Test processing
        audio_tensor = torch.from_numpy(sample_audio)
        result = await model.process_audio_stream(audio_tensor, "Transcribe this")
        
        assert isinstance(result, str)
        assert result == "Test transcription"

class TestWebSocketServer:
    """Test cases for WebSocket server"""
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, audio_base64):
        """Test WebSocket message handling"""
        # WebSocket server tests - using UI server WebSocket endpoint instead
# from src.streaming.websocket_server import WebSocketServer
        
        server = WebSocketServer()
        
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Test message handling
        test_message = {
            "type": "ping"
        }
        
        await server.handle_message(mock_websocket, json.dumps(test_message))
        
        # Verify pong response was sent
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["type"] == "pong"

class TestTCPServer:
    """Test cases for TCP server"""
    
    @pytest.mark.asyncio
    async def test_tcp_message_reading(self):
        """Test TCP message reading functionality"""
        from src.streaming.tcp_server import TCPStreamingServer
        import struct
        
        server = TCPStreamingServer()
        
        # Create mock reader
        mock_reader = AsyncMock()
        
        # Prepare test data
        test_data = {"type": "ping"}
        test_json = json.dumps(test_data)
        test_bytes = test_json.encode('utf-8')
        
        # Mock the reading process
        mock_reader.readexactly.side_effect = [
            struct.pack('!I', len(test_bytes)),  # Length prefix
            test_bytes  # Actual data
        ]
        
        # Test reading
        result = await server.read_message(mock_reader)
        
        assert result == test_data

class TestConfiguration:
    """Test cases for configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        from src.utils.config import Config, load_config
        
        # Test default config
        config = Config()
        
        assert config.server.host == "0.0.0.0"
        assert config.server.http_port == 8000
        assert config.server.health_port == 8005
        assert config.audio.sample_rate == 16000
        assert config.spectrogram.n_mels == 128

class TestHealthCheck:
    """Test cases for health check API"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        from src.api.health_check import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

def test_integration_audio_pipeline(sample_audio):
    """Integration test for complete audio processing pipeline"""
    from src.models.audio_processor import AudioProcessor
    
    processor = AudioProcessor()
    
    # Test complete pipeline
    audio_tensor = processor.preprocess_audio(sample_audio)
    log_mel_spec = processor.generate_log_mel_spectrogram(audio_tensor)
    
    # Verify output shapes and types
    assert isinstance(audio_tensor, torch.Tensor)
    assert isinstance(log_mel_spec, torch.Tensor)
    assert log_mel_spec.shape[0] == 128  # n_mels
    
    # Test streaming processing
    processed = processor.process_streaming_audio(sample_audio)
    assert isinstance(processed, torch.Tensor)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
