"""
Example client for testing Voxtral Real-time Streaming
Demonstrates WebSocket and TCP connections
"""
import asyncio
import websockets
import json
import base64
import numpy as np
import socket
import struct
from typing import Optional
import argparse

class VoxtralStreamingClient:
    """Client for testing Voxtral streaming server"""
    
    def __init__(self, host: str = "localhost"):
        self.host = host
        self.websocket_port = 8000  # UI server WebSocket endpoint
        self.tcp_port = 8766
        
    def generate_test_audio(self, duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate test audio signal"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a simple sine wave (440 Hz - A4 note)
        frequency = 440
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return audio
    
    async def test_websocket(self, audio_data: Optional[np.ndarray] = None):
        """Test WebSocket streaming"""
        print(f"Testing WebSocket connection to {self.host}:{self.websocket_port}")
        
        if audio_data is None:
            audio_data = self.generate_test_audio()
        
        try:
            uri = f"ws://{self.host}:{self.websocket_port}/ws"  # Use UI server WebSocket endpoint
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket server")
                
                # Wait for welcome message
                welcome = await websocket.recv()
                print(f"Server: {welcome}")
                
                # Send test audio
                audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                message = {
                    "type": "audio_chunk",  # Use new message type
                    "audio_data": audio_b64,
                    "mode": "conversation",
                    "chunk_id": 1,
                    "timestamp": time.time()
                }
                
                print("Sending audio data...")
                await websocket.send(json.dumps(message))
                
                # Receive response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                print(f"Response: {response_data}")
                print(f"Transcription: {response_data.get('text', 'No text')}")
                print(f"Processing time: {response_data.get('processing_time_ms', 'Unknown')}ms")
                
        except Exception as e:
            print(f"WebSocket test failed: {e}")
    
    def test_tcp(self, audio_data: Optional[np.ndarray] = None):
        """Test TCP streaming"""
        print(f"Testing TCP connection to {self.host}:{self.tcp_port}")
        
        if audio_data is None:
            audio_data = self.generate_test_audio()
        
        try:
            # Connect to TCP server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.tcp_port))
            print("Connected to TCP server")
            
            # Receive welcome message
            welcome_length = struct.unpack('!I', sock.recv(4))[0]
            welcome_data = sock.recv(welcome_length).decode('utf-8')
            welcome_message = json.loads(welcome_data)
            print(f"Server: {welcome_message}")
            
            # Send test audio
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            message = {
                "type": "audio",
                "audio_data": audio_b64,
                "mode": "transcribe",
                "prompt": "Transcribe this test audio"
            }
            
            # Send message with length prefix
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            
            print("Sending audio data...")
            sock.send(struct.pack('!I', len(message_bytes)) + message_bytes)
            
            # Receive response
            response_length = struct.unpack('!I', sock.recv(4))[0]
            response_data = sock.recv(response_length).decode('utf-8')
            response = json.loads(response_data)
            
            print(f"Response: {response}")
            print(f"Transcription: {response.get('text', 'No text')}")
            print(f"Processing time: {response.get('processing_time_ms', 'Unknown')}ms")
            
            sock.close()
            
        except Exception as e:
            print(f"TCP test failed: {e}")
    
    async def test_health_check(self):
        """Test health check endpoint"""
        import aiohttp
        
        print(f"Testing health check at http://{self.host}:8005")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test basic health
                async with session.get(f"http://{self.host}:8005/health") as response:
                    health_data = await response.json()
                    print(f"Health: {health_data}")
                
                # Test detailed status
                async with session.get(f"http://{self.host}:8005/status") as response:
                    status_data = await response.json()
                    print(f"Status: {status_data}")
                    
        except Exception as e:
            print(f"Health check failed: {e}")
    
    def test_web_ui(self):
        """Test web UI accessibility"""
        import requests
        
        print(f"Testing web UI at http://{self.host}:8000")
        
        try:
            response = requests.get(f"http://{self.host}:8000", timeout=10)
            if response.status_code == 200:
                print("✓ Web UI is accessible")
            else:
                print(f"✗ Web UI returned status code: {response.status_code}")
                
        except Exception as e:
            print(f"Web UI test failed: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Test Voxtral streaming server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--test", choices=["websocket", "tcp", "health", "web", "all"], 
                       default="all", help="Test to run")
    
    args = parser.parse_args()
    
    client = VoxtralStreamingClient(args.host)
    
    print("=" * 50)
    print("Voxtral Streaming Client Test")
    print("=" * 50)
    
    if args.test in ["websocket", "all"]:
        print("\n🔌 Testing WebSocket Streaming")
        print("-" * 30)
        await client.test_websocket()
    
    if args.test in ["tcp", "all"]:
        print("\n🌐 Testing TCP Streaming")
        print("-" * 30)
        client.test_tcp()
    
    if args.test in ["health", "all"]:
        print("\n💚 Testing Health Check")
        print("-" * 30)
        await client.test_health_check()
    
    if args.test in ["web", "all"]:
        print("\n🌍 Testing Web UI")
        print("-" * 30)
        client.test_web_ui()
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import aiohttp
        import requests
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp", "requests"])
        import aiohttp
        import requests
    
    asyncio.run(main())