#!/usr/bin/env python3
"""
Comprehensive Streaming Voice Agent Test Suite
Tests streaming functionality, interruption detection, and performance targets
"""

import asyncio
import websockets
import json
import time
import base64
import numpy as np
import logging
from typing import Dict, List, Any
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streaming_test")

class StreamingVoiceAgentTester:
    """Comprehensive tester for streaming voice agent functionality"""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.test_results = {}
        self.performance_metrics = {
            'first_word_latency': [],
            'word_to_audio_latency': [],
            'interruption_response_time': [],
            'total_streaming_latency': [],
            'token_generation_rate': []
        }
    
    def generate_test_audio(self, duration: float = 2.0, frequency: int = 440) -> bytes:
        """Generate test audio for streaming tests"""
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate speech-like audio with multiple frequencies
        audio = (
            0.3 * np.sin(2 * np.pi * frequency * t) +
            0.2 * np.sin(2 * np.pi * (frequency * 1.5) * t) +
            0.1 * np.sin(2 * np.pi * (frequency * 2) * t)
        )
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, len(audio))
        audio = audio + noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32).tobytes()
    
    async def test_streaming_response(self) -> Dict[str, Any]:
        """Test streaming token generation and word-level TTS triggering"""
        logger.info("🧪 Testing streaming response generation...")
        
        test_start = time.time()
        results = {
            'success': False,
            'first_word_latency_ms': None,
            'total_words_received': 0,
            'audio_chunks_received': 0,
            'streaming_complete': False,
            'error': None
        }
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Generate test audio
                audio_data = self.generate_test_audio(duration=3.0)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                # Send streaming request
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",  # Enable streaming mode
                    "streaming": True,
                    "chunk_id": f"stream_test_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                logger.info("📤 Sent streaming audio request")
                
                first_word_received = False
                words_received = []
                audio_chunks = []
                
                # Listen for streaming responses
                timeout = 30  # 30 second timeout
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'streaming_words':
                            if not first_word_received:
                                first_word_latency = (time.time() - test_start) * 1000
                                results['first_word_latency_ms'] = first_word_latency
                                first_word_received = True
                                logger.info(f"⚡ First words received in {first_word_latency:.1f}ms: '{data.get('text', '')}'")
                            
                            words_received.append(data.get('text', ''))
                            logger.info(f"📝 Words: '{data.get('text', '')}'")
                        
                        elif data.get('type') == 'streaming_audio':
                            audio_chunks.append(data)
                            if len(audio_chunks) % 5 == 0:  # Log every 5th chunk
                                logger.info(f"🎵 Audio chunk {len(audio_chunks)} received")
                        
                        elif data.get('type') == 'streaming_complete':
                            results['streaming_complete'] = True
                            results['total_words_received'] = len(words_received)
                            results['audio_chunks_received'] = len(audio_chunks)
                            logger.info(f"✅ Streaming complete: {len(words_received)} words, {len(audio_chunks)} audio chunks")
                            break
                        
                        elif data.get('type') == 'error':
                            results['error'] = data.get('message', 'Unknown error')
                            logger.error(f"❌ Streaming error: {results['error']}")
                            break
                    
                    except asyncio.TimeoutError:
                        continue
                
                if results['streaming_complete'] and first_word_received:
                    results['success'] = True
                    
                    # Calculate performance metrics
                    if results['first_word_latency_ms']:
                        self.performance_metrics['first_word_latency'].append(results['first_word_latency_ms'])
                    
                    total_time = (time.time() - test_start) * 1000
                    self.performance_metrics['total_streaming_latency'].append(total_time)
                    
                    logger.info(f"✅ Streaming test successful: {results['first_word_latency_ms']:.1f}ms first word")
                else:
                    results['error'] = "Streaming did not complete successfully"
                    logger.error(f"❌ Streaming test failed: {results['error']}")
        
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"❌ Streaming test exception: {e}")
        
        return results
    
    async def test_interruption_detection(self) -> Dict[str, Any]:
        """Test user interruption detection and TTS cancellation"""
        logger.info("🧪 Testing interruption detection...")
        
        results = {
            'success': False,
            'interruption_detected': False,
            'response_time_ms': None,
            'error': None
        }
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # First, start a streaming session
                audio_data = self.generate_test_audio(duration=5.0)  # Longer audio
                audio_b64 = base64.b64encode(audio_data).decode()
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"interrupt_test_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                logger.info("📤 Started streaming session for interruption test")
                
                # Wait for TTS to start
                await asyncio.sleep(2.0)
                
                # Send interruption signal (new audio while TTS is playing)
                interrupt_start = time.time()
                interrupt_audio = self.generate_test_audio(duration=1.0, frequency=880)  # Different frequency
                interrupt_b64 = base64.b64encode(interrupt_audio).decode()
                
                interrupt_message = {
                    "type": "audio_chunk",
                    "audio_data": interrupt_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"interrupt_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(interrupt_message))
                logger.info("🛑 Sent interruption signal")
                
                # Listen for interruption response
                timeout = 10
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'interruption':
                            response_time = (time.time() - interrupt_start) * 1000
                            results['interruption_detected'] = True
                            results['response_time_ms'] = response_time
                            results['success'] = True
                            
                            self.performance_metrics['interruption_response_time'].append(response_time)
                            
                            logger.info(f"✅ Interruption detected in {response_time:.1f}ms")
                            break
                    
                    except asyncio.TimeoutError:
                        continue
                
                if not results['interruption_detected']:
                    results['error'] = "Interruption not detected within timeout"
                    logger.error("❌ Interruption detection failed")
        
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"❌ Interruption test exception: {e}")
        
        return results
    
    async def test_250_token_generation(self) -> Dict[str, Any]:
        """Test 250 token generation with streaming for perceived low latency"""
        logger.info("🧪 Testing 250 token generation with streaming...")
        
        results = {
            'success': False,
            'tokens_generated': 0,
            'first_word_latency_ms': None,
            'total_generation_time_ms': None,
            'streaming_effective': False,
            'error': None
        }
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Generate longer audio to trigger more comprehensive response
                audio_data = self.generate_test_audio(duration=4.0)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                test_start = time.time()
                
                message = {
                    "type": "audio_chunk",
                    "audio_data": audio_b64,
                    "mode": "streaming",
                    "streaming": True,
                    "chunk_id": f"long_test_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(message))
                logger.info("📤 Sent request for 250 token generation test")
                
                first_word_received = False
                total_text = ""
                word_count = 0
                
                timeout = 60  # Longer timeout for 250 tokens
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'streaming_words':
                            if not first_word_received:
                                first_word_latency = (time.time() - test_start) * 1000
                                results['first_word_latency_ms'] = first_word_latency
                                first_word_received = True
                                logger.info(f"⚡ First words in {first_word_latency:.1f}ms")
                            
                            words = data.get('text', '')
                            total_text += " " + words
                            word_count += len(words.split())
                            
                            if word_count % 10 == 0:  # Log every 10 words
                                logger.info(f"📝 {word_count} words received so far...")
                        
                        elif data.get('type') == 'streaming_complete':
                            total_time = (time.time() - test_start) * 1000
                            results['total_generation_time_ms'] = total_time
                            results['tokens_generated'] = len(total_text.split())
                            
                            # Check if streaming was effective (first word < 200ms)
                            if results['first_word_latency_ms'] and results['first_word_latency_ms'] < 200:
                                results['streaming_effective'] = True
                            
                            results['success'] = True
                            logger.info(f"✅ Generated {results['tokens_generated']} tokens in {total_time:.1f}ms")
                            logger.info(f"📊 Streaming effective: {results['streaming_effective']}")
                            break
                        
                        elif data.get('type') == 'error':
                            results['error'] = data.get('message', 'Unknown error')
                            break
                    
                    except asyncio.TimeoutError:
                        continue
                
                if not results['success']:
                    results['error'] = "250 token test did not complete"
        
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"❌ 250 token test exception: {e}")
        
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all streaming tests and generate comprehensive report"""
        logger.info("🚀 Starting comprehensive streaming voice agent tests...")
        
        # Test 1: Basic streaming response
        streaming_result = await self.test_streaming_response()
        self.test_results['streaming_response'] = streaming_result
        
        # Test 2: Interruption detection
        interruption_result = await self.test_interruption_detection()
        self.test_results['interruption_detection'] = interruption_result
        
        # Test 3: 250 token generation
        token_result = await self.test_250_token_generation()
        self.test_results['token_generation'] = token_result
        
        # Generate comprehensive report
        report = self.generate_test_report()
        
        return report
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report with performance analysis"""
        report = {
            'timestamp': time.time(),
            'test_results': self.test_results,
            'performance_metrics': {},
            'success_summary': {},
            'recommendations': []
        }
        
        # Calculate performance metrics
        for metric_name, values in self.performance_metrics.items():
            if values:
                report['performance_metrics'][metric_name] = {
                    'count': len(values),
                    'avg_ms': sum(values) / len(values),
                    'min_ms': min(values),
                    'max_ms': max(values),
                    'target_met': self._check_target_met(metric_name, values)
                }
        
        # Success summary
        report['success_summary'] = {
            'streaming_response': self.test_results.get('streaming_response', {}).get('success', False),
            'interruption_detection': self.test_results.get('interruption_detection', {}).get('success', False),
            'token_generation': self.test_results.get('token_generation', {}).get('success', False),
            'overall_success': all([
                self.test_results.get('streaming_response', {}).get('success', False),
                self.test_results.get('interruption_detection', {}).get('success', False),
                self.test_results.get('token_generation', {}).get('success', False)
            ])
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _check_target_met(self, metric_name: str, values: List[float]) -> bool:
        """Check if performance targets are met"""
        targets = {
            'first_word_latency': 100,
            'word_to_audio_latency': 150,
            'interruption_response_time': 100,
            'total_streaming_latency': 300
        }
        
        target = targets.get(metric_name, 1000)
        avg_value = sum(values) / len(values)
        return avg_value <= target
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results"""
        recommendations = []
        
        # Check streaming performance
        if self.test_results.get('streaming_response', {}).get('success'):
            first_word_latency = self.test_results['streaming_response'].get('first_word_latency_ms', 0)
            if first_word_latency > 100:
                recommendations.append(f"⚠️ First word latency ({first_word_latency:.1f}ms) exceeds 100ms target")
            else:
                recommendations.append(f"✅ First word latency ({first_word_latency:.1f}ms) meets target")
        
        # Check interruption detection
        if self.test_results.get('interruption_detection', {}).get('success'):
            response_time = self.test_results['interruption_detection'].get('response_time_ms', 0)
            if response_time > 100:
                recommendations.append(f"⚠️ Interruption response time ({response_time:.1f}ms) exceeds 100ms target")
            else:
                recommendations.append(f"✅ Interruption response time ({response_time:.1f}ms) meets target")
        
        # Check token generation
        if self.test_results.get('token_generation', {}).get('success'):
            streaming_effective = self.test_results['token_generation'].get('streaming_effective', False)
            if streaming_effective:
                recommendations.append("✅ 250 token generation with effective streaming")
            else:
                recommendations.append("⚠️ 250 token generation streaming needs optimization")
        
        return recommendations

async def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Voice Agent Test Suite")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    parser.add_argument("--output", default="streaming_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    tester = StreamingVoiceAgentTester(args.server)
    
    try:
        logger.info(f"🔗 Connecting to server: {args.server}")
        report = await tester.run_comprehensive_tests()
        
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 STREAMING VOICE AGENT TEST RESULTS")
        print("="*60)
        
        success_summary = report['success_summary']
        print(f"Streaming Response: {'✅' if success_summary['streaming_response'] else '❌'}")
        print(f"Interruption Detection: {'✅' if success_summary['interruption_detection'] else '❌'}")
        print(f"250 Token Generation: {'✅' if success_summary['token_generation'] else '❌'}")
        print(f"Overall Success: {'✅' if success_summary['overall_success'] else '❌'}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        for metric, data in report['performance_metrics'].items():
            target_status = "✅" if data['target_met'] else "❌"
            print(f"  {metric}: {data['avg_ms']:.1f}ms avg {target_status}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print(f"\n💾 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
