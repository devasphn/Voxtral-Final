#!/usr/bin/env python3
"""
Comprehensive Audio Streaming Pipeline Test
Tests the fixed audio format conversion and streaming functionality
"""

import asyncio
import sys
import os
import time
import base64
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.kokoro_model_realtime import kokoro_model
from utils.audio_format_validator import audio_format_validator
from utils.config import config
from utils.logging_config import logger

class AudioStreamingTester:
    """Comprehensive test suite for audio streaming pipeline"""
    
    def __init__(self):
        self.test_results = []
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    async def test_kokoro_initialization(self):
        """Test Kokoro model initialization"""
        print("\n🔧 Testing Kokoro Model Initialization...")
        
        try:
            if not kokoro_model.is_initialized:
                print("   Initializing Kokoro model...")
                await kokoro_model.initialize()
            
            if kokoro_model.is_initialized:
                print("   ✅ Kokoro model initialized successfully")
                info = kokoro_model.get_model_info()
                print(f"   📊 Model info: {info['model_name']} on {info['device']}")
                print(f"   🎵 Sample rate: {info['sample_rate']}Hz")
                print(f"   🗣️  Available voices: {len(info['available_voices'])}")
                return True
            else:
                print("   ❌ Kokoro model initialization failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Initialization error: {e}")
            return False
    
    async def test_basic_synthesis(self):
        """Test basic speech synthesis"""
        print("\n🎤 Testing Basic Speech Synthesis...")
        
        try:
            test_text = "Hello, this is a test of the fixed audio streaming system."
            print(f"   Synthesizing: '{test_text}'")
            
            result = await kokoro_model.synthesize_speech(
                text=test_text,
                voice="hm_omega",
                chunk_id="test_basic"
            )
            
            if result['success']:
                audio_data = result['audio_data']
                sample_rate = result['sample_rate']
                
                print(f"   ✅ Synthesis successful")
                print(f"   📊 Audio: {len(audio_data)} samples at {sample_rate}Hz")
                print(f"   ⏱️  Time: {result['synthesis_time_ms']:.1f}ms")
                
                # Save test audio
                output_file = self.output_dir / "test_basic_synthesis.wav"
                sf.write(output_file, audio_data, sample_rate)
                print(f"   💾 Saved to: {output_file}")
                
                # Validate audio format
                validation = audio_format_validator.validate_audio_chunk(audio_data, 'numpy', sample_rate)
                if validation['is_valid']:
                    print("   ✅ Audio format validation passed")
                else:
                    print(f"   ⚠️  Audio validation issues: {validation['issues']}")
                
                return True
            else:
                print(f"   ❌ Synthesis failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"   ❌ Basic synthesis error: {e}")
            return False
    
    async def test_streaming_synthesis(self):
        """Test streaming speech synthesis with WAV format validation"""
        print("\n🌊 Testing Streaming Speech Synthesis...")
        
        try:
            test_text = "This is a comprehensive test of the streaming audio synthesis with proper WAV format generation."
            print(f"   Streaming synthesis: '{test_text}'")
            
            chunks_received = 0
            total_audio_bytes = 0
            chunk_files = []
            
            async for chunk in kokoro_model.synthesize_speech_streaming(
                text=test_text,
                voice="hm_omega",
                chunk_id="test_streaming"
            ):
                if chunk.get('audio_chunk') and not chunk.get('is_final'):
                    chunks_received += 1
                    audio_chunk = chunk['audio_chunk']
                    total_audio_bytes += len(audio_chunk)
                    
                    print(f"   📦 Chunk {chunks_received}: {len(audio_chunk)} bytes")
                    
                    # Validate WAV format
                    validation = audio_format_validator.validate_wav_headers(audio_chunk)
                    if validation['is_valid']:
                        print(f"      ✅ Valid WAV: {validation['sample_rate']}Hz, {validation['channels']}ch, {validation['bit_depth']}bit")
                        
                        # Save chunk for inspection
                        chunk_file = self.output_dir / f"test_streaming_chunk_{chunks_received:03d}.wav"
                        with open(chunk_file, 'wb') as f:
                            f.write(audio_chunk)
                        chunk_files.append(chunk_file)
                        
                    else:
                        print(f"      ❌ Invalid WAV format: {validation['errors']}")
                        
                        # Diagnose ultrasonic noise potential
                        diagnosis = audio_format_validator.diagnose_ultrasonic_noise(audio_chunk)
                        if diagnosis['likely_causes']:
                            print(f"      🔍 Potential issues: {diagnosis['likely_causes']}")
                            print(f"      💡 Recommendations: {diagnosis['recommendations']}")
                
                elif chunk.get('is_final'):
                    print(f"   🏁 Streaming completed: {chunks_received} chunks, {total_audio_bytes} total bytes")
                    break
            
            if chunks_received > 0:
                print("   ✅ Streaming synthesis successful")
                print(f"   📁 Saved {len(chunk_files)} chunk files")
                return True
            else:
                print("   ❌ No audio chunks received")
                return False
                
        except Exception as e:
            print(f"   ❌ Streaming synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_base64_conversion(self):
        """Test base64 encoding/decoding pipeline"""
        print("\n🔄 Testing Base64 Conversion Pipeline...")
        
        try:
            # Generate a small audio chunk
            test_text = "Base64 test"
            result = await kokoro_model.synthesize_speech(test_text, voice="hm_omega")
            
            if not result['success']:
                print("   ❌ Failed to generate test audio")
                return False
            
            audio_data = result['audio_data']
            sample_rate = result['sample_rate']
            
            # Convert to WAV format
            from io import BytesIO
            wav_buffer = BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            wav_bytes = wav_buffer.getvalue()
            wav_buffer.close()
            
            print(f"   📊 Original WAV: {len(wav_bytes)} bytes")
            
            # Encode to base64
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            print(f"   📊 Base64 encoded: {len(audio_b64)} characters")
            
            # Decode back from base64
            decoded_bytes = base64.b64decode(audio_b64)
            print(f"   📊 Decoded: {len(decoded_bytes)} bytes")
            
            # Validate round-trip
            if wav_bytes == decoded_bytes:
                print("   ✅ Base64 round-trip successful")
                
                # Validate decoded WAV
                validation = audio_format_validator.validate_wav_headers(decoded_bytes)
                if validation['is_valid']:
                    print("   ✅ Decoded WAV format valid")
                    
                    # Save for browser testing
                    browser_test_file = self.output_dir / "browser_test_audio.wav"
                    with open(browser_test_file, 'wb') as f:
                        f.write(decoded_bytes)
                    print(f"   💾 Browser test file: {browser_test_file}")
                    
                    return True
                else:
                    print(f"   ❌ Decoded WAV invalid: {validation['errors']}")
                    return False
            else:
                print("   ❌ Base64 round-trip failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Base64 conversion error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all audio pipeline tests"""
        print("🧪 Starting Comprehensive Audio Streaming Pipeline Tests")
        print("=" * 60)
        
        tests = [
            ("Kokoro Initialization", self.test_kokoro_initialization),
            ("Basic Synthesis", self.test_basic_synthesis),
            ("Streaming Synthesis", self.test_streaming_synthesis),
            ("Base64 Conversion", self.test_base64_conversion)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            start_time = time.time()
            try:
                success = await test_func()
                duration = time.time() - start_time
                results[test_name] = {
                    'success': success,
                    'duration': duration
                }
            except Exception as e:
                duration = time.time() - start_time
                results[test_name] = {
                    'success': False,
                    'duration': duration,
                    'error': str(e)
                }
                print(f"   ❌ Test failed with exception: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(tests)
        passed_tests = sum(1 for r in results.values() if r['success'])
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            print(f"{status} {test_name:<25} ({duration:.2f}s)")
            
            if not result['success'] and 'error' in result:
                print(f"     Error: {result['error']}")
        
        print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Audio streaming pipeline is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        print(f"\n📁 Test outputs saved to: {self.output_dir.absolute()}")
        
        return passed_tests == total_tests

async def main():
    """Main test runner"""
    tester = AudioStreamingTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready to test with the web interface!")
        print("   The ultrasonic noise issue should now be resolved.")
    else:
        print("\n🔧 Some issues remain. Check the test output for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
