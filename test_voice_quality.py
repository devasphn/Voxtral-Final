#!/usr/bin/env python3
"""
Voice Quality Testing Script for Voxtral + Kokoro TTS
Tests the new Indian female voice configuration and audio quality improvements
"""

import asyncio
import time
import logging
import sys
import os
import numpy as np
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_quality_test")

class VoiceQualityTester:
    """Tests voice quality and configuration for Indian female voices"""
    
    def __init__(self):
        self.test_results = {}
        self.tts_service = None
        self.kokoro_model = None
        
    async def initialize_services(self):
        """Initialize TTS services for testing"""
        logger.info("🎤 Initializing TTS services for voice quality testing...")
        
        try:
            # Import TTS services
            from src.tts.tts_service import TTSService
            from src.models.kokoro_model_realtime import KokoroTTSModel
            from src.utils.voice_profiles import voice_profile_manager
            
            self.tts_service = TTSService()
            self.kokoro_model = KokoroTTSModel()
            self.voice_manager = voice_profile_manager
            
            # Initialize services
            logger.info("🔊 Initializing TTS service...")
            await self.tts_service.initialize()
            
            logger.info("🎵 Initializing Kokoro TTS model...")
            await self.kokoro_model.initialize()
            
            logger.info("✅ All TTS services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ TTS service initialization failed: {e}")
            return False
    
    async def test_indian_female_voices(self):
        """Test Indian female voices for quality and accent"""
        logger.info("\n🧪 Testing Indian Female Voices")
        
        test_texts = [
            "Hello, my name is Priya and I am here to help you today.",
            "Welcome to our voice assistant. How can I assist you?",
            "The weather is quite pleasant today in Mumbai.",
            "Thank you for using our service. Have a wonderful day!"
        ]
        
        indian_voices = ["hf_alpha", "hf_beta"]
        results = {}
        
        for voice in indian_voices:
            logger.info(f"🎤 Testing voice: {voice}")
            voice_results = []
            
            for i, text in enumerate(test_texts):
                try:
                    start_time = time.time()
                    result = await self.tts_service.generate_speech_async(text, voice)
                    generation_time = (time.time() - start_time) * 1000
                    
                    if result.get('success', False):
                        audio_data = result.get('audio_data')
                        if audio_data is not None and len(audio_data) > 0:
                            # Calculate audio quality metrics
                            audio_rms = np.sqrt(np.mean(audio_data**2))
                            audio_peak = np.max(np.abs(audio_data))
                            audio_duration = len(audio_data) / 24000  # 24kHz sample rate
                            
                            voice_results.append({
                                'text_index': i,
                                'text_length': len(text),
                                'generation_time_ms': generation_time,
                                'audio_duration_s': audio_duration,
                                'audio_rms': audio_rms,
                                'audio_peak': audio_peak,
                                'audio_samples': len(audio_data),
                                'success': True
                            })
                            
                            logger.info(f"  ✅ Text {i+1}: {generation_time:.1f}ms, {audio_duration:.2f}s audio")
                        else:
                            voice_results.append({'text_index': i, 'success': False, 'error': 'No audio data'})
                            logger.warning(f"  ⚠️  Text {i+1}: No audio generated")
                    else:
                        voice_results.append({'text_index': i, 'success': False, 'error': result.get('error', 'Unknown')})
                        logger.error(f"  ❌ Text {i+1}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    voice_results.append({'text_index': i, 'success': False, 'error': str(e)})
                    logger.error(f"  ❌ Text {i+1}: Exception - {e}")
            
            results[voice] = voice_results
        
        self.test_results['indian_female_voices'] = results
        
        # Calculate summary statistics
        for voice, voice_results in results.items():
            successful_tests = [r for r in voice_results if r.get('success', False)]
            if successful_tests:
                avg_generation_time = sum(r['generation_time_ms'] for r in successful_tests) / len(successful_tests)
                avg_audio_quality = sum(r['audio_rms'] for r in successful_tests) / len(successful_tests)
                
                logger.info(f"📊 {voice} Summary:")
                logger.info(f"   Success Rate: {len(successful_tests)}/{len(voice_results)} ({len(successful_tests)/len(voice_results)*100:.1f}%)")
                logger.info(f"   Avg Generation Time: {avg_generation_time:.1f}ms")
                logger.info(f"   Avg Audio Quality (RMS): {avg_audio_quality:.4f}")
    
    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        logger.info("\n🧪 Testing Voice Mapping")
        
        test_mappings = [
            ("indian", "hf_alpha"),
            ("hindi", "hf_alpha"),
            ("ऋतिका", "hf_alpha"),
            ("indian_female", "hf_alpha"),
            ("english", "af_bella"),
            ("professional", "af_nicole"),
            ("default", "hf_alpha")
        ]
        
        mapping_results = []
        
        for request_voice, expected_voice in test_mappings:
            try:
                from src.tts.tts_service import map_voice_to_kokoro
                mapped_voice = map_voice_to_kokoro(request_voice)
                
                success = mapped_voice == expected_voice
                mapping_results.append({
                    'request': request_voice,
                    'expected': expected_voice,
                    'actual': mapped_voice,
                    'success': success
                })
                
                status = "✅" if success else "❌"
                logger.info(f"  {status} '{request_voice}' -> '{mapped_voice}' (expected: '{expected_voice}')")
                
            except Exception as e:
                mapping_results.append({
                    'request': request_voice,
                    'expected': expected_voice,
                    'actual': None,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"  ❌ '{request_voice}' -> Error: {e}")
        
        self.test_results['voice_mapping'] = mapping_results
        
        success_rate = sum(1 for r in mapping_results if r['success']) / len(mapping_results)
        logger.info(f"📊 Voice Mapping Success Rate: {success_rate*100:.1f}%")
    
    async def test_latency_performance(self):
        """Test latency performance with new quality settings"""
        logger.info("\n🧪 Testing Latency Performance")
        
        test_texts = [
            "Hi there!",
            "How are you doing today?",
            "This is a medium length sentence for testing purposes.",
            "This is a longer sentence that contains more words and should take more time to process but still maintain good quality output."
        ]
        
        latency_results = []
        
        for i, text in enumerate(test_texts):
            try:
                start_time = time.time()
                result = await self.tts_service.generate_speech_async(text, "hf_alpha")
                total_time = (time.time() - start_time) * 1000
                
                if result.get('success', False):
                    latency_results.append({
                        'text_length': len(text),
                        'total_latency_ms': total_time,
                        'target_met': total_time < 300,  # Target <300ms
                        'success': True
                    })
                    
                    status = "✅" if total_time < 300 else "⚠️"
                    logger.info(f"  {status} Text {i+1} ({len(text)} chars): {total_time:.1f}ms")
                else:
                    latency_results.append({
                        'text_length': len(text),
                        'success': False,
                        'error': result.get('error', 'Unknown')
                    })
                    logger.error(f"  ❌ Text {i+1}: Failed - {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                latency_results.append({
                    'text_length': len(text),
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"  ❌ Text {i+1}: Exception - {e}")
        
        self.test_results['latency_performance'] = latency_results
        
        # Calculate performance statistics
        successful_tests = [r for r in latency_results if r.get('success', False)]
        if successful_tests:
            avg_latency = sum(r['total_latency_ms'] for r in successful_tests) / len(successful_tests)
            target_met_count = sum(1 for r in successful_tests if r.get('target_met', False))
            
            logger.info(f"📊 Latency Performance Summary:")
            logger.info(f"   Average Latency: {avg_latency:.1f}ms")
            logger.info(f"   Target Met: {target_met_count}/{len(successful_tests)} ({target_met_count/len(successful_tests)*100:.1f}%)")
    
    async def test_voice_profiles(self):
        """Test voice profile management"""
        logger.info("\n🧪 Testing Voice Profiles")
        
        try:
            # Test getting Indian female voices
            indian_voices = self.voice_manager.get_indian_female_voices()
            logger.info(f"✅ Found {len(indian_voices)} Indian female voices")
            
            for voice in indian_voices:
                logger.info(f"   - {voice.name} ({voice.voice_id}): {voice.description}")
            
            # Test recommendations
            recommended = self.voice_manager.get_recommended_voice_for_indian_accent()
            logger.info(f"✅ Recommended voice for Indian accent: {recommended}")
            
            # Test UI info
            ui_info = self.voice_manager.get_voice_info_for_ui()
            logger.info(f"✅ UI voice info generated: {len(ui_info['all_voices'])} total voices")
            logger.info(f"   Indian female: {len(ui_info['indian_female'])}")
            logger.info(f"   English female: {len(ui_info['english_female'])}")
            
            self.test_results['voice_profiles'] = {
                'indian_voices_count': len(indian_voices),
                'recommended_voice': recommended,
                'ui_info_generated': True,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Voice profiles test failed: {e}")
            self.test_results['voice_profiles'] = {
                'success': False,
                'error': str(e)
            }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📊 VOICE QUALITY TEST REPORT")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test_name, result in self.test_results.items() 
                          if isinstance(result, dict) and result.get('success', True))
        
        logger.info(f"📈 Overall Results: {passed_tests}/{total_tests} test categories passed")
        logger.info("")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            logger.info(f"🔍 {test_name.upper().replace('_', ' ')}:")
            
            if test_name == 'indian_female_voices':
                for voice, voice_results in result.items():
                    successful = sum(1 for r in voice_results if r.get('success', False))
                    logger.info(f"   {voice}: {successful}/{len(voice_results)} tests passed")
            
            elif test_name == 'voice_mapping':
                successful = sum(1 for r in result if r.get('success', False))
                logger.info(f"   Mapping accuracy: {successful}/{len(result)} ({successful/len(result)*100:.1f}%)")
            
            elif test_name == 'latency_performance':
                successful = [r for r in result if r.get('success', False)]
                if successful:
                    avg_latency = sum(r['total_latency_ms'] for r in successful) / len(successful)
                    target_met = sum(1 for r in successful if r.get('target_met', False))
                    logger.info(f"   Average latency: {avg_latency:.1f}ms")
                    logger.info(f"   Target <300ms met: {target_met}/{len(successful)} times")
            
            elif isinstance(result, dict) and 'success' in result:
                status = "✅ PASSED" if result['success'] else "❌ FAILED"
                logger.info(f"   Status: {status}")
        
        logger.info("")
        logger.info("🎯 VOICE CONFIGURATION STATUS:")
        if passed_tests == total_tests:
            logger.info("   ✅ ALL TESTS PASSED - INDIAN FEMALE VOICE READY FOR PRODUCTION")
        else:
            logger.info(f"   ⚠️  {total_tests - passed_tests} ISSUES DETECTED - REVIEW REQUIRED")
        
        return passed_tests == total_tests

async def main():
    """Main test execution"""
    logger.info("🎤 VOICE QUALITY AND CONFIGURATION TESTING")
    logger.info("=" * 60)
    
    tester = VoiceQualityTester()
    
    # Initialize services
    if not await tester.initialize_services():
        logger.error("❌ Service initialization failed - cannot proceed with tests")
        return False
    
    # Run all tests
    await tester.test_voice_profiles()
    await tester.test_voice_mapping()
    await tester.test_indian_female_voices()
    await tester.test_latency_performance()
    
    # Generate report
    all_passed = tester.generate_report()
    
    return all_passed

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)
