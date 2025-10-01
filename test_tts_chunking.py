#!/usr/bin/env python3
"""
Test TTS chunking behavior to understand why so many audio chunks are generated
"""

import asyncio
import sys
import os
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

async def test_tts_chunking():
    """Test how many chunks TTS generates for different inputs"""
    print("🔧 Testing TTS Chunking Behavior...")
    
    try:
        # Import and initialize Kokoro TTS model
        from src.models.kokoro_model_realtime import KokoroTTSModel
        
        print("📦 Initializing Kokoro TTS model...")
        kokoro = KokoroTTSModel()
        await kokoro.initialize()
        print("✅ Kokoro TTS model initialized successfully")
        
        # Test different text inputs
        test_cases = [
            "hello",           # Single word
            "hello world",     # Two words
            "I",              # Single letter
            "sorry",          # Word from our streaming test
            "process",        # Another word from streaming test
            "Hi there! How are you?"  # Full sentence
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: '{text}'")
            
            chunk_count = 0
            audio_chunks = []
            start_time = time.time()
            
            try:
                async for tts_chunk in kokoro.synthesize_speech_streaming(text, voice="hf_alpha"):
                    if tts_chunk.get('audio_chunk'):
                        chunk_count += 1
                        audio_size = len(tts_chunk['audio_chunk'])
                        audio_chunks.append(audio_size)
                        print(f"   Chunk {chunk_count}: {audio_size} bytes")
                    elif tts_chunk.get('is_final'):
                        break
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
            
            total_time = (time.time() - start_time) * 1000
            total_audio_bytes = sum(audio_chunks)
            
            print(f"   📊 Results:")
            print(f"      Total chunks: {chunk_count}")
            print(f"      Total audio: {total_audio_bytes} bytes")
            print(f"      Generation time: {total_time:.1f}ms")
            print(f"      Chunks per word: {chunk_count / len(text.split()):.1f}")
            
            # Analyze chunking pattern
            if chunk_count > len(text.split()) * 5:  # More than 5 chunks per word
                print(f"   ⚠️  High chunk count detected!")
            elif chunk_count <= len(text.split()) * 2:  # 2 or fewer chunks per word
                print(f"   ✅ Reasonable chunk count")
            else:
                print(f"   ⚡ Moderate chunk count")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test TTS chunking: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

async def test_tts_pipeline_settings():
    """Test different TTS pipeline settings to optimize chunking"""
    print("\n🔧 Testing TTS Pipeline Settings...")
    
    try:
        # Import Kokoro directly to test pipeline settings
        from kokoro import KPipeline
        
        print("📦 Testing different pipeline configurations...")
        
        # Test with different settings
        configs = [
            {"lang_code": "h", "description": "Hindi (current)"},
            {"lang_code": "a", "description": "Auto (default)"},
            {"lang_code": "en", "description": "English"},
        ]
        
        for config in configs:
            print(f"\n   Testing {config['description']} (lang_code: {config['lang_code']})...")
            
            try:
                pipeline = KPipeline(lang_code=config['lang_code'])
                
                # Test with a simple word
                test_text = "hello"
                chunk_count = 0
                
                generator = pipeline(test_text, voice="hf_alpha", speed=1.0)
                
                for i, (gs, ps, audio) in enumerate(generator):
                    if audio is not None and len(audio) > 0:
                        chunk_count += 1
                        if chunk_count >= 10:  # Limit to first 10 chunks
                            break
                
                print(f"      Generated {chunk_count} chunks for '{test_text}'")
                
            except Exception as e:
                print(f"      ❌ Error with {config['description']}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test pipeline settings: {e}")
        return False

async def main():
    """Run all TTS chunking tests"""
    print("🚀 Starting TTS Chunking Diagnostic Tests...\n")
    
    # Test 1: TTS chunking behavior
    success1 = await test_tts_chunking()
    
    # Test 2: Pipeline settings
    success2 = await test_tts_pipeline_settings()
    
    print(f"\n📊 Test Results:")
    print(f"   TTS chunking test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Pipeline settings test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n✅ All tests completed - check results above for optimization opportunities")
    else:
        print("\n❌ Some tests failed - TTS chunking issues detected")

if __name__ == "__main__":
    asyncio.run(main())
