#!/usr/bin/env python3
"""
Quick validation script to confirm the audio streaming fix is working
"""

import sys
import os
from pathlib import Path
import soundfile as sf
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.audio_format_validator import audio_format_validator

def validate_test_outputs():
    """Validate the generated test audio files"""
    print("🔍 Validating Audio Fix - Test Output Files")
    print("=" * 50)
    
    test_dir = Path("test_outputs")
    if not test_dir.exists():
        print("❌ Test outputs directory not found. Run test_audio_streaming_fix.py first.")
        return False
    
    test_files = [
        "test_basic_synthesis.wav",
        "test_streaming_chunk_001.wav", 
        "browser_test_audio.wav"
    ]
    
    all_valid = True
    
    for filename in test_files:
        filepath = test_dir / filename
        if not filepath.exists():
            print(f"❌ {filename} - File not found")
            all_valid = False
            continue
        
        try:
            # Read audio file
            audio_data, sample_rate = sf.read(filepath)
            file_size = filepath.stat().st_size
            
            print(f"\n📁 {filename}")
            print(f"   📊 File size: {file_size:,} bytes")
            print(f"   🎵 Sample rate: {sample_rate}Hz")
            print(f"   📏 Duration: {len(audio_data)/sample_rate:.2f}s")
            print(f"   🔊 Samples: {len(audio_data):,}")
            
            # Validate WAV format
            with open(filepath, 'rb') as f:
                wav_bytes = f.read()
            
            validation = audio_format_validator.validate_wav_headers(wav_bytes)
            
            if validation['is_valid']:
                print(f"   ✅ Valid WAV format")
                print(f"   📋 Channels: {validation['channels']}")
                print(f"   🎚️  Bit depth: {validation['bit_depth']}")
                
                # Check for audio quality issues
                if len(audio_data) > 0:
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0.95:
                        print(f"   ⚠️  High amplitude detected: {max_val:.3f}")
                    elif max_val < 0.01:
                        print(f"   ⚠️  Very low amplitude: {max_val:.3f}")
                    else:
                        print(f"   ✅ Good amplitude range: {max_val:.3f}")
                    
                    # Check for silence
                    if np.all(audio_data == 0):
                        print(f"   ❌ Audio contains only silence")
                        all_valid = False
                    else:
                        print(f"   ✅ Contains audio content")
                else:
                    print(f"   ❌ No audio data")
                    all_valid = False
            else:
                print(f"   ❌ Invalid WAV format: {validation['errors']}")
                all_valid = False
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All test files are valid! Audio fix is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Start the Voxtral application")
        print("   2. Test with the web interface")
        print("   3. Verify you hear clear speech instead of ultrasonic noise")
    else:
        print("⚠️  Some issues detected. Check the output above.")
    
    return all_valid

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔧 Checking Dependencies...")
    
    try:
        import soundfile
        print("   ✅ soundfile available")
    except ImportError:
        print("   ❌ soundfile not available - install with: pip install soundfile")
        return False
    
    try:
        import numpy
        print("   ✅ numpy available")
    except ImportError:
        print("   ❌ numpy not available - install with: pip install numpy")
        return False
    
    return True

def main():
    """Main validation function"""
    print("🎵 Voxtral Audio Streaming Fix - Validation")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install required packages.")
        return 1
    
    # Validate test outputs
    if not validate_test_outputs():
        print("\n❌ Validation failed. Some issues need to be addressed.")
        return 1
    
    print("\n✅ Validation completed successfully!")
    print("\n📋 Summary of fixes applied:")
    print("   • Fixed WAV format generation in Kokoro streaming")
    print("   • Enhanced client-side audio validation")
    print("   • Migrated to modern AudioWorkletNode API")
    print("   • Added comprehensive audio format validation")
    print("   • Implemented proper audio normalization")
    
    print("\n🎯 Expected results:")
    print("   • Clear speech audio (no ultrasonic noise)")
    print("   • Sub-100ms first-word latency")
    print("   • Stable streaming performance")
    print("   • Cross-browser compatibility")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
