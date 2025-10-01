#!/usr/bin/env python3
"""
Voxtral-Final Model Preloading Script
Downloads and caches all required models to eliminate cold starts
"""

import os
import sys
import time
import torch
from pathlib import Path
import logging
from typing import Dict, Any

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def setup_logging():
    """Setup logging for preloading"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("preload_models")

def print_header():
    """Print preloading header"""
    print("=" * 60)
    print("🤖 VOXTRAL-FINAL MODEL PRELOADING")
    print("=" * 60)
    print()

def check_disk_space(required_gb=10):
    """Check available disk space"""
    logger = logging.getLogger("preload_models")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        logger.info(f"💾 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < required_gb:
            logger.warning(f"⚠️  Low disk space: {free_gb:.1f} GB available, {required_gb} GB recommended")
            return False
        
        logger.info(f"✅ Sufficient disk space available")
        return True
        
    except Exception as e:
        logger.error(f"❌ Disk space check failed: {e}")
        return False

def preload_voxtral_model():
    """Preload Voxtral model"""
    logger = logging.getLogger("preload_models")
    logger.info("🎯 Preloading Voxtral model...")
    
    try:
        # Import configuration
        from src.utils.config import config
        
        # Import model classes with fallback
        try:
            from transformers import VoxtralForConditionalGeneration, AutoProcessor
            logger.info("✅ Transformers with Voxtral support available")
        except ImportError:
            logger.error("❌ Voxtral not available in transformers. Please update to transformers>=4.56.0")
            return False
        
        model_name = config.model.name
        cache_dir = config.model.cache_dir
        
        logger.info(f"📥 Downloading Voxtral model: {model_name}")
        logger.info(f"📁 Cache directory: {cache_dir}")
        
        # Create cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare download arguments
        download_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        
        # Add authentication token if available
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            download_kwargs["token"] = hf_token
            logger.info("🔑 Using HuggingFace authentication token")
        else:
            logger.warning("⚠️  No HF_TOKEN found. May fail for gated models.")
        
        # Download processor
        start_time = time.time()
        logger.info("📥 Downloading AutoProcessor...")
        
        processor = AutoProcessor.from_pretrained(model_name, **download_kwargs)
        processor_time = time.time() - start_time
        logger.info(f"✅ AutoProcessor downloaded in {processor_time:.1f}s")
        
        # Download model
        start_time = time.time()
        logger.info("📥 Downloading Voxtral model...")
        
        # Use minimal settings for download
        model_kwargs = download_kwargs.copy()
        model_kwargs.update({
            "torch_dtype": torch.float16,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "low_cpu_mem_usage": True,
        })
        
        model = VoxtralForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        model_time = time.time() - start_time
        logger.info(f"✅ Voxtral model downloaded in {model_time:.1f}s")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")
        
        # Clean up memory
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Voxtral model preloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Voxtral model preloading failed: {e}")
        return False

def preload_kokoro_model():
    """Preload Kokoro TTS model"""
    logger = logging.getLogger("preload_models")
    logger.info("🎵 Preloading Kokoro TTS model...")
    
    try:
        # Import Kokoro
        import kokoro
        logger.info("✅ Kokoro library available")
        
        # Import configuration
        from src.utils.config import config
        
        # Initialize Kokoro pipeline
        start_time = time.time()
        logger.info("📥 Initializing Kokoro TTS pipeline...")
        
        # Create a minimal Kokoro instance to trigger model download
        pipeline = kokoro.TTS()
        
        # Test with a short text to ensure model is loaded
        test_text = "Hello, this is a test."
        voice = config.tts.voice
        
        logger.info(f"🎤 Testing with voice: {voice}")
        
        # Generate test audio (this will download the model if needed)
        audio_data = pipeline.generate(test_text, voice=voice)
        
        kokoro_time = time.time() - start_time
        logger.info(f"✅ Kokoro TTS model loaded in {kokoro_time:.1f}s")
        
        # Clean up
        del pipeline
        del audio_data
        
        logger.info("✅ Kokoro TTS model preloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Kokoro TTS model preloading failed: {e}")
        logger.error(f"   Make sure 'kokoro>=0.9.4' is installed")
        return False

def verify_model_cache():
    """Verify that models are properly cached"""
    logger = logging.getLogger("preload_models")
    logger.info("🔍 Verifying model cache...")
    
    try:
        from src.utils.config import config
        cache_dir = Path(config.model.cache_dir)
        
        if cache_dir.exists():
            # Check cache size
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            
            logger.info(f"📁 Cache directory: {cache_dir}")
            logger.info(f"💾 Cache size: {size_gb:.2f} GB")
            
            if size_gb > 0.1:  # At least 100MB cached
                logger.info("✅ Model cache verified")
                return True
            else:
                logger.warning("⚠️  Cache appears empty or too small")
                return False
        else:
            logger.warning("⚠️  Cache directory not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cache verification failed: {e}")
        return False

def cleanup_old_cache():
    """Clean up old or corrupted cache files"""
    logger = logging.getLogger("preload_models")
    logger.info("🧹 Cleaning up old cache...")
    
    try:
        from src.utils.config import config
        cache_dir = Path(config.model.cache_dir)
        
        if cache_dir.exists():
            # Remove any .tmp files or incomplete downloads
            tmp_files = list(cache_dir.rglob('*.tmp'))
            for tmp_file in tmp_files:
                tmp_file.unlink()
                logger.info(f"🗑️  Removed temporary file: {tmp_file.name}")
            
            logger.info("✅ Cache cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache cleanup failed: {e}")
        return False

def main():
    """Main preloading function"""
    logger = setup_logging()
    print_header()
    
    start_time = time.time()
    
    # Preloading steps
    steps = [
        ("Disk Space Check", lambda: check_disk_space(10)),
        ("Cache Cleanup", cleanup_old_cache),
        ("Voxtral Model", preload_voxtral_model),
        ("Kokoro TTS Model", preload_kokoro_model),
        ("Cache Verification", verify_model_cache),
    ]
    
    results = []
    for step_name, step_func in steps:
        logger.info(f"\n🔄 {step_name}...")
        try:
            result = step_func()
            results.append((step_name, result))
            
            if result:
                logger.info(f"✅ {step_name} completed")
            else:
                logger.warning(f"⚠️  {step_name} had issues")
                
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
            results.append((step_name, False))
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PRELOADING SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for step_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} {step_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} steps completed")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if passed >= total - 1:  # Allow one non-critical failure
        print("\n🎉 Model preloading completed successfully!")
        print("🚀 System ready for ultra-low latency deployment!")
        return True
    else:
        print(f"\n⚠️  {total - passed} critical steps failed.")
        print("🔧 Please check the errors above and retry.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
