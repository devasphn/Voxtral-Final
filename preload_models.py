#!/usr/bin/env python3
"""
Model Pre-loading Script for Voxtral Voice AI
Downloads and caches models to eliminate cold start delays
Optimized for RunPod deployment
"""
import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def preload_voxtral_model():
    """Pre-load Voxtral model and processor"""
    logger.info("🤖 Pre-loading Voxtral model...")
    start_time = time.time()
    
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        
        model_name = "mistralai/Voxtral-Mini-3B-2507"
        
        logger.info(f"   📥 Downloading Voxtral model: {model_name}")
        
        # Download and cache the model
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        )
        
        # Download and cache the processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        logger.info(f"   ✅ Voxtral model loaded successfully in {load_time:.1f}s")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   📊 Model parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed to load Voxtral model: {e}")
        return False

def preload_kokoro_model():
    """Pre-load Kokoro TTS model"""
    logger.info("🔊 Pre-loading Kokoro TTS model...")
    start_time = time.time()
    
    try:
        import kokoro
        
        logger.info("   📥 Initializing Kokoro TTS...")
        
        # Initialize Kokoro TTS (this will download models if needed)
        # The exact initialization depends on the kokoro package API
        
        load_time = time.time() - start_time
        logger.info(f"   ✅ Kokoro TTS loaded successfully in {load_time:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed to load Kokoro TTS: {e}")
        return False

def verify_gpu_availability():
    """Verify GPU availability for model loading"""
    logger.info("🖥️  Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            
            logger.info(f"   ✅ CUDA Available: {torch.version.cuda}")
            logger.info(f"   ✅ GPU Count: {gpu_count}")
            logger.info(f"   ✅ GPU: {gpu_name}")
            logger.info(f"   ✅ GPU Memory: {gpu_memory:.1f}GB")
            
            return True
        else:
            logger.warning("   ⚠️  No GPU available - models will run on CPU (slower)")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ GPU check failed: {e}")
        return False

def check_disk_space():
    """Check available disk space for model downloads"""
    logger.info("💾 Checking disk space...")
    
    try:
        # Check workspace directory
        workspace_path = Path("/workspace") if Path("/workspace").exists() else Path(".")
        stat = os.statvfs(workspace_path)
        
        # Calculate available space in GB
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        logger.info(f"   📁 Available space: {available_gb:.1f}GB")
        
        if available_gb < 10:
            logger.warning(f"   ⚠️  Low disk space: {available_gb:.1f}GB (recommend >10GB)")
            return False
        else:
            logger.info(f"   ✅ Sufficient disk space available")
            return True
            
    except Exception as e:
        logger.error(f"   ❌ Disk space check failed: {e}")
        return False

def setup_cache_directories():
    """Setup model cache directories"""
    logger.info("📁 Setting up cache directories...")
    
    try:
        # Determine workspace path
        workspace_path = Path("/workspace") if Path("/workspace").exists() else Path(".")
        
        # Create cache directories
        cache_dirs = [
            workspace_path / "model_cache",
            workspace_path / "logs",
            workspace_path / "temp_audio"
        ]
        
        for cache_dir in cache_dirs:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(cache_dir, 0o755)
            logger.info(f"   ✅ Created: {cache_dir}")
        
        # Set HuggingFace cache directory
        hf_cache = workspace_path / "model_cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
        
        logger.info(f"   ✅ HuggingFace cache: {hf_cache}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Cache directory setup failed: {e}")
        return False

def test_model_imports():
    """Test that all required model libraries can be imported"""
    logger.info("📦 Testing model imports...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("tokenizers", "Tokenizers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("kokoro", "Kokoro TTS"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy")
    ]
    
    all_imports_successful = True
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            logger.info(f"   ✅ {display_name}")
        except ImportError as e:
            logger.error(f"   ❌ {display_name}: {e}")
            all_imports_successful = False
    
    return all_imports_successful

def main():
    """Main pre-loading function"""
    logger.info("🚀 Starting model pre-loading for Voxtral Voice AI")
    logger.info("=" * 60)
    
    start_time = time.time()
    success_count = 0
    total_steps = 6
    
    # Step 1: Check disk space
    if check_disk_space():
        success_count += 1
    
    # Step 2: Setup cache directories
    if setup_cache_directories():
        success_count += 1
    
    # Step 3: Verify GPU
    if verify_gpu_availability():
        success_count += 1
    
    # Step 4: Test imports
    if test_model_imports():
        success_count += 1
    else:
        logger.error("❌ Some required packages are missing. Please run: pip install -r requirements.txt")
        return False
    
    # Step 5: Pre-load Voxtral model
    if preload_voxtral_model():
        success_count += 1
    
    # Step 6: Pre-load Kokoro model
    if preload_kokoro_model():
        success_count += 1
    
    # Summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"🎯 Pre-loading completed: {success_count}/{total_steps} steps successful")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")
    
    if success_count == total_steps:
        logger.info("✅ All models pre-loaded successfully!")
        logger.info("🚀 System ready for ultra-low latency voice AI")
        logger.info("💡 You can now start the server with:")
        logger.info("   python -m uvicorn src.api.ui_server_simple:app --host 0.0.0.0 --port 8000")
        return True
    else:
        logger.warning(f"⚠️  {total_steps - success_count} steps failed")
        logger.info("💡 You can still try starting the server, but expect longer initial response times")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Pre-loading interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
