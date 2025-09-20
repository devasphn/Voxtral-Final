#!/usr/bin/env python3
"""
Voxtral Model Loading Diagnostic Script
Tests the fixed model loading with comprehensive error reporting
"""

import os
import sys
import traceback
import time
from typing import Optional

def test_environment():
    """Test environment setup"""
    print("🔍 TESTING ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Check HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"✅ HF_TOKEN is set (length: {len(hf_token)})")
    else:
        print("❌ HF_TOKEN is not set!")
        print("   Set it with: export HF_TOKEN='your_token_here'")
        return False
    
    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️ CUDA not available, will use CPU")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
    
    # Check cache directory
    cache_dir = "./model_cache"
    if os.path.exists(cache_dir):
        print(f"✅ Cache directory exists: {cache_dir}")
    else:
        print(f"📁 Creating cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
    
    print()
    return True

def test_imports():
    """Test required imports"""
    print("🔍 TESTING IMPORTS")
    print("=" * 50)
    
    try:
        import transformers
        print(f"✅ transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        print("✅ Voxtral classes imported successfully")
    except ImportError as e:
        print(f"❌ Voxtral classes import failed: {e}")
        print("   Install with: pip install transformers>=4.56.0")
        return False
    
    try:
        import torch
        print(f"✅ torch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    print()
    return True

def test_processor_loading():
    """Test AutoProcessor loading"""
    print("🔍 TESTING AUTOPROCESSOR LOADING")
    print("=" * 50)
    
    try:
        from transformers import AutoProcessor
        
        processor_kwargs = {
            "cache_dir": "./model_cache",
            "trust_remote_code": True
        }
        
        # Add token if available
        if os.getenv('HF_TOKEN'):
            processor_kwargs["token"] = os.getenv('HF_TOKEN')
            print("🔑 Using HuggingFace authentication token")
        
        print("🔄 Loading AutoProcessor...")
        start_time = time.time()
        
        processor = AutoProcessor.from_pretrained(
            "mistralai/Voxtral-Mini-3B-2507",
            **processor_kwargs
        )
        
        load_time = time.time() - start_time
        print(f"✅ AutoProcessor loaded successfully in {load_time:.2f}s")
        print(f"   Processor type: {type(processor).__name__}")
        
        return processor
        
    except Exception as e:
        print(f"❌ AutoProcessor loading failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def test_model_loading():
    """Test Voxtral model loading with fallbacks"""
    print("🔍 TESTING VOXTRAL MODEL LOADING")
    print("=" * 50)
    
    try:
        from transformers import VoxtralForConditionalGeneration
        import torch
        
        # Primary loading attempt
        model_kwargs = {
            "cache_dir": "./model_cache",
            "dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            "use_safetensors": True,
        }
        
        # Add token if available
        if os.getenv('HF_TOKEN'):
            model_kwargs["token"] = os.getenv('HF_TOKEN')
            print("🔑 Using HuggingFace authentication token")
        
        print("🔄 Loading Voxtral model (primary attempt)...")
        start_time = time.time()
        
        try:
            model = VoxtralForConditionalGeneration.from_pretrained(
                "mistralai/Voxtral-Mini-3B-2507",
                **model_kwargs
            )
            load_time = time.time() - start_time
            print(f"✅ Voxtral model loaded successfully in {load_time:.2f}s")
            print(f"   Model device: {model.device}")
            print(f"   Model dtype: {model.dtype}")
            print(f"   Attention: flash_attention_2")
            return model
            
        except Exception as primary_error:
            print(f"⚠️ Primary loading failed: {primary_error}")
            
            # Fallback 1: Eager attention
            print("🔄 Trying fallback 1: eager attention...")
            model_kwargs["attn_implementation"] = "eager"
            
            try:
                model = VoxtralForConditionalGeneration.from_pretrained(
                    "mistralai/Voxtral-Mini-3B-2507",
                    **model_kwargs
                )
                load_time = time.time() - start_time
                print(f"✅ Model loaded with eager attention in {load_time:.2f}s")
                return model
                
            except Exception as eager_error:
                print(f"⚠️ Eager attention failed: {eager_error}")
                
                # Fallback 2: No safetensors
                print("🔄 Trying fallback 2: no safetensors...")
                model_kwargs["use_safetensors"] = False
                
                try:
                    model = VoxtralForConditionalGeneration.from_pretrained(
                        "mistralai/Voxtral-Mini-3B-2507",
                        **model_kwargs
                    )
                    load_time = time.time() - start_time
                    print(f"✅ Model loaded without safetensors in {load_time:.2f}s")
                    return model
                    
                except Exception as safetensors_error:
                    print(f"❌ All fallbacks failed: {safetensors_error}")
                    raise safetensors_error
        
    except Exception as e:
        print(f"❌ Model loading completely failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def test_model_inference(model, processor):
    """Test basic model inference"""
    print("🔍 TESTING MODEL INFERENCE")
    print("=" * 50)
    
    try:
        import torch
        import numpy as np
        
        # Create dummy audio input
        sample_rate = 16000
        duration = 1.0  # 1 second
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        print("🔄 Testing model inference...")
        start_time = time.time()
        
        # Process audio
        inputs = processor(audio=audio_data, sampling_rate=sample_rate, return_tensors="pt")
        
        # Move to same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.1,
                use_cache=True
            )
        
        inference_time = time.time() - start_time
        print(f"✅ Model inference successful in {inference_time:.3f}s")
        print(f"   Output shape: {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function"""
    print("🚀 VOXTRAL MODEL LOADING DIAGNOSTIC")
    print("=" * 60)
    print()
    
    # Test environment
    if not test_environment():
        print("❌ Environment test failed. Please fix issues above.")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed. Please install required packages.")
        return False
    
    # Test processor loading
    processor = test_processor_loading()
    if processor is None:
        print("❌ Processor loading failed. Check authentication and network.")
        return False
    
    # Test model loading
    model = test_model_loading()
    if model is None:
        print("❌ Model loading failed. Check logs above for details.")
        return False
    
    # Test inference
    if not test_model_inference(model, processor):
        print("❌ Model inference failed. Model may be corrupted.")
        return False
    
    print()
    print("🎉 ALL TESTS PASSED!")
    print("✅ Voxtral model loading is working correctly")
    print("✅ Your speech-to-speech system should now work")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
