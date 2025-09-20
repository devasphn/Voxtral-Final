"""
Compatibility Layer for Missing Dependencies
Provides fallback implementations and graceful degradation when packages are missing
"""

import sys
import logging
from typing import Any, Optional, Dict, List
import warnings

# Setup logging
compat_logger = logging.getLogger("compatibility")
compat_logger.setLevel(logging.INFO)

class MissingPackageError(ImportError):
    """Custom exception for missing packages"""
    pass

class CompatibilityManager:
    """Manages compatibility and fallbacks for missing packages"""
    
    def __init__(self):
        self.missing_packages = []
        self.available_packages = []
        self.fallback_mode = False
        
    def check_package(self, package_name: str, import_name: str = None) -> bool:
        """Check if a package is available"""
        if import_name is None:
            import_name = package_name
            
        try:
            __import__(import_name)
            self.available_packages.append(package_name)
            compat_logger.info(f"[OK] {package_name} is available")
            return True
        except ImportError:
            self.missing_packages.append(package_name)
            compat_logger.warning(f"[WARN] {package_name} is not available")
            return False
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get a comprehensive status report"""
        return {
            "available_packages": self.available_packages,
            "missing_packages": self.missing_packages,
            "fallback_mode": self.fallback_mode,
            "total_checked": len(self.available_packages) + len(self.missing_packages)
        }

# Global compatibility manager
compat_manager = CompatibilityManager()

# Check critical packages
def check_transformers_voxtral():
    """Check if Voxtral is available in transformers"""
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        compat_logger.info("[OK] Voxtral classes available in transformers")
        return True
    except ImportError:
        compat_logger.warning("[WARN] Voxtral classes not available - using fallback")
        return False

def check_mistral_common():
    """Check if mistral-common is available"""
    return compat_manager.check_package("mistral-common", "mistral_common")

# Orpheus TTS support removed - using Kokoro TTS only

def check_pydantic_settings():
    """Check if pydantic-settings is available"""
    return compat_manager.check_package("pydantic-settings", "pydantic_settings")

def check_vllm():
    """Check if vllm is available"""
    return compat_manager.check_package("vllm")

# Fallback implementations
class FallbackVoxtralModel:
    """Fallback implementation when Voxtral is not available"""
    
    def __init__(self):
        self.model_name = "mistralai/Voxtral-Mini-3B-2507"
        self.is_initialized = False
        compat_logger.warning("Using fallback Voxtral model - limited functionality")
    
    def get_model_info(self):
        return {
            "status": "fallback_mode",
            "model_name": self.model_name,
            "message": "Voxtral not available - install transformers>=4.56.0 or from source"
        }
    
    async def process_realtime_chunk(self, audio_data, chunk_id):
        raise MissingPackageError("Voxtral not available - please install transformers>=4.56.0")

class FallbackAutoProcessor:
    """Fallback implementation when AutoProcessor is not available"""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise MissingPackageError("AutoProcessor not available - please install transformers>=4.56.0")

# Orpheus TTS fallback model removed - using Kokoro TTS only

class FallbackConfig:
    """Fallback configuration when pydantic-settings is not available"""
    
    def __init__(self):
        # Default configuration values
        self.server = type('obj', (object,), {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False
        })()
        
        self.model = type('obj', (object,), {
            'voxtral': type('obj', (object,), {
                'model_name': 'mistralai/Voxtral-Mini-3B-2507',
                'max_context_length': 130072,
                'torch_dtype': 'bfloat16'
            })(),
            'kokoro': type('obj', (object,), {
                'model_name': 'hexgrad/Kokoro-82M',
                'default_voice': 'hm_omega'
            })()
        })()
        
        self.audio = type('obj', (object,), {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'channels': 1,
            'format': 'float32'
        })()
        
        compat_logger.warning("Using fallback configuration - pydantic-settings not available")

# Smart imports with fallbacks
def get_voxtral_classes():
    """Get Voxtral classes with fallback"""
    if check_transformers_voxtral():
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        return VoxtralForConditionalGeneration, AutoProcessor
    else:
        compat_logger.warning("Voxtral not available - using fallback")
        return FallbackVoxtralModel, None

# Orpheus TTS support removed - using Kokoro TTS only

def get_config():
    """Get configuration with fallback"""
    if check_pydantic_settings():
        try:
            from src.utils.config import config
            return config
        except ImportError:
            pass
    
    compat_logger.warning("Configuration not available - using fallback")
    return FallbackConfig()

# Initialization function
def initialize_compatibility():
    """Initialize compatibility layer and check all packages"""
    compat_logger.info("Initializing compatibility layer...")
    
    # Check all critical packages
    packages_to_check = [
        ("transformers", "transformers"),
        ("mistral-common", "mistral_common"),
        ("kokoro", "kokoro"),
        ("pydantic-settings", "pydantic_settings"),
        ("torch", "torch"),
        ("fastapi", "fastapi"),
        ("websockets", "websockets"),
        ("librosa", "librosa"),
        ("numpy", "numpy")
    ]
    
    for package_name, import_name in packages_to_check:
        compat_manager.check_package(package_name, import_name)
    
    # Check Voxtral specifically
    voxtral_available = check_transformers_voxtral()
    
    # Determine if we need fallback mode
    critical_missing = [
        pkg for pkg in ["transformers", "torch", "fastapi", "numpy"] 
        if pkg in compat_manager.missing_packages
    ]
    
    if critical_missing:
        compat_manager.fallback_mode = True
        compat_logger.error(f"[ERROR] Critical packages missing: {critical_missing}")
        compat_logger.error("System will run in limited fallback mode")
    elif not voxtral_available:
        compat_manager.fallback_mode = True
        compat_logger.warning("[WARN] Voxtral not available - some features will be limited")
    else:
        compat_logger.info("[OK] All critical packages available")

    # Print status report
    status = compat_manager.get_status_report()
    compat_logger.info(f"[INFO] Compatibility Status:")
    compat_logger.info(f"   Available: {len(status['available_packages'])}")
    compat_logger.info(f"   Missing: {len(status['missing_packages'])}")
    compat_logger.info(f"   Fallback mode: {status['fallback_mode']}")
    
    if status['missing_packages']:
        compat_logger.warning(f"   Missing packages: {', '.join(status['missing_packages'])}")
    
    return status

# Auto-initialize when imported
if __name__ != "__main__":
    initialize_compatibility()

# Test function
def test_compatibility():
    """Test the compatibility layer"""
    print("[EMOJI] Testing Compatibility Layer")
    print("=" * 50)
    
    status = initialize_compatibility()
    
    print(f"Available packages: {status['available_packages']}")
    print(f"Missing packages: {status['missing_packages']}")
    print(f"Fallback mode: {status['fallback_mode']}")
    
    # Test fallback classes
    print("\nTesting fallback implementations:")
    
    fallback_voxtral = FallbackVoxtralModel()
    print(f"Fallback Voxtral: {fallback_voxtral.get_model_info()}")
    
    # Orpheus fallback removed - using Kokoro TTS only
    print("Orpheus TTS support removed - using Kokoro TTS exclusively")
    
    fallback_config = FallbackConfig()
    print(f"Fallback Config: server.port = {fallback_config.server.port}")
    
    print("[OK] Compatibility layer test completed")

if __name__ == "__main__":
    test_compatibility()
