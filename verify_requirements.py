#!/usr/bin/env python3
"""
Requirements Verification Script for Voxtral Voice AI
Verifies all dependencies are correctly installed and compatible
Optimized for RunPod deployment
"""
import sys
import subprocess
import importlib
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RequirementsVerifier:
    """Verify and validate all project requirements"""
    
    def __init__(self):
        self.results = {
            "core_packages": {},
            "ai_packages": {},
            "web_packages": {},
            "audio_packages": {},
            "utility_packages": {},
            "errors": [],
            "warnings": []
        }
        
        # Define package categories and their requirements
        self.package_categories = {
            "core_packages": [
                ("torch", "2.4.1", "PyTorch with CUDA support"),
                ("torchvision", "0.19.1", "PyTorch vision utilities"),
                ("torchaudio", "2.4.1", "PyTorch audio processing")
            ],
            "ai_packages": [
                ("transformers", "4.45.2", "Hugging Face Transformers"),
                ("accelerate", "0.24.1", "Hugging Face Accelerate"),
                ("tokenizers", "0.15.0", "Fast tokenizers"),
                ("huggingface_hub", "0.25.1", "Hugging Face Hub")
            ],
            "web_packages": [
                ("fastapi", "0.104.1", "FastAPI web framework"),
                ("uvicorn", "0.24.0", "ASGI server"),
                ("websockets", "12.0", "WebSocket support"),
                ("pydantic", "2.5.3", "Data validation"),
                ("httpx", "0.25.2", "HTTP client")
            ],
            "audio_packages": [
                ("librosa", "0.10.1", "Audio analysis"),
                ("soundfile", "0.12.1", "Audio file I/O"),
                ("numpy", "1.24.4", "Numerical computing"),
                ("scipy", "1.11.4", "Scientific computing")
            ],
            "utility_packages": [
                ("pyyaml", "6.0.1", "YAML parser"),
                ("psutil", "5.9.6", "System utilities"),
                ("pillow", "10.1.0", "Image processing")
            ]
        }
    
    def verify_all_packages(self) -> bool:
        """Verify all package categories"""
        logger.info("🔍 Starting comprehensive package verification...")
        
        all_passed = True
        
        for category, packages in self.package_categories.items():
            logger.info(f"\n📦 Verifying {category.replace('_', ' ').title()}...")
            category_passed = self._verify_category(category, packages)
            all_passed = all_passed and category_passed
        
        # Additional checks
        self._verify_cuda_support()
        self._verify_model_compatibility()
        self._verify_runpod_compatibility()
        
        return all_passed
    
    def _verify_category(self, category: str, packages: List[Tuple[str, str, str]]) -> bool:
        """Verify a category of packages"""
        category_passed = True
        
        for package_name, expected_version, description in packages:
            try:
                # Import the package
                module = importlib.import_module(package_name)
                
                # Get version
                version = getattr(module, '__version__', 'unknown')
                
                # Check version compatibility
                if self._is_version_compatible(version, expected_version):
                    logger.info(f"  ✅ {package_name} {version} - {description}")
                    self.results[category][package_name] = {
                        "status": "ok",
                        "version": version,
                        "expected": expected_version,
                        "description": description
                    }
                else:
                    logger.warning(f"  ⚠️  {package_name} {version} (expected {expected_version}) - {description}")
                    self.results[category][package_name] = {
                        "status": "version_mismatch",
                        "version": version,
                        "expected": expected_version,
                        "description": description
                    }
                    self.results["warnings"].append(f"{package_name} version mismatch")
                    
            except ImportError as e:
                logger.error(f"  ❌ {package_name} - NOT INSTALLED - {description}")
                self.results[category][package_name] = {
                    "status": "missing",
                    "error": str(e),
                    "expected": expected_version,
                    "description": description
                }
                self.results["errors"].append(f"{package_name} not installed")
                category_passed = False
                
            except Exception as e:
                logger.error(f"  ❌ {package_name} - ERROR: {e}")
                self.results[category][package_name] = {
                    "status": "error",
                    "error": str(e),
                    "expected": expected_version,
                    "description": description
                }
                self.results["errors"].append(f"{package_name} error: {e}")
                category_passed = False
        
        return category_passed
    
    def _is_version_compatible(self, actual: str, expected: str) -> bool:
        """Check if version is compatible (allowing patch version differences)"""
        if actual == "unknown":
            return False
        
        try:
            # Extract major.minor from both versions
            actual_parts = actual.split('.')[:2]
            expected_parts = expected.split('.')[:2]
            
            return actual_parts == expected_parts
        except:
            return actual == expected
    
    def _verify_cuda_support(self):
        """Verify CUDA support"""
        logger.info("\n🖥️  Verifying CUDA Support...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                
                logger.info(f"  ✅ CUDA Available: {cuda_version}")
                logger.info(f"  ✅ GPU Count: {gpu_count}")
                logger.info(f"  ✅ GPU Name: {gpu_name}")
                
                self.results["cuda"] = {
                    "available": True,
                    "version": cuda_version,
                    "gpu_count": gpu_count,
                    "gpu_name": gpu_name
                }
            else:
                logger.warning("  ⚠️  CUDA not available - CPU mode only")
                self.results["cuda"] = {"available": False}
                self.results["warnings"].append("CUDA not available")
                
        except Exception as e:
            logger.error(f"  ❌ CUDA verification failed: {e}")
            self.results["cuda"] = {"error": str(e)}
            self.results["errors"].append(f"CUDA verification failed: {e}")
    
    def _verify_model_compatibility(self):
        """Verify model compatibility"""
        logger.info("\n🤖 Verifying Model Compatibility...")
        
        # Test Voxtral model import
        try:
            from transformers import VoxtralForConditionalGeneration, AutoProcessor
            logger.info("  ✅ Voxtral model classes available")
            self.results["voxtral_compatibility"] = True
        except Exception as e:
            logger.error(f"  ❌ Voxtral model import failed: {e}")
            self.results["voxtral_compatibility"] = False
            self.results["errors"].append(f"Voxtral import failed: {e}")
        
        # Test Kokoro TTS
        try:
            import kokoro
            logger.info("  ✅ Kokoro TTS available")
            self.results["kokoro_compatibility"] = True
        except Exception as e:
            logger.warning(f"  ⚠️  Kokoro TTS import failed: {e}")
            self.results["kokoro_compatibility"] = False
            self.results["warnings"].append(f"Kokoro import failed: {e}")
    
    def _verify_runpod_compatibility(self):
        """Verify RunPod-specific compatibility"""
        logger.info("\n🚀 Verifying RunPod Compatibility...")
        
        # Check for RunPod environment
        import os
        runpod_indicators = [
            "/workspace",
            "RUNPOD_POD_ID",
            "RUNPOD_API_KEY"
        ]
        
        runpod_detected = any(
            os.path.exists(indicator) if indicator.startswith('/') 
            else os.environ.get(indicator) 
            for indicator in runpod_indicators
        )
        
        if runpod_detected:
            logger.info("  ✅ RunPod environment detected")
            self.results["runpod_environment"] = True
        else:
            logger.info("  ℹ️  Not running in RunPod environment")
            self.results["runpod_environment"] = False
        
        # Check WebSocket support
        try:
            import websockets
            logger.info("  ✅ WebSocket support available")
            self.results["websocket_support"] = True
        except Exception as e:
            logger.error(f"  ❌ WebSocket support failed: {e}")
            self.results["websocket_support"] = False
            self.results["errors"].append(f"WebSocket support failed: {e}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive verification report"""
        report = "\n" + "="*60 + "\n"
        report += "VOXTRAL VOICE AI - REQUIREMENTS VERIFICATION REPORT\n"
        report += "="*60 + "\n"
        
        # Summary
        total_errors = len(self.results["errors"])
        total_warnings = len(self.results["warnings"])
        
        if total_errors == 0:
            report += "✅ OVERALL STATUS: PASSED\n"
        else:
            report += "❌ OVERALL STATUS: FAILED\n"
        
        report += f"Errors: {total_errors}, Warnings: {total_warnings}\n\n"
        
        # Errors
        if self.results["errors"]:
            report += "❌ ERRORS:\n"
            for error in self.results["errors"]:
                report += f"  - {error}\n"
            report += "\n"
        
        # Warnings
        if self.results["warnings"]:
            report += "⚠️  WARNINGS:\n"
            for warning in self.results["warnings"]:
                report += f"  - {warning}\n"
            report += "\n"
        
        # CUDA Info
        if "cuda" in self.results:
            cuda_info = self.results["cuda"]
            if cuda_info.get("available"):
                report += f"🖥️  CUDA: {cuda_info.get('version')} ({cuda_info.get('gpu_name')})\n"
            else:
                report += "🖥️  CUDA: Not available\n"
        
        # RunPod Info
        if self.results.get("runpod_environment"):
            report += "🚀 RunPod Environment: Detected\n"
        else:
            report += "🚀 RunPod Environment: Not detected\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report

def main():
    """Main verification function"""
    verifier = RequirementsVerifier()
    
    # Run verification
    success = verifier.verify_all_packages()
    
    # Generate and print report
    report = verifier.generate_report()
    print(report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
