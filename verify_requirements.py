#!/usr/bin/env python3
"""
Voxtral-Final Requirements Verification Script
Verifies all dependencies are installed correctly for RunPod deployment
"""

import sys
import subprocess
import importlib
import pkg_resources
from packaging import version
import os
import torch
from pathlib import Path

def print_header():
    """Print verification header"""
    print("=" * 60)
    print("🔍 VOXTRAL-FINAL REQUIREMENTS VERIFICATION")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("📋 Checking Python version...")
    
    python_version = sys.version_info
    required_major, required_minor = 3, 8
    
    if python_version.major < required_major or (python_version.major == required_major and python_version.minor < required_minor):
        print(f"❌ Python {required_major}.{required_minor}+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - OK")
    return True

def check_cuda_availability():
    """Check CUDA availability and version"""
    print("\n🔧 Checking CUDA availability...")
    
    try:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"✅ CUDA {cuda_version} available")
            print(f"✅ {device_count} GPU(s) detected")
            print(f"✅ Current device: {device_name}")
            
            # Check VRAM
            if device_count > 0:
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ GPU Memory: {memory_total:.1f} GB")
                
                if memory_total < 4.0:
                    print("⚠️  Warning: Less than 4GB VRAM detected. Performance may be limited.")
            
            return True
        else:
            print("❌ CUDA not available - CPU mode will be used")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

def check_package_version(package_name, required_version=None, import_name=None):
    """Check if a package is installed with correct version"""
    try:
        # Use import_name if provided, otherwise use package_name
        actual_import_name = import_name or package_name
        
        # Try to import the package
        importlib.import_module(actual_import_name)
        
        # Check version if required
        if required_version:
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                if version.parse(installed_version) >= version.parse(required_version):
                    print(f"✅ {package_name} {installed_version} (>= {required_version})")
                    return True
                else:
                    print(f"❌ {package_name} {installed_version} (< {required_version} required)")
                    return False
            except pkg_resources.DistributionNotFound:
                print(f"⚠️  {package_name} imported but version check failed")
                return True  # Package works but version unknown
        else:
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                print(f"✅ {package_name} {installed_version}")
            except pkg_resources.DistributionNotFound:
                print(f"✅ {package_name} (version unknown)")
            return True
            
    except ImportError as e:
        print(f"❌ {package_name} not found: {e}")
        return False
    except Exception as e:
        print(f"❌ {package_name} check failed: {e}")
        return False

def check_core_dependencies():
    """Check core Python dependencies"""
    print("\n📦 Checking core dependencies...")
    
    dependencies = [
        # Core ML libraries
        ("torch", "2.0.0"),
        ("transformers", "4.56.0"),
        ("accelerate", "0.25.0"),
        ("tokenizers", "0.20.0"),
        
        # Audio processing
        ("librosa", "0.10.1"),
        ("soundfile", "0.12.1"),
        ("numpy", "1.24.4"),
        ("scipy", "1.11.4"),
        
        # Web framework
        ("fastapi", "0.104.1"),
        ("uvicorn", "0.24.0"),
        ("websockets", "12.0"),
        ("pydantic", "2.5.3"),
        ("aiofiles", "23.2.1"),
        
        # Configuration
        ("pyyaml", None, "yaml"),
        ("python-dotenv", "1.0.0", "dotenv"),
        ("psutil", "5.9.6"),
        
        # System utilities
        ("pillow", "10.1.0", "PIL"),
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for dep in dependencies:
        if len(dep) == 2:
            package_name, required_version = dep
            import_name = None
        else:
            package_name, required_version, import_name = dep
            
        if check_package_version(package_name, required_version, import_name):
            success_count += 1
    
    print(f"\n📊 Core dependencies: {success_count}/{total_count} OK")
    return success_count == total_count

def check_specialized_dependencies():
    """Check specialized dependencies"""
    print("\n🎯 Checking specialized dependencies...")
    
    specialized_deps = [
        ("mistral-common", "1.8.1", "mistral_common"),
        ("kokoro", "0.9.4"),
        ("huggingface-hub", "0.26.0", "huggingface_hub"),
    ]
    
    success_count = 0
    total_count = len(specialized_deps)
    
    for dep in specialized_deps:
        if len(dep) == 2:
            package_name, required_version = dep
            import_name = None
        else:
            package_name, required_version, import_name = dep
            
        if check_package_version(package_name, required_version, import_name):
            success_count += 1
    
    print(f"\n📊 Specialized dependencies: {success_count}/{total_count} OK")
    return success_count == total_count

def check_environment_variables():
    """Check required environment variables"""
    print("\n🔑 Checking environment variables...")
    
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("✅ HF_TOKEN is set")
        return True
    else:
        print("⚠️  HF_TOKEN not set (may be required for Voxtral model)")
        return False

def check_file_structure():
    """Check critical file structure"""
    print("\n📁 Checking file structure...")
    
    critical_files = [
        "config.yaml",
        "src/api/ui_server_realtime.py",
        "src/models/voxtral_model_realtime.py",
        "src/models/kokoro_model_realtime.py",
        "src/models/unified_model_manager.py",
        "src/utils/config.py",
        "src/utils/logging_config.py",
    ]
    
    success_count = 0
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            success_count += 1
        else:
            print(f"❌ {file_path} - Missing")
    
    print(f"\n📊 File structure: {success_count}/{len(critical_files)} OK")
    return success_count == len(critical_files)

def main():
    """Main verification function"""
    print_header()
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA Availability", check_cuda_availability),
        ("Core Dependencies", check_core_dependencies),
        ("Specialized Dependencies", check_specialized_dependencies),
        ("Environment Variables", check_environment_variables),
        ("File Structure", check_file_structure),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All verification checks passed! System ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
