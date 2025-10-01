#!/usr/bin/env python3
"""
Complete RunPod Deployment Script for Voxtral Voice AI
Handles all deployment steps including missing packages and model pre-loading
"""
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, check=True):
    """Run a shell command with logging"""
    logger.info(f"🔧 {description}")
    logger.info(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"   ❌ Failed: {e}")
        if e.stderr:
            logger.error(f"   Error: {e.stderr.strip()}")
        return False

def install_missing_packages():
    """Install any missing packages"""
    logger.info("📦 Installing missing packages...")
    
    missing_packages = [
        "pyyaml>=6.0.1",
        "pillow>=10.1.0"  # Optional but useful
    ]
    
    for package in missing_packages:
        success = run_command(
            f"pip install {package}",
            f"Installing {package}",
            check=False
        )
        if success:
            logger.info(f"   ✅ {package} installed")
        else:
            logger.warning(f"   ⚠️  {package} installation failed (may be optional)")

def verify_installation():
    """Run the updated verification script"""
    logger.info("🔍 Running verification script...")
    
    try:
        result = subprocess.run([sys.executable, "verify_requirements.py"], 
                              capture_output=True, text=True, timeout=60)
        
        logger.info("Verification output:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Verification warnings:")
            logger.warning(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Verification script timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def preload_models():
    """Pre-load models to eliminate cold starts"""
    logger.info("🤖 Pre-loading models...")
    
    try:
        result = subprocess.run([sys.executable, "preload_models.py"], 
                              capture_output=True, text=True, timeout=300)
        
        logger.info("Model pre-loading output:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Model pre-loading warnings:")
            logger.warning(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Model pre-loading timed out (5 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Model pre-loading failed: {e}")
        return False

def check_environment():
    """Check RunPod environment and requirements"""
    logger.info("🌍 Checking environment...")
    
    # Check if we're in RunPod
    runpod_indicators = [
        Path("/workspace").exists(),
        os.environ.get("RUNPOD_POD_ID"),
        os.environ.get("RUNPOD_API_KEY")
    ]
    
    if any(runpod_indicators):
        logger.info("   ✅ RunPod environment detected")
    else:
        logger.info("   ℹ️  Not in RunPod environment (local development)")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("   ✅ Virtual environment active")
    else:
        logger.warning("   ⚠️  No virtual environment detected")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"   🐍 Python version: {python_version}")
    
    if sys.version_info >= (3, 8):
        logger.info("   ✅ Python version compatible")
    else:
        logger.error("   ❌ Python 3.8+ required")
        return False
    
    return True

def start_server(port=8000):
    """Start the Voxtral Voice AI server"""
    logger.info(f"🚀 Starting Voxtral Voice AI server on port {port}...")
    
    # Check if port is available
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        logger.warning(f"   ⚠️  Port {port} is already in use")
        return False
    
    # Start the server
    command = f"python -m uvicorn src.api.ui_server_simple:app --host 0.0.0.0 --port {port} --workers 1"
    
    logger.info(f"   Command: {command}")
    logger.info(f"   🌐 Server will be available at:")
    
    if Path("/workspace").exists():
        # RunPod environment
        logger.info(f"   📱 Web Interface: https://[POD_ID]-{port}.proxy.runpod.net")
        logger.info(f"   🔌 WebSocket: wss://[POD_ID]-8765.proxy.runpod.net")
    else:
        # Local environment
        logger.info(f"   📱 Web Interface: http://localhost:{port}")
        logger.info(f"   🔌 WebSocket: ws://localhost:8765")
    
    logger.info("   💡 Replace [POD_ID] with your actual RunPod pod ID")
    logger.info("   🛑 Press Ctrl+C to stop the server")
    
    try:
        # Start server (this will block)
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main deployment function"""
    logger.info("🚀 Voxtral Voice AI - Complete RunPod Deployment")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    # Step 2: Install missing packages
    install_missing_packages()
    
    # Step 3: Verify installation
    logger.info("\n" + "=" * 40)
    verification_success = verify_installation()
    
    if verification_success:
        logger.info("✅ All required packages verified")
    else:
        logger.warning("⚠️  Some verification issues found, but continuing...")
    
    # Step 4: Pre-load models
    logger.info("\n" + "=" * 40)
    preload_success = preload_models()
    
    if preload_success:
        logger.info("✅ Models pre-loaded successfully")
    else:
        logger.warning("⚠️  Model pre-loading had issues, but continuing...")
    
    # Step 5: Final summary
    setup_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"⏱️  Setup completed in {setup_time:.1f}s")
    
    if verification_success and preload_success:
        logger.info("🎉 Deployment successful! System ready for ultra-low latency voice AI")
    else:
        logger.info("⚠️  Deployment completed with warnings")
    
    # Step 6: Ask user if they want to start the server
    logger.info("\n" + "=" * 40)
    logger.info("🚀 Ready to start the server!")
    
    try:
        response = input("Start the Voxtral Voice AI server now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            start_server()
        else:
            logger.info("💡 To start the server manually, run:")
            logger.info("   python -m uvicorn src.api.ui_server_simple:app --host 0.0.0.0 --port 8000")
    except KeyboardInterrupt:
        logger.info("\n🛑 Deployment interrupted by user")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
