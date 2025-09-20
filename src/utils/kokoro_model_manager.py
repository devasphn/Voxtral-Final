"""
Kokoro TTS Model Manager
Handles downloading, verification, and management of Kokoro TTS model files
"""

import os
import hashlib
import logging
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Setup logging
kokoro_manager_logger = logging.getLogger("kokoro_model_manager")
kokoro_manager_logger.setLevel(logging.INFO)

class KokoroModelManager:
    """
    Manages Kokoro TTS model files including download, verification, and caching
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the Kokoro model manager

        Args:
            cache_dir: Custom cache directory for model files
        """
        self.repo_id = "hexgrad/Kokoro-82M"
        # Use HuggingFace's default cache directory structure
        self.cache_dir = cache_dir
        self.model_files = {
            # Core model files
            "kokoro-v1_0.pth": "kokoro-v1_0.pth",

            # Voice files (key ones we need)
            "voices/af_heart.pt": "voices/af_heart.pt",
            "voices/af_bella.pt": "voices/af_bella.pt",
            "voices/af_nicole.pt": "voices/af_nicole.pt",
            "voices/af_sarah.pt": "voices/af_sarah.pt",
            "voices/hf_alpha.pt": "voices/hf_alpha.pt",
            "voices/hf_beta.pt": "voices/hf_beta.pt",
            "voices/hm_omega.pt": "voices/hm_omega.pt",
            "voices/hm_psi.pt": "voices/hm_psi.pt"
        }
        
        # Expected file sizes (approximate, in bytes) for verification
        self.expected_sizes = {
            "kokoro-v1_0.pth": 327_000_000,  # ~327MB
            "voices/af_bella.pt": 500_000,  # ~500KB
            "voices/af_heart.pt": 500_000,  # ~500KB
            "voices/af_nicole.pt": 500_000,  # ~500KB
            "voices/af_sarah.pt": 500_000,  # ~500KB
            "voices/hf_alpha.pt": 500_000,  # ~500KB
            "voices/hf_beta.pt": 500_000,  # ~500KB
            "voices/hm_omega.pt": 500_000,  # ~500KB
            "voices/hm_psi.pt": 500_000,  # ~500KB
        }

        # Initialize cache directory (will be set after download)
        self.actual_cache_dir = None
        kokoro_manager_logger.info(f"[AUDIO] Kokoro Model Manager initialized for repo: {self.repo_id}")
    
    def check_model_availability(self) -> Dict[str, bool]:
        """
        Check which model files are available locally

        Returns:
            Dict mapping file names to availability status
        """
        availability = {}

        # If we don't have the actual cache dir yet, try to find it
        if not self.actual_cache_dir:
            self._find_cache_directory()

        if not self.actual_cache_dir:
            # No cache directory found, all files unavailable
            for file_key in self.model_files.keys():
                availability[file_key] = False
            return availability

        for file_key, file_path in self.model_files.items():
            local_path = os.path.join(self.actual_cache_dir, file_path)
            availability[file_key] = os.path.exists(local_path) and os.path.getsize(local_path) > 0

        return availability

    def _find_cache_directory(self):
        """Find the actual HuggingFace cache directory for our model"""
        try:
            from huggingface_hub import snapshot_download
            # This will return the path without downloading if it exists
            cache_path = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                allow_patterns=["*.pth", "voices/*.pt"]
            )
            self.actual_cache_dir = cache_path
            kokoro_manager_logger.info(f"[FILE] Found cache directory: {cache_path}")
        except Exception as e:
            kokoro_manager_logger.debug(f"Cache directory not found: {e}")
            self.actual_cache_dir = None

    def verify_model_integrity(self) -> Dict[str, bool]:
        """
        Verify the integrity of downloaded model files

        Returns:
            Dict mapping file names to integrity status
        """
        integrity = {}

        # Make sure we have the cache directory
        if not self.actual_cache_dir:
            self._find_cache_directory()

        if not self.actual_cache_dir:
            # No cache directory, all files invalid
            for file_key in self.model_files.keys():
                integrity[file_key] = False
            return integrity

        for file_key, file_path in self.model_files.items():
            local_path = os.path.join(self.actual_cache_dir, file_path)

            if not os.path.exists(local_path):
                integrity[file_key] = False
                continue
            
            try:
                file_size = os.path.getsize(local_path)
                expected_size = self.expected_sizes.get(file_key, 0)
                
                # Check if file size is reasonable (within 20% of expected)
                if expected_size > 0:
                    size_ratio = file_size / expected_size
                    if 0.8 <= size_ratio <= 1.2:
                        integrity[file_key] = True
                    else:
                        kokoro_manager_logger.warning(f"[WARN] File size mismatch for {file_key}: {file_size} vs expected {expected_size}")
                        integrity[file_key] = False
                else:
                    # If no expected size, just check that file exists and is not empty
                    integrity[file_key] = file_size > 0
                
                # Additional verification for PyTorch files
                if file_path.endswith('.pt'):
                    try:
                        torch.load(local_path, map_location='cpu')
                        kokoro_manager_logger.debug(f"[OK] PyTorch file verified: {file_key}")
                    except Exception as e:
                        kokoro_manager_logger.error(f"[ERROR] PyTorch file corrupted: {file_key} - {e}")
                        integrity[file_key] = False
                        
            except Exception as e:
                kokoro_manager_logger.error(f"[ERROR] Error verifying {file_key}: {e}")
                integrity[file_key] = False
        
        return integrity
    
    def download_model_files(self, force_download: bool = False) -> bool:
        """
        Download all required Kokoro model files
        
        Args:
            force_download: Force re-download even if files exist
            
        Returns:
            True if all files downloaded successfully, False otherwise
        """
        kokoro_manager_logger.info("[INPUT] Starting Kokoro model download...")
        
        try:
            # Check what's already available
            if not force_download:
                availability = self.check_model_availability()
                integrity = self.verify_model_integrity()
                
                missing_files = [f for f, available in availability.items() if not available]
                corrupted_files = [f for f, valid in integrity.items() if not valid]
                
                if not missing_files and not corrupted_files:
                    kokoro_manager_logger.info("[OK] All Kokoro model files already available and verified")
                    return True
                
                if missing_files:
                    kokoro_manager_logger.info(f"[EMOJI] Missing files: {missing_files}")
                if corrupted_files:
                    kokoro_manager_logger.info(f"[CONFIG] Corrupted files to re-download: {corrupted_files}")
            
            # Download using huggingface_hub
            kokoro_manager_logger.info(f"[EMOJI] Downloading from repository: {self.repo_id}")

            start_time = time.time()

            # Download the entire repository to ensure we get all files
            local_dir = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=self.cache_dir,
                force_download=force_download,
                allow_patterns=["*.pth", "voices/*.pt"]  # Only download what we need
            )

            # Update our cache directory reference
            self.actual_cache_dir = local_dir

            download_time = time.time() - start_time
            kokoro_manager_logger.info(f"[EMOJI] Model downloaded to: {local_dir}")
            kokoro_manager_logger.info(f"[TIME] Download completed in {download_time:.2f}s")
            
            # Verify the download
            integrity = self.verify_model_integrity()
            failed_files = [f for f, valid in integrity.items() if not valid]
            
            if failed_files:
                kokoro_manager_logger.error(f"[ERROR] Download verification failed for: {failed_files}")
                return False
            
            kokoro_manager_logger.info("[OK] All Kokoro model files downloaded and verified successfully!")
            return True
            
        except HfHubHTTPError as e:
            kokoro_manager_logger.error(f"[ERROR] HTTP error downloading model: {e}")
            return False
        except Exception as e:
            kokoro_manager_logger.error(f"[ERROR] Error downloading Kokoro model: {e}")
            import traceback
            kokoro_manager_logger.error(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return False
    
    def get_model_status(self) -> Dict[str, any]:
        """
        Get comprehensive status of Kokoro model files
        
        Returns:
            Dict containing availability, integrity, and other status information
        """
        availability = self.check_model_availability()
        integrity = self.verify_model_integrity()
        
        total_files = len(self.model_files)
        available_files = sum(availability.values())
        valid_files = sum(integrity.values())
        
        status = {
            "total_files": total_files,
            "available_files": available_files,
            "valid_files": valid_files,
            "availability_percentage": (available_files / total_files) * 100,
            "integrity_percentage": (valid_files / total_files) * 100,
            "cache_directory": self.actual_cache_dir or "Not found",
            "repository": self.repo_id,
            "file_details": {
                file_key: {
                    "available": availability.get(file_key, False),
                    "valid": integrity.get(file_key, False),
                    "path": os.path.join(self.actual_cache_dir or "", file_path)
                }
                for file_key, file_path in self.model_files.items()
            }
        }
        
        return status
    
    def cleanup_cache(self) -> bool:
        """
        Clean up the model cache directory
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            import shutil
            if self.actual_cache_dir and os.path.exists(self.actual_cache_dir):
                shutil.rmtree(self.actual_cache_dir)
                kokoro_manager_logger.info(f"[CLEANUP] Cleaned up cache directory: {self.actual_cache_dir}")
                self.actual_cache_dir = None
                return True
            return True
        except Exception as e:
            kokoro_manager_logger.error(f"[ERROR] Error cleaning up cache: {e}")
            return False
    
    def get_voice_files(self) -> List[str]:
        """
        Get list of available voice files
        
        Returns:
            List of voice file names (without .pt extension)
        """
        voice_files = []
        for file_key in self.model_files.keys():
            if file_key.startswith("voices/") and file_key.endswith(".pt"):
                voice_name = os.path.basename(file_key).replace(".pt", "")
                voice_files.append(voice_name)
        
        return sorted(voice_files)

# Global instance
kokoro_model_manager = KokoroModelManager()
