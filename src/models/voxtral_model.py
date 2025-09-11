"""
Compatibility shim for legacy voxtral_model imports
Redirects to the real implementation in voxtral_model_realtime.py
"""

# Import everything from the real implementation
from .voxtral_model_realtime import *

# Ensure voxtral_model instance is available for legacy imports
from .voxtral_model_realtime import voxtral_model

# Export the main components for backward compatibility
__all__ = ["VoxtralModel", "voxtral_model"]
