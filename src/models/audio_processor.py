"""
Compatibility shim for legacy audio_processor imports
Redirects to the real implementation in audio_processor_realtime.py
"""

# Import everything from the real implementation
from .audio_processor_realtime import *

# Export the main class for backward compatibility
__all__ = ["AudioProcessor"]
