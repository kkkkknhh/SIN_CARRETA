"""
Version Validator
Ensures Python 3.10 is being used across all system modules.
"""

import os
import sys
import warnings
from typing import Tuple

def validate_python_310() -> bool:
    """
    Validate that Python 3.10 is being used.
    
    Returns:
        bool: True if Python 3.10, False otherwise
        
    Raises:
        RuntimeError: If Python version is incompatible
    """
    version = sys.version_info
    
    if version.major != 3 or version.minor != 10:
        error_msg = (
            f"Python 3.10 required for SIN_CARRETA system. "
            f"Found Python {version.major}.{version.minor}.{version.micro}. "
            f"Please use Python 3.10 to ensure compatibility with NumPy and other dependencies."
        )
        raise RuntimeError(error_msg)
    
    return True

def validate_numpy_compatibility() -> bool:
    """
    Validate NumPy version is compatible with Python 3.10.
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        import numpy as np
        
        version_parts = np.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        # Check minimum version for Python 3.10 support
        if major < 1 or (major == 1 and minor < 21):
            warnings.warn(
                f"NumPy {np.__version__} may not fully support Python 3.10. "
                f"Upgrade to NumPy >= 1.21.0 recommended.",
                UserWarning
            )
            return False
            
        # Check for potentially problematic newer versions
        if major > 1 or (major == 1 and minor >= 25):
            warnings.warn(
                f"NumPy {np.__version__} may have breaking changes. "
                f"Consider using NumPy < 1.25.0 for stability.",
                UserWarning
            )
        
        return True
        
    except ImportError:
        warnings.warn("NumPy not installed. Install with: pip install 'numpy>=1.21.0,<1.25.0'", UserWarning)
        return False


def get_python_version_info() -> Tuple[int, int, int]:
    """Get current Python version as tuple."""
    return sys.version_info[:3]


# Automatic validation on import (can be disabled by env var)
if os.getenv('SKIP_VERSION_VALIDATION') != '1':
    validate_python_310()
    validate_numpy_compatibility()
