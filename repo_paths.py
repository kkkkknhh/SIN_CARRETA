"""
Central Path Resolver for Canonical Artifacts

This module provides the single source of truth for locating the two canonical JSON artifacts:
- DECALOGO: /bundles/decalogo-industrial.latest.clean.json
- DNP: /standards/dnp-standards.latest.clean.json

All code must use these functions to resolve paths. Direct path construction is forbidden.
"""

import re
from pathlib import Path
from typing import Optional

# Resolve repository root (this file is in the root)
REPO_ROOT = Path(__file__).resolve().parent

# Canonical directories
BUNDLES_DIR = REPO_ROOT / "bundles"
STANDARDS_DIR = REPO_ROOT / "standards"

# Canonical file paths
DECALOGO_PATH = BUNDLES_DIR / "decalogo-industrial.latest.clean.json"
DNP_PATH = STANDARDS_DIR / "dnp-standards.latest.clean.json"

# Validation patterns (exact match only)
_RX_DEC = re.compile(r'^decalogo-industrial\.latest\.clean\.json$', re.IGNORECASE)
_RX_DNP = re.compile(r'^dnp-standards\.latest\.clean\.json$', re.IGNORECASE)


def assert_canonical_file(path: Path, kind: str) -> None:
    """
    Validate that a path points to a canonical file with the correct name.
    
    Args:
        path: Path to validate
        kind: Either "decalogo" or "dnp"
        
    Raises:
        ValueError: If the filename doesn't match the canonical pattern
        FileNotFoundError: If the file doesn't exist
    """
    name = path.name
    
    if kind == "decalogo":
        if not _RX_DEC.match(name):
            raise ValueError(
                f"NON-CANONICAL_DECALOGO: got '{name}', "
                f"expected 'decalogo-industrial.latest.clean.json'"
            )
    elif kind == "dnp":
        if not _RX_DNP.match(name):
            raise ValueError(
                f"NON-CANONICAL_DNP: got '{name}', "
                f"expected 'dnp-standards.latest.clean.json'"
            )
    else:
        raise ValueError(f"Unknown artifact kind: {kind}")
    
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_{kind.upper()}: {path}")


def get_decalogo_path(override: Optional[str] = None) -> Path:
    """
    Get the canonical path to the DECALOGO artifact.
    
    Args:
        override: Optional override path (must still have canonical filename)
        
    Returns:
        Path to decalogo-industrial.latest.clean.json
        
    Raises:
        ValueError: If override has non-canonical filename
        FileNotFoundError: If file doesn't exist
    """
    path = Path(override) if override else DECALOGO_PATH
    assert_canonical_file(path, "decalogo")
    return path


def get_dnp_path(override: Optional[str] = None) -> Path:
    """
    Get the canonical path to the DNP standards artifact.
    
    Args:
        override: Optional override path (must still have canonical filename)
        
    Returns:
        Path to dnp-standards.latest.clean.json
        
    Raises:
        ValueError: If override has non-canonical filename
        FileNotFoundError: If file doesn't exist
    """
    path = Path(override) if override else DNP_PATH
    assert_canonical_file(path, "dnp")
    return path


def get_repo_root() -> Path:
    """
    Get the repository root directory.
    
    Returns:
        Path to repository root
    """
    return REPO_ROOT


# Public API
__all__ = [
    'get_decalogo_path',
    'get_dnp_path',
    'get_repo_root',
    'assert_canonical_file',
    'REPO_ROOT',
    'BUNDLES_DIR',
    'STANDARDS_DIR',
    'DECALOGO_PATH',
    'DNP_PATH',
]
