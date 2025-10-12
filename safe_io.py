"""
Safe I/O operations with fallback mechanisms.

This module provides safe file writing operations with multiple fallback
strategies in case of permission errors or disk space issues.
"""

import errno
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# In-memory store for when disk operations fail
_IN_MEMORY_STORE: Dict[str, bytes] = {}

# Fallback directory for when primary path fails
_FALLBACK_DIRECTORY: Optional[Path] = None


@dataclass
class WriteResult:
    """Result of a safe write operation."""

    status: str  # "success", "fallback", or "memory"
    path: Optional[Path] = None
    key: Optional[str] = None
    error: Optional[str] = None


def set_fallback_directory(path: Path) -> None:
    """Set the fallback directory for failed writes."""
    global _FALLBACK_DIRECTORY
    _FALLBACK_DIRECTORY = path
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def get_fallback_directory() -> Optional[Path]:
    """Get the current fallback directory."""
    return _FALLBACK_DIRECTORY


def clear_in_memory_store() -> None:
    """Clear the in-memory store."""
    global _IN_MEMORY_STORE
    _IN_MEMORY_STORE.clear()


def get_in_memory_store_snapshot() -> Dict[str, bytes]:
    """Get a snapshot of the in-memory store."""
    return _IN_MEMORY_STORE.copy()


def safe_write_text(
    target: Path, content: str, label: str = "default", encoding: str = "utf-8"
) -> WriteResult:
    """
    Safely write text to a file with fallback mechanisms.

    Args:
        target: Target file path
        content: Text content to write
        label: Label for this write operation
        encoding: Text encoding (default: utf-8)

    Returns:
        WriteResult indicating success or fallback status
    """
    content_bytes = content.encode(encoding)

    # Try primary path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return WriteResult(status="success", path=target)
    except PermissionError as e:
        # Try fallback directory
        if _FALLBACK_DIRECTORY is not None:
            try:
                fallback_path = _FALLBACK_DIRECTORY / target.name
                fallback_path.write_text(content, encoding=encoding)
                return WriteResult(status="fallback", path=fallback_path)
            except Exception:
                pass
        # Fall through to memory storage
    except OSError as e:
        # Check if it's a disk full error
        if e.errno == errno.ENOSPC:
            pass  # Fall through to memory storage
        else:
            return WriteResult(
                status="error", error=f"OS error: {e}"
            )
    except Exception as e:
        return WriteResult(status="error", error=str(e))

    # Final fallback: store in memory
    key = f"{label}:{target.name}"
    _IN_MEMORY_STORE[key] = content_bytes
    return WriteResult(status="memory", key=key)


def safe_write_bytes(
    target: Path, content: bytes, label: str = "default"
) -> WriteResult:
    """
    Safely write bytes to a file with fallback mechanisms.

    Args:
        target: Target file path
        content: Bytes content to write
        label: Label for this write operation

    Returns:
        WriteResult indicating success or fallback status
    """
    # Try primary path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return WriteResult(status="success", path=target)
    except PermissionError:
        # Try fallback directory
        if _FALLBACK_DIRECTORY is not None:
            try:
                fallback_path = _FALLBACK_DIRECTORY / target.name
                fallback_path.write_bytes(content)
                return WriteResult(status="fallback", path=fallback_path)
            except Exception:
                pass
        # Fall through to memory storage
    except OSError as e:
        if e.errno == errno.ENOSPC:
            pass  # Fall through to memory storage
        else:
            return WriteResult(status="error", error=f"OS error: {e}")
    except Exception as e:
        return WriteResult(status="error", error=str(e))

    # Final fallback: store in memory
    key = f"{label}:{target.name}"
    _IN_MEMORY_STORE[key] = content
    return WriteResult(status="memory", key=key)
