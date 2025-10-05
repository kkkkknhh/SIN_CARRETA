"""Robust disk persistence helpers with explicit fallbacks."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafeWriteResult:
    """Outcome returned by safe persistence helpers."""

    status: str
    path: Optional[Path]
    key: Optional[str]
    reason: Optional[str]


_FALLBACK_DIR = Path(__file__).resolve().parent / "fallback_error_logs"
_DEFAULT_FALLBACK_DIR = _FALLBACK_DIR
_IN_MEMORY_STORE: Dict[str, Union[str, bytes]] = {}


def set_fallback_directory(path: Path) -> None:
    """Override the fallback directory (useful for tests)."""

    global _FALLBACK_DIR
    _FALLBACK_DIR = path.resolve()


def get_fallback_directory() -> Path:
    """Return the currently configured fallback directory."""

    return _FALLBACK_DIR


def reset_fallback_directory() -> None:
    """Restore the original fallback directory."""

    set_fallback_directory(_DEFAULT_FALLBACK_DIR)


def clear_in_memory_store() -> None:
    """Remove all in-memory fallback entries."""

    _IN_MEMORY_STORE.clear()


def get_in_memory_store_snapshot() -> Dict[str, Union[str, bytes]]:
    """Return a shallow copy of the in-memory fallback entries."""

    return dict(_IN_MEMORY_STORE)


def safe_write_json(path: Path, payload: Dict[str, object], *, label: str) -> SafeWriteResult:
    """Safely persist JSON payload with fallback handling."""

    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    return safe_write_text(path, serialized, label=label)


def safe_write_text(path: Path, content: str, *, label: str, encoding: str = "utf-8") -> SafeWriteResult:
    """Safely persist text content and provide explicit fallback information."""

    return _safe_write_bytes(path, content.encode(encoding), label=label)


def safe_write_bytes(path: Path, content: bytes, *, label: str) -> SafeWriteResult:
    """Safely persist binary payloads."""

    return _safe_write_bytes(path, content, label=label)


def _safe_write_bytes(path: Path, content: bytes, *, label: str) -> SafeWriteResult:
    """Core persistence helper handling permission errors and disk exhaustion."""

    path = path.resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        if not _is_recoverable_error(exc):
            raise
        logger.warning(
            "Evento '%s': no se pudo crear el directorio %s (%s)",
            label,
            path.parent,
            exc,
        )
        return _store_in_memory(path, content, label, f"mkdir:{exc}")

    try:
        with open(path, "wb") as handle:
            handle.write(content)
        logger.info("Evento '%s': contenido persistido en %s", label, path)
        return SafeWriteResult(status="primary", path=path, key=None, reason=None)
    except (PermissionError, OSError) as exc:
        if not _is_recoverable_error(exc):
            raise
        logger.warning(
            "Evento '%s': fallo al escribir en %s (%s). Se intentará fallback.",
            label,
            path,
            exc,
        )
        return _persist_to_fallback(path, content, label, exc)


def _persist_to_fallback(path: Path, content: bytes, label: str, exc: Exception) -> SafeWriteResult:
    fallback_dir = _FALLBACK_DIR
    try:
        fallback_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as dir_exc:
        if not _is_recoverable_error(dir_exc):
            raise
        logger.error(
            "Evento '%s': no se pudo preparar el directorio fallback %s (%s). Se usará memoria.",
            label,
            fallback_dir,
            dir_exc,
        )
        return _store_in_memory(path, content, label, f"fallback_mkdir:{dir_exc}")

    fallback_name = _build_fallback_name(path, label)
    fallback_path = fallback_dir / fallback_name

    try:
        with open(fallback_path, "wb") as handle:
            handle.write(content)
        logger.warning(
            "Evento '%s': contenido persistido en fallback %s tras fallo en %s",
            label,
            fallback_path,
            path,
        )
        return SafeWriteResult(
            status="fallback",
            path=fallback_path,
            key=None,
            reason=str(exc),
        )
    except (PermissionError, OSError) as fallback_exc:
        if not _is_recoverable_error(fallback_exc):
            raise
        logger.error(
            "Evento '%s': fallback en disco %s también falló (%s). Se usará memoria.",
            label,
            fallback_path,
            fallback_exc,
        )
        return _store_in_memory(path, content, label, f"fallback_write:{fallback_exc}")


def _store_in_memory(path: Path, content: bytes, label: str, reason: str) -> SafeWriteResult:
    key = f"{label}:{hashlib.sha1(str(path).encode('utf-8')).hexdigest()}"
    _IN_MEMORY_STORE[key] = content
    logger.error(
        "Evento '%s': contenido almacenado en memoria bajo la clave %s (motivo: %s)",
        label,
        key,
        reason,
    )
    return SafeWriteResult(status="memory", path=None, key=key, reason=reason)


def _is_recoverable_error(exc: Exception) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {errno.EACCES, errno.EROFS, errno.ENOSPC}
    return False


def _build_fallback_name(path: Path, label: str) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    suffix = path.suffix or ".bin"
    return f"{label}.{digest}{suffix}"
