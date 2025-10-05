"""Minimal dependency guard used to warn about outdated libraries."""

from __future__ import annotations

from importlib import metadata
from typing import Dict, List, Tuple

REQUIRED_PACKAGES: Dict[str, str] = {
    "sentence-transformers": "2.0.0",
    "torch": "1.10.0",
    "scikit-learn": "1.0.0",
}


def check_dependencies() -> Tuple[bool, List[str]]:
    """Verify that required packages meet minimum versions."""

    issues: List[str] = []
    all_valid = True

    for package, minimum in REQUIRED_PACKAGES.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            issues.append(f"Paquete faltante: {package}")
            all_valid = False
            continue

        if _normalize(installed) < _normalize(minimum):
            issues.append(f"{package} {installed} < {minimum}")
            all_valid = False

    return all_valid, issues


def _normalize(version: str) -> Tuple[int, ...]:
    parts = []
    for chunk in version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


__all__ = ["check_dependencies"]
