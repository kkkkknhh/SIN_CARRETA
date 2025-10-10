"""Environment validation contracts for SIN_CARRETA.

This module exposes deterministic validation routines that must be invoked
explicitly by consumers (CLI entry points, notebooks, CI pipelines, etc.).
No validation executes at import time; callers are responsible for invoking
``validate_environment`` and handling raised errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ValidationDiagnostic:
    """Structured diagnostic data describing a validation check.

    Attributes:
        check: Identifier for the validation routine.
        passed: ``True`` when the validation succeeded.
        details: Contextual metadata that aids in troubleshooting failures.
    """

    check: str
    passed: bool
    details: Dict[str, str]


class EnvironmentValidationError(RuntimeError):
    """Raised when an environment contract is violated."""

    def __init__(self, message: str, diagnostic: ValidationDiagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


Observer = Callable[[ValidationDiagnostic], None]


def get_python_version_info() -> Tuple[int, int, int]:
    """Return the active interpreter version as ``(major, minor, micro)``."""

    return sys.version_info[:3]


def _parse_version_segment(segment: str) -> int:
    """Extract the leading integer component from a version segment."""

    digits = []
    for character in segment:
        if character.isdigit():
            digits.append(character)
        else:
            break

    if not digits:
        raise ValueError(f"Unable to parse numeric component from '{segment}'.")

    return int("".join(digits))


def validate_python_310() -> ValidationDiagnostic:
    """Ensure that Python 3.10 is being used.

    Invariants:
        - The interpreter major version must be ``3``.
        - The interpreter minor version must be ``10``.

    Postconditions:
        - Returns a :class:`ValidationDiagnostic` describing the interpreter.
        - Raises :class:`EnvironmentValidationError` when the invariants are
          violated. The exception contains the same diagnostic payload.
    """

    version = get_python_version_info()
    passed = (version[0], version[1]) == (3, 10)
    diagnostic = ValidationDiagnostic(
        check="python_310",
        passed=passed,
        details={
            "expected_major": "3",
            "expected_minor": "10",
            "actual": ".".join(str(part) for part in version),
        },
    )

    if not passed:
        message = (
            "Python 3.10 is required for SIN_CARRETA. "
            f"Detected interpreter {diagnostic.details['actual']}."
        )
        raise EnvironmentValidationError(message, diagnostic)

    return diagnostic


def validate_numpy_compatibility() -> ValidationDiagnostic:
    """Validate that NumPy satisfies the supported contract for Python 3.10.

    Preconditions:
        - NumPy must be importable in the active environment.

    Postconditions:
        - Returns a :class:`ValidationDiagnostic` describing the NumPy version
          when the contract is satisfied.
        - Raises :class:`EnvironmentValidationError` with diagnostics when the
          contract is violated or NumPy cannot be imported.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded by contract
        diagnostic = ValidationDiagnostic(
            check="numpy_import",
            passed=False,
            details={
                "expected_min_version": ">=1.21.0",
                "expected_max_version": "<1.25.0",
                "actual": "missing",
            },
        )
        message = (
            "NumPy is required but could not be imported. "
            "Install with: pip install 'numpy>=1.21.0,<1.25.0'."
        )
        raise EnvironmentValidationError(message, diagnostic) from exc

    version_str = np.__version__
    parts = version_str.split(".")
    try:
        major = _parse_version_segment(parts[0])
        minor = _parse_version_segment(parts[1])
    except (IndexError, ValueError) as exc:
        diagnostic = ValidationDiagnostic(
            check="numpy_version",
            passed=False,
            details={
                "expected_min_version": ">=1.21.0",
                "expected_max_version": "<1.25.0",
                "actual": version_str,
                "error": str(exc),
            },
        )
        message = (
            "NumPy version string could not be parsed. "
            f"Detected value '{version_str}'."
        )
        raise EnvironmentValidationError(message, diagnostic) from exc

    version_tuple = (major, minor)
    min_supported = (1, 21)
    max_supported = (1, 25)
    passed = min_supported <= version_tuple < max_supported
    diagnostic = ValidationDiagnostic(
        check="numpy_version",
        passed=passed,
        details={
            "expected_min_version": ">=1.21.0",
            "expected_max_version": "<1.25.0",
            "actual": version_str,
        },
    )

    if not passed:
        message = (
            "NumPy version incompatible with Python 3.10. "
            f"Detected {version_str}; expected {min_supported[0]}.{min_supported[1]}.x "
            f"through {max_supported[0]}.{max_supported[1]-1}.x."
        )
        raise EnvironmentValidationError(message, diagnostic)

    return diagnostic


def validate_environment(observer: Optional[Observer] = None) -> List[ValidationDiagnostic]:
    """Run all environment validations and return their diagnostics.

    This function should be invoked explicitly by upstream entry points. Every
    successful validation emits its diagnostic payload to ``observer`` when
    provided, enabling tests or CI systems to capture structured data.

    Postconditions:
        - Returns a list of diagnostics for successful validations.
        - Raises :class:`EnvironmentValidationError` on the first failure after
          invoking ``observer`` with the failing diagnostic (if supplied).
    """

    diagnostics: List[ValidationDiagnostic] = []
    checks = (validate_python_310, validate_numpy_compatibility)

    for check in checks:
        try:
            diagnostic = check()
        except EnvironmentValidationError as error:
            if observer is not None:
                observer(error.diagnostic)
            raise
        else:
            diagnostics.append(diagnostic)
            if observer is not None:
                observer(diagnostic)

    return diagnostics


__all__ = [
    "EnvironmentValidationError",
    "ValidationDiagnostic",
    "get_python_version_info",
    "validate_environment",
    "validate_numpy_compatibility",
    "validate_python_310",
]
