"""
Environment validation module for MINIMINIMOON.

Validates Python version and critical dependencies like NumPy.
"""

import re
import sys
from typing import Tuple

# Version requirements
REQUIRED_PYTHON = (3, 10)
NUMPY_MIN = (1, 21, 0)
NUMPY_MAX_EXCL = (1, 25, 0)


class EnvironmentValidationError(Exception):
    """Raised when environment validation fails."""

    pass


def _parse_ver(version_str: str) -> Tuple[int, ...]:
    """
    Parse a version string into a tuple of integers.

    Args:
        version_str: Version string like "1.21.0" or "1.21.0rc1"

    Returns:
        Tuple of version numbers, e.g., (1, 21, 0)
    """
    # Remove any tags like rc1, +cpu, etc.
    clean_version = re.sub(r"[+\-].*$", "", version_str)
    clean_version = re.sub(r"[a-zA-Z].*$", "", clean_version)

    parts = clean_version.split(".")
    # Ensure we have at least 3 parts
    while len(parts) < 3:
        parts.append("0")

    return tuple(int(p) for p in parts[:3])


def _cmp(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """
    Compare two version tuples.

    Args:
        a: First version tuple
        b: Second version tuple

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def validate_environment() -> None:
    """
    Validate the Python environment.

    Checks:
    - Python version >= 3.10
    - NumPy version >= 1.21.0 and < 1.25.0

    Raises:
        EnvironmentValidationError: If validation fails
    """
    # Check Python version
    python_version = sys.version_info[:2]
    if python_version < REQUIRED_PYTHON:
        raise EnvironmentValidationError(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ "
            f"required, but running "
            f"{sys.version_info[0]}.{sys.version_info[1]}"
        )

    # Check NumPy version
    try:
        import numpy as np

        numpy_version = _parse_ver(np.__version__)

        if _cmp(numpy_version, NUMPY_MIN) < 0:
            raise EnvironmentValidationError(
                f"NumPy >={NUMPY_MIN[0]}.{NUMPY_MIN[1]}.{NUMPY_MIN[2]} required, "
                f"but found {np.__version__}"
            )

        if _cmp(numpy_version, NUMPY_MAX_EXCL) >= 0:
            max_ver = NUMPY_MAX_EXCL
            raise EnvironmentValidationError(
                f"NumPy <{max_ver[0]}.{max_ver[1]}.{max_ver[2]} required, "
                f"but found {np.__version__}"
            )
    except ImportError:
        raise EnvironmentValidationError("NumPy is not installed")


def cli() -> int:
    """
    Command-line interface for environment validation.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        validate_environment()
        print("✓ Environment validation passed")
        ver = sys.version_info
        print(f"  Python: {ver[0]}.{ver[1]}.{ver[2]}")
        try:
            import numpy as np

            print(f"  NumPy: {np.__version__}")
        except ImportError:
            pass
        return 0
    except EnvironmentValidationError as e:
        print(f"✗ Environment validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(cli())
