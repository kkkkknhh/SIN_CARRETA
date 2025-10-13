#!/usr/bin/env python3
"""
Python Version Enforcement
Strict Python 3.10-3.12 compatibility checker for MINIMINIMOON.

This module provides runtime Python version validation with clear error
messages and installation guidance.
"""

import sys
from typing import NoReturn, Tuple

MIN_PYTHON_VERSION: Tuple[int, int] = (3, 10)
MAX_PYTHON_VERSION: Tuple[int, int] = (3, 12)


def check_python_version(strict: bool = True) -> Tuple[bool, str]:
    """
    Check if current Python version is within supported range.

    Args:
        strict: If True, exit on version mismatch. If False, return status.

    Returns:
        Tuple of (is_valid, message)
    """
    current = sys.version_info
    current_version = f"{current.major}.{current.minor}.{current.micro}"

    # Check minimum version
    if (current.major, current.minor) < MIN_PYTHON_VERSION:
        message = f"""
╔════════════════════════════════════════════════════════════════════╗
║  PYTHON VERSION TOO OLD                                            ║
╚════════════════════════════════════════════════════════════════════╝

Current Python version: {current_version}
Required: Python >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}

MINIMINIMOON requires Python 3.10 or newer for:
  • NumPy >=1.21.0 (first version with Python 3.10 wheels)
  • Pattern matching syntax (PEP 634)
  • Modern type hints (PEP 604, 612, 613)
  • Performance improvements in dict and set operations

INSTALLATION INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option 1: Using pyenv (Recommended)
  $ curl https://pyenv.run | bash
  $ pyenv install 3.10.13
  $ pyenv local 3.10.13

Option 2: Using Anaconda/Miniconda
  $ conda create -n miniminimoon python=3.10
  $ conda activate miniminimoon

Option 3: System package manager
  Ubuntu/Debian:
    $ sudo apt update
    $ sudo apt install python3.10 python3.10-venv
  
  macOS (Homebrew):
    $ brew install python@3.10

After installation, verify:
  $ python3.10 --version
  $ python3.10 -m pip install -r requirements/torch-cpu.txt

For more information, see: README.md#installation
"""
        return False, message

    # Check maximum version
    if (current.major, current.minor) > MAX_PYTHON_VERSION:
        message = f"""
╔════════════════════════════════════════════════════════════════════╗
║  PYTHON VERSION TOO NEW                                            ║
╚════════════════════════════════════════════════════════════════════╝

Current Python version: {current_version}
Supported: Python <= {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}

MINIMINIMOON has only been tested up to Python {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}.
While it may work with newer versions, we cannot guarantee compatibility.

RECOMMENDED ACTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Use a supported Python version (3.10-3.12):
   $ pyenv install 3.10.13
   $ pyenv local 3.10.13

2. If you must use Python {current_version}, run dependency check:
   $ python scripts/check_conflicts.py

3. Report compatibility issues at:
   https://github.com/kkkkknhh/SIN_CARRETA/issues

WARNING: Proceeding with untested Python version may cause runtime errors.
"""
        return False, message

    # Version is within range
    message = f"✓ Python {current_version} is supported (3.10-3.12 range)"
    return True, message


def enforce_python_version() -> None:
    """
    Enforce Python version requirement. Exits process if version is incompatible.
    Call this from module imports to fail fast.
    """
    is_valid, message = check_python_version(strict=True)

    if not is_valid:
        print(message, file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """Command-line interface for version checking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check Python version compatibility for MINIMINIMOON"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if version is incompatible",
    )

    args = parser.parse_args()

    is_valid, message = check_python_version(strict=args.strict)
    print(message)

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
