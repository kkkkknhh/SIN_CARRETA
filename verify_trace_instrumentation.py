#!/usr/bin/env python3.10
"""Verify trace instrumentation is syntactically correct."""

import ast
import sys
from pathlib import Path


def verify_syntax(file_path):
    """Verify Python syntax."""
    try:
        code = Path(file_path).read_text()
        ast.parse(code)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Verify orchestrator syntax."""
    print("Verifying miniminimoon_orchestrator.py syntax...")

    success, message = verify_syntax("miniminimoon_orchestrator.py")

    if success:
        print(f"✓ {message}")
        return 0
    else:
        print(f"✗ {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
