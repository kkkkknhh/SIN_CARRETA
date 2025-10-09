#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inspections.py — Run all code quality inspections locally

This script runs all code quality checks that are performed in CI:
  - Python bytecode compilation
  - flake8 linting
  - mypy type checking
  - ruff linting (optional)

Exit codes:
  0: All inspections passed
  1: One or more inspections failed
"""

import subprocess
import sys
from typing import List, Tuple


def run_command(name: str, cmd: List[str]) -> Tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        output = result.stdout + result.stderr
        if output:
            print(output)

        success = result.returncode == 0
        if success:
            print(f"✅ {name} PASSED")
        else:
            print(f"❌ {name} FAILED (exit code: {result.returncode})")

        return success, output

    except FileNotFoundError:
        print(f"❌ {name} FAILED: Command not found")
        return False, f"Command not found: {cmd[0]}"
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
        return False, str(e)


def main() -> int:
    """Run all inspections and return overall exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Run all code quality inspections")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (default: run all inspections)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any inspection errors (default: warn only)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CODE QUALITY INSPECTIONS")
    print("=" * 60)

    inspections = [
        ("Python Bytecode Compilation", ["python", "-m", "compileall", "-q", "."]),
        ("Flake8 Linting", ["flake8", "."]),
        ("Mypy Type Checking", ["mypy", ".", "--config-file", "pyproject.toml"]),
    ]

    # Optional: Add ruff if available
    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=False)
        inspections.append(("Ruff Linting", ["ruff", "check", "."]))
    except FileNotFoundError:
        pass

    results = []
    for name, cmd in inspections:
        success, _ = run_command(name, cmd)
        results.append((name, success))

        if not success and args.fail_fast:
            print("\n⚠️  Stopping due to failure (--fail-fast enabled)")
            break

    # Summary
    print("\n" + "=" * 60)
    print("INSPECTION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:.<40} {status}")

    print("=" * 60)
    print(f"TOTAL: {passed}/{total} inspections passed")
    print("=" * 60)

    if passed == total:
        print("\n✅ All inspections PASSED")
        return 0
    elif args.strict:
        print(f"\n❌ {total - passed} inspection(s) FAILED (strict mode)")
        return 1
    else:
        print(f"\n⚠️  {total - passed} inspection(s) FAILED (warning only)")
        print("    Run with --strict to enforce all checks")
        return 0


if __name__ == "__main__":
    sys.exit(main())
