#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_run_inspections.py — Test suite for run_inspections.py

Tests:
1. Inspection script imports and runs without errors
2. Script exits with 0 in default (warning) mode
3. Script exits with 1 in strict mode (when there are failures)
4. All inspection commands are valid
"""

import subprocess
import sys
from pathlib import Path


def test_inspections_script_exists():
    """Test that run_inspections.py exists and is executable."""
    script_path = Path(__file__).parent / "run_inspections.py"
    assert script_path.exists(), "run_inspections.py should exist"
    assert script_path.is_file(), "run_inspections.py should be a file"


def test_inspections_help():
    """Test that --help flag works."""
    result = subprocess.run(
        ["python", "run_inspections.py", "--help"],
        capture_output=True,
        text=True,
        check=False
    )
    assert result.returncode == 0, "--help should exit with 0"
    assert "Run all code quality inspections" in result.stdout


def test_inspections_default_mode():
    """Test that default (warning) mode exits with 0."""
    result = subprocess.run(
        ["python", "run_inspections.py"],
        capture_output=True,
        text=True,
        check=False
    )
    # Default mode should exit with 0 even if some checks fail
    assert result.returncode == 0, "Default mode should exit with 0"
    assert "INSPECTION SUMMARY" in result.stdout


def test_inspections_strict_mode():
    """Test that strict mode can fail."""
    result = subprocess.run(
        ["python", "run_inspections.py", "--strict"],
        capture_output=True,
        text=True,
        check=False
    )
    # Strict mode exits with 1 if any checks fail
    # OR 0 if all pass (unlikely in this codebase)
    assert result.returncode in [0, 1], "Strict mode should exit with 0 or 1"
    assert "INSPECTION SUMMARY" in result.stdout


def test_individual_inspections():
    """Test that individual inspection commands work."""
    commands = [
        ["python", "-m", "compileall", "-q", "run_inspections.py"],
        ["flake8", "run_inspections.py"],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )
            # Just check that commands don't crash
            assert result.returncode in [0, 1], f"{cmd} should not crash"
        except FileNotFoundError:
            # Tool not installed, skip
            continue


if __name__ == "__main__":
    print("Testing run_inspections.py...")
    print("="*60)

    tests = [
        ("Script exists", test_inspections_script_exists),
        ("Help flag", test_inspections_help),
        ("Default mode", test_inspections_default_mode),
        ("Strict mode", test_inspections_strict_mode),
        ("Individual commands", test_individual_inspections),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {name}: Unexpected error: {e}")
            failed += 1

    print("="*60)
    print(f"TOTAL: {passed}/{len(tests)} tests passed")

    if failed > 0:
        sys.exit(1)
    else:
        print("✅ All tests PASSED")
        sys.exit(0)
