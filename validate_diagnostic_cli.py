#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for diagnostic CLI functionality.
Tests CLI argument parsing and help output.
"""

import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Test that CLI help includes diagnostic command."""
    result = subprocess.run(
        [sys.executable, "miniminimoon_cli.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "CLI help should exit with 0"
    assert "diagnostic" in result.stdout, "Help should list diagnostic command"
    print("✓ CLI help includes diagnostic command")


def test_diagnostic_help():
    """Test that diagnostic command help works."""
    result = subprocess.run(
        [sys.executable, "miniminimoon_cli.py", "diagnostic", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Diagnostic help should exit with 0"
    assert "plan_path" in result.stdout, "Help should mention plan_path"
    assert "--repo" in result.stdout, "Help should mention --repo"
    assert "--rubric" in result.stdout, "Help should mention --rubric"
    print("✓ Diagnostic command help works correctly")


def test_file_structure():
    """Test that required files exist."""
    files = [
        "miniminimoon_cli.py",
        "diagnostic_runner.py",
        "test_diagnostic_runner.py"
    ]
    
    for file in files:
        path = Path(file)
        assert path.exists(), f"{file} should exist"
        print(f"✓ {file} exists")


def test_syntax():
    """Test that Python files have valid syntax."""
    files = [
        "miniminimoon_cli.py",
        "diagnostic_runner.py",
        "test_diagnostic_runner.py"
    ]
    
    for file in files:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"{file} should have valid syntax"
        print(f"✓ {file} has valid syntax")


def main():
    print("=" * 80)
    print("DIAGNOSTIC CLI VALIDATION")
    print("=" * 80)
    print()
    
    try:
        test_file_structure()
        print()
        test_syntax()
        print()
        test_cli_help()
        print()
        test_diagnostic_help()
        print()
        print("=" * 80)
        print("✅ ALL VALIDATION TESTS PASSED")
        print("=" * 80)
        return 0
    except AssertionError as e:
        print()
        print("=" * 80)
        print(f"❌ VALIDATION FAILED: {e}")
        print("=" * 80)
        return 1
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
