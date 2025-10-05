#!/usr/bin/env python3
"""
Validation script for the DAG validation implementation.
"""

import subprocess
import sys


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    result = subprocess.run([sys.executable, 'test_dag_validation.py'],
                            capture_output=True, text=True, check=True)

    if result.returncode == 0:
        print("✅ All tests passed")
        print(result.stdout)
        return True
    else:
        print("❌ Tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def run_lint():
    """Check code syntax and basic linting."""
    print("Checking syntax...")

    files = ['dag_validation.py', 'test_dag_validation.py', 'verify_reproducibility.py']

    for file in files:
        result = subprocess.run([sys.executable, '-m', 'py_compile', file],
                                capture_output=True, text=True, check=True)
        if result.returncode != 0:
            print(f"❌ Syntax error in {file}")
            print(result.stderr)
            return False

    print("✅ Syntax check passed")
    return True


def run_reproducibility_test():
    """Test reproducibility manually."""
    print("Testing reproducibility...")

    result = subprocess.run([sys.executable, 'verify_reproducibility.py'],
                            capture_output=True, text=True, check=True)

    if result.returncode == 0 and "PASSED" in result.stdout:
        print("✅ Reproducibility test passed")
        print(result.stdout)
        return True
    else:
        print("❌ Reproducibility test failed")
        print(result.stdout)
        print(result.stderr)
        return False


def main():
    """Run all validation checks."""
    print("=== DAG Validation System Validation ===")

    checks = [
        ("Syntax/Lint", run_lint),
        ("Unit Tests", run_tests),
        ("Reproducibility", run_reproducibility_test)
    ]

    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            success = check_func()
            results.append(success)
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append(False)

    print("\n=== Final Results ===")
    for i, (name, _) in enumerate(checks):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
