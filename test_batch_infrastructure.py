"""
Integration test for batch testing infrastructure.
Validates that all components work together correctly.
"""

import json
import subprocess
import sys
from pathlib import Path


def test_batch_load_test_syntax():
    """Verify batch load test compiles correctly"""
    result = subprocess.run(
        ["python3", "-m", "py_compile", "test_batch_load.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Syntax error in test_batch_load.py: {result.stderr}"


def test_stress_test_syntax():
    """Verify stress test compiles correctly"""
    result = subprocess.run(
        ["python3", "-m", "py_compile", "test_stress_test.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Syntax error in test_stress_test.py: {result.stderr}"


def test_ci_configuration():
    """Verify CI configuration includes batch testing jobs"""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists(), "CI configuration file not found"
    
    ci_content = ci_path.read_text()
    
    # Check for batch_load_test job
    assert "batch_load_test:" in ci_content, "batch_load_test job not found in CI"
    
    # Check for stress_test job
    assert "stress_test:" in ci_content, "stress_test job not found in CI"
    
    # Check for Redis service
    assert "redis:" in ci_content, "Redis service not configured in CI"
    assert "6379:6379" in ci_content, "Redis port not exposed in CI"
    
    # Check for artifact archival
    assert "batch-metrics" in ci_content, "batch-metrics artifact not configured"
    assert "stress-test-metrics" in ci_content, "stress-test-metrics artifact not configured"
    
    # Check for dependency on performance job
    assert "needs: performance" in ci_content, "Jobs don't depend on performance job"


def test_requirements_updated():
    """Verify requirements.txt includes pytest-asyncio"""
    req_path = Path("requirements.txt")
    assert req_path.exists(), "requirements.txt not found"
    
    req_content = req_path.read_text()
    assert "pytest-asyncio" in req_content, "pytest-asyncio not in requirements.txt"


def test_validation_script():
    """Verify validation script works correctly"""
    result = subprocess.run(
        ["python3", "validate_batch_tests.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Validation script failed: {result.stdout}\n{result.stderr}"
    assert "All validations passed" in result.stdout, "Not all validations passed"


def test_documentation_exists():
    """Verify documentation was created"""
    doc_path = Path("BATCH_TESTING.md")
    assert doc_path.exists(), "BATCH_TESTING.md documentation not found"
    
    doc_content = doc_path.read_text()
    assert "Batch Load Test" in doc_content, "Batch load test not documented"
    assert "Stress Test" in doc_content, "Stress test not documented"
    assert "processing_times.json" in doc_content, "Output files not documented"
    assert "memory_profile.json" in doc_content, "Memory profile not documented"
    assert "throughput_report.json" in doc_content, "Throughput report not documented"


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Batch Testing Infrastructure Integration Test")
    print("=" * 60)
    
    tests = [
        ("Batch load test syntax", test_batch_load_test_syntax),
        ("Stress test syntax", test_stress_test_syntax),
        ("CI configuration", test_ci_configuration),
        ("Requirements updated", test_requirements_updated),
        ("Validation script", test_validation_script),
        ("Documentation exists", test_documentation_exists),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n📝 Testing: {name}...")
            test_func()
            print(f"✅ {name} - PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name} - FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All integration tests passed!")
        return 0
    else:
        print(f"❌ {failed} integration test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
