#!/usr/bin/env python3
"""
Validation script for batch load and stress test modules.
Verifies syntax and basic functionality without running actual tests.
"""

import ast
import sys
from pathlib import Path


def validate_python_file(file_path: str) -> bool:
    """Validate Python file syntax and imports"""
    try:
        content = Path(file_path).read_text()
        ast.parse(content)
        print(f"✅ {file_path}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax error - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Validation error - {e}")
        return False


def check_required_imports(file_path: str, required_imports: list) -> bool:
    """Check if file contains required imports"""
    try:
        content = Path(file_path).read_text()
        tree = ast.parse(content)
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        missing = [imp for imp in required_imports if imp not in imports]
        if missing:
            print(f"⚠️  {file_path}: Missing imports: {missing}")
            return False
        
        print(f"✅ {file_path}: All required imports present")
        return True
    except Exception as e:
        print(f"❌ {file_path}: Import check error - {e}")
        return False


def validate_test_structure(file_path: str, expected_functions: list) -> bool:
    """Validate test file structure"""
    try:
        content = Path(file_path).read_text()
        tree = ast.parse(content)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        
        missing = [func for func in expected_functions if func not in functions]
        if missing:
            print(f"⚠️  {file_path}: Missing functions: {missing}")
            return False
        
        print(f"✅ {file_path}: All expected functions present")
        return True
    except Exception as e:
        print(f"❌ {file_path}: Structure validation error - {e}")
        return False


def main():
    """Main validation function"""
    print("=" * 60)
    print("Batch Load Test & Stress Test Validation")
    print("=" * 60)
    
    all_valid = True
    
    # Validate test_batch_load.py
    print("\n📝 Validating test_batch_load.py...")
    all_valid &= validate_python_file("test_batch_load.py")
    all_valid &= check_required_imports("test_batch_load.py", ["asyncio", "json", "pytest"])
    all_valid &= validate_test_structure("test_batch_load.py", [
        "evaluate_document_concurrent",
        "test_batch_load_10_concurrent"
    ])
    
    # Validate test_stress_test.py
    print("\n📝 Validating test_stress_test.py...")
    all_valid &= validate_python_file("test_stress_test.py")
    all_valid &= check_required_imports("test_stress_test.py", ["asyncio", "json", "tracemalloc", "psutil", "pytest"])
    all_valid &= validate_test_structure("test_stress_test.py", [
        "stress_test_document",
        "test_stress_test_50_concurrent"
    ])
    
    # Check for output files structure
    print("\n📝 Checking output file generation...")
    
    # Check that both tests write expected JSON files
    batch_content = Path("test_batch_load.py").read_text()
    if "processing_times.json" in batch_content and "throughput_report.json" in batch_content:
        print("✅ test_batch_load.py: Generates processing_times.json and throughput_report.json")
    else:
        print("❌ test_batch_load.py: Missing expected output files")
        all_valid = False
    
    stress_content = Path("test_stress_test.py").read_text()
    if "memory_profile.json" in stress_content:
        print("✅ test_stress_test.py: Generates memory_profile.json")
    else:
        print("❌ test_stress_test.py: Missing expected output files")
        all_valid = False
    
    # Validate CI configuration
    print("\n📝 Validating CI configuration...")
    ci_content = Path(".github/workflows/ci.yml").read_text()
    
    required_jobs = ["batch_load_test", "stress_test"]
    missing_jobs = [job for job in required_jobs if job not in ci_content]
    
    if missing_jobs:
        print(f"❌ CI configuration: Missing jobs: {missing_jobs}")
        all_valid = False
    else:
        print("✅ CI configuration: All required jobs present")
    
    # Check for artifact archival
    if "batch-metrics" in ci_content and "stress-test-metrics" in ci_content:
        print("✅ CI configuration: Artifact archival configured")
    else:
        print("❌ CI configuration: Missing artifact archival")
        all_valid = False
    
    # Check for Redis service
    if "redis:" in ci_content and "6379:6379" in ci_content:
        print("✅ CI configuration: Redis service configured")
    else:
        print("⚠️  CI configuration: Redis service not found")
    
    # Final result
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All validations passed!")
        return 0
    else:
        print("❌ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
