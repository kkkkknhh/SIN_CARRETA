#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone test for validate_question_id_format() method
"""

import json
import tempfile
import shutil
from pathlib import Path
from system_validators import SystemHealthValidator, ValidationError


def test_valid_rubric():
    """Test with valid 300-question rubric"""
    print("Test 1: Valid rubric (300 questions, D1-D6, Q1-Q50)")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rubric_"))
    
    try:
        weights = {f"D{d}-Q{q}": 0.0033333333 for d in range(1, 7) for q in range(1, 51)}
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        
        validator = SystemHealthValidator(str(temp_dir))
        validator.validate_question_id_format()
        print("  ✓ PASS: Valid rubric passed validation")
    except ValidationError as e:
        print(f"  ✗ FAIL: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


def test_malformed_ids():
    """Test detection of malformed question IDs"""
    print("\nTest 2: Malformed question IDs")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rubric_"))
    
    try:
        weights = {
            "D1-Q1": 0.0033,
            "D7-Q1": 0.0033,      # Invalid dimension (D7)
            "D1-Q301": 0.0033,    # Invalid question number (>300)
            "D1-Q0": 0.0033,      # Invalid question number (0)
            "invalid-id": 0.0033  # Completely invalid
        }
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        
        validator = SystemHealthValidator(str(temp_dir))
        validator.validate_question_id_format()
        print("  ✗ FAIL: Should have raised ValidationError for malformed IDs")
        return False
    except ValidationError as e:
        error_msg = str(e)
        if "malformed" in error_msg.lower() and ("D7-Q1" in error_msg or "4 malformed" in error_msg):
            print("  ✓ PASS: Correctly detected malformed IDs")
            print(f"    Error message: {error_msg[:150]}...")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: {error_msg}")
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_count_mismatch():
    """Test detection of question count mismatch"""
    print("\nTest 3: Question count mismatch (180 instead of 300)")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rubric_"))
    
    try:
        weights = {f"D{d}-Q{q}": 0.005 for d in range(1, 7) for q in range(1, 31)}
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        
        validator = SystemHealthValidator(str(temp_dir))
        validator.validate_question_id_format()
        print("  ✗ FAIL: Should have raised ValidationError for count mismatch")
        return False
    except ValidationError as e:
        error_msg = str(e)
        if "count mismatch" in error_msg.lower() and "180" in error_msg and "300" in error_msg:
            print("  ✓ PASS: Correctly detected count mismatch")
            print(f"    Error message: {error_msg[:150]}...")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: {error_msg}")
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_missing_rubric():
    """Test validation with missing rubric file"""
    print("\nTest 4: Missing rubric file")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rubric_"))
    
    try:
        validator = SystemHealthValidator(str(temp_dir))
        validator.validate_question_id_format()
        print("  ✗ FAIL: Should have raised ValidationError for missing rubric")
        return False
    except ValidationError as e:
        error_msg = str(e)
        if "missing" in error_msg.lower():
            print("  ✓ PASS: Correctly detected missing rubric")
            print(f"    Error message: {error_msg}")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: {error_msg}")
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_empty_weights():
    """Test validation with empty weights section"""
    print("\nTest 5: Empty weights section")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rubric_"))
    
    try:
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": {}}))
        
        validator = SystemHealthValidator(str(temp_dir))
        validator.validate_question_id_format()
        print("  ✗ FAIL: Should have raised ValidationError for empty weights")
        return False
    except ValidationError as e:
        error_msg = str(e)
        if "empty" in error_msg.lower() and "weights" in error_msg.lower():
            print("  ✓ PASS: Correctly detected empty weights")
            print(f"    Error message: {error_msg}")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: {error_msg}")
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pre_execution_integration():
    """Test that pre_execution validates question IDs"""
    print("\nTest 6: Integration with validate_pre_execution()")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rubric_"))
    
    try:
        weights = {"D1-Q1": 0.005, "invalid-id": 0.005}
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        (temp_dir / "tools").mkdir(exist_ok=True)
        (temp_dir / "tools" / "flow_doc.json").write_text('{"canonical_order": []}')
        
        validator = SystemHealthValidator(str(temp_dir))
        result = validator.validate_pre_execution()
        
        if not result["ok"] and any("malformed" in err.lower() for err in result["errors"]):
            print("  ✓ PASS: Pre-execution validation includes question ID check")
            print(f"    Errors: {result['errors']}")
            return True
        else:
            print("  ✗ FAIL: Pre-execution did not catch malformed IDs")
            print(f"    Result: {result}")
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_actual_rubric():
    """Test with the actual rubric_scoring.json in the repo"""
    print("\nTest 7: Actual rubric_scoring.json file")
    
    rubric_path = Path("rubric_scoring.json")
    if not rubric_path.exists():
        print("  ⊘ SKIP: rubric_scoring.json not found in current directory")
        return True
    
    try:
        validator = SystemHealthValidator(".")
        validator.validate_question_id_format()
        print("  ✓ PASS: Actual rubric_scoring.json passed validation")
        return True
    except ValidationError as e:
        print(f"  ✗ FAIL: Actual rubric has issues: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing validate_question_id_format() implementation")
    print("=" * 70)
    
    tests = [
        test_valid_rubric,
        test_malformed_ids,
        test_count_mismatch,
        test_missing_rubric,
        test_empty_weights,
        test_pre_execution_integration,
        test_actual_rubric
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("✓ All tests passed!")
        exit(0)
    else:
        print(f"✗ {total - passed} test(s) failed")
        exit(1)
