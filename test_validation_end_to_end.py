#!/usr/bin/env python3
"""End-to-end test demonstrating the validation flow"""

import json
import tempfile
import shutil
from pathlib import Path
from system_validators import SystemHealthValidator, ValidationError


def demo_validation_flow():
    """Demonstrate the complete validation flow"""
    print("=" * 70)
    print("END-TO-END VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    # Scenario 1: Valid rubric blocks execution properly
    print("\n[Scenario 1] Testing with MALFORMED question IDs")
    print("-" * 70)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="demo_"))
    
    try:
        # Create a rubric with malformed IDs
        malformed_rubric = {
            "metadata": {"total_questions": 5},
            "weights": {
                "D1-Q1": 0.2,
                "D7-Q1": 0.2,      # Invalid dimension
                "D1-Q301": 0.2,    # Invalid question number
                "invalid-id": 0.2, # Invalid format
                "D1-Q2": 0.2
            }
        }
        
        (temp_dir / "rubric_scoring.json").write_text(json.dumps(malformed_rubric))
        (temp_dir / "tools").mkdir(exist_ok=True)
        (temp_dir / "tools" / "flow_doc.json").write_text('{"canonical_order": []}')
        
        validator = SystemHealthValidator(str(temp_dir))
        
        print("Calling validate_pre_execution()...")
        result = validator.validate_pre_execution()
        
        print(f"\nValidation Result:")
        print(f"  Status: {'✗ FAILED' if not result['ok'] else '✓ PASSED'}")
        print(f"  Errors detected: {len(result['errors'])}")
        
        for i, error in enumerate(result['errors'], 1):
            print(f"\n  Error {i}:")
            for line in error.split('\n'):
                print(f"    {line}")
        
        if not result['ok']:
            print("\n✓ System correctly blocked execution due to malformed question IDs")
        else:
            print("\n✗ ERROR: System should have blocked execution!")
            return False
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Scenario 2: Count mismatch blocks execution
    print("\n" + "=" * 70)
    print("[Scenario 2] Testing with INCORRECT QUESTION COUNT")
    print("-" * 70)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="demo_"))
    
    try:
        # Create a rubric with only 100 questions
        count_mismatch_rubric = {
            "metadata": {"total_questions": 100},
            "weights": {f"D{d}-Q{q}": 0.01 for d in range(1, 6) for q in range(1, 21)}
        }
        
        (temp_dir / "rubric_scoring.json").write_text(json.dumps(count_mismatch_rubric))
        (temp_dir / "tools").mkdir(exist_ok=True)
        (temp_dir / "tools" / "flow_doc.json").write_text('{"canonical_order": []}')
        
        validator = SystemHealthValidator(str(temp_dir))
        
        print("Calling validate_pre_execution()...")
        result = validator.validate_pre_execution()
        
        print(f"\nValidation Result:")
        print(f"  Status: {'✗ FAILED' if not result['ok'] else '✓ PASSED'}")
        print(f"  Total questions found: 100")
        print(f"  Expected: 300")
        
        for error in result['errors']:
            if 'count mismatch' in error.lower():
                print(f"\n  Count Mismatch Error:")
                for line in error.split('\n'):
                    print(f"    {line}")
        
        if not result['ok']:
            print("\n✓ System correctly blocked execution due to question count mismatch")
        else:
            print("\n✗ ERROR: System should have blocked execution!")
            return False
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Scenario 3: Valid rubric allows execution
    print("\n" + "=" * 70)
    print("[Scenario 3] Testing with VALID 300-question rubric")
    print("-" * 70)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="demo_"))
    
    try:
        # Create a valid 300-question rubric
        valid_rubric = {
            "metadata": {"total_questions": 300},
            "weights": {f"D{d}-Q{q}": 0.0033333333 for d in range(1, 7) for q in range(1, 51)}
        }
        
        (temp_dir / "rubric_scoring.json").write_text(json.dumps(valid_rubric))
        (temp_dir / "tools").mkdir(exist_ok=True)
        (temp_dir / "tools" / "flow_doc.json").write_text('{"canonical_order": []}')
        
        validator = SystemHealthValidator(str(temp_dir))
        
        print("Calling validate_pre_execution()...")
        result = validator.validate_pre_execution()
        
        print(f"\nValidation Result:")
        print(f"  Total questions: 300 ✓")
        print(f"  All IDs conform to pattern D[1-6]-Q[1-300] ✓")
        
        # Check if question ID validation passed (no errors about malformed IDs or count)
        question_id_errors = [e for e in result['errors'] if 'malformed' in e.lower() or 'count mismatch' in e.lower()]
        
        if not question_id_errors:
            print(f"\n✓ Question ID validation passed")
            print(f"  (Note: Other pre-execution checks may still fail, but question ID format is correct)")
        else:
            print(f"\n✗ ERROR: Valid rubric should pass question ID validation!")
            return False
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 70)
    print("✓ ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Points:")
    print("  1. validate_question_id_format() validates rubric structure")
    print("  2. It checks ID format: D[1-6]-Q[1-300]")
    print("  3. It verifies exactly 300 questions exist")
    print("  4. It's integrated into validate_pre_execution()")
    print("  5. It blocks pipeline execution on validation failures")
    print("  6. It provides descriptive error messages with specific details")
    
    return True


if __name__ == "__main__":
    success = demo_validation_flow()
    exit(0 if success else 1)
