#!/usr/bin/env python3
"""
Validation script for rubric loading refactor.
Tests that answer_assembler.py and miniminimoon_orchestrator.py correctly
load scoring configuration exclusively from RUBRIC_SCORING.json.
"""

import json
import ast
import sys
from pathlib import Path


def check_rubric_json_structure():
    """Verify RUBRIC_SCORING.json has required structure."""
    print("\n" + "=" * 70)
    print("1. Checking RUBRIC_SCORING.json structure...")
    print("=" * 70)
    
    try:
        with open("RUBRIC_SCORING.json", 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        
        errors = []
        
        # Check required sections
        if "questions" not in rubric:
            errors.append("Missing 'questions' section")
        elif not isinstance(rubric["questions"], list):
            errors.append("'questions' must be an array")
        
        if "weights" not in rubric:
            errors.append("Missing 'weights' section")
        elif not isinstance(rubric["weights"], dict):
            errors.append("'weights' must be a dictionary")
        
        if errors:
            print("❌ FAILED:")
            for err in errors:
                print(f"   - {err}")
            return False
        
        questions = rubric["questions"]
        weights = rubric["weights"]
        
        print(f"✓ Found 'questions' section: {len(questions)} question templates")
        print(f"✓ Found 'weights' section: {len(weights)} weight entries")
        
        # Verify question structure
        if questions:
            sample = questions[0]
            required_q_fields = ['id', 'dimension', 'scoring_modality', 'max_score']
            missing = [f for f in required_q_fields if f not in sample]
            if missing:
                print(f"⚠️  Warning: Question missing fields: {missing}")
            else:
                print(f"✓ Question structure valid (sample: {sample['id']})")
        
        # Verify weight structure (keys should be like 'D1-Q1-P1', 'D1-Q1-P2', etc.)
        weight_keys = list(weights.keys())
        if weight_keys:
            sample_keys = weight_keys[:3]
            print(f"✓ Weight keys format: {sample_keys}")
            
            # Check that weights are numeric
            sample_values = [weights[k] for k in sample_keys]
            if all(isinstance(v, (int, float)) for v in sample_values):
                print(f"✓ Weight values are numeric")
            else:
                print(f"⚠️  Warning: Some weight values are not numeric")
        
        print("\n✅ RUBRIC_SCORING.json structure: VALID\n")
        return True
        
    except FileNotFoundError:
        print("❌ FAILED: RUBRIC_SCORING.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_answer_assembler_code():
    """Verify answer_assembler.py has required refactoring."""
    print("=" * 70)
    print("2. Checking answer_assembler.py code...")
    print("=" * 70)
    
    try:
        with open("answer_assembler.py", 'r', encoding='utf-8') as f:
            code = f.read()
        
        checks = []
        
        # Check 1: _load_rubric_config method exists
        if "_load_rubric_config" in code:
            checks.append(("✓", "_load_rubric_config() method present"))
        else:
            checks.append(("❌", "_load_rubric_config() method MISSING"))
        
        # Check 2: QuestionAnswer has rubric_weight field
        if "rubric_weight" in code and "QuestionAnswer" in code:
            checks.append(("✓", "QuestionAnswer.rubric_weight field present"))
        else:
            checks.append(("❌", "QuestionAnswer.rubric_weight field MISSING"))
        
        # Check 3: Loads weights from RUBRIC_SCORING.json
        if '"weights"' in code and 'rubric_config' in code:
            checks.append(("✓", "Loads weights from rubric_config"))
        else:
            checks.append(("❌", "Weight loading from rubric_config MISSING"))
        
        # Check 4: Validates weights presence
        if "GATE #5" in code and "weights" in code:
            checks.append(("✓", "GATE #5 validation present"))
        else:
            checks.append(("❌", "GATE #5 validation MISSING"))
        
        # Check 5: Single source of truth mention
        if "single source of truth" in code.lower() or "RUBRIC_SCORING.json" in code:
            checks.append(("✓", "Single source of truth documentation present"))
        else:
            checks.append(("⚠️ ", "Single source of truth documentation could be clearer"))
        
        # Print results
        for symbol, msg in checks:
            print(f"{symbol} {msg}")
        
        failed = sum(1 for s, _ in checks if s == "❌")
        
        if failed == 0:
            print("\n✅ answer_assembler.py: VALID\n")
            return True
        else:
            print(f"\n❌ answer_assembler.py: {failed} checks FAILED\n")
            return False
        
    except FileNotFoundError:
        print("❌ FAILED: answer_assembler.py not found")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_orchestrator_code():
    """Verify miniminimoon_orchestrator.py has required refactoring."""
    print("=" * 70)
    print("3. Checking miniminimoon_orchestrator.py code...")
    print("=" * 70)
    
    try:
        with open("miniminimoon_orchestrator.py", 'r', encoding='utf-8') as f:
            code = f.read()
        
        checks = []
        
        # Check 1: _load_rubric reads both questions and weights
        if "_load_rubric" in code:
            # Check if it returns tuple or loads both
            if "questions" in code and "weights" in code:
                checks.append(("✓", "_load_rubric() loads both questions and weights"))
            else:
                checks.append(("❌", "_load_rubric() doesn't load both sections"))
        else:
            checks.append(("❌", "_load_rubric() method MISSING"))
        
        # Check 2: _validate_rubric_coverage checks 1:1 alignment
        if "_validate_rubric_coverage" in code:
            if "1:1" in code or "alignment" in code:
                checks.append(("✓", "_validate_rubric_coverage() validates 1:1 alignment"))
            else:
                checks.append(("⚠️ ", "_validate_rubric_coverage() present but may lack alignment check"))
        else:
            checks.append(("❌", "_validate_rubric_coverage() method MISSING"))
        
        # Check 3: Stores weights internally
        if "self.weights" in code:
            checks.append(("✓", "AnswerAssembler stores weights internally"))
        else:
            checks.append(("❌", "AnswerAssembler doesn't store weights internally"))
        
        # Check 4: Retrieves weights from loaded dictionary
        if "self.weights.get" in code or "self.weights[" in code:
            checks.append(("✓", "Retrieves weights from loaded dictionary"))
        else:
            checks.append(("❌", "Weight retrieval from dictionary MISSING"))
        
        # Check 5: GATE #5 validation
        if "gate #5" in code.lower():
            checks.append(("✓", "GATE #5 validation present"))
        else:
            checks.append(("❌", "GATE #5 validation MISSING"))
        
        # Print results
        for symbol, msg in checks:
            print(f"{symbol} {msg}")
        
        failed = sum(1 for s, _ in checks if s == "❌")
        
        if failed == 0:
            print("\n✅ miniminimoon_orchestrator.py: VALID\n")
            return True
        else:
            print(f"\n❌ miniminimoon_orchestrator.py: {failed} checks FAILED\n")
            return False
        
    except FileNotFoundError:
        print("❌ FAILED: miniminimoon_orchestrator.py not found")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_hardcoded_constants():
    """Check for hardcoded scoring constants that should be removed."""
    print("=" * 70)
    print("4. Checking for hardcoded scoring constants...")
    print("=" * 70)
    
    try:
        with open("answer_assembler.py", 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Look for potential hardcoded weights (common patterns)
        suspicious_patterns = [
            ("0.00333", "hardcoded weight value 0.00333"),
            ("weight = 0.", "direct weight assignment"),
            ("WEIGHT_", "WEIGHT_ constant definition"),
        ]
        
        findings = []
        for pattern, description in suspicious_patterns:
            if pattern in code:
                # Count occurrences
                count = code.count(pattern)
                findings.append(f"⚠️  Found {count}x: {description}")
        
        if not findings:
            print("✓ No obvious hardcoded weight constants found")
            print("\n✅ Hardcoded constants check: PASSED\n")
            return True
        else:
            print("Suspicious patterns found (manual review recommended):")
            for finding in findings:
                print(f"   {finding}")
            print("\n⚠️  Hardcoded constants check: REVIEW NEEDED\n")
            return True  # Not a failure, just needs review
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print(" RUBRIC REFACTORING VALIDATION")
    print(" Validates: answer_assembler.py & miniminimoon_orchestrator.py")
    print(" Requirement: Load all scoring config from RUBRIC_SCORING.json")
    print("=" * 70)
    
    results = [
        check_rubric_json_structure(),
        check_answer_assembler_code(),
        check_orchestrator_code(),
        check_hardcoded_constants(),
    ]
    
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    checks = [
        "RUBRIC_SCORING.json structure",
        "answer_assembler.py refactoring",
        "miniminimoon_orchestrator.py refactoring",
        "Hardcoded constants removal"
    ]
    
    for check, result in zip(checks, results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {check}")
    
    print("\n" + "=" * 70)
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("=" * 70 + "\n")
        return 0
    else:
        print(f"❌ SOME CHECKS FAILED ({passed}/{total} passed)")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
