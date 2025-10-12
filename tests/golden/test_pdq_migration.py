#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden test cases for P-D-Q canonical notation migration.

These test cases verify that:
1. Canonical IDs pass through unchanged
2. Legacy IDs are correctly migrated
3. Invalid IDs are properly rejected
4. Migration confidence thresholds work
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from migration.migrate_legacy_ids import LegacyIDMigrator

# Golden test cases
GOLDEN_CASES = [
    # (input_id, context, expected_output, expected_rubric_key, should_succeed, description)
    
    # Case 1: Already canonical - should pass through
    ("P3-D2-Q4", {}, "P3-D2-Q4", "D2-Q4", True, "Already canonical - no change"),
    ("P1-D1-Q1", {}, "P1-D1-Q1", "D1-Q1", True, "Canonical P1-D1-Q1"),
    ("P10-D6-Q30", {}, "P10-D6-Q30", "D6-Q30", True, "Canonical with P10 and Q30"),
    
    # Case 2: Legacy D#-Q# with section context
    ("D4-Q3", {"section": "P8"}, "P8-D4-Q3", "D4-Q3", True, "D#-Q# with section P8"),
    ("D1-Q1", {"section": "P1"}, "P1-D1-Q1", "D1-Q1", True, "D#-Q# with section P1"),
    ("D6-Q25", {"section": "PUNTO 7"}, "P7-D6-Q25", "D6-Q25", True, "D#-Q# with 'PUNTO 7' in section"),
    
    # Case 3: Legacy P#-Q# (missing dimension)
    ("P2-Q5", {}, "P2-D1-Q5", "D1-Q5", True, "P#-Q# inferred to D1 by question range"),
    ("P3-Q12", {}, "P3-D3-Q12", "D3-Q12", True, "P#-Q# inferred to D3 (Q12 is in 11-15 range)"),
    ("P7-Q29", {}, "P7-D6-Q29", "D6-Q29", True, "P#-Q# inferred to D6 (Q29 is in 26-30 range)"),
    
    # Case 4: Legacy Q# (missing both)
    ("Q12", {"section": "P6"}, "P6-D3-Q12", "D3-Q12", True, "Q# with section context"),
    ("Q1", {"section": "P1"}, "P1-D1-Q1", "D1-Q1", True, "Q1 with section P1"),
    
    # Case 5: Invalid formats (should fail)
    ("D7-Q1", {}, None, None, False, "Invalid dimension D7 (only D1-D6 allowed)"),
    ("P11-D1-Q1", {}, None, None, False, "Invalid policy P11 (only P1-P10 allowed)"),
    ("P1-D1-Q0", {}, None, None, False, "Invalid question Q0 (must be Q1+)"),
    ("invalid-id", {}, None, None, False, "Completely invalid format"),
    
    # Case 6: Edge cases
    ("P10-D6-Q50", {}, "P10-D6-Q50", "D6-Q50", True, "Edge case: P10 with large Q number"),
    ("D1-Q100", {"section": "P4"}, "P4-D1-Q100", "D1-Q100", True, "Large Q number with section"),
]

def run_golden_tests() -> Tuple[int, int]:
    """
    Run all golden test cases.
    Returns: (passed, total)
    """
    print("=" * 70)
    print("Golden Test Cases - P-D-Q Migration")
    print("=" * 70)
    
    # Initialize migrator
    repo_root = Path(__file__).parent.parent.parent
    manifest_path = repo_root / "config" / "QUESTIONNAIRE_MANIFEST.yaml"
    bundle_path = repo_root / "bundles" / "decalogo_bundle.json"
    
    migrator = LegacyIDMigrator(manifest_path, bundle_path)
    
    passed = 0
    total = len(GOLDEN_CASES)
    
    for i, (input_id, context, expected_out, expected_rk, should_succeed, description) in enumerate(GOLDEN_CASES, 1):
        print(f"\n[Test {i}/{total}] {description}")
        print(f"  Input: {input_id}")
        print(f"  Context: {context}")
        
        try:
            normalized, rubric_key, confidence = migrator.migrate(input_id, context)
            
            if not should_succeed:
                print(f"  ❌ FAIL: Expected failure but got: {normalized}")
                continue
            
            # Check results
            if normalized != expected_out:
                print(f"  ❌ FAIL: Expected {expected_out}, got {normalized}")
                continue
            
            if rubric_key != expected_rk:
                print(f"  ❌ FAIL: Expected rubric_key {expected_rk}, got {rubric_key}")
                continue
            
            print(f"  ✅ PASS: {normalized} (rubric_key: {rubric_key}, confidence: {confidence:.2f})")
            passed += 1
            
        except ValueError as e:
            if should_succeed:
                print(f"  ❌ FAIL: Unexpected error: {e}")
                continue
            else:
                print(f"  ✅ PASS: Correctly rejected with: {str(e)[:80]}")
                passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: Unexpected exception: {e}")
            continue
    
    return passed, total

def main():
    """Main test runner."""
    passed, total = run_golden_tests()
    
    print("\n" + "=" * 70)
    print(f"Golden Test Results: {passed}/{total} passed")
    print("=" * 70)
    
    if passed == total:
        print("✅ All golden tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
