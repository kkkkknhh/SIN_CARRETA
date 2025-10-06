#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rubric_check.py - GATE #5 Validation Tool

Validates strict 1:1 alignment between RUBRIC_SCORING.json weights
and DECALOGO_FULL.json questions.

Exit codes:
  0 - All checks passed
  1 - Validation failed
"""

import json
import sys
import pathlib
from typing import Set, Tuple


def load_json(path: pathlib.Path) -> dict:
    """Load and parse JSON file."""
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ ERROR: Failed to load {path}: {e}")
        sys.exit(1)


def get_rubric_weights(rubric: dict) -> Set[str]:
    """Extract weight keys from rubric."""
    if 'weights' not in rubric:
        print("✗ ERROR: 'weights' key missing from RUBRIC_SCORING.json")
        sys.exit(1)
    return set(rubric['weights'].keys())


def get_decalogo_questions(decalogo: dict) -> Set[str]:
    """Extract question IDs from decalogo."""
    if 'questions' not in decalogo:
        print("✗ ERROR: 'questions' key missing from DECALOGO_FULL.json")
        sys.exit(1)
    
    questions = set()
    for q in decalogo['questions']:
        unique_id = f"{q['id']}-{q['point_code']}"
        questions.add(unique_id)
    return questions


def validate_alignment(
    weights: Set[str],
    questions: Set[str]
) -> Tuple[bool, list]:
    """Validate 1:1 alignment between weights and questions."""
    errors = []
    
    missing_weights = questions - weights
    extra_weights = weights - questions
    
    if missing_weights:
        sample = sorted(missing_weights)[:10]
        errors.append(
            f"Missing weights for {len(missing_weights)} questions: {sample}"
            + (" ..." if len(missing_weights) > 10 else "")
        )
    
    if extra_weights:
        sample = sorted(extra_weights)[:10]
        errors.append(
            f"Extra weights for {len(extra_weights)} non-existent questions: {sample}"
            + (" ..." if len(extra_weights) > 10 else "")
        )
    
    return len(errors) == 0, errors


def main():
    """Main validation routine."""
    print("=" * 70)
    print("GATE #5: Rubric Weight Validation")
    print("=" * 70)
    
    # Load files
    rubric_path = pathlib.Path("rubric_scoring.json")
    decalogo_path = pathlib.Path("DECALOGO_FULL.json")
    
    if not rubric_path.exists():
        print(f"✗ ERROR: {rubric_path} not found")
        sys.exit(1)
    
    if not decalogo_path.exists():
        print(f"✗ ERROR: {decalogo_path} not found")
        sys.exit(1)
    
    print(f"\n1. Loading {rubric_path}...")
    rubric = load_json(rubric_path)
    print(f"   ✓ Loaded successfully")
    
    print(f"\n2. Loading {decalogo_path}...")
    decalogo = load_json(decalogo_path)
    print(f"   ✓ Loaded successfully")
    
    print("\n3. Extracting weights and questions...")
    weights = get_rubric_weights(rubric)
    questions = get_decalogo_questions(decalogo)
    print(f"   ✓ Found {len(weights)} weights")
    print(f"   ✓ Found {len(questions)} questions")
    
    print("\n4. Validating 1:1 alignment...")
    ok, errors = validate_alignment(weights, questions)
    
    if ok:
        print(f"   ✓ Perfect 1:1 alignment: {len(questions)}/300 questions have weights")
        print("\n" + "=" * 70)
        print("✅ GATE #5 VALIDATION PASSED")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"   ✗ Alignment validation failed:")
        for error in errors:
            print(f"      - {error}")
        print("\n" + "=" * 70)
        print("❌ GATE #5 VALIDATION FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
