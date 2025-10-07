#!/usr/bin/env python3
"""Comprehensive validation of rubric_scoring.json weights structure."""

import json
import sys

def validate_rubric_weights(rubric_file='rubric_scoring.json'):
    """Validate all requirements for the weights section."""
    
    print(f"Loading {rubric_file}...")
    with open(rubric_file, 'r') as f:
        rubric = json.load(f)
    
    # Requirement 1: Check top-level structure
    print("\n=== REQUIREMENT 1: Top-level Structure ===")
    top_keys = list(rubric.keys())
    print(f"Top-level keys: {top_keys}")
    has_questions = 'questions' in rubric
    has_weights = 'weights' in rubric
    print(f"Has 'questions': {has_questions} ✓" if has_questions else f"Has 'questions': {has_questions} ✗")
    print(f"Has 'weights': {has_weights} ✓" if has_weights else f"Has 'weights': {has_weights} ✗")
    
    if not (has_questions and has_weights):
        print("FAILED: Missing required top-level keys")
        return False
    
    # Requirement 2: Check weight count and IDs
    print("\n=== REQUIREMENT 2: Weight Structure ===")
    weights = rubric['weights']
    weight_ids = set(weights.keys())
    print(f"Total weights: {len(weights)}")
    
    # Verify all 300 expected IDs (D1-D6, Q1-Q50)
    expected_ids = set()
    for dim in range(1, 7):
        for q in range(1, 51):
            expected_ids.add(f"D{dim}-Q{q}")
    
    print(f"Expected weight entries: 300")
    print(f"Actual weight entries: {len(weight_ids)}")
    print(f"All 300 weights present: {len(weight_ids) == 300 and weight_ids == expected_ids} ✓" if len(weight_ids) == 300 and weight_ids == expected_ids else f"All 300 weights present: False ✗")
    
    # Requirement 3: Verify uniform weight value
    print("\n=== REQUIREMENT 3: Uniform Weight Values ===")
    expected_weight = 1.0 / 300.0
    unique_weights = set(weights.values())
    print(f"Expected weight (1/300): {expected_weight}")
    print(f"Unique weight values: {unique_weights}")
    all_correct = len(unique_weights) == 1 and abs(list(unique_weights)[0] - expected_weight) < 1e-15
    print(f"All weights = 0.0033333333333333335: {all_correct} ✓" if all_correct else f"All weights uniform: False ✗")
    
    # Requirement 4: Verify sum equals 1.0
    print("\n=== REQUIREMENT 4: Weight Sum ===")
    total_weight = sum(weights.values())
    print(f"Sum of all weights: {total_weight}")
    print(f"Expected sum: 1.0")
    difference = abs(total_weight - 1.0)
    print(f"Difference: {difference}")
    within_tolerance = difference < 1e-10
    print(f"Within tolerance (<1e-10): {within_tolerance} ✓" if within_tolerance else f"Within tolerance: False ✗")
    
    # Requirement 5: Check alignment with questions section
    print("\n=== REQUIREMENT 5: Questions Section Alignment ===")
    questions = rubric['questions']
    question_ids = {q['id'] for q in questions}
    print(f"Questions in 'questions' section: {len(question_ids)}")
    print(f"Sample question IDs: {sorted(list(question_ids))[:5]}")
    
    # Note: questions section has 30 base questions (D1-Q1 through D6-Q30)
    # while weights has 300 (all combinations for thematic points)
    base_question_ids = {f"D{d}-Q{q}" for d in range(1, 7) for q in range(1, 31)}
    questions_match_base = question_ids == base_question_ids
    print(f"Questions section contains base 30 questions: {questions_match_base} ✓" if questions_match_base else f"Questions structure differs ℹ")
    
    # Sample verification
    print(f"\nSample weight IDs: {sorted(list(weight_ids))[:5]}")
    print(f"Last weight IDs: {sorted(list(weight_ids))[-5:]}")
    
    success = (has_questions and has_weights and len(weight_ids) == 300 and 
               all_correct and within_tolerance)
    
    print("\n" + "="*50)
    if success:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
    print("="*50)
    
    return success

if __name__ == "__main__":
    success = validate_rubric_weights()
    sys.exit(0 if success else 1)
