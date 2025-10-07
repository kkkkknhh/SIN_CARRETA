#!/usr/bin/env python3
"""
Regenerate RUBRIC_SCORING.json weights section with correct 300-question pattern.

Architecture (from questionnaire_engine.py):
- 30 base questions (D1-Q1 to D6-Q30) defined in questionnaire_engine.py
- Each base question is parametrized for 10 thematic points (P1-P10)
- Question ID format: P{point}-D{dimension}-Q{question} (e.g., P1-D1-Q1, P2-D1-Q1, etc.)
- Total: 30 base questions × 10 thematic points = 300 unique question evaluations
- Weight per question: 1/300 = 0.003333333333333333
"""

import json
import re

def generate_all_300_weights():
    """
    Generate weights for all 300 questions using the correct ID pattern.
    
    Pattern from questionnaire_engine.py line 124:
      question_id: str  # P1-D1-Q1, P2-D1-Q1, etc.
    
    And line 1607:
      question_id = f"{point.id}-{base_q.id}"
    
    Where:
    - point.id = P1, P2, ..., P10 (10 thematic points)
    - base_q.id = D1-Q1, D1-Q2, ..., D6-Q30 (30 base questions)
    
    Base question mapping per dimension:
    - D1: Q1, Q2, Q3, Q4, Q5
    - D2: Q6, Q7, Q8, Q9, Q10
    - D3: Q11, Q12, Q13, Q14, Q15
    - D4: Q16, Q17, Q18, Q19, Q20
    - D5: Q21, Q22, Q23, Q24, Q25
    - D6: Q26, Q27, Q28, Q29, Q30
    """
    
    weights = {}
    weight_per_question = 1.0 / 300.0
    
    # Define base questions per dimension (from questionnaire_engine.py)
    dimension_questions = {
        'D1': [1, 2, 3, 4, 5],
        'D2': [6, 7, 8, 9, 10],
        'D3': [11, 12, 13, 14, 15],
        'D4': [16, 17, 18, 19, 20],
        'D5': [21, 22, 23, 24, 25],
        'D6': [26, 27, 28, 29, 30]
    }
    
    # Generate all 300 question IDs
    for point in range(1, 11):  # P1-P10
        point_id = f"P{point}"
        for dimension in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
            for q_num in dimension_questions[dimension]:
                question_id = f"{point_id}-{dimension}-Q{q_num}"
                weights[question_id] = weight_per_question
    
    return weights

def verify_weights(weights):
    """Verify the generated weights meet all requirements"""
    
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print()
    
    # Check 1: Exactly 300 entries
    num_entries = len(weights)
    print(f"1. Number of entries: {num_entries}")
    if num_entries == 300:
        print("   ✓ PASS: Exactly 300 entries")
    else:
        print(f"   ✗ FAIL: Expected 300 entries, got {num_entries}")
        return False
    print()
    
    # Check 2: All weights sum to 1.0
    total_weight = sum(weights.values())
    print(f"2. Sum of all weights: {total_weight:.15f}")
    if abs(total_weight - 1.0) < 1e-10:
        print(f"   ✓ PASS: Weights sum to 1.0 (within tolerance {1e-10})")
    else:
        print(f"   ✗ FAIL: Expected sum of 1.0, got {total_weight}")
        print(f"   Difference: {abs(total_weight - 1.0):.15e}")
        return False
    print()
    
    # Check 3: Each weight is exactly 1/300
    expected_weight = 1.0 / 300.0
    print(f"3. Expected weight per question: {expected_weight:.15f}")
    all_correct = True
    incorrect_weights = []
    for question_id, weight in weights.items():
        if abs(weight - expected_weight) > 1e-15:
            incorrect_weights.append((question_id, weight))
            if len(incorrect_weights) <= 3:  # Show first few
                print(f"   ✗ {question_id}: {weight:.15f} (expected {expected_weight:.15f})")
    
    if not incorrect_weights:
        print(f"   ✓ PASS: All weights equal {expected_weight:.15f}")
    else:
        print(f"   ✗ FAIL: {len(incorrect_weights)} weights are incorrect")
        return False
    print()
    
    # Check 4: All keys follow P{point}-D{dimension}-Q{question} pattern
    pattern = re.compile(r'^P([1-9]|10)-D[1-6]-Q([1-9]|[12][0-9]|30)$')
    print("4. Verify ID pattern P{1-10}-D{1-6}-Q{1-30}:")
    invalid_keys = [k for k in weights.keys() if not pattern.match(k)]
    if not invalid_keys:
        print("   ✓ PASS: All keys follow P{point}-D{dimension}-Q{question} pattern")
    else:
        print(f"   ✗ FAIL: Found {len(invalid_keys)} invalid keys")
        print(f"   First few invalid: {invalid_keys[:5]}")
        return False
    print()
    
    # Check 5: Each thematic point has exactly 30 questions
    print("5. Questions per thematic point:")
    all_correct = True
    for point in range(1, 11):
        point_id = f"P{point}"
        point_questions = [k for k in weights.keys() if k.startswith(f"{point_id}-")]
        count = len(point_questions)
        status = "✓" if count == 30 else "✗"
        print(f"   {status} {point_id}: {count} questions")
        if count != 30:
            all_correct = False
    if not all_correct:
        return False
    print()
    
    # Check 6: Each dimension appears 10 times (once per thematic point)
    print("6. Questions per dimension (across all thematic points):")
    all_correct = True
    for dim in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        dim_questions = [k for k in weights.keys() if f"-{dim}-" in k]
        # Each dimension has 5 questions, each appears 10 times
        expected_count = 5 * 10
        count = len(dim_questions)
        status = "✓" if count == expected_count else "✗"
        print(f"   {status} {dim}: {count} questions (5 base × 10 points)")
        if count != expected_count:
            all_correct = False
    if not all_correct:
        return False
    print()
    
    # Check 7: Verify question number ranges per dimension
    print("7. Question number ranges per dimension:")
    dimension_ranges = {
        'D1': list(range(1, 6)),
        'D2': list(range(6, 11)),
        'D3': list(range(11, 16)),
        'D4': list(range(16, 21)),
        'D5': list(range(21, 26)),
        'D6': list(range(26, 31))
    }
    
    all_correct = True
    for dim, expected_qs in dimension_ranges.items():
        # Extract unique question numbers for this dimension
        dim_keys = [k for k in weights.keys() if f"-{dim}-Q" in k]
        q_numbers = sorted(set([int(k.split('-Q')[1]) for k in dim_keys]))
        
        if q_numbers == expected_qs:
            print(f"   ✓ {dim}: Q{expected_qs[0]}-Q{expected_qs[-1]} (appears 10 times each)")
        else:
            print(f"   ✗ {dim}: Expected {expected_qs}, got {q_numbers}")
            all_correct = False
    
    if not all_correct:
        return False
    print()
    
    print("=" * 80)
    print("ALL VERIFICATION CHECKS PASSED ✓")
    print("=" * 80)
    return True

def update_rubric_scoring_json(weights):
    """Update RUBRIC_SCORING.json with new weights"""
    
    # Read current RUBRIC_SCORING.json
    with open('RUBRIC_SCORING.json', 'r', encoding='utf-8') as f:
        rubric = json.load(f)
    
    # Update weights section
    rubric['weights'] = weights
    
    # Update metadata
    rubric['metadata']['total_questions'] = 300
    rubric['metadata']['description'] = "Complete scoring system for 300-question PDM evaluation (30 base questions × 10 thematic points)"
    
    # Write back with sorted keys for readability
    with open('RUBRIC_SCORING.json', 'w', encoding='utf-8') as f:
        json.dump(rubric, f, indent=2, ensure_ascii=False)
    
    print()
    print("✓ RUBRIC_SCORING.json updated successfully")
    print(f"✓ Weights section now contains {len(weights)} entries")

def main():
    print("=" * 80)
    print("REGENERATING RUBRIC_SCORING.json WEIGHTS")
    print("=" * 80)
    print()
    print("Architecture:")
    print("  • 30 base questions (D1-Q1 to D6-Q30)")
    print("  • 10 thematic points (P1-P10)")
    print("  • Total evaluations: 30 × 10 = 300")
    print("  • Weight per question: 1/300 = 0.003333333333333333")
    print()
    print("ID Pattern (from questionnaire_engine.py):")
    print("  • Format: P{point}-D{dimension}-Q{question}")
    print("  • Example: P1-D1-Q1, P1-D1-Q2, ..., P10-D6-Q30")
    print("  • Points: P1-P10")
    print("  • Dimensions: D1-D6")
    print("  • Questions: Q1-Q30 (distributed across dimensions)")
    print()
    
    # Generate weights
    print("Generating weights...")
    weights = generate_all_300_weights()
    print(f"✓ Generated {len(weights)} weight entries")
    print()
    
    # Verify weights
    if not verify_weights(weights):
        print()
        print("✗ VERIFICATION FAILED - aborting")
        return False
    
    # Show sample entries
    print()
    print("Sample weight entries:")
    sample_keys = [
        'P1-D1-Q1',   # First question of first point
        'P1-D1-Q5',   # Last question of D1 in first point
        'P1-D2-Q6',   # First question of D2 in first point
        'P5-D3-Q15',  # Middle point, middle dimension
        'P10-D6-Q26', # Last point, last dimension first question
        'P10-D6-Q30'  # Last question overall
    ]
    for key in sample_keys:
        if key in weights:
            print(f"  {key}: {weights[key]:.15f}")
    print()
    
    # Update RUBRIC_SCORING.json
    print("Updating RUBRIC_SCORING.json...")
    update_rubric_scoring_json(weights)
    
    print()
    print("=" * 80)
    print("REGENERATION COMPLETE ✓")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
