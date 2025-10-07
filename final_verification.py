#!/usr/bin/env python3
"""
Final verification of RUBRIC_SCORING.json weights section
"""
import json
import re

print("=" * 80)
print("FINAL VERIFICATION: RUBRIC_SCORING.json WEIGHTS")
print("=" * 80)
print()

# Load RUBRIC_SCORING.json
with open('RUBRIC_SCORING.json', 'r', encoding='utf-8') as f:
    rubric = json.load(f)

weights = rubric['weights']

# Verification 1: Exactly 300 entries
num_entries = len(weights)
print(f"✓ CHECK 1: Number of entries")
print(f"  Result: {num_entries}")
print(f"  Expected: 300")
print(f"  Status: {'PASS' if num_entries == 300 else 'FAIL'}")
print()

# Verification 2: Sum to 1.0
total = sum(weights.values())
print(f"✓ CHECK 2: Sum of all weights")
print(f"  Result: {total:.15f}")
print(f"  Expected: 1.000000000000000")
print(f"  Difference: {abs(total - 1.0):.2e}")
print(f"  Status: {'PASS' if abs(total - 1.0) < 1e-10 else 'FAIL'}")
print()

# Verification 3: Each weight is exactly 1/300
expected_weight = 1.0 / 300.0
incorrect = [k for k, v in weights.items() if abs(v - expected_weight) > 1e-15]
print(f"✓ CHECK 3: Each weight equals 0.003333333333333333")
print(f"  Expected value: {expected_weight:.15f}")
print(f"  Incorrect weights: {len(incorrect)}")
print(f"  Status: {'PASS' if len(incorrect) == 0 else 'FAIL'}")
print()

# Verification 4: ID pattern P{point}-D{dimension}-Q{question}
pattern = re.compile(r'^P([1-9]|10)-D[1-6]-Q([1-9]|[12][0-9]|30)$')
invalid_ids = [k for k in weights.keys() if not pattern.match(k)]
print(f"✓ CHECK 4: ID pattern P{{point}}-D{{dimension}}-Q{{question}}")
print(f"  Pattern: P{{1-10}}-D{{1-6}}-Q{{1-30}}")
print(f"  Invalid IDs: {len(invalid_ids)}")
if invalid_ids:
    print(f"  Examples: {invalid_ids[:5]}")
print(f"  Status: {'PASS' if len(invalid_ids) == 0 else 'FAIL'}")
print()

# Verification 5: Each thematic point has exactly 30 questions
print(f"✓ CHECK 5: Questions per thematic point (should be 30 each)")
all_points_correct = True
for point in range(1, 11):
    point_questions = [k for k in weights.keys() if k.startswith(f'P{point}-')]
    count = len(point_questions)
    if count != 30:
        print(f"  P{point}: {count} ✗")
        all_points_correct = False
print(f"  Status: {'PASS - all points have 30 questions' if all_points_correct else 'FAIL'}")
print()

# Verification 6: Each dimension appears 50 times (5 base q × 10 points)
print(f"✓ CHECK 6: Questions per dimension (should be 50 each)")
all_dims_correct = True
for dim in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
    dim_questions = [k for k in weights.keys() if f'-{dim}-' in k]
    count = len(dim_questions)
    if count != 50:
        print(f"  {dim}: {count} (expected 50) ✗")
        all_dims_correct = False
print(f"  Status: {'PASS - all dimensions have 50 questions' if all_dims_correct else 'FAIL'}")
print()

# Verification 7: Question number ranges per dimension
print(f"✓ CHECK 7: Question number ranges per dimension")
dimension_ranges = {
    'D1': list(range(1, 6)),
    'D2': list(range(6, 11)),
    'D3': list(range(11, 16)),
    'D4': list(range(16, 21)),
    'D5': list(range(21, 26)),
    'D6': list(range(26, 31))
}
all_ranges_correct = True
for dim, expected_qs in dimension_ranges.items():
    dim_keys = [k for k in weights.keys() if f'-{dim}-Q' in k]
    q_numbers = sorted(set([int(k.split('-Q')[1]) for k in dim_keys]))
    if q_numbers != expected_qs:
        print(f"  {dim}: Expected Q{expected_qs[0]}-Q{expected_qs[-1]}, got {q_numbers} ✗")
        all_ranges_correct = False
print(f"  Status: {'PASS - all ranges correct' if all_ranges_correct else 'FAIL'}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total entries: {num_entries}")
print(f"Sum of weights: {total:.15f}")
print(f"Weight per question: {expected_weight:.15f}")
print()
print("Sample entries:")
samples = ['P1-D1-Q1', 'P1-D6-Q30', 'P5-D3-Q15', 'P10-D1-Q1', 'P10-D6-Q30']
for s in samples:
    if s in weights:
        print(f"  {s}: {weights[s]:.15f}")
print()

# Overall result
all_pass = (
    num_entries == 300 and
    abs(total - 1.0) < 1e-10 and
    len(incorrect) == 0 and
    len(invalid_ids) == 0 and
    all_points_correct and
    all_dims_correct and
    all_ranges_correct
)

print("=" * 80)
if all_pass:
    print("✓ ALL VERIFICATION CHECKS PASSED")
else:
    print("✗ SOME VERIFICATION CHECKS FAILED")
print("=" * 80)
