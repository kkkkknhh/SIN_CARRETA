#!/usr/bin/env python3
import json
import re

with open('RUBRIC_SCORING.json', 'r') as f:
    rubric = json.load(f)

weights = rubric['weights']

print('='*80)
print('RUBRIC_SCORING.json WEIGHTS VERIFICATION')
print('='*80)
print()

# Check 1: Number of entries
print(f'1. Number of entries: {len(weights)}')
print(f'   Expected: 300')
print(f'   Status: {"✓ PASS" if len(weights) == 300 else "✗ FAIL"}')
print()

# Check 2: Sum of weights
total = sum(weights.values())
print(f'2. Sum of weights: {total:.15f}')
print(f'   Expected: 1.0')
print(f'   Difference: {abs(total - 1.0):.2e}')
print(f'   Status: {"✓ PASS" if abs(total - 1.0) < 1e-10 else "✗ FAIL"}')
print()

# Check 3: Each weight value
expected = 1.0 / 300.0
incorrect = [k for k, v in weights.items() if abs(v - expected) > 1e-15]
print(f'3. Weight values:')
print(f'   Expected: {expected:.15f}')
print(f'   Incorrect: {len(incorrect)}')
print(f'   Status: {"✓ PASS" if len(incorrect) == 0 else "✗ FAIL"}')
print()

# Check 4: ID pattern
pattern = re.compile(r'^P([1-9]|10)-D[1-6]-Q([1-9]|[12][0-9]|30)$')
invalid = [k for k in weights.keys() if not pattern.match(k)]
print(f'4. ID pattern P{{1-10}}-D{{1-6}}-Q{{1-30}}:')
print(f'   Invalid: {len(invalid)}')
if invalid[:5]:
    print(f'   Examples: {invalid[:5]}')
print(f'   Status: {"✓ PASS" if len(invalid) == 0 else "✗ FAIL"}')
print()

# Check 5: Count per point
print(f'5. Questions per thematic point:')
for p in range(1, 11):
    count = len([k for k in weights.keys() if k.startswith(f'P{p}-')])
    status = "✓" if count == 30 else "✗"
    print(f'   {status} P{p}: {count}')
print()

# Check 6: Count per dimension
print(f'6. Questions per dimension (across all points):')
for d in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
    count = len([k for k in weights.keys() if f'-{d}-' in k])
    expected_count = 50  # 5 base questions × 10 points
    status = "✓" if count == expected_count else "✗"
    print(f'   {status} {d}: {count} (expected {expected_count})')
print()

print('='*80)
print('Sample entries:')
samples = ['P1-D1-Q1', 'P1-D6-Q30', 'P5-D3-Q15', 'P10-D6-Q30']
for s in samples:
    if s in weights:
        print(f'  {s}: {weights[s]:.15f}')
print('='*80)
