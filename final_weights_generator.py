#!/usr/bin/env python3
import json

# Generate correct 300 weights based on questionnaire_engine.py pattern
weights = {}
w = 1.0 / 300.0

# Base question distribution per dimension (from questionnaire_engine.py)
dq = {
    'D1': [1, 2, 3, 4, 5],
    'D2': [6, 7, 8, 9, 10],
    'D3': [11, 12, 13, 14, 15],
    'D4': [16, 17, 18, 19, 20],
    'D5': [21, 22, 23, 24, 25],
    'D6': [26, 27, 28, 29, 30]
}

# Generate all 300 question IDs: P{point}-D{dim}-Q{q_num}
for p in range(1, 11):  # P1-P10
    for dim in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        for q in dq[dim]:
            qid = f'P{p}-{dim}-Q{q}'
            weights[qid] = w

# Load and update RUBRIC_SCORING.json
with open('RUBRIC_SCORING.json', 'r') as f:
    rubric = json.load(f)

rubric['weights'] = weights
rubric['metadata']['description'] = 'Complete scoring system for 300-question PDM evaluation (30 base questions × 10 thematic points)'

with open('RUBRIC_SCORING.json', 'w') as f:
    json.dump(rubric, f, indent=2, ensure_ascii=False)

# Verification
print('Updated RUBRIC_SCORING.json')
print(f'Total entries: {len(weights)}')
print(f'Sum: {sum(weights.values()):.15f}')
print(f'Weight: 0.003333333333333333')
print(f'Sample: {list(weights.keys())[:3]} ... {list(sorted(weights.keys()))[-3:]}')
