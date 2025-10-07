#!/usr/bin/env python3
"""Generate clean weights section for RUBRIC_SCORING.json"""

import json

# Read the current file
with open('RUBRIC_SCORING.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate all 180 question identifiers (D{1-6}-Q{1-30})
weights = {}
weight_value = 1/300

for dimension in range(1, 7):  # D1 through D6
    for question in range(1, 31):  # Q1 through Q30
        question_id = f"D{dimension}-Q{question}"
        weights[question_id] = weight_value

# Remove old weights if they exist
if 'weights' in data:
    del data['weights']

# Insert weights after metadata
new_data = {}
for key in data:
    new_data[key] = data[key]
    if key == 'metadata':
        new_data['weights'] = weights

# If metadata wasn't in the expected position, add at end
if 'weights' not in new_data:
    new_data['weights'] = weights

# Write back
with open('RUBRIC_SCORING.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"Generated {len(weights)} weight entries")
print(f"Weight value: {weight_value}")
print(f"First 5: {list(weights.items())[:5]}")
print(f"Last 5: {list(weights.items())[-5:]}")
