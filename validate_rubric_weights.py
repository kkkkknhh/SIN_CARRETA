#!/usr/bin/env python3
"""Validate RUBRIC_SCORING.json weights structure."""

import json

with open('RUBRIC_SCORING.json', 'r') as f:
    data = json.load(f)

weights = data.get('weights', {})
print(f"Total weight entries: {len(weights)}")

# Get all keys
all_keys = list(weights.keys())
print(f"\nFirst 15 keys: {all_keys[:15]}")
print(f"Last 15 keys: {all_keys[-15:]}")

# Check for different patterns
pattern_P_D_Q = [k for k in all_keys if k.startswith('P') and 'D' in k and 'Q' in k]
pattern_D_Q_P = [k for k in all_keys if k.startswith('D') and '-Q' in k and '-P' in k]
pattern_D_Q = [k for k in all_keys if k.startswith('D') and '-Q' in k and k.count('-') == 1]

print(f"\nPattern P#-D#-Q#: {len(pattern_P_D_Q)} entries")
print(f"Pattern D#-Q#-P#: {len(pattern_D_Q_P)} entries")
print(f"Pattern D#-Q# (no P): {len(pattern_D_Q)} entries")

# Check unique weight values
values = set(weights.values())
print(f"\nUnique weight values: {values}")

# Expected weight
expected_weight = 1/300
print(f"Expected weight (1/300): {expected_weight}")

# Check if both keys exist at top level
print(f"\nTop-level keys: {list(data.keys())}")
print(f"'questions' exists: {'questions' in data}")
print(f"'weights' exists: {'weights' in data}")
