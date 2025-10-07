#!/usr/bin/env python3
"""Validate answers_report.json format and structure"""
import json
import re

# Load answers
with open('artifacts/answers_report.json', 'r') as f:
    data = json.load(f)

answers = data.get('answers', [])
print(f"✅ Total answers: {len(answers)}")

# Check IDs
ids = [a['question_id'] for a in answers]
print("\n📋 Question ID samples:")
print(f"   First 10: {ids[:10]}")
print(f"   Last 10: {ids[-10:]}")

# Validate format: D{N}-Q{N}
pattern = re.compile(r'^D\d+-Q\d+$')
valid_format = all(pattern.match(id) for id in ids)
print(f"\n✅ All IDs match D{{N}}-Q{{N}} format: {valid_format}")

# Count per dimension
from collections import Counter
dimensions = [id.split('-')[0] for id in ids]
dim_counts = Counter(dimensions)
print("\n📊 Questions per dimension:")
for dim in sorted(dim_counts.keys()):
    print(f"   {dim}: {dim_counts[dim]} questions")

# Check expected range
expected_dims = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
all_dims_present = all(dim in dim_counts for dim in expected_dims)
print(f"\n✅ All dimensions D1-D6 present: {all_dims_present}")

# Total validation
if len(answers) == 300 and valid_format and all_dims_present:
    print("\n✅ VALIDATION PASSED: 300 questions with standardized D{{N}}-Q{{N}} format")
else:
    print("\n❌ VALIDATION FAILED")
    if len(answers) != 300:
        print(f"   - Expected 300 answers, got {len(answers)}")
    if not valid_format:
        print("   - ID format validation failed")
    if not all_dims_present:
        print("   - Not all dimensions D1-D6 present")
