#!/usr/bin/env python3
"""Simple integration test for output_quality_assessor"""
import sys
from pathlib import Path
from output_quality_assessor import validate_output_quality

test_data = Path(__file__).parent / "test_data"

print("Running output quality assessment...")
print(f"Test data directory: {test_data}")

results = validate_output_quality(
    answers_path=str(test_data / "answers_300.json"),
    rubric_path=str(test_data / "rubric.json"),
    evidence_registry_path=str(test_data / "evidence_registry_15.json"),
    flow_runtime_path=str(test_data / "flow_runtime.json"),
    flow_doc_path=str(test_data / "flow_doc.json"),
    validation_gates_path=str(test_data / "validation_gates_pass.json"),
    output_path=str(test_data / "output.json")
)

print(f"\nOverall Pass: {results['overall_pass']}")
print(f"Passing Criteria: {results['summary']['passing_criteria']}/{results['summary']['total_criteria']}")

for criterion, result in results["criteria"].items():
    status = "✓" if result.get("pass") else "✗"
    print(f"  {status} {criterion}")

if results['summary']['failing_criteria']:
    print(f"\nFailing: {', '.join(results['summary']['failing_criteria'])}")

sys.exit(0 if results['overall_pass'] else 1)
