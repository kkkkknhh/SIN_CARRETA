#!/usr/bin/env python3
import json
from pathlib import Path

test_data_dir = Path(__file__).parent

# Create answers with 300 questions
answers_300 = {
    "question_answers": [
        {
            "question_id": f"Q{i}",
            "confidence": 0.85 + (i % 10) * 0.01,
            "evidence_count": (i % 4),
            "rationale": f"Rationale for question {i}"
        }
        for i in range(300)
    ],
    "answers": [
        {
            "question_id": f"Q{i}",
            "confidence": 0.85 + (i % 10) * 0.01,
            "evidence_count": (i % 4),
            "rationale": f"Rationale for question {i}"
        }
        for i in range(300)
    ]
}
with open(test_data_dir / "answers_300.json", 'w') as f:
    json.dump(answers_300, f)

# Create evidence registry with 15 stages
evidence_registry = {
    "evidences": [
        {
            "evidence_id": f"ev_{stage}_{i}",
            "pipeline_stage": f"stage_{stage}"
        }
        for stage in range(1, 16)
        for i in range(3)
    ]
}
with open(test_data_dir / "evidence_registry_15.json", 'w') as f:
    json.dump(evidence_registry, f)

# Create flow runtime and doc with matching order
flow_order = [f"stage_{i}" for i in range(1, 11)]
with open(test_data_dir / "flow_runtime.json", 'w') as f:
    json.dump({"stage_order": flow_order}, f)
with open(test_data_dir / "flow_doc.json", 'w') as f:
    json.dump({"canonical_order": flow_order}, f)

# Create passing validation gates
gates = {
    "immutability_verified": {"status": "pass"},
    "flow_order_match": {"status": "pass"},
    "evidence_deterministic_hash_consistency": {"status": "pass"},
    "coverage_300_300": {"status": "pass"},
    "rubric_alignment": {"status": "pass"},
    "triple_run_determinism": {"status": "pass"}
}
with open(test_data_dir / "validation_gates_pass.json", 'w') as f:
    json.dump(gates, f)

# Create rubric file for rubric_check.py
rubric = {
    "weights": {f"Q{i}": 1.0 for i in range(300)}
}
with open(test_data_dir / "rubric.json", 'w') as f:
    json.dump(rubric, f)

print("Test fixtures created successfully")
