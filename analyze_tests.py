#!/usr/bin/env python3
"""Simple test file analyzer"""
import pathlib
import json

repo_root = pathlib.Path(".")
test_files = []

# Find all test files
for f in repo_root.glob("test_*.py"):
    if "venv" not in str(f):
        test_files.append(f)

for f in repo_root.glob("tests/**/*.py"):
    if "venv" not in str(f) and f.name != "__init__.py":
        test_files.append(f)

results = {
    "total_files": len(test_files),
    "deprecated_refs": [],
    "missing_rubric": [],
    "missing_answer_assembler": [],
    "missing_artifacts": [],
    "missing_determinism": []
}

for tf in test_files:
    content = tf.read_text(encoding='utf-8', errors='ignore')
    fname = str(tf.relative_to(repo_root))
    
    # Check for deprecated
    if 'decalogo_pipeline_orchestrator' in content:
        results["deprecated_refs"].append(fname)
    
    # Check rubric in evaluator tests
    if any(x in tf.stem for x in ['answer_assembler', 'evaluation', 'pipeline', 'unified', 'orchestrator', 'rubric']):
        if 'RUBRIC_SCORING' not in content and 'rubric_scoring' not in content:
            results["missing_rubric"].append(fname)
    
    # Check AnswerAssembler
    if any(x in tf.stem for x in ['evaluation', 'pipeline', 'unified', 'e2e']):
        if 'AnswerAssembler' not in content and 'answer_assembler' not in content:
            results["missing_answer_assembler"].append(fname)
    
    # Check artifacts
    if any(x in tf.stem for x in ['e2e', 'pipeline', 'orchestrator', 'unified']):
        if 'artifact' in content.lower():
            if 'answers_report.json' not in content or 'flow_runtime.json' not in content:
                results["missing_artifacts"].append(fname)
    
    # Check determinism
    if any(x in tf.stem for x in ['orchestrator', 'pipeline', 'e2e', 'unified', 'critical']):
        if 'deterministic' not in content.lower() and 'hash' not in content.lower():
            results["missing_determinism"].append(fname)

print(json.dumps(results, indent=2))
