#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock execution script to verify test infrastructure without running full pipeline
"""

import json
import sys
import hashlib
from pathlib import Path

def generate_mock_artifacts(artifacts_dir: Path):
    """Generate minimal mock artifacts for testing"""
    
    # Create artifacts directory
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate answers_report.json (300 questions)
    # Use varied scores and confidences for doctoral-level quality
    answers = []
    for i in range(300):
        question_id = f"D{i//50 + 1}-Q{i%50 + 1}"
        
        # Vary scores across range [0, 3] with some distribution
        score_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        score = score_options[i % len(score_options)]
        
        # Vary confidences in range [0.6, 0.9]
        confidence = 0.65 + (i % 50) * 0.005  # Ranges from 0.65 to 0.895
        
        # Vary rationale length (50-150 chars)
        rationale_base = f"Comprehensive evidence-based analysis for {question_id} considering multiple factors including "
        rationale_detail = "policy alignment, implementation feasibility, budget adequacy, stakeholder engagement, and long-term sustainability metrics"[:i % 100]
        
        answers.append({
            "question_id": question_id,
            "evidence_ids": [f"EV-{i*3+1}", f"EV-{i*3+2}", f"EV-{i*3+3}"],
            "confidence": round(confidence, 2),
            "rationale": rationale_base + rationale_detail,
            "score": score
        })
    
    answers_report = {
        "metadata": {
            "version": "2.0",
            "timestamp": "2024-10-07T09:00:00",
            "evaluator": "mock_evaluator"
        },
        "summary": {
            "total_questions": 300,
            "answered_questions": 300
        },
        "answers": answers
    }
    
    with open(artifacts_dir / "answers_report.json", 'w', encoding='utf-8') as f:
        json.dump(answers_report, f, indent=2, ensure_ascii=True)
    
    # 2. Generate flow_runtime.json (canonical order)
    canonical_order = [
        "sanitization",
        "plan_processing",
        "document_segmentation",
        "embedding",
        "responsibility_detection",
        "contradiction_detection",
        "monetary_detection",
        "feasibility_scoring",
        "causal_detection",
        "teoria_cambio",
        "dag_validation",
        "evidence_registry_build",
        "decalogo_evaluation",
        "questionnaire_evaluation",
        "answers_assembly"
    ]
    
    flow_runtime = {
        "execution_order": canonical_order,
        "order": canonical_order,
        "nodes": [{"name": node, "status": "completed"} for node in canonical_order],
        "timestamp": "2024-10-07T09:00:00"
    }
    
    with open(artifacts_dir / "flow_runtime.json", 'w', encoding='utf-8') as f:
        json.dump(flow_runtime, f, indent=2, ensure_ascii=True)
    
    # 3. Generate coverage_report.json
    coverage_report = {
        "total_questions": 300,
        "answered_questions": 300,
        "coverage_percentage": 100.0,
        "dimensions": {
            f"D{i+1}": {"questions": 50, "answered": 50} for i in range(6)
        }
    }
    
    with open(artifacts_dir / "coverage_report.json", 'w', encoding='utf-8') as f:
        json.dump(coverage_report, f, indent=2, ensure_ascii=True)
    
    # 4. Generate evidence_registry.json
    evidence_entries = []
    for i in range(900):  # 3 evidence per question * 300 questions
        evidence_entries.append({
            "evidence_id": f"EV-{i+1}",
            "text": f"Mock evidence text {i+1}",
            "source": "mock_document",
            "confidence": 0.85
        })
    
    evidence_registry = {
        "metadata": {
            "total_evidence": len(evidence_entries),
            "timestamp": "2024-10-07T09:00:00"
        },
        "entries": evidence_entries,
        "deterministic_hash": hashlib.sha256(
            json.dumps(evidence_entries, sort_keys=True).encode()
        ).hexdigest()
    }
    
    with open(artifacts_dir / "evidence_registry.json", 'w', encoding='utf-8') as f:
        json.dump(evidence_registry, f, indent=2, ensure_ascii=True)
    
    # Print to stderr so it doesn't interfere with JSON output parsing
    print(f"✅ Generated mock artifacts in {artifacts_dir}", file=sys.stderr)
    print(f"   - answers_report.json (300 questions)", file=sys.stderr)
    print(f"   - flow_runtime.json (15 nodes)", file=sys.stderr)
    print(f"   - coverage_report.json (300/300)", file=sys.stderr)
    print(f"   - evidence_registry.json (900 entries)", file=sys.stderr)


if __name__ == "__main__":
    repo_root = Path(__file__).parent
    artifacts_dir = repo_root / "artifacts"
    
    # Check if --quiet flag passed
    quiet = "--quiet" in sys.argv
    
    if not quiet:
        print("Mock Execution Script", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
    
    generate_mock_artifacts(artifacts_dir)
    
    # Output success JSON (mimics CLI output) - ALWAYS to stdout for parsing
    result = {
        "ok": True,
        "action": "evaluate",
        "repo": str(repo_root),
        "artifacts_dir": str(artifacts_dir),
        "results": {
            "status": "completed",
            "total_questions": 300,
            "evidence_count": 900
        },
        "pre_validation": {"ok": True},
        "post_validation": {"ok": True}
    }
    
    print(json.dumps(result, indent=2))
    sys.exit(0)
