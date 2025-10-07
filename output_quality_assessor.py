import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics


def validate_output_quality(
    answers_path: Optional[str] = None,
    rubric_path: Optional[str] = None,
    evidence_registry_path: Optional[str] = None,
    flow_runtime_path: Optional[str] = None,
    flow_doc_path: Optional[str] = None,
    validation_gates_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Programmatically validate all acceptance criteria for output quality.
    
    Returns structured dictionary with:
    - boolean pass/fail per criterion
    - detailed metric values
    - consolidated overall_pass boolean
    """
    repo_root = Path(__file__).parent
    
    answers_path = Path(answers_path) if answers_path else repo_root / "artifacts" / "answers_report.json"
    rubric_path = Path(rubric_path) if rubric_path else repo_root / "RUBRIC_SCORING.json"
    evidence_registry_path = Path(evidence_registry_path) if evidence_registry_path else repo_root / "artifacts" / "evidence_registry.json"
    flow_runtime_path = Path(flow_runtime_path) if flow_runtime_path else repo_root / "artifacts" / "flow_runtime.json"
    flow_doc_path = Path(flow_doc_path) if flow_doc_path else repo_root / "tools" / "flow_doc.json"
    validation_gates_path = Path(validation_gates_path) if validation_gates_path else repo_root / "artifacts" / "validation_gates.json"
    output_path = Path(output_path) if output_path else repo_root / "reports" / "output_quality_assessment.json"
    
    results = {
        "criteria": {},
        "metrics": {},
        "overall_pass": False,
        "summary": {},
        "errors": []
    }
    
    # Criterion 1: Exactly 300 questions present
    try:
        if not answers_path.exists():
            results["criteria"]["question_count"] = {
                "pass": False,
                "expected": 300,
                "actual": None,
                "error": f"answers_report.json not found at {answers_path}"
            }
        else:
            with open(answers_path) as f:
                answers_data = json.load(f)
            
            question_count = len(answers_data.get("question_answers", []))
            results["criteria"]["question_count"] = {
                "pass": question_count == 300,
                "expected": 300,
                "actual": question_count
            }
    except Exception as e:
        results["criteria"]["question_count"] = {
            "pass": False,
            "expected": 300,
            "actual": None,
            "error": str(e)
        }
        results["errors"].append(f"Question count validation failed: {e}")
    
    # Criterion 2: Rubric alignment via subprocess
    try:
        rubric_check_script = repo_root / "tools" / "rubric_check.py"
        if not rubric_check_script.exists():
            results["criteria"]["rubric_alignment"] = {
                "pass": False,
                "exit_code": None,
                "error": f"rubric_check.py not found at {rubric_check_script}"
            }
        else:
            cmd = [sys.executable, str(rubric_check_script), str(answers_path), str(rubric_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            results["criteria"]["rubric_alignment"] = {
                "pass": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip() if proc.stderr else None
            }
    except subprocess.TimeoutExpired:
        results["criteria"]["rubric_alignment"] = {
            "pass": False,
            "exit_code": None,
            "error": "rubric_check.py subprocess timeout"
        }
        results["errors"].append("Rubric check subprocess timed out")
    except Exception as e:
        results["criteria"]["rubric_alignment"] = {
            "pass": False,
            "exit_code": None,
            "error": str(e)
        }
        results["errors"].append(f"Rubric alignment check failed: {e}")
    
    # Criterion 3: All 15 pipeline stages contributed evidence
    try:
        if not evidence_registry_path.exists():
            results["criteria"]["pipeline_stage_coverage"] = {
                "pass": False,
                "expected_stages": 15,
                "actual_stages": 0,
                "contributing_stages": [],
                "error": f"evidence_registry.json not found at {evidence_registry_path}"
            }
        else:
            with open(evidence_registry_path) as f:
                evidence_data = json.load(f)
            
            stage_contributions = defaultdict(int)
            for evidence in evidence_data.get("evidences", []):
                stage = evidence.get("pipeline_stage")
                if stage:
                    stage_contributions[stage] += 1
            
            contributing_stages = sorted(stage_contributions.keys())
            stage_count = len(contributing_stages)
            
            results["criteria"]["pipeline_stage_coverage"] = {
                "pass": stage_count >= 15,
                "expected_stages": 15,
                "actual_stages": stage_count,
                "contributing_stages": contributing_stages,
                "stage_evidence_counts": dict(stage_contributions)
            }
    except Exception as e:
        results["criteria"]["pipeline_stage_coverage"] = {
            "pass": False,
            "expected_stages": 15,
            "actual_stages": 0,
            "contributing_stages": [],
            "error": str(e)
        }
        results["errors"].append(f"Pipeline stage coverage check failed: {e}")
    
    # Criterion 4: Flow order matches canonical order
    try:
        flow_order_pass = True
        flow_order_details = {}
        
        if not flow_runtime_path.exists():
            flow_order_pass = False
            flow_order_details["error"] = f"flow_runtime.json not found at {flow_runtime_path}"
        elif not flow_doc_path.exists():
            flow_order_pass = False
            flow_order_details["error"] = f"flow_doc.json not found at {flow_doc_path}"
        else:
            with open(flow_runtime_path) as f:
                runtime_data = json.load(f)
            with open(flow_doc_path) as f:
                flow_doc_data = json.load(f)
            
            runtime_order = runtime_data.get("stage_order", [])
            canonical_order = flow_doc_data.get("canonical_order", [])
            
            deviations = []
            for i, (runtime_stage, canonical_stage) in enumerate(zip(runtime_order, canonical_order)):
                if runtime_stage != canonical_stage:
                    deviations.append({
                        "position": i,
                        "expected": canonical_stage,
                        "actual": runtime_stage
                    })
            
            if len(runtime_order) != len(canonical_order):
                deviations.append({
                    "issue": "length_mismatch",
                    "expected_length": len(canonical_order),
                    "actual_length": len(runtime_order)
                })
            
            flow_order_pass = len(deviations) == 0
            flow_order_details = {
                "runtime_order": runtime_order,
                "canonical_order": canonical_order,
                "deviations": deviations
            }
        
        results["criteria"]["flow_order_match"] = {
            "pass": flow_order_pass,
            **flow_order_details
        }
    except Exception as e:
        results["criteria"]["flow_order_match"] = {
            "pass": False,
            "error": str(e)
        }
        results["errors"].append(f"Flow order validation failed: {e}")
    
    # Criterion 5: All 6 validation gates have passing status
    try:
        required_gates = [
            "immutability_verified",
            "flow_order_match",
            "evidence_deterministic_hash_consistency",
            "coverage_300_300",
            "rubric_alignment",
            "triple_run_determinism"
        ]
        
        if not validation_gates_path.exists():
            results["criteria"]["validation_gates"] = {
                "pass": False,
                "expected_gates": 6,
                "passing_gates": 0,
                "gate_status": {},
                "error": f"validation_gates.json not found at {validation_gates_path}"
            }
        else:
            with open(validation_gates_path) as f:
                gates_data = json.load(f)
            
            gate_status = {}
            for gate in required_gates:
                gate_info = gates_data.get(gate, {})
                if isinstance(gate_info, dict):
                    gate_status[gate] = gate_info.get("status") == "pass"
                else:
                    gate_status[gate] = gate_info == "pass"
            
            passing_gates = sum(1 for status in gate_status.values() if status)
            
            results["criteria"]["validation_gates"] = {
                "pass": passing_gates == 6,
                "expected_gates": 6,
                "passing_gates": passing_gates,
                "gate_status": gate_status
            }
    except Exception as e:
        results["criteria"]["validation_gates"] = {
            "pass": False,
            "expected_gates": 6,
            "passing_gates": 0,
            "gate_status": {},
            "error": str(e)
        }
        results["errors"].append(f"Validation gates check failed: {e}")
    
    # Quality Metrics: confidence scores, evidence distribution, rationale completeness
    try:
        if answers_path.exists():
            with open(answers_path) as f:
                answers_data = json.load(f)
            
            question_answers = answers_data.get("question_answers", [])
            
            # Confidence scores
            confidences = [qa.get("confidence", 0) for qa in question_answers if qa.get("confidence") is not None]
            if confidences:
                results["metrics"]["confidence_scores"] = {
                    "mean": statistics.mean(confidences),
                    "median": statistics.median(confidences),
                    "min": min(confidences),
                    "max": max(confidences),
                    "stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                    "count": len(confidences)
                }
            else:
                results["metrics"]["confidence_scores"] = {
                    "mean": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "stdev": None,
                    "count": 0
                }
            
            # Evidence distribution
            evidence_counts = [qa.get("evidence_count", 0) for qa in question_answers]
            evidence_distribution = defaultdict(int)
            for count in evidence_counts:
                evidence_distribution[count] += 1
            
            results["metrics"]["evidence_distribution"] = {
                "mean": statistics.mean(evidence_counts) if evidence_counts else 0,
                "median": statistics.median(evidence_counts) if evidence_counts else 0,
                "min": min(evidence_counts) if evidence_counts else 0,
                "max": max(evidence_counts) if evidence_counts else 0,
                "distribution": dict(evidence_distribution),
                "questions_with_zero_evidence": evidence_distribution[0]
            }
            
            # Rationale completeness
            rationales_present = sum(1 for qa in question_answers if qa.get("rationale"))
            rationale_lengths = [len(qa.get("rationale", "")) for qa in question_answers if qa.get("rationale")]
            
            results["metrics"]["rationale_completeness"] = {
                "total_questions": len(question_answers),
                "questions_with_rationale": rationales_present,
                "completeness_percentage": (rationales_present / len(question_answers) * 100) if question_answers else 0,
                "mean_length": statistics.mean(rationale_lengths) if rationale_lengths else 0,
                "min_length": min(rationale_lengths) if rationale_lengths else 0,
                "max_length": max(rationale_lengths) if rationale_lengths else 0
            }
    except Exception as e:
        results["errors"].append(f"Quality metrics computation failed: {e}")
    
    # Determine overall pass
    all_criteria_pass = all(
        criterion.get("pass", False)
        for criterion in results["criteria"].values()
    )
    results["overall_pass"] = all_criteria_pass
    
    # Generate summary
    results["summary"] = {
        "total_criteria": len(results["criteria"]),
        "passing_criteria": sum(1 for c in results["criteria"].values() if c.get("pass", False)),
        "failing_criteria": [name for name, c in results["criteria"].items() if not c.get("pass", False)],
        "has_errors": len(results["errors"]) > 0,
        "error_count": len(results["errors"])
    }
    
    # Write results to output file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        results["errors"].append(f"Failed to write output file: {e}")
    
    return results


def main():
    """CLI entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate output quality against acceptance criteria")
    parser.add_argument("--answers", help="Path to answers_report.json")
    parser.add_argument("--rubric", help="Path to RUBRIC_SCORING.json")
    parser.add_argument("--evidence", help="Path to evidence_registry.json")
    parser.add_argument("--flow-runtime", help="Path to flow_runtime.json")
    parser.add_argument("--flow-doc", help="Path to flow_doc.json")
    parser.add_argument("--gates", help="Path to validation_gates.json")
    parser.add_argument("--output", help="Output path for results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    results = validate_output_quality(
        answers_path=args.answers,
        rubric_path=args.rubric,
        evidence_registry_path=args.evidence,
        flow_runtime_path=args.flow_runtime,
        flow_doc_path=args.flow_doc,
        validation_gates_path=args.gates,
        output_path=args.output
    )
    
    if args.verbose:
        print(json.dumps(results, indent=2))
    else:
        print(f"Overall Pass: {results['overall_pass']}")
        print(f"Passing Criteria: {results['summary']['passing_criteria']}/{results['summary']['total_criteria']}")
        if results['summary']['failing_criteria']:
            print(f"Failing: {', '.join(results['summary']['failing_criteria'])}")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
    
    sys.exit(0 if results['overall_pass'] else 1)


if __name__ == "__main__":
    main()
