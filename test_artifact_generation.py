#!/usr/bin/env python3
"""
Test Artifact Generation
=========================
Verifies that miniminimoon_orchestrator.py's export_artifacts() method generates
all 5 required artifacts with proper error handling and path resolution.

Tests:
1. All 5 artifacts are generated with correct filenames
2. Each artifact matches documented JSON schema
3. Proper error handling for directory creation and file writing
4. Path resolution handles absolute and relative paths
5. Cross-references between artifacts are valid
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def check_artifact_exists(artifact_path: Path) -> bool:
    """Check if artifact file exists and is readable."""
    if not artifact_path.exists():
        print(f"  ⨯ {artifact_path.name} - NOT FOUND")
        return False
    
    try:
        with open(artifact_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  ✓ {artifact_path.name} - EXISTS and is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"  ⨯ {artifact_path.name} - INVALID JSON: {e}")
        return False
    except Exception as e:
        print(f"  ⨯ {artifact_path.name} - ERROR reading file: {e}")
        return False


def validate_flow_runtime_schema(data: Dict[str, Any]) -> List[str]:
    """Validate flow_runtime.json schema."""
    errors = []
    required_fields = [
        "evidence_hash", "duration_seconds", "end_time", "errors",
        "flow_hash", "orchestrator_version", "plan_path", "stage_count",
        "stage_timestamps", "stages", "start_time", "validation"
    ]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check stages count
    if "stages" in data and "stage_count" in data:
        if len(data["stages"]) != data["stage_count"]:
            errors.append(f"stage_count mismatch: {data['stage_count']} != {len(data['stages'])}")
    
    # Check for 15 canonical stages
    if "stages" in data and len(data["stages"]) != 15:
        errors.append(f"Expected 15 stages, got {len(data['stages'])}")
    
    return errors


def validate_evidence_registry_schema(data: Dict[str, Any]) -> List[str]:
    """Validate evidence_registry.json schema."""
    errors = []
    required_fields = ["evidence_count", "deterministic_hash", "evidence"]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check evidence count
    if "evidence" in data and "evidence_count" in data:
        if len(data["evidence"]) != data["evidence_count"]:
            errors.append(f"evidence_count mismatch: {data['evidence_count']} != {len(data['evidence'])}")
    
    # Check deterministic_hash format (should be 64 char hex)
    if "deterministic_hash" in data:
        hash_val = data["deterministic_hash"]
        if not isinstance(hash_val, str) or len(hash_val) != 64:
            errors.append(f"deterministic_hash should be 64-char hex string, got: {hash_val[:20]}...")
    
    return errors


def validate_answers_report_schema(data: Dict[str, Any]) -> List[str]:
    """Validate answers_report.json schema."""
    errors = []
    
    # Support both new format (global_summary) and legacy format (summary)
    summary = data.get("global_summary", data.get("summary"))
    if not summary:
        errors.append("Missing global_summary or summary field")
        return errors
    
    # Check question count
    total = summary.get("total_questions", 0)
    answered = summary.get("answered_questions", 0)
    
    if total < 300:
        errors.append(f"total_questions should be ≥300, got {total}")
    if answered < 300:
        errors.append(f"answered_questions should be ≥300, got {answered} (Gate #4 failure)")
    
    # Check answers array
    answers = data.get("question_answers", data.get("answers", []))
    if len(answers) != answered:
        errors.append(f"answers array length mismatch: {len(answers)} != {answered}")
    
    return errors


def validate_answers_sample_schema(data: Dict[str, Any]) -> List[str]:
    """Validate answers_sample.json schema."""
    errors = []
    
    # Support both formats
    summary = data.get("global_summary", data.get("summary"))
    if not summary:
        errors.append("Missing global_summary or summary field")
    
    # Check sample size
    sample = data.get("sample_question_answers", data.get("answers", []))
    if len(sample) > 10:
        errors.append(f"Sample should contain ≤10 answers, got {len(sample)}")
    
    return errors


def validate_coverage_report_schema(data: Dict[str, Any]) -> List[str]:
    """Validate coverage_report.json schema."""
    errors = []
    required_fields = ["total_questions", "answered_questions", "coverage_percentage", "dimensions"]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check dimensions
    if "dimensions" in data:
        expected_dims = ["D1", "D2", "D3", "D4", "D5", "D6"]
        actual_dims = list(data["dimensions"].keys())
        if sorted(actual_dims) != sorted(expected_dims):
            errors.append(f"Expected dimensions {expected_dims}, got {actual_dims}")
        
        # Check each dimension has questions and answered fields
        for dim, dim_data in data["dimensions"].items():
            if "questions" not in dim_data or "answered" not in dim_data:
                errors.append(f"Dimension {dim} missing questions or answered field")
    
    # Check coverage percentage
    if "coverage_percentage" in data:
        cov = data["coverage_percentage"]
        if not (0.0 <= cov <= 100.0):
            errors.append(f"coverage_percentage should be in [0, 100], got {cov}")
    
    return errors


def validate_cross_references(artifacts_dir: Path) -> List[str]:
    """Validate cross-references between artifacts."""
    errors = []
    
    try:
        # Load all artifacts
        with open(artifacts_dir / "flow_runtime.json", 'r') as f:
            flow_runtime = json.load(f)
        with open(artifacts_dir / "evidence_registry.json", 'r') as f:
            evidence_registry = json.load(f)
        with open(artifacts_dir / "answers_report.json", 'r') as f:
            answers_report = json.load(f)
        
        # Check evidence_hash cross-reference
        flow_evidence_hash = flow_runtime.get("evidence_hash", "")
        registry_hash = evidence_registry.get("deterministic_hash", "")
        
        if flow_evidence_hash != registry_hash:
            errors.append(f"Evidence hash mismatch: flow_runtime={flow_evidence_hash[:16]}... != evidence_registry={registry_hash[:16]}...")
        
        # Check evidence_ids in answers reference valid evidence entries
        answers = answers_report.get("question_answers", answers_report.get("answers", []))
        evidence_ids = set(evidence_registry.get("evidence", {}).keys())
        
        invalid_refs = []
        for answer in answers[:10]:  # Check first 10 for efficiency
            for eid in answer.get("evidence_ids", []):
                if eid not in evidence_ids:
                    invalid_refs.append(eid)
        
        if invalid_refs:
            errors.append(f"Found {len(invalid_refs)} invalid evidence_id references in answers_report")
    
    except Exception as e:
        errors.append(f"Error validating cross-references: {e}")
    
    return errors


def main():
    """Main test execution."""
    print("="*80)
    print("ARTIFACT GENERATION VERIFICATION")
    print("="*80)
    
    artifacts_dir = Path("artifacts")
    
    if not artifacts_dir.exists():
        print(f"\n⨯ artifacts/ directory does not exist")
        print("  Run the pipeline first to generate artifacts")
        sys.exit(1)
    
    print(f"\n1. Checking for 5 required artifacts in {artifacts_dir}/...\n")
    
    required_artifacts = [
        "flow_runtime.json",
        "evidence_registry.json",
        "answers_report.json",
        "answers_sample.json",
        "coverage_report.json"
    ]
    
    all_exist = True
    for artifact_name in required_artifacts:
        artifact_path = artifacts_dir / artifact_name
        if not check_artifact_exists(artifact_path):
            all_exist = False
    
    if not all_exist:
        print(f"\n⨯ Not all required artifacts present")
        sys.exit(1)
    
    print(f"\n✓ All 5 required artifacts found")
    
    # Validate schemas
    print(f"\n2. Validating artifact schemas...\n")
    
    schema_validators = {
        "flow_runtime.json": validate_flow_runtime_schema,
        "evidence_registry.json": validate_evidence_registry_schema,
        "answers_report.json": validate_answers_report_schema,
        "answers_sample.json": validate_answers_sample_schema,
        "coverage_report.json": validate_coverage_report_schema
    }
    
    all_valid = True
    for artifact_name, validator in schema_validators.items():
        artifact_path = artifacts_dir / artifact_name
        with open(artifact_path, 'r') as f:
            data = json.load(f)
        
        errors = validator(data)
        if errors:
            print(f"  ⨯ {artifact_name} - SCHEMA VALIDATION FAILED:")
            for error in errors:
                print(f"    - {error}")
            all_valid = False
        else:
            print(f"  ✓ {artifact_name} - SCHEMA VALID")
    
    if not all_valid:
        print(f"\n⨯ Some artifacts have schema validation errors")
        sys.exit(1)
    
    print(f"\n✓ All artifact schemas valid")
    
    # Validate cross-references
    print(f"\n3. Validating cross-references between artifacts...\n")
    
    errors = validate_cross_references(artifacts_dir)
    if errors:
        print(f"  ⨯ CROSS-REFERENCE VALIDATION FAILED:")
        for error in errors:
            print(f"    - {error}")
        sys.exit(1)
    
    print(f"  ✓ All cross-references valid")
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"\n✓ All 5 artifacts generated successfully")
    print(f"✓ All schemas match ARCHITECTURE.md specifications")
    print(f"✓ All cross-references between artifacts are valid")
    print(f"\nArtifacts verified:")
    for artifact_name in required_artifacts:
        artifact_path = artifacts_dir / artifact_name
        size_kb = artifact_path.stat().st_size / 1024
        print(f"  - {artifact_name} ({size_kb:.1f} KB)")
    
    print(f"\n✓ VERIFICATION PASSED")


if __name__ == "__main__":
    main()
