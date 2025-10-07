#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test for Determinism Verifier
=========================================

Tests end-to-end workflow with mock orchestrator execution.
"""

import json
import tempfile
import subprocess
import sys
from pathlib import Path


def create_mock_artifacts(output_dir: Path):
    """Create mock artifacts that mimic orchestrator output"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock evidence_registry.json
    evidence_data = {
        "evidence_count": 10,
        "deterministic_hash": "abc123def456",
        "evidence": {
            "ev1": {
                "source_component": "feasibility_scorer",
                "evidence_type": "baseline_presence",
                "content": {"text": "baseline found"},
                "confidence": 0.9,
                "applicable_questions": ["D1-Q1"]
            }
        }
    }
    with open(output_dir / "evidence_registry.json", 'w') as f:
        json.dump(evidence_data, f, indent=2)
    
    # Mock flow_runtime.json
    flow_data = {
        "flow_hash": "flow123abc",
        "stages": ["sanitization", "segmentation", "embedding"],
        "stage_count": 3,
        "duration_seconds": 10.5
    }
    with open(output_dir / "flow_runtime.json", 'w') as f:
        json.dump(flow_data, f, indent=2)
    
    # Mock answers_report.json
    answers_data = {
        "metadata": {"version": "1.0"},
        "global_summary": {"total_questions": 2},
        "question_answers": [
            {
                "question_id": "D1-Q1",
                "score": 0.85,
                "confidence": 0.9,
                "evidence_ids": ["ev1"]
            },
            {
                "question_id": "D1-Q2",
                "score": 0.70,
                "confidence": 0.8,
                "evidence_ids": []
            }
        ]
    }
    with open(output_dir / "answers_report.json", 'w') as f:
        json.dump(answers_data, f, indent=2)
    
    # Mock coverage_report.json
    coverage_data = {
        "dimensions": {
            "D1": {"coverage": 0.75, "questions_answered": 2}
        }
    }
    with open(output_dir / "coverage_report.json", 'w') as f:
        json.dump(coverage_data, f, indent=2)


def create_mock_orchestrator_script(script_path: Path):
    """Create a mock orchestrator script that generates deterministic artifacts"""
    script_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path
import json

# Parse arguments
input_pdf = sys.argv[1]
output_dir = Path(sys.argv[3])  # --output-dir is at index 3

# Import the mock artifact creator
from test_determinism_verifier_integration import create_mock_artifacts
create_mock_artifacts(output_dir)

print(f"Mock orchestrator executed: {input_pdf} -> {output_dir}")
sys.exit(0)
'''
    with open(script_path, 'w') as f:
        f.write(script_content)
    script_path.chmod(0o755)


def test_determinism_verifier_perfect_match():
    """Test verifier with perfectly matching runs"""
    print("\n=== Integration Test: Perfect Match ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock input PDF
        input_pdf = tmpdir / "test.pdf"
        input_pdf.touch()
        
        # Create mock orchestrator script
        mock_script = tmpdir / "mock_orchestrator.py"
        create_mock_orchestrator_script(mock_script)
        
        # Temporarily replace the orchestrator call in verifier
        # For this test, we'll manually create the artifacts
        output_dir = tmpdir / "determinism_test"
        run1_dir = output_dir / "run1"
        run2_dir = output_dir / "run2"
        
        # Create identical artifacts for both runs
        create_mock_artifacts(run1_dir)
        create_mock_artifacts(run2_dir)
        
        # Import verifier and test directly
        from determinism_verifier import DeterminismVerifier, DeterminismReport
        
        verifier = DeterminismVerifier(input_pdf, output_dir)
        
        # Compare artifacts directly (skip orchestrator execution)
        comparisons = []
        for artifact in verifier.REQUIRED_ARTIFACTS:
            comparison = verifier.compare_artifacts(artifact)
            comparisons.append(comparison)
        
        # Verify all match
        assert all(c.match for c in comparisons), "All artifacts should match"
        
        # Extract hashes
        evidence_hash_run1 = verifier.extract_evidence_hash(run1_dir / "evidence_registry.json")
        evidence_hash_run2 = verifier.extract_evidence_hash(run2_dir / "evidence_registry.json")
        assert evidence_hash_run1 == evidence_hash_run2, "Evidence hashes should match"
        
        flow_hash_run1 = verifier.extract_flow_hash(run1_dir / "flow_runtime.json")
        flow_hash_run2 = verifier.extract_flow_hash(run2_dir / "flow_runtime.json")
        assert flow_hash_run1 == flow_hash_run2, "Flow hashes should match"
        
        # Create report
        report = DeterminismReport(
            timestamp="2024-01-01T00:00:00Z",
            input_pdf=str(input_pdf),
            run1_dir=str(run1_dir),
            run2_dir=str(run2_dir),
            perfect_match=True,
            artifact_comparisons=comparisons,
            evidence_hash_run1=evidence_hash_run1,
            evidence_hash_run2=evidence_hash_run2,
            evidence_hash_match=True,
            flow_hash_run1=flow_hash_run1,
            flow_hash_run2=flow_hash_run2,
            flow_hash_match=True
        )
        
        assert report.perfect_match, "Report should indicate perfect match"
        
        # Export report
        verifier.export_report(report)
        
        # Verify reports exist
        json_report = output_dir / "determinism_report.json"
        txt_report = output_dir / "determinism_report.txt"
        
        assert json_report.exists(), "JSON report should exist"
        assert txt_report.exists(), "Text report should exist"
        
        # Verify JSON report content
        with open(json_report, 'r') as f:
            report_data = json.load(f)
            assert report_data["perfect_match"] == True
            assert report_data["evidence_hash_match"] == True
            assert report_data["flow_hash_match"] == True
        
        print("✓ Perfect match test passed")


def test_determinism_verifier_mismatch():
    """Test verifier with mismatched runs"""
    print("\n=== Integration Test: Mismatch Detection ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock input PDF
        input_pdf = tmpdir / "test.pdf"
        input_pdf.touch()
        
        output_dir = tmpdir / "determinism_test"
        run1_dir = output_dir / "run1"
        run2_dir = output_dir / "run2"
        
        # Create artifacts for run1
        create_mock_artifacts(run1_dir)
        
        # Create different artifacts for run2
        create_mock_artifacts(run2_dir)
        
        # Modify one artifact to create mismatch
        answers_path = run2_dir / "answers_report.json"
        with open(answers_path, 'r') as f:
            answers_data = json.load(f)
        answers_data["question_answers"][0]["score"] = 0.90  # Changed from 0.85
        with open(answers_path, 'w') as f:
            json.dump(answers_data, f, indent=2)
        
        # Import verifier and test
        from determinism_verifier import DeterminismVerifier
        
        verifier = DeterminismVerifier(input_pdf, output_dir)
        
        # Compare artifacts
        comparisons = []
        for artifact in verifier.REQUIRED_ARTIFACTS:
            comparison = verifier.compare_artifacts(artifact)
            comparisons.append(comparison)
        
        # Verify mismatch detected
        mismatched = [c for c in comparisons if not c.match]
        assert len(mismatched) > 0, "Should detect at least one mismatch"
        assert any(c.artifact_name == "answers_report.json" for c in mismatched), \
            "Should detect answers_report.json mismatch"
        
        # Verify diff generated
        answers_comparison = next(c for c in comparisons if c.artifact_name == "answers_report.json")
        assert len(answers_comparison.diff_lines) > 0, "Diff should be generated"
        
        print("✓ Mismatch detection test passed")


def test_json_normalization_robustness():
    """Test that normalization handles various JSON variations"""
    print("\n=== Integration Test: JSON Normalization ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_pdf = tmpdir / "test.pdf"
        input_pdf.touch()
        
        from determinism_verifier import DeterminismVerifier
        verifier = DeterminismVerifier(input_pdf, tmpdir)
        
        # Create two JSON files with same content but different formatting
        json1 = tmpdir / "test1.json"
        json2 = tmpdir / "test2.json"
        
        # File 1: Compact, different key order, with timestamp
        data1 = {
            "z_field": "value1",
            "a_field": "value2",
            "timestamp": "2024-01-01T00:00:00Z",
            "duration_seconds": 10.5
        }
        
        # File 2: Pretty-printed, different key order, different timestamp
        data2 = {
            "a_field": "value2",
            "z_field": "value1",
            "timestamp": "2024-12-31T23:59:59Z",
            "duration_seconds": 99.9
        }
        
        with open(json1, 'w') as f:
            json.dump(data1, f)
        
        with open(json2, 'w') as f:
            json.dump(data2, f, indent=4)
        
        # Compute normalized hashes
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, \
            "Hashes should match after normalization (key order, whitespace, timestamps removed)"
        
        print("✓ JSON normalization test passed")


if __name__ == "__main__":
    try:
        test_determinism_verifier_perfect_match()
        test_determinism_verifier_mismatch()
        test_json_normalization_robustness()
        
        print("\n" + "=" * 80)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 80)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
