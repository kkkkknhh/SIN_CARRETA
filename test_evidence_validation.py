#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone test for evidence validation layer functionality.

Demonstrates validation without external dependencies.
"""

import unittest

from evidence_registry import EvidenceProvenance, EvidenceRegistry


def test_validation_basic():
    """Basic validation test"""
    print("Testing evidence validation layer...")

    registry = EvidenceRegistry()

    # Register evidence for questions
    for q_num in range(1, 4):
        question_id = f"D1-Q{q_num}"
        for stage in range(1, 4):
            provenance = EvidenceProvenance(
                detector_type="monetary",
                stage_number=stage,
                source_text_location={"page": 1, "line": 10},
                execution_timestamp="2024-01-01T00:00:00Z",
                quality_metrics={"precision": 0.9},
            )
            registry.register(
                source_component=f"detector_stage_{stage}",
                evidence_type="test_evidence",
                content={"value": f"evidence_{stage}"},
                confidence=0.85,
                applicable_questions=[question_id],
                provenance=provenance,
            )

    # Validate
    result = registry.validate_evidence_counts(
        all_question_ids=["D1-Q1", "D1-Q2", "D1-Q3"], min_evidence_threshold=3
    )

    assert result["valid"] == True
    assert result["total_questions"] == 3
    assert len(result["questions_below_threshold"]) == 0

    print("✓ All 3 questions meet threshold of 3 evidence sources")

    # Check stage tracking
    for qid in ["D1-Q1", "D1-Q2", "D1-Q3"]:
        summary = result["evidence_summary"][qid]
        assert summary["evidence_count"] == 3
        assert 1 in summary["stage_contributions"]
        assert 2 in summary["stage_contributions"]
        assert 3 in summary["stage_contributions"]
        print(
            f"✓ {qid}: Evidence from stages {sorted(summary['stage_contributions'].keys())}"
        )

    print("\n✅ Basic validation test passed")


def test_validation_insufficient_evidence():
    """Test detection of insufficient evidence"""
    print("\nTesting insufficient evidence detection...")

    registry = EvidenceRegistry()

    # D1-Q1: 3 evidence (sufficient)
    for stage in range(1, 4):
        provenance = EvidenceProvenance(
            detector_type="monetary",
            stage_number=stage,
            source_text_location={"page": 1, "line": 10},
            execution_timestamp="2024-01-01T00:00:00Z",
        )
        registry.register(
            source_component=f"detector_stage_{stage}",
            evidence_type="test_evidence",
            content={"value": stage},
            confidence=0.85,
            applicable_questions=["D1-Q1"],
            provenance=provenance,
        )

    # D1-Q2: 1 evidence (insufficient)
    provenance = EvidenceProvenance(
        detector_type="responsibility",
        stage_number=5,
        source_text_location={"page": 2, "line": 20},
        execution_timestamp="2024-01-01T01:00:00Z",
    )
    registry.register(
        source_component="detector_stage_5",
        evidence_type="test_evidence",
        content={"value": 1},
        confidence=0.80,
        applicable_questions=["D1-Q2"],
        provenance=provenance,
    )

    # Validate
    result = registry.validate_evidence_counts(
        all_question_ids=["D1-Q1", "D1-Q2"], min_evidence_threshold=3
    )

    assert result["valid"] == False
    assert result["total_questions"] == 2
    assert len(result["questions_below_threshold"]) == 1
    assert "D1-Q2" in result["questions_below_threshold"]

    print("✓ Correctly identified D1-Q2 as having insufficient evidence")
    print(
        f"  D1-Q1: {result['evidence_summary']['D1-Q1']['evidence_count']} evidence (sufficient)"
    )
    print(
        f"  D1-Q2: {result['evidence_summary']['D1-Q2']['evidence_count']} evidence (insufficient)"
    )

    # Check stage information for insufficient question
    d1q2_summary = result["evidence_summary"]["D1-Q2"]
    print(
        f"  D1-Q2 contributing stages: {sorted(d1q2_summary['stage_contributions'].keys())}"
    )
    print(f"  D1-Q2 missing stages: {d1q2_summary['missing_stages'][:5]}...")

    print("\n✅ Insufficient evidence detection test passed")


def test_validation_provenance_tracking():
    """Test provenance metadata tracking"""
    print("\nTesting provenance metadata tracking...")

    registry = EvidenceRegistry()

    provenance = EvidenceProvenance(
        detector_type="responsibility",
        stage_number=2,
        source_text_location={
            "page": 5,
            "line": 42,
            "char_start": 100,
            "char_end": 250,
        },
        execution_timestamp="2024-01-15T10:30:00Z",
        quality_metrics={"precision": 0.92, "recall": 0.88, "f1": 0.90},
    )

    registry.register(
        source_component="responsibility_detector",
        evidence_type="entity_detection",
        content={"entity": "Ministerio de Educación"},
        confidence=0.92,
        applicable_questions=["D1-Q1"],
        provenance=provenance,
    )

    # Validate
    result = registry.validate_evidence_counts(
        all_question_ids=["D1-Q1"], min_evidence_threshold=1
    )

    evidence_sources = result["evidence_summary"]["D1-Q1"]["evidence_sources"]
    assert len(evidence_sources) == 1

    source = evidence_sources[0]
    assert source["detector_type"] == "responsibility"
    assert source["stage_number"] == 2
    assert source["confidence"] == 0.92
    assert source["execution_timestamp"] == "2024-01-15T10:30:00Z"
    assert source["quality_metrics"]["f1"] == 0.90

    print("✓ Full provenance metadata tracked:")
    print(f"  Detector Type: {source['detector_type']}")
    print(f"  Stage Number: {source['stage_number']}")
    print(f"  Confidence: {source['confidence']:.2%}")
    print(f"  Execution Timestamp: {source['execution_timestamp']}")
    print(f"  Quality Metrics: F1={source['quality_metrics']['f1']:.2%}")

    print("\n✅ Provenance tracking test passed")


def test_validation_export():
    """Test validation result export"""
    print("\nTesting validation result export...")

    import json
    import os
    import tempfile

    registry = EvidenceRegistry()

    # Add evidence
    for stage in range(1, 4):
        provenance = EvidenceProvenance(
            detector_type="monetary",
            stage_number=stage,
            source_text_location={"page": stage},
            execution_timestamp="2024-01-01T00:00:00Z",
        )
        registry.register(
            source_component=f"detector_{stage}",
            evidence_type="test",
            content={},
            confidence=0.8,
            applicable_questions=["D1-Q1"],
            provenance=provenance,
        )

    # Validate
    result = registry.validate_evidence_counts(
        all_question_ids=["D1-Q1"], min_evidence_threshold=3
    )

    # Export
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    registry.export_validation_results(result, temp_path)

    # Verify
    assert os.path.exists(temp_path)

    with open(temp_path, "r") as f:
        loaded = json.load(f)

    assert loaded["total_questions"] == 1
    assert loaded["valid"] == True
    assert "evidence_summary" in loaded
    assert "stage_coverage_summary" in loaded

    print("✓ Validation results exported successfully")
    print(f"  File: {temp_path}")
    print(f"  Total Questions: {loaded['total_questions']}")
    print(f"  Valid: {loaded['valid']}")
    print(f"  Evidence Summary Keys: {list(loaded['evidence_summary'].keys())}")

    os.unlink(temp_path)

    print("\n✅ Export test passed")


if __name__ == "__main__":
    test_validation_basic()
    test_validation_insufficient_evidence()
    test_validation_provenance_tracking()
    test_validation_export()

    print("\n" + "=" * 70)
    print("All validation tests passed successfully!")
    print("=" * 70)
