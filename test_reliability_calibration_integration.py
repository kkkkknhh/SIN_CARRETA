#!/usr/bin/env python3
"""
Test suite for ReliabilityCalibrator integration in questionnaire_engine.py
Stage 15 QUESTIONNAIRE_EVAL validation
"""

import sys
from dataclasses import asdict

from questionnaire_engine import (
    QuestionnaireEngine,
    BaseQuestion,
    ThematicPoint,
    SearchPattern,
    ScoringRule,
    ScoringModality,
)


def test_calibrator_initialization():
    """Test that QuestionnaireEngine initializes ReliabilityCalibrator"""
    engine = QuestionnaireEngine()

    # Verify calibrator exists
    assert hasattr(
        engine, "reliability_calibrator"
    ), "Engine should have reliability_calibrator attribute"

    # Verify calibrator configuration
    calibrator = engine.reliability_calibrator
    assert calibrator.detector_name == "questionnaire_evaluator"
    assert calibrator.precision_a == 5.0
    assert calibrator.precision_b == 1.0
    assert calibrator.recall_a == 5.0
    assert calibrator.recall_b == 1.0

    print("✓ ReliabilityCalibrator initialization verified")


def test_calibrated_score_in_evaluation():
    """Test that evaluation results include calibrated_score and uncertainty"""

    # Create a simple test case
    engine = QuestionnaireEngine()

    # Create a mock thematic point
    point = ThematicPoint(
        id="P1", title="Test Point", keywords=[], hints=[], relevant_programs=[]
    )

    # Create a simple base question
    base_q = BaseQuestion(
        id="D1-Q1",
        dimension="D1",
        question_no=1,
        template="Test question for {PUNTO_TEMATICO}?",
        search_patterns={
            "test_element": SearchPattern(
                pattern_type="regex", pattern=r"test", description="Test pattern"
            )
        },
        scoring_rule=ScoringRule(
            modality=ScoringModality.TYPE_A, formula="(found/1) × 3"
        ),
        expected_elements=["test_element"],
    )

    # Create mock evidence list
    evidence_list = []

    # Evaluate question
    result = engine._evaluate_question_with_evidence(base_q, point, evidence_list)

    # Verify calibration fields are present
    assert hasattr(
        result, "calibrated_score"
    ), "EvaluationResult should have calibrated_score"
    assert hasattr(result, "uncertainty"), "EvaluationResult should have uncertainty"

    # Verify calibration was applied
    assert (
        result.calibrated_score is not None
    ), "calibrated_score should not be None after evaluation"
    assert (
        result.uncertainty is not None
    ), "uncertainty should not be None after evaluation"

    # Verify calibrated score is different from raw score (unless reliability is 1.0)
    calibrator = engine.reliability_calibrator
    expected_f1 = calibrator.expected_f1
    if expected_f1 != 1.0:
        assert result.score != result.calibrated_score, (
            f"Calibrated score should differ from raw score when F1={expected_f1:.3f}. "
            f"Raw: {result.score}, Calibrated: {result.calibrated_score}"
        )

    # Verify uncertainty is computed
    assert result.uncertainty >= 0.0, "Uncertainty should be non-negative"
    assert result.uncertainty <= 1.0, "Uncertainty should be <= 1.0"

    print(f"✓ Calibration applied: Raw={result.score:.2f}, Calibrated={result.calibrated_score:.2f}, Uncertainty=±{result.uncertainty:.3f}")


def test_calibration_evidence_registration():
    """Test that calibration evidence is registered in question evaluation"""

    engine = QuestionnaireEngine()

    point = ThematicPoint(
        id="P1", title="Test Point", keywords=[], hints=[], relevant_programs=[]
    )

    base_q = BaseQuestion(
        id="D1-Q1",
        dimension="D1",
        question_no=1,
        template="Test question for {PUNTO_TEMATICO}?",
        search_patterns={
            "test_element": SearchPattern(
                pattern_type="regex", pattern=r"test", description="Test pattern"
            )
        },
        scoring_rule=ScoringRule(
            modality=ScoringModality.TYPE_A, formula="(found/1) × 3"
        ),
        expected_elements=["test_element"],
    )

    evidence_list = []
    result = engine._evaluate_question_with_evidence(base_q, point, evidence_list)

    # Verify calibration evidence is in evidence list
    calibration_evidence_found = False
    for evidence in result.evidence:
        if evidence.get("source") == "reliability_calibrator":
            calibration_evidence_found = True
            assert (
                evidence.get("type") == "bayesian_calibration"
            ), "Evidence type should be bayesian_calibration"
            assert "confidence" in evidence, "Evidence should include confidence"
            assert (
                "content_summary" in evidence
            ), "Evidence should include content_summary"
            # Verify content summary includes key metrics
            summary = evidence["content_summary"]
            assert "Raw score" in summary, "Summary should mention raw score"
            assert "Calibrated" in summary, "Summary should mention calibrated score"
            assert "Uncertainty" in summary, "Summary should mention uncertainty"
            assert "F1" in summary, "Summary should mention F1 score"

    assert (
        calibration_evidence_found
    ), "Calibration evidence should be registered in evidence list"

    print("✓ Calibration evidence registration verified")


def test_300_questions_calibrated():
    """Test that all 300 questions receive calibration treatment"""

    # Create mock evidence registry
    from questionnaire_engine import EvidenceRegistry

    evidence_registry = EvidenceRegistry()
    evidence_registry.freeze()

    engine = QuestionnaireEngine(evidence_registry=evidence_registry)

    # Execute full evaluation with evidence registry
    results = engine.execute_full_evaluation_parallel(
        evidence_registry=evidence_registry,
        municipality="Test Municipality",
        department="Test Department",
    )

    # Verify 300 questions were evaluated
    all_questions = results["results"]["all_questions"]
    assert (
        len(all_questions) == 300
    ), f"Should have 300 questions, got {len(all_questions)}"

    # Verify all have calibration
    calibrated_count = 0
    for q in all_questions:
        assert (
            "calibrated_score" in q
        ), f"Question {q['question_id']} missing calibrated_score"
        assert "uncertainty" in q, f"Question {q['question_id']} missing uncertainty"

        if q["calibrated_score"] is not None:
            calibrated_count += 1

        # Verify calibration evidence in each question
        has_calibration_evidence = any(
            e.get("source") == "reliability_calibrator" for e in q.get("evidence", [])
        )
        assert (
            has_calibration_evidence
        ), f"Question {q['question_id']} missing calibration evidence"

    assert (
        calibrated_count == 300
    ), f"All 300 questions should be calibrated, got {calibrated_count}"

    print(f"✓ All 300 questions calibrated with uncertainty quantification")


def test_calculation_detail_includes_calibration():
    """Test that calculation_detail field includes calibration information"""

    engine = QuestionnaireEngine()

    point = ThematicPoint(
        id="P1", title="Test Point", keywords=[], hints=[], relevant_programs=[]
    )

    base_q = BaseQuestion(
        id="D1-Q1",
        dimension="D1",
        question_no=1,
        template="Test question for {PUNTO_TEMATICO}?",
        search_patterns={
            "test_element": SearchPattern(
                pattern_type="regex", pattern=r"test", description="Test pattern"
            )
        },
        scoring_rule=ScoringRule(
            modality=ScoringModality.TYPE_A, formula="(found/1) × 3"
        ),
        expected_elements=["test_element"],
    )

    evidence_list = []
    result = engine._evaluate_question_with_evidence(base_q, point, evidence_list)

    # Verify calculation_detail includes calibration info
    detail = result.calculation_detail
    assert "Raw:" in detail, "calculation_detail should mention raw score"
    assert "Calibrated:" in detail, "calculation_detail should mention calibrated score"
    assert "±" in detail, "calculation_detail should include uncertainty notation"

    print(f"✓ Calculation detail includes calibration: '{detail}'")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ReliabilityCalibrator Integration in Stage 15")
    print("=" * 60)

    try:
        test_calibrator_initialization()
        test_calibrated_score_in_evaluation()
        test_calibration_evidence_registration()
        test_300_questions_calibrated()
        test_calculation_detail_includes_calibration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
