"""Integration-focused diagnostics for the scoring subsystems.

These tests exercise the canonical pipeline ordering, feasibility scorer
normalization, questionnaire engine structural guarantees, reliability
calibration weighting, and the output quality assessor's comprehensive
checks. They are designed to detect regressions in the scoring
infrastructure described in the scoring system specification.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.reliability_calibration import (
    ReliabilityCalibrator,
    reliability_weighted_score,
)
from feasibility_scorer import FeasibilityScorer
from output_quality_assessor import validate_output_quality
from questionnaire_engine import QuestionnaireEngine


def test_canonical_pipeline_order_and_feasibility_position() -> None:
    """The canonical pipeline must keep feasibility scoring in position eight."""

    module_ast = ast.parse(Path("miniminimoon_orchestrator.py").read_text(encoding="utf-8"))
    pipeline_class = None
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineStage":
            pipeline_class = node
            break

    assert pipeline_class is not None, "PipelineStage definition must exist"

    stage_names = [
        stmt.targets[0].id
        for stmt in pipeline_class.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    ]

    canonical_prefix = [
        "SANITIZATION",
        "PLAN_PROCESSING",
        "SEGMENTATION",
        "EMBEDDING",
        "RESPONSIBILITY",
        "CONTRADICTION",
        "MONETARY",
        "FEASIBILITY",
        "CAUSAL",
        "TEORIA",
        "DAG",
    ]

    assert stage_names[: len(canonical_prefix)] == canonical_prefix
    assert stage_names[7] == "FEASIBILITY"


@pytest.fixture()
def feasibility_scorer() -> FeasibilityScorer:
    scorer = FeasibilityScorer()
    scorer.configure_parallel(enable_parallel=False)
    return scorer


def test_feasibility_scorer_outputs_are_normalized(feasibility_scorer: FeasibilityScorer) -> None:
    """Overall feasibility and indicator scores must stay within [0, 1]."""

    sample_text = (
        "Indicador estratégico: ampliar la cobertura de agua potable. "
        "Línea base: 65% de hogares atendidos en 2020 con unidad porcentual. "
        "Meta cuantificable: alcanzar el 85% para el año 2025. "
        "Responsable: Secretaría de Infraestructura municipal con presupuesto anual identificado."
    )

    result = feasibility_scorer.evaluate_plan_feasibility(sample_text)

    assert 0.0 <= result["overall_feasibility"] <= 1.0
    assert set(result["decalogo_answers"].keys()) == {"DE1_Q3", "DE4_Q1", "DE4_Q2"}
    assert result["indicators"], "Expected at least one extracted indicator"

    first_indicator = result["indicators"][0]
    assert 0.0 <= first_indicator["feasibility_score"] <= 1.0
    assert any(ind["has_baseline"] for ind in result["indicators"])
    assert any(ind["has_target"] for ind in result["indicators"])


def test_questionnaire_engine_structure_and_rubric_alignment() -> None:
    """The questionnaire engine must guarantee the 10×6×5 structure with 300 weights."""

    engine = QuestionnaireEngine()

    assert engine.structure.validate_structure() is True
    assert len(engine.base_questions) == 30
    assert len(engine.thematic_points) == 10

    generated_ids = set()
    for point_index, _ in enumerate(engine.thematic_points):
        for question in engine.base_questions:
            dimension_no = int(question.dimension[1:])
            base_position = ((question.question_no - 1) % 5) + 1
            sequential_question_no = (base_position - 1) * 10 + (point_index + 1)
            question_id = f"D{dimension_no}-Q{sequential_question_no}"
            generated_ids.add(question_id)

    assert len(generated_ids) == 300

    rubric_path = Path("rubric_scoring.json")
    assert rubric_path.exists(), "rubric_scoring.json must be present"

    rubric_data = json.loads(rubric_path.read_text(encoding="utf-8"))
    weights = rubric_data["weights"]
    assert len(weights) == 300

    expected_weight = pytest.approx(1.0 / 300.0, rel=1e-6)
    for question_id, weight in weights.items():
        assert question_id in generated_ids
        assert weight == expected_weight


def test_reliability_weighted_score_uses_expected_metrics() -> None:
    """Reliability weighting should multiply scores by the posterior expectation."""

    calibrator = ReliabilityCalibrator(detector_name="feasibility_detector")
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    calibrator.update(y_true, y_pred)

    raw_score = 0.8
    precision_weighted = reliability_weighted_score(raw_score, calibrator, metric="precision")
    f1_weighted = reliability_weighted_score(raw_score, calibrator, metric="f1")

    assert precision_weighted == pytest.approx(raw_score * calibrator.expected_precision)
    assert f1_weighted == pytest.approx(raw_score * calibrator.expected_f1)


def test_output_quality_assessor_detects_pipeline_gaps(tmp_path: Path) -> None:
    """The assessor must flag missing questions, stage coverage, and flow order issues."""

    answers_path = tmp_path / "answers.json"
    answers_payload = {
        "question_answers": [
            {"question_id": f"D1-Q{i}", "confidence": 0.5, "evidence_count": 0, "rationale": ""}
            for i in range(1, 299)
        ],
        "answers": [{"question_id": "D1-Q1"}],
    }
    answers_path.write_text(json.dumps(answers_payload), encoding="utf-8")

    rubric_path = tmp_path / "rubric.json"
    rubric_payload = {"weights": {"D1-Q1": 1.0}}
    rubric_path.write_text(json.dumps(rubric_payload), encoding="utf-8")

    evidence_registry_path = tmp_path / "evidence_registry.json"
    evidence_payload = {
        "evidences": [
            {"pipeline_stage": "sanitization"},
            {"pipeline_stage": "feasibility_scoring"},
        ]
    }
    evidence_registry_path.write_text(json.dumps(evidence_payload), encoding="utf-8")

    flow_runtime_path = tmp_path / "flow_runtime.json"
    flow_runtime_path.write_text(json.dumps({"stage_order": ["sanitization", "monetary_detection"]}), encoding="utf-8")

    flow_doc_path = tmp_path / "flow_doc.json"
    flow_doc_path.write_text(
        json.dumps({"canonical_order": ["sanitization", "plan_processing", "feasibility_scoring"]}),
        encoding="utf-8",
    )

    validation_gates_path = tmp_path / "validation_gates.json"
    validation_gates_path.write_text(
        json.dumps({
            "immutability_verified": "pass",
            "flow_order_match": "fail",
            "evidence_deterministic_hash_consistency": "fail",
            "coverage_300_300": "fail",
            "rubric_alignment": "fail",
            "triple_run_determinism": "fail",
        }),
        encoding="utf-8",
    )

    results = validate_output_quality(
        answers_path=str(answers_path),
        rubric_path=str(rubric_path),
        evidence_registry_path=str(evidence_registry_path),
        flow_runtime_path=str(flow_runtime_path),
        flow_doc_path=str(flow_doc_path),
        validation_gates_path=str(validation_gates_path),
        output_path=str(tmp_path / "report.json"),
    )

    question_count = results["criteria"]["question_count"]
    assert question_count["pass"] is False
    assert question_count["actual"] == 298

    pipeline_coverage = results["criteria"]["pipeline_stage_coverage"]
    assert pipeline_coverage["pass"] is False
    assert pipeline_coverage["actual_stages"] == 2

    flow_order = results["criteria"]["flow_order_match"]
    assert flow_order["pass"] is False
    assert flow_order["deviations"], "Expected deviations when orders differ"

    validation_gates = results["criteria"]["validation_gates"]
    assert validation_gates["pass"] is False
    assert validation_gates["passing_gates"] == 1
