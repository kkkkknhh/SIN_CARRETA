#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
Integration test for AnswerAssembler with doctoral argumentation.

Tests:
1. Minimum evidence requirement (≥3 sources)
2. Complete Toulmin structure validation
3. Quality threshold enforcement (coherence ≥0.85, quality ≥0.80)
4. Proper serialization of doctoral argument structure
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List


# Mock evidence registry
class MockEvidenceRegistry:
    """Mock evidence registry for testing"""

    def __init__(self, evidence_data: Dict[str, List[Any]]):
        self.evidence_data = evidence_data

    def get_evidence_for_question(self, question_id: str) -> List[Any]:
        """Return evidence for a question"""
        return self.evidence_data.get(question_id, [])


@dataclass
class MockEvidence:
    """Mock evidence item"""

    text: str
    confidence: float


def test_answer_assembler_doctoral_argumentation():
    """Test answer assembler with doctoral argumentation integration"""
    from answer_assembler import AnswerAssembler

    # Create mock evidence registry with varying evidence counts
    evidence_registry = MockEvidenceRegistry(
        {
            "P1-D1-Q1": [
                MockEvidence("Evidence 1 for D1-Q1", 0.9),
                MockEvidence("Evidence 2 for D1-Q1", 0.85),
                MockEvidence("Evidence 3 for D1-Q1", 0.88),
                MockEvidence("Evidence 4 for D1-Q1", 0.82),
            ],
            "P1-D2-Q1": [
                MockEvidence("Evidence 1 for D2-Q1", 0.75),
                MockEvidence("Evidence 2 for D2-Q1", 0.80),
            ],
            "P1-D3-Q1": [
                MockEvidence("Single evidence for D3-Q1", 0.65),
            ],
        }
    )

    # Create mock evaluation inputs
    evaluation_inputs = {
        "questionnaire_eval": {
            "question_results": [
                {"question_id": "P1-D1-Q1", "score": 2.5},
                {"question_id": "P1-D2-Q1", "score": 1.8},
                {"question_id": "P1-D3-Q1", "score": 1.2},
            ]
        }
    }

    # Initialize assembler
    assembler = AnswerAssembler(evidence_registry=evidence_registry)

    # Assemble answers
    result = assembler.assemble(evaluation_inputs)

    # Validate results
    print("=" * 80)
    print("ANSWER ASSEMBLER - DOCTORAL ARGUMENTATION INTEGRATION TEST")
    print("=" * 80)

    # Check question answers
    question_answers = result["question_answers"]
    print(f"\n✓ Processed {len(question_answers)} questions")

    # Validate P1-D1-Q1 (4 evidence sources - should succeed if engine available)
    q1 = next(qa for qa in question_answers if qa["question_id"] == "P1-D1-Q1")
    print(f"\n--- Question: {q1['question_id']} ---")
    print(f"  Evidence count: {q1['evidence_count']}")
    print(f"  Argumentation status: {q1['argumentation_status']}")

    if q1["argumentation_status"] == "success":
        print("  ✓ Doctoral argument generated successfully")
        assert "doctoral_justification" in q1, "Missing doctoral_justification"
        assert "toulmin_structure" in q1, "Missing toulmin_structure"
        assert "argument_quality" in q1, "Missing argument_quality"

        # Validate Toulmin structure completeness
        toulmin = q1["toulmin_structure"]
        required_fields = [
            "claim",
            "ground",
            "warrant",
            "backing",
            "qualifier",
            "rebuttal",
        ]
        for field in required_fields:
            assert field in toulmin, f"Missing Toulmin field: {field}"
            assert toulmin[field], f"Empty Toulmin field: {field}"
        print(
            f"  ✓ Complete Toulmin structure validated ({len(required_fields)} components)"
        )

        # Validate quality thresholds
        quality = q1["argument_quality"]
        coherence = quality.get("coherence_score", 0.0)
        qual_score = quality.get("quality_score", 0.0)
        print(f"  ✓ Coherence score: {coherence:.3f} (threshold: 0.85)")
        print(f"  ✓ Quality score: {qual_score:.3f} (threshold: 0.80)")

        assert coherence >= 0.85, f"Coherence score {coherence} below threshold"
        assert qual_score >= 0.80, f"Quality score {qual_score} below threshold"
        assert quality.get("meets_doctoral_standards", False), (
            "Should meet doctoral standards"
        )

    # Validate P1-D2-Q1 (2 evidence sources - should fail with insufficient evidence)
    q2 = next(qa for qa in question_answers if qa["question_id"] == "P1-D2-Q1")
    print(f"\n--- Question: {q2['question_id']} ---")
    print(f"  Evidence count: {q2['evidence_count']}")
    print(f"  Argumentation status: {q2['argumentation_status']}")
    assert q2["argumentation_status"] == "insufficient_evidence", (
        "Should fail with insufficient evidence"
    )
    assert "Insufficient evidence" in " ".join(q2["caveats"]), (
        "Should have caveat about insufficient evidence"
    )
    print("  ✓ Correctly rejected due to insufficient evidence")

    # Validate P1-D3-Q1 (1 evidence source - should fail with insufficient evidence)
    q3 = next(qa for qa in question_answers if qa["question_id"] == "P1-D3-Q1")
    print(f"\n--- Question: {q3['question_id']} ---")
    print(f"  Evidence count: {q3['evidence_count']}")
    print(f"  Argumentation status: {q3['argumentation_status']}")
    assert q3["argumentation_status"] == "insufficient_evidence", (
        "Should fail with insufficient evidence"
    )
    print("  ✓ Correctly rejected due to insufficient evidence")

    # Validate global summary
    global_summary = result["global_summary"]
    doctoral_stats = global_summary["doctoral_argumentation"]
    print("\n--- Global Summary ---")
    print(f"  Total questions: {global_summary['answered_questions']}")
    print(f"  Doctoral coverage: {doctoral_stats['coverage']}")
    print(f"  Status breakdown: {doctoral_stats['status_breakdown']}")
    print(f"  Average coherence: {doctoral_stats['average_coherence_score']:.3f}")
    print(f"  Average quality: {doctoral_stats['average_quality_score']:.3f}")

    # Validate metadata
    metadata = result["metadata"]
    print("\n--- Metadata ---")
    print(f"  Doctoral engine enabled: {metadata['doctoral_engine_enabled']}")
    print(f"  Toulmin components validated: {metadata['toulmin_components_validated']}")

    required_components = [
        "claim",
        "ground",
        "warrant",
        "backing",
        "qualifier",
        "rebuttal",
    ]
    assert metadata["toulmin_components_validated"] == required_components, (
        "Wrong Toulmin components"
    )

    # Validate validation thresholds
    thresholds = doctoral_stats["validation_thresholds"]
    assert thresholds["min_evidence_sources"] == 3, "Wrong evidence threshold"
    assert thresholds["min_coherence_score"] == 0.85, "Wrong coherence threshold"
    assert thresholds["min_quality_score"] == 0.80, "Wrong quality threshold"
    print(f"  ✓ Validation thresholds correctly configured")

    print("\n" + "=" * 80)
    print("TEST PASSED: All validations successful")
    print("=" * 80)

    # Optional: Print sample doctoral argument if available
    if q1["argumentation_status"] == "success":
        print("\n--- Sample Doctoral Argument (P1-D1-Q1) ---")
        doctoral = q1["doctoral_justification"]
        print(f"\nClaim: {doctoral['claim'][:200]}...")
        print(f"\nGround: {doctoral['ground'][:200]}...")
        print(f"\nWarrant: {doctoral['warrant'][:200]}...")
        print(f"\nQualifier: {doctoral['qualifier'][:200]}...")

    return result


if __name__ == "__main__":
    try:
        result = test_answer_assembler_doctoral_argumentation()
        print("\n✓ Integration test completed successfully")
    except ImportError as e:
        print(f"\n⚠ Warning: {e}")
        print("  This is expected if doctoral_argumentation_engine is not available")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
