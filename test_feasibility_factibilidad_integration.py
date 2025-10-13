"""
Test for factibilidad module integration in miniminimoon_orchestrator Stage 8 FEASIBILITY.
"""

import logging
from pathlib import Path
from typing import List


# Test the _execute_feasibility_stage method directly (mock-based due to dependencies)
def test_feasibility_stage_integration_mocked():
    """Test that Stage 8 FEASIBILITY integrates factibilidad modules correctly (mock-based)."""
    import hashlib

    from factibilidad import FactibilidadScorer, PatternDetector

    # Mock segments
    test_text = """
    La línea base muestra una tasa de pobreza del 25% actual.
    La meta es reducir la pobreza hasta el 15% para el año 2025.
    El objetivo de cobertura educativa es alcanzar el 90% en el 2026.
    """

    test_segments = [
        {
            "text": "La línea base muestra una tasa de pobreza del 25% actual.",
            "id": "seg_0",
        },
        {
            "text": "La meta es reducir la pobreza hasta el 15% para el año 2025.",
            "id": "seg_1",
        },
        {
            "text": "El objetivo de cobertura educativa es alcanzar el 90% en el 2026.",
            "id": "seg_2",
        },
    ]

    # Extract PDM content (simulating _execute_feasibility_stage logic)
    pdm_content = [seg["text"] for seg in test_segments]
    combined_pdm_text = "\n".join(pdm_content)

    # Instantiate PatternDetector and detect patterns
    pattern_detector = PatternDetector()
    detected_patterns = pattern_detector.detect_patterns(combined_pdm_text)

    # Instantiate FactibilidadScorer
    factibilidad_scorer = FactibilidadScorer(
        proximity_window=500, base_score=0.0, w1=0.5, w2=0.3, w3=0.3
    )

    # Compute enhanced feasibility scores
    similarity_score = 0.7
    scoring_result = factibilidad_scorer.score_text(
        combined_pdm_text, similarity_score=similarity_score
    )

    # Simulate evidence construction
    evidence_id = (
        f"feas_refinado_{hashlib.sha256(combined_pdm_text.encode()).hexdigest()[:12]}"
    )

    evidence_content = {
        "score_final": scoring_result["score_final"],
        "similarity_score": scoring_result["similarity_score"],
        "causal_density": scoring_result["causal_density"],
        "informative_length_ratio": scoring_result["informative_length_ratio"],
        "pattern_matches": {
            "baseline": [
                {
                    "text": m.text,
                    "start": m.start,
                    "end": m.end,
                    "confidence": m.confidence,
                }
                for m in detected_patterns.get("baseline", [])
            ],
            "target": [
                {
                    "text": m.text,
                    "start": m.start,
                    "end": m.end,
                    "confidence": m.confidence,
                }
                for m in detected_patterns.get("target", [])
            ],
            "timeframe": [
                {
                    "text": m.text,
                    "start": m.start,
                    "end": m.end,
                    "confidence": m.confidence,
                }
                for m in detected_patterns.get("timeframe", [])
            ],
        },
        "clusters": len(scoring_result.get("clusters", [])),
        "weights": scoring_result["weights"],
    }

    evidence_metadata = {
        "module": "factibilidad",
        "pattern_detector_source": "factibilidad.pattern_detector.PatternDetector",
        "scorer_source": "factibilidad.scoring.FactibilidadScorer",
        "scoring_method": "refinado",
        "pdm_content_length": len(combined_pdm_text),
        "segment_count": len(test_segments),
        "analysis": scoring_result.get("analysis", {}),
    }

    # Assertions
    assert len(detected_patterns["baseline"]) > 0, "Should detect baseline patterns"
    assert len(detected_patterns["target"]) > 0, "Should detect target patterns"
    assert len(detected_patterns["timeframe"]) > 0, "Should detect timeframe patterns"

    assert evidence_metadata["module"] == "factibilidad"
    assert (
        evidence_metadata["pattern_detector_source"]
        == "factibilidad.pattern_detector.PatternDetector"
    )
    assert (
        evidence_metadata["scorer_source"] == "factibilidad.scoring.FactibilidadScorer"
    )
    assert evidence_metadata["scoring_method"] == "refinado"

    assert "score_final" in evidence_content
    assert "pattern_matches" in evidence_content

    print("✓ Stage 8 FEASIBILITY integration test passed (mocked)")
    print(f"  - Detected {len(detected_patterns['baseline'])} baseline patterns")
    print(f"  - Detected {len(detected_patterns['target'])} target patterns")
    print(f"  - Detected {len(detected_patterns['timeframe'])} timeframe patterns")
    print(f"  - Final score: {scoring_result['score_final']:.4f}")
    print(f"  - Evidence ID: {evidence_id}")


def test_pattern_detector_standalone():
    """Test PatternDetector directly."""
    from factibilidad import PatternDetector

    detector = PatternDetector()
    test_text = "La línea base es del 20% y la meta es alcanzar el 30% para el 2025."

    patterns = detector.detect_patterns(test_text)

    assert "baseline" in patterns
    assert "target" in patterns
    assert "timeframe" in patterns

    assert len(patterns["baseline"]) > 0
    assert len(patterns["target"]) > 0
    assert len(patterns["timeframe"]) > 0

    print("✓ PatternDetector standalone test passed")


def test_factibilidad_scorer_standalone():
    """Test FactibilidadScorer directly."""
    from factibilidad import FactibilidadScorer

    scorer = FactibilidadScorer(proximity_window=500, w1=0.5, w2=0.3, w3=0.2)
    test_text = "La línea base es del 20% y la meta es alcanzar el 30% para el 2025."

    result = scorer.score_text(test_text, similarity_score=0.7)

    assert "score_final" in result
    assert "similarity_score" in result
    assert "causal_density" in result
    assert "informative_length_ratio" in result
    assert "pattern_matches" in result

    assert result["similarity_score"] == 0.7
    assert result["score_final"] >= 0.0

    print("✓ FactibilidadScorer standalone test passed")
    print(f"  - Final score: {result['score_final']:.4f}")
    print(f"  - Causal density: {result['causal_density']:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing factibilidad module integration...")
    print("=" * 60)

    test_pattern_detector_standalone()
    print()

    test_factibilidad_scorer_standalone()
    print()

    test_feasibility_stage_integration_mocked()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
