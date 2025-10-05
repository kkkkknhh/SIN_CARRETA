#!/usr/bin/env python3
"""
Test script for zero evidence support functionality in feasibility_scorer.py
"""

import logging

from feasibility_scorer import FeasibilityScorer

# Set up logging to see the override messages
logging.basicConfig(level=logging.INFO)


def test_zero_evidence_support():
    """Test that zero evidencia_soporte overrides normal scoring logic."""
    scorer = FeasibilityScorer()
    text = "línea base 50% meta 80% año 2025 responsable Secretaría"

    print("Testing zero evidence support override...")
    print("=" * 50)

    # Test normal scoring
    print("\n1. Normal scoring (no evidencia_soporte parameter):")
    normal_result = scorer.calculate_feasibility_score(text)
    print(f"   Score: {normal_result.feasibility_score:.2f}")
    print(f"   Quality tier: {normal_result.quality_tier}")
    print(f"   Components detected: {len(normal_result.components_detected)}")

    # Test with zero evidence support
    print("\n2. With zero evidence support (evidencia_soporte=0):")
    zero_evidence_result = scorer.calculate_feasibility_score(
        text, evidencia_soporte=0)
    print(f"   Score: {zero_evidence_result.feasibility_score:.2f}")
    print(f"   Quality tier: {zero_evidence_result.quality_tier}")
    print(
        f"   Components detected: {len(zero_evidence_result.components_detected)}")
    print("   Risk level: HIGH (due to zero evidence)")

    # Test with non-zero evidence support
    print("\n3. With non-zero evidence support (evidencia_soporte=1):")
    normal_result_with_evidence = scorer.calculate_feasibility_score(
        text, evidencia_soporte=1
    )
    print(f"   Score: {normal_result_with_evidence.feasibility_score:.2f}")
    print(f"   Quality tier: {normal_result_with_evidence.quality_tier}")
    print(
        f"   Components detected: {len(normal_result_with_evidence.components_detected)}"
    )

    # Verify expectations
    assert normal_result.feasibility_score > 0.0, (
        "Normal scoring should have positive score"
    )
    assert normal_result.quality_tier != "REQUIERE MAYOR EVIDENCIA", (
        "Normal scoring should not require more evidence"
    )

    assert zero_evidence_result.feasibility_score == 0.0, (
        "Zero evidence should result in zero score"
    )
    assert zero_evidence_result.quality_tier == "REQUIERE MAYOR EVIDENCIA", (
        "Zero evidence should require more evidence"
    )
    assert len(zero_evidence_result.components_detected) == 0, (
        "Zero evidence should detect no components"
    )

    assert normal_result_with_evidence.feasibility_score > 0.0, (
        "Non-zero evidence should score normally"
    )
    assert normal_result_with_evidence.quality_tier != "REQUIERE MAYOR EVIDENCIA", (
        "Non-zero evidence should not require more evidence"
    )

    print("\n✓ All assertions passed!")


def test_batch_scoring_with_evidence():
    """Test batch scoring with evidencia_soporte values."""
    scorer = FeasibilityScorer()
    indicators = [
        "línea base 50% meta 80% año 2025",
        "situación actual mejorar objetivo",
        "aumentar servicios región",
    ]
    evidencia_list = [0, 1, 2]  # First one has zero evidence

    print("\n\nTesting batch scoring with evidence support...")
    print("=" * 50)

    results = scorer.batch_score(
        indicators, evidencia_soporte_list=evidencia_list)

    for i, (indicator, evidencia, result) in enumerate(
        zip(indicators, evidencia_list, results)
    ):
        print(f"\nIndicator {i + 1} (evidencia_soporte={evidencia}):")
        print(f"   Text: '{indicator}'")
        print(f"   Score: {result.feasibility_score:.2f}")
        print(f"   Quality tier: {result.quality_tier}")

    # Verify first result is overridden due to zero evidence
    assert results[0].feasibility_score == 0.0, "First result should have zero score"
    assert results[0].quality_tier == "REQUIERE MAYOR EVIDENCIA", (
        "First result should require more evidence"
    )

    # Other results should score normally (though may still be low due to missing components)
    assert results[1].quality_tier != "REQUIERE MAYOR EVIDENCIA", (
        "Second result should not require more evidence"
    )
    assert results[2].quality_tier != "REQUIERE MAYOR EVIDENCIA", (
        "Third result should not require more evidence"
    )

    print("\n✓ Batch scoring test passed!")


if __name__ == "__main__":
    test_zero_evidence_support()
    test_batch_scoring_with_evidence()
    print("\n🎉 All tests completed successfully!")
    print("The zero evidence support override is working correctly.")
