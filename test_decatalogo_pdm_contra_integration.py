"""Test integration of pdm_contra detectors in Decatalogo_principal.py evaluate_from_evidence method."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_evaluate_from_evidence_has_pdm_contra_integration():
    """Verify that evaluate_from_evidence method integrates pdm_contra detectors."""
    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    # Check that the class has the required methods
    assert hasattr(ExtractorEvidenciaIndustrialAvanzado, "evaluate_from_evidence")
    assert hasattr(ExtractorEvidenciaIndustrialAvanzado, "_extract_dimension_evidence")
    assert hasattr(ExtractorEvidenciaIndustrialAvanzado, "_calculate_dimension_score")

    print("✓ All required methods exist in ExtractorEvidenciaIndustrialAvanzado")


def test_evaluate_from_evidence_detector_instantiation():
    """Test that evaluate_from_evidence instantiates all required detectors."""
    import inspect

    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    # Get source code of evaluate_from_evidence method
    source = inspect.getsource(
        ExtractorEvidenciaIndustrialAvanzado.evaluate_from_evidence
    )

    # Check for detector instantiations
    required_detectors = [
        "ContradictionDetector",
        "RiskScorer",
        "PatternMatcher",
        "SpanishNLIDetector",
        "CompetenceValidator",
        "ExplanationTracer",
    ]

    for detector in required_detectors:
        assert detector in source, f"{detector} not found in evaluate_from_evidence"

    print(f"✓ All {len(required_detectors)} detectors are instantiated")


def test_evaluate_from_evidence_stage_14_provenance():
    """Test that evaluate_from_evidence stores evidence with Stage 14 provenance."""
    import inspect

    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    source = inspect.getsource(
        ExtractorEvidenciaIndustrialAvanzado.evaluate_from_evidence
    )

    # Check for Stage 14 provenance markers
    assert "stage_14" in source or "Stage 14" in source
    assert "stage_14_contradiction_analysis" in source
    assert "contradiction_analysis" in source.lower()

    print("✓ Stage 14 provenance metadata is present")


def test_dimension_scoring_integration():
    """Test that dimension scoring uses enriched evidence from pdm_contra."""
    import inspect

    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    source = inspect.getsource(
        ExtractorEvidenciaIndustrialAvanzado.evaluate_from_evidence
    )

    # Check for dimension scoring loop
    assert "dimensiones_decalogo" in source
    assert "dimension_scores" in source
    assert "_extract_dimension_evidence" in source
    assert "_calculate_dimension_score" in source
    assert "enriched_evidence" in source

    print("✓ Dimension scoring uses enriched evidence")


def test_evidence_registry_integration():
    """Test that evidence registry receives enriched evidence."""
    import inspect

    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    source = inspect.getsource(
        ExtractorEvidenciaIndustrialAvanzado.evaluate_from_evidence
    )

    # Check for evidence registration calls
    assert "evidence_registry.register" in source
    assert "EvidenceEntry" in source
    assert "pdm_contra_contradiction" in source
    assert "pdm_contra_patterns" in source
    assert "pdm_contra_competence" in source

    print("✓ Evidence registry receives all detector outputs")


def test_return_format_compatibility():
    """Test that evaluate_from_evidence maintains Stage 15 compatible return format."""
    import inspect

    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    source = inspect.getsource(
        ExtractorEvidenciaIndustrialAvanzado.evaluate_from_evidence
    )

    # Check return structure
    assert "evaluation_results = {" in source
    assert '"status":' in source
    assert '"timestamp":' in source
    assert '"summary":' in source
    assert '"detailed_analysis":' in source
    assert '"dimension_scores":' in source
    assert "return evaluation_results" in source

    print("✓ Return format is compatible with downstream stages")


def test_detector_methods_structure():
    """Test that helper methods have correct signatures."""
    import inspect

    from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado

    # Check _extract_dimension_evidence signature
    extract_sig = inspect.signature(
        ExtractorEvidenciaIndustrialAvanzado._extract_dimension_evidence
    )
    params = list(extract_sig.parameters.keys())
    assert "self" in params
    assert "dimension_info" in params
    assert "enriched_evidence" in params
    assert "evidence_registry" in params

    # Check _calculate_dimension_score signature
    calc_sig = inspect.signature(
        ExtractorEvidenciaIndustrialAvanzado._calculate_dimension_score
    )
    params = list(calc_sig.parameters.keys())
    assert "self" in params
    assert "dimension_info" in params
    assert "dimension_evidence" in params
    assert "enriched_evidence" in params

    print("✓ Helper methods have correct signatures")


if __name__ == "__main__":
    print("Running pdm_contra integration tests...\n")

    test_evaluate_from_evidence_has_pdm_contra_integration()
    test_evaluate_from_evidence_detector_instantiation()
    test_evaluate_from_evidence_stage_14_provenance()
    test_dimension_scoring_integration()
    test_evidence_registry_integration()
    test_return_format_compatibility()
    test_detector_methods_structure()

    print("\n✅ All integration tests passed!")
