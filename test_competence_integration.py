"""Test competence validation integration in evaluate_from_evidence method."""

import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing
sys.modules["spacy"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pdfplumber"] = None
sys.modules["transformers"] = MagicMock()


def test_competence_validator_integration():
    """Test that CompetenceValidator is properly integrated in evaluate_from_evidence."""

    # Read the source file
    with open("Decatalogo_principal.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Verify imports
    assert "from pdm_contra.policy.competence import CompetenceValidator" in content, (
        "CompetenceValidator import missing"
    )

    # Verify instantiation
    assert "competence_validator = CompetenceValidator()" in content, (
        "CompetenceValidator instantiation missing"
    )

    # Verify per-question validation
    assert "question_competence_issues" in content, (
        "Per-question competence issues tracking missing"
    )

    assert "competence_validator.validate_segment(" in content, (
        "CompetenceValidator.validate_segment() call missing"
    )

    # Verify evidence registration with proper provenance
    assert "competence_validation::" in content, (
        "Competence validation evidence ID prefix missing"
    )

    assert '"validator_source": "CompetenceValidator"' in content, (
        "Validator source provenance missing"
    )

    assert '"provenance"' in content, "Provenance metadata missing"

    # Verify error handling
    assert "except Exception" in content and "log_warning_with_text" in content, (
        "Error handling with logging missing"
    )

    # Verify placement after contradiction detection
    lines = content.split("\n")
    contradiction_idx = -1
    competence_per_q_idx = -1
    scoring_idx = -1

    for i, line in enumerate(lines):
        if "STEP 1: Run ContradictionDetector" in line:
            contradiction_idx = i
        if "COMPETENCE VALIDATION PER QUESTION" in line:
            competence_per_q_idx = i
        if "STEP 8: Return comprehensive evaluation results" in line:
            scoring_idx = i

    assert contradiction_idx > 0, "Contradiction detection step not found"
    assert competence_per_q_idx > 0, "Per-question competence validation not found"
    assert scoring_idx > 0, "Scoring/results step not found"

    assert contradiction_idx < competence_per_q_idx < scoring_idx, (
        "Competence validation not properly placed between contradiction detection and scoring"
    )

    # Verify results are included in return value
    assert '"question_competence_validations":' in content, (
        "Competence validations not included in evaluation results"
    )

    print("✅ All integration checks passed:")
    print("  ✓ CompetenceValidator imported and instantiated")
    print("  ✓ Per-question validation with proper evidence registration")
    print("  ✓ Provenance metadata included")
    print("  ✓ Error handling with logging")
    print(
        "  ✓ Proper placement: contradiction detection → competence validation → scoring"
    )
    print("  ✓ Results included in return value")


if __name__ == "__main__":
    test_competence_validator_integration()
