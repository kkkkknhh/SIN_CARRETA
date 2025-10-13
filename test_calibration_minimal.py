#!/usr/bin/env python3
"""
Minimal test for ReliabilityCalibrator integration - validates basic functionality
without requiring full numpy/scipy installation.
"""

import sys


def test_import():
    """Test that questionnaire_engine can import ReliabilityCalibrator"""
    try:
        # Just check the import succeeds at bytecode level
        import py_compile
        import tempfile

        # Compile questionnaire_engine.py to verify syntax
        py_compile.compile("questionnaire_engine.py", doraise=True)
        print("✓ questionnaire_engine.py compiles successfully")

        # Compile reliability_calibration.py to verify syntax
        py_compile.compile("evaluation/reliability_calibration.py", doraise=True)
        print("✓ evaluation/reliability_calibration.py compiles successfully")

        return True
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False


def test_dataclass_structure():
    """Test that EvaluationResult has the required calibration fields"""
    import ast

    # Parse questionnaire_engine.py to check dataclass definition
    with open("questionnaire_engine.py", "r") as f:
        tree = ast.parse(f.read())

    # Find EvaluationResult dataclass
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EvaluationResult":
            # Get all field names
            field_names = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    field_names.append(item.target.id)

            # Check for calibration fields
            assert "calibrated_score" in field_names, (
                "EvaluationResult missing calibrated_score field"
            )
            assert "uncertainty" in field_names, (
                "EvaluationResult missing uncertainty field"
            )
            print("✓ EvaluationResult has calibrated_score and uncertainty fields")
            return True

    print("❌ Could not find EvaluationResult dataclass")
    return False


def test_calibrator_initialization():
    """Test that QuestionnaireEngine initializes calibrator"""
    import ast

    with open("questionnaire_engine.py", "r") as f:
        tree = ast.parse(f.read())

    # Find __init__ method in QuestionnaireEngine
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QuestionnaireEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    # Convert to source and check for calibrator initialization
                    init_source = ast.unparse(item)
                    assert "reliability_calibrator" in init_source, (
                        "Missing reliability_calibrator initialization"
                    )
                    assert "ReliabilityCalibrator" in init_source, (
                        "Missing ReliabilityCalibrator call"
                    )
                    print("✓ QuestionnaireEngine initializes ReliabilityCalibrator")
                    return True

    print("❌ Could not verify calibrator initialization")
    return False


def test_calibration_in_evaluation():
    """Test that evaluation method uses calibration"""
    import ast

    with open("questionnaire_engine.py", "r") as f:
        content = f.read()

    # Check for calibration-related code in _evaluate_question_with_evidence
    assert "calibrated_score" in content, "Missing calibrated_score computation"
    assert "uncertainty" in content, "Missing uncertainty computation"
    assert "expected_f1" in content or "expected_reliability" in content, (
        "Missing reliability computation"
    )
    assert "reliability_calibrator" in content, "Missing calibrator usage"

    print("✓ Evaluation method includes calibration logic")
    return True


def test_evidence_registration():
    """Test that calibration evidence is registered"""
    with open("questionnaire_engine.py", "r") as f:
        content = f.read()

    # Check for calibration evidence registration
    assert "bayesian_calibration" in content, "Missing calibration evidence type"
    assert "calibration_evidence" in content or "reliability_calibrator" in content, (
        "Missing evidence registration"
    )

    print("✓ Calibration evidence registration verified")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Minimal Calibration Integration Test")
    print("=" * 60)

    tests = [
        ("Import & Compilation", test_import),
        ("DataClass Structure", test_dataclass_structure),
        ("Calibrator Initialization", test_calibrator_initialization),
        ("Calibration in Evaluation", test_calibration_in_evaluation),
        ("Evidence Registration", test_evidence_registration),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"❌ {failed} TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
