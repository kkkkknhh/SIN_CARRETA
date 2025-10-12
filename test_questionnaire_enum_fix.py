#!/usr/bin/env python3
"""
Test to verify the ScoreBand Enum and QuestionnaireEngine initialization fixes.
This test validates that the issues mentioned in the problem statement are resolved.
"""

import ast
import sys
from pathlib import Path


def test_scoreband_enum_structure():
    """Verify ScoreBand Enum uses @property pattern instead of custom __init__."""
    print("\n[TEST 1] Checking ScoreBand Enum structure...")

    # Parse the questionnaire_engine.py file
    qe_path = Path(__file__).parent / "questionnaire_engine.py"
    with open(qe_path, "r") as f:
        tree = ast.parse(f.read(), filename="questionnaire_engine.py")

    # Find ScoreBand class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ScoreBand":
            has_custom_init = False
            properties = []

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        has_custom_init = True

                    # Check for @property decorator
                    if any(
                        isinstance(dec, ast.Name) and dec.id == "property"
                        for dec in item.decorator_list
                    ):
                        properties.append(item.name)

            assert not has_custom_init, (
                "ScoreBand should NOT have custom __init__ (causes Enum instantiation error)"
            )

            expected_properties = ["min_score", "max_score", "color", "description"]
            for prop in expected_properties:
                assert prop in properties, f"ScoreBand missing @property {prop}"

            print("    ✓ ScoreBand uses @property pattern correctly")
            print(f"    ✓ Properties: {', '.join(properties)}")
            return True

    raise AssertionError("ScoreBand class not found")


def test_questionnaire_engine_init_signature():
    """Verify QuestionnaireEngine.__init__ accepts correct parameters."""
    print("\n[TEST 2] Checking QuestionnaireEngine.__init__ signature...")

    # Parse the questionnaire_engine.py file
    qe_path = Path(__file__).parent / "questionnaire_engine.py"
    with open(qe_path, "r") as f:
        tree = ast.parse(f.read(), filename="questionnaire_engine.py")

    # Find QuestionnaireEngine class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QuestionnaireEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    params = [arg.arg for arg in item.args.args]
                    defaults_count = len(item.args.defaults)

                    assert "self" in params, "__init__ must have 'self' parameter"
                    assert "evidence_registry" in params, (
                        "__init__ must accept 'evidence_registry' parameter"
                    )
                    assert "rubric_path" in params, (
                        "__init__ must accept 'rubric_path' parameter"
                    )

                    # Both parameters should have defaults (None)
                    assert defaults_count == 2, (
                        "evidence_registry and rubric_path should have default values"
                    )

                    print(f"    ✓ Signature: def __init__({', '.join(params)})")
                    print(f"    ✓ {defaults_count} parameters have defaults")
                    return True

            raise AssertionError("__init__ method not found in QuestionnaireEngine")

    raise AssertionError("QuestionnaireEngine class not found")


def test_orchestrator_call_compatibility():
    """Verify orchestrator calls QuestionnaireEngine with correct arguments."""
    print("\n[TEST 3] Checking orchestrator instantiation compatibility...")

    # Parse the orchestrator file
    orch_path = Path(__file__).parent / "miniminimoon_orchestrator.py"
    with open(orch_path, "r") as f:
        tree = ast.parse(f.read(), filename="miniminimoon_orchestrator.py")

    # Find QuestionnaireEngine instantiation
    found_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "QuestionnaireEngine"
            ):
                keyword_args = [kw.arg for kw in node.keywords]

                # The orchestrator should pass evidence_registry and rubric_path as keyword args
                assert "evidence_registry" in keyword_args, (
                    "Orchestrator should pass evidence_registry to QuestionnaireEngine"
                )
                assert "rubric_path" in keyword_args, (
                    "Orchestrator should pass rubric_path to QuestionnaireEngine"
                )

                print(
                    f"    ✓ Orchestrator calls with keyword args: {', '.join(keyword_args)}"
                )
                found_call = True

    assert found_call, "QuestionnaireEngine instantiation not found in orchestrator"
    return True


def test_syntax_validity():
    """Verify Python syntax is valid in modified files."""
    print("\n[TEST 4] Checking Python syntax validity...")

    import py_compile

    files = [
        Path(__file__).parent / "questionnaire_engine.py",
        Path(__file__).parent / "miniminimoon_orchestrator.py",
    ]

    for file_path in files:
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"    ✓ {file_path.name} has valid syntax")
        except py_compile.PyCompileError as e:
            raise AssertionError(f"Syntax error in {file_path.name}: {e}")

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING: ScoreBand Enum and QuestionnaireEngine initialization fixes")
    print("=" * 80)

    tests = [
        test_scoreband_enum_structure,
        test_questionnaire_engine_init_signature,
        test_orchestrator_call_compatibility,
        test_syntax_validity,
    ]

    try:
        for test in tests:
            test()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return 0

    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
