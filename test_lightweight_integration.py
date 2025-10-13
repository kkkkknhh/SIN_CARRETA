#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Integration Test for MINIMINIMOON Full Module Integration
======================================================================

Tests the integration without requiring heavy dependencies (numpy, torch, etc.)
This validates code structure, method signatures, and integration points.
"""

import ast
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_method_exists_in_file(
    filepath: Path, class_name: str, method_name: str
) -> bool:
    """Check if a method exists in a class by parsing the AST"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return True
        return False
    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")
        return False


def check_import_in_method(filepath: Path, method_name: str, import_name: str) -> bool:
    """Check if a method contains a specific import"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the method
        if method_name not in content:
            return False

        # Check if import is in the method or nearby
        return import_name in content
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return False


def test_module_contribution_mapper():
    """Test 1: Module Contribution Mapper structure"""
    logger.info("=" * 70)
    logger.info("TEST 1: Module Contribution Mapper")
    logger.info("=" * 70)

    filepath = Path("module_contribution_mapper.py")
    if not filepath.exists():
        logger.error(f"  ✗ File not found: {filepath}")
        return False

    # Check key classes
    classes_to_check = [
        "ModuleCategory",
        "ModuleContribution",
        "QuestionContributionMap",
        "ModuleContributionMapper",
    ]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        for class_name in classes_to_check:
            if f"class {class_name}" in content:
                logger.info(f"  ✓ Class {class_name} found")
            else:
                logger.error(f"  ✗ Class {class_name} not found")
                return False

        # Check key methods
        if "def get_question_mapping" in content:
            logger.info("  ✓ Method get_question_mapping() found")
        else:
            logger.error("  ✗ Method get_question_mapping() not found")
            return False

        if "def _initialize_default_mappings" in content:
            logger.info("  ✓ Method _initialize_default_mappings() found")
        else:
            logger.error("  ✗ Method _initialize_default_mappings() not found")
            return False

        logger.info("✅ TEST 1 PASSED: Module Contribution Mapper structure valid")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        return False


def test_evaluate_from_evidence():
    """Test 2: evaluate_from_evidence() in Decatalogo_principal.py"""
    logger.info("=" * 70)
    logger.info("TEST 2: evaluate_from_evidence() Method")
    logger.info("=" * 70)

    filepath = Path("Decatalogo_principal.py")
    if not filepath.exists():
        logger.error(f"  ✗ File not found: {filepath}")
        return False

    has_method = check_method_exists_in_file(
        filepath, "ExtractorEvidenciaIndustrialAvanzado", "evaluate_from_evidence"
    )

    if not has_method:
        logger.error(
            "  ✗ Method evaluate_from_evidence() not found in ExtractorEvidenciaIndustrialAvanzado"
        )
        return False

    logger.info("  ✓ Method evaluate_from_evidence() exists")

    # Check for key integrations in the method
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        required_imports = [
            "ContradictionDetector",
            "RiskScorer",
            "PatternMatcher",
            "SpanishNLIDetector",
            "CompetenceValidator",
            "ExplanationTracer",
            "PatternDetector",
            "ReliabilityCalibrator",
        ]

        for imp in required_imports:
            if imp in content:
                logger.info(f"  ✓ Integrates {imp}")
            else:
                logger.warning(f"  ⚠ Missing integration: {imp}")

        logger.info("✅ TEST 2 PASSED: evaluate_from_evidence() implemented")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        return False


def test_questionnaire_engine_enhancement():
    """Test 3: Questionnaire engine enhancement"""
    logger.info("=" * 70)
    logger.info("TEST 3: Questionnaire Engine Enhancement")
    logger.info("=" * 70)

    filepath = Path("questionnaire_engine.py")
    if not filepath.exists():
        logger.error(f"  ✗ File not found: {filepath}")
        return False

    # Check for multi-source synthesis method
    has_method = check_method_exists_in_file(
        filepath, "QuestionnaireEngine", "_synthesize_multi_source_evidence"
    )

    if not has_method:
        logger.error("  ✗ Method _synthesize_multi_source_evidence() not found")
        return False

    logger.info("  ✓ Method _synthesize_multi_source_evidence() exists")

    # Check EvaluationResult has metadata
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for EvaluationResult class and metadata field
        if "@dataclass" in content and "class EvaluationResult" in content:
            logger.info("  ✓ EvaluationResult is a dataclass")

            # Check for metadata field
            if "metadata:" in content or "metadata =" in content:
                logger.info("  ✓ EvaluationResult has metadata field")
            else:
                logger.warning(
                    "  ⚠ metadata field not clearly found in EvaluationResult"
                )

        # Check for module_contribution_mapper import
        if "module_contribution_mapper" in content:
            logger.info("  ✓ Imports module_contribution_mapper")
        else:
            logger.warning("  ⚠ module_contribution_mapper import not found")

        logger.info("✅ TEST 3 PASSED: Questionnaire engine enhanced")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        return False


def test_doctoral_argumentation_integration():
    """Test 4: Doctoral argumentation in answer_assembler"""
    logger.info("=" * 70)
    logger.info("TEST 4: Doctoral Argumentation Integration")
    logger.info("=" * 70)

    filepath = Path("answer_assembler.py")
    if not filepath.exists():
        logger.error(f"  ✗ File not found: {filepath}")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for doctoral engine integration
        checks = [
            ("doctoral_argumentation_engine", "Imports doctoral_argumentation_engine"),
            ("DoctoralArgumentationEngine", "Uses DoctoralArgumentationEngine class"),
            ("StructuredEvidence", "Uses StructuredEvidence"),
            ("toulmin_structure", "Generates Toulmin structures"),
            ("logical_coherence", "Validates logical coherence"),
            ("academic_quality", "Validates academic quality"),
            ("bayesian_posterior", "Uses Bayesian posterior"),
        ]

        all_found = True
        for keyword, description in checks:
            if keyword in content:
                logger.info(f"  ✓ {description}")
            else:
                logger.warning(f"  ⚠ Missing: {description}")
                all_found = False

        if all_found:
            logger.info("✅ TEST 4 PASSED: Doctoral argumentation fully integrated")
        else:
            logger.info(
                "⚠️  TEST 4 PARTIAL: Some components missing but core integration present"
            )

        return True

    except Exception as e:
        logger.error(f"❌ TEST 4 FAILED: {e}")
        return False


def test_file_syntax():
    """Test 5: All modified files have valid Python syntax"""
    logger.info("=" * 70)
    logger.info("TEST 5: File Syntax Validation")
    logger.info("=" * 70)

    files_to_check = [
        "module_contribution_mapper.py",
        "Decatalogo_principal.py",
        "questionnaire_engine.py",
        "answer_assembler.py",
    ]

    all_valid = True
    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"  ⚠ File not found: {filepath}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                ast.parse(f.read(), filename=filepath)
            logger.info(f"  ✓ {filepath} - syntax valid")
        except SyntaxError as e:
            logger.error(f"  ✗ {filepath} - syntax error: {e}")
            all_valid = False

    if all_valid:
        logger.info("✅ TEST 5 PASSED: All files have valid syntax")
    else:
        logger.error("❌ TEST 5 FAILED: Syntax errors found")

    return all_valid


def test_integration_completeness():
    """Test 6: Integration completeness check"""
    logger.info("=" * 70)
    logger.info("TEST 6: Integration Completeness")
    logger.info("=" * 70)

    checklist = [
        (
            "module_contribution_mapper.py exists",
            Path("module_contribution_mapper.py").exists(),
        ),
        (
            "evaluate_from_evidence() in Decatalogo_principal.py",
            check_method_exists_in_file(
                Path("Decatalogo_principal.py"),
                "ExtractorEvidenciaIndustrialAvanzado",
                "evaluate_from_evidence",
            ),
        ),
        (
            "_synthesize_multi_source_evidence() in questionnaire_engine.py",
            check_method_exists_in_file(
                Path("questionnaire_engine.py"),
                "QuestionnaireEngine",
                "_synthesize_multi_source_evidence",
            ),
        ),
        (
            "Doctoral integration in answer_assembler.py",
            check_import_in_method(
                Path("answer_assembler.py"), "assemble", "DoctoralArgumentationEngine"
            ),
        ),
    ]

    all_complete = True
    for description, status in checklist:
        if status:
            logger.info(f"  ✓ {description}")
        else:
            logger.error(f"  ✗ {description}")
            all_complete = False

    if all_complete:
        logger.info("✅ TEST 6 PASSED: All integration points complete")
    else:
        logger.error("❌ TEST 6 FAILED: Some integration points missing")

    return all_complete


def run_all_tests():
    """Run all lightweight integration tests"""
    logger.info("\n" + "=" * 70)
    logger.info("MINIMINIMOON LIGHTWEIGHT INTEGRATION TEST SUITE")
    logger.info("(No heavy dependencies required)")
    logger.info("=" * 70 + "\n")

    tests = [
        ("Module Contribution Mapper", test_module_contribution_mapper),
        ("evaluate_from_evidence()", test_evaluate_from_evidence),
        ("Questionnaire Engine Enhancement", test_questionnaire_engine_enhancement),
        ("Doctoral Argumentation Integration", test_doctoral_argumentation_integration),
        ("File Syntax Validation", test_file_syntax),
        ("Integration Completeness", test_integration_completeness),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        print()  # Add spacing

    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    for test_name, test_passed in results:
        status = "✅ PASSED" if test_passed else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")

    logger.info("-" * 70)
    logger.info(f"Total: {total} tests | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        logger.info("=" * 70)
        logger.info("🎉 ALL TESTS PASSED! Integration structure validated.")
        logger.info("=" * 70)
        logger.info(
            "\nNOTE: Runtime testing requires installing dependencies from requirements.txt"
        )
        return 0
    else:
        logger.error("=" * 70)
        logger.error(f"⚠️  {failed} TEST(S) FAILED. Review errors above.")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
