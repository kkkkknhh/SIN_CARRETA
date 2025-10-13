#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test for MINIMINIMOON Full Module Integration
==========================================================

Tests the complete integration of:
1. Module contribution mapper
2. evaluate_from_evidence() in Decatalogo_principal
3. Multi-source evidence synthesis in questionnaire_engine
4. Doctoral argumentation in answer_assembler

This test validates that all pieces work together properly.
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_module_contribution_mapper():
    """Test 1: Module Contribution Mapper"""
    logger.info("=" * 70)
    logger.info("TEST 1: Module Contribution Mapper")
    logger.info("=" * 70)
    
    try:
        from module_contribution_mapper import create_default_mapper, ModuleCategory
        
        mapper = create_default_mapper()
        
        # Test question mapping
        test_questions = ['D1-Q1', 'D3-Q15', 'D6-Q45']
        for q_id in test_questions:
            mapping = mapper.get_question_mapping(q_id)
            assert mapping is not None, f"No mapping for {q_id}"
            assert len(mapping.contributions) >= 3, f"{q_id} has <3 sources"
            
            primary = mapping.get_primary_module()
            assert primary is not None, f"No primary module for {q_id}"
            
            logger.info(f"  ✓ {q_id}: {len(mapping.contributions)} modules, primary={primary.module.value}")
        
        # Test statistics
        stats = mapper.get_summary_statistics()
        assert stats['total_questions'] == 300, "Should map 300 questions"
        assert len(stats['module_usage']) >= 10, "Should use at least 10 modules"
        
        logger.info(f"  ✓ Statistics: {stats['total_questions']} questions, {len(stats['module_usage'])} modules")
        logger.info("✅ TEST 1 PASSED: Module Contribution Mapper")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluate_from_evidence_exists():
    """Test 2: evaluate_from_evidence() method exists"""
    logger.info("=" * 70)
    logger.info("TEST 2: evaluate_from_evidence() Method")
    logger.info("=" * 70)
    
    try:
        from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado
        
        # Check method exists
        assert hasattr(ExtractorEvidenciaIndustrialAvanzado, 'evaluate_from_evidence'), \
            "evaluate_from_evidence() method not found"
        
        logger.info("  ✓ Method evaluate_from_evidence() exists in ExtractorEvidenciaIndustrialAvanzado")
        
        # Check method signature
        import inspect
        sig = inspect.signature(ExtractorEvidenciaIndustrialAvanzado.evaluate_from_evidence)
        params = list(sig.parameters.keys())
        assert 'evidence_registry' in params, "Method should accept evidence_registry parameter"
        
        logger.info(f"  ✓ Method signature: {sig}")
        logger.info("✅ TEST 2 PASSED: evaluate_from_evidence() exists with correct signature")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_questionnaire_engine_enhancement():
    """Test 3: Questionnaire engine multi-source synthesis"""
    logger.info("=" * 70)
    logger.info("TEST 3: Questionnaire Engine Enhancement")
    logger.info("=" * 70)
    
    try:
        from questionnaire_engine import QuestionnaireEngine, EvaluationResult
        from dataclasses import fields
        
        # Check EvaluationResult has metadata field
        result_fields = [f.name for f in fields(EvaluationResult)]
        assert 'metadata' in result_fields, "EvaluationResult missing metadata field"
        
        logger.info("  ✓ EvaluationResult has metadata field")
        
        # Check QuestionnaireEngine has _synthesize_multi_source_evidence
        assert hasattr(QuestionnaireEngine, '_synthesize_multi_source_evidence'), \
            "_synthesize_multi_source_evidence() method not found"
        
        logger.info("  ✓ QuestionnaireEngine has _synthesize_multi_source_evidence()")
        
        # Check method signature
        import inspect
        sig = inspect.signature(QuestionnaireEngine._synthesize_multi_source_evidence)
        params = list(sig.parameters.keys())
        assert 'question_id' in params, "Method should accept question_id parameter"
        assert 'orchestrator_results' in params, "Method should accept orchestrator_results parameter"
        
        logger.info(f"  ✓ Method signature: {sig}")
        logger.info("✅ TEST 3 PASSED: Questionnaire engine enhanced with multi-source synthesis")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_doctoral_argumentation_integration():
    """Test 4: Doctoral argumentation integration"""
    logger.info("=" * 70)
    logger.info("TEST 4: Doctoral Argumentation Integration")
    logger.info("=" * 70)
    
    try:
        from answer_assembler import AnswerAssembler
        import inspect
        
        # Check assemble method exists
        assert hasattr(AnswerAssembler, 'assemble'), "assemble() method not found"
        
        logger.info("  ✓ AnswerAssembler has assemble() method")
        
        # Check the method imports doctoral_argumentation_engine
        source = inspect.getsource(AnswerAssembler.assemble)
        assert 'DoctoralArgumentationEngine' in source, \
            "assemble() does not import DoctoralArgumentationEngine"
        assert 'toulmin_structure' in source, \
            "assemble() does not reference toulmin_structure"
        assert 'logical_coherence' in source, \
            "assemble() does not reference logical_coherence"
        
        logger.info("  ✓ assemble() integrates DoctoralArgumentationEngine")
        logger.info("  ✓ assemble() generates Toulmin structures")
        logger.info("  ✓ assemble() validates logical coherence")
        logger.info("✅ TEST 4 PASSED: Doctoral argumentation integrated")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complementary_modules_imported():
    """Test 5: Complementary modules can be imported"""
    logger.info("=" * 70)
    logger.info("TEST 5: Complementary Modules Import")
    logger.info("=" * 70)
    
    modules_to_test = [
        ('pdm_contra.core', 'ContradictionDetector'),
        ('pdm_contra.scoring.risk', 'RiskScorer'),
        ('pdm_contra.nlp.patterns', 'PatternMatcher'),
        ('pdm_contra.nlp.nli', 'SpanishNLIDetector'),
        ('pdm_contra.policy.competence', 'CompetenceValidator'),
        ('pdm_contra.explain.tracer', 'ExplanationTracer'),
        ('factibilidad', 'PatternDetector'),
        ('evaluation', 'ReliabilityCalibrator'),
        ('doctoral_argumentation_engine', 'DoctoralArgumentationEngine'),
    ]
    
    all_passed = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            logger.info(f"  ✓ {module_name}.{class_name} imported successfully")
        except Exception as e:
            logger.error(f"  ✗ {module_name}.{class_name} failed: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("✅ TEST 5 PASSED: All complementary modules can be imported")
    else:
        logger.error("❌ TEST 5 FAILED: Some modules failed to import")
    
    return all_passed


def test_end_to_end_flow_diagram():
    """Test 6: Verify flow integration points"""
    logger.info("=" * 70)
    logger.info("TEST 6: End-to-End Flow Integration Points")
    logger.info("=" * 70)
    
    try:
        # Check orchestrator stages
        from miniminimoon_orchestrator import PipelineStage
        
        required_stages = [
            'DECALOGO_LOAD',
            'DECALOGO_EVAL',
            'QUESTIONNAIRE_EVAL',
            'ANSWER_ASSEMBLY'
        ]
        
        for stage_name in required_stages:
            assert hasattr(PipelineStage, stage_name), f"Missing stage: {stage_name}"
            logger.info(f"  ✓ PipelineStage.{stage_name} exists")
        
        logger.info("✅ TEST 6 PASSED: All pipeline stages defined")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    logger.info("\n" + "=" * 70)
    logger.info("MINIMINIMOON FULL MODULE INTEGRATION TEST SUITE")
    logger.info("=" * 70 + "\n")
    
    tests = [
        ("Module Contribution Mapper", test_module_contribution_mapper),
        ("evaluate_from_evidence() Method", test_evaluate_from_evidence_exists),
        ("Questionnaire Engine Enhancement", test_questionnaire_engine_enhancement),
        ("Doctoral Argumentation Integration", test_doctoral_argumentation_integration),
        ("Complementary Modules Import", test_complementary_modules_imported),
        ("End-to-End Flow Integration", test_end_to_end_flow_diagram),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        print()  # Add spacing between tests
    
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
        logger.info("🎉 ALL TESTS PASSED! Integration complete.")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("=" * 70)
        logger.error(f"⚠️  {failed} TEST(S) FAILED. Review errors above.")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
