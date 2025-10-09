#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test FLOW #3 Fix: Verify document_segmenter receives doc_struct (dict) not sanitized_text (str)

This test verifies that the orchestrator correctly passes doc_struct (dict) from 
plan_processor (FLOW #2) to document_segmenter (FLOW #3), as specified in the 
critical flows documentation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json


class TestFlow3Fix(unittest.TestCase):
    """Test FLOW #3: document_segmentation receives correct input type"""
    
    def test_document_segmenter_receives_doc_struct(self):
        """
        Verify document_segmenter.segment() is called with doc_struct (dict), 
        not sanitized_text (str).
        
        This tests the critical fix for FLOW #3:
        - Input: {doc_struct:dict} from plan_processor (FLOW #2)
        - Not: sanitized_text (str)
        """
        # Import here to avoid import errors if dependencies missing
        try:
            from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
        except ImportError as e:
            self.skipTest(f"Cannot import orchestrator: {e}")
        
        # Create a temporary plan file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Plan de Desarrollo Municipal 2024-2028\n\n")
            f.write("Objetivo: Mejorar la calidad de vida de los ciudadanos.\n")
            f.write("Meta: Reducir la pobreza en un 20%.\n")
            temp_plan_path = f.name
        
        try:
            # Create config directory with minimal config files
            with tempfile.TemporaryDirectory() as config_dir:
                config_path = Path(config_dir)
                
                # Create minimal RUBRIC_SCORING.json
                rubric = {
                    "weights": {
                        "D1-Q1": 1.0,
                        "D1-Q2": 1.0
                    }
                }
                with open(config_path / "RUBRIC_SCORING.json", 'w', encoding='utf-8') as f:
                    json.dump(rubric, f)
                
                # Create minimal DECALOGO_FULL.json
                decalogo = {
                    "dimensions": []
                }
                with open(config_path / "DECALOGO_FULL.json", 'w', encoding='utf-8') as f:
                    json.dump(decalogo, f)
                
                # Patch components to track calls
                with patch('miniminimoon_orchestrator.PlanSanitizer') as MockSanitizer, \
                     patch('miniminimoon_orchestrator.PlanProcessor') as MockProcessor, \
                     patch('miniminimoon_orchestrator.DocumentSegmenter') as MockSegmenter, \
                     patch('miniminimoon_orchestrator.EmbeddingModel') as MockEmbedding, \
                     patch('miniminimoon_orchestrator.ResponsibilityDetector') as MockResp, \
                     patch('miniminimoon_orchestrator.ContradictionDetector') as MockContra, \
                     patch('miniminimoon_orchestrator.MonetaryDetector') as MockMonetary, \
                     patch('miniminimoon_orchestrator.FeasibilityScorer') as MockFeas, \
                     patch('miniminimoon_orchestrator.CausalPatternDetector') as MockCausal, \
                     patch('miniminimoon_orchestrator.TeoriaCambioValidator') as MockTeoria, \
                     patch('miniminimoon_orchestrator.DAGValidator') as MockDAG, \
                     patch('miniminimoon_orchestrator.QuestionnaireEngine') as MockQuestionnaire, \
                     patch('miniminimoon_orchestrator.ExtractorEvidenciaIndustrialAvanzado') as MockExtractor, \
                     patch('miniminimoon_orchestrator.EnhancedImmutabilityContract') as MockImmutability:
                    
                    # Setup mocks
                    mock_sanitizer = MockSanitizer.return_value
                    mock_sanitizer.sanitize_text.return_value = "sanitized text"
                    
                    # CRITICAL: plan_processor returns doc_struct (dict)
                    mock_processor = MockProcessor.return_value
                    mock_doc_struct = {
                        "full_text": "sanitized text",
                        "metadata": {"title": "Test Plan"},
                        "sections": {},
                        "evidence": {},
                        "cluster_evidence": {},
                        "processing_status": "success"
                    }
                    mock_processor.process.return_value = mock_doc_struct
                    
                    # document_segmenter should receive doc_struct (dict)
                    mock_segmenter = MockSegmenter.return_value
                    mock_segment = Mock()
                    mock_segment.text = "segment text"
                    mock_segmenter.segment.return_value = [mock_segment]
                    
                    # Mock other components
                    mock_embedding = MockEmbedding.return_value
                    mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
                    
                    mock_resp.return_value.detect_responsibilities.return_value = []
                    mock_contra.return_value.detect_contradictions.return_value = Mock(contradictions=[])
                    mock_monetary.return_value.detect_monetary.return_value = []
                    mock_feas.return_value.score_feasibility.return_value = {"indicators": []}
                    mock_causal.return_value.detect_patterns.return_value = {"patterns": []}
                    mock_teoria.return_value.validate.return_value = {"toc_graph": {}}
                    mock_dag.return_value.validate.return_value = {"is_valid": True}
                    
                    mock_questionnaire = MockQuestionnaire.return_value
                    mock_questionnaire.evaluate.return_value = {"question_results": []}
                    
                    mock_extractor = MockExtractor.return_value
                    mock_extractor.evaluate_from_evidence.return_value = {"dimensions": []}
                    
                    mock_immutability.return_value.has_snapshot.return_value = False
                    mock_immutability.return_value.freeze_configuration.return_value = None
                    
                    # Create orchestrator and process plan
                    try:
                        orchestrator = CanonicalDeterministicOrchestrator(
                            config_dir=config_path,
                            enable_validation=False
                        )
                        
                        # Process the plan
                        result = orchestrator.process_plan_deterministic(temp_plan_path)
                        
                        # CRITICAL ASSERTION: Verify document_segmenter.segment() was called with doc_struct (dict)
                        mock_segmenter.segment.assert_called()
                        call_args = mock_segmenter.segment.call_args
                        
                        # Get the first positional argument
                        if call_args[0]:  # positional args
                            actual_input = call_args[0][0]
                        else:  # keyword args
                            actual_input = call_args[1].get('doc_struct')
                        
                        # VERIFY: Input is a dict, not a string
                        self.assertIsInstance(
                            actual_input, 
                            dict, 
                            f"FLOW #3 VIOLATION: document_segmenter.segment() received {type(actual_input).__name__}, "
                            f"expected dict. This violates the canonical flow specification."
                        )
                        
                        # VERIFY: Dict contains expected fields from plan_processor
                        self.assertIn("full_text", actual_input, 
                                     "doc_struct must contain 'full_text' field")
                        self.assertEqual(actual_input["full_text"], "sanitized text",
                                       "full_text should match sanitized text")
                        
                        print("✓ FLOW #3 VERIFIED: document_segmenter receives doc_struct (dict) from plan_processor")
                        print(f"✓ doc_struct type: {type(actual_input).__name__}")
                        print(f"✓ doc_struct keys: {sorted(actual_input.keys())}")
                        
                    except Exception as e:
                        self.fail(f"Orchestrator execution failed: {e}")
        
        finally:
            # Cleanup
            import os
            os.unlink(temp_plan_path)


if __name__ == '__main__':
    unittest.main()
