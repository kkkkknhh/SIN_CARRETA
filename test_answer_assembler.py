#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for AnswerAssembler with rubric weight validation.

Tests cover:
1. Enum import fix
2. _load_rubric() method validation
3. _validate_rubric_coverage() method for 1:1 alignment
4. Initialization-time error handling
5. Integration with RUBRIC_SCORING.json and DECALOGO_FULL.json
"""

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from answer_assembler import AnswerAssembler, EvidenceQuality


class TestAnswerAssemblerRubricValidation(unittest.TestCase):
    """Test suite for AnswerAssembler rubric weight validation (GATE #5)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.rubric_path = pathlib.Path(self.test_dir) / "test_rubric.json"
        self.decalogo_path = pathlib.Path(self.test_dir) / "test_decalogo.json"
        
        # Minimal valid rubric structure
        self.valid_rubric = {
            "score_bands": {},
            "scoring_modalities": {},
            "dimensions": {},
            "questions": [
                {"id": "D1-Q1", "scoring_modality": "TYPE_A"},
                {"id": "D1-Q2", "scoring_modality": "TYPE_B"}
            ],
            "weights": {
                "D1-Q1-P1": 1.0,
                "D1-Q2-P1": 1.0
            }
        }
        
        # Minimal valid decalogo structure
        self.valid_decalogo = {
            "questions": [
                {"id": "D1-Q1", "point_code": "P1", "dimension": "D1", "question_no": 1},
                {"id": "D1-Q2", "point_code": "P1", "dimension": "D1", "question_no": 2}
            ]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_json(self, path, data):
        """Helper to write JSON data to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def test_enum_import(self):
        """Test that Enum is properly imported and EvidenceQuality works."""
        self.assertTrue(hasattr(EvidenceQuality, 'EXCELENTE'))
        self.assertTrue(hasattr(EvidenceQuality, 'BUENA'))
        self.assertTrue(hasattr(EvidenceQuality, 'ACEPTABLE'))
        self.assertTrue(hasattr(EvidenceQuality, 'DEBIL'))
        self.assertTrue(hasattr(EvidenceQuality, 'INSUFICIENTE'))
        
        # Test enum usage
        quality = EvidenceQuality.EXCELENTE
        self.assertEqual(quality.value, "excelente")

    def test_initialization_with_valid_rubric(self):
        """Test successful initialization with valid rubric containing weights."""
        self._write_json(self.rubric_path, self.valid_rubric)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        assembler = AnswerAssembler(
            rubric_path=str(self.rubric_path),
            decalogo_path=str(self.decalogo_path)
        )
        
        self.assertEqual(len(assembler.weights), 2)
        self.assertEqual(len(assembler.questions_by_unique_id), 2)
        self.assertIn("D1-Q1-P1", assembler.weights)
        self.assertIn("D1-Q2-P1", assembler.weights)

    def test_load_rubric_missing_questions_key(self):
        """Test _load_rubric() fails when 'questions' key is missing."""
        rubric_no_questions = {
            "score_bands": {},
            "weights": {"D1-Q1-P1": 1.0}
        }
        self._write_json(self.rubric_path, rubric_no_questions)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        with self.assertRaises(ValueError) as context:
            AnswerAssembler(
                rubric_path=str(self.rubric_path),
                decalogo_path=str(self.decalogo_path)
            )
        
        self.assertIn("GATE #5 FAILED", str(context.exception))
        self.assertIn("'questions' key missing", str(context.exception))

    def test_load_rubric_missing_weights_key(self):
        """Test _load_rubric() fails when 'weights' key is missing."""
        rubric_no_weights = {
            "score_bands": {},
            "questions": [{"id": "D1-Q1", "scoring_modality": "TYPE_A"}]
        }
        self._write_json(self.rubric_path, rubric_no_weights)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        with self.assertRaises(ValueError) as context:
            AnswerAssembler(
                rubric_path=str(self.rubric_path),
                decalogo_path=str(self.decalogo_path)
            )
        
        self.assertIn("GATE #5 FAILED", str(context.exception))
        self.assertIn("'weights' key missing", str(context.exception))

    def test_validate_rubric_coverage_missing_weights(self):
        """Test _validate_rubric_coverage() fails when some questions lack weights."""
        rubric_missing_weight = {
            "score_bands": {},
            "scoring_modalities": {},
            "dimensions": {},
            "questions": [
                {"id": "D1-Q1", "scoring_modality": "TYPE_A"},
                {"id": "D1-Q2", "scoring_modality": "TYPE_B"}
            ],
            "weights": {
                "D1-Q1-P1": 1.0
                # Missing D1-Q2-P1
            }
        }
        self._write_json(self.rubric_path, rubric_missing_weight)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        with self.assertRaises(ValueError) as context:
            AnswerAssembler(
                rubric_path=str(self.rubric_path),
                decalogo_path=str(self.decalogo_path)
            )
        
        self.assertIn("GATE #5 FAILED", str(context.exception))
        self.assertIn("Rubric weight coverage mismatch", str(context.exception))
        self.assertIn("Missing weights", str(context.exception))

    def test_validate_rubric_coverage_extra_weights(self):
        """Test _validate_rubric_coverage() fails when rubric has extra weights."""
        rubric_extra_weights = {
            "score_bands": {},
            "scoring_modalities": {},
            "dimensions": {},
            "questions": [
                {"id": "D1-Q1", "scoring_modality": "TYPE_A"},
                {"id": "D1-Q2", "scoring_modality": "TYPE_B"}
            ],
            "weights": {
                "D1-Q1-P1": 1.0,
                "D1-Q2-P1": 1.0,
                "D1-Q3-P1": 1.0  # Extra weight for non-existent question
            }
        }
        self._write_json(self.rubric_path, rubric_extra_weights)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        with self.assertRaises(ValueError) as context:
            AnswerAssembler(
                rubric_path=str(self.rubric_path),
                decalogo_path=str(self.decalogo_path)
            )
        
        self.assertIn("GATE #5 FAILED", str(context.exception))
        self.assertIn("Rubric weight coverage mismatch", str(context.exception))
        self.assertIn("Extra weights", str(context.exception))

    def test_validate_rubric_coverage_perfect_match(self):
        """Test _validate_rubric_coverage() succeeds with perfect 1:1 alignment."""
        self._write_json(self.rubric_path, self.valid_rubric)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        # Should not raise any exception
        assembler = AnswerAssembler(
            rubric_path=str(self.rubric_path),
            decalogo_path=str(self.decalogo_path)
        )
        
        # Verify weights are loaded
        self.assertEqual(set(assembler.weights.keys()), 
                        set(assembler.questions_by_unique_id.keys()))

    def test_weights_stored_internally(self):
        """Test that weights are stored as instance variable."""
        self._write_json(self.rubric_path, self.valid_rubric)
        self._write_json(self.decalogo_path, self.valid_decalogo)
        
        assembler = AnswerAssembler(
            rubric_path=str(self.rubric_path),
            decalogo_path=str(self.decalogo_path)
        )
        
        self.assertIsInstance(assembler.weights, dict)
        self.assertEqual(assembler.weights["D1-Q1-P1"], 1.0)
        self.assertEqual(assembler.weights["D1-Q2-P1"], 1.0)


class TestAnswerAssemblerIntegration(unittest.TestCase):
    """Integration tests with actual RUBRIC_SCORING.json and DECALOGO_FULL.json."""

    def test_initialization_with_real_files(self):
        """Test initialization with actual project files."""
        # This test requires the actual files to exist
        rubric_path = pathlib.Path("rubric_scoring.json")
        decalogo_path = pathlib.Path("DECALOGO_FULL.json")
        
        if not (rubric_path.exists() and decalogo_path.exists()):
            self.skipTest("Real project files not available")
        
        assembler = AnswerAssembler(
            rubric_path=str(rubric_path),
            decalogo_path=str(decalogo_path)
        )
        
        # Verify 300 questions and 300 weights
        self.assertEqual(len(assembler.questions_by_unique_id), 300)
        self.assertEqual(len(assembler.weights), 300)
        
        # Verify perfect 1:1 alignment
        self.assertEqual(
            set(assembler.weights.keys()),
            set(assembler.questions_by_unique_id.keys())
        )

    def test_gate5_validation_log_message(self):
        """Test that GATE #5 validation produces correct log message."""
        rubric_path = pathlib.Path("rubric_scoring.json")
        decalogo_path = pathlib.Path("DECALOGO_FULL.json")
        
        if not (rubric_path.exists() and decalogo_path.exists()):
            self.skipTest("Real project files not available")
        
        with self.assertLogs('AnswerAssembler', level='INFO') as cm:
            assembler = AnswerAssembler(
                rubric_path=str(rubric_path),
                decalogo_path=str(decalogo_path)
            )
        
        # Check for GATE #5 validation log
        log_messages = '\n'.join(cm.output)
        self.assertIn("Rubric validated (gate #5)", log_messages)
        self.assertIn("300/300 questions with weights", log_messages)


class TestAnswerAssemblerErrorMessages(unittest.TestCase):
    """Test error message clarity and helpfulness."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.rubric_path = pathlib.Path(self.test_dir) / "test_rubric.json"
        self.decalogo_path = pathlib.Path(self.test_dir) / "test_decalogo.json"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_error_message_includes_gate_number(self):
        """Test that error messages include GATE #5 reference."""
        rubric_no_weights = {
            "questions": [{"id": "D1-Q1", "scoring_modality": "TYPE_A"}]
        }
        decalogo = {
            "questions": [{"id": "D1-Q1", "point_code": "P1", "dimension": "D1", "question_no": 1}]
        }
        
        with open(self.rubric_path, 'w') as f:
            json.dump(rubric_no_weights, f)
        with open(self.decalogo_path, 'w') as f:
            json.dump(decalogo, f)
        
        with self.assertRaises(ValueError) as context:
            AnswerAssembler(
                rubric_path=str(self.rubric_path),
                decalogo_path=str(self.decalogo_path)
            )
        
        error_message = str(context.exception)
        self.assertIn("GATE #5", error_message)

    def test_error_message_shows_sample_missing_weights(self):
        """Test that error messages show sample of missing weights."""
        # Create rubric with many missing weights
        rubric = {
            "questions": [{"id": f"D{i}-Q{i}", "scoring_modality": "TYPE_A"} for i in range(1, 16)],
            "weights": {}  # No weights
        }
        decalogo = {
            "questions": [
                {"id": f"D{i}-Q{i}", "point_code": "P1", "dimension": f"D{i}", "question_no": i}
                for i in range(1, 16)
            ]
        }
        
        with open(self.rubric_path, 'w') as f:
            json.dump(rubric, f)
        with open(self.decalogo_path, 'w') as f:
            json.dump(decalogo, f)
        
        with self.assertRaises(ValueError) as context:
            AnswerAssembler(
                rubric_path=str(self.rubric_path),
                decalogo_path=str(self.decalogo_path)
            )
        
        error_message = str(context.exception)
        self.assertIn("Missing weights for 15 questions", error_message)
        # Should show sample of missing questions
        self.assertTrue(any(f"D{i}-Q{i}-P1" in error_message for i in range(1, 11)))


if __name__ == "__main__":
    unittest.main()
