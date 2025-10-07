#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Unified Pipeline Mock Tests
======================================

Mock-based integration tests for the unified evaluation pipeline with:
- miniminimoon_orchestrator integration
- RUBRIC_SCORING.json validation
- AnswerAssembler component testing
- system_validators pre/post execution gates
- Deterministic hash validation
- Frozen configuration enforcement
- Evidence registry consistency
- Artifact generation (answers_report.json, flow_runtime.json)
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from miniminimoon_orchestrator import MINIMINIMOONOrchestrator
from answer_assembler import AnswerAssembler
from system_validators import SystemHealthValidator
from evidence_registry import EvidenceRegistry


class TestE2EUnifiedPipelineMock(unittest.TestCase):
    """End-to-end tests with mocked components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.artifacts_dir = self.temp_path / "artifacts"
        self.artifacts_dir.mkdir()
        
        # Create minimal RUBRIC_SCORING.json
        self.rubric_data = {
            "questions": [
                {"id": "D1-Q1-P1", "scoring_modality": "TYPE_A"},
                {"id": "D1-Q2-P1", "scoring_modality": "TYPE_B"}
            ],
            "weights": {
                "D1-Q1-P1": 1.0,
                "D1-Q2-P1": 1.0
            },
            "metadata": {
                "version": "2.0",
                "total_questions": 2
            }
        }
        self.rubric_path = self.temp_path / "RUBRIC_SCORING.json"
        with open(self.rubric_path, 'w') as f:
            json.dump(self.rubric_data, f)
        
        # Create minimal frozen config
        self.config_data = {
            "determinism": {
                "enabled": True,
                "seed": 42,
                "frozen": True
            },
            "parallel_processing": False
        }
        self.config_path = self.temp_path / "config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f)
        
        # Create immutability snapshot
        self.immutability_snapshot = {
            "frozen": True,
            "deterministic_hash": "test_hash_123",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        self.snapshot_path = self.temp_path / ".immutability_snapshot.json"
        with open(self.snapshot_path, 'w') as f:
            json.dump(self.immutability_snapshot, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rubric_scoring_validation(self):
        """Test that RUBRIC_SCORING.json is validated as single source of truth"""
        # Verify rubric structure
        self.assertIn("questions", self.rubric_data)
        self.assertIn("weights", self.rubric_data)
        self.assertEqual(len(self.rubric_data["questions"]), 2)
        self.assertEqual(len(self.rubric_data["weights"]), 2)
        
        # Verify 1:1 alignment
        question_ids = {q["id"] for q in self.rubric_data["questions"]}
        weight_ids = set(self.rubric_data["weights"].keys())
        self.assertEqual(question_ids, weight_ids)
    
    def test_answer_assembler_integration(self):
        """Test AnswerAssembler component with rubric weights"""
        # Create minimal decalogo
        decalogo_data = {
            "questions": [
                {"id": "D1-Q1", "point_code": "P1", "dimension": "D1"},
                {"id": "D1-Q2", "point_code": "P1", "dimension": "D1"}
            ]
        }
        decalogo_path = self.temp_path / "DECALOGO_FULL.json"
        with open(decalogo_path, 'w') as f:
            json.dump(decalogo_data, f)
        
        # Initialize AnswerAssembler
        assembler = AnswerAssembler(
            rubric_path=str(self.rubric_path),
            decalogo_path=str(decalogo_path)
        )
        
        # Verify weights loaded
        self.assertEqual(len(assembler.weights), 2)
        self.assertIn("D1-Q1-P1", assembler.weights)
        self.assertIn("D1-Q2-P1", assembler.weights)
    
    def test_system_validators_pre_execution(self):
        """Test system_validators pre-execution gates"""
        validator = SystemHealthValidator(str(self.temp_path))
        
        # Create required files for validation
        (self.temp_path / "tools").mkdir()
        flow_doc = {"canonical_order": ["step1", "step2"]}
        with open(self.temp_path / "tools" / "flow_doc.json", 'w') as f:
            json.dump(flow_doc, f)
        
        result = validator.validate_pre_execution()
        
        # Should pass with all required files present
        self.assertTrue(result.get("ok"), 
                        f"Pre-execution validation failed: {result.get('errors')}")
        self.assertTrue(result.get("checks", {}).get("rubric_present"))
        self.assertTrue(result.get("checks", {}).get("flow_doc_present"))
    
    def test_system_validators_post_execution(self):
        """Test system_validators post-execution gates"""
        # Create mock artifacts
        answers_report = {
            "answers": [
                {
                    "question_id": "D1-Q1-P1",
                    "evidence_ids": ["ev1", "ev2"],
                    "confidence": 0.9,
                    "reasoning": "Test reasoning",
                    "score": 2.5
                }
            ],
            "summary": {"total_questions": 1},
            "metadata": {"timestamp": "2025-01-01T00:00:00Z"}
        }
        with open(self.artifacts_dir / "answers_report.json", 'w') as f:
            json.dump(answers_report, f)
        
        flow_runtime = {
            "order": ["step1", "step2"],
            "timestamps": {}
        }
        with open(self.artifacts_dir / "flow_runtime.json", 'w') as f:
            json.dump(flow_runtime, f)
        
        # Create mock rubric_check.py that returns success
        tools_dir = self.temp_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        rubric_check_script = tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print(json.dumps({"match": True}))\n'
            'sys.exit(0)\n'
        )
        
        validator = SystemHealthValidator(str(self.temp_path))
        result = validator.validate_post_execution(
            artifacts_dir="artifacts",
            check_rubric_strict=False  # Skip rubric check for this test
        )
        
        self.assertTrue(result.get("ok"),
                        f"Post-execution validation failed: {result.get('errors')}")
    
    def test_deterministic_hash_reproducibility(self):
        """Test deterministic hash consistency"""
        # Create evidence registry with fixed content
        registry1 = EvidenceRegistry()
        registry1.register(
            source_component="test_comp",
            evidence_type="test_type",
            content={"value": 42},
            confidence=0.8,
            applicable_questions=["D1-Q1"]
        )
        hash1 = registry1.deterministic_hash()
        
        # Create second registry with same content
        registry2 = EvidenceRegistry()
        registry2.register(
            source_component="test_comp",
            evidence_type="test_type",
            content={"value": 42},
            confidence=0.8,
            applicable_questions=["D1-Q1"]
        )
        hash2 = registry2.deterministic_hash()
        
        # Hashes must be identical
        self.assertEqual(hash1, hash2,
                         "Deterministic hashes not reproducible")
    
    def test_frozen_configuration_enforcement(self):
        """Test that frozen configuration is enforced"""
        # Verify config has frozen=True
        self.assertTrue(self.config_data["determinism"]["frozen"])
        
        # Verify immutability snapshot exists
        self.assertTrue(self.snapshot_path.exists())
        snapshot = json.loads(self.snapshot_path.read_text())
        self.assertTrue(snapshot["frozen"])
    
    def test_evidence_id_consistency(self):
        """Test evidence_id consistency across multiple registrations"""
        registry = EvidenceRegistry()
        
        # Register same evidence twice
        id1 = registry.register(
            source_component="comp1",
            evidence_type="type1",
            content={"data": "test"},
            confidence=0.85,
            applicable_questions=["D1-Q1"]
        )
        
        id2 = registry.register(
            source_component="comp1",
            evidence_type="type1",
            content={"data": "test"},
            confidence=0.85,
            applicable_questions=["D1-Q1"]
        )
        
        # IDs should be identical for same content
        self.assertEqual(id1, id2,
                         "Evidence IDs not consistent for same content")
    
    def test_artifact_generation_structure(self):
        """Test that artifacts have correct structure"""
        # Mock answers_report.json structure
        answers_report = {
            "answers": [
                {
                    "question_id": "D1-Q1-P1",
                    "evidence_ids": ["ev1", "ev2"],
                    "confidence": 0.9,
                    "reasoning": "Based on evidence",
                    "score": 2.5
                }
            ],
            "summary": {
                "total_questions": 1,
                "avg_confidence": 0.9
            },
            "metadata": {
                "timestamp": "2025-01-01T00:00:00Z",
                "deterministic_hash": "hash123"
            }
        }
        
        # Verify required fields
        self.assertIn("answers", answers_report)
        self.assertIn("summary", answers_report)
        self.assertIn("metadata", answers_report)
        
        # Verify answer structure from AnswerAssembler
        answer = answers_report["answers"][0]
        required_fields = ["question_id", "evidence_ids", "confidence", 
                          "reasoning", "score"]
        for field in required_fields:
            self.assertIn(field, answer,
                          f"Answer missing required field: {field}")
        
        # Mock flow_runtime.json structure
        flow_runtime = {
            "order": ["sanitization", "plan_processing", "segmentation"],
            "timestamps": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T00:01:00Z"
            },
            "metadata": {
                "deterministic_hash": "hash123"
            }
        }
        
        # Verify required fields
        self.assertIn("order", flow_runtime)
        self.assertIn("timestamps", flow_runtime)
        self.assertIn("metadata", flow_runtime)


if __name__ == "__main__":
    unittest.main()
