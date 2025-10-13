#!/usr/bin/env python3
"""
Test suite for audit_system.py

Validates that audit system correctly identifies:
- 16 canonical stages
- 5 expected module integrations
- Evidence coverage analysis
- Determinism validation framework
"""

import unittest
from pathlib import Path

from audit_system import (
    AuditReport,
    DeterminismResult,
    EvidenceCoverageResult,
    MINIMINIMOONAuditor,
    ModuleIntegrationResult,
    StageAuditResult,
)


class TestAuditSystemStructure(unittest.TestCase):
    """Test audit system data structures and configuration."""

    def test_canonical_stages_count(self):
        """Verify 16 canonical stages are defined."""
        stages = MINIMINIMOONAuditor.CANONICAL_STAGES
        self.assertEqual(len(stages), 16, "Expected exactly 16 canonical stages")

    def test_canonical_stages_order(self):
        """Verify canonical stage order matches orchestrator."""
        stages = MINIMINIMOONAuditor.CANONICAL_STAGES
        expected_stages = [
            "sanitization",
            "plan_processing",
            "document_segmentation",
            "embedding",
            "responsibility_detection",
            "contradiction_detection",
            "monetary_detection",
            "feasibility_scoring",
            "causal_detection",
            "teoria_cambio",
            "dag_validation",
            "evidence_registry_build",
            "decalogo_load",
            "decalogo_evaluation",
            "questionnaire_evaluation",
            "answers_assembly",
        ]
        self.assertEqual(stages, expected_stages)

    def test_expected_modules_count(self):
        """Verify 5 expected modules are configured."""
        modules = MINIMINIMOONAuditor.EXPECTED_MODULES
        self.assertEqual(len(modules), 5, "Expected exactly 5 modules to audit")

    def test_expected_module_names(self):
        """Verify correct module names are configured."""
        modules = MINIMINIMOONAuditor.EXPECTED_MODULES
        expected = {
            "pdm_contra",
            "factibilidad",
            "CompetenceValidator",
            "ReliabilityCalibrator",
            "doctoral_argumentation_engine",
        }
        self.assertEqual(set(modules.keys()), expected)

    def test_module_stage_mappings(self):
        """Verify module-to-stage mappings are correct."""
        modules = MINIMINIMOONAuditor.EXPECTED_MODULES
        self.assertEqual(
            modules["pdm_contra"], 6, "pdm_contra should map to stage 6 (CONTRADICTION)"
        )
        self.assertEqual(
            modules["factibilidad"],
            8,
            "factibilidad should map to stage 8 (FEASIBILITY)",
        )
        self.assertEqual(
            modules["CompetenceValidator"],
            15,
            "CompetenceValidator should map to stage 15 (QUESTIONNAIRE_EVAL)",
        )
        self.assertEqual(
            modules["ReliabilityCalibrator"],
            15,
            "ReliabilityCalibrator should map to stage 15 (QUESTIONNAIRE_EVAL)",
        )
        self.assertEqual(
            modules["doctoral_argumentation_engine"],
            16,
            "doctoral_argumentation_engine should map to stage 16 (ANSWER_ASSEMBLY)",
        )

    def test_stage_audit_result_structure(self):
        """Verify StageAuditResult has required fields."""
        result = StageAuditResult(
            stage_name="test_stage",
            stage_number=1,
            executed=True,
            has_evidence=True,
            evidence_count=5,
            error=None,
        )
        self.assertEqual(result.stage_name, "test_stage")
        self.assertEqual(result.stage_number, 1)
        self.assertTrue(result.executed)
        self.assertTrue(result.has_evidence)
        self.assertEqual(result.evidence_count, 5)
        self.assertIsNone(result.error)

    def test_module_integration_result_structure(self):
        """Verify ModuleIntegrationResult has required fields."""
        result = ModuleIntegrationResult(
            module_name="test_module",
            expected_stage=8,
            found_in_codebase=False,
            imported_in_orchestrator=False,
            invoked_in_stage=False,
            file_location=None,
            import_statement=None,
            invocation_method=None,
            issues=["Not found in codebase"],
        )
        self.assertEqual(result.module_name, "test_module")
        self.assertEqual(result.expected_stage, 8)
        self.assertFalse(result.found_in_codebase)
        self.assertEqual(len(result.issues), 1)

    def test_evidence_coverage_result_structure(self):
        """Verify EvidenceCoverageResult has required fields."""
        result = EvidenceCoverageResult(
            total_questions_expected=300,
            total_questions_found=250,
            questions_with_0_sources=10,
            questions_with_1_source=20,
            questions_with_2_sources=30,
            questions_with_3plus_sources=190,
            questions_with_doctoral_justification=200,
            coverage_percentage=63.3,
            meets_requirements=False,
        )
        self.assertEqual(result.total_questions_expected, 300)
        self.assertEqual(result.questions_with_3plus_sources, 190)
        self.assertFalse(result.meets_requirements)

    def test_determinism_result_structure(self):
        """Verify DeterminismResult has required fields."""
        result = DeterminismResult(
            run1_document_hash="abc123",
            run2_document_hash="abc123",
            run1_evidence_hash="def456",
            run2_evidence_hash="def456",
            run1_flow_hash="ghi789",
            run2_flow_hash="ghi789",
            document_hashes_match=True,
            evidence_hashes_match=True,
            flow_hashes_match=True,
            total_questions_compared=300,
            questions_with_score_mismatch=0,
            score_mismatches=[],
            max_score_difference=0.0,
            is_deterministic=True,
            errors=[],
        )
        self.assertTrue(result.is_deterministic)
        self.assertTrue(result.evidence_hashes_match)
        self.assertEqual(result.questions_with_score_mismatch, 0)


class TestModuleIntegrationChecks(unittest.TestCase):
    """Test module integration checking logic."""

    def test_doctoral_argumentation_engine_exists(self):
        """Verify doctoral_argumentation_engine.py exists."""
        path = Path("doctoral_argumentation_engine.py")
        self.assertTrue(
            path.exists(), "doctoral_argumentation_engine.py should exist in codebase"
        )

    def test_missing_modules_identified(self):
        """Verify missing modules are correctly identified."""
        missing_modules = [
            "pdm_contra",
            "factibilidad",
            "CompetenceValidator",
            "ReliabilityCalibrator",
        ]
        for module in missing_modules:
            path = Path(f"{module}.py")
            # These should NOT exist (expected to be missing)
            if path.exists():
                self.fail(
                    f"{module}.py exists but is expected to be missing per requirements"
                )


if __name__ == "__main__":
    unittest.main()
