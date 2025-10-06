#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Canonical Integration Validation
================================================

Tests the canonical integration validation system including:
- All 11 node validations
- 5 target component smoke tests
- Report generation
- Dashboard metrics output
- SLO compliance checking
- CI/CD integration
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from data_flow_contract import CanonicalFlowValidator, DataType
from slo_monitoring import SLOThresholds
from validate_canonical_integration import (
    CanonicalIntegrationValidator,
    ComponentSmokeTest,
    NodeValidationResult,
    IntegrationReport,
)


class TestCanonicalIntegrationValidator(unittest.TestCase):
    """Test cases for CanonicalIntegrationValidator"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = CanonicalIntegrationValidator(
            enable_cache=True, cache_size=100
        )

    def test_initialization(self):
        """Test validator initialization"""
        self.assertIsNotNone(self.validator.validator)
        self.assertIsNotNone(self.validator.slo_monitor)
        self.assertEqual(len(self.validator.target_components), 5)
        self.assertEqual(len(self.validator.performance_baselines), 11)

    def test_target_components_defined(self):
        """Test that all 5 target components are defined"""
        expected_components = [
            "Decatalogo_principal",
            "dag_validation",
            "embedding_model",
            "plan_processor",
            "validate_teoria_cambio",
        ]

        for component in expected_components:
            self.assertIn(component, self.validator.target_components)
            smoke_test = self.validator.target_components[component]
            self.assertIsInstance(smoke_test, ComponentSmokeTest)
            self.assertTrue(smoke_test.critical)

    def test_mock_data_generation(self):
        """Test mock data generation for all data types"""
        for data_type in DataType:
            mock_data = self.validator._generate_mock_data(data_type)
            self.assertIsNotNone(mock_data)

            # Verify data type structure
            if data_type == DataType.RAW_TEXT:
                self.assertIsInstance(mock_data, str)
            elif data_type == DataType.SEGMENTS:
                self.assertIsInstance(mock_data, list)
                self.assertGreater(len(mock_data), 0)
            elif data_type == DataType.EMBEDDINGS:
                self.assertIsInstance(mock_data, list)
            elif data_type == DataType.ENTITIES:
                self.assertIsInstance(mock_data, list)

    def test_validate_node_success(self):
        """Test successful node validation"""
        available_data = {"raw_text": "Sample plan text for testing validation"}

        result = self.validator.validate_node("sanitization", available_data)

        self.assertIsInstance(result, NodeValidationResult)
        self.assertTrue(result.success)
        self.assertEqual(result.node_name, "sanitization")
        self.assertGreater(result.execution_time_ms, 0)
        self.assertTrue(result.input_valid)

    def test_validate_node_missing_input(self):
        """Test node validation with missing required input"""
        available_data = {}  # Missing raw_text

        result = self.validator.validate_node("sanitization", available_data)

        self.assertIsInstance(result, NodeValidationResult)
        self.assertFalse(result.success)
        self.assertGreater(len(result.errors), 0)

    def test_validate_all_nodes(self):
        """Test validation of all 11 canonical nodes"""
        node_results = self.validator.validate_all_nodes()

        # Should have results for all 11 nodes
        self.assertEqual(len(node_results), 11)

        # Check canonical order
        expected_nodes = [
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
        ]

        for node_name in expected_nodes:
            self.assertIn(node_name, node_results)
            result = node_results[node_name]
            self.assertIsInstance(result, NodeValidationResult)
            self.assertEqual(result.node_name, node_name)

    def test_smoke_test_mock_success(self):
        """Test smoke test with mocked successful import"""
        component = ComponentSmokeTest(
            component_name="test_component",
            module_path="sys",  # Use sys module which always exists
            test_function="test_function",
            expected_attributes=["path", "version"],
            timeout_seconds=5.0,
            critical=True,
        )

        success, message = self.validator.run_smoke_test(component)

        self.assertTrue(success)
        self.assertIn("successful", message.lower())

    def test_smoke_test_missing_attribute(self):
        """Test smoke test with missing expected attribute"""
        component = ComponentSmokeTest(
            component_name="test_component",
            module_path="sys",
            test_function="test_function",
            expected_attributes=["nonexistent_attribute"],
            timeout_seconds=5.0,
            critical=True,
        )

        success, message = self.validator.run_smoke_test(component)

        self.assertFalse(success)
        self.assertIn("Missing expected attribute", message)

    def test_check_slo_compliance_all_pass(self):
        """Test SLO compliance check when all nodes pass"""
        node_results = {
            "node1": NodeValidationResult(
                node_name="node1",
                success=True,
                execution_time_ms=5.0,
                input_valid=True,
            ),
            "node2": NodeValidationResult(
                node_name="node2",
                success=True,
                execution_time_ms=10.0,
                input_valid=True,
            ),
        }

        compliance = self.validator.check_slo_compliance(node_results)

        self.assertIn("availability", compliance)
        self.assertIn("p95_latency", compliance)
        self.assertIn("error_rate", compliance)
        self.assertTrue(compliance["availability"])
        self.assertTrue(compliance["error_rate"])

    def test_check_slo_compliance_availability_breach(self):
        """Test SLO compliance check with availability breach"""
        # Create results with high failure rate
        node_results = {}
        for i in range(10):
            node_results[f"node{i}"] = NodeValidationResult(
                node_name=f"node{i}",
                success=i < 5,  # Only 50% success
                execution_time_ms=5.0,
                input_valid=True,
            )

        compliance = self.validator.check_slo_compliance(node_results)

        # 50% availability should breach 99.5% threshold
        self.assertFalse(compliance["availability"])

    def test_capture_baseline_metrics(self):
        """Test baseline metrics capture"""
        node_results = {
            "node1": NodeValidationResult(
                node_name="node1",
                success=True,
                execution_time_ms=12.5,
                input_valid=True,
            ),
            "node2": NodeValidationResult(
                node_name="node2",
                success=True,
                execution_time_ms=8.3,
                input_valid=True,
            ),
        }

        baseline = self.validator.capture_baseline_metrics(node_results)

        self.assertEqual(len(baseline), 2)
        self.assertEqual(baseline["node1"], 12.5)
        self.assertEqual(baseline["node2"], 8.3)

    def test_generate_dashboard_metrics(self):
        """Test dashboard metrics generation"""
        node_results = {
            "sanitization": NodeValidationResult(
                node_name="sanitization",
                success=True,
                execution_time_ms=4.5,
                input_valid=True,
            )
        }
        smoke_results = {"Decatalogo_principal": True}

        metrics = self.validator.generate_dashboard_metrics(node_results, smoke_results)

        self.assertIn("timestamp", metrics)
        self.assertIn("summary", metrics)
        self.assertIn("components", metrics)
        self.assertIn("nodes", metrics)
        self.assertIn("slo_compliance", metrics)

        # Check summary
        self.assertIn("overall_health", metrics["summary"])
        self.assertIn("node_success_rate", metrics["summary"])

        # Check component health
        self.assertIn("Decatalogo_principal", metrics["components"])
        self.assertEqual(metrics["components"]["Decatalogo_principal"]["status"], "healthy")

    def test_generate_report(self):
        """Test complete report generation"""
        node_results = {
            "sanitization": NodeValidationResult(
                node_name="sanitization",
                success=True,
                execution_time_ms=4.5,
                input_valid=True,
            ),
            "plan_processing": NodeValidationResult(
                node_name="plan_processing",
                success=True,
                execution_time_ms=9.2,
                input_valid=True,
            ),
        }
        smoke_results = {
            "Decatalogo_principal": True,
            "dag_validation": True,
            "embedding_model": True,
            "plan_processor": True,
            "validate_teoria_cambio": True,
        }

        report = self.validator.generate_report(node_results, smoke_results)

        self.assertIsInstance(report, IntegrationReport)
        self.assertTrue(report.overall_success)
        self.assertEqual(report.total_nodes, 2)
        self.assertEqual(report.passed_nodes, 2)
        self.assertEqual(report.failed_nodes, 0)
        self.assertEqual(report.ci_exit_code, 0)
        self.assertIsNotNone(report.dashboard_metrics)

    def test_generate_report_with_failures(self):
        """Test report generation with node failures"""
        node_results = {
            "sanitization": NodeValidationResult(
                node_name="sanitization",
                success=True,
                execution_time_ms=4.5,
                input_valid=True,
            ),
            "plan_processing": NodeValidationResult(
                node_name="plan_processing",
                success=False,
                execution_time_ms=9.2,
                input_valid=False,
                errors=["Missing required input"],
            ),
        }
        smoke_results = {
            "Decatalogo_principal": True,
            "dag_validation": False,
            "embedding_model": True,
            "plan_processor": True,
            "validate_teoria_cambio": True,
        }

        report = self.validator.generate_report(node_results, smoke_results)

        self.assertFalse(report.overall_success)
        self.assertEqual(report.failed_nodes, 1)
        self.assertEqual(report.ci_exit_code, 1)

    def test_save_report(self):
        """Test report saving to JSON file"""
        node_results = {
            "sanitization": NodeValidationResult(
                node_name="sanitization",
                success=True,
                execution_time_ms=4.5,
                input_valid=True,
            )
        }
        smoke_results = {"Decatalogo_principal": True}

        report = self.validator.generate_report(node_results, smoke_results)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = Path(f.name)

        try:
            self.validator.save_report(report, temp_path)

            # Verify file was created and is valid JSON
            self.assertTrue(temp_path.exists())

            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            self.assertIn("timestamp", saved_data)
            self.assertIn("overall_success", saved_data)
            self.assertIn("node_results", saved_data)
            self.assertIn("smoke_test_results", saved_data)

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_performance_baseline_thresholds(self):
        """Test that performance baselines are defined for all nodes"""
        canonical_nodes = [
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
        ]

        for node_name in canonical_nodes:
            self.assertIn(node_name, self.validator.performance_baselines)
            baseline = self.validator.performance_baselines[node_name]
            self.assertGreater(baseline, 0)

    def test_ci_mode_exit_codes(self):
        """Test CI mode exit codes"""
        # Success case - need all 5 critical components
        node_results_success = {
            "node1": NodeValidationResult(
                node_name="node1", success=True, execution_time_ms=5.0, input_valid=True
            )
        }
        smoke_results_success = {
            "Decatalogo_principal": True,
            "dag_validation": True,
            "embedding_model": True,
            "plan_processor": True,
            "validate_teoria_cambio": True,
        }
        report_success = self.validator.generate_report(
            node_results_success, smoke_results_success
        )
        self.assertEqual(report_success.ci_exit_code, 0)

        # Failure case
        node_results_fail = {
            "node1": NodeValidationResult(
                node_name="node1",
                success=False,
                execution_time_ms=5.0,
                input_valid=False,
            )
        }
        smoke_results_fail = {"Decatalogo_principal": False}
        report_fail = self.validator.generate_report(node_results_fail, smoke_results_fail)
        self.assertEqual(report_fail.ci_exit_code, 1)


class TestNodeValidationResult(unittest.TestCase):
    """Test cases for NodeValidationResult dataclass"""

    def test_creation(self):
        """Test NodeValidationResult creation"""
        result = NodeValidationResult(
            node_name="test_node",
            success=True,
            execution_time_ms=12.5,
            input_valid=True,
            errors=["error1"],
            warnings=["warning1"],
            cached=True,
        )

        self.assertEqual(result.node_name, "test_node")
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time_ms, 12.5)
        self.assertTrue(result.input_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.warnings), 1)
        self.assertTrue(result.cached)


class TestIntegrationReport(unittest.TestCase):
    """Test cases for IntegrationReport dataclass"""

    def test_creation(self):
        """Test IntegrationReport creation"""
        node_result = NodeValidationResult(
            node_name="test_node",
            success=True,
            execution_time_ms=5.0,
            input_valid=True,
        )

        report = IntegrationReport(
            timestamp=datetime.now(),
            overall_success=True,
            total_nodes=1,
            passed_nodes=1,
            failed_nodes=0,
            node_results={"test_node": node_result},
            smoke_test_results={"component1": True},
            performance_metrics={"total_time": 100.0},
            baseline_metrics={"test_node": 5.0},
            slo_compliance={"availability": True},
            ci_exit_code=0,
        )

        self.assertTrue(report.overall_success)
        self.assertEqual(report.total_nodes, 1)
        self.assertEqual(report.ci_exit_code, 0)


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
