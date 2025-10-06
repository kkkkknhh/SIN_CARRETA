#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Integration Validation Script
========================================

Programmatically verifies all 11 canonical nodes are properly integrated by
executing CanonicalFlowValidator.validate_node_execution() for each node.

Features:
- Validates all 11 canonical pipeline nodes
- Smoke tests for 5 target components (Decatalogo_principal, dag_validation, 
  embedding_model, plan_processor, validate_teoria_cambio)
- Generates validation report with pass/fail status
- Captures baseline performance metrics
- CI/CD compatible with automatic PR blocking on failures
- Structured metrics output for dashboard integration
- SLO tracking integration

Target Components (Highlighted):
1. Decatalogo_principal - Main evaluation system
2. dag_validation - DAG structure validator  
3. embedding_model - Document embedding generator
4. plan_processor - Plan metadata extractor
5. validate_teoria_cambio - Theory of Change validator

Canonical Nodes (11 Total):
1. sanitization
2. plan_processing
3. document_segmentation
4. embedding
5. responsibility_detection
6. contradiction_detection
7. monetary_detection
8. feasibility_scoring
9. causal_detection
10. teoria_cambio
11. dag_validation
"""

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Core validation infrastructure
from data_flow_contract import CanonicalFlowValidator, DataType

# SLO monitoring integration
from slo_monitoring import (
    SLOMonitor,
    SLOThresholds,
    FlowMetrics,
    Alert,
    AlertType,
    AlertSeverity,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CanonicalIntegrationValidator")


@dataclass
class ComponentSmokeTest:
    """Smoke test definition for target component"""

    component_name: str
    module_path: str
    test_function: str
    expected_attributes: List[str] = field(default_factory=list)
    timeout_seconds: float = 5.0
    critical: bool = True


@dataclass
class NodeValidationResult:
    """Result of validating a single node"""

    node_name: str
    success: bool
    execution_time_ms: float
    input_valid: bool
    output_valid: Optional[bool] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cached: bool = False


@dataclass
class IntegrationReport:
    """Complete integration validation report"""

    timestamp: datetime
    overall_success: bool
    total_nodes: int
    passed_nodes: int
    failed_nodes: int
    node_results: Dict[str, NodeValidationResult]
    smoke_test_results: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    slo_compliance: Dict[str, bool]
    cache_stats: Optional[Dict[str, Any]] = None
    ci_exit_code: int = 0
    dashboard_metrics: Dict[str, Any] = field(default_factory=dict)


class CanonicalIntegrationValidator:
    """
    Validates integration of all 11 canonical nodes with comprehensive smoke tests
    for the 5 target components.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 1000,
        slo_thresholds: Optional[SLOThresholds] = None,
    ):
        """
        Initialize validator.

        Args:
            enable_cache: Enable validation caching
            cache_size: Cache size for validation results
            slo_thresholds: Custom SLO thresholds (uses defaults if None)
        """
        self.validator = CanonicalFlowValidator(
            enable_cache=enable_cache, cache_size=cache_size
        )
        self.slo_monitor = SLOMonitor(
            thresholds=slo_thresholds or SLOThresholds(), 
            aggregation_window_seconds=300
        )

        # Define 5 target components for smoke testing
        self.target_components = {
            "Decatalogo_principal": ComponentSmokeTest(
                component_name="Decatalogo_principal",
                module_path="Decatalogo_principal",
                test_function="test_decatalogo_import",
                expected_attributes=["AdvancedDeviceConfig", "BUNDLE"],
                timeout_seconds=10.0,
                critical=True,
            ),
            "dag_validation": ComponentSmokeTest(
                component_name="dag_validation",
                module_path="dag_validation",
                test_function="test_dag_validator_import",
                expected_attributes=["AdvancedDAGValidator"],
                timeout_seconds=5.0,
                critical=True,
            ),
            "embedding_model": ComponentSmokeTest(
                component_name="embedding_model",
                module_path="embedding_model",
                test_function="test_embedding_model_import",
                expected_attributes=["IndustrialEmbeddingModel"],
                timeout_seconds=5.0,
                critical=True,
            ),
            "plan_processor": ComponentSmokeTest(
                component_name="plan_processor",
                module_path="plan_processor",
                test_function="test_plan_processor_import",
                expected_attributes=["PlanProcessor"],
                timeout_seconds=5.0,
                critical=True,
            ),
            "validate_teoria_cambio": ComponentSmokeTest(
                component_name="validate_teoria_cambio",
                module_path="validate_teoria_cambio",
                test_function="test_teoria_cambio_validator_import",
                expected_attributes=[],  # Validation script, not class
                timeout_seconds=5.0,
                critical=True,
            ),
        }

        # Performance baseline thresholds (ms)
        self.performance_baselines = {
            "sanitization": 5.0,
            "plan_processing": 10.0,
            "document_segmentation": 15.0,
            "embedding": 50.0,
            "responsibility_detection": 20.0,
            "contradiction_detection": 15.0,
            "monetary_detection": 10.0,
            "feasibility_scoring": 15.0,
            "causal_detection": 20.0,
            "teoria_cambio": 30.0,
            "dag_validation": 25.0,
        }

        logger.info("CanonicalIntegrationValidator initialized")
        logger.info(f"Monitoring {len(self.target_components)} target components")
        logger.info(f"Validating {len(self.validator.execution_order)} canonical nodes")

    def run_smoke_test(self, component: ComponentSmokeTest) -> Tuple[bool, str]:
        """
        Run smoke test for a target component.

        Args:
            component: Component smoke test definition

        Returns:
            (success, message)
        """
        logger.info(f"Running smoke test for {component.component_name}...")
        start_time = time.time()

        try:
            # Try to import the module
            module = __import__(component.module_path)

            # Check expected attributes
            for attr in component.expected_attributes:
                if not hasattr(module, attr):
                    return (
                        False,
                        f"Missing expected attribute: {attr} in {component.module_path}",
                    )

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > component.timeout_seconds * 1000:
                return (
                    False,
                    f"Import timeout: {elapsed_ms:.1f}ms > {component.timeout_seconds*1000}ms",
                )

            logger.info(f"✓ {component.component_name} smoke test passed ({elapsed_ms:.1f}ms)")
            return True, f"Import successful in {elapsed_ms:.1f}ms"

        except ImportError as e:
            logger.error(f"✗ {component.component_name} import failed: {e}")
            return False, f"Import error: {str(e)}"
        except Exception as e:
            logger.error(f"✗ {component.component_name} test failed: {e}")
            return False, f"Test error: {str(e)}"

    def validate_node(
        self, node_name: str, available_data: Dict[str, Any]
    ) -> NodeValidationResult:
        """
        Validate a single canonical node.

        Args:
            node_name: Name of the node
            available_data: Data available for the node

        Returns:
            Node validation result
        """
        logger.info(f"Validating node: {node_name}")
        start_time = time.time()

        try:
            is_valid, report = self.validator.validate_node_execution(
                node_name=node_name, available_data=available_data, use_cache=True
            )

            elapsed_ms = (time.time() - start_time) * 1000

            result = NodeValidationResult(
                node_name=node_name,
                success=is_valid,
                execution_time_ms=elapsed_ms,
                input_valid=report.get("input_validation", {}).get("valid", False),
                errors=report.get("errors", []),
                cached=report.get("cached", False),
            )

            # Track metrics for SLO monitoring
            self.slo_monitor.record_request(
                flow_name=node_name,
                success=is_valid,
                latency_ms=elapsed_ms,
                contract_valid=is_valid
            )

            if is_valid:
                logger.info(f"✓ {node_name} validation passed ({elapsed_ms:.2f}ms)")
            else:
                logger.error(f"✗ {node_name} validation failed: {result.errors}")

            return result

        except Exception as e:
            logger.error(f"✗ {node_name} validation error: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return NodeValidationResult(
                node_name=node_name,
                success=False,
                execution_time_ms=elapsed_ms,
                input_valid=False,
                errors=[f"Validation exception: {str(e)}"],
            )

    def validate_all_nodes(self) -> Dict[str, NodeValidationResult]:
        """
        Validate all 11 canonical nodes in execution order.

        Returns:
            Dictionary mapping node names to validation results
        """
        logger.info("=" * 80)
        logger.info("CANONICAL NODE VALIDATION - 11 NODES")
        logger.info("=" * 80)

        results = {}
        available_data = {}

        # Build up data as we go through the canonical flow
        for node_name in self.validator.execution_order:
            contract = self.validator.get_contract(node_name)

            # Simulate data for required inputs
            for required_input in contract.required_inputs:
                if required_input.value not in available_data:
                    # Generate mock data for validation
                    available_data[required_input.value] = self._generate_mock_data(
                        required_input
                    )

            # Validate the node
            result = self.validate_node(node_name, available_data)
            results[node_name] = result

            # Add outputs to available data for downstream nodes
            if result.success:
                for output in contract.produces:
                    available_data[output.value] = self._generate_mock_data(output)

        return results

    def run_smoke_tests(self) -> Dict[str, bool]:
        """
        Run smoke tests for all 5 target components.

        Returns:
            Dictionary mapping component names to success status
        """
        logger.info("=" * 80)
        logger.info("SMOKE TESTS - 5 TARGET COMPONENTS")
        logger.info("=" * 80)

        results = {}
        for component_name, component in self.target_components.items():
            success, message = self.run_smoke_test(component)
            results[component_name] = success

            if not success and component.critical:
                logger.error(f"CRITICAL: {component_name} smoke test failed: {message}")

        return results

    def check_slo_compliance(
        self, node_results: Dict[str, NodeValidationResult]
    ) -> Dict[str, bool]:
        """
        Check SLO compliance for all nodes.

        Args:
            node_results: Validation results for all nodes

        Returns:
            Dictionary mapping SLO names to compliance status
        """
        compliance = {}

        # Check availability
        total_nodes = len(node_results)
        successful_nodes = sum(1 for r in node_results.values() if r.success)
        availability = (successful_nodes / total_nodes * 100) if total_nodes > 0 else 0
        compliance["availability"] = (
            availability >= self.slo_monitor.thresholds.availability_percent
        )

        # Check p95 latency
        latencies = [r.execution_time_ms for r in node_results.values()]
        if latencies:
            latencies_sorted = sorted(latencies)
            p95_idx = int(len(latencies_sorted) * 0.95)
            p95_latency = latencies_sorted[p95_idx] if p95_idx < len(latencies_sorted) else latencies_sorted[-1]
            compliance["p95_latency"] = (
                p95_latency <= self.slo_monitor.thresholds.p95_latency_ms
            )

        # Check error rate
        failed_nodes = sum(1 for r in node_results.values() if not r.success)
        error_rate = (failed_nodes / total_nodes * 100) if total_nodes > 0 else 0
        compliance["error_rate"] = (
            error_rate <= self.slo_monitor.thresholds.error_rate_percent
        )

        return compliance

    def generate_dashboard_metrics(
        self, node_results: Dict[str, NodeValidationResult], smoke_results: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Generate structured metrics for dashboard integration.

        Args:
            node_results: Node validation results
            smoke_results: Smoke test results

        Returns:
            Dashboard-compatible metrics
        """
        timestamp = datetime.now().isoformat()

        # Component health metrics
        component_health = {}
        for component_name, success in smoke_results.items():
            component_health[component_name] = {
                "status": "healthy" if success else "unhealthy",
                "timestamp": timestamp,
                "critical": self.target_components[component_name].critical,
            }

        # Node execution metrics
        node_metrics = {}
        for node_name, result in node_results.items():
            node_metrics[node_name] = {
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "baseline_ms": self.performance_baselines.get(node_name, 0),
                "baseline_deviation_pct": (
                    ((result.execution_time_ms - self.performance_baselines.get(node_name, 0))
                     / self.performance_baselines.get(node_name, 1))
                    * 100
                    if self.performance_baselines.get(node_name, 0) > 0
                    else 0
                ),
                "cached": result.cached,
                "timestamp": timestamp,
            }

        # Aggregate metrics
        total_nodes = len(node_results)
        successful_nodes = sum(1 for r in node_results.values() if r.success)
        total_components = len(smoke_results)
        healthy_components = sum(1 for s in smoke_results.values() if s)

        dashboard_metrics = {
            "timestamp": timestamp,
            "summary": {
                "overall_health": "healthy"
                if successful_nodes == total_nodes and healthy_components == total_components
                else "degraded"
                if successful_nodes >= total_nodes * 0.8
                else "unhealthy",
                "node_success_rate": successful_nodes / total_nodes if total_nodes > 0 else 0,
                "component_health_rate": healthy_components / total_components
                if total_components > 0
                else 0,
            },
            "components": component_health,
            "nodes": node_metrics,
            "slo_compliance": self.check_slo_compliance(node_results),
        }

        return dashboard_metrics

    def capture_baseline_metrics(
        self, node_results: Dict[str, NodeValidationResult]
    ) -> Dict[str, float]:
        """
        Capture current performance metrics as baseline.

        Args:
            node_results: Node validation results

        Returns:
            Dictionary mapping node names to baseline execution times
        """
        baselines = {}
        for node_name, result in node_results.items():
            if result.success:
                baselines[node_name] = result.execution_time_ms

        return baselines

    def generate_report(
        self,
        node_results: Dict[str, NodeValidationResult],
        smoke_results: Dict[str, bool],
    ) -> IntegrationReport:
        """
        Generate comprehensive integration validation report.

        Args:
            node_results: Node validation results
            smoke_results: Smoke test results

        Returns:
            Complete integration report
        """
        total_nodes = len(node_results)
        passed_nodes = sum(1 for r in node_results.values() if r.success)
        failed_nodes = total_nodes - passed_nodes

        # Check overall success
        all_smoke_tests_passed = all(smoke_results.values())
        all_critical_smoke_tests_passed = all(
            smoke_results.get(name, False)
            for name, component in self.target_components.items()
            if component.critical
        )
        all_nodes_passed = passed_nodes == total_nodes

        overall_success = all_nodes_passed and all_critical_smoke_tests_passed

        # Performance metrics
        performance_metrics = {
            "total_validation_time_ms": sum(r.execution_time_ms for r in node_results.values()),
            "average_node_time_ms": sum(r.execution_time_ms for r in node_results.values())
            / total_nodes
            if total_nodes > 0
            else 0,
            "slowest_node": max(node_results.items(), key=lambda x: x[1].execution_time_ms)[0]
            if node_results
            else None,
            "cached_validations": sum(1 for r in node_results.values() if r.cached),
        }

        # Capture baseline metrics
        baseline_metrics = self.capture_baseline_metrics(node_results)

        # SLO compliance
        slo_compliance = self.check_slo_compliance(node_results)

        # Cache statistics
        cache_stats = self.validator.get_cache_stats()

        # Dashboard metrics
        dashboard_metrics = self.generate_dashboard_metrics(node_results, smoke_results)

        # Determine CI exit code
        ci_exit_code = 0 if overall_success else 1

        report = IntegrationReport(
            timestamp=datetime.now(),
            overall_success=overall_success,
            total_nodes=total_nodes,
            passed_nodes=passed_nodes,
            failed_nodes=failed_nodes,
            node_results=node_results,
            smoke_test_results=smoke_results,
            performance_metrics=performance_metrics,
            baseline_metrics=baseline_metrics,
            slo_compliance=slo_compliance,
            cache_stats=cache_stats,
            ci_exit_code=ci_exit_code,
            dashboard_metrics=dashboard_metrics,
        )

        return report

    def print_report(self, report: IntegrationReport):
        """
        Print human-readable validation report.

        Args:
            report: Integration validation report
        """
        print("\n" + "=" * 80)
        print("CANONICAL INTEGRATION VALIDATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp.isoformat()}")
        print(f"Overall Status: {'✓ PASSED' if report.overall_success else '✗ FAILED'}")
        print()

        # Smoke test results
        print("-" * 80)
        print("SMOKE TEST RESULTS (5 Target Components)")
        print("-" * 80)
        for component_name, success in report.smoke_test_results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            critical = " [CRITICAL]" if self.target_components[component_name].critical else ""
            print(f"  {component_name:30s} {status}{critical}")
        print()

        # Node validation results
        print("-" * 80)
        print("NODE VALIDATION RESULTS (11 Canonical Nodes)")
        print("-" * 80)
        for node_name in self.validator.execution_order:
            result = report.node_results.get(node_name)
            if result:
                status = "✓ PASS" if result.success else "✗ FAIL"
                cached = " [CACHED]" if result.cached else ""
                baseline = self.performance_baselines.get(node_name, 0)
                deviation = (
                    ((result.execution_time_ms - baseline) / baseline * 100)
                    if baseline > 0
                    else 0
                )
                print(
                    f"  {node_name:25s} {status}  "
                    f"{result.execution_time_ms:6.2f}ms "
                    f"(baseline: {baseline:6.2f}ms, {deviation:+.1f}%){cached}"
                )
                if result.errors:
                    for error in result.errors[:2]:  # Show first 2 errors
                        print(f"      Error: {error}")
        print()

        # Performance metrics
        print("-" * 80)
        print("PERFORMANCE METRICS")
        print("-" * 80)
        print(f"  Total Validation Time: {report.performance_metrics['total_validation_time_ms']:.2f}ms")
        print(f"  Average Node Time:     {report.performance_metrics['average_node_time_ms']:.2f}ms")
        print(f"  Slowest Node:          {report.performance_metrics['slowest_node']}")
        print(f"  Cached Validations:    {report.performance_metrics['cached_validations']}/{report.total_nodes}")
        if report.cache_stats:
            print(f"  Cache Hit Rate:        {report.cache_stats['hit_rate']:.1%}")
        print()

        # SLO compliance
        print("-" * 80)
        print("SLO COMPLIANCE")
        print("-" * 80)
        for slo_name, compliant in report.slo_compliance.items():
            status = "✓ COMPLIANT" if compliant else "✗ BREACH"
            print(f"  {slo_name:20s} {status}")
        print()

        # Summary
        print("-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"  Total Nodes:    {report.total_nodes}")
        print(f"  Passed:         {report.passed_nodes} ({report.passed_nodes/report.total_nodes*100:.1f}%)")
        print(f"  Failed:         {report.failed_nodes}")
        print(f"  CI Exit Code:   {report.ci_exit_code}")
        print("=" * 80)
        print()

    def save_report(self, report: IntegrationReport, output_path: Path):
        """
        Save validation report to JSON file.

        Args:
            report: Integration validation report
            output_path: Path to save report
        """
        # Convert dataclasses to dicts
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "overall_success": report.overall_success,
            "total_nodes": report.total_nodes,
            "passed_nodes": report.passed_nodes,
            "failed_nodes": report.failed_nodes,
            "node_results": {
                name: {
                    "node_name": result.node_name,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "input_valid": result.input_valid,
                    "output_valid": result.output_valid,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "cached": result.cached,
                }
                for name, result in report.node_results.items()
            },
            "smoke_test_results": report.smoke_test_results,
            "performance_metrics": report.performance_metrics,
            "baseline_metrics": report.baseline_metrics,
            "slo_compliance": report.slo_compliance,
            "cache_stats": report.cache_stats,
            "ci_exit_code": report.ci_exit_code,
            "dashboard_metrics": report.dashboard_metrics,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved to: {output_path}")

    def _generate_mock_data(self, data_type: DataType) -> Any:
        """Generate mock data for a given data type"""
        mock_data_map = {
            DataType.RAW_TEXT: "Sample development plan text for municipal evaluation and analysis. " * 5,
            DataType.SANITIZED_TEXT: "Sample development plan text for municipal evaluation and analysis. " * 5,
            DataType.SEGMENTS: [
                "This is segment one of the plan.",
                "This is segment two discussing objectives.",
                "This is segment three with budget information.",
            ],
            DataType.EMBEDDINGS: [[0.1] * 384 for _ in range(3)],  # Mock embeddings
            DataType.ENTITIES: [
                {"text": "Municipality Director", "type": "PERSON", "confidence": 0.95}
            ],
            DataType.CONTRADICTIONS: [],
            DataType.MONETARY_VALUES: [{"amount": 1000000, "currency": "COP", "context": "budget"}],
            DataType.FEASIBILITY_SCORES: {"technical": 0.8, "financial": 0.7, "social": 0.9},
            DataType.CAUSAL_PATTERNS: [
                {"cause": "Investment", "effect": "Development", "confidence": 0.85}
            ],
            DataType.TEORIA_CAMBIO: {
                "nodes": ["Input", "Activity", "Output", "Outcome"],
                "edges": [("Input", "Activity"), ("Activity", "Output"), ("Output", "Outcome")],
            },
            DataType.DAG_STRUCTURE: {"is_valid": True, "cycles": [], "disconnected": []},
            DataType.METADATA: {"author": "Test", "date": "2024-01-01", "version": "1.0"},
        }

        return mock_data_map.get(data_type, {})


def main():
    """Main entry point for validation script"""
    parser = argparse.ArgumentParser(
        description="Validate canonical integration of MINIMINIMOON pipeline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/canonical_integration_validation.json"),
        help="Output path for validation report",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable validation caching"
    )
    parser.add_argument(
        "--cache-size", type=int, default=1000, help="Validation cache size"
    )
    parser.add_argument(
        "--ci", action="store_true", help="CI mode (exit with error code on failure)"
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        help="Output path for performance baseline metrics",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=Path("reports/dashboard_metrics.json"),
        help="Output path for dashboard metrics",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dashboard_output:
        args.dashboard_output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize validator
    validator = CanonicalIntegrationValidator(
        enable_cache=not args.no_cache, cache_size=args.cache_size
    )

    try:
        # Run smoke tests for target components
        smoke_results = validator.run_smoke_tests()

        # Validate all canonical nodes
        node_results = validator.validate_all_nodes()

        # Generate report
        report = validator.generate_report(node_results, smoke_results)

        # Print report
        validator.print_report(report)

        # Save report
        validator.save_report(report, args.output)

        # Save dashboard metrics
        if args.dashboard_output:
            with open(args.dashboard_output, "w", encoding="utf-8") as f:
                json.dump(report.dashboard_metrics, f, indent=2)
            logger.info(f"Dashboard metrics saved to: {args.dashboard_output}")

        # Save baseline metrics if requested
        if args.baseline_output:
            with open(args.baseline_output, "w", encoding="utf-8") as f:
                json.dump(report.baseline_metrics, f, indent=2)
            logger.info(f"Baseline metrics saved to: {args.baseline_output}")

        # Exit with appropriate code
        if args.ci:
            sys.exit(report.ci_exit_code)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        logger.error(traceback.format_exc())
        if args.ci:
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()
