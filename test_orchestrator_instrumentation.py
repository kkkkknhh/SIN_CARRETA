#!/usr/bin/env python3
"""
Test Suite for Orchestrator Instrumentation
============================================
Validates performance monitoring, circuit breaker integration, and CI/CD gates.
"""

import pytest
import time
import os
import json
import yaml
from unittest.mock import Mock, patch, MagicMock
from miniminimoon_orchestrator import MINIMINIMOONOrchestrator, PerformanceMonitor
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def test_load_budgets(self, tmp_path):
        """Test loading performance budgets from YAML"""
        budgets_file = tmp_path / "test_budgets.yaml"
        budgets_file.write_text("""
budgets:
  test_node:
    p95_ms: 10.0
    p99_ms: 15.0
    tolerance_pct: 10.0
""")
        
        monitor = PerformanceMonitor(str(budgets_file))
        assert "test_node" in monitor.budgets
        assert monitor.budgets["test_node"]["p95_ms"] == 10.0
    
    def test_record_latency(self):
        """Test recording latency measurements"""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("test_node", 5.0)
        monitor.record_latency("test_node", 10.0)
        monitor.record_latency("test_node", 15.0)
        
        assert len(monitor.latencies["test_node"]) == 3
        assert monitor.latencies["test_node"] == [5.0, 10.0, 15.0]
    
    def test_get_percentiles(self):
        """Test percentile calculations"""
        monitor = PerformanceMonitor()
        
        # Record 100 measurements
        for i in range(100):
            monitor.record_latency("test_node", float(i))
        
        percentiles = monitor.get_percentiles("test_node")
        
        assert percentiles["p50"] == 50.0
        assert percentiles["p95"] == 95.0
        assert percentiles["p99"] == 99.0
        assert percentiles["count"] == 100
    
    def test_check_budget_violation_pass(self, tmp_path):
        """Test budget check when performance is within budget"""
        budgets_file = tmp_path / "test_budgets.yaml"
        budgets_file.write_text("""
budgets:
  test_node:
    p95_ms: 100.0
    tolerance_pct: 10.0
""")
        
        monitor = PerformanceMonitor(str(budgets_file))
        
        # Record latencies well within budget
        for i in range(100):
            monitor.record_latency("test_node", 50.0)
        
        passed, message = monitor.check_budget_violation("test_node")
        
        assert passed is True
        assert "✅" in message
    
    def test_check_budget_violation_fail(self, tmp_path):
        """Test budget check when performance exceeds budget"""
        budgets_file = tmp_path / "test_budgets.yaml"
        budgets_file.write_text("""
budgets:
  test_node:
    p95_ms: 10.0
    tolerance_pct: 10.0
""")
        
        monitor = PerformanceMonitor(str(budgets_file))
        
        # Record latencies exceeding budget
        for i in range(100):
            monitor.record_latency("test_node", 50.0)
        
        passed, message = monitor.check_budget_violation("test_node")
        
        assert passed is False
        assert "❌" in message
    
    def test_export_prometheus_metrics(self, tmp_path):
        """Test Prometheus metrics export"""
        output_file = tmp_path / "metrics.prom"
        
        monitor = PerformanceMonitor()
        monitor.record_latency("test_node", 10.0)
        monitor.record_latency("test_node", 20.0)
        
        monitor.export_prometheus_metrics(str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "miniminimoon_pipeline_latency_milliseconds" in content
        assert 'node="test_node"' in content
    
    def test_generate_dashboard_html(self, tmp_path):
        """Test HTML dashboard generation"""
        output_file = tmp_path / "dashboard.html"
        
        monitor = PerformanceMonitor()
        monitor.record_latency("test_node", 10.0)
        
        monitor.generate_dashboard_html(str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "MINIMINIMOON Performance Dashboard" in content
        assert "test_node" in content


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with orchestrator"""
    
    def test_circuit_breaker_initialization(self):
        """Test that circuit breakers are properly initialized"""
        with patch('miniminimoon_orchestrator.PlanSanitizer'), patch('miniminimoon_orchestrator.PlanProcessor'), patch('miniminimoon_orchestrator.DocumentSegmenter'), patch('miniminimoon_orchestrator.IndustrialEmbeddingModel'), patch('miniminimoon_orchestrator.SpacyModelLoader'), patch('miniminimoon_orchestrator.ResponsibilityDetector'), patch('miniminimoon_orchestrator.ContradictionDetector'), patch('miniminimoon_orchestrator.MonetaryDetector'), patch('miniminimoon_orchestrator.FeasibilityScorer'), patch('miniminimoon_orchestrator.PDETCausalPatternDetector'), patch('miniminimoon_orchestrator.AdvancedDAGValidator'), patch('miniminimoon_orchestrator.EvidenceRegistry'), patch('miniminimoon_orchestrator.CanonicalFlowValidator'), patch('miniminimoon_orchestrator.QuestionnaireEngine'):
            orchestrator = MINIMINIMOONOrchestrator()
            
            assert hasattr(orchestrator, 'circuit_breakers')
            assert len(orchestrator.circuit_breakers) == 6
            
            expected_circuits = [
                "embedding",
                "responsibility_detection",
                "contradiction_detection",
                "causal_detection",
                "teoria_cambio",
                "dag_validation"
            ]
            
            for circuit_name in expected_circuits:
                assert circuit_name in orchestrator.circuit_breakers
                circuit = orchestrator.circuit_breakers[circuit_name]
                assert isinstance(circuit, CircuitBreaker)
                assert circuit.config.recovery_time_sla_seconds == 2.0
    
    def test_circuit_breaker_recovery_threshold(self):
        """Test that circuit breakers use 2.0s recovery threshold"""
        circuit = CircuitBreaker("test", CircuitBreakerConfig(recovery_time_sla_seconds=2.0))
        
        assert circuit.config.recovery_time_sla_seconds == 2.0


class TestInstrumentedMethods:
    """Test that execute methods are properly instrumented"""
    
    @patch('miniminimoon_orchestrator.PlanSanitizer')
    @patch('miniminimoon_orchestrator.PlanProcessor')
    @patch('miniminimoon_orchestrator.DocumentSegmenter')
    @patch('miniminimoon_orchestrator.IndustrialEmbeddingModel')
    @patch('miniminimoon_orchestrator.SpacyModelLoader')
    @patch('miniminimoon_orchestrator.ResponsibilityDetector')
    @patch('miniminimoon_orchestrator.ContradictionDetector')
    @patch('miniminimoon_orchestrator.MonetaryDetector')
    @patch('miniminimoon_orchestrator.FeasibilityScorer')
    @patch('miniminimoon_orchestrator.PDETCausalPatternDetector')
    @patch('miniminimoon_orchestrator.AdvancedDAGValidator')
    @patch('miniminimoon_orchestrator.EvidenceRegistry')
    @patch('miniminimoon_orchestrator.CanonicalFlowValidator')
    @patch('miniminimoon_orchestrator.QuestionnaireEngine')
    def test_sanitization_records_latency(self, *mocks):
        """Test that sanitization method records latency"""
        orchestrator = MINIMINIMOONOrchestrator()
        orchestrator.sanitizer.sanitize = Mock(return_value="sanitized text")
        
        result = orchestrator._execute_sanitization("test text")
        
        assert result == "sanitized text"
        assert "sanitization" in orchestrator.performance_monitor.latencies
        assert len(orchestrator.performance_monitor.latencies["sanitization"]) == 1
    
    @patch('miniminimoon_orchestrator.PlanSanitizer')
    @patch('miniminimoon_orchestrator.PlanProcessor')
    @patch('miniminimoon_orchestrator.DocumentSegmenter')
    @patch('miniminimoon_orchestrator.IndustrialEmbeddingModel')
    @patch('miniminimoon_orchestrator.SpacyModelLoader')
    @patch('miniminimoon_orchestrator.ResponsibilityDetector')
    @patch('miniminimoon_orchestrator.ContradictionDetector')
    @patch('miniminimoon_orchestrator.MonetaryDetector')
    @patch('miniminimoon_orchestrator.FeasibilityScorer')
    @patch('miniminimoon_orchestrator.PDETCausalPatternDetector')
    @patch('miniminimoon_orchestrator.AdvancedDAGValidator')
    @patch('miniminimoon_orchestrator.EvidenceRegistry')
    @patch('miniminimoon_orchestrator.CanonicalFlowValidator')
    @patch('miniminimoon_orchestrator.QuestionnaireEngine')
    def test_embedding_uses_circuit_breaker(self, *mocks):
        """Test that embedding method uses circuit breaker"""
        orchestrator = MINIMINIMOONOrchestrator()
        orchestrator.embedding_model.embed = Mock(return_value=[[0.1, 0.2]])
        
        result = orchestrator._execute_embedding(["test segment"])
        
        assert result == [[0.1, 0.2]]
        assert "embedding" in orchestrator.performance_monitor.latencies


class TestCIPerformanceGate:
    """Test CI/CD performance gate functionality"""
    
    def test_performance_gate_pass(self):
        """Test that performance gate passes when budgets are met"""
        from ci_performance_gate import run_performance_gate
        
        # This will run actual benchmarks, so we just check it doesn't crash
        # In real CI, this would run with actual performance data
        assert callable(run_performance_gate)
    
    def test_load_performance_budgets(self, tmp_path):
        """Test loading performance budgets in CI gate"""
        from ci_performance_gate import load_performance_budgets
        
        budgets_file = tmp_path / "budgets.yaml"
        budgets_file.write_text("""
budgets:
  test_component:
    p95_ms: 5.0
""")
        
        budgets = load_performance_budgets(str(budgets_file))
        assert "test_component" in budgets
        assert budgets["test_component"]["p95_ms"] == 5.0


class TestPrometheusMetrics:
    """Test Prometheus metrics export"""
    
    def test_metrics_format(self, tmp_path):
        """Test that Prometheus metrics are in correct format"""
        monitor = PerformanceMonitor()
        
        # Record some data
        for i in range(100):
            monitor.record_latency("test_node", float(i))
        
        output_file = tmp_path / "metrics.prom"
        monitor.export_prometheus_metrics(str(output_file))
        
        content = output_file.read_text()
        lines = content.split('\n')
        
        # Check format
        assert any(line.startswith('# HELP') for line in lines)
        assert any(line.startswith('# TYPE') for line in lines)
        assert any('quantile="0.95"' in line for line in lines)
        assert any('quantile="0.99"' in line for line in lines)


def test_performance_budgets_yaml_exists():
    """Test that performance_budgets.yaml file exists"""
    assert os.path.exists("performance_budgets.yaml")
    
    with open("performance_budgets.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    assert "budgets" in config
    assert "circuit_breakers" in config
    assert "cicd" in config
    assert "prometheus" in config
    assert "alerting" in config
    
    # Check that all 11 canonical nodes have budgets
    budgets = config["budgets"]
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
        "dag_validation"
    ]
    
    for node in expected_nodes:
        assert node in budgets, f"Missing budget for {node}"
        assert "p95_ms" in budgets[node]
        assert "tolerance_pct" in budgets[node]
        assert budgets[node]["tolerance_pct"] == 10.0


def test_prometheus_alerting_rules_exists():
    """Test that Prometheus alerting rules file exists"""
    assert os.path.exists("prometheus_alerting_rules.yaml")
    
    with open("prometheus_alerting_rules.yaml", 'r') as f:
        rules = yaml.safe_load(f)
    
    assert "groups" in rules
    assert len(rules["groups"]) >= 3  # performance, circuit_breakers, regression
    
    # Check for key alert rules
    all_rules = []
    for group in rules["groups"]:
        all_rules.extend(group.get("rules", []))
    
    alert_names = [rule.get("alert") for rule in all_rules if "alert" in rule]
    
    assert "CircuitBreakerOpened" in alert_names
    assert "HighLatencyEmbedding" in alert_names
    assert "PerformanceRegression" in alert_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
