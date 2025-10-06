#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for enhanced MINIMINIMOON orchestrator features.
Tests trace IDs, metrics collection, health checks, and retry logic.
"""

import unittest
import time
from miniminimoon_orchestrator import (
    ExecutionContext,
    MetricsCollector,
    HealthChecker,
    MINIMINIMOONOrchestrator
)


class TestExecutionContext(unittest.TestCase):
    """Test ExecutionContext with trace IDs and intermediate outputs"""
    
    def test_trace_id_generation(self):
        """Test that trace IDs are generated as 32-char hex strings"""
        ctx = ExecutionContext()
        self.assertIsNotNone(ctx.trace_id)
        self.assertEqual(len(ctx.trace_id), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in ctx.trace_id))
    
    def test_trace_id_uniqueness(self):
        """Test that each context gets a unique trace ID"""
        ctx1 = ExecutionContext()
        ctx2 = ExecutionContext()
        self.assertNotEqual(ctx1.trace_id, ctx2.trace_id)
    
    def test_intermediate_output_storage(self):
        """Test storing intermediate outputs"""
        ctx = ExecutionContext()
        ctx.store_intermediate_output("node1", {"result": "test"})
        ctx.store_intermediate_output("node2", [1, 2, 3])
        
        self.assertEqual(len(ctx.intermediate_outputs), 2)
        self.assertIn("node1", ctx.intermediate_outputs)
        self.assertIn("node2", ctx.intermediate_outputs)
        self.assertEqual(ctx.intermediate_outputs["node1"]["output"], {"result": "test"})
        self.assertIn("trace_id", ctx.intermediate_outputs["node1"])
    
    def test_latency_tracking(self):
        """Test latency sample collection"""
        ctx = ExecutionContext()
        ctx.register_component_execution("comp1", 0.0, 0.5, "success")
        ctx.register_component_execution("comp2", 1.0, 1.8, "success")
        
        self.assertEqual(len(ctx.latency_samples), 2)
        self.assertAlmostEqual(ctx.latency_samples[0], 0.5, places=2)
        self.assertAlmostEqual(ctx.latency_samples[1], 0.8, places=2)
    
    def test_percentile_calculation(self):
        """Test p95 and p99 percentile calculation"""
        ctx = ExecutionContext()
        
        # Add 100 samples from 0.0 to 0.99
        for i in range(100):
            ctx.latency_samples.append(i / 100.0)
        
        percentiles = ctx.calculate_percentiles()
        
        self.assertIn("p95", percentiles)
        self.assertIn("p99", percentiles)
        self.assertIn("mean", percentiles)
        
        # P95 should be around 0.94-0.95
        self.assertGreater(percentiles["p95"], 0.90)
        self.assertLess(percentiles["p95"], 1.0)
        
        # P99 should be around 0.98-0.99
        self.assertGreater(percentiles["p99"], 0.95)
        self.assertLess(percentiles["p99"], 1.0)
    
    def test_execution_summary_includes_trace_id(self):
        """Test that execution summary includes trace ID"""
        ctx = ExecutionContext()
        summary = ctx.get_execution_summary()
        
        self.assertIn("trace_id", summary)
        self.assertEqual(summary["trace_id"], ctx.trace_id)
        self.assertIn("latency_p95", summary)
        self.assertIn("latency_p99", summary)


class TestMetricsCollector(unittest.TestCase):
    """Test Prometheus-compatible metrics collection"""
    
    def test_initialization(self):
        """Test MetricsCollector initialization"""
        metrics = MetricsCollector()
        
        self.assertEqual(len(metrics._get_all_components()), 11)
        self.assertIn("embedding", metrics._get_all_components())
        self.assertIn("dag_validation", metrics._get_all_components())
    
    def test_latency_recording(self):
        """Test latency histogram recording"""
        metrics = MetricsCollector()
        
        metrics.record_latency("embedding", 0.05)
        metrics.record_latency("embedding", 0.5)
        metrics.record_latency("embedding", 1.5)
        
        # Check buckets updated
        self.assertGreater(metrics.latency_histogram["embedding"][0.1], 0)
        self.assertGreater(metrics.latency_histogram["embedding"][1.0], 0)
        self.assertGreater(metrics.latency_histogram["embedding"][float('inf')], 0)
    
    def test_error_counting(self):
        """Test error counter recording"""
        metrics = MetricsCollector()
        
        metrics.record_error("embedding")
        metrics.record_error("embedding")
        metrics.record_error("dag_validation")
        
        self.assertEqual(metrics.error_counters["embedding"], 2)
        self.assertEqual(metrics.error_counters["dag_validation"], 1)
    
    def test_request_counting(self):
        """Test request throughput gauge"""
        metrics = MetricsCollector()
        
        for _ in range(5):
            metrics.record_request()
        
        self.assertEqual(metrics.total_requests, 5)
        self.assertEqual(metrics.throughput_gauge, 5)
    
    def test_prometheus_format(self):
        """Test Prometheus text exposition format"""
        metrics = MetricsCollector()
        
        metrics.record_latency("embedding", 0.5)
        metrics.record_error("embedding")
        metrics.record_request()
        
        output = metrics.get_prometheus_metrics()
        
        # Check format
        self.assertIn("# HELP miniminimoon_latency_seconds", output)
        self.assertIn("# TYPE miniminimoon_latency_seconds histogram", output)
        self.assertIn('miniminimoon_latency_seconds_bucket{component="embedding"', output)
        self.assertIn("# HELP miniminimoon_errors_total", output)
        self.assertIn("# TYPE miniminimoon_errors_total counter", output)
        self.assertIn('miniminimoon_errors_total{component="embedding"}', output)
        self.assertIn("miniminimoon_requests_total", output)


class TestHealthChecker(unittest.TestCase):
    """Test health check functionality"""
    
    def test_health_checker_initialization(self):
        """Test HealthChecker initialization requires orchestrator"""
        orchestrator = MINIMINIMOONOrchestrator()
        health_checker = HealthChecker(orchestrator)
        
        self.assertIsNotNone(health_checker.orchestrator)
    
    def test_component_health_check_speed(self):
        """Test that component health check completes quickly"""
        orchestrator = MINIMINIMOONOrchestrator()
        health_checker = HealthChecker(orchestrator)
        
        start = time.perf_counter()
        result = health_checker.check_component("embedding")
        duration_ms = (time.perf_counter() - start) * 1000
        
        self.assertLess(duration_ms, 100)  # Must be under 100ms
        self.assertIn("status", result)
        self.assertIn("component", result)
        self.assertIn("check_duration_ms", result)
    
    def test_all_components_health_check_speed(self):
        """Test that checking all 11 components completes under 100ms"""
        orchestrator = MINIMINIMOONOrchestrator()
        health_checker = HealthChecker(orchestrator)
        
        start = time.perf_counter()
        result = health_checker.check_all_components()
        duration_ms = (time.perf_counter() - start) * 1000
        
        self.assertLess(duration_ms, 100)  # Must be under 100ms total
        self.assertIn("overall_status", result)
        self.assertIn("components", result)
        self.assertEqual(result["total_components"], 11)
    
    def test_health_status_detection(self):
        """Test health status reflects component state"""
        orchestrator = MINIMINIMOONOrchestrator()
        health_checker = HealthChecker(orchestrator)
        
        # All components should be initialized/healthy after orchestrator init
        result = health_checker.check_all_components()
        
        self.assertGreater(result["healthy_components"], 0)
        self.assertEqual(len(result["components"]), 11)


class TestOrchestratorEnhancements(unittest.TestCase):
    """Test orchestrator enhancements"""
    
    def test_orchestrator_has_metrics_collector(self):
        """Test orchestrator initializes MetricsCollector"""
        orchestrator = MINIMINIMOONOrchestrator()
        
        self.assertIsNotNone(orchestrator.metrics_collector)
        self.assertIsInstance(orchestrator.metrics_collector, MetricsCollector)
    
    def test_orchestrator_has_health_checker(self):
        """Test orchestrator initializes HealthChecker"""
        orchestrator = MINIMINIMOONOrchestrator()
        
        self.assertIsNotNone(orchestrator.health_checker)
        self.assertIsInstance(orchestrator.health_checker, HealthChecker)
    
    def test_get_health_method(self):
        """Test get_health() method"""
        orchestrator = MINIMINIMOONOrchestrator()
        health = orchestrator.get_health()
        
        self.assertIn("overall_status", health)
        self.assertIn("components", health)
        self.assertEqual(health["total_components"], 11)
    
    def test_get_metrics_method(self):
        """Test get_metrics() method returns Prometheus format"""
        orchestrator = MINIMINIMOONOrchestrator()
        metrics = orchestrator.get_metrics()
        
        self.assertIsInstance(metrics, str)
        self.assertIn("miniminimoon_latency_seconds", metrics)
        self.assertIn("miniminimoon_errors_total", metrics)
    
    def test_context_has_trace_id(self):
        """Test orchestrator context has trace ID"""
        orchestrator = MINIMINIMOONOrchestrator()
        
        self.assertIsNotNone(orchestrator.context.trace_id)
        self.assertEqual(len(orchestrator.context.trace_id), 32)


if __name__ == "__main__":
    unittest.main()
