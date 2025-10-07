#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for diagnostic_runner.py
Validates diagnostic report structure and basic functionality.
"""

import unittest
from pathlib import Path
from dataclasses import asdict

from diagnostic_runner import (
    NodeMetrics,
    DiagnosticReport,
    generate_reports
)


class TestDiagnosticRunner(unittest.TestCase):
    """Test diagnostic runner components."""
    
    def test_node_metrics_creation(self):
        """Test NodeMetrics dataclass creation."""
        metric = NodeMetrics(
            node_name="test_node",
            start_time=0.0,
            end_time=1.0,
            duration_ms=1000.0,
            status="success",
            input_state={"key": "value"}
        )
        
        self.assertEqual(metric.node_name, "test_node")
        self.assertEqual(metric.duration_ms, 1000.0)
        self.assertEqual(metric.status, "success")
        self.assertIsInstance(metric.input_state, dict)
    
    def test_diagnostic_report_creation(self):
        """Test DiagnosticReport dataclass creation."""
        metrics = [
            NodeMetrics("node1", 0.0, 1.0, 1000.0, "success"),
            NodeMetrics("node2", 1.0, 2.0, 1000.0, "success")
        ]
        
        report = DiagnosticReport(
            total_execution_time_ms=2000.0,
            node_metrics=metrics,
            connection_stability={"status": "stable"},
            output_quality={"quality_status": "passed"},
            determinism_check={"deterministic": True},
            status="success"
        )
        
        self.assertEqual(report.total_execution_time_ms, 2000.0)
        self.assertEqual(len(report.node_metrics), 2)
        self.assertEqual(report.status, "success")
        self.assertTrue(report.determinism_check["deterministic"])
    
    def test_diagnostic_report_to_dict(self):
        """Test DiagnosticReport serialization to dict."""
        metrics = [NodeMetrics("test", 0.0, 1.0, 1000.0, "success")]
        report = DiagnosticReport(
            total_execution_time_ms=1000.0,
            node_metrics=metrics,
            connection_stability={},
            output_quality={},
            determinism_check={},
            status="success"
        )
        
        report_dict = asdict(report)
        self.assertIsInstance(report_dict, dict)
        self.assertIn("total_execution_time_ms", report_dict)
        self.assertIn("node_metrics", report_dict)
        self.assertIn("status", report_dict)
    
    def test_report_generation(self):
        """Test report file generation."""
        import tempfile
        import json
        
        metrics = [
            NodeMetrics("node1", 0.0, 0.5, 500.0, "success"),
            NodeMetrics("node2", 0.5, 1.5, 1000.0, "success"),
            NodeMetrics("node3", 1.5, 2.0, 500.0, "success")
        ]
        
        report = DiagnosticReport(
            total_execution_time_ms=2000.0,
            node_metrics=metrics,
            connection_stability={"warmup_duration_ms": 100.0, "status": "stable"},
            output_quality={"quality_status": "passed", "segments_count": 10},
            determinism_check={"deterministic": True},
            status="success"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            json_path, md_path = generate_reports(report, output_dir)
            
            # Check files were created
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            
            # Validate JSON content
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            self.assertEqual(json_data["status"], "success")
            self.assertEqual(json_data["total_execution_time_ms"], 2000.0)
            self.assertEqual(len(json_data["node_metrics"]), 3)
            
            # Validate Markdown content
            with open(md_path, 'r') as f:
                md_content = f.read()
            
            self.assertIn("# MINIMINIMOON Diagnostic Report", md_content)
            self.assertIn("**Status**: SUCCESS", md_content)
            self.assertIn("Connection Stability", md_content)
            self.assertIn("Output Quality", md_content)
            self.assertIn("Determinism Check", md_content)
            self.assertIn("Node Execution Metrics", md_content)
    
    def test_failed_report_generation(self):
        """Test report generation for failed runs."""
        import tempfile
        
        metrics = [
            NodeMetrics("node1", 0.0, 0.5, 500.0, "success"),
            NodeMetrics("node2", 0.5, 1.0, 500.0, "failed", 
                       error_msg="Test error message")
        ]
        
        report = DiagnosticReport(
            total_execution_time_ms=1000.0,
            node_metrics=metrics,
            connection_stability={"status": "failed"},
            output_quality={"quality_status": "failed"},
            determinism_check={"deterministic": False},
            status="failed",
            error_summary="Pipeline execution failed"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            json_path, md_path = generate_reports(report, output_dir)
            
            # Validate Markdown content includes error info
            with open(md_path, 'r') as f:
                md_content = f.read()
            
            self.assertIn("**Status**: FAILED", md_content)
            self.assertIn("Error Summary", md_content)
            self.assertIn("Pipeline execution failed", md_content)
            self.assertIn("Failed Nodes", md_content)
            self.assertIn("node2", md_content)
            self.assertIn("Test error message", md_content)


if __name__ == "__main__":
    unittest.main()
