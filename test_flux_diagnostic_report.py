#!/usr/bin/env python3
"""
Test suite for flux diagnostic report generator.
"""

import json
import pytest
import tempfile
from pathlib import Path
from generate_flux_diagnostic_report import (
    format_bytes,
    format_latency,
    format_throughput,
    assess_node_health,
    identify_top_risks,
    generate_report
)


def test_format_bytes():
    """Test byte formatting."""
    assert format_bytes(512) == "512.0 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(1024**2 * 5) == "5.0 MB"
    assert format_bytes(1024**3 * 1.5) == "1.50 GB"


def test_format_latency():
    """Test latency formatting."""
    assert "μs" in format_latency(0.0001)
    assert "ms" in format_latency(0.05)
    assert "s" in format_latency(2.5)


def test_format_throughput():
    """Test throughput formatting."""
    assert "items/s" in format_throughput(10.5)
    assert "k items/s" in format_throughput(5000)


def test_assess_node_health():
    """Test node health assessment."""
    # PASS case
    node = {"latency_ms": 100, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.01}
    status, reason = assess_node_health(node)
    assert status == "PASS"
    
    # WARN case - high latency
    node = {"latency_ms": 2500, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.01}
    status, reason = assess_node_health(node)
    assert status == "WARN"
    assert "latency" in reason.lower()
    
    # FAIL case - high error rate
    node = {"latency_ms": 100, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.1}
    status, reason = assess_node_health(node)
    assert status == "FAIL"
    assert "error" in reason.lower()
    
    # FAIL case - excessive latency
    node = {"latency_ms": 6000, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.01}
    status, reason = assess_node_health(node)
    assert status == "FAIL"


def test_identify_top_risks():
    """Test risk identification."""
    nodes = {
        "stage_1": {"latency_ms": 100, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.01},
        "stage_2": {"latency_ms": 6000, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.01},
        "stage_3": {"latency_ms": 100, "peak_memory_mb": 256, "throughput": 50, "error_rate": 0.15},
    }
    connections = {
        "stage_1->stage_2": {"stability": 0.99, "type_mismatches": []},
        "stage_2->stage_3": {"stability": 0.75, "type_mismatches": ["field_a", "field_b"]},
    }
    
    risks = identify_top_risks(nodes, connections)
    
    # Should identify critical error rate, performance degradation, and unstable connection
    assert len(risks) >= 3
    assert any("Error Rate" in r[1] for r in risks)
    # stage_2 has 6000ms latency which is a FAIL but gets classified as "Performance Degradation"
    assert any("Performance" in r[1] or "Degradation" in r[1] for r in risks)
    assert any("Unstable" in r[1] for r in risks)
    
    # Should be sorted by severity (descending)
    assert all(risks[i][0] >= risks[i+1][0] for i in range(len(risks)-1))


def test_generate_report():
    """Test full report generation."""
    # Create sample JSON data
    test_data = {
        "nodes": {
            "plan_sanitizer": {
                "latency_ms": 150,
                "peak_memory_mb": 128,
                "throughput": 100,
                "error_rate": 0.005
            },
            "document_segmenter": {
                "latency_ms": 2200,
                "peak_memory_mb": 512,
                "throughput": 20,
                "error_rate": 0.01
            },
            "embedding_model": {
                "latency_ms": 450,
                "peak_memory_mb": 768,
                "throughput": 30,
                "error_rate": 0.002
            },
        },
        "connections": {
            "sanitizer->segmenter": {"stability": 0.99, "type_mismatches": []},
            "segmenter->embedder": {"stability": 0.95, "type_mismatches": ["field_x"]},
        },
        "output_quality": {
            "determinism_verified": True,
            "determinism_runs": {
                "run_1": "abc123",
                "run_2": "abc123"
            },
            "question_coverage": 300,
            "rubric_check_exit_code": 0,
            "rubric_aligned": True,
            "all_gates_passed": True,
            "failed_gates": []
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_diagnostic.json"
        output_path = Path(tmpdir) / "test_report.md"
        
        # Write test JSON
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Generate report
        success = generate_report(json_path, output_path)
        assert success
        assert output_path.exists()
        
        # Verify report content
        content = output_path.read_text()
        assert "Pipeline Flux Diagnostic Report" in content
        assert "Executive Summary" in content
        assert "Node-by-Node Performance" in content
        assert "Inter-Node Connection Assessment" in content
        assert "Final Output Quality" in content
        assert "Top 5 Risks" in content
        assert "Top 5 Recommended Fixes" in content
        
        # Verify node data appears
        assert "plan_sanitizer" in content
        assert "document_segmenter" in content
        assert "embedding_model" in content
        
        # Verify output quality results
        assert "300/300" in content
        assert "✓" in content  # Pass indicators


def test_generate_report_with_failures():
    """Test report generation with failures."""
    test_data = {
        "nodes": {
            "failing_stage": {
                "latency_ms": 12000,
                "peak_memory_mb": 5000,
                "throughput": 0.5,
                "error_rate": 0.2
            }
        },
        "connections": {
            "conn_1": {"stability": 0.60, "type_mismatches": ["a", "b", "c", "d", "e", "f"]}
        },
        "output_quality": {
            "determinism_verified": False,
            "question_coverage": 250,
            "rubric_check_exit_code": 3,
            "rubric_aligned": False,
            "all_gates_passed": False,
            "failed_gates": ["Gate 1", "Gate 4"]
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_diagnostic.json"
        output_path = Path(tmpdir) / "test_report.md"
        
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        success = generate_report(json_path, output_path)
        assert success
        
        content = output_path.read_text()
        
        # Should show failures
        assert "FAIL" in content or "CRITICAL" in content
        assert "250/300" in content
        assert "Gate 1" in content
        assert "Gate 4" in content


def test_generate_report_missing_file():
    """Test report generation with missing input file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "nonexistent.json"
        output_path = Path(tmpdir) / "output.md"
        
        success = generate_report(json_path, output_path)
        assert not success


def test_generate_report_invalid_json():
    """Test report generation with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "invalid.json"
        output_path = Path(tmpdir) / "output.md"
        
        # Write invalid JSON
        with open(json_path, 'w') as f:
            f.write("{invalid json content")
        
        success = generate_report(json_path, output_path)
        assert not success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
