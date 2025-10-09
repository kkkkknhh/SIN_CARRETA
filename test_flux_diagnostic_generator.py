#!/usr/bin/env python3
"""Comprehensive test suite for the flux diagnostic report generator."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add reports directory to path
sys.path.insert(0, str(Path(__file__).parent / "reports"))

from flux_diagnostic_generator import FluxDiagnosticGenerator


class TestFluxDiagnosticGenerator:
    """Test suite for diagnostic report generator"""
    
    @pytest.fixture
    def valid_diagnostic_data(self):
        """Valid pipeline diagnostic data"""
        return {
            "pipeline_stages": [
                {
                    "stage_name": "document_segmentation",
                    "latency_ms": 120.5,
                    "cpu_ms": 95.2,
                    "peak_memory_mb": 256.8,
                    "throughput": 8.3,
                    "cache_hits": 42,
                    "contract_checks": {"pass": 5, "fail": 0}
                },
                {
                    "stage_name": "embedding_generation",
                    "latency_ms": 450.2,
                    "cpu_ms": 420.1,
                    "peak_memory_mb": 1024.3,
                    "throughput": 2.2,
                    "cache_hits": 128,
                    "contract_checks": {"pass": 8, "fail": 0}
                },
                {
                    "stage_name": "semantic_search",
                    "latency_ms": 85.3,
                    "cpu_ms": 75.1,
                    "peak_memory_mb": 512.2,
                    "throughput": 11.7,
                    "cache_hits": 256,
                    "contract_checks": {"pass": 6, "fail": 0}
                },
                {
                    "stage_name": "evidence_extraction",
                    "latency_ms": 320.8,
                    "cpu_ms": 280.5,
                    "peak_memory_mb": 768.4,
                    "throughput": 3.1,
                    "cache_hits": 64,
                    "contract_checks": {"pass": 12, "fail": 1}
                },
                {
                    "stage_name": "responsibility_detection",
                    "latency_ms": 210.4,
                    "cpu_ms": 190.3,
                    "peak_memory_mb": 512.1,
                    "throughput": 4.8,
                    "cache_hits": 32,
                    "contract_checks": {"pass": 7, "fail": 0}
                },
                {
                    "stage_name": "temporal_analysis",
                    "latency_ms": 95.6,
                    "cpu_ms": 85.2,
                    "peak_memory_mb": 256.5,
                    "throughput": 10.5,
                    "cache_hits": 16,
                    "contract_checks": {"pass": 4, "fail": 0}
                },
                {
                    "stage_name": "causal_graph_construction",
                    "latency_ms": 540.3,
                    "cpu_ms": 510.8,
                    "peak_memory_mb": 1536.2,
                    "throughput": 1.9,
                    "cache_hits": 8,
                    "contract_checks": {"pass": 15, "fail": 2}
                },
                {
                    "stage_name": "intervention_identification",
                    "latency_ms": 180.7,
                    "cpu_ms": 165.3,
                    "peak_memory_mb": 512.8,
                    "throughput": 5.5,
                    "cache_hits": 24,
                    "contract_checks": {"pass": 9, "fail": 0}
                },
                {
                    "stage_name": "outcome_mapping",
                    "latency_ms": 220.5,
                    "cpu_ms": 200.1,
                    "peak_memory_mb": 640.3,
                    "throughput": 4.5,
                    "cache_hits": 16,
                    "contract_checks": {"pass": 11, "fail": 1}
                },
                {
                    "stage_name": "rubric_alignment",
                    "latency_ms": 150.2,
                    "cpu_ms": 135.8,
                    "peak_memory_mb": 384.6,
                    "throughput": 6.7,
                    "cache_hits": 48,
                    "contract_checks": {"pass": 6, "fail": 0}
                },
                {
                    "stage_name": "scoring_computation",
                    "latency_ms": 75.4,
                    "cpu_ms": 68.9,
                    "peak_memory_mb": 128.2,
                    "throughput": 13.3,
                    "cache_hits": 96,
                    "contract_checks": {"pass": 5, "fail": 0}
                },
                {
                    "stage_name": "contract_validation",
                    "latency_ms": 110.8,
                    "cpu_ms": 98.5,
                    "peak_memory_mb": 256.4,
                    "throughput": 9.0,
                    "cache_hits": 72,
                    "contract_checks": {"pass": 18, "fail": 0}
                },
                {
                    "stage_name": "determinism_verification",
                    "latency_ms": 65.3,
                    "cpu_ms": 58.7,
                    "peak_memory_mb": 192.1,
                    "throughput": 15.3,
                    "cache_hits": 128,
                    "contract_checks": {"pass": 4, "fail": 0}
                },
                {
                    "stage_name": "coverage_validation",
                    "latency_ms": 85.7,
                    "cpu_ms": 75.2,
                    "peak_memory_mb": 256.8,
                    "throughput": 11.7,
                    "cache_hits": 64,
                    "contract_checks": {"pass": 5, "fail": 0}
                },
                {
                    "stage_name": "report_generation",
                    "latency_ms": 45.2,
                    "cpu_ms": 38.6,
                    "peak_memory_mb": 128.5,
                    "throughput": 22.1,
                    "cache_hits": 32,
                    "contract_checks": {"pass": 3, "fail": 0}
                }
            ],
            "coverage_300": {
                "met": True,
                "actual_count": 315,
                "required_count": 300
            },
            "quality_metrics": {
                "confidence_scores": {
                    "overall": 0.87,
                    "evidence_extraction": 0.92,
                    "responsibility_detection": 0.85,
                    "intervention_mapping": 0.84
                },
                "evidence_completeness": {
                    "percentage": 94.5,
                    "missing_categories": ["long_term_impacts"]
                }
            }
        }
    
    @pytest.fixture
    def valid_connection_data(self):
        """Valid connection stability data with 72 contracts"""
        connections = []
        
        # Generate 72 documented flow contracts
        flow_pairs = [
            ("document_segmentation", "embedding_generation"),
            ("embedding_generation", "semantic_search"),
            ("semantic_search", "evidence_extraction"),
            ("evidence_extraction", "responsibility_detection"),
            ("responsibility_detection", "temporal_analysis"),
            ("temporal_analysis", "causal_graph_construction"),
            ("causal_graph_construction", "intervention_identification"),
            ("intervention_identification", "outcome_mapping"),
            ("outcome_mapping", "rubric_alignment"),
            ("rubric_alignment", "scoring_computation"),
            ("scoring_computation", "contract_validation"),
            ("contract_validation", "determinism_verification"),
            ("determinism_verification", "coverage_validation"),
            ("coverage_validation", "report_generation")
        ]
        
        # Create 72 connections (5+ per flow pair with various types)
        contract_types = ["data", "interface", "stability", "performance", "determinism"]
        verdicts = ["SUITABLE", "SUITABLE", "SUITABLE", "UNSTABLE", "SUITABLE"]
        
        for from_node, to_node in flow_pairs:
            for i, ctype in enumerate(contract_types):
                connections.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "contract_type": ctype,
                    "interface_check": {
                        "passed": verdicts[i] == "SUITABLE",
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    "stability_rate": 98.5 if verdicts[i] == "SUITABLE" else 85.2,
                    "suitability_verdict": verdicts[i],
                    "mismatch_examples": [] if verdicts[i] == "SUITABLE" else [
                        {
                            "expected": "dict",
                            "actual": "list",
                            "location": f"{from_node}.output"
                        }
                    ]
                })
        
        # Add 2 more to reach exactly 72
        connections.append({
            "from_node": "embedding_generation",
            "to_node": "causal_graph_construction",
            "contract_type": "cache",
            "interface_check": {"passed": True, "timestamp": "2024-01-15T10:30:00Z"},
            "stability_rate": 99.2,
            "suitability_verdict": "SUITABLE",
            "mismatch_examples": []
        })
        connections.append({
            "from_node": "evidence_extraction",
            "to_node": "scoring_computation",
            "contract_type": "bypass",
            "interface_check": {"passed": True, "timestamp": "2024-01-15T10:30:00Z"},
            "stability_rate": 97.8,
            "suitability_verdict": "SUITABLE",
            "mismatch_examples": []
        })
        
        return {"connections": connections}
    
    @pytest.fixture
    def valid_determinism_data(self):
        """Valid determinism verification data"""
        return {
            "determinism_verified": True,
            "runs_compared": 10,
            "hash_matches": 10,
            "variance": 0.0
        }
    
    def test_valid_data_generates_report(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that valid data generates a complete report"""
        generator = FluxDiagnosticGenerator(
            diagnostic_data=valid_diagnostic_data,
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data,
            rubric_exit_code=0
        )
        
        report = generator.generate_report()
        
        assert "pipeline_stages" in report
        assert "connections" in report
        assert "final_output" in report
        assert "environment" in report
        
        assert len(report["pipeline_stages"]) == 15
        assert len(report["connections"]) == 72
        
    def test_missing_pipeline_stages_fails_validation(self, valid_connection_data, valid_determinism_data):
        """Test that missing pipeline_stages key fails validation"""
        generator = FluxDiagnosticGenerator(
            diagnostic_data={},
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_report()
            
        assert "pipeline_stages" in str(exc_info.value)
        
    def test_missing_stage_fields_fails_validation(self, valid_connection_data, valid_determinism_data):
        """Test that stages with missing required fields fail validation"""
        incomplete_data = {
            "pipeline_stages": [
                {
                    "stage_name": "document_segmentation",
                    "latency_ms": 120.5
                    # Missing other required fields
                }
            ]
        }
        
        generator = FluxDiagnosticGenerator(
            diagnostic_data=incomplete_data,
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_report()
            
        assert "missing fields" in str(exc_info.value).lower()
        
    def test_missing_expected_stages_fails_validation(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that missing expected stages fails validation"""
        # Remove a required stage
        incomplete_stages = [s for s in valid_diagnostic_data["pipeline_stages"] if s["stage_name"] != "rubric_alignment"]
        
        generator = FluxDiagnosticGenerator(
            diagnostic_data={"pipeline_stages": incomplete_stages},
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_report()
            
        assert "Missing expected stages" in str(exc_info.value)
        
    def test_invalid_suitability_verdict_fails(self, valid_diagnostic_data, valid_determinism_data):
        """Test that invalid verdict enum fails validation"""
        invalid_connections = {
            "connections": [
                {
                    "from_node": "stage1",
                    "to_node": "stage2",
                    "interface_check": {"passed": True},
                    "stability_rate": 95.0,
                    "suitability_verdict": "INVALID_VERDICT",
                    "mismatch_examples": []
                }
            ] * 72  # Ensure 72 connections
        }
        
        generator = FluxDiagnosticGenerator(
            diagnostic_data=valid_diagnostic_data,
            connection_data=invalid_connections,
            determinism_data=valid_determinism_data
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_report()
            
        assert "Invalid suitability_verdict" in str(exc_info.value)
        
    def test_insufficient_connections_fails(self, valid_diagnostic_data, valid_determinism_data):
        """Test that fewer than 72 connections fails validation"""
        insufficient_connections = {
            "connections": [
                {
                    "from_node": "stage1",
                    "to_node": "stage2",
                    "interface_check": {"passed": True},
                    "stability_rate": 95.0,
                    "suitability_verdict": "SUITABLE",
                    "mismatch_examples": []
                }
            ] * 50  # Only 50 connections
        }
        
        generator = FluxDiagnosticGenerator(
            diagnostic_data=valid_diagnostic_data,
            connection_data=insufficient_connections,
            determinism_data=valid_determinism_data
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_report()
            
        assert "Expected 72 flow contracts" in str(exc_info.value)
        
    def test_environment_metadata_collection(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that environment metadata is properly collected"""
        generator = FluxDiagnosticGenerator(
            diagnostic_data=valid_diagnostic_data,
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data,
            rubric_exit_code=0
        )
        
        env = generator.collect_environment_metadata()
        
        assert "repo_commit_hash" in env
        assert "python_version" in env
        assert "os_platform" in env
        assert "library_versions" in env
        assert "execution_timestamp" in env
        
        # Verify timestamp format (ISO 8601)
        datetime.fromisoformat(env["execution_timestamp"].replace("Z", "+00:00"))
        
    def test_final_output_aggregation(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that final_output section properly aggregates results"""
        generator = FluxDiagnosticGenerator(
            diagnostic_data=valid_diagnostic_data,
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data,
            rubric_exit_code=0
        )
        
        final_output = generator.generate_final_output_section()
        
        assert final_output["determinism_verified"] is True
        assert final_output["coverage_300"]["met"] is True
        assert final_output["rubric_alignment"] == "ALIGNED"
        assert "quality_metrics" in final_output
        
    def test_rubric_exit_code_mapping(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that rubric exit codes map correctly to status"""
        test_cases = [
            (0, "ALIGNED"),
            (1, "ERROR"),
            (2, "MISSING_FILES"),
            (3, "MISALIGNED"),
            (-1, "UNKNOWN")
        ]
        
        for exit_code, expected_status in test_cases:
            generator = FluxDiagnosticGenerator(
                diagnostic_data=valid_diagnostic_data,
                connection_data=valid_connection_data,
                determinism_data=valid_determinism_data,
                rubric_exit_code=exit_code
            )
            
            final_output = generator.generate_final_output_section()
            assert final_output["rubric_alignment"] == expected_status
            
    def test_save_report_creates_file(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that save_report creates a valid JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.json"
            
            generator = FluxDiagnosticGenerator(
                diagnostic_data=valid_diagnostic_data,
                connection_data=valid_connection_data,
                determinism_data=valid_determinism_data,
                rubric_exit_code=0
            )
            
            generator.save_report(output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                report = json.load(f)
                
            assert "pipeline_stages" in report
            assert "connections" in report
            assert "final_output" in report
            assert "environment" in report
            
    def test_contract_checks_structure(self, valid_diagnostic_data, valid_connection_data, valid_determinism_data):
        """Test that contract_checks have proper pass/fail structure"""
        generator = FluxDiagnosticGenerator(
            diagnostic_data=valid_diagnostic_data,
            connection_data=valid_connection_data,
            determinism_data=valid_determinism_data,
            rubric_exit_code=0
        )
        
        report = generator.generate_report()
        
        for stage in report["pipeline_stages"]:
            assert "contract_checks" in stage
            assert "pass" in stage["contract_checks"]
            assert "fail" in stage["contract_checks"]
            assert isinstance(stage["contract_checks"]["pass"], int)
            assert isinstance(stage["contract_checks"]["fail"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
