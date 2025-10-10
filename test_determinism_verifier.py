#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Determinism Verifier
===================================

Comprehensive tests for determinism_verifier.py covering:
- JSON normalization and hashing
- Non-deterministic field removal
- Artifact comparison logic
- Diff generation
- Report generation
- Exit code handling
"""

import pytest
import json
import hashlib
from determinism_verifier import (
    DeterminismVerifier,
    ArtifactComparison,
    DeterminismReport
)


class TestJSONNormalization:
    """Test JSON normalization and hashing"""
    
    @staticmethod
    def test_identical_content_different_order(tmp_path):
        """Test that identical content with different key order produces same hash"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        # Create two JSON files with same content but different key order
        json1 = tmp_path / "test1.json"
        json2 = tmp_path / "test2.json"
        
        data1 = {"z": 1, "a": 2, "m": 3}
        data2 = {"a": 2, "m": 3, "z": 1}
        
        with open(json1, 'w') as f:
            json.dump(data1, f)
        with open(json2, 'w') as f:
            json.dump(data2, f)
        
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, "Hashes should match for identical content with different key order"
    
    @staticmethod
    def test_different_whitespace(tmp_path):
        """Test that different whitespace produces same hash"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        json1 = tmp_path / "test1.json"
        json2 = tmp_path / "test2.json"
        
        # Compact vs pretty-printed
        with open(json1, 'w') as f:
            f.write('{"a":1,"b":2}')
        with open(json2, 'w') as f:
            json.dump({"a": 1, "b": 2}, f, indent=4)
        
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, "Hashes should match regardless of whitespace"
    
    @staticmethod
    def test_nested_object_normalization(tmp_path):
        """Test normalization of nested objects"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        json1 = tmp_path / "test1.json"
        json2 = tmp_path / "test2.json"
        
        data1 = {
            "outer": {"z": {"deep": 1}, "a": {"deep": 2}},
            "list": [{"b": 1}, {"a": 2}]
        }
        data2 = {
            "list": [{"b": 1}, {"a": 2}],
            "outer": {"a": {"deep": 2}, "z": {"deep": 1}}
        }
        
        with open(json1, 'w') as f:
            json.dump(data1, f)
        with open(json2, 'w') as f:
            json.dump(data2, f)
        
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, "Hashes should match for nested structures with different key order"


class TestNonDeterministicFieldRemoval:
    """Test removal of non-deterministic fields"""
    
    @staticmethod
    def test_timestamp_removal(tmp_path):
        """Test that timestamps are removed from comparison"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        json1 = tmp_path / "test1.json"
        json2 = tmp_path / "test2.json"
        
        data1 = {"value": 42, "timestamp": "2024-01-01T00:00:00Z"}
        data2 = {"value": 42, "timestamp": "2024-12-31T23:59:59Z"}
        
        with open(json1, 'w') as f:
            json.dump(data1, f)
        with open(json2, 'w') as f:
            json.dump(data2, f)
        
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, "Timestamps should be ignored in hash computation"
    
    @staticmethod
    def test_duration_removal(tmp_path):
        """Test that duration fields are removed"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        json1 = tmp_path / "test1.json"
        json2 = tmp_path / "test2.json"
        
        data1 = {"result": "success", "duration_seconds": 12.5, "execution_time": 100}
        data2 = {"result": "success", "duration_seconds": 45.2, "execution_time": 500}
        
        with open(json1, 'w') as f:
            json.dump(data1, f)
        with open(json2, 'w') as f:
            json.dump(data2, f)
        
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, "Duration fields should be ignored"
    
    @staticmethod
    def test_nested_timestamp_removal(tmp_path):
        """Test that nested timestamps are removed"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        json1 = tmp_path / "test1.json"
        json2 = tmp_path / "test2.json"
        
        data1 = {
            "stages": [
                {"name": "stage1", "timestamp": "2024-01-01T00:00:00Z"},
                {"name": "stage2", "timestamp": "2024-01-01T00:01:00Z"}
            ]
        }
        data2 = {
            "stages": [
                {"name": "stage1", "timestamp": "2024-12-31T00:00:00Z"},
                {"name": "stage2", "timestamp": "2024-12-31T00:01:00Z"}
            ]
        }
        
        with open(json1, 'w') as f:
            json.dump(data1, f)
        with open(json2, 'w') as f:
            json.dump(data2, f)
        
        hash1 = verifier.compute_normalized_hash(json1)
        hash2 = verifier.compute_normalized_hash(json2)
        
        assert hash1 == hash2, "Nested timestamps should be ignored"


class TestArtifactComparison:
    """Test artifact comparison logic"""
    
    @staticmethod
    def test_matching_artifacts(tmp_path):
        """Test comparison of matching artifacts"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        # Create matching artifacts
        verifier.run1_dir.mkdir(parents=True)
        verifier.run2_dir.mkdir(parents=True)
        
        data = {"evidence_count": 42, "hash": "abc123"}
        
        artifact1 = verifier.run1_dir / "test.json"
        artifact2 = verifier.run2_dir / "test.json"
        
        with open(artifact1, 'w') as f:
            json.dump(data, f)
        with open(artifact2, 'w') as f:
            json.dump(data, f)
        
        comparison = verifier.compare_artifacts("test.json")
        
        assert comparison.match, "Identical artifacts should match"
        assert comparison.run1_hash == comparison.run2_hash
        assert len(comparison.diff_lines) == 0, "No diff lines for matching artifacts"
    
    @staticmethod
    def test_mismatched_artifacts(tmp_path):
        """Test comparison of mismatched artifacts"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        verifier.run1_dir.mkdir(parents=True)
        verifier.run2_dir.mkdir(parents=True)
        
        data1 = {"evidence_count": 42}
        data2 = {"evidence_count": 43}
        
        artifact1 = verifier.run1_dir / "test.json"
        artifact2 = verifier.run2_dir / "test.json"
        
        with open(artifact1, 'w') as f:
            json.dump(data1, f)
        with open(artifact2, 'w') as f:
            json.dump(data2, f)
        
        comparison = verifier.compare_artifacts("test.json")
        
        assert not comparison.match, "Different artifacts should not match"
        assert comparison.run1_hash != comparison.run2_hash
        assert len(comparison.diff_lines) > 0, "Diff lines should be generated for mismatches"


class TestEvidenceHashExtraction:
    """Test evidence hash extraction"""
    
    @staticmethod
    def test_extract_evidence_hash(tmp_path):
        """Test extraction of deterministic hash from evidence registry"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        evidence_data = {
            "evidence_count": 100,
            "deterministic_hash": "abc123def456",
            "evidence": {}
        }
        
        evidence_path = tmp_path / "evidence_registry.json"
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f)
        
        extracted_hash = verifier.extract_evidence_hash(evidence_path)
        
        assert extracted_hash == "abc123def456"
    
    @staticmethod
    def test_extract_missing_evidence_hash(tmp_path):
        """Test extraction when hash is missing"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        evidence_data = {"evidence_count": 0}
        
        evidence_path = tmp_path / "evidence_registry.json"
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f)
        
        extracted_hash = verifier.extract_evidence_hash(evidence_path)
        
        assert extracted_hash == ""


class TestFlowHashExtraction:
    """Test flow hash extraction"""
    
    @staticmethod
    def test_extract_flow_hash(tmp_path):
        """Test extraction of flow hash from runtime"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        flow_data = {
            "flow_hash": "xyz789abc123",
            "stages": ["stage1", "stage2"]
        }
        
        flow_path = tmp_path / "flow_runtime.json"
        with open(flow_path, 'w') as f:
            json.dump(flow_data, f)
        
        extracted_hash = verifier.extract_flow_hash(flow_path)
        
        assert extracted_hash == "xyz789abc123"


class TestDiffGeneration:
    """Test diff generation"""
    
    @staticmethod
    def test_diff_for_different_values(tmp_path):
        """Test diff generation for different values"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        file1 = tmp_path / "file1.json"
        file2 = tmp_path / "file2.json"
        
        data1 = {"score": 0.85, "evidence": ["A", "B"]}
        data2 = {"score": 0.90, "evidence": ["A", "C"]}
        
        with open(file1, 'w') as f:
            json.dump(data1, f)
        with open(file2, 'w') as f:
            json.dump(data2, f)
        
        diff = verifier.generate_json_diff(file1, file2)
        
        assert len(diff) > 0, "Diff should be generated"
        diff_text = "\n".join(diff)
        assert "0.85" in diff_text or "0.90" in diff_text, "Diff should show changed values"
    
    @staticmethod
    def test_diff_for_nested_changes(tmp_path):
        """Test diff for nested structure changes"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        
        file1 = tmp_path / "file1.json"
        file2 = tmp_path / "file2.json"
        
        data1 = {
            "answers": [
                {"id": "Q1", "score": 0.5},
                {"id": "Q2", "score": 0.7}
            ]
        }
        data2 = {
            "answers": [
                {"id": "Q1", "score": 0.5},
                {"id": "Q2", "score": 0.8}  # Changed
            ]
        }
        
        with open(file1, 'w') as f:
            json.dump(data1, f)
        with open(file2, 'w') as f:
            json.dump(data2, f)
        
        diff = verifier.generate_json_diff(file1, file2)
        
        assert len(diff) > 0, "Diff should be generated for nested changes"


class TestReportGeneration:
    """Test report generation"""
    
    @staticmethod
    def test_perfect_match_report():
        """Test report for perfect match"""
        comparisons = [
            ArtifactComparison("test1.json", "hash1", "hash1", True, 100, 100),
            ArtifactComparison("test2.json", "hash2", "hash2", True, 200, 200)
        ]
        
        report = DeterminismReport(
            timestamp="2024-01-01T00:00:00Z",
            input_pdf="/path/to/input.pdf",
            run1_dir="/path/to/run1",
            run2_dir="/path/to/run2",
            perfect_match=True,
            artifact_comparisons=comparisons,
            evidence_hash_run1="evidence_hash",
            evidence_hash_run2="evidence_hash",
            evidence_hash_match=True,
            flow_hash_run1="flow_hash",
            flow_hash_run2="flow_hash",
            flow_hash_match=True
        )
        
        assert report.perfect_match
        assert report.evidence_hash_match
        assert report.flow_hash_match
        assert len(report.execution_errors) == 0
    
    @staticmethod
    def test_mismatch_report():
        """Test report for mismatches"""
        comparisons = [
            ArtifactComparison("test1.json", "hash1", "hash1", True, 100, 100),
            ArtifactComparison("test2.json", "hash2a", "hash2b", False, 200, 201, ["diff line"])
        ]
        
        report = DeterminismReport(
            timestamp="2024-01-01T00:00:00Z",
            input_pdf="/path/to/input.pdf",
            run1_dir="/path/to/run1",
            run2_dir="/path/to/run2",
            perfect_match=False,
            artifact_comparisons=comparisons,
            evidence_hash_run1="hash_a",
            evidence_hash_run2="hash_b",
            evidence_hash_match=False,
            flow_hash_run1="flow_a",
            flow_hash_run2="flow_b",
            flow_hash_match=False
        )
        
        assert not report.perfect_match
        assert not report.evidence_hash_match
        assert not report.flow_hash_match
    
    @staticmethod
    def test_execution_error_report():
        """Test report with execution errors"""
        report = DeterminismReport(
            timestamp="2024-01-01T00:00:00Z",
            input_pdf="/path/to/input.pdf",
            run1_dir="/path/to/run1",
            run2_dir="/path/to/run2",
            perfect_match=False,
            artifact_comparisons=[],
            evidence_hash_run1="",
            evidence_hash_run2="",
            evidence_hash_match=False,
            flow_hash_run1="",
            flow_hash_run2="",
            flow_hash_match=False,
            execution_errors=["Run 1 failed: timeout", "Run 2 failed: missing file"]
        )
        
        assert len(report.execution_errors) == 2
        assert not report.perfect_match


class TestReportExport:
    """Test report export functionality"""
    
    @staticmethod
    def test_json_export(tmp_path):
        """Test JSON report export"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        verifier.output_dir.mkdir(parents=True, exist_ok=True)
        
        report = DeterminismReport(
            timestamp="2024-01-01T00:00:00Z",
            input_pdf=str(tmp_path / "input.pdf"),
            run1_dir=str(tmp_path / "run1"),
            run2_dir=str(tmp_path / "run2"),
            perfect_match=True,
            artifact_comparisons=[],
            evidence_hash_run1="hash1",
            evidence_hash_run2="hash1",
            evidence_hash_match=True,
            flow_hash_run1="flow1",
            flow_hash_run2="flow1",
            flow_hash_match=True
        )
        
        verifier.export_report(report)
        
        json_path = verifier.output_dir / "determinism_report.json"
        txt_path = verifier.output_dir / "determinism_report.txt"
        
        assert json_path.exists(), "JSON report should be created"
        assert txt_path.exists(), "Text report should be created"
        
        # Verify JSON is valid
        with open(json_path, 'r') as f:
            loaded = json.load(f)
            assert loaded["perfect_match"] == True
            assert loaded["evidence_hash_match"] == True
    
    @staticmethod
    def test_text_report_content(tmp_path):
        """Test text report contains expected sections"""
        verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
        verifier.output_dir.mkdir(parents=True, exist_ok=True)
        
        report = DeterminismReport(
            timestamp="2024-01-01T00:00:00Z",
            input_pdf=str(tmp_path / "input.pdf"),
            run1_dir=str(tmp_path / "run1"),
            run2_dir=str(tmp_path / "run2"),
            perfect_match=False,
            artifact_comparisons=[
                ArtifactComparison("test.json", "hash1", "hash2", False, 100, 101)
            ],
            evidence_hash_run1="hash_a",
            evidence_hash_run2="hash_b",
            evidence_hash_match=False,
            flow_hash_run1="flow_a",
            flow_hash_run2="flow_b",
            flow_hash_match=False
        )
        
        verifier.export_report(report)
        
        txt_path = verifier.output_dir / "determinism_report.txt"
        with open(txt_path, 'r') as f:
            content = f.read()
        
        assert "DETERMINISM VERIFICATION REPORT" in content
        assert "EVIDENCE HASH COMPARISON" in content
        assert "FLOW HASH COMPARISON" in content
        assert "ARTIFACT COMPARISONS" in content
        assert "MISMATCH" in content


def test_file_hash_computation(tmp_path):
    """Test SHA-256 file hash computation"""
    verifier = DeterminismVerifier(tmp_path / "dummy.pdf", tmp_path)
    
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    computed_hash = verifier.compute_file_hash(test_file)
    expected_hash = hashlib.sha256(test_content).hexdigest()
    
    assert computed_hash == expected_hash


def test_verifier_initialization(tmp_path):
    """Test verifier initialization"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    
    verifier = DeterminismVerifier(pdf_path)
    
    assert verifier.input_pdf == pdf_path.resolve()
    assert verifier.output_dir.name.startswith("determinism_run_")
    assert verifier.run1_dir == verifier.output_dir / "run1"
    assert verifier.run2_dir == verifier.output_dir / "run2"


def test_verifier_missing_input_pdf(tmp_path):
    """Test verifier fails with missing input PDF"""
    with pytest.raises(FileNotFoundError):
        DeterminismVerifier(tmp_path / "nonexistent.pdf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
