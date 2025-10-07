#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch_validators.py — Tests for batch validation functions
"""

import json
import pathlib
import tempfile
import shutil
from typing import Dict, Any
import pytest
from system_validators import (
    validate_batch_pre_execution,
    validate_batch_post_execution,
    BatchValidationResult,
    ValidationError
)


class TestBatchValidationResult:
    def test_dataclass_initialization(self):
        result = BatchValidationResult(ok=True)
        assert result.ok is True
        assert result.errors == []
        assert result.memory_available_gb is None
        assert result.total_documents == 0
        
    def test_dataclass_with_values(self):
        result = BatchValidationResult(
            ok=False,
            errors=["test error"],
            memory_available_gb=10.5,
            memory_ok=True,
            total_documents=5
        )
        assert result.ok is False
        assert result.errors == ["test error"]
        assert result.memory_available_gb == 10.5
        assert result.total_documents == 5


class TestValidateBatchPreExecution:
    def test_raises_validation_error_on_failure(self):
        # This test might pass or fail depending on system resources
        # We just verify the function raises ValidationError when checks fail
        try:
            result = validate_batch_pre_execution()
            # If it succeeds, result should be BatchValidationResult
            assert isinstance(result, BatchValidationResult)
            assert result.ok is True
        except ValidationError as e:
            # If it fails, it should raise ValidationError
            assert "Batch pre-execution validation failed" in str(e)
            
    def test_validates_memory_threshold(self):
        # Test that memory check is performed (result or exception)
        try:
            result = validate_batch_pre_execution()
            assert result.memory_available_gb is not None
        except ValidationError:
            # Expected if memory is insufficient
            pass
        
    def test_validates_disk_threshold(self):
        # Test that disk check is performed (result or exception)
        try:
            result = validate_batch_pre_execution()
            assert result.disk_available_gb is not None
        except ValidationError:
            # Expected if disk space is insufficient
            pass


class TestValidateBatchPostExecution:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.artifacts_dir = pathlib.Path(self.temp_dir) / "artifacts"
        self.artifacts_dir.mkdir()
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
        
    def _create_doc_artifacts(self, doc_id: str, coverage: int, deterministic_hash: str):
        doc_dir = self.artifacts_dir / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        # Create coverage_report.json
        coverage_data = {
            "summary": {
                "total_questions": coverage
            }
        }
        (doc_dir / "coverage_report.json").write_text(json.dumps(coverage_data))
        
        # Create evidence_registry.json
        evidence_data = {
            "evidence_count": 25,
            "deterministic_hash": deterministic_hash,
            "evidence": {}
        }
        (doc_dir / "evidence_registry.json").write_text(json.dumps(evidence_data))
    
    def test_empty_batch(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_post_execution([], str(self.artifacts_dir))
        assert "validation failed" in str(exc_info.value).lower()
        
    def test_single_document_success(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        batch_results = [
            {"document_id": "doc1", "status": "success", "processing_time": 120.0}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert isinstance(result, dict)
        assert result["ok"] is True
        assert result["aggregate_coverage_statistics"]["total_documents"] == 1
        assert result["aggregate_coverage_statistics"]["coverage_passed"] == 1
        assert result["aggregate_coverage_statistics"]["coverage_failed"] == 0
        assert result["hash_verification"]["consistent"] is True
        
    def test_insufficient_coverage(self):
        self._create_doc_artifacts("doc1", 250, "hash123")
        batch_results = [
            {"document_id": "doc1", "status": "success"}
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert "coverage validation" in str(exc_info.value).lower()
        assert "insufficient coverage" in str(exc_info.value).lower()
        
    def test_hash_consistency_multiple_docs(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash123")
        self._create_doc_artifacts("doc3", 300, "hash123")
        
        batch_results = [
            {"document_id": "doc1", "status": "success"},
            {"document_id": "doc2", "status": "success"},
            {"document_id": "doc3", "status": "success"}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result["hash_verification"]["consistent"] is True
        assert result["aggregate_coverage_statistics"]["total_documents"] == 3
        assert result["aggregate_coverage_statistics"]["coverage_passed"] == 3
        
    def test_hash_inconsistency(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash456")
        
        batch_results = [
            {"document_id": "doc1", "status": "success"},
            {"document_id": "doc2", "status": "success"}
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert "hash consistency" in str(exc_info.value).lower()
        
    def test_with_failed_documents(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash123")
        
        batch_results = [
            {"document_id": "doc1", "status": "success"},
            {"document_id": "doc2", "status": "success"},
            {"document_id": "doc3", "status": "failed", "error": "Test error"}
        ]
        
        # Should succeed since doc1 and doc2 pass, doc3 is marked failed
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result["ok"] is True
        assert result["aggregate_coverage_statistics"]["coverage_passed"] == 2
        assert result["performance_metrics"]["error_rate_percent"] == pytest.approx(33.33, rel=0.01)
        
    def test_processing_time_stats(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash123")
        self._create_doc_artifacts("doc3", 300, "hash123")
        
        batch_results = [
            {"document_id": "doc1", "status": "success", "processing_time": 100.0},
            {"document_id": "doc2", "status": "success", "processing_time": 200.0},
            {"document_id": "doc3", "status": "success", "processing_time": 150.0}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        stats = result["performance_metrics"]["processing_time_distribution"]
        assert stats is not None
        assert stats["mean"] == pytest.approx(150.0)
        assert stats["p50"] == pytest.approx(150.0)
        assert stats["p95"] == pytest.approx(200.0)
        assert stats["min"] == 100.0
        assert stats["max"] == 200.0
        assert stats["count"] == 3
        
    def test_throughput_calculation(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash123")
        
        batch_results = [
            {"document_id": "doc1", "status": "success", "processing_time": 1800.0},  # 30 min
            {"document_id": "doc2", "status": "success", "processing_time": 1800.0}   # 30 min
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        # Total time: 3600s = 1 hour, 2 documents = 2 docs/hour
        assert result["performance_metrics"]["throughput_docs_per_hour"] == pytest.approx(2.0)
        
    def test_missing_artifacts(self):
        batch_results = [
            {"document_id": "doc1", "status": "success"}
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert "missing coverage_report.json" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
