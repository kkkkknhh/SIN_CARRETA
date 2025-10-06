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
    BatchValidationResult
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
    def test_returns_batch_validation_result(self):
        result = validate_batch_pre_execution()
        assert isinstance(result, BatchValidationResult)
        assert isinstance(result.errors, list)
        
    def test_memory_check(self):
        result = validate_batch_pre_execution()
        assert result.memory_available_gb is not None or not result.memory_ok
        
    def test_disk_check(self):
        result = validate_batch_pre_execution()
        assert result.disk_available_gb is not None or not result.disk_ok


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
        result = validate_batch_post_execution([], str(self.artifacts_dir))
        assert isinstance(result, BatchValidationResult)
        assert result.total_documents == 0
        assert not result.ok
        
    def test_single_document_success(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        batch_results = [
            {"document_id": "doc1", "status": "success", "processing_time": 120.0}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result.total_documents == 1
        assert result.coverage_passed == 1
        assert result.coverage_failed == 0
        assert result.hash_consistency_ok is True
        
    def test_insufficient_coverage(self):
        self._create_doc_artifacts("doc1", 250, "hash123")
        batch_results = [
            {"document_id": "doc1", "status": "success"}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result.coverage_passed == 0
        assert result.coverage_failed == 1
        assert not result.ok
        assert any("insufficient coverage" in err.lower() for err in result.errors)
        
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
        assert result.hash_consistency_ok is True
        assert result.total_documents == 3
        assert result.coverage_passed == 3
        
    def test_hash_inconsistency(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash456")
        
        batch_results = [
            {"document_id": "doc1", "status": "success"},
            {"document_id": "doc2", "status": "success"}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result.hash_consistency_ok is False
        assert any("hash inconsistency" in err.lower() for err in result.errors)
        
    def test_error_rate_calculation(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        
        batch_results = [
            {"document_id": "doc1", "status": "success"},
            {"document_id": "doc2", "status": "failed", "error": "Test error"},
            {"document_id": "doc3", "status": "failed", "error": "Test error"}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result.total_documents == 3
        assert result.error_rate_percent == pytest.approx(66.67, rel=0.01)
        
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
        assert result.processing_time_stats is not None
        assert result.processing_time_stats["mean"] == pytest.approx(150.0)
        assert result.processing_time_stats["median"] == pytest.approx(150.0)
        assert result.processing_time_stats["min"] == 100.0
        assert result.processing_time_stats["max"] == 200.0
        assert result.processing_time_stats["count"] == 3
        
    def test_throughput_calculation(self):
        self._create_doc_artifacts("doc1", 300, "hash123")
        self._create_doc_artifacts("doc2", 300, "hash123")
        
        batch_results = [
            {"document_id": "doc1", "status": "success", "processing_time": 1800.0},  # 30 min
            {"document_id": "doc2", "status": "success", "processing_time": 1800.0}   # 30 min
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        # Total time: 3600s = 1 hour, 2 documents = 2 docs/hour
        assert result.throughput_docs_per_hour == pytest.approx(2.0)
        
    def test_missing_artifacts(self):
        batch_results = [
            {"document_id": "doc1", "status": "success"}
        ]
        
        result = validate_batch_post_execution(batch_results, str(self.artifacts_dir))
        assert result.coverage_failed == 1
        assert any("missing coverage_report.json" in err for err in result.errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
