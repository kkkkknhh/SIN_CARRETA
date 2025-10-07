#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_system_validators.py — Tests for System Validators

Tests pre/post execution validation including batch validation functions.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from system_validators import (
    SystemHealthValidator,
    validate_batch_pre_execution,
    validate_batch_post_execution,
    ValidationError,
    BatchValidationResult
)


class TestSystemHealthValidator:
    """Tests for SystemHealthValidator"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository structure"""
        temp_dir = Path(tempfile.mkdtemp(prefix="validator_test_"))
        
        # Create required files with D{N}-Q{N} format weights
        rubric = {
            "weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 0.0033333333333333335 for i in range(300)}
        }
        (temp_dir / "RUBRIC_SCORING.json").write_text(json.dumps(rubric))
        (temp_dir / "tools").mkdir(exist_ok=True)
        (temp_dir / "tools" / "flow_doc.json").write_text('{"canonical_order": ["node1", "node2"]}')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_pre_execution_validation_success(self, temp_repo):
        """Test pre-execution validation with all checks passing"""
        validator = SystemHealthValidator(str(temp_repo))
        result = validator.validate_pre_execution()
        
        assert "ok" in result
        assert "errors" in result
        assert "checks" in result
    
    def test_pre_execution_missing_rubric(self):
        """Test pre-execution validation detects missing rubric"""
        temp_dir = Path(tempfile.mkdtemp(prefix="validator_test_"))
        (temp_dir / "tools").mkdir(exist_ok=True)
        (temp_dir / "tools" / "flow_doc.json").write_text('{}')
        
        validator = SystemHealthValidator(str(temp_dir))
        result = validator.validate_pre_execution()
        
        assert result["ok"] is False
        assert any("rubric_scoring.json" in err for err in result["errors"])
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validate_question_id_format_valid(self, temp_repo):
        """Test question ID format validation with valid rubric"""
        validator = SystemHealthValidator(str(temp_repo))
        validator.validate_question_id_format()
    
    def test_validate_question_id_format_malformed_ids(self):
        """Test detection of malformed question IDs"""
        temp_dir = Path(tempfile.mkdtemp(prefix="validator_test_"))
        
        weights = {
            "D1-Q1": 0.0033,
            "D7-Q1": 0.0033,
            "D1-Q301": 0.0033,
            "D1-Q0": 0.0033,
            "invalid-id": 0.0033
        }
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        
        validator = SystemHealthValidator(str(temp_dir))
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_question_id_format()
        
        error_msg = str(exc_info.value)
        assert "malformed question ID" in error_msg.lower()
        assert "D7-Q1" in error_msg or "4 malformed" in error_msg
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validate_question_id_format_count_mismatch(self):
        """Test detection of question count mismatch"""
        temp_dir = Path(tempfile.mkdtemp(prefix="validator_test_"))
        
        weights = {f"D{d}-Q{q}": 0.005 for d in range(1, 7) for q in range(1, 31)}
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        
        validator = SystemHealthValidator(str(temp_dir))
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_question_id_format()
        
        error_msg = str(exc_info.value)
        assert "count mismatch" in error_msg.lower()
        assert "180" in error_msg
        assert "300" in error_msg
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validate_question_id_format_missing_rubric(self):
        """Test validation with missing rubric file"""
        temp_dir = Path(tempfile.mkdtemp(prefix="validator_test_"))
        
        validator = SystemHealthValidator(str(temp_dir))
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_question_id_format()
        
        assert "RUBRIC_SCORING.json missing" in str(exc_info.value)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validate_question_id_format_empty_weights(self):
        """Test validation with empty weights section"""
        temp_dir = Path(tempfile.mkdtemp(prefix="validator_test_"))
        
        (temp_dir / "rubric_scoring.json").write_text(json.dumps({"weights": {}}))
        
        validator = SystemHealthValidator(str(temp_dir))
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_question_id_format()
        
        assert "weights" in str(exc_info.value).lower()
        assert "empty" in str(exc_info.value).lower()
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_pre_execution_validates_question_ids(self, temp_repo):
        """Test that pre-execution includes question ID validation"""
        weights = {"D1-Q1": 0.005, "invalid-id": 0.005}
        (temp_repo / "rubric_scoring.json").write_text(json.dumps({"weights": weights}))
        
        validator = SystemHealthValidator(str(temp_repo))
        result = validator.validate_pre_execution()
        
        assert result["ok"] is False
        assert any("malformed" in err.lower() for err in result["errors"])
    
    def test_post_execution_validation_success(self, temp_repo):
        """Test post-execution validation with valid artifacts"""
        # Create artifacts directory
        artifacts_dir = temp_repo / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Create runtime trace
        runtime_trace = {
            "order": ["node1", "node2"],
            "execution_time": 100
        }
        (artifacts_dir / "flow_runtime.json").write_text(json.dumps(runtime_trace))
        
        # Create answers report with 300+ questions in D{N}-Q{N} format
        answers_report = {
            "summary": {"total_questions": 300},
            "answers": [{"question_id": f"D{i//50 + 1}-Q{i%50 + 1}"} for i in range(300)]
        }
        (artifacts_dir / "answers_report.json").write_text(json.dumps(answers_report))
        
        # Update RUBRIC_SCORING.json with matching D{N}-Q{N} format weights
        rubric = {
            "weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 0.0033333333333333335 for i in range(300)}
        }
        (temp_repo / "RUBRIC_SCORING.json").write_text(json.dumps(rubric))
        
        validator = SystemHealthValidator(str(temp_repo))
        result = validator.validate_post_execution(artifacts_dir="artifacts")
        
        assert result["ok"] is True
        assert result["ok_coverage"] is True
    
    def test_post_execution_insufficient_coverage(self, temp_repo):
        """Test post-execution validation detects insufficient coverage"""
        artifacts_dir = temp_repo / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        runtime_trace = {"order": ["node1", "node2"]}
        (artifacts_dir / "flow_runtime.json").write_text(json.dumps(runtime_trace))
        
        # Only 250 questions (below 300 threshold) in D{N}-Q{N} format
        answers_report = {
            "summary": {"total_questions": 250},
            "answers": [{"question_id": f"D{i//50 + 1}-Q{i%50 + 1}"} for i in range(250)]
        }
        (artifacts_dir / "answers_report.json").write_text(json.dumps(answers_report))
        
        # Update RUBRIC_SCORING.json with matching D{N}-Q{N} format weights (250 only)
        rubric = {
            "weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 0.0033333333333333335 for i in range(250)}
        }
        (temp_repo / "RUBRIC_SCORING.json").write_text(json.dumps(rubric))
        
        validator = SystemHealthValidator(str(temp_repo))
        result = validator.validate_post_execution(artifacts_dir="artifacts")
        
        assert result["ok_coverage"] is False
        assert any("below 300 questions" in err for err in result["errors"])


class TestBatchPreExecutionValidation:
    """Tests for validate_batch_pre_execution"""
    
    @patch('system_validators.PSUTIL_AVAILABLE', True)
    @patch('system_validators.psutil')
    @patch('system_validators.redis.Redis')
    def test_pre_execution_all_checks_pass(self, mock_redis_class, mock_psutil):
        """Test all pre-execution checks passing"""
        # Mock memory check (10GB available)
        mock_mem = Mock()
        mock_mem.available = 10 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem
        
        # Mock disk check (20GB available)
        mock_disk = Mock(free=20 * 1024**3)
        with patch('system_validators.shutil.disk_usage', return_value=mock_disk):
            # Mock Redis connectivity
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            
            # Mock Celery workers
            with patch('system_validators.Celery') as mock_celery_class:
                mock_celery = Mock()
                mock_inspect = Mock()
                mock_inspect.stats.return_value = {"worker1": {}, "worker2": {}}
                mock_celery.control.inspect.return_value = mock_inspect
                mock_celery_class.return_value = mock_celery
                
                result = validate_batch_pre_execution()
                
                assert result.ok is True
                assert result.memory_ok is True
                assert result.disk_ok is True
                assert result.redis_ok is True
                assert result.workers_ok is True
    
    @patch('system_validators.PSUTIL_AVAILABLE', True)
    @patch('system_validators.psutil')
    def test_pre_execution_insufficient_memory(self, mock_psutil):
        """Test detection of insufficient memory"""
        # Mock memory check (only 4GB available, need 8GB)
        mock_mem = Mock()
        mock_mem.available = 4 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem
        
        # Mock disk with sufficient space
        mock_disk = Mock(free=20 * 1024**3)
        with patch('system_validators.shutil.disk_usage', return_value=mock_disk):
            # Mock Redis connectivity
            with patch('system_validators.redis.Redis') as mock_redis_class:
                mock_redis = Mock()
                mock_redis.ping.return_value = True
                mock_redis_class.return_value = mock_redis
                
                # Mock Celery
                with patch('system_validators.Celery') as mock_celery_class:
                    mock_celery = Mock()
                    mock_inspect = Mock()
                    mock_inspect.stats.return_value = {"worker1": {}}
                    mock_celery.control.inspect.return_value = mock_inspect
                    mock_celery_class.return_value = mock_celery
                    
                    with pytest.raises(ValidationError) as exc_info:
                        validate_batch_pre_execution()
                    
                    assert "Insufficient memory" in str(exc_info.value)


class TestBatchPostExecutionValidation:
    """Tests for validate_batch_post_execution"""
    
    @pytest.fixture
    def artifacts_dir(self):
        """Create temporary artifacts directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix="artifacts_test_"))
        yield temp_dir
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_post_execution_all_documents_success(self, artifacts_dir):
        """Test validation with all documents processed successfully"""
        batch_results = []
        
        for i in range(10):
            doc_id = f"doc_{i:02d}"
            doc_dir = artifacts_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Create coverage report with 300 questions in D{N}-Q{N} format
            coverage = {
                "summary": {
                    "total_questions": 300,
                    "answered_questions": 300
                },
                "questions": [f"D{j//50 + 1}-Q{j%50 + 1}" for j in range(300)]
            }
            (doc_dir / "coverage_report.json").write_text(json.dumps(coverage))
            
            # Create evidence registry with deterministic hash
            evidence = {
                "deterministic_hash": "consistent_hash_12345",
                "evidence": [{"id": j, "text": f"evidence_{j}"} for j in range(50)]
            }
            (doc_dir / "evidence_registry.json").write_text(json.dumps(evidence))
            
            batch_results.append({
                "document_id": doc_id,
                "status": "success",
                "processing_time": 15.5
            })
        
        result = validate_batch_post_execution(batch_results, str(artifacts_dir))
        
        assert result["ok"] is True
        assert result["coverage_passed"] == 10
        assert result["coverage_failed"] == 0
        assert result["hash_consistency_ok"] is True
        assert result["throughput_docs_per_hour"] > 0
    
    def test_post_execution_insufficient_coverage(self, artifacts_dir):
        """Test detection of insufficient coverage"""
        batch_results = []
        
        doc_id = "doc_00"
        doc_dir = artifacts_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Only 250 questions (insufficient) in D{N}-Q{N} format
        coverage = {
            "summary": {
                "total_questions": 250,
                "answered_questions": 250
            },
            "questions": [f"D{j//50 + 1}-Q{j%50 + 1}" for j in range(250)]
        }
        (doc_dir / "coverage_report.json").write_text(json.dumps(coverage))
        
        batch_results.append({
            "document_id": doc_id,
            "status": "success",
            "processing_time": 15.0
        })
        
        result = validate_batch_post_execution(batch_results, str(artifacts_dir))
        
        assert result["ok"] is False
        assert result["coverage_passed"] == 0
        assert result["coverage_failed"] == 1
        assert any("insufficient coverage" in err for err in result["errors"])
    
    def test_post_execution_hash_inconsistency(self, artifacts_dir):
        """Test detection of hash inconsistency across documents"""
        batch_results = []
        
        for i in range(3):
            doc_id = f"doc_{i:02d}"
            doc_dir = artifacts_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            coverage = {"summary": {"total_questions": 300}}
            (doc_dir / "coverage_report.json").write_text(json.dumps(coverage))
            
            # Different hash for each document (inconsistency)
            evidence = {
                "deterministic_hash": f"hash_{i}_different",
                "evidence": []
            }
            (doc_dir / "evidence_registry.json").write_text(json.dumps(evidence))
            
            batch_results.append({
                "document_id": doc_id,
                "status": "success",
                "processing_time": 15.0
            })
        
        result = validate_batch_post_execution(batch_results, str(artifacts_dir))
        
        assert result["hash_consistency_ok"] is False
        assert any("Hash inconsistency" in err for err in result["errors"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
