#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Batch Job Manager with Redis and Celery
=================================================

Tests:
- BatchJobManager initialization with/without Redis
- Job lifecycle state transitions (queued → processing → completed/failed)
- Redis connection pooling and graceful degradation
- File system artifact storage
- Job creation, status queries, result retrieval
- Celery task execution (mocked)
- Compatibility with unified_evaluation_pipeline interface
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest

from batch_processor import (
    BatchJobManager,
    JobState,
    _process_pdm_document_impl,
    REDIS_AVAILABLE,
    CELERY_AVAILABLE
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_artifact_dir():
    """Create temporary artifact directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available")
    
    with mock.patch('batch_processor.redis.Redis') as mock_redis_class:
        mock_client = mock.MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_redis_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def job_manager_degraded(temp_artifact_dir):
    """Job manager in degraded mode (no Redis)"""
    manager = BatchJobManager(
        redis_url="redis://invalid:9999/0",
        artifact_storage_dir=temp_artifact_dir,
        enable_degraded_mode=True
    )
    yield manager
    manager.close()


@pytest.fixture
def mock_unified_pipeline():
    """Mock UnifiedEvaluationPipeline"""
    with mock.patch('batch_processor.UnifiedEvaluationPipeline') as mock_pipeline_class:
        mock_pipeline = mock.MagicMock()
        mock_pipeline.evaluate.return_value = {
            "status": "success",
            "metadata": {
                "pdm_path": "/test/doc.pdf",
                "execution_time_seconds": 10.5
            },
            "evidence_registry": {
                "deterministic_hash": "abc123",
                "statistics": {"total_evidence": 42}
            },
            "evaluations": {
                "decalogo": {"score": 85.5},
                "questionnaire": {"score": 90.0}
            }
        }
        mock_pipeline_class.return_value = mock_pipeline
        yield mock_pipeline


# ============================================================================
# BatchJobManager Tests
# ============================================================================

def test_batch_job_manager_initialization_degraded(temp_artifact_dir):
    """Test BatchJobManager initialization in degraded mode"""
    manager = BatchJobManager(
        redis_url="redis://invalid:9999/0",
        artifact_storage_dir=temp_artifact_dir,
        enable_degraded_mode=True
    )
    
    assert manager.artifact_storage_dir == Path(temp_artifact_dir)
    assert manager.enable_degraded_mode is True
    assert manager.redis_available is False
    assert manager.fallback_storage == {}
    
    manager.close()


def test_batch_job_manager_artifact_dir_creation(temp_artifact_dir):
    """Test artifact directory creation"""
    artifact_dir = Path(temp_artifact_dir) / "custom_artifacts"
    assert not artifact_dir.exists()
    
    manager = BatchJobManager(
        artifact_storage_dir=str(artifact_dir),
        enable_degraded_mode=True
    )
    
    assert artifact_dir.exists()
    manager.close()


def test_create_job_degraded_mode(job_manager_degraded):
    """Test job creation in degraded mode"""
    # Mock Celery to avoid submission
    with mock.patch('batch_processor.CELERY_AVAILABLE', False):
        with pytest.raises(RuntimeError, match="Celery not available"):
            job_manager_degraded.create_job(
                pdm_path="/test/document.pdf",
                municipality="TestCity",
                department="TestDept"
            )


def test_job_lifecycle_state_transitions(job_manager_degraded):
    """Test job state transitions: queued → processing → completed"""
    job_id = str(uuid.uuid4())
    
    # Create initial job data
    job_data = {
        "job_id": job_id,
        "pdm_path": "/test/doc.pdf",
        "state": JobState.QUEUED.value,
        "created_at": "2024-01-01T00:00:00"
    }
    job_manager_degraded._set_job_data(job_id, job_data)
    
    # Transition to PROCESSING
    job_manager_degraded.update_job_status(job_id, JobState.PROCESSING)
    status = job_manager_degraded.get_job_status(job_id)
    assert status["state"] == JobState.PROCESSING.value
    
    # Transition to COMPLETED
    result_data = {"test": "result"}
    job_manager_degraded.update_job_status(
        job_id,
        JobState.COMPLETED,
        result_data=result_data
    )
    status = job_manager_degraded.get_job_status(job_id)
    assert status["state"] == JobState.COMPLETED.value
    
    # Verify artifact was created
    result_file = Path(job_manager_degraded.artifact_storage_dir) / f"{job_id}_result.json"
    assert result_file.exists()
    
    with open(result_file, 'r') as f:
        stored_result = json.load(f)
    assert stored_result == result_data


def test_job_lifecycle_with_failure(job_manager_degraded):
    """Test job state transition to FAILED"""
    job_id = str(uuid.uuid4())
    
    # Create initial job data
    job_data = {
        "job_id": job_id,
        "pdm_path": "/test/doc.pdf",
        "state": JobState.QUEUED.value
    }
    job_manager_degraded._set_job_data(job_id, job_data)
    
    # Transition to PROCESSING
    job_manager_degraded.update_job_status(job_id, JobState.PROCESSING)
    
    # Transition to FAILED
    error_msg = "Pipeline execution failed"
    job_manager_degraded.update_job_status(
        job_id,
        JobState.FAILED,
        error=error_msg
    )
    
    status = job_manager_degraded.get_job_status(job_id)
    assert status["state"] == JobState.FAILED.value
    assert status["error"] == error_msg


def test_get_job_status_not_found(job_manager_degraded):
    """Test getting status for non-existent job"""
    status = job_manager_degraded.get_job_status("nonexistent-job-id")
    assert status["state"] == "not_found"
    assert "error" in status


def test_get_job_result_not_completed(job_manager_degraded):
    """Test getting result for non-completed job"""
    job_id = str(uuid.uuid4())
    
    job_data = {
        "job_id": job_id,
        "state": JobState.PROCESSING.value
    }
    job_manager_degraded._set_job_data(job_id, job_data)
    
    result = job_manager_degraded.get_job_result(job_id)
    assert result["state"] == JobState.PROCESSING.value
    assert result["message"] == "Job not completed"


def test_get_job_result_completed(job_manager_degraded):
    """Test getting result for completed job"""
    job_id = str(uuid.uuid4())
    
    # Create completed job with result
    result_data = {
        "status": "success",
        "score": 95.0
    }
    
    job_data = {
        "job_id": job_id,
        "state": JobState.COMPLETED.value,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:10:00"
    }
    job_manager_degraded._set_job_data(job_id, job_data)
    job_manager_degraded.update_job_status(
        job_id,
        JobState.COMPLETED,
        result_data=result_data
    )
    
    result = job_manager_degraded.get_job_result(job_id)
    assert result["state"] == JobState.COMPLETED.value
    assert result["result"] == result_data
    assert "artifacts" in result
    assert "result_file" in result["artifacts"]


def test_fallback_storage_operations(job_manager_degraded):
    """Test in-memory fallback storage get/set operations"""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "test_key": "test_value"
    }
    
    # Set data
    job_manager_degraded._set_job_data(job_id, job_data)
    assert job_id in job_manager_degraded.fallback_storage
    
    # Get data
    retrieved = job_manager_degraded._get_job_data(job_id)
    assert retrieved == job_data
    
    # Get non-existent data
    retrieved = job_manager_degraded._get_job_data("nonexistent")
    assert retrieved is None


def test_artifact_storage_path_generation(job_manager_degraded):
    """Test artifact file path generation"""
    job_id = "test-job-123"
    
    result_path = job_manager_degraded.artifact_storage_dir / f"{job_id}_result.json"
    evidence_path = job_manager_degraded.artifact_storage_dir / f"{job_id}_evidence.json"
    output_dir = job_manager_degraded.artifact_storage_dir / job_id
    
    assert result_path.name == f"{job_id}_result.json"
    assert evidence_path.name == f"{job_id}_evidence.json"
    assert output_dir.name == job_id


# ============================================================================
# Celery Task Tests
# ============================================================================

def test_process_pdm_document_success(mock_unified_pipeline, temp_artifact_dir):
    """Test successful PDM document processing"""
    job_id = str(uuid.uuid4())
    pdm_path = "/test/document.pdf"
    
    with mock.patch('batch_processor.BatchJobManager') as mock_manager_class:
        mock_manager = mock.MagicMock()
        mock_manager.artifact_storage_dir = Path(temp_artifact_dir)
        mock_manager_class.return_value = mock_manager
        
        result = _process_pdm_document_impl(
            job_id=job_id,
            pdm_path=pdm_path,
            municipality="TestCity",
            department="TestDept"
        )
        
        assert result["status"] == "completed"
        assert result["job_id"] == job_id
        assert "results" in result
        assert "artifacts" in result
        
        # Verify pipeline was called with correct parameters
        mock_unified_pipeline.evaluate.assert_called_once()
        call_kwargs = mock_unified_pipeline.evaluate.call_args[1]
        assert call_kwargs["pdm_path"] == pdm_path
        assert call_kwargs["municipality"] == "TestCity"
        assert call_kwargs["department"] == "TestDept"
        assert call_kwargs["export_json"] is True
        
        # Verify state transitions
        assert mock_manager.update_job_status.call_count >= 2


def test_process_pdm_document_pipeline_error(mock_unified_pipeline, temp_artifact_dir):
    """Test PDM processing with pipeline error"""
    job_id = str(uuid.uuid4())
    
    # Mock pipeline to return error
    mock_unified_pipeline.evaluate.return_value = {
        "status": "pipeline_error",
        "error": "Failed to parse document"
    }
    
    with mock.patch('batch_processor.BatchJobManager') as mock_manager_class:
        mock_manager = mock.MagicMock()
        mock_manager.artifact_storage_dir = Path(temp_artifact_dir)
        mock_manager_class.return_value = mock_manager
        
        result = _process_pdm_document_impl(
            job_id=job_id,
            pdm_path="/test/doc.pdf"
        )
        
        assert result["status"] == "failed"
        assert "error" in result
        
        # Verify job was marked as FAILED
        mock_manager.update_job_status.assert_any_call(
            job_id,
            JobState.FAILED,
            error=mock.ANY
        )


def test_process_pdm_document_exception(mock_unified_pipeline, temp_artifact_dir):
    """Test PDM processing with exception"""
    job_id = str(uuid.uuid4())
    
    # Mock pipeline to raise exception
    mock_unified_pipeline.evaluate.side_effect = RuntimeError("Unexpected error")
    
    with mock.patch('batch_processor.BatchJobManager') as mock_manager_class:
        mock_manager = mock.MagicMock()
        mock_manager.artifact_storage_dir = Path(temp_artifact_dir)
        mock_manager_class.return_value = mock_manager
        
        result = _process_pdm_document_impl(
            job_id=job_id,
            pdm_path="/test/doc.pdf"
        )
        
        assert result["status"] == "failed"
        assert "error" in result
        assert result["error_type"] == "RuntimeError"
        
        # Verify job was marked as FAILED
        mock_manager.update_job_status.assert_any_call(
            job_id,
            JobState.FAILED,
            error=mock.ANY
        )


def test_process_pdm_document_creates_output_directory(mock_unified_pipeline, temp_artifact_dir):
    """Test that processing creates job-specific output directory"""
    job_id = str(uuid.uuid4())
    
    with mock.patch('batch_processor.BatchJobManager') as mock_manager_class:
        mock_manager = mock.MagicMock()
        mock_manager.artifact_storage_dir = Path(temp_artifact_dir)
        mock_manager_class.return_value = mock_manager
        
        _process_pdm_document_impl(
            job_id=job_id,
            pdm_path="/test/doc.pdf"
        )
        
        # Verify output directory was created
        output_dir = Path(temp_artifact_dir) / job_id
        assert output_dir.exists()
        assert output_dir.is_dir()


def test_pipeline_compatibility_all_parameters(mock_unified_pipeline, temp_artifact_dir):
    """Test full parameter compatibility with unified_evaluation_pipeline"""
    job_id = str(uuid.uuid4())
    
    with mock.patch('batch_processor.BatchJobManager') as mock_manager_class:
        mock_manager = mock.MagicMock()
        mock_manager.artifact_storage_dir = Path(temp_artifact_dir)
        mock_manager_class.return_value = mock_manager
        
        _process_pdm_document_impl(
            job_id=job_id,
            pdm_path="/test/plan.pdf",
            municipality="Bogotá",
            department="Cundinamarca"
        )
        
        # Verify all parameters were passed correctly
        call_kwargs = mock_unified_pipeline.evaluate.call_args[1]
        assert call_kwargs["pdm_path"] == "/test/plan.pdf"
        assert call_kwargs["municipality"] == "Bogotá"
        assert call_kwargs["department"] == "Cundinamarca"
        assert call_kwargs["export_json"] is True
        assert "output_dir" in call_kwargs


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_job_workflow_degraded(job_manager_degraded, mock_unified_pipeline):
    """Test end-to-end job workflow in degraded mode"""
    job_id = str(uuid.uuid4())
    
    # Manually create job (skipping Celery submission)
    job_data = {
        "job_id": job_id,
        "pdm_path": "/test/doc.pdf",
        "municipality": "TestCity",
        "department": "TestDept",
        "state": JobState.QUEUED.value,
        "created_at": "2024-01-01T00:00:00"
    }
    job_manager_degraded._set_job_data(job_id, job_data)
    
    # Verify queued state
    status = job_manager_degraded.get_job_status(job_id)
    assert status["state"] == JobState.QUEUED.value
    
    # Simulate processing
    job_manager_degraded.update_job_status(job_id, JobState.PROCESSING)
    status = job_manager_degraded.get_job_status(job_id)
    assert status["state"] == JobState.PROCESSING.value
    
    # Simulate completion
    result_data = {"test": "success"}
    job_manager_degraded.update_job_status(
        job_id,
        JobState.COMPLETED,
        result_data=result_data
    )
    
    # Verify completed state and result retrieval
    status = job_manager_degraded.get_job_status(job_id)
    assert status["state"] == JobState.COMPLETED.value
    
    result = job_manager_degraded.get_job_result(job_id)
    assert result["state"] == JobState.COMPLETED.value
    assert result["result"] == result_data


def test_graceful_degradation_redis_unavailable():
    """Test graceful degradation when Redis is unavailable"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create manager with invalid Redis URL but degraded mode enabled
        manager = BatchJobManager(
            redis_url="redis://invalid.host:9999/0",
            artifact_storage_dir=tmpdir,
            enable_degraded_mode=True
        )
        
        assert manager.redis_available is False
        assert manager.fallback_storage == {}
        
        # Should be able to store/retrieve data using fallback
        job_id = str(uuid.uuid4())
        job_data = {"test": "data"}
        manager._set_job_data(job_id, job_data)
        
        retrieved = manager._get_job_data(job_id)
        assert retrieved == job_data
        
        manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
