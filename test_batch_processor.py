#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch_processor.py — Unit Tests for BatchJobManager

Tests job lifecycle management, state transitions, and Redis integration.
"""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile

from batch_processor import BatchJobManager, JobState


class TestBatchJobManager:
    """Unit tests for BatchJobManager"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.llen.return_value = 0
        mock_client.scan_iter.return_value = []
        return mock_client
    
    @pytest.fixture
    def manager(self, mock_redis):
        """Create BatchJobManager with mock Redis"""
        with patch('batch_processor.redis.Redis', return_value=mock_redis):
            temp_dir = tempfile.mkdtemp(prefix="batch_test_")
            manager = BatchJobManager(artifacts_base_dir=temp_dir)
            manager.redis_client = mock_redis
            yield manager
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_get_job_key(self, manager):
        """Test job key generation"""
        job_id = "test-job-123"
        key = manager.get_job_key(job_id)
        assert key == "job:test-job-123"
    
    def test_get_job_data_not_found(self, manager, mock_redis):
        """Test retrieving non-existent job"""
        mock_redis.get.return_value = None
        
        result = manager.get_job_data("nonexistent-job")
        assert result is None
    
    def test_get_job_data_success(self, manager, mock_redis):
        """Test retrieving existing job"""
        job_data = {
            "job_id": "test-job",
            "status": JobState.QUEUED.value,
            "document_count": 5
        }
        mock_redis.get.return_value = json.dumps(job_data)
        
        result = manager.get_job_data("test-job")
        assert result is not None
        assert result["job_id"] == "test-job"
        assert result["status"] == JobState.QUEUED.value
    
    def test_update_job_data(self, manager, mock_redis):
        """Test updating job data"""
        existing_job = {
            "job_id": "test-job",
            "status": JobState.QUEUED.value,
            "document_count": 5
        }
        mock_redis.get.return_value = json.dumps(existing_job)
        
        updates = {"status": JobState.PROCESSING.value}
        success = manager.update_job_data("test-job", updates)
        
        assert success is True
        mock_redis.setex.assert_called_once()
    
    def test_transition_to_processing(self, manager, mock_redis):
        """Test QUEUED → PROCESSING transition"""
        job_data = {
            "job_id": "test-job",
            "status": JobState.QUEUED.value
        }
        mock_redis.get.return_value = json.dumps(job_data)
        
        success = manager.transition_to_processing("test-job")
        assert success is True
    
    def test_transition_to_completed(self, manager, mock_redis):
        """Test PROCESSING → COMPLETED transition"""
        job_data = {
            "job_id": "test-job",
            "status": JobState.PROCESSING.value
        }
        mock_redis.get.return_value = json.dumps(job_data)
        
        results = {"documents_processed": 5, "coverage": 300}
        success = manager.transition_to_completed("test-job", results)
        assert success is True
    
    def test_transition_to_failed(self, manager, mock_redis):
        """Test transition to FAILED state"""
        job_data = {
            "job_id": "test-job",
            "status": JobState.PROCESSING.value
        }
        mock_redis.get.return_value = json.dumps(job_data)
        
        error_msg = "Processing failed"
        success = manager.transition_to_failed("test-job", error_msg)
        assert success is True
    
    def test_update_progress(self, manager, mock_redis):
        """Test progress updates"""
        job_data = {
            "job_id": "test-job",
            "status": JobState.PROCESSING.value,
            "progress": {}
        }
        mock_redis.get.return_value = json.dumps(job_data)
        
        success = manager.update_progress(
            job_id="test-job",
            current_step="processing documents",
            completed_steps=3,
            total_steps=10
        )
        assert success is True
    
    def test_store_artifacts(self, manager):
        """Test artifact storage to filesystem"""
        artifacts = {
            "job_id": "test-job",
            "results": {"score": 85}
        }
        
        artifacts_dir = manager.store_artifacts("test-job", artifacts)
        assert artifacts_dir.exists()
        
        results_file = artifacts_dir / "evaluation_results.json"
        assert results_file.exists()
        
        with open(results_file, "r") as f:
            stored_data = json.load(f)
        assert stored_data == artifacts
    
    def test_get_queue_depth(self, manager, mock_redis):
        """Test queue depth retrieval"""
        mock_redis.llen.return_value = 42
        
        depth = manager.get_queue_depth("pdm_evaluation_queue")
        assert depth == 42
        mock_redis.llen.assert_called_once_with("pdm_evaluation_queue")
    
    def test_get_active_jobs(self, manager, mock_redis):
        """Test active job retrieval"""
        job_keys = ["job:test-1", "job:test-2", "job:test-3"]
        mock_redis.scan_iter.return_value = job_keys
        
        job_1 = {"job_id": "test-1", "status": JobState.QUEUED.value}
        job_2 = {"job_id": "test-2", "status": JobState.PROCESSING.value}
        job_3 = {"job_id": "test-3", "status": JobState.COMPLETED.value}
        
        mock_redis.get.side_effect = [
            json.dumps(job_1),
            json.dumps(job_2),
            json.dumps(job_3)
        ]
        
        active_jobs = manager.get_active_jobs()
        assert len(active_jobs) == 2
        assert "test-1" in active_jobs
        assert "test-2" in active_jobs
        assert "test-3" not in active_jobs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
