#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_processor.py — Batch Job Manager for PDM Document Processing

Manages job lifecycle states: queued → processing → completed/failed
Integrates with Redis-backed Celery task queue and unified_evaluation_pipeline
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
import redis

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    """Job processing states"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchJobManager:
    """
    Manages batch job lifecycle and coordinates with Celery workers.
    
    Responsibilities:
    - Job state transitions (queued → processing → completed/failed)
    - Progress tracking and updates
    - Result aggregation
    - Job cleanup and TTL management
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        job_ttl_seconds: int = 86400,
        artifacts_base_dir: str = "./data/results"
    ):
        """
        Initialize BatchJobManager.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (optional)
            job_ttl_seconds: Job data TTL in seconds (default: 24 hours)
            artifacts_base_dir: Base directory for result artifacts
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self.job_ttl_seconds = job_ttl_seconds
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.artifacts_base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("BatchJobManager initialized with Redis at %s:%s", redis_host, redis_port)
    
    @staticmethod
    def get_job_key(job_id: str) -> str:
        """Get Redis key for job data"""
        return f"job:{job_id}"
    
    def get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job data from Redis.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Job data dictionary or None if not found
        """
        job_key = self.get_job_key(job_id)
        job_data_json = self.redis_client.get(job_key)
        
        if not job_data_json:
            logger.warning("Job %s not found in Redis", job_id)
            return None
        
        try:
            return json.loads(job_data_json)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode job data for %s: %s", job_id, e)
            return None
    
    def update_job_data(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update job data in Redis.
        
        Args:
            job_id: Unique job identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        job_data = self.get_job_data(job_id)
        if not job_data:
            logger.error("Cannot update non-existent job %s", job_id)
            return False
        
        job_data.update(updates)
        
        job_key = self.get_job_key(job_id)
        try:
            self.redis_client.setex(
                job_key,
                self.job_ttl_seconds,
                json.dumps(job_data)
            )
            return True
        except Exception as e:
            logger.error("Failed to update job data for %s: %s", job_id, e)
            return False
    
    def transition_to_processing(self, job_id: str) -> bool:
        """
        Transition job from QUEUED to PROCESSING.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if transition successful
        """
        updates = {
            "status": JobState.PROCESSING.value,
            "started_time": datetime.utcnow().isoformat(),
            "progress": {
                "current_step": "processing",
                "total_steps": 10,
                "completed_steps": 1,
                "progress_percentage": 10.0
            }
        }
        
        success = self.update_job_data(job_id, updates)
        if success:
            logger.info("Job %s transitioned to PROCESSING", job_id)
        return success
    
    def update_progress(
        self,
        job_id: str,
        current_step: str,
        completed_steps: int,
        total_steps: int
    ) -> bool:
        """
        Update job progress indicators.
        
        Args:
            job_id: Unique job identifier
            current_step: Description of current processing step
            completed_steps: Number of completed steps
            total_steps: Total number of steps
            
        Returns:
            True if update successful
        """
        progress_percentage = (completed_steps / total_steps * 100.0) if total_steps > 0 else 0.0
        
        updates = {
            "progress": {
                "current_step": current_step,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "progress_percentage": progress_percentage
            }
        }
        
        return self.update_job_data(job_id, updates)
    
    def transition_to_completed(
        self,
        job_id: str,
        results: Dict[str, Any]
    ) -> bool:
        """
        Transition job to COMPLETED state with results.
        
        Args:
            job_id: Unique job identifier
            results: Evaluation results dictionary
            
        Returns:
            True if transition successful
        """
        updates = {
            "status": JobState.COMPLETED.value,
            "completed_time": datetime.utcnow().isoformat(),
            "results": results,
            "progress": {
                "current_step": "completed",
                "total_steps": 10,
                "completed_steps": 10,
                "progress_percentage": 100.0
            }
        }
        
        success = self.update_job_data(job_id, updates)
        if success:
            logger.info("Job %s transitioned to COMPLETED", job_id)
        return success
    
    def transition_to_failed(
        self,
        job_id: str,
        error_message: str
    ) -> bool:
        """
        Transition job to FAILED state with error message.
        
        Args:
            job_id: Unique job identifier
            error_message: Description of failure
            
        Returns:
            True if transition successful
        """
        updates = {
            "status": JobState.FAILED.value,
            "completed_time": datetime.utcnow().isoformat(),
            "error_message": error_message
        }
        
        success = self.update_job_data(job_id, updates)
        if success:
            logger.error("Job %s transitioned to FAILED: %s", job_id, error_message)
        return success
    
    def store_artifacts(
        self,
        job_id: str,
        artifacts: Dict[str, Any]
    ) -> Path:
        """
        Store evaluation artifacts to filesystem.
        
        Args:
            job_id: Unique job identifier
            artifacts: Dictionary of artifacts to store
            
        Returns:
            Path to artifacts directory
        """
        job_artifacts_dir = self.artifacts_base_dir / job_id
        job_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Store main results as JSON
        results_file = job_artifacts_dir / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2, ensure_ascii=False)
        
        logger.info("Stored artifacts for job %s at %s", job_id, job_artifacts_dir)
        return job_artifacts_dir
    
    def get_queue_depth(self, queue_name: str = "pdm_evaluation_queue") -> int:
        """
        Get current depth of task queue.
        
        Args:
            queue_name: Name of Redis list used as queue
            
        Returns:
            Number of items in queue
        """
        try:
            return self.redis_client.llen(queue_name)
        except Exception as e:
            logger.error("Failed to get queue depth: %s", e)
            return 0
    
    def get_active_jobs(self) -> List[str]:
        """
        Get list of active job IDs (queued or processing).
        
        Returns:
            List of job IDs
        """
        try:
            pattern = "job:*"
            job_keys = list(self.redis_client.scan_iter(match=pattern, count=100))
            
            active_jobs = []
            for job_key in job_keys:
                job_data_json = self.redis_client.get(job_key)
                if job_data_json:
                    job_data = json.loads(job_data_json)
                    status = job_data.get("status")
                    if status in [JobState.QUEUED.value, JobState.PROCESSING.value]:
                        job_id = job_data.get("job_id")
                        if job_id:
                            active_jobs.append(job_id)
            
            return active_jobs
        except Exception as e:
            logger.error("Failed to get active jobs: %s", e)
            return []
    
    def cleanup_expired_jobs(self) -> int:
        """
        Clean up expired job artifacts from filesystem.
        
        Redis handles TTL automatically, but filesystem artifacts need manual cleanup.
        
        Returns:
            Number of jobs cleaned up
        """
        cleaned = 0
        
        try:
            for job_dir in self.artifacts_base_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                
                job_id = job_dir.name
                job_data = self.get_job_data(job_id)
                
                # If job not in Redis (expired), clean up artifacts
                if not job_data:
                    import shutil
                    shutil.rmtree(job_dir)
                    logger.info("Cleaned up expired artifacts for job %s", job_id)
                    cleaned += 1
        except Exception as e:
            logger.error("Error during cleanup: %s", e)
        
        return cleaned
