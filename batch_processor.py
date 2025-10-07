#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Job Manager with Redis-backed State Machine and Celery Workers
====================================================================

Batch processing system with:
- Redis-backed job lifecycle (queued → processing → completed → failed)
- Celery worker pool with optimized configuration
- File system artifact storage
- Graceful degradation when Redis unavailable
- Full compatibility with unified_evaluation_pipeline
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis package not available - degraded mode only")

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logging.warning("celery package not available - task submission disabled")

from unified_evaluation_pipeline import UnifiedEvaluationPipeline

logger = logging.getLogger(__name__)


# ============================================================================
# Job State Enum
# ============================================================================

class JobState(str, Enum):
    """Job lifecycle states"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Celery Application Configuration
# ============================================================================

# Redis broker and result backend URLs
REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")
REDIS_RESULT_BACKEND = os.getenv("REDIS_RESULT_BACKEND", "redis://localhost:6379/1")

# Initialize Celery application (if available)
if CELERY_AVAILABLE:
    celery_app = Celery(
        "batch_processor",
        broker=REDIS_BROKER_URL,
        backend=REDIS_RESULT_BACKEND
    )

    # Configure Celery for optimal worker performance
    celery_app.conf.update(
        worker_prefetch_multiplier=4,  # Prefetch 4 tasks per worker
        worker_concurrency=8,  # 8 concurrent worker processes
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,  # Acknowledge after task completion
        worker_max_tasks_per_child=100,  # Restart workers after 100 tasks (memory management)
        broker_connection_retry_on_startup=True,
        result_expires=86400,  # Results expire after 24 hours
    )

    logger.info(f"Celery configured: broker={REDIS_BROKER_URL}, concurrency=8, prefetch=4")
else:
    celery_app = None
    logger.warning("Celery not available - task submission will fail")


# ============================================================================
# Batch Job Manager with Redis State Management
# ============================================================================

class BatchJobManager:
    """
    Manages batch job lifecycle with Redis-backed state transitions.
    
    State machine:
        queued → processing → completed
                          ↘ failed
    
    Features:
    - Redis connection pooling
    - Graceful degradation if Redis unavailable
    - File system artifact storage
    - Job creation, status queries, result retrieval
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        artifact_storage_dir: Optional[str] = None,
        redis_pool_size: int = 10,
        enable_degraded_mode: bool = True
    ):
        """
        Initialize BatchJobManager.
        
        Args:
            redis_url: Redis connection URL (default: from env or localhost)
            artifact_storage_dir: Directory for artifact storage (default: ./batch_artifacts)
            redis_pool_size: Redis connection pool size
            enable_degraded_mode: Enable graceful degradation if Redis unavailable
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/2")
        self.artifact_storage_dir = Path(artifact_storage_dir or "batch_artifacts")
        self.artifact_storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_degraded_mode = enable_degraded_mode
        self.redis_available = False
        self.redis_client: Optional[redis.Redis] = None
        self.fallback_storage: Dict[str, Dict] = {}  # In-memory fallback
        
        # Initialize Redis connection pool
        if REDIS_AVAILABLE:
            try:
                self.redis_pool = ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=redis_pool_size,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    retry_on_timeout=True
                )
                self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"✅ Redis connection established: {self.redis_url}")
                
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                if self.enable_degraded_mode:
                    logger.warning("→ Operating in degraded mode (in-memory storage)")
                else:
                    raise
        else:
            logger.warning("⚠️ Redis library not available")
            if not self.enable_degraded_mode:
                raise RuntimeError("Redis library required but not available")
            logger.warning("→ Operating in degraded mode (in-memory storage)")
        
        logger.info(f"BatchJobManager initialized: artifacts_dir={self.artifact_storage_dir}")
    
    def create_job(
        self,
        pdm_path: str,
        municipality: str = "",
        department: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new batch job and submit to Celery.
        
        Args:
            pdm_path: Path to PDM document
            municipality: Municipality name
            department: Department name
            metadata: Additional job metadata
        
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "pdm_path": pdm_path,
            "municipality": municipality,
            "department": department,
            "state": JobState.QUEUED.value,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Store job state
        self._set_job_data(job_id, job_data)
        
        # Submit Celery task asynchronously
        if CELERY_AVAILABLE and celery_app:
            try:
                task = process_pdm_document.apply_async(
                    args=[job_id, pdm_path, municipality, department],
                    task_id=job_id
                )
                logger.info(f"Job created and submitted: {job_id} (task_id={task.id})")
            except Exception as e:
                logger.error(f"Failed to submit Celery task for job {job_id}: {e}")
                self.update_job_status(job_id, JobState.FAILED, error=str(e))
                raise
        else:
            error_msg = "Celery not available - cannot submit task"
            logger.error(error_msg)
            self.update_job_status(job_id, JobState.FAILED, error=error_msg)
            raise RuntimeError(error_msg)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get current job status.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job status dict with state, timestamps, and metadata
        """
        job_data = self._get_job_data(job_id)
        
        if not job_data:
            return {
                "job_id": job_id,
                "state": "not_found",
                "error": "Job not found"
            }
        
        return {
            "job_id": job_id,
            "state": job_data.get("state"),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at"),
            "pdm_path": job_data.get("pdm_path"),
            "municipality": job_data.get("municipality"),
            "department": job_data.get("department"),
            "error": job_data.get("error"),
            "metadata": job_data.get("metadata", {})
        }
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job results and artifacts.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Complete job results including evaluation data and artifact paths
        """
        job_data = self._get_job_data(job_id)
        
        if not job_data:
            return None
        
        if job_data.get("state") != JobState.COMPLETED.value:
            return {
                "job_id": job_id,
                "state": job_data.get("state"),
                "error": job_data.get("error"),
                "message": "Job not completed"
            }
        
        # Load result from artifact storage
        result_path = self.artifact_storage_dir / f"{job_id}_result.json"
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load result for job {job_id}: {e}")
                result_data = {}
        else:
            result_data = {}
        
        return {
            "job_id": job_id,
            "state": job_data.get("state"),
            "result": result_data,
            "artifacts": {
                "result_file": str(result_path) if result_path.exists() else None,
                "evidence_file": str(self.artifact_storage_dir / f"{job_id}_evidence.json"),
                "output_dir": str(self.artifact_storage_dir / job_id)
            },
            "created_at": job_data.get("created_at"),
            "completed_at": job_data.get("updated_at")
        }
    
    def update_job_status(
        self,
        job_id: str,
        state: JobState,
        error: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ):
        """
        Update job status with state transition.
        
        Args:
            job_id: Job identifier
            state: New job state
            error: Error message (if failed)
            result_data: Result data to store (if completed)
        """
        job_data = self._get_job_data(job_id) or {}
        
        old_state = job_data.get("state", "unknown")
        job_data["state"] = state.value
        job_data["updated_at"] = datetime.utcnow().isoformat()
        
        if error:
            job_data["error"] = error
        
        # Store result data as artifact
        if result_data and state == JobState.COMPLETED:
            result_path = self.artifact_storage_dir / f"{job_id}_result.json"
            try:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Stored result artifact: {result_path}")
            except Exception as e:
                logger.error(f"Failed to store result artifact for job {job_id}: {e}")
        
        self._set_job_data(job_id, job_data)
        
        logger.info(f"Job {job_id} state transition: {old_state} → {state.value}")
    
    def _get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data from Redis or fallback storage"""
        if self.redis_available and self.redis_client:
            try:
                data = self.redis_client.get(f"job:{job_id}")
                return json.loads(data) if data else None
            except Exception as e:
                logger.warning(f"Redis read failed for job {job_id}: {e}")
                if self.enable_degraded_mode:
                    return self.fallback_storage.get(job_id)
                raise
        else:
            return self.fallback_storage.get(job_id)
    
    def _set_job_data(self, job_id: str, job_data: Dict[str, Any]):
        """Set job data in Redis or fallback storage"""
        if self.redis_available and self.redis_client:
            try:
                self.redis_client.set(
                    f"job:{job_id}",
                    json.dumps(job_data),
                    ex=86400  # Expire after 24 hours
                )
            except Exception as e:
                logger.warning(f"Redis write failed for job {job_id}: {e}")
                if self.enable_degraded_mode:
                    self.fallback_storage[job_id] = job_data
                else:
                    raise
        else:
            self.fallback_storage[job_id] = job_data
    
    def close(self):
        """Close Redis connection pool"""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")


# ============================================================================
# Celery Worker Task
# ============================================================================

def _process_pdm_document_impl(
    job_id: str,
    pdm_path: str,
    municipality: str = "",
    department: str = ""
) -> Dict[str, Any]:
    """
    Celery task that processes PDM document through unified_evaluation_pipeline.
    
    This task:
    1. Updates job state to PROCESSING
    2. Invokes UnifiedEvaluationPipeline.evaluate()
    3. Captures results and artifacts
    4. Updates job state to COMPLETED or FAILED
    5. Maintains full compatibility with pipeline interface
    
    Args:
        job_id: Job identifier
        pdm_path: Path to PDM document
        municipality: Municipality name
        department: Department name
    
    Returns:
        Evaluation results dict
    """
    # Initialize job manager for status updates
    job_manager = BatchJobManager()
    
    logger.info(f"Processing job {job_id}: {pdm_path}")
    
    # Update state to PROCESSING
    job_manager.update_job_status(job_id, JobState.PROCESSING)
    
    try:
        # Create job-specific output directory
        output_dir = job_manager.artifact_storage_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize unified evaluation pipeline
        logger.info(f"Initializing UnifiedEvaluationPipeline for job {job_id}")
        pipeline = UnifiedEvaluationPipeline(config_path="system_configuration.json")
        
        # Execute evaluation with full parameter compatibility
        logger.info(f"Starting evaluation for job {job_id}")
        start_time = time.time()
        
        results = pipeline.evaluate(
            pdm_path=pdm_path,
            municipality=municipality,
            department=department,
            export_json=True,
            output_dir=str(output_dir)
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Evaluation completed for job {job_id} in {execution_time:.2f}s")
        
        # Validate results
        if results.get("status") == "pipeline_error":
            error_msg = results.get("error", "Unknown pipeline error")
            logger.error(f"Pipeline error for job {job_id}: {error_msg}")
            job_manager.update_job_status(job_id, JobState.FAILED, error=error_msg)
            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
        
        # Extract artifact paths from results
        artifact_info = {
            "output_dir": str(output_dir),
            "evidence_hash": results.get("evidence_registry", {}).get("deterministic_hash"),
            "execution_time_seconds": execution_time
        }
        
        # Store results as artifacts
        job_manager.update_job_status(
            job_id,
            JobState.COMPLETED,
            result_data={
                "results": results,
                "artifacts": artifact_info
            }
        )
        
        logger.info(f"✅ Job {job_id} completed successfully")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results": results,
            "artifacts": artifact_info
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}", exc_info=True)
        
        # Update state to FAILED
        job_manager.update_job_status(job_id, JobState.FAILED, error=error_msg)
        
        return {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
            "error_type": type(e).__name__
        }
    
    finally:
        job_manager.close()


# Register as Celery task if available
if CELERY_AVAILABLE and celery_app:
    process_pdm_document = celery_app.task(
        bind=True,
        name="batch_processor.process_pdm_document"
    )(_process_pdm_document_impl)
else:
    # Provide stub function for testing
    def process_pdm_document(*args, **kwargs):
        raise RuntimeError("Celery not available")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI for batch job management"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_processor.py create <pdm_path> [municipality] [department]")
        print("  python batch_processor.py status <job_id>")
        print("  python batch_processor.py result <job_id>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    manager = BatchJobManager()
    
    try:
        if command == "create":
            if len(sys.argv) < 3:
                print("Error: pdm_path required")
                sys.exit(1)
            
            pdm_path = sys.argv[2]
            municipality = sys.argv[3] if len(sys.argv) > 3 else ""
            department = sys.argv[4] if len(sys.argv) > 4 else ""
            
            job_id = manager.create_job(pdm_path, municipality, department)
            print(f"Job created: {job_id}")
        
        elif command == "status":
            if len(sys.argv) < 3:
                print("Error: job_id required")
                sys.exit(1)
            
            job_id = sys.argv[2]
            status = manager.get_job_status(job_id)
            print(json.dumps(status, indent=2))
        
        elif command == "result":
            if len(sys.argv) < 3:
                print("Error: job_id required")
                sys.exit(1)
            
            job_id = sys.argv[2]
            result = manager.get_job_result(job_id)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print(f"No result found for job {job_id}")
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    finally:
        manager.close()


if __name__ == "__main__":
    main()
