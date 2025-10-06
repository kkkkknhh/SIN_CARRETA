"""
FastAPI server for PDM document evaluation system.

Provides REST API endpoints for document upload, job status tracking,
results retrieval, and system health monitoring.
"""

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DECALOGO PDM Evaluation API",
    description="REST API for uploading and evaluating PDM documents against DECALOGO standards",
    version="1.0.0",
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

STAGING_DIR = Path(os.getenv("STAGING_DIR", "./data/staging"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./data/results"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))  # 24 hours
TASK_QUEUE_NAME = "pdm_evaluation_queue"

STAGING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_redis_client() -> redis.Redis:
    """Create Redis client with connection pooling."""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        client.ping()
        return client
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis service unavailable",
        )


class JobState(str, Enum):
    """Job processing states."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingConfig(BaseModel):
    """Optional processing configuration parameters."""
    enable_causal_analysis: bool = Field(True, description="Enable causal pattern detection")
    enable_contradiction_detection: bool = Field(True, description="Enable contradiction detection")
    enable_monetary_analysis: bool = Field(True, description="Enable monetary value detection")
    enable_teoria_cambio: bool = Field(True, description="Enable Theory of Change graph construction")
    strictness_level: str = Field("standard", description="Validation strictness: lenient, standard, strict")
    
    @validator("strictness_level")
    def validate_strictness(cls, v):
        allowed = ["lenient", "standard", "strict"]
        if v not in allowed:
            raise ValueError(f"strictness_level must be one of {allowed}")
        return v


class DocumentMetadata(BaseModel):
    """Required metadata for PDM document submission."""
    municipality: str = Field(..., min_length=1, max_length=200, description="Municipality name")
    department: str = Field(..., min_length=1, max_length=200, description="Department name")
    year: int = Field(..., ge=2000, le=2100, description="PDM year")
    period: str = Field(..., description="PDM period (e.g., '2024-2027')")
    requester_name: Optional[str] = Field(None, max_length=200, description="Name of requester")
    requester_email: Optional[str] = Field(None, max_length=200, description="Email of requester")
    notes: Optional[str] = Field(None, max_length=1000, description="Additional notes")
    
    @validator("period")
    def validate_period(cls, v):
        if not v or len(v) < 7:
            raise ValueError("period must be non-empty and formatted like '2024-2027'")
        return v


class UploadResponse(BaseModel):
    """Response for document upload."""
    job_id: str
    status: JobState
    document_count: int
    submission_time: str
    estimated_completion_time: Optional[str] = None


class JobProgress(BaseModel):
    """Job progress indicators."""
    current_step: str
    total_steps: int
    completed_steps: int
    progress_percentage: float
    
    @validator("progress_percentage")
    def validate_percentage(cls, v):
        return max(0.0, min(100.0, v))


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: JobState
    submission_time: str
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    progress: Optional[JobProgress] = None
    error_message: Optional[str] = None
    document_count: int
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    timestamp: str
    redis_connected: bool
    workers_available: int
    queue_size: int
    staging_dir_writable: bool
    results_dir_writable: bool


def validate_pdf_format(file_content: bytes, filename: str) -> bool:
    """
    Validate that file is a valid PDF document.
    
    Args:
        file_content: Raw file bytes
        filename: Original filename
        
    Returns:
        True if valid PDF
    """
    if not filename.lower().endswith('.pdf'):
        return False
    
    if len(file_content) < 4:
        return False
    
    pdf_header = file_content[:4]
    return pdf_header == b'%PDF'


def generate_deterministic_filename(job_id: str, original_filename: str, index: int) -> str:
    """
    Generate deterministic filename for uploaded document.
    
    Format: {job_id}_{index}_{sanitized_original_name}
    
    Args:
        job_id: Unique job identifier
        original_filename: Original uploaded filename
        index: Document index in batch
        
    Returns:
        Deterministic filename
    """
    sanitized = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_filename)
    sanitized = sanitized[:100]
    return f"{job_id}_{index:03d}_{sanitized}"


def store_job_metadata(
    redis_client: redis.Redis,
    job_id: str,
    metadata: DocumentMetadata,
    document_count: int,
    filenames: List[str],
    config: ProcessingConfig,
) -> None:
    """
    Store job metadata in Redis with TTL.
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier
        metadata: Document metadata
        document_count: Number of documents in batch
        filenames: List of stored filenames
        config: Processing configuration
    """
    job_data = {
        "job_id": job_id,
        "status": JobState.QUEUED.value,
        "submission_time": datetime.utcnow().isoformat(),
        "started_time": None,
        "completed_time": None,
        "document_count": document_count,
        "filenames": filenames,
        "metadata": metadata.dict(),
        "config": config.dict(),
        "progress": {
            "current_step": "queued",
            "total_steps": 10,
            "completed_steps": 0,
            "progress_percentage": 0.0,
        },
        "error_message": None,
    }
    
    redis_key = f"job:{job_id}"
    redis_client.setex(
        redis_key,
        JOB_TTL_SECONDS,
        json.dumps(job_data),
    )
    logger.info(f"Stored metadata for job {job_id} with TTL {JOB_TTL_SECONDS}s")


def enqueue_job(redis_client: redis.Redis, job_id: str) -> None:
    """
    Enqueue job to Redis-backed task queue for Celery workers.
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier to enqueue
    """
    task_data = {
        "job_id": job_id,
        "enqueued_at": datetime.utcnow().isoformat(),
        "task_type": "evaluate_pdm_documents",
    }
    
    redis_client.lpush(TASK_QUEUE_NAME, json.dumps(task_data))
    logger.info(f"Enqueued job {job_id} to task queue {TASK_QUEUE_NAME}")


def get_worker_count(redis_client: redis.Redis) -> int:
    """
    Get count of active workers by checking heartbeat keys.
    
    Workers should set heartbeat keys like worker:{worker_id}:heartbeat
    with short TTL that they refresh periodically.
    
    Args:
        redis_client: Redis client instance
        
    Returns:
        Number of active workers
    """
    try:
        pattern = "worker:*:heartbeat"
        worker_keys = list(redis_client.scan_iter(match=pattern, count=100))
        return len(worker_keys)
    except Exception as e:
        logger.error(f"Failed to get worker count: {e}")
        return 0


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDM documents (PDF format)"),
    municipality: str = Form(..., description="Municipality name"),
    department: str = Form(..., description="Department name"),
    year: int = Form(..., description="PDM year"),
    period: str = Form(..., description="PDM period"),
    requester_name: Optional[str] = Form(None, description="Requester name"),
    requester_email: Optional[str] = Form(None, description="Requester email"),
    notes: Optional[str] = Form(None, description="Additional notes"),
    enable_causal_analysis: bool = Form(True),
    enable_contradiction_detection: bool = Form(True),
    enable_monetary_analysis: bool = Form(True),
    enable_teoria_cambio: bool = Form(True),
    strictness_level: str = Form("standard"),
) -> UploadResponse:
    """
    Upload single or batch PDM documents for evaluation.
    
    Accepts multipart/form-data with:
    - One or more PDF files
    - Required metadata: municipality, department, year, period
    - Optional metadata: requester_name, requester_email, notes
    - Optional processing configuration parameters
    
    Returns:
        job_id: Unique identifier for tracking job status and retrieving results
        status: Initial job state (queued)
        document_count: Number of documents uploaded
        submission_time: UTC timestamp of submission
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided",
        )
    
    if len(files) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 files per batch",
        )
    
    metadata = DocumentMetadata(
        municipality=municipality,
        department=department,
        year=year,
        period=period,
        requester_name=requester_name,
        requester_email=requester_email,
        notes=notes,
    )
    
    config = ProcessingConfig(
        enable_causal_analysis=enable_causal_analysis,
        enable_contradiction_detection=enable_contradiction_detection,
        enable_monetary_analysis=enable_monetary_analysis,
        enable_teoria_cambio=enable_teoria_cambio,
        strictness_level=strictness_level,
    )
    
    job_id = str(uuid.uuid4())
    job_dir = STAGING_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    stored_filenames = []
    
    try:
        for idx, file in enumerate(files):
            file_content = await file.read()
            
            if len(file_content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE_MB}MB",
                )
            
            if not validate_pdf_format(file_content, file.filename):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a valid PDF document",
                )
            
            deterministic_filename = generate_deterministic_filename(
                job_id, file.filename, idx
            )
            file_path = job_dir / deterministic_filename
            
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            stored_filenames.append(deterministic_filename)
            logger.info(f"Stored file {file.filename} as {deterministic_filename}")
        
        redis_client = get_redis_client()
        
        store_job_metadata(
            redis_client=redis_client,
            job_id=job_id,
            metadata=metadata,
            document_count=len(files),
            filenames=stored_filenames,
            config=config,
        )
        
        enqueue_job(redis_client, job_id)
        
        submission_time = datetime.utcnow()
        estimated_completion = submission_time + timedelta(minutes=len(files) * 2)
        
        return UploadResponse(
            job_id=job_id,
            status=JobState.QUEUED,
            document_count=len(files),
            submission_time=submission_time.isoformat(),
            estimated_completion_time=estimated_completion.isoformat(),
        )
        
    except HTTPException:
        if job_dir.exists():
            shutil.rmtree(job_dir)
        raise
    except Exception as e:
        if job_dir.exists():
            shutil.rmtree(job_dir)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get current status and progress of a job.
    
    Returns job state with progress indicators:
    - queued: Job waiting in queue
    - processing: Job being evaluated by worker
    - completed: Evaluation finished, results available
    - failed: Evaluation failed with error message
    
    Args:
        job_id: Unique job identifier returned from upload
        
    Returns:
        Job status with progress information
    """
    redis_client = get_redis_client()
    redis_key = f"job:{job_id}"
    
    job_data_json = redis_client.get(redis_key)
    if not job_data_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or expired",
        )
    
    try:
        job_data = json.loads(job_data_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode job data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job data",
        )
    
    progress_data = job_data.get("progress")
    progress = None
    if progress_data:
        progress = JobProgress(**progress_data)
    
    return JobStatusResponse(
        job_id=job_id,
        status=JobState(job_data["status"]),
        submission_time=job_data["submission_time"],
        started_time=job_data.get("started_time"),
        completed_time=job_data.get("completed_time"),
        progress=progress,
        error_message=job_data.get("error_message"),
        document_count=job_data["document_count"],
        metadata=job_data.get("metadata", {}),
    )


@app.get("/results/{job_id}")
async def get_job_results(job_id: str, format: str = "json") -> FileResponse:
    """
    Retrieve completed evaluation results.
    
    Returns evaluation artifacts for a completed job.
    Supports multiple formats: json (default), pdf (report), zip (full package).
    
    Args:
        job_id: Unique job identifier
        format: Result format - 'json', 'pdf', or 'zip'
        
    Returns:
        FileResponse with requested artifact
    """
    redis_client = get_redis_client()
    redis_key = f"job:{job_id}"
    
    job_data_json = redis_client.get(redis_key)
    if not job_data_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or expired",
        )
    
    try:
        job_data = json.loads(job_data_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job data",
        )
    
    if job_data["status"] != JobState.COMPLETED.value:
        current_status = job_data["status"]
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed (current status: {current_status})",
        )
    
    results_dir = RESULTS_DIR / job_id
    if not results_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results not found",
        )
    
    if format == "json":
        result_file = results_dir / "evaluation_results.json"
        media_type = "application/json"
        filename = f"results_{job_id}.json"
    elif format == "pdf":
        result_file = results_dir / "evaluation_report.pdf"
        media_type = "application/pdf"
        filename = f"report_{job_id}.pdf"
    elif format == "zip":
        result_file = results_dir / "evaluation_package.zip"
        media_type = "application/zip"
        filename = f"package_{job_id}.zip"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format}. Use 'json', 'pdf', or 'zip'",
        )
    
    if not result_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result file not available in {format} format",
        )
    
    return FileResponse(
        path=result_file,
        media_type=media_type,
        filename=filename,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    System health and readiness check.
    
    Verifies:
    - Redis connectivity
    - Worker availability (active workers with heartbeat)
    - Queue size (pending jobs)
    - Filesystem writability for staging and results directories
    
    Returns:
        Health status with component checks
    """
    timestamp = datetime.utcnow().isoformat()
    
    redis_connected = False
    workers_available = 0
    queue_size = 0
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        redis_client.ping()
        redis_connected = True
        
        workers_available = get_worker_count(redis_client)
        queue_size = redis_client.llen(TASK_QUEUE_NAME)
        
    except Exception as e:
        logger.warning(f"Health check Redis error: {e}")
    
    staging_dir_writable = False
    try:
        test_file = STAGING_DIR / f".health_check_{time.time()}"
        test_file.write_text("health check")
        test_file.unlink()
        staging_dir_writable = True
    except Exception as e:
        logger.warning(f"Staging directory not writable: {e}")
    
    results_dir_writable = False
    try:
        test_file = RESULTS_DIR / f".health_check_{time.time()}"
        test_file.write_text("health check")
        test_file.unlink()
        results_dir_writable = True
    except Exception as e:
        logger.warning(f"Results directory not writable: {e}")
    
    overall_healthy = (
        redis_connected
        and workers_available > 0
        and staging_dir_writable
        and results_dir_writable
    )
    
    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        timestamp=timestamp,
        redis_connected=redis_connected,
        workers_available=workers_available,
        queue_size=queue_size,
        staging_dir_writable=staging_dir_writable,
        results_dir_writable=results_dir_writable,
    )


@app.get("/")
async def root():
    """API root with service information."""
    return {
        "service": "DECALOGO PDM Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload PDM documents",
            "status": "GET /status/{job_id} - Check job status",
            "results": "GET /results/{job_id} - Download results",
            "health": "GET /health - System health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting DECALOGO API server on {host}:{port}")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Staging: {STAGING_DIR}")
    logger.info(f"Results: {RESULTS_DIR}")
    
    uvicorn.run(app, host=host, port=port)
