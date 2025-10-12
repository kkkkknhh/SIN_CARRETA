"""
Comprehensive test suite for FastAPI server.

Tests all endpoints including upload, status, results, and health checks.
"""

import io
import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import (
    JobState,
    app,
    generate_deterministic_filename,
    validate_pdf_format,
)


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch("api.server.get_redis_client") as mock:
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        redis_mock.setex.return_value = True
        redis_mock.lpush.return_value = 1
        redis_mock.get.return_value = None
        redis_mock.llen.return_value = 0
        redis_mock.scan_iter.return_value = []
        mock.return_value = redis_mock
        yield redis_mock


@pytest.fixture
def temp_dirs():
    """Create temporary staging and results directories."""
    temp_staging = Path(tempfile.mkdtemp())
    temp_results = Path(tempfile.mkdtemp())

    with patch("api.server.STAGING_DIR", temp_staging):
        with patch("api.server.RESULTS_DIR", temp_results):
            yield temp_staging, temp_results

    shutil.rmtree(temp_staging, ignore_errors=True)
    shutil.rmtree(temp_results, ignore_errors=True)


@pytest.fixture
def sample_pdf():
    """Create a minimal valid PDF file for testing."""
    pdf_content = (
        b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"
    )
    return io.BytesIO(pdf_content)


def test_root_endpoint(client):
    """Test root endpoint returns service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "DECALOGO PDM Evaluation API"
    assert "endpoints" in data


def test_validate_pdf_format_valid():
    """Test PDF validation accepts valid PDF files."""
    valid_pdf = b"%PDF-1.4\nsome content"
    assert validate_pdf_format(valid_pdf, "test.pdf") is True


def test_validate_pdf_format_invalid_extension():
    """Test PDF validation rejects non-PDF extensions."""
    content = b"%PDF-1.4\nsome content"
    assert validate_pdf_format(content, "test.txt") is False


def test_validate_pdf_format_invalid_header():
    """Test PDF validation rejects files without PDF header."""
    invalid_content = b"not a pdf file"
    assert validate_pdf_format(invalid_content, "test.pdf") is False


def test_validate_pdf_format_too_short():
    """Test PDF validation rejects files that are too short."""
    short_content = b"PDF"
    assert validate_pdf_format(short_content, "test.pdf") is False


def test_generate_deterministic_filename():
    """Test deterministic filename generation."""
    job_id = "test-job-123"
    original = "my_document.pdf"
    index = 5

    result = generate_deterministic_filename(job_id, original, index)

    assert result.startswith("test-job-123_005_")
    assert "my_document.pdf" in result


def test_generate_deterministic_filename_sanitization():
    """Test filename sanitization removes special characters."""
    job_id = "job-123"
    original = "my/doc#@!ument.pdf"
    index = 0

    result = generate_deterministic_filename(job_id, original, index)

    assert "/" not in result
    assert "#" not in result
    assert "@" not in result
    assert "!" not in result


def test_generate_deterministic_filename_truncation():
    """Test filename truncation for very long names."""
    job_id = "job-123"
    original = "a" * 200 + ".pdf"
    index = 0

    result = generate_deterministic_filename(job_id, original, index)

    assert len(result) < 200


def test_upload_single_document(client, mock_redis, temp_dirs, sample_pdf):
    """Test uploading a single PDF document."""
    staging_dir, _ = temp_dirs

    files = [("files", ("document.pdf", sample_pdf, "application/pdf"))]
    data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca",
        "year": 2024,
        "period": "2024-2027",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 201
    result = response.json()
    assert "job_id" in result
    assert result["status"] == "queued"
    assert result["document_count"] == 1
    assert "submission_time" in result

    mock_redis.setex.assert_called_once()
    mock_redis.lpush.assert_called_once()


def test_upload_batch_documents(client, mock_redis, temp_dirs):
    """Test uploading multiple documents in batch."""
    staging_dir, _ = temp_dirs

    pdf_content = b"%PDF-1.4\ntest content"
    files = [
        ("files", ("doc1.pdf", io.BytesIO(pdf_content), "application/pdf")),
        ("files", ("doc2.pdf", io.BytesIO(pdf_content), "application/pdf")),
        ("files", ("doc3.pdf", io.BytesIO(pdf_content), "application/pdf")),
    ]
    data = {
        "municipality": "Medellín",
        "department": "Antioquia",
        "year": 2024,
        "period": "2024-2027",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 201
    result = response.json()
    assert result["document_count"] == 3


def test_upload_with_optional_metadata(client, mock_redis, temp_dirs, sample_pdf):
    """Test upload with optional requester metadata."""
    staging_dir, _ = temp_dirs

    files = [("files", ("document.pdf", sample_pdf, "application/pdf"))]
    data = {
        "municipality": "Cali",
        "department": "Valle del Cauca",
        "year": 2024,
        "period": "2024-2027",
        "requester_name": "John Doe",
        "requester_email": "john@example.com",
        "notes": "Test upload",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 201


def test_upload_with_processing_config(client, mock_redis, temp_dirs, sample_pdf):
    """Test upload with custom processing configuration."""
    staging_dir, _ = temp_dirs

    files = [("files", ("document.pdf", sample_pdf, "application/pdf"))]
    data = {
        "municipality": "Cartagena",
        "department": "Bolívar",
        "year": 2024,
        "period": "2024-2027",
        "enable_causal_analysis": False,
        "enable_contradiction_detection": True,
        "strictness_level": "strict",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 201


def test_upload_missing_required_field(client, mock_redis, temp_dirs, sample_pdf):
    """Test upload fails when required metadata field is missing."""
    files = [("files", ("document.pdf", sample_pdf, "application/pdf"))]
    data = {
        "municipality": "Bogotá",
        "year": 2024,
        "period": "2024-2027",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 422


def test_upload_invalid_year(client, mock_redis, temp_dirs, sample_pdf):
    """Test upload fails with invalid year."""
    files = [("files", ("document.pdf", sample_pdf, "application/pdf"))]
    data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca",
        "year": 1999,
        "period": "2024-2027",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 422


def test_upload_invalid_strictness_level(client, mock_redis, temp_dirs, sample_pdf):
    """Test upload fails with invalid strictness level."""
    files = [("files", ("document.pdf", sample_pdf, "application/pdf"))]
    data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca",
        "year": 2024,
        "period": "2024-2027",
        "strictness_level": "invalid",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 422


def test_upload_no_files(client, mock_redis, temp_dirs):
    """Test upload fails when no files provided."""
    data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca",
        "year": 2024,
        "period": "2024-2027",
    }

    response = client.post("/upload", data=data)

    assert response.status_code == 422


def test_upload_invalid_pdf(client, mock_redis, temp_dirs):
    """Test upload fails with invalid PDF file."""
    staging_dir, _ = temp_dirs

    invalid_content = b"not a pdf file"
    files = [
        ("files", ("document.pdf", io.BytesIO(invalid_content), "application/pdf"))
    ]
    data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca",
        "year": 2024,
        "period": "2024-2027",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_upload_too_many_files(client, mock_redis, temp_dirs):
    """Test upload fails when exceeding batch limit."""
    staging_dir, _ = temp_dirs

    pdf_content = b"%PDF-1.4\ntest"
    files = [
        ("files", (f"doc{i}.pdf", io.BytesIO(pdf_content), "application/pdf"))
        for i in range(101)
    ]
    data = {
        "municipality": "Bogotá",
        "department": "Cundinamarca",
        "year": 2024,
        "period": "2024-2027",
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 400
    assert "Maximum 100 files" in response.json()["detail"]


def test_get_status_queued(client, mock_redis):
    """Test getting status of queued job."""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.QUEUED.value,
        "submission_time": datetime.utcnow().isoformat(),
        "started_time": None,
        "completed_time": None,
        "document_count": 2,
        "metadata": {},
        "progress": {
            "current_step": "queued",
            "total_steps": 10,
            "completed_steps": 0,
            "progress_percentage": 0.0,
        },
        "error_message": None,
    }
    mock_redis.get.return_value = json.dumps(job_data)

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    result = response.json()
    assert result["job_id"] == job_id
    assert result["status"] == "queued"
    assert result["document_count"] == 2
    assert result["progress"]["progress_percentage"] == 0.0


def test_get_status_processing(client, mock_redis):
    """Test getting status of processing job."""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.PROCESSING.value,
        "submission_time": datetime.utcnow().isoformat(),
        "started_time": datetime.utcnow().isoformat(),
        "completed_time": None,
        "document_count": 1,
        "metadata": {},
        "progress": {
            "current_step": "embedding_generation",
            "total_steps": 10,
            "completed_steps": 5,
            "progress_percentage": 50.0,
        },
        "error_message": None,
    }
    mock_redis.get.return_value = json.dumps(job_data)

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "processing"
    assert result["started_time"] is not None
    assert result["progress"]["progress_percentage"] == 50.0


def test_get_status_completed(client, mock_redis):
    """Test getting status of completed job."""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.COMPLETED.value,
        "submission_time": datetime.utcnow().isoformat(),
        "started_time": datetime.utcnow().isoformat(),
        "completed_time": datetime.utcnow().isoformat(),
        "document_count": 1,
        "metadata": {},
        "progress": {
            "current_step": "completed",
            "total_steps": 10,
            "completed_steps": 10,
            "progress_percentage": 100.0,
        },
        "error_message": None,
    }
    mock_redis.get.return_value = json.dumps(job_data)

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["completed_time"] is not None
    assert result["progress"]["progress_percentage"] == 100.0


def test_get_status_failed(client, mock_redis):
    """Test getting status of failed job."""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.FAILED.value,
        "submission_time": datetime.utcnow().isoformat(),
        "started_time": datetime.utcnow().isoformat(),
        "completed_time": datetime.utcnow().isoformat(),
        "document_count": 1,
        "metadata": {},
        "progress": None,
        "error_message": "Processing error occurred",
    }
    mock_redis.get.return_value = json.dumps(job_data)

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed"
    assert result["error_message"] == "Processing error occurred"


def test_get_status_not_found(client, mock_redis):
    """Test getting status of non-existent job."""
    job_id = str(uuid.uuid4())
    mock_redis.get.return_value = None

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 404
    assert "not found or expired" in response.json()["detail"]


def test_get_results_json(client, mock_redis, temp_dirs):
    """Test retrieving results in JSON format."""
    _, results_dir = temp_dirs

    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.COMPLETED.value,
        "submission_time": datetime.utcnow().isoformat(),
        "completed_time": datetime.utcnow().isoformat(),
        "document_count": 1,
        "metadata": {},
    }
    mock_redis.get.return_value = json.dumps(job_data)

    job_results_dir = results_dir / job_id
    job_results_dir.mkdir(parents=True)
    result_file = job_results_dir / "evaluation_results.json"
    result_file.write_text(json.dumps({"score": 85.5}))

    response = client.get(f"/results/{job_id}?format=json")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


def test_get_results_not_completed(client, mock_redis):
    """Test retrieving results for non-completed job."""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.PROCESSING.value,
        "submission_time": datetime.utcnow().isoformat(),
        "document_count": 1,
        "metadata": {},
    }
    mock_redis.get.return_value = json.dumps(job_data)

    response = client.get(f"/results/{job_id}")

    assert response.status_code == 409
    assert "not completed" in response.json()["detail"]


def test_get_results_not_found(client, mock_redis):
    """Test retrieving results for non-existent job."""
    job_id = str(uuid.uuid4())
    mock_redis.get.return_value = None

    response = client.get(f"/results/{job_id}")

    assert response.status_code == 404


def test_get_results_invalid_format(client, mock_redis, temp_dirs):
    """Test retrieving results with invalid format."""
    _, results_dir = temp_dirs

    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "status": JobState.COMPLETED.value,
        "submission_time": datetime.utcnow().isoformat(),
        "document_count": 1,
        "metadata": {},
    }
    mock_redis.get.return_value = json.dumps(job_data)

    job_results_dir = results_dir / job_id
    job_results_dir.mkdir(parents=True)

    response = client.get(f"/results/{job_id}?format=invalid")

    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_health_check_all_healthy(client, mock_redis):
    """Test health check when all components are healthy."""
    mock_redis.ping.return_value = True
    mock_redis.llen.return_value = 5
    mock_redis.scan_iter.return_value = ["worker:1:heartbeat", "worker:2:heartbeat"]

    response = client.get("/health")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "healthy"
    assert result["redis_connected"] is True
    assert result["workers_available"] == 2
    assert result["queue_size"] == 5
    assert result["staging_dir_writable"] is True
    assert result["results_dir_writable"] is True


def test_health_check_degraded_no_workers(client, mock_redis):
    """Test health check when no workers available."""
    mock_redis.ping.return_value = True
    mock_redis.llen.return_value = 10
    mock_redis.scan_iter.return_value = []

    response = client.get("/health")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "degraded"
    assert result["workers_available"] == 0


def test_health_check_redis_unavailable(client):
    """Test health check when Redis is unavailable."""
    with patch("api.server.redis.Redis") as mock_redis_class:
        mock_instance = MagicMock()
        mock_instance.ping.side_effect = Exception("Connection failed")
        mock_redis_class.return_value = mock_instance

        response = client.get("/health")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "degraded"
        assert result["redis_connected"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
