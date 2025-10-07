# Batch Processing Infrastructure

## Overview

The DECALOGO PDM evaluation system includes comprehensive batch processing infrastructure for high-throughput document evaluation with deterministic guarantees.

## Architecture

```
┌──────────────┐
│  FastAPI     │  POST /upload → multipart form data (PDFs + metadata)
│  Server      │  GET /status/{job_id} → job progress tracking
│  (api/       │  GET /results/{job_id} → evaluation results
│   server.py) │  GET /metrics → Prometheus metrics
└──────┬───────┘
       │
       ↓
┌──────────────┐
│   Redis      │  Job metadata storage (24h TTL)
│   Backend    │  Task queue: pdm_evaluation_queue
│              │  Result caching
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────────────┐
│  Celery Workers (concurrency=8, prefetch=4)  │
│  ┌────────────────────────────────────────┐  │
│  │  process_document_task                 │  │
│  │  - Invokes unified_evaluation_pipeline │  │
│  │  - Stores artifacts to filesystem      │  │
│  │  - Updates job progress in Redis       │  │
│  │  - Emits Prometheus metrics            │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │  aggregate_batch_results_task          │  │
│  │  - Collects results from all documents │  │
│  │  - Validates coverage (300/300)        │  │
│  │  - Verifies deterministic hashes       │  │
│  └────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
       │
       ↓
┌──────────────┐
│  Filesystem  │  Artifacts: evaluation_results.json
│  Persistence │            evidence_registry.json
│              │            coverage_report.json
└──────────────┘
```

## Components

### 1. FastAPI Server (`api/server.py`)

REST API for document submission and result retrieval.

**Endpoints:**

- **POST /upload**: Upload PDM documents (multipart form data)
  - Files: PDF documents (max 50MB each, 100 files per batch)
  - Metadata: municipality, department, year, period
  - Optional: requester_name, requester_email, notes
  - Processing config: enable_causal_analysis, enable_contradiction_detection, etc.
  - Returns: `job_id`, `status`, `document_count`, `submission_time`

- **GET /status/{job_id}**: Track job progress
  - Returns: `status` (queued|processing|completed|failed), `progress`, `error_message`

- **GET /results/{job_id}**: Retrieve evaluation results
  - Query param `format`: json (default), pdf, zip
  - Returns: FileResponse with evaluation artifacts

- **GET /metrics**: Prometheus metrics endpoint

### 2. BatchJobManager (`batch_processor.py`)

Manages job lifecycle states and coordinates with Celery workers.

**Job States:**
- `QUEUED`: Job submitted, waiting for worker
- `PROCESSING`: Worker actively processing documents
- `COMPLETED`: All documents processed successfully
- `FAILED`: Processing failed with error message

**Key Methods:**
- `transition_to_processing(job_id)`: QUEUED → PROCESSING
- `update_progress(job_id, current_step, completed_steps, total_steps)`
- `transition_to_completed(job_id, results)`: PROCESSING → COMPLETED
- `transition_to_failed(job_id, error_message)`: → FAILED
- `store_artifacts(job_id, artifacts)`: Persist results to filesystem
- `get_queue_depth(queue_name)`: Monitor queue backlog

### 3. Celery Tasks (`celery_tasks.py`)

Asynchronous task definitions for document processing.

**Tasks:**

- **`process_document_task`**: Process single document
  - Invokes `UnifiedEvaluationPipeline.run_evaluation()`
  - Stores artifacts: `evaluation_results.json`, `evidence_registry.json`, `coverage_report.json`
  - Updates job progress in Redis
  - Emits Prometheus metrics
  - Retry policy: max 3 retries with exponential backoff

- **`aggregate_batch_results_task`**: Aggregate batch results
  - Collects results from all documents
  - Validates 300/300 coverage for each document
  - Verifies deterministic hash consistency
  - Transitions job to COMPLETED state

### 4. System Validators (`system_validators.py`)

Pre/post execution validation with resource and quality checks.

**Pre-Execution Validation (`validate_batch_pre_execution`):**
- Memory: ≥8GB available
- Disk space: ≥10GB available
- Redis: Connectivity check
- Celery workers: At least one worker available

**Post-Execution Validation (`validate_batch_post_execution`):**
- Coverage: 300/300 questions for each document
- Deterministic hashes: Consistency across documents
- Processing times: Statistics (mean, p50, p95)
- Throughput: Documents per hour calculation
- Error rate: Percentage of failed documents

### 5. Celery Configuration (`celeryconfig.py`)

Worker configuration optimized for batch processing:

```python
worker_concurrency = 8               # 8 parallel workers
worker_prefetch_multiplier = 4       # Prefetch 4 tasks per worker
worker_max_tasks_per_child = 100     # Restart worker after 100 tasks
task_time_limit = 600                # 10 minutes hard limit
task_soft_time_limit = 540           # 9 minutes soft limit
result_expires = 86400               # 24 hours result TTL
```

## Prometheus Metrics

Exported via `/metrics` endpoint:

### Counters
- **`documents_processed_total{status}`**: Total documents processed (by status: success, error)

### Gauges
- **`batch_throughput_per_hour`**: Current throughput (5-min rate * 3600)
- **`worker_utilization{worker_id}`**: Worker utilization percentage
- **`queue_depth{queue_name}`**: Number of documents in queue

### Histograms
- **`batch_document_processing_latency_seconds`**: Document processing latency distribution
  - Buckets: [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120]

## Alerting Rules (`prometheus_alerting_rules.yaml`)

### Critical Alerts

**BatchThroughputBelowSLA** (severity: critical)
- Threshold: < 170 docs/hour for 10 minutes
- Action: Immediate investigation required

**QueueDepthExceedsThreshold** (severity: critical)
- Threshold: > 100 documents for 5 minutes
- Action: Scale workers or investigate bottlenecks

**BatchP95LatencyExceedsSLA** (severity: critical)
- Threshold: > 21.2 seconds (p95) for 3 minutes
- Action: Performance optimization required

### Warning Alerts

**BatchThroughputDegraded** (severity: warning)
- Threshold: < 200 docs/hour (buffer zone) for 10 minutes

**WorkerUtilizationLow** (severity: warning)
- Threshold: < 50% for 10 minutes
- Possible causes: Queue starvation, worker failures

**BatchErrorRateHigh** (severity: warning)
- Threshold: > 5% error rate for 5 minutes

## CI/CD Integration (`.github/workflows/ci.yml`)

### `batch_load_test` Job

Validates batch processing with 10 concurrent documents:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports: [6379:6379]

steps:
  - Run batch load test (10 concurrent documents)
  - Verify throughput ≥ 170 docs/hour
  - Validate p95 latency ≤ 21.2 seconds
  - Check 300/300 coverage for all documents
  - Verify deterministic hash consistency
```

**Success Criteria:**
- Throughput: ≥ 170 documents/hour
- P95 Latency: ≤ 21.2 seconds per document
- Coverage: 300/300 questions for each document
- Hash Consistency: All documents produce same deterministic hash

### `stress_test` Job

Memory leak detection with 50 concurrent uploads:

```yaml
steps:
  - Process 50 documents in batches of 10
  - Monitor memory usage with psutil
  - Calculate memory growth percentage
  - Assert memory growth ≤ 20%
```

## Deployment

### 1. Start Redis

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### 2. Start Celery Workers

```bash
celery -A celery_tasks worker \
  --loglevel=info \
  --concurrency=8 \
  --prefetch-multiplier=4 \
  --max-tasks-per-child=100
```

### 3. Start FastAPI Server

```bash
uvicorn api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### 4. Monitor with Prometheus

Configure Prometheus scrape target:

```yaml
scrape_configs:
  - job_name: 'pdm-evaluation-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Load alerting rules:

```bash
prometheus --config.file=prometheus.yml \
  --rules.file=prometheus_alerting_rules.yaml
```

## Usage Example

### Upload Batch for Evaluation

```python
import httpx

files = [
    ("files", ("pdm_2024.pdf", open("pdm_2024.pdf", "rb"), "application/pdf")),
    ("files", ("pdm_2025.pdf", open("pdm_2025.pdf", "rb"), "application/pdf"))
]

data = {
    "municipality": "La Paz",
    "department": "La Paz",
    "year": 2024,
    "period": "2024-2027",
    "enable_causal_analysis": True,
    "enable_contradiction_detection": True
}

response = httpx.post("http://localhost:8000/upload", files=files, data=data)
job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")
```

### Poll Job Status

```python
import time

while True:
    response = httpx.get(f"http://localhost:8000/status/{job_id}")
    status = response.json()["status"]
    
    if status == "completed":
        print("✅ Job completed!")
        break
    elif status == "failed":
        print("❌ Job failed!")
        break
    
    progress = response.json()["progress"]["progress_percentage"]
    print(f"Processing: {progress:.1f}%")
    time.sleep(5)
```

### Download Results

```python
response = httpx.get(f"http://localhost:8000/results/{job_id}", params={"format": "json"})
with open("results.json", "wb") as f:
    f.write(response.content)
```

## Performance Targets

### Throughput
- **Target**: ≥ 170 documents/hour
- **Buffer**: 200 documents/hour (warning threshold)

### Latency
- **P95**: ≤ 21.2 seconds per document
- **P50**: ≤ 12 seconds per document

### Concurrency
- **Workers**: 8 concurrent workers
- **Prefetch**: 4 tasks per worker (32 total prefetched)

### Resource Requirements
- **Memory**: 8GB minimum, 16GB recommended
- **Disk**: 10GB minimum free space
- **Redis**: 2GB memory allocation

## Deterministic Guarantees

All batch processing respects existing deterministic pipeline guarantees:

1. **Freeze Verification**: Configuration snapshot verified before execution
2. **Canonical Flow Order**: 11-node execution order preserved
3. **Evidence Registry Integrity**: Deterministic hashing ensures reproducibility
4. **Coverage Validation**: 300/300 question coverage verified post-execution

## Testing

```bash
# Run batch processor unit tests
pytest test_batch_processor.py -v

# Run system validator tests
pytest test_system_validators.py -v

# Run batch load test (10 concurrent)
pytest test_batch_load.py -v

# Run stress test (50 concurrent)
pytest test_stress_test.py -v

# Run full CI suite
pytest test_*.py -v
```

## Monitoring Dashboard

Key metrics to monitor:

1. **Throughput**: `rate(documents_processed_total{status="success"}[5m]) * 3600`
2. **Queue Depth**: `queue_depth{queue_name="pdm_evaluation_queue"}`
3. **Worker Utilization**: `avg(worker_utilization) by (worker_id)`
4. **P95 Latency**: `histogram_quantile(0.95, batch_document_processing_latency_seconds)`
5. **Error Rate**: `rate(documents_processed_total{status="error"}[5m]) / rate(documents_processed_total[5m])`

## Troubleshooting

### High Queue Depth

**Symptom**: `queue_depth` > 100 for extended period

**Solutions:**
1. Scale up Celery workers: `celery -A celery_tasks worker --concurrency=16`
2. Check worker health: `celery -A celery_tasks inspect active`
3. Review error logs for stuck tasks

### Low Throughput

**Symptom**: `batch_throughput_per_hour` < 170

**Solutions:**
1. Increase worker concurrency
2. Optimize document preprocessing
3. Check resource constraints (CPU, memory, I/O)

### Memory Leaks

**Symptom**: Worker memory growth > 20%

**Solutions:**
1. Reduce `worker_max_tasks_per_child` to recycle workers more frequently
2. Review model caching strategies
3. Check for circular references in task results

### Worker Failures

**Symptom**: Workers crashing or becoming unresponsive

**Solutions:**
1. Check task time limits: `task_time_limit`, `task_soft_time_limit`
2. Review exception handling in `process_document_task`
3. Verify Redis connectivity and memory limits
4. Check for resource exhaustion (memory, file descriptors)

## Security Considerations

1. **File Validation**: All uploaded PDFs validated for format and size
2. **Input Sanitization**: Metadata fields sanitized and validated
3. **Job TTL**: Job data expires after 24 hours (configurable)
4. **Redis Authentication**: Use `REDIS_PASSWORD` environment variable
5. **Rate Limiting**: Implement rate limiting on `/upload` endpoint
6. **Access Control**: Add authentication/authorization middleware

## References

- [Celery Documentation](https://docs.celeryq.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Redis Best Practices](https://redis.io/topics/best-practices)
