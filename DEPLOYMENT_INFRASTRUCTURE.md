# Deployment Infrastructure

Complete canary deployment, distributed tracing, and SLO monitoring infrastructure for the DECALOGO evaluation system.

## Overview

This infrastructure provides:

1. **Canary Deployment** - Progressive traffic routing (5% → 25% → 100%) with automated rollback
2. **OpenTelemetry Distributed Tracing** - Instrumentation for 28 critical flows and 11 pipeline components
3. **SLO Monitoring** - Real-time monitoring with 99.5% availability, 200ms p95 latency, and 0.1% error rate thresholds
4. **Alerting** - Automated alerts for contract violations, performance regressions, and fault recovery failures

## Components

### 1. Canary Deployment (`canary_deployment.py`)

Progressive traffic routing with automated rollback on metric violations.

#### Features

- **Progressive Release**: 5% → 25% → 100% traffic routing
- **Configurable Hold Durations**: Set hold times at each stage
- **Automated Rollback Triggers**:
  - Contract violations (immediate)
  - Error rate > 10%
  - P95 latency > 500ms
- **Real-time Metrics Collection**: Request counts, latencies, error rates

#### Usage

```python
from canary_deployment import create_canary_controller

# Create controller
controller = create_canary_controller(
    deployment_id="my-deployment-v2.0",
    canary_5_hold_seconds=300,      # 5 minutes at 5%
    canary_25_hold_seconds=600,     # 10 minutes at 25%
    full_rollout_hold_seconds=1800, # 30 minutes at 100%
    max_error_rate=10.0,            # Rollback if error rate > 10%
    max_p95_latency_ms=500.0        # Rollback if p95 > 500ms
)

# Define request handler
def request_generator():
    request_id = generate_request_id()
    version = controller.router.route_request(request_id)
    
    # Process request with appropriate version
    result = process_request(request_id, version)
    
    return (
        request_id,
        result.latency_ms,
        result.success,
        result.contract_valid
    )

# Execute deployment
result = controller.execute_deployment(request_generator)

if result.success:
    print(f"Deployment completed: {result.final_stage}")
else:
    print(f"Rollback triggered: {result.rollback_reason}")

# Export metrics
controller.export_metrics("output/deployment_metrics.json")
```

#### API Reference

**`create_canary_controller(deployment_id, ...)`**
- Creates a configured canary deployment controller
- Returns: `CanaryDeploymentController`

**`CanaryDeploymentController.execute_deployment(request_generator)`**
- Executes full canary deployment progression
- Args: `request_generator` - Callable returning `(request_id, latency_ms, success, contract_valid)`
- Returns: `DeploymentResult`

**`TrafficRouter.route_request(request_id)`**
- Routes a request to baseline or canary version
- Args: `request_id` - Unique request identifier
- Returns: `"baseline"` or `"canary"`

### 2. OpenTelemetry Instrumentation (`opentelemetry_instrumentation.py`)

Distributed tracing for all critical flows and pipeline components.

#### 28 Critical Flows Instrumented

**Document Processing (5)**:
- Document ingestion
- Document segmentation
- Text normalization
- Embedding generation
- Similarity calculation

**Evidence Extraction (8)**:
- Causal pattern detection
- Monetary detection
- Responsibility detection
- Feasibility scoring
- Contradiction detection
- Teoria Cambio analysis
- Policy alignment
- Indicator extraction

**Evaluation (5)**:
- Decálogo evaluation
- Questionnaire evaluation
- Rubric scoring
- Evidence aggregation
- Result synthesis

**Validation (5)**:
- Contract validation
- DAG validation
- Immutability verification
- Determinism verification
- Reproducibility verification

**Infrastructure (5)**:
- Circuit breaker
- Memory watchdog
- Error recovery
- Health check
- Metric collection

#### 11 Pipeline Components Instrumented

- Document segmenter
- Embedding model
- Causal pattern detector
- Monetary detector
- Responsibility detector
- Feasibility scorer
- Contradiction detector
- Teoria Cambio
- Questionnaire engine
- Evidence registry
- Pipeline orchestrator

#### Usage

```python
from opentelemetry_instrumentation import (
    initialize_tracing,
    trace_flow,
    trace_component,
    FlowType,
    ComponentType,
    create_span_logger
)

# Initialize tracing
initialize_tracing(
    service_name="decalogo-evaluation-system",
    service_version="1.0.0",
    environment="production"
)

# Trace a critical flow
@trace_flow(FlowType.DECALOGO_EVALUATION)
def evaluate_plan(plan_text: str) -> dict:
    results = {}
    # ... evaluation logic ...
    return results

# Trace a pipeline component
@trace_component(ComponentType.EMBEDDING_MODEL)
def encode_text(text: str) -> list:
    embeddings = model.encode(text)
    return embeddings

# Create span-aware logger with trace correlation
logger = create_span_logger(__name__)
logger.info("Processing plan", plan_id="12345")
```

#### Context Propagation

```python
from opentelemetry_instrumentation import get_tracing_manager

manager = get_tracing_manager()

# Inject context for cross-service calls
headers = {}
manager.inject_context(headers)
response = requests.post(url, headers=headers)

# Extract context from incoming requests
context = manager.extract_context(request.headers)
```

### 3. SLO Monitoring (`slo_monitoring.py`)

Real-time SLO monitoring with alerting.

#### SLO Thresholds

- **Availability**: 99.5%
- **P95 Latency**: 200ms
- **Error Rate**: 0.1%

#### Additional Thresholds

- **Performance Regression** (Phase 3): 10% threshold
- **Fault Recovery** (Phase 4): 1.5s p99 recovery time
- **Contract Violations** (Phase 6): Immediate alert

#### Usage

```python
from slo_monitoring import create_slo_monitor, DashboardDataGenerator

# Create SLO monitor
monitor = create_slo_monitor(
    availability_threshold=99.5,
    p95_latency_threshold_ms=200.0,
    error_rate_threshold=0.1,
    performance_regression_threshold=10.0,
    p99_recovery_time_threshold_ms=1500.0
)

# Record requests
monitor.record_request(
    flow_name="decalogo_evaluation",
    success=True,
    latency_ms=150.0,
    contract_valid=True
)

# Record fault recovery
monitor.record_recovery(
    flow_name="document_processing",
    success=True,
    recovery_time_ms=1200.0
)

# Set baseline for performance regression detection
monitor.set_baseline("decalogo_evaluation")

# Check SLO status
status = monitor.check_slo_status("decalogo_evaluation")
print(f"Availability: {status.availability}%")
print(f"P95 Latency: {status.p95_latency_ms}ms")
print(f"Error Rate: {status.error_rate_percent}%")
print(f"Overall SLO Met: {status.overall_slo_met}")

# Evaluate alert rules
alerts = monitor.evaluate_alert_rules()
for alert in alerts:
    print(f"Alert: {alert.message}")
    print(f"Severity: {alert.severity.value}")

# Generate dashboard data
generator = DashboardDataGenerator(monitor)
dashboard = generator.generate_dashboard_data()
generator.export_dashboard_json("output/slo_dashboard.json")
```

#### Dashboard Structure

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "overall": {
    "total_flows": 28,
    "flows_meeting_slo": 27,
    "slo_compliance_percent": 96.4,
    "active_alerts": 1
  },
  "thresholds": {
    "availability_percent": 99.5,
    "p95_latency_ms": 200.0,
    "error_rate_percent": 0.1
  },
  "flows": {
    "decalogo_evaluation": {
      "availability": {
        "value": 99.8,
        "threshold": 99.5,
        "slo_met": true,
        "status_indicator": "green"
      },
      "p95_latency": {
        "value_ms": 180.5,
        "threshold_ms": 200.0,
        "slo_met": true,
        "status_indicator": "green"
      },
      "error_rate": {
        "value_percent": 0.05,
        "threshold_percent": 0.1,
        "slo_met": true,
        "status_indicator": "green"
      },
      "overall_slo_met": true
    }
  },
  "alerts": []
}
```

### 4. Alert Types

#### Contract Violation Alerts
- **Severity**: CRITICAL
- **Trigger**: Any contract validation failure
- **Action**: Immediate rollback in canary deployment

#### Performance Regression Alerts (Phase 3)
- **Severity**: WARNING
- **Trigger**: P95 latency > 10% above baseline
- **Integration**: Compares against baseline metrics

#### Fault Recovery Alerts (Phase 4)
- **Severity**: CRITICAL
- **Trigger**: P99 recovery time > 1.5s
- **Integration**: Monitors circuit breaker and retry mechanisms

#### SLO Breach Alerts
- **Availability Degraded**: Availability < 99.5%
- **Latency Exceeded**: P95 latency > 200ms
- **Error Rate Exceeded**: Error rate > 0.1%

## Integration with Existing Phases

### Phase 0: Structured Logging
- OpenTelemetry trace IDs automatically correlated with log messages
- `SpanLogger` adds trace context to all log entries
- Enables distributed trace debugging across services

### Phase 3: Performance Benchmarking
- SLO monitor tracks performance regression against baselines
- Automated alerts when performance degrades > 10%
- Historical metrics for trend analysis

### Phase 4: Fault Injection
- Monitors fault recovery times
- Alerts on p99 recovery time > 1.5s
- Integrates with circuit breaker patterns

### Phase 6: Contract Validation
- Immediate rollback on contract violations
- Contract validation results tracked in metrics
- CRITICAL alerts for any validation failures

## Testing

### Unit Tests

```bash
# Run all tests
python3 -m pytest test_canary_deployment.py test_opentelemetry_instrumentation.py test_slo_monitoring.py -v

# Run specific test classes
python3 -m pytest test_canary_deployment.py::TestTrafficRouting -v
python3 -m pytest test_slo_monitoring.py::TestAlertGeneration -v
```

### Integration Test

```bash
# Run full integration test
python3 test_deployment_integration.py
```

### Example Deployment

```bash
# Run example deployment
python3 deployment_example.py
```

## Architecture

### Canary Deployment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Canary Deployment Controller                  │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │ Traffic Router │                │   Metrics   │
        │   5%→25%→100%  │                │  Collector  │
        └───────┬────────┘                └──────┬──────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │   Baseline     │                │  Rollback   │
        │    Version     │                │   Triggers  │
        └────────────────┘                └─────────────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │    Canary      │                │  Contract   │
        │    Version     │◄───────────────┤  Error Rate │
        └────────────────┘                │   Latency   │
                                          └─────────────┘
```

### OpenTelemetry Tracing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tracing Manager                             │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │  Span Creation │                │   Context   │
        │  28 Flows      │                │ Propagation │
        │  11 Components │                │             │
        └───────┬────────┘                └──────┬──────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │  Decorators    │                │  Injection  │
        │  @trace_flow   │                │  Extraction │
        │  @trace_comp.  │                │             │
        └───────┬────────┘                └──────┬──────┘
                │                                 │
        ┌───────▼────────────────────────────────▼──────┐
        │         Structured Logging Integration         │
        │            (Phase 0 Correlation)               │
        └────────────────────────────────────────────────┘
```

### SLO Monitoring Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        SLO Monitor                               │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │    Metrics     │                │   Alert     │
        │  Aggregator    │                │   Rules     │
        │  (5 min win.)  │                │             │
        └───────┬────────┘                └──────┬──────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │  Availability  │                │  Contract   │
        │  P95 Latency   │                │  Violations │
        │  Error Rate    │                │ (Phase 6)   │
        └───────┬────────┘                └──────┬──────┘
                │                                 │
        ┌───────▼────────┐                ┌──────▼──────┐
        │  Performance   │                │    Fault    │
        │  Regression    │                │  Recovery   │
        │  (Phase 3)     │                │  (Phase 4)  │
        └───────┬────────┘                └──────┬──────┘
                │                                 │
        ┌───────▼────────────────────────────────▼──────┐
        │            Dashboard Data Generator            │
        └────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# OpenTelemetry
export OTEL_SERVICE_NAME="decalogo-evaluation-system"
export OTEL_SERVICE_VERSION="1.0.0"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# SLO Thresholds
export SLO_AVAILABILITY_THRESHOLD=99.5
export SLO_P95_LATENCY_MS=200.0
export SLO_ERROR_RATE_PERCENT=0.1

# Canary Configuration
export CANARY_5_HOLD_SECONDS=300
export CANARY_25_HOLD_SECONDS=600
export CANARY_FULL_HOLD_SECONDS=1800
```

### Configuration File

```json
{
  "deployment": {
    "canary_stages": [
      {"percentage": 5, "hold_seconds": 300},
      {"percentage": 25, "hold_seconds": 600},
      {"percentage": 100, "hold_seconds": 1800}
    ],
    "rollback_thresholds": {
      "max_error_rate": 10.0,
      "max_p95_latency_ms": 500.0,
      "metric_window_seconds": 60,
      "min_samples": 10
    }
  },
  "tracing": {
    "service_name": "decalogo-evaluation-system",
    "service_version": "1.0.0",
    "environment": "production",
    "sampling_rate": 1.0
  },
  "slo": {
    "availability_percent": 99.5,
    "p95_latency_ms": 200.0,
    "error_rate_percent": 0.1,
    "performance_regression_percent": 10.0,
    "p99_recovery_time_ms": 1500.0,
    "aggregation_window_seconds": 300
  }
}
```

## Deployment Checklist

1. **Pre-Deployment**
   - [ ] Initialize OpenTelemetry tracing
   - [ ] Set baseline metrics for performance regression detection
   - [ ] Configure SLO thresholds
   - [ ] Verify contract validation is enabled

2. **During Deployment**
   - [ ] Monitor canary metrics in real-time
   - [ ] Watch for automated rollback triggers
   - [ ] Check SLO dashboard for compliance
   - [ ] Review distributed traces for issues

3. **Post-Deployment**
   - [ ] Verify all stages completed successfully
   - [ ] Review alert history
   - [ ] Update baseline metrics
   - [ ] Export and archive deployment metrics

## Troubleshooting

### High Latency in Canary

**Symptom**: P95 latency exceeds 500ms threshold

**Investigation**:
1. Check distributed traces for slow operations
2. Review SLO dashboard for affected flows
3. Compare with baseline metrics for regression

**Resolution**:
- Automated rollback triggers if threshold exceeded
- Manual investigation of trace spans
- Performance optimization of affected components

### Contract Violations

**Symptom**: Immediate rollback due to contract validation failure

**Investigation**:
1. Review contract violation details in metrics
2. Check trace context for failing request
3. Examine Phase 6 contract validation logs

**Resolution**:
- Automated rollback to baseline version
- Fix contract validation issue
- Re-deploy after validation passes

### SLO Breaches

**Symptom**: Active alerts for SLO threshold breaches

**Investigation**:
1. Check SLO dashboard for affected flows
2. Review metrics history for trends
3. Correlate with deployment events

**Resolution**:
- Investigate root cause using distributed traces
- Implement fixes for affected components
- Re-deploy with canary monitoring

## Performance Considerations

### Canary Deployment
- **Overhead**: ~1-2ms per request for routing decision
- **Memory**: ~10MB for metrics storage per 10k requests
- **CPU**: Negligible impact (<1%)

### OpenTelemetry Tracing
- **Overhead**: ~0.5-1ms per traced operation
- **Memory**: ~5MB per 10k spans (with batching)
- **CPU**: ~2-3% with sampling rate 1.0

### SLO Monitoring
- **Overhead**: ~0.5ms per metric recording
- **Memory**: ~20MB for 5-minute window across all flows
- **CPU**: ~1% for metric aggregation

## Production Recommendations

1. **Sampling Rate**: Use 0.1-1.0 sampling for production tracing
2. **Metric Retention**: Keep 30 days of SLO metrics
3. **Alert Fatigue**: Set appropriate thresholds to avoid noise
4. **Dashboard Refresh**: Update every 30-60 seconds
5. **Canary Duration**: Use longer hold times (15-30 minutes) per stage
6. **Baseline Updates**: Refresh baselines weekly during stable periods

## Batch Processing Infrastructure

High-throughput asynchronous batch processing system for evaluating large document collections.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Client Applications                           │
│  (Python scripts, Shell scripts, Postman, Web UI)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP/HTTPS
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Nginx Reverse Proxy                            │
│  • SSL termination                                                    │
│  • Rate limiting (100 req/min per IP)                                │
│  • Load balancing across FastAPI instances                           │
│  • Request buffering (client_max_body_size: 100MB)                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server(s)                              │
│  • POST /api/v1/batch/upload       - Upload documents                │
│  • GET  /api/v1/batch/status/{id}  - Check job status                │
│  • GET  /api/v1/batch/results/{id} - Retrieve results                │
│  • GET  /api/v1/health             - Health check                    │
│  • Token-based authentication (Bearer tokens)                        │
│  • Request validation (1-100 documents per batch)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Enqueue tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Redis Message Broker                             │
│  • Task queue: decalogo_batch_tasks                                  │
│  • Result backend: decalogo_batch_results                            │
│  • Priority queuing (high/normal/low)                                │
│  • TTL: 7 days for tasks, 30 days for results                       │
│  • Persistence: AOF + RDB snapshots every 60s                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Consume tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Celery Workers (N instances)                  │
│  • Concurrency: 4 processes per worker (prefork)                     │
│  • Task timeout: 300s per document                                   │
│  • Memory limit: 4GB per worker process                              │
│  • Autoscaling: 2-8 workers based on queue depth                     │
│  • Soft time limit: 280s (20s grace for cleanup)                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Store results
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Artifact Storage Layer                           │
│  • Redis: Metadata + small results (<1MB)                            │
│  • S3/Minio: Large result artifacts (>1MB)                           │
│  • Retention: 30 days automatic cleanup                              │
│  • Compression: gzip for JSON results                                │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   Upload   │────▶│   Queue    │────▶│  Process   │────▶│   Store    │
│  Batch Job │     │   Tasks    │     │ Documents  │     │  Results   │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
      │                  │                   │                   │
      ▼                  ▼                   ▼                   ▼
  Validate          Prioritize          Execute            Compress
  Documents         by priority         evaluation         & persist
  Generate ID       Enqueue to          Track progress     Update status
  Store metadata    Redis queue         Handle errors      Generate artifacts
```

### API Endpoint Specifications

#### 1. Upload Batch Job

**Endpoint**: `POST /api/v1/batch/upload`

**Authentication**: Bearer token required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "documents": [
    {
      "id": "doc-001",
      "title": "Plan Nacional de Desarrollo 2024",
      "content": "Full document text content...",
      "metadata": {
        "country": "Mexico",
        "year": 2024,
        "sector": "education"
      }
    },
    {
      "id": "doc-002",
      "title": "Estrategia de Infraestructura",
      "content": "Full document text content...",
      "metadata": {
        "country": "Colombia",
        "year": 2024,
        "sector": "infrastructure"
      }
    }
  ],
  "evaluation_config": {
    "enable_decalogo": true,
    "enable_questionnaire": true,
    "enable_teoria_cambio": false,
    "detect_responsibilities": true,
    "detect_monetary_values": true,
    "detect_contradictions": false
  },
  "priority": "normal",
  "callback_url": "https://example.com/webhook/batch-complete",
  "notification_email": "user@example.com"
}
```

**Request Validation Rules**:
- `documents`: Array of 1-100 documents (min: 1, max: 100)
- `documents[].id`: Required, unique within batch, max 100 chars
- `documents[].title`: Required, max 500 chars
- `documents[].content`: Required, min 100 chars, max 1MB per document
- `documents[].metadata`: Optional object
- `evaluation_config`: Optional, defaults shown above
- `priority`: Optional, one of ["high", "normal", "low"], default "normal"
- `callback_url`: Optional, valid HTTPS URL
- `notification_email`: Optional, valid email address

**Response** (201 Created):
```json
{
  "job_id": "batch-20240115-abc123def456",
  "status": "queued",
  "document_count": 2,
  "estimated_completion_time": "2024-01-15T11:45:00Z",
  "created_at": "2024-01-15T11:30:00Z",
  "priority": "normal",
  "_links": {
    "self": "/api/v1/batch/upload",
    "status": "/api/v1/batch/status/batch-20240115-abc123def456",
    "results": "/api/v1/batch/results/batch-20240115-abc123def456"
  }
}
```

**Error Response** (400 Bad Request):
```json
{
  "error": "validation_error",
  "message": "Invalid request payload",
  "details": [
    {
      "field": "documents",
      "message": "Batch size exceeds maximum of 100 documents"
    }
  ]
}
```

**Error Response** (401 Unauthorized):
```json
{
  "error": "authentication_error",
  "message": "Invalid or missing authentication token"
}
```

**Error Response** (429 Too Many Requests):
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit of 100 requests per minute exceeded",
  "retry_after_seconds": 45
}
```

#### 2. Check Job Status

**Endpoint**: `GET /api/v1/batch/status/{job_id}`

**Authentication**: Bearer token required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK) - Queued:
```json
{
  "job_id": "batch-20240115-abc123def456",
  "status": "queued",
  "document_count": 2,
  "documents_processed": 0,
  "documents_successful": 0,
  "documents_failed": 0,
  "created_at": "2024-01-15T11:30:00Z",
  "started_at": null,
  "completed_at": null,
  "estimated_completion_time": "2024-01-15T11:45:00Z",
  "priority": "normal",
  "progress_percent": 0.0,
  "_links": {
    "self": "/api/v1/batch/status/batch-20240115-abc123def456",
    "results": "/api/v1/batch/results/batch-20240115-abc123def456"
  }
}
```

**Response** (200 OK) - Processing:
```json
{
  "job_id": "batch-20240115-abc123def456",
  "status": "processing",
  "document_count": 2,
  "documents_processed": 1,
  "documents_successful": 1,
  "documents_failed": 0,
  "created_at": "2024-01-15T11:30:00Z",
  "started_at": "2024-01-15T11:32:00Z",
  "completed_at": null,
  "estimated_completion_time": "2024-01-15T11:45:00Z",
  "priority": "normal",
  "progress_percent": 50.0,
  "current_document": {
    "id": "doc-002",
    "title": "Estrategia de Infraestructura",
    "started_at": "2024-01-15T11:40:00Z"
  },
  "_links": {
    "self": "/api/v1/batch/status/batch-20240115-abc123def456",
    "results": "/api/v1/batch/results/batch-20240115-abc123def456"
  }
}
```

**Response** (200 OK) - Completed:
```json
{
  "job_id": "batch-20240115-abc123def456",
  "status": "completed",
  "document_count": 2,
  "documents_processed": 2,
  "documents_successful": 2,
  "documents_failed": 0,
  "created_at": "2024-01-15T11:30:00Z",
  "started_at": "2024-01-15T11:32:00Z",
  "completed_at": "2024-01-15T11:44:30Z",
  "priority": "normal",
  "progress_percent": 100.0,
  "processing_time_seconds": 750,
  "average_time_per_document_seconds": 375,
  "_links": {
    "self": "/api/v1/batch/status/batch-20240115-abc123def456",
    "results": "/api/v1/batch/results/batch-20240115-abc123def456"
  }
}
```

**Response** (200 OK) - Failed:
```json
{
  "job_id": "batch-20240115-abc123def456",
  "status": "failed",
  "document_count": 2,
  "documents_processed": 2,
  "documents_successful": 1,
  "documents_failed": 1,
  "created_at": "2024-01-15T11:30:00Z",
  "started_at": "2024-01-15T11:32:00Z",
  "completed_at": "2024-01-15T11:44:30Z",
  "priority": "normal",
  "progress_percent": 100.0,
  "error_summary": {
    "total_errors": 1,
    "error_types": {
      "timeout": 1
    }
  },
  "_links": {
    "self": "/api/v1/batch/status/batch-20240115-abc123def456",
    "results": "/api/v1/batch/results/batch-20240115-abc123def456"
  }
}
```

**Error Response** (404 Not Found):
```json
{
  "error": "job_not_found",
  "message": "Job with ID 'batch-20240115-abc123def456' not found or expired"
}
```

#### 3. Retrieve Results

**Endpoint**: `GET /api/v1/batch/results/{job_id}`

**Authentication**: Bearer token required

**Query Parameters**:
- `format`: Optional, one of ["json", "jsonl", "csv"], default "json"
- `include_metadata`: Optional, boolean, default true
- `compression`: Optional, one of ["none", "gzip"], default "none"

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK) - JSON Format:
```json
{
  "job_id": "batch-20240115-abc123def456",
  "status": "completed",
  "completed_at": "2024-01-15T11:44:30Z",
  "document_count": 2,
  "documents_successful": 2,
  "documents_failed": 0,
  "results": [
    {
      "document_id": "doc-001",
      "document_title": "Plan Nacional de Desarrollo 2024",
      "status": "success",
      "processing_time_seconds": 345,
      "evaluation": {
        "decalogo_score": 8.5,
        "decalogo_items": [
          {
            "criterion": "Objetivos SMART",
            "score": 9.0,
            "rationale": "Objetivos claramente definidos con métricas específicas"
          }
        ],
        "questionnaire_score": 7.8,
        "detected_entities": {
          "responsibilities": [
            {
              "entity": "Secretaría de Educación Pública",
              "type": "institution",
              "confidence": 0.95
            }
          ],
          "monetary_values": [
            {
              "value": 50000000000,
              "currency": "MXN",
              "context": "presupuesto anual"
            }
          ]
        },
        "contract_validation": {
          "valid": true,
          "violations": []
        }
      },
      "metadata": {
        "country": "Mexico",
        "year": 2024,
        "sector": "education"
      }
    },
    {
      "document_id": "doc-002",
      "document_title": "Estrategia de Infraestructura",
      "status": "success",
      "processing_time_seconds": 405,
      "evaluation": {
        "decalogo_score": 7.2,
        "decalogo_items": [...],
        "questionnaire_score": 6.9,
        "detected_entities": {...},
        "contract_validation": {
          "valid": true,
          "violations": []
        }
      },
      "metadata": {
        "country": "Colombia",
        "year": 2024,
        "sector": "infrastructure"
      }
    }
  ],
  "_links": {
    "self": "/api/v1/batch/results/batch-20240115-abc123def456",
    "status": "/api/v1/batch/status/batch-20240115-abc123def456",
    "download_csv": "/api/v1/batch/results/batch-20240115-abc123def456?format=csv",
    "download_jsonl": "/api/v1/batch/results/batch-20240115-abc123def456?format=jsonl"
  }
}
```

**Response** (200 OK) - JSONL Format:
```
{"document_id":"doc-001","document_title":"Plan Nacional de Desarrollo 2024","status":"success",...}
{"document_id":"doc-002","document_title":"Estrategia de Infraestructura","status":"success",...}
```

**Response** (200 OK) - CSV Format:
```csv
document_id,document_title,status,processing_time_seconds,decalogo_score,questionnaire_score
doc-001,Plan Nacional de Desarrollo 2024,success,345,8.5,7.8
doc-002,Estrategia de Infraestructura,success,405,7.2,6.9
```

**Error Response** (202 Accepted) - Job Not Complete:
```json
{
  "error": "job_incomplete",
  "message": "Job is still processing",
  "status": "processing",
  "progress_percent": 50.0,
  "retry_after_seconds": 60,
  "_links": {
    "status": "/api/v1/batch/status/batch-20240115-abc123def456"
  }
}
```

#### 4. Health Check

**Endpoint**: `GET /api/v1/health`

**Authentication**: None required

**Response** (200 OK) - Healthy:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T11:30:00Z",
  "version": "1.0.0",
  "components": {
    "api": {
      "status": "healthy",
      "uptime_seconds": 86400
    },
    "redis": {
      "status": "healthy",
      "connection": "ok",
      "queue_depth": 15,
      "memory_usage_mb": 128
    },
    "celery_workers": {
      "status": "healthy",
      "active_workers": 4,
      "busy_workers": 2,
      "available_capacity": 14
    },
    "artifact_storage": {
      "status": "healthy",
      "storage_used_gb": 15.2,
      "storage_available_gb": 84.8
    }
  },
  "metrics": {
    "jobs_queued": 15,
    "jobs_processing": 3,
    "jobs_completed_last_hour": 42,
    "average_processing_time_seconds": 380
  }
}
```

**Response** (503 Service Unavailable) - Unhealthy:
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T11:30:00Z",
  "version": "1.0.0",
  "components": {
    "api": {
      "status": "healthy"
    },
    "redis": {
      "status": "unhealthy",
      "error": "Connection refused"
    },
    "celery_workers": {
      "status": "degraded",
      "active_workers": 1,
      "expected_workers": 4,
      "warning": "Worker capacity below threshold"
    }
  }
}
```

### Performance Targets

**Throughput**: 170 documents per hour (sustained)
- With 4 workers × 4 processes = 16 concurrent tasks
- Average processing time: 6-7 minutes per document
- Queue processing rate: ~3 documents/minute

**Latency**:
- API response time: <100ms (upload, status endpoints)
- Job queuing time: <1 second
- First document processing start: <30 seconds after upload

**Availability**:
- API uptime: 99.9%
- Worker availability: 99.5%
- Redis availability: 99.95%

**Resource Allocation**:
- FastAPI: 2 vCPU, 4GB RAM per instance (2 instances minimum)
- Redis: 4 vCPU, 8GB RAM (primary + replica)
- Celery Workers: 4 vCPU, 8GB RAM per worker (4 workers minimum)
- Total: 32 vCPU, 72GB RAM for baseline deployment

## Batch Processing Architecture

High-throughput asynchronous batch processing system for evaluating large document collections.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Client Applications                           │
│  (Python scripts, Shell scripts, Postman, Web UI)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP/HTTPS
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Nginx Reverse Proxy                            │
│  • SSL termination                                                    │
│  • Rate limiting (100 req/min per IP)                                │
│  • Load balancing across FastAPI instances                           │
│  • Request buffering                                                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server(s)                              │
│  • POST /api/v1/batch/upload       - Upload documents                │
│  • GET  /api/v1/batch/status/{id}  - Check job status                │
│  • GET  /api/v1/batch/results/{id} - Retrieve results                │
│  • GET  /api/v1/health             - Health check                    │
│  • Token-based authentication (Bearer tokens)                        │
│  • Request validation (1-100 documents per batch)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Enqueue tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Redis Message Broker                             │
│  • Task queue: decalogo_batch_tasks                                  │
│  • Result backend: decalogo_batch_results                            │
│  • Priority queuing (high/normal/low)                                │
│  • TTL: 7 days for tasks, 30 days for results                       │
│  • Persistence: AOF + RDB snapshots                                  │
└────────────┬──────────────────┬──────────────────┬───────────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Celery Worker 1  │ │ Celery Worker 2  │ │ Celery Worker N  │
│ • Concurrency: 4 │ │ • Concurrency: 4 │ │ • Concurrency: 4 │
│ • Prefetch: 2    │ │ • Prefetch: 2    │ │ • Prefetch: 2    │
│ • Max tasks: 100 │ │ • Max tasks: 100 │ │ • Max tasks: 100 │
│ • CPU: 4 cores   │ │ • CPU: 4 cores   │ │ • CPU: 4 cores   │
│ • RAM: 8GB       │ │ • RAM: 8GB       │ │ • RAM: 8GB       │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Artifact Storage                                 │
│  • S3-compatible object storage (MinIO/AWS S3)                       │
│  • Results: JSON files with evaluation data                          │
│  • Artifacts: Extracted evidence, embeddings, graphs                 │
│  • Retention: 90 days                                                │
│  • Access: Presigned URLs (24-hour expiry)                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Message Flow Diagram

```
1. Client Upload
   ├─► POST /api/v1/batch/upload
   │   └─► {"documents": [...], "priority": "normal"}
   │
2. FastAPI Validation & Enqueue
   ├─► Validate authentication token
   ├─► Validate request schema
   ├─► Generate batch_id (UUID)
   ├─► Store metadata in Redis
   ├─► Enqueue tasks to Celery
   └─► Return: {"batch_id": "...", "status": "queued"}
   │
3. Redis Queue
   ├─► Task stored in queue: decalogo_batch_tasks
   ├─► Metadata stored: batch:{batch_id}:metadata
   └─► Status tracked: batch:{batch_id}:status
   │
4. Celery Worker Processing
   ├─► Worker picks up task from queue
   ├─► Initialize evaluation pipeline
   ├─► Process documents (with progress updates)
   │   └─► Update Redis: batch:{batch_id}:progress
   ├─► Generate results and artifacts
   ├─► Upload to S3 storage
   └─► Update status to "completed"
   │
5. Client Status Check
   ├─► GET /api/v1/batch/status/{batch_id}
   └─► Return: {"status": "completed", "progress": "100/100"}
   │
6. Client Results Retrieval
   ├─► GET /api/v1/batch/results/{batch_id}
   └─► Return: {"results_url": "s3://...", "artifacts": [...]}
```

### Worker Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Celery Worker                                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Task Initialization │
                    │  • Load models        │
                    │  • Setup tracing      │
                    │  • Initialize storage │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Document Segmentation │
                    │  • Text normalization │
                    │  • Sentence splitting │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Embedding Generation │
                    │  • MPNet/MiniLM       │
                    │  • Batch processing   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Evidence Extraction  │
                    │  • Causal patterns    │
                    │  • Monetary amounts   │
                    │  • Responsibilities   │
                    │  • Feasibility        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Decálogo Evaluation   │
                    │  • 10 principles      │
                    │  • Evidence scoring   │
                    │  • Questionnaire      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Results Assembly    │
                    │  • JSON generation    │
                    │  • Artifact bundling  │
                    │  • S3 upload          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Status Update       │
                    │  • Redis update       │
                    │  • Callback (if set)  │
                    └───────────────────────┘
```

### API Endpoint Specifications

#### 1. Upload Documents for Batch Processing

**Endpoint**: `POST /api/v1/batch/upload`

**Authentication**: Bearer token required

**Request Headers**:
```http
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body Schema**:
```json
{
  "documents": [
    {
      "id": "doc-001",
      "text": "Plan text content...",
      "metadata": {
        "filename": "plan_nacional_2024.pdf",
        "author": "Ministry of Planning",
        "date": "2024-01-15"
      }
    }
  ],
  "options": {
    "priority": "normal",
    "callback_url": "https://example.com/webhook",
    "include_artifacts": true,
    "enable_tracing": true
  }
}
```

**Request Validation**:
- `documents`: Array, required, 1-100 items
- `documents[].id`: String, required, max 255 chars
- `documents[].text`: String, required, 100-1,000,000 chars
- `documents[].metadata`: Object, optional
- `options.priority`: Enum ["high", "normal", "low"], default "normal"
- `options.callback_url`: URL, optional, must be HTTPS
- `options.include_artifacts`: Boolean, default true
- `options.enable_tracing`: Boolean, default true

**Response (202 Accepted)**:
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "document_count": 25,
  "estimated_completion_time": "2024-01-15T14:30:00Z",
  "created_at": "2024-01-15T14:00:00Z"
}
```

**Error Responses**:

```json
// 401 Unauthorized
{
  "error": "invalid_token",
  "message": "Authentication token is invalid or expired"
}

// 400 Bad Request
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": [
    {
      "field": "documents",
      "error": "Array must contain 1-100 items"
    }
  ]
}

// 429 Too Many Requests
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded: 100 requests per minute",
  "retry_after": 45
}

// 503 Service Unavailable
{
  "error": "service_unavailable",
  "message": "Service temporarily unavailable, please retry",
  "retry_after": 300
}
```

#### 2. Check Batch Job Status

**Endpoint**: `GET /api/v1/batch/status/{batch_id}`

**Authentication**: Bearer token required

**Request Headers**:
```http
Authorization: Bearer <token>
```

**Path Parameters**:
- `batch_id`: UUID of the batch job

**Response (200 OK)**:
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "document_count": 25,
  "progress": {
    "completed": 15,
    "failed": 1,
    "pending": 9,
    "percent": 60.0
  },
  "created_at": "2024-01-15T14:00:00Z",
  "started_at": "2024-01-15T14:01:30Z",
  "estimated_completion": "2024-01-15T14:30:00Z",
  "worker_id": "celery-worker-2",
  "current_document": "doc-015"
}
```

**Status Values**:
- `queued`: Job is waiting in queue
- `processing`: Job is being processed
- `completed`: All documents processed successfully
- `partial_success`: Some documents failed
- `failed`: Job failed completely
- `cancelled`: Job was cancelled by user

**Error Responses**:

```json
// 404 Not Found
{
  "error": "batch_not_found",
  "message": "Batch job with ID 550e8400-e29b-41d4-a716-446655440000 not found"
}

// 403 Forbidden
{
  "error": "access_denied",
  "message": "You do not have permission to access this batch job"
}
```

#### 3. Retrieve Batch Results

**Endpoint**: `GET /api/v1/batch/results/{batch_id}`

**Authentication**: Bearer token required

**Request Headers**:
```http
Authorization: Bearer <token>
```

**Query Parameters**:
- `format`: Response format (`json` or `presigned_url`), default `json`
- `include_artifacts`: Include artifact URLs (boolean), default `true`

**Response (200 OK) - JSON Format**:
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "completed_at": "2024-01-15T14:25:00Z",
  "processing_time_seconds": 1410,
  "results": [
    {
      "document_id": "doc-001",
      "status": "success",
      "evaluation": {
        "overall_score": 8.5,
        "principle_scores": {
          "principle_1_universality": 9.0,
          "principle_2_sustainability": 8.5,
          "principle_3_intersectoriality": 8.0
        },
        "evidence_count": 47,
        "key_findings": [
          "Strong causal evidence for poverty reduction",
          "Clear monetary commitments identified"
        ]
      },
      "artifacts": {
        "detailed_report": "https://storage.example.com/batch-550e8400/doc-001/report.json?expires=86400",
        "evidence_graph": "https://storage.example.com/batch-550e8400/doc-001/graph.json?expires=86400",
        "embeddings": "https://storage.example.com/batch-550e8400/doc-001/embeddings.npy?expires=86400"
      }
    }
  ],
  "summary": {
    "total_documents": 25,
    "successful": 24,
    "failed": 1,
    "average_score": 7.8,
    "total_evidence_extracted": 1152
  },
  "artifacts": {
    "batch_report": "https://storage.example.com/batch-550e8400/summary.json?expires=86400",
    "aggregate_metrics": "https://storage.example.com/batch-550e8400/metrics.json?expires=86400"
  }
}
```

**Response (200 OK) - Presigned URL Format**:
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "results_url": "https://storage.example.com/batch-550e8400/results.json?expires=86400",
  "expires_at": "2024-01-16T14:25:00Z"
}
```

**Error Responses**:

```json
// 404 Not Found
{
  "error": "results_not_found",
  "message": "Results for batch 550e8400-e29b-41d4-a716-446655440000 not found or expired"
}

// 202 Accepted (Job still processing)
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Results not yet available, job is still processing",
  "progress": 60.0,
  "retry_after": 60
}
```

#### 4. Health Check

**Endpoint**: `GET /api/v1/health`

**Authentication**: Not required

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00Z",
  "version": "1.0.0",
  "components": {
    "api": {
      "status": "healthy",
      "latency_ms": 5
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 2,
      "queue_size": 15
    },
    "workers": {
      "status": "healthy",
      "active_workers": 5,
      "total_capacity": 20,
      "utilization_percent": 75.0
    },
    "storage": {
      "status": "healthy",
      "latency_ms": 12,
      "available_space_gb": 500
    }
  },
  "metrics": {
    "requests_per_minute": 45,
    "average_processing_time_seconds": 58,
    "queue_wait_time_seconds": 15
  }
}
```

**Response (503 Service Unavailable)**:
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T14:30:00Z",
  "components": {
    "redis": {
      "status": "unhealthy",
      "error": "Connection timeout"
    }
  }
}
```

### Performance Specifications

**Throughput Target**: 170 documents per hour

**Recommended Configuration**:
- **Workers**: 5 Celery workers
- **Concurrency per worker**: 4 (20 total concurrent tasks)
- **Processing time per document**: ~3 minutes average
- **Queue capacity**: 1000 pending tasks
- **Max batch size**: 100 documents

**Calculation**:
```
20 concurrent tasks × 60 minutes/hour ÷ 3 minutes/task = 400 documents/hour
(2.4× target, provides safety margin for complex documents)
```

**Resource Requirements per Worker**:
- **CPU**: 4 cores
- **RAM**: 8GB (model loading + processing buffers)
- **Disk**: 20GB (temporary files + models)
- **Network**: 100 Mbps

**Redis Configuration**:
- **Memory**: 4GB
- **Max connections**: 100
- **Persistence**: AOF every second + RDB every 5 minutes

**Storage Requirements**:
- **Results**: ~5MB per document
- **Artifacts**: ~10MB per document (embeddings, graphs)
- **Retention**: 90 days
- **Total**: ~25TB per year at 170 docs/hour × 24/7

See `BATCH_PROCESSING_GUIDE.md` for detailed setup and deployment instructions.

## Batch Processing Architecture

High-throughput asynchronous batch processing system for evaluating large document collections with a target throughput of **170 documents per hour**.

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Client Applications                           │
│  (Python scripts, Shell scripts, Postman, Web UI)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP/HTTPS (REST API)
                                │ Authentication: Bearer Token
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Nginx Reverse Proxy                            │
│  • SSL termination (HTTPS → HTTP)                                    │
│  • Rate limiting (100 req/min per IP)                                │
│  • Load balancing across FastAPI instances                           │
│  • Request buffering (max 100MB)                                     │
│  • Timeout: 60s upload, 30s status, 300s results                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FastAPI Application Server(s)                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ API Endpoints                                                   │ │
│  │  • POST /api/v1/batch/upload       - Upload documents          │ │
│  │  • GET  /api/v1/batch/status/{id}  - Check job status         │ │
│  │  • GET  /api/v1/batch/results/{id} - Retrieve results         │ │
│  │  • GET  /api/v1/health             - Health check              │ │
│  │  • POST /api/v1/batch/cancel/{id}  - Cancel running job       │ │
│  │  • GET  /api/v1/metrics            - Prometheus metrics        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Security & Validation Layer                                     │ │
│  │  • Bearer token authentication (JWT)                           │ │
│  │  • Rate limiting (100 requests/min per user)                   │ │
│  │  • Request validation (1-100 documents per batch)              │ │
│  │  • File size limits (10MB per doc, 100MB per batch)            │ │
│  │  • Content-Type validation (application/json, multipart)       │ │
│  └────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Redis Message Broker                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Celery Task Queues                                              │ │
│  │  • celery:high_priority   - Urgent jobs (< 10 docs)            │ │
│  │  • celery:default         - Standard jobs (10-50 docs)         │ │
│  │  • celery:batch           - Large batches (> 50 docs)          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Task State Storage                                              │ │
│  │  • Job metadata (status, progress, timestamps)                 │ │
│  │  • Result references (artifact paths)                          │ │
│  │  • Task expiry: 24 hours                                       │ │
│  │  • Persistence: RDB snapshots + AOF                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Celery Workers (N instances)                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Worker Configuration                                            │ │
│  │  • Concurrency: 4 processes per worker (CPU-bound)             │ │
│  │  • Max tasks per child: 100 (memory leak prevention)           │ │
│  │  • Task timeout: 600s (10 minutes)                             │ │
│  │  • Prefetch multiplier: 1 (fair distribution)                  │ │
│  │  • Autoscale: 2-4 processes per worker                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Task Processing Pipeline                                        │ │
│  │  1. Retrieve document from artifact storage                    │ │
│  │  2. Run DECALOGO evaluation with tracing                       │ │
│  │  3. Store results in artifact storage                          │ │
│  │  4. Update job status in Redis                                 │ │
│  │  5. Record metrics (latency, success, contract)                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Artifact Storage System                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Storage Backend (Filesystem or S3-compatible)                   │ │
│  │  • Input documents: /data/artifacts/inputs/{job_id}/           │ │
│  │  • Output results: /data/artifacts/outputs/{job_id}/           │ │
│  │  • Retention: 7 days (configurable)                            │ │
│  │  • Compression: gzip for JSON results                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Access Patterns                                                 │ │
│  │  • Write: FastAPI uploads, Worker results                      │ │
│  │  • Read: Workers (input), FastAPI (results download)           │ │
│  │  • Cleanup: Daily cron job (artifacts older than 7 days)       │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

                                ┌────────────────────┐
                                │  Monitoring Stack  │
                                ├────────────────────┤
                                │ • SLO Monitor      │
                                │ • OpenTelemetry    │
                                │ • Prometheus       │
                                │ • Grafana          │
                                └────────────────────┘
```

### Data Flow Diagram

```
┌───────────┐                    ┌───────────┐
│  Client   │                    │  FastAPI  │
│  Upload   │                    │  Server   │
└─────┬─────┘                    └─────┬─────┘
      │                                │
      │ 1. POST /batch/upload          │
      │ (JSON with doc texts)          │
      ├───────────────────────────────►│
      │                                │
      │                                │ 2. Validate & Auth
      │                                │    Generate job_id
      │                                ▼
      │                          ┌──────────┐
      │                          │  Artifact│
      │                          │  Storage │
      │                          └─────┬────┘
      │                                │
      │                                │ 3. Store input docs
      │                                │
      │                                ▼
      │                          ┌──────────┐
      │                          │  Redis   │
      │                          │  Queue   │
      │                          └─────┬────┘
      │                                │
      │ 4. Return job_id               │ 5. Enqueue tasks
      │    {"job_id": "uuid"}          │    (1 task per doc)
      │◄───────────────────────────────┤
      │                                ▼
      │                          ┌──────────┐
      │                          │  Celery  │
      │                          │  Worker  │
      │                          └─────┬────┘
      │                                │
      │ 6. GET /status/{job_id}        │ 7. Process tasks
      │    Poll every 5-10s            │    - Load doc
      │◄───────────────────────────────┤    - Evaluate
      │ {"status": "processing",       │    - Store result
      │  "progress": "12/50"}          │
      │                                ▼
      │                          ┌──────────┐
      │                          │  Artifact│
      │                          │  Storage │
      │                          └─────┬────┘
      │                                │
      │                                │ 8. Store results
      │                                │
      │ 9. GET /status/{job_id}        ▼
      │◄───────────────────────────────┤
      │ {"status": "completed"}        │
      │                                │
      │ 10. GET /results/{job_id}      │
      │◄───────────────────────────────┤
      │ {"results": [...]}             │
      │                                │
      ▼                                ▼
```

### Throughput Capacity

**Target**: 170 documents/hour

**Configuration for Target Throughput**:
- **4 Celery workers** with 4 processes each = **16 concurrent evaluations**
- **Processing time per document**: ~3.5 minutes average
- **Theoretical max**: (16 workers × 60 min) / 3.5 min = ~274 docs/hour
- **Real-world (75% efficiency)**: ~205 docs/hour
- **Headroom**: 20% above target for peak loads

### API Specifications

#### Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer <JWT_TOKEN>
```

JWT tokens contain:
```json
{
  "sub": "user_id",
  "username": "john.doe@example.com",
  "role": "evaluator",
  "exp": 1234567890,
  "rate_limit": 100
}
```

#### Endpoint 1: Upload Documents for Batch Processing

**Endpoint**: `POST /api/v1/batch/upload`

**Description**: Upload 1-100 documents for asynchronous evaluation.

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "documents": [
    {
      "document_id": "doc_001",
      "text": "Plan text content here...",
      "metadata": {
        "source": "municipal_plan",
        "year": 2024,
        "region": "Region_X"
      }
    },
    {
      "document_id": "doc_002",
      "text": "Another plan text...",
      "metadata": {}
    }
  ],
  "priority": "normal",
  "callback_url": "https://example.com/webhook",
  "options": {
    "enable_tracing": true,
    "strict_validation": true
  }
}
```

**Request Schema**:
```json
{
  "type": "object",
  "required": ["documents"],
  "properties": {
    "documents": {
      "type": "array",
      "minItems": 1,
      "maxItems": 100,
      "items": {
        "type": "object",
        "required": ["document_id", "text"],
        "properties": {
          "document_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 255
          },
          "text": {
            "type": "string",
            "minLength": 100,
            "maxLength": 10485760
          },
          "metadata": {
            "type": "object"
          }
        }
      }
    },
    "priority": {
      "type": "string",
      "enum": ["high", "normal", "low"],
      "default": "normal"
    },
    "callback_url": {
      "type": "string",
      "format": "uri"
    },
    "options": {
      "type": "object",
      "properties": {
        "enable_tracing": {"type": "boolean", "default": true},
        "strict_validation": {"type": "boolean", "default": true}
      }
    }
  }
}
```

**Response** (201 Created):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "document_count": 2,
  "estimated_completion": "2024-01-15T11:15:00Z",
  "created_at": "2024-01-15T11:00:00Z",
  "status_url": "/api/v1/batch/status/550e8400-e29b-41d4-a716-446655440000",
  "results_url": "/api/v1/batch/results/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses**:
- **400 Bad Request**: Invalid request body or validation failure
- **401 Unauthorized**: Missing or invalid authentication token
- **413 Payload Too Large**: Batch exceeds 100MB limit
- **429 Too Many Requests**: Rate limit exceeded (100 req/min)
- **503 Service Unavailable**: Queue is full (>10000 pending tasks)

**Example Error Response**:
```json
{
  "error": "validation_error",
  "message": "Document count exceeds maximum of 100",
  "details": {
    "provided": 150,
    "maximum": 100
  }
}
```

#### Endpoint 2: Check Job Status

**Endpoint**: `GET /api/v1/batch/status/{job_id}`

**Description**: Retrieve current status and progress of a batch job.

**Request Headers**:
```
Authorization: Bearer <token>
```

**Path Parameters**:
- `job_id` (string, UUID): Job identifier returned from upload

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": {
    "total": 50,
    "completed": 12,
    "failed": 1,
    "pending": 37,
    "percentage": 24.0
  },
  "created_at": "2024-01-15T11:00:00Z",
  "started_at": "2024-01-15T11:00:15Z",
  "updated_at": "2024-01-15T11:05:30Z",
  "estimated_completion": "2024-01-15T14:30:00Z",
  "error_summary": {
    "total_errors": 1,
    "error_types": {
      "validation_error": 1
    }
  }
}
```

**Status Values**:
- `queued`: Job accepted, tasks enqueued
- `processing`: Workers actively processing documents
- `completed`: All documents processed successfully
- `completed_with_errors`: Some documents failed, partial results available
- `failed`: Job failed critically (cannot continue)
- `cancelled`: Job cancelled by user

**Error Responses**:
- **401 Unauthorized**: Missing or invalid authentication token
- **404 Not Found**: Job ID does not exist or expired
- **403 Forbidden**: User does not own this job

#### Endpoint 3: Retrieve Results

**Endpoint**: `GET /api/v1/batch/results/{job_id}`

**Description**: Download evaluation results for a completed job.

**Request Headers**:
```
Authorization: Bearer <token>
Accept: application/json
```

**Query Parameters**:
- `format` (string, optional): Response format - `json` (default) or `jsonl`
- `include_failures` (boolean, optional): Include failed documents (default: `false`)
- `compress` (boolean, optional): Gzip compression (default: `true`)

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_documents": 50,
  "successful": 49,
  "failed": 1,
  "results": [
    {
      "document_id": "doc_001",
      "status": "success",
      "evaluation": {
        "decalogo_score": 8.5,
        "questionnaire_results": {...},
        "evidence": [...],
        "contract_valid": true
      },
      "processing_time_ms": 3420,
      "completed_at": "2024-01-15T11:03:30Z",
      "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736"
    },
    {
      "document_id": "doc_002",
      "status": "success",
      "evaluation": {...},
      "processing_time_ms": 3215,
      "completed_at": "2024-01-15T11:06:45Z",
      "trace_id": "5ac83f4688c45eb7b4df93ae1f1f5847"
    }
  ],
  "failures": [
    {
      "document_id": "doc_003",
      "status": "failed",
      "error": "validation_error",
      "error_message": "Document text too short (minimum 100 characters)",
      "failed_at": "2024-01-15T11:02:00Z"
    }
  ],
  "summary": {
    "avg_processing_time_ms": 3350,
    "p95_latency_ms": 3800,
    "total_processing_time_ms": 164150
  }
}
```

**Error Responses**:
- **401 Unauthorized**: Missing or invalid authentication token
- **404 Not Found**: Job ID does not exist, expired, or no results available
- **403 Forbidden**: User does not own this job
- **425 Too Early**: Job still processing (use status endpoint)

#### Endpoint 4: Health Check

**Endpoint**: `GET /api/v1/health`

**Description**: Check system health and component status.

**Request Headers**: None required (public endpoint)

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T11:00:00Z",
  "components": {
    "api_server": {
      "status": "healthy",
      "response_time_ms": 2
    },
    "redis": {
      "status": "healthy",
      "connection": "active",
      "memory_used_mb": 245,
      "queue_depth": {
        "high_priority": 5,
        "default": 23,
        "batch": 120
      }
    },
    "celery_workers": {
      "status": "healthy",
      "active_workers": 4,
      "total_capacity": 16,
      "busy_workers": 12,
      "available_workers": 4
    },
    "artifact_storage": {
      "status": "healthy",
      "available_space_gb": 450,
      "total_space_gb": 500
    }
  },
  "metrics": {
    "total_jobs_today": 45,
    "documents_processed_today": 1823,
    "current_throughput_docs_per_hour": 168,
    "avg_processing_time_ms": 3420
  }
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T11:00:00Z",
  "components": {
    "redis": {
      "status": "unhealthy",
      "error": "Connection refused"
    },
    "celery_workers": {
      "status": "degraded",
      "active_workers": 1,
      "expected_workers": 4
    }
  }
}
```

#### Endpoint 5: Cancel Job

**Endpoint**: `POST /api/v1/batch/cancel/{job_id}`

**Description**: Cancel a running or queued batch job.

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "cancelled_at": "2024-01-15T11:10:00Z",
  "progress": {
    "total": 50,
    "completed": 12,
    "cancelled": 38
  },
  "partial_results_available": true
}
```

**Error Responses**:
- **401 Unauthorized**: Missing or invalid authentication token
- **404 Not Found**: Job ID does not exist
- **403 Forbidden**: User does not own this job
- **409 Conflict**: Job already completed (cannot cancel)

#### Endpoint 6: Prometheus Metrics

**Endpoint**: `GET /api/v1/metrics`

**Description**: Export Prometheus metrics for monitoring.

**Request Headers**: None required

**Response** (200 OK, text/plain):
```
# HELP batch_jobs_total Total number of batch jobs submitted
# TYPE batch_jobs_total counter
batch_jobs_total{status="completed"} 145
batch_jobs_total{status="failed"} 3
batch_jobs_total{status="cancelled"} 2

# HELP batch_documents_processed_total Total documents processed
# TYPE batch_documents_processed_total counter
batch_documents_processed_total{status="success"} 6823
batch_documents_processed_total{status="failed"} 45

# HELP batch_processing_time_seconds Document processing time
# TYPE batch_processing_time_seconds histogram
batch_processing_time_seconds_bucket{le="1.0"} 234
batch_processing_time_seconds_bucket{le="2.0"} 1456
batch_processing_time_seconds_bucket{le="5.0"} 5234
batch_processing_time_seconds_bucket{le="+Inf"} 6823
batch_processing_time_seconds_sum 23456.78
batch_processing_time_seconds_count 6823

# HELP batch_queue_depth Current queue depth by priority
# TYPE batch_queue_depth gauge
batch_queue_depth{priority="high"} 5
batch_queue_depth{priority="normal"} 23
batch_queue_depth{priority="low"} 120

# HELP batch_worker_capacity Worker capacity metrics
# TYPE batch_worker_capacity gauge
batch_worker_capacity{state="busy"} 12
batch_worker_capacity{state="idle"} 4
```

### Rate Limiting and Authentication

#### Rate Limits

**Per-IP Rate Limits** (enforced by Nginx):
- 100 requests/minute for upload endpoint
- 300 requests/minute for status/results endpoints
- 1000 requests/minute for health/metrics endpoints

**Per-User Rate Limits** (enforced by FastAPI):
- 100 batch uploads per day
- 10,000 API requests per day
- Configurable per user role in JWT

**Rate Limit Headers** (included in all responses):
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1610723400
Retry-After: 45
```

#### JWT Token Generation

**Token Issuer**: Authentication service (not part of batch system)

**Token Validation**:
- Signature verification using RS256 algorithm
- Expiration check (typical: 1 hour lifetime)
- Role-based access control (RBAC)
- Token revocation list (Redis-backed)

**Example Token Payload**:
```json
{
  "sub": "user_123456",
  "username": "evaluator@example.com",
  "role": "evaluator",
  "permissions": ["batch:upload", "batch:read"],
  "rate_limits": {
    "uploads_per_day": 100,
    "api_requests_per_day": 10000
  },
  "iat": 1610720000,
  "exp": 1610723600,
  "iss": "auth.decalogo.system",
  "aud": "batch.api.decalogo.system"
}
```

### Security Considerations

1. **Input Validation**: All document text sanitized to prevent injection attacks
2. **Size Limits**: Enforced at Nginx (100MB) and application (10MB per doc) layers
3. **Rate Limiting**: Multi-layer protection (IP, user, global)
4. **Token Security**: Short-lived JWT tokens, secure signing keys rotation
5. **Artifact Isolation**: Each job's artifacts stored in isolated directories
6. **Access Control**: Users can only access their own jobs
7. **Audit Logging**: All API requests logged with user ID and trace ID
8. **DDoS Protection**: Nginx connection limits and request buffering

### Monitoring and Alerting Integration

The batch processing system integrates with the existing deployment infrastructure:

1. **OpenTelemetry Tracing**: Each document evaluation traced end-to-end
2. **SLO Monitoring**: Batch processing metrics feed into SLO dashboard
3. **Prometheus**: Metrics exported for Grafana visualization
4. **Alert Rules**:
   - Queue depth > 5000 tasks (WARNING)
   - Worker count < 2 (CRITICAL)
   - Processing time p95 > 5 minutes (WARNING)
   - Error rate > 5% (CRITICAL)
   - Throughput < 100 docs/hour (WARNING)

See **BATCH_PROCESSING_GUIDE.md** for detailed setup, deployment, and troubleshooting instructions.──────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Enqueue Celery tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Redis Message Broker                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Task Queues                                                     │ │
│  │  • decalogo_high_priority   - Critical evaluations             │ │
│  │  • decalogo_normal          - Standard batch processing        │ │
│  │  • decalogo_low_priority    - Background re-processing         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Result Backend                                                  │ │
│  │  • Job status tracking (PENDING/RUNNING/SUCCESS/FAILURE)       │ │
│  │  • Progress updates (documents processed)                      │ │
│  │  • Result metadata (evaluation scores, errors)                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  • TTL: 7 days for tasks, 30 days for results                       │
│  • Persistence: AOF (append-only file) + RDB snapshots every 5 min  │
│  • Memory: 2GB recommended for 10k concurrent tasks                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Dequeue and execute
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Celery Worker Pool                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Worker Node 1   │  │  Worker Node 2   │  │  Worker Node N   │  │
│  │  • Concurrency:4 │  │  • Concurrency:4 │  │  • Concurrency:4 │  │
│  │  • CPU: 4 cores  │  │  • CPU: 4 cores  │  │  • CPU: 4 cores  │  │
│  │  • RAM: 8GB      │  │  • RAM: 8GB      │  │  • RAM: 8GB      │  │
│  │  • Task timeout: │  │  • Task timeout: │  │  • Task timeout: │  │
│  │    10 min/doc    │  │    10 min/doc    │  │    10 min/doc    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                       │
│  Processing Pipeline per Document:                                   │
│  1. Load embedding model                                             │
│  2. Segment and normalize text                                       │
│  3. Extract evidence (causal, monetary, responsibility)              │
│  4. Evaluate against DECALOGO and questionnaire                      │
│  5. Generate evaluation report                                       │
│  6. Store artifacts                                                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Store results
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Artifact Storage Layer                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ File System Storage (NFS/Shared Volume)                         │ │
│  │  /artifacts/                                                    │ │
│  │    ├── batch_{job_id}/                                          │ │
│  │    │   ├── documents/          - Original uploaded files       │ │
│  │    │   ├── evaluations/        - JSON evaluation results       │ │
│  │    │   ├── reports/            - Formatted HTML/PDF reports    │ │
│  │    │   └── metadata.json       - Job metadata and metrics      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Optional: S3-Compatible Object Storage                          │ │
│  │  • MinIO/AWS S3/Google Cloud Storage                            │ │
│  │  • Versioned artifacts                                          │ │
│  │  • Lifecycle policies (archive after 90 days)                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Upload Phase:
  Client → Nginx → FastAPI → Redis (enqueue task) → Return job_id

Processing Phase:
  Redis → Celery Worker → Load Model → Process Document → Store Results
                      ↓
                  Update Status (Redis)
                      ↓
              Progress: 1/10, 2/10, ... 10/10

Retrieval Phase:
  Client → Nginx → FastAPI → Redis (fetch status/results) → Return data
                           ↓
                   Artifact Storage (if completed)
```

### API Specifications

#### 1. Upload Documents (POST /api/v1/batch/upload)

**Authentication**: Bearer token required

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Request Body** (multipart/form-data):
```
files: File[] (1-100 documents)
  - Supported formats: .txt, .pdf, .docx
  - Max size per file: 10MB
  - Max total size: 100MB

priority: string (optional)
  - Values: "high", "normal", "low"
  - Default: "normal"

metadata: JSON (optional)
  - Example: {"project_id": "abc123", "user_id": "user@example.com"}
```

**Response** (201 Created):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "created_at": "2024-01-15T10:30:00Z",
  "document_count": 10,
  "estimated_completion_time": "2024-01-15T10:45:00Z",
  "priority": "normal",
  "status_url": "/api/v1/batch/status/550e8400-e29b-41d4-a716-446655440000",
  "results_url": "/api/v1/batch/results/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses**:
```json
// 400 Bad Request - Invalid request
{
  "error": "INVALID_REQUEST",
  "message": "Document count must be between 1 and 100",
  "details": {
    "provided_count": 150,
    "max_allowed": 100
  }
}

// 401 Unauthorized - Missing/invalid token
{
  "error": "UNAUTHORIZED",
  "message": "Invalid or expired authentication token"
}

// 413 Payload Too Large - File size exceeded
{
  "error": "PAYLOAD_TOO_LARGE",
  "message": "Total upload size exceeds 100MB limit",
  "details": {
    "provided_size_mb": 125.5,
    "max_allowed_mb": 100
  }
}

// 429 Too Many Requests - Rate limit exceeded
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Try again in 60 seconds.",
  "retry_after": 60
}
```

#### 2. Check Job Status (GET /api/v1/batch/status/{job_id})

**Authentication**: Bearer token required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "RUNNING",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:15Z",
  "updated_at": "2024-01-15T10:32:00Z",
  "completed_at": null,
  "progress": {
    "total_documents": 10,
    "processed_documents": 3,
    "successful_documents": 3,
    "failed_documents": 0,
    "percentage": 30.0
  },
  "estimated_completion_time": "2024-01-15T10:45:00Z",
  "priority": "normal",
  "worker_id": "celery-worker-2@hostname",
  "current_stage": "evidence_extraction"
}

// Status values: PENDING, RUNNING, SUCCESS, FAILURE, CANCELLED
```

**Error Responses**:
```json
// 404 Not Found - Job doesn't exist
{
  "error": "JOB_NOT_FOUND",
  "message": "Job with ID '550e8400-...' not found or expired"
}
```

#### 3. Retrieve Results (GET /api/v1/batch/results/{job_id})

**Authentication**: Bearer token required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Query Parameters**:
```
format: string (optional)
  - Values: "json", "summary"
  - Default: "json"

include_artifacts: boolean (optional)
  - Include download URLs for reports/evaluations
  - Default: false
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "SUCCESS",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:15Z",
  "completed_at": "2024-01-15T10:44:30Z",
  "duration_seconds": 855.0,
  "results": [
    {
      "document_id": "doc_001",
      "filename": "plan_educacion.pdf",
      "status": "success",
      "evaluation": {
        "decalogo_score": 8.5,
        "questionnaire_score": 7.8,
        "overall_score": 8.15,
        "recommendations": [
          "Strengthen causal chain for education outcomes",
          "Add more specific monetary estimates"
        ]
      },
      "evidence_summary": {
        "causal_patterns": 15,
        "monetary_mentions": 8,
        "responsibilities": 12,
        "contradictions": 2
      },
      "processing_time_seconds": 82.5,
      "artifacts": {
        "evaluation_json": "/api/v1/artifacts/550e8400.../doc_001/evaluation.json",
        "report_html": "/api/v1/artifacts/550e8400.../doc_001/report.html"
      }
    },
    {
      "document_id": "doc_002",
      "filename": "corrupted_file.pdf",
      "status": "failure",
      "error": {
        "code": "PROCESSING_ERROR",
        "message": "Failed to extract text from PDF",
        "details": "PDF appears to be corrupted or encrypted"
      },
      "processing_time_seconds": 5.2
    }
  ],
  "summary": {
    "total_documents": 10,
    "successful": 9,
    "failed": 1,
    "average_score": 8.23,
    "average_processing_time_seconds": 78.5
  }
}
```

**Error Responses**:
```json
// 404 Not Found - Job doesn't exist
{
  "error": "JOB_NOT_FOUND",
  "message": "Job with ID '550e8400-...' not found or expired"
}

// 425 Too Early - Job not completed
{
  "error": "JOB_NOT_COMPLETED",
  "message": "Job is still processing. Check status endpoint for progress.",
  "status": "RUNNING",
  "progress_percentage": 30.0
}
```

#### 4. Cancel Job (POST /api/v1/batch/cancel/{job_id})

**Authentication**: Bearer token required

**Request Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "CANCELLED",
  "message": "Job cancellation requested. Processing will stop after current document.",
  "documents_processed_before_cancel": 3,
  "total_documents": 10
}
```

#### 5. Health Check (GET /api/v1/health)

**Authentication**: Not required

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api_server": {
      "status": "up",
      "uptime_seconds": 86400
    },
    "redis_broker": {
      "status": "up",
      "connection_pool_size": 50,
      "used_memory_mb": 512.5
    },
    "celery_workers": {
      "status": "up",
      "active_workers": 8,
      "available_workers": 8,
      "queued_tasks": 15,
      "processing_tasks": 24
    },
    "artifact_storage": {
      "status": "up",
      "available_space_gb": 458.3,
      "used_space_gb": 41.7
    }
  },
  "metrics": {
    "requests_per_minute": 45.2,
    "average_response_time_ms": 125.8,
    "jobs_completed_last_hour": 170,
    "error_rate_percent": 0.05
  }
}
```

### Authentication & Authorization

**Token-Based Authentication**:
```python
# Generate API token (admin only)
import secrets
token = secrets.token_urlsafe(32)
# Example: "xJk7Lm9Qp3Rz8Nv2Yw5Tx4Fb6Gh1Dc0"

# Store securely (environment variable or secrets manager)
export DECALOGO_API_TOKEN="xJk7Lm9Qp3Rz8Nv2Yw5Tx4Fb6Gh1Dc0"
```

**Using Authentication**:
```bash
# curl
curl -H "Authorization: Bearer xJk7Lm9Qp3Rz8Nv2Yw5Tx4Fb6Gh1Dc0" \
     https://api.decalogo.example.com/api/v1/health

# Python requests
import requests
headers = {"Authorization": "Bearer xJk7Lm9Qp3Rz8Nv2Yw5Tx4Fb6Gh1Dc0"}
response = requests.get("https://api.decalogo.example.com/api/v1/health", 
                       headers=headers)
```

### Rate Limiting

**Per-IP Rate Limits** (Nginx level):
- 100 requests per minute per IP address
- Burst: 20 requests
- Response: 429 Too Many Requests with Retry-After header

**Per-User Rate Limits** (Application level):
- 100 requests per minute per API token
- 10 concurrent batch jobs per user
- 1000 documents per hour per user

**Configuration**:
```nginx
# Nginx rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req zone=api_limit burst=20 nodelay;
```

### Performance Targets

**Throughput**: 170 documents per hour (per worker)
- Based on 21-second average processing time per document
- Scales linearly with worker count

**Target Configuration for 170 docs/hour**:
- **Single Worker**: 1 worker @ 4 concurrency = ~170 docs/hour
- **High Throughput**: 5 workers @ 4 concurrency = ~850 docs/hour
- **Production Scale**: 10 workers @ 4 concurrency = ~1700 docs/hour

**Latency SLOs**:
- API response time: <100ms (p95) for status/upload endpoints
- Job start latency: <5s (time from upload to processing start)
- Result retrieval: <200ms (p95)

**Resource Requirements** (per worker):
- CPU: 4 cores (3.5 cores utilized during processing)
- RAM: 8GB (6GB peak usage with embedding model loaded)
- Disk I/O: 50 MB/s sustained (document reading + artifact writing)
- Network: 10 Mbps (document upload + result retrieval)

For complete setup instructions, client examples, troubleshooting guides, and performance tuning recommendations, see **[BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md)**.

## Batch Processing Architecture

High-throughput asynchronous batch processing system for evaluating large document collections.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Client Applications                           │
│  (Python scripts, Shell scripts, Postman, Web UI)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP/HTTPS
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Nginx Reverse Proxy                            │
│  • SSL termination                                                    │
│  • Rate limiting (100 req/min per IP)                                │
│  • Load balancing across FastAPI instances                           │
│  • Request buffering                                                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server(s)                              │
│  • POST /api/v1/batch/upload       - Upload documents                │
│  • GET  /api/v1/batch/status/{id}  - Check job status                │
│  • GET  /api/v1/batch/results/{id} - Retrieve results                │
│  • GET  /api/v1/health             - Health check                    │
│  • Token-based authentication (Bearer tokens)                        │
│  • Request validation (1-100 documents per batch)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Enqueue tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Redis Message Broker                             │
│  • Task queue: decalogo_batch_tasks                                  │
│  • Result backend: decalogo_batch_results                            │
│  • Priority queuing (high/normal/low)                                │
│  • TTL: 7 days for tasks, 30 days for results                       │
│  • Persistence: AOF + RDB snapshots                                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Worker pool
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Celery Worker Pool                                │
│                                                                       │
│  Worker 1 (8 threads)    Worker 2 (8 threads)    Worker N (8 threads)│
│  ├─ Document Loader      ├─ Document Loader      ├─ Document Loader │
│  ├─ Segmentation         ├─ Segmentation         ├─ Segmentation    │
│  ├─ Embedding Model      ├─ Embedding Model      ├─ Embedding Model │
│  ├─ Evidence Extraction  ├─ Evidence Extraction  ├─ Evidence Extract.│
│  ├─ Evaluation           ├─ Evaluation           ├─ Evaluation      │
│  └─ Result Aggregation   └─ Result Aggregation   └─ Result Aggreg.  │
│                                                                       │
│  Configuration:                                                       │
│  • concurrency=8 (threads per worker)                                │
│  • prefetch_multiplier=4 (32 tasks prefetched per worker)           │
│  • max_tasks_per_child=100 (worker restart after 100 tasks)         │
│  • task_acks_late=True (acknowledge after completion)               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Store results
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Artifact Storage                                 │
│                                                                       │
│  artifacts/                                                           │
│  ├── batch_jobs/                                                      │
│  │   ├── <job_id>/                                                    │
│  │   │   ├── metadata.json       (job metadata, timestamps)          │
│  │   │   ├── documents/           (uploaded documents)               │
│  │   │   ├── results/             (evaluation results per doc)       │
│  │   │   └── summary.json         (aggregated results)               │
│  │   └── ...                                                          │
│  └── logs/                                                            │
│      └── batch_<job_id>.log       (detailed processing logs)         │
│                                                                       │
│  Storage policies:                                                    │
│  • Compression: gzip for results > 1MB                               │
│  • Retention: 90 days, then archive to cold storage                  │
│  • Backup: Daily incremental, weekly full                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Upload Request                Status Check              Result Retrieval
     │                             │                            │
     ▼                             ▼                            ▼
┌─────────┐                 ┌─────────┐                 ┌─────────┐
│ FastAPI │                 │ FastAPI │                 │ FastAPI │
│ Endpoint│                 │ Endpoint│                 │ Endpoint│
└────┬────┘                 └────┬────┘                 └────┬────┘
     │                           │                            │
     │ 1. Validate              │ 3. Query Redis            │ 5. Read from
     │    documents             │    for task status        │    artifact
     │                          │                           │    storage
     ▼                          ▼                           ▼
┌─────────┐                ┌─────────┐                ┌──────────┐
│  Redis  │                │  Redis  │                │ Artifact │
│  Queue  │                │ Backend │                │ Storage  │
└────┬────┘                └─────────┘                └──────────┘
     │                           ▲                            ▲
     │ 2. Enqueue                │ 4. Update                 │ 6. Return
     │    batch task             │    progress               │    results
     ▼                           │                           │
┌─────────┐                     │                           │
│ Celery  │─────────────────────┴───────────────────────────┘
│ Worker  │
└─────────┘
```

### API Endpoint Specifications

#### 1. Upload Documents (POST /api/v1/batch/upload)

**Request Headers**
```
Content-Type: application/json
Authorization: Bearer <token>
```

**Request Body**
```json
{
  "documents": [
    {
      "id": "doc_001",
      "content": "Plan text content...",
      "metadata": {
        "title": "Plan Nacional de Desarrollo",
        "year": 2024,
        "sector": "Social"
      }
    }
  ],
  "options": {
    "priority": "normal",
    "include_evidence": true,
    "include_traces": false,
    "webhook_url": "https://example.com/webhook"
  }
}
```

**Validation Rules**
- `documents`: Required, array of 1-100 documents
- `documents[].id`: Required, unique string (max 100 chars)
- `documents[].content`: Required, non-empty string (max 1MB)
- `documents[].metadata`: Optional, object with arbitrary key-value pairs
- `options.priority`: Optional, enum ["high", "normal", "low"], default "normal"
- `options.include_evidence`: Optional, boolean, default true
- `options.include_traces`: Optional, boolean, default false
- `options.webhook_url`: Optional, valid HTTPS URL

**Response (202 Accepted)**
```json
{
  "job_id": "batch_20240115_abc123",
  "status": "queued",
  "document_count": 15,
  "estimated_completion_seconds": 300,
  "created_at": "2024-01-15T10:30:00Z",
  "status_url": "/api/v1/batch/status/batch_20240115_abc123",
  "results_url": "/api/v1/batch/results/batch_20240115_abc123"
}
```

**Error Responses**
```json
// 400 Bad Request - Invalid input
{
  "error": "validation_error",
  "message": "Document count exceeds maximum of 100",
  "details": {
    "received": 150,
    "maximum": 100
  }
}

// 401 Unauthorized - Missing/invalid token
{
  "error": "unauthorized",
  "message": "Invalid or missing authentication token"
}

// 429 Too Many Requests - Rate limit exceeded
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit of 100 requests per minute exceeded",
  "retry_after_seconds": 45
}

// 503 Service Unavailable - Queue full
{
  "error": "service_unavailable",
  "message": "Queue capacity exceeded, please retry later",
  "queue_size": 10000,
  "queue_capacity": 10000
}
```

#### 2. Check Job Status (GET /api/v1/batch/status/{job_id})

**Request Headers**
```
Authorization: Bearer <token>
```

**Response (200 OK) - Queued**
```json
{
  "job_id": "batch_20240115_abc123",
  "status": "queued",
  "document_count": 15,
  "documents_completed": 0,
  "documents_failed": 0,
  "progress_percent": 0.0,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": null,
  "completed_at": null,
  "estimated_completion_seconds": 300
}
```

**Response (200 OK) - Processing**
```json
{
  "job_id": "batch_20240115_abc123",
  "status": "processing",
  "document_count": 15,
  "documents_completed": 8,
  "documents_failed": 1,
  "progress_percent": 53.3,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:31:00Z",
  "completed_at": null,
  "estimated_completion_seconds": 120,
  "current_worker": "worker-2@host-01"
}
```

**Response (200 OK) - Completed**
```json
{
  "job_id": "batch_20240115_abc123",
  "status": "completed",
  "document_count": 15,
  "documents_completed": 14,
  "documents_failed": 1,
  "progress_percent": 100.0,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:31:00Z",
  "completed_at": "2024-01-15T10:36:00Z",
  "total_processing_seconds": 300,
  "results_url": "/api/v1/batch/results/batch_20240115_abc123"
}
```

**Response (200 OK) - Failed**
```json
{
  "job_id": "batch_20240115_abc123",
  "status": "failed",
  "document_count": 15,
  "documents_completed": 5,
  "documents_failed": 10,
  "progress_percent": 33.3,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:31:00Z",
  "failed_at": "2024-01-15T10:33:00Z",
  "error": "Worker crashed during processing",
  "error_details": {
    "worker_id": "worker-3@host-02",
    "last_document": "doc_008",
    "exception": "OutOfMemoryError"
  }
}
```

**Error Responses**
```json
// 404 Not Found - Job doesn't exist
{
  "error": "job_not_found",
  "message": "No job found with ID: batch_20240115_xyz789"
}
```

#### 3. Retrieve Results (GET /api/v1/batch/results/{job_id})

**Request Headers**
```
Authorization: Bearer <token>
```

**Query Parameters**
- `format`: Optional, enum ["json", "csv"], default "json"
- `include_evidence`: Optional, boolean, default true
- `include_traces`: Optional, boolean, default false

**Response (200 OK)**
```json
{
  "job_id": "batch_20240115_abc123",
  "status": "completed",
  "document_count": 15,
  "documents_completed": 14,
  "documents_failed": 1,
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:36:00Z",
  "summary": {
    "average_score": 7.8,
    "median_score": 8.0,
    "total_evidence_extracted": 142,
    "total_contradictions": 3,
    "average_processing_time_seconds": 20.5
  },
  "results": [
    {
      "document_id": "doc_001",
      "status": "completed",
      "score": 8.5,
      "evaluation": {
        "D1_DIAGNOSTICO_SUFICIENTE": {
          "score": 9.0,
          "evidence_count": 5,
          "confidence": 0.92
        },
        "D2_CAUSALIDAD_FUNDAMENTADA": {
          "score": 8.0,
          "evidence_count": 3,
          "confidence": 0.88
        }
      },
      "evidence": [
        {
          "type": "causal_relationship",
          "text": "La reducción de la pobreza se logrará mediante...",
          "confidence": 0.95,
          "decalogo_alignment": ["D2_CAUSALIDAD_FUNDAMENTADA"]
        }
      ],
      "contradictions": [],
      "processing_time_seconds": 18.2
    },
    {
      "document_id": "doc_002",
      "status": "failed",
      "error": "Document parsing failed",
      "error_details": {
        "exception": "InvalidDocumentFormat",
        "message": "Unable to extract text from document"
      }
    }
  ]
}
```

**Response (200 OK) - CSV Format**
```csv
document_id,status,score,D1_score,D2_score,...,processing_time_seconds,error
doc_001,completed,8.5,9.0,8.0,...,18.2,
doc_002,failed,,,,...,,InvalidDocumentFormat
```

**Error Responses**
```json
// 404 Not Found
{
  "error": "job_not_found",
  "message": "No job found with ID: batch_20240115_xyz789"
}

// 202 Accepted - Job still processing
{
  "error": "job_not_ready",
  "message": "Job is still processing",
  "status": "processing",
  "progress_percent": 65.0,
  "estimated_completion_seconds": 90
}
```

#### 4. Health Check (GET /api/v1/health)

**Response (200 OK)**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api": {
      "status": "healthy",
      "uptime_seconds": 86400
    },
    "redis": {
      "status": "healthy",
      "queue_size": 245,
      "queue_capacity": 10000,
      "memory_usage_mb": 128,
      "memory_capacity_mb": 2048
    },
    "workers": {
      "status": "healthy",
      "total_workers": 4,
      "active_workers": 4,
      "idle_workers": 2,
      "busy_workers": 2,
      "total_capacity": 32,
      "current_load": 8
    },
    "storage": {
      "status": "healthy",
      "disk_usage_percent": 45.2,
      "available_gb": 500
    }
  },
  "metrics": {
    "jobs_completed_last_hour": 85,
    "jobs_failed_last_hour": 2,
    "average_processing_time_seconds": 21.3,
    "throughput_documents_per_hour": 170
  }
}
```

**Response (503 Service Unavailable)**
```json
{
  "status": "unhealthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api": {
      "status": "healthy"
    },
    "redis": {
      "status": "unhealthy",
      "error": "Connection timeout after 5 seconds"
    },
    "workers": {
      "status": "degraded",
      "total_workers": 4,
      "active_workers": 2,
      "message": "2 workers unresponsive"
    }
  }
}
```

### Performance Characteristics

**Throughput**
- Target: 170 documents per hour (with 4 workers @ 8 threads each)
- Actual: 150-180 documents per hour (depends on document size and complexity)
- Peak: 250 documents per hour (with 8 workers @ 8 threads each)

**Latency**
- Small batch (1-10 docs): 30-120 seconds
- Medium batch (11-50 docs): 2-10 minutes
- Large batch (51-100 docs): 10-30 minutes

**Resource Usage**
- Memory per worker: 2-4GB (depends on embedding model)
- CPU per worker: 80-95% utilization (8 threads)
- Redis memory: 100-500MB (for 1000 queued tasks)
- Disk I/O: 50-100MB/s (artifact storage writes)

### Scaling Guidelines

**Horizontal Scaling**
- Add workers across multiple machines to increase throughput linearly
- Each worker adds ~40-50 documents/hour capacity
- Redis can handle 50+ concurrent workers without performance degradation

**Vertical Scaling**
- Increase worker concurrency (8 → 16 threads) on machines with 16+ CPU cores
- Increase Redis memory allocation for larger queue depths
- Use NVMe storage for artifact storage to reduce I/O bottlenecks

**Auto-scaling Triggers**
- Scale up when queue depth > 500 tasks for > 5 minutes
- Scale down when queue depth < 50 tasks for > 15 minutes
- Min workers: 2 (for redundancy)
- Max workers: 20 (to prevent Redis overload)

## Batch Processing Infrastructure

High-throughput asynchronous batch processing system for evaluating large document collections.

### Architecture Diagrams

#### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Client Applications                           │
│  (Python scripts, Shell scripts, Postman, Web UI)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP/HTTPS
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Nginx Reverse Proxy                            │
│  • SSL termination (TLS 1.2+)                                        │
│  • Rate limiting (100 req/min per IP)                                │
│  • Load balancing across FastAPI instances (round-robin)             │
│  • Request buffering (client_max_body_size: 100MB)                   │
│  • Connection pooling (keepalive_timeout: 65s)                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server(s)                              │
│  • POST /api/v1/batch/upload       - Upload documents                │
│  • GET  /api/v1/batch/status/{id}  - Check job status                │
│  • GET  /api/v1/batch/results/{id} - Retrieve results                │
│  • DELETE /api/v1/batch/{id}       - Cancel job                      │
│  • GET  /api/v1/health             - Health check                    │
│  • GET  /api/v1/metrics            - Prometheus metrics              │
│  • Token-based authentication (Bearer tokens)                        │
│  • Request validation (1-100 documents per batch)                    │
│  • Async request handling (uvicorn workers)                          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Enqueue tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Redis Message Broker                             │
│  • Task queue: decalogo_batch_tasks                                  │
│  • Result backend: decalogo_batch_results                            │
│  • Priority queuing (high/normal/low)                                │
│  • TTL: 7 days for tasks, 30 days for results                       │
│  • Persistence: AOF + RDB snapshots every 60s                        │
│  • High availability: Redis Sentinel for failover                    │
│  • Memory: 4GB minimum, 8GB recommended                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Dequeue tasks
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Celery Worker Pool                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  Worker 1       │  │  Worker 2       │  │  Worker N       │    │
│  │  • 4 processes  │  │  • 4 processes  │  │  • 4 processes  │    │
│  │  • 8GB RAM      │  │  • 8GB RAM      │  │  • 8GB RAM      │    │
│  │  • 4 CPU cores  │  │  • 4 CPU cores  │  │  • 4 CPU cores  │    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    │
│           │                    │                    │              │
│           └────────────────────┼────────────────────┘              │
│                                │                                    │
│  • Concurrency: 4-8 processes per worker                           │
│  • Prefetch multiplier: 2 (8-16 tasks buffered)                    │
│  • Max retries: 3 with exponential backoff                         │
│  • Task timeout: 600s (10 minutes)                                 │
│  • Graceful shutdown: 30s timeout                                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ Write results
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Artifact Storage (MinIO/S3)                        │
│  • Bucket: decalogo-batch-artifacts                                  │
│  • Structure: /{job_id}/results.json                                 │
│             /{job_id}/documents/{doc_id}.json                        │
│             /{job_id}/logs/{worker_id}.log                           │
│  • Encryption: AES-256 at rest                                       │
│  • Lifecycle: 90 days retention, then archive to Glacier             │
│  • Replication: Cross-region for disaster recovery                   │
└──────────────────────────────────────────────────────────────────────┘
```

#### Data Flow Diagram

```
Upload Request                  Task Execution                Result Retrieval
    │                                │                              │
    ├─ POST /batch/upload           ├─ Worker pulls task           ├─ GET /batch/results/{id}
    │  └─ Documents[]               │  from Redis queue            │  └─ Fetch from MinIO/S3
    │                                │                              │     or Redis cache
    ├─ Validate request             ├─ Execute evaluation          │
    │  └─ 1-100 docs                │  pipeline:                   ├─ Response:
    │  └─ Auth token                │  ├─ Document segmentation    │  └─ Status: completed
    │                                │  ├─ Embedding generation     │  └─ Documents[]
    ├─ Generate job_id              │  ├─ Evidence extraction      │     ├─ doc_id
    │  └─ UUID v4                   │  ├─ Decálogo evaluation      │     ├─ scores{}
    │                                │  ├─ Questionnaire scoring    │     ├─ evidence[]
    ├─ Enqueue to Redis             │  └─ Result synthesis         │     └─ violations[]
    │  └─ Priority: normal          │                              │
    │                                ├─ Update status in Redis     ├─ GET /batch/status/{id}
    ├─ Return 202 Accepted          │  └─ progress: 50%            │  └─ {status, progress, eta}
    │  └─ {job_id, status}          │                              │
    │                                ├─ Write results to MinIO     │
    │                                │  └─ {job_id}/results.json   │
    │                                │                              │
    │                                ├─ Cache results in Redis     │
    │                                │  └─ TTL: 24 hours            │
    │                                │                              │
    │                                └─ Mark as completed          │
```

### API Endpoint Specifications

#### 1. Upload Batch Job

**Endpoint**: `POST /api/v1/batch/upload`

**Authentication**: Bearer token required

**Request Headers**:
```http
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "documents": [
    {
      "id": "doc-001",
      "title": "Plan de Desarrollo Municipal 2024",
      "content": "Full plan text...",
      "metadata": {
        "municipality": "Florencia",
        "department": "Caquetá",
        "year": 2024
      }
    }
  ],
  "options": {
    "priority": "normal",
    "include_evidence": true,
    "include_questionnaire": true,
    "webhook_url": "https://example.com/webhook",
    "timeout_seconds": 600
  }
}
```

**Request Schema**:
- `documents` (array, required): 1-100 documents
  - `id` (string, required): Unique document identifier
  - `title` (string, required): Document title
  - `content` (string, required): Full document text (max 1MB per doc)
  - `metadata` (object, optional): Additional metadata
- `options` (object, optional): Processing options
  - `priority` (enum: high/normal/low, default: normal)
  - `include_evidence` (boolean, default: true)
  - `include_questionnaire` (boolean, default: true)
  - `webhook_url` (string, optional): Callback URL on completion
  - `timeout_seconds` (integer, default: 600, max: 3600)

**Response**: `202 Accepted`
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "submitted_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:45:00Z",
  "document_count": 25,
  "status_url": "/api/v1/batch/status/550e8400-e29b-41d4-a716-446655440000",
  "results_url": "/api/v1/batch/results/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses**:

`400 Bad Request`: Invalid request (validation errors)
```json
{
  "error": "validation_error",
  "message": "Document count must be between 1 and 100",
  "details": {
    "field": "documents",
    "provided": 150,
    "allowed_range": [1, 100]
  }
}
```

`401 Unauthorized`: Missing or invalid authentication token
```json
{
  "error": "unauthorized",
  "message": "Invalid or expired authentication token"
}
```

`429 Too Many Requests`: Rate limit exceeded
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded: 100 requests per minute",
  "retry_after": 45
}
```

`503 Service Unavailable`: Queue full or workers unavailable
```json
{
  "error": "service_unavailable",
  "message": "All workers are currently busy. Please retry later.",
  "queue_length": 500
}
```

#### 2. Check Job Status

**Endpoint**: `GET /api/v1/batch/status/{job_id}`

**Authentication**: Bearer token required

**Response**: `200 OK`
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": {
    "total_documents": 25,
    "completed_documents": 12,
    "failed_documents": 0,
    "percent_complete": 48.0
  },
  "submitted_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:15Z",
  "estimated_completion": "2024-01-15T10:45:00Z",
  "worker_id": "worker-01",
  "current_phase": "evidence_extraction",
  "metrics": {
    "avg_processing_time_per_doc_seconds": 35.2,
    "queue_position": null
  }
}
```

**Status Values**:
- `queued`: Job is waiting in queue
- `processing`: Job is being processed
- `completed`: Job completed successfully
- `failed`: Job failed with errors
- `cancelled`: Job was cancelled by user
- `timeout`: Job exceeded timeout limit

**Error Responses**:

`404 Not Found`: Job ID not found
```json
{
  "error": "not_found",
  "message": "Job ID not found or expired",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 3. Retrieve Results

**Endpoint**: `GET /api/v1/batch/results/{job_id}`

**Authentication**: Bearer token required

**Query Parameters**:
- `include_evidence` (boolean, default: true): Include detailed evidence
- `format` (enum: json/csv, default: json): Response format

**Response**: `200 OK`
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "submitted_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:44:32Z",
  "total_processing_time_seconds": 872,
  "document_count": 25,
  "documents": [
    {
      "id": "doc-001",
      "title": "Plan de Desarrollo Municipal 2024",
      "status": "success",
      "processing_time_seconds": 35.2,
      "scores": {
        "decalogo": {
          "total_score": 78.5,
          "max_score": 100.0,
          "principles": {
            "p1_diagnostic": 8.0,
            "p2_objectives": 7.5,
            "p3_indicators": 6.8,
            "p4_causal_model": 7.2,
            "p5_feasibility": 8.1,
            "p6_financing": 7.0,
            "p7_prioritization": 7.8,
            "p8_alignment": 8.2,
            "p9_transparency": 9.0,
            "p10_monitoring": 8.9
          }
        },
        "questionnaire": {
          "d1_score": 7.5,
          "d2_score": 8.0,
          "d3_score": 7.2,
          "d4_score": 6.8,
          "d5_score": 8.1,
          "d6_score": 7.0,
          "total_score": 44.6,
          "max_score": 60.0
        }
      },
      "evidence": {
        "causal_patterns": 23,
        "monetary_amounts": 15,
        "responsibilities": 18,
        "feasibility_signals": 12,
        "contradictions": 2
      },
      "violations": [
        {
          "principle": "p3_indicators",
          "severity": "warning",
          "message": "Insufficient SMART indicators detected"
        }
      ],
      "metadata": {
        "municipality": "Florencia",
        "department": "Caquetá",
        "year": 2024
      }
    }
  ],
  "summary": {
    "avg_decalogo_score": 76.3,
    "avg_questionnaire_score": 42.1,
    "total_violations": 5,
    "documents_with_warnings": 3,
    "documents_with_errors": 0
  },
  "artifacts": {
    "full_results": "https://artifacts.decalogo.gov.co/550e8400/results.json",
    "evidence_details": "https://artifacts.decalogo.gov.co/550e8400/evidence.json",
    "execution_logs": "https://artifacts.decalogo.gov.co/550e8400/logs.txt"
  }
}
```

**Error Responses**:

`202 Accepted`: Job not yet completed
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Job is still processing. Check status endpoint for progress.",
  "progress": 48.0,
  "estimated_completion": "2024-01-15T10:45:00Z"
}
```

`500 Internal Server Error`: Job failed with errors
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "processing_error",
  "message": "Job failed during evidence extraction",
  "failed_at": "2024-01-15T10:35:22Z",
  "details": {
    "phase": "evidence_extraction",
    "error_type": "ModelLoadError",
    "affected_documents": ["doc-015", "doc-016"]
  }
}
```

#### 4. Cancel Job

**Endpoint**: `DELETE /api/v1/batch/{job_id}`

**Authentication**: Bearer token required

**Response**: `200 OK`
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Job cancellation requested",
  "cancelled_at": "2024-01-15T10:35:00Z",
  "documents_completed": 12,
  "documents_cancelled": 13
}
```

**Error Responses**:

`409 Conflict`: Job already completed or failed
```json
{
  "error": "conflict",
  "message": "Cannot cancel job with status: completed"
}
```

#### 5. Health Check

**Endpoint**: `GET /api/v1/health`

**Authentication**: None

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "api": {
      "status": "up",
      "response_time_ms": 5.2
    },
    "redis": {
      "status": "up",
      "connection_pool_size": 50,
      "used_connections": 12
    },
    "workers": {
      "status": "up",
      "active_workers": 5,
      "total_workers": 5,
      "queue_length": 23
    },
    "storage": {
      "status": "up",
      "available_space_gb": 250.5
    }
  },
  "metrics": {
    "total_jobs_processed": 1523,
    "jobs_in_queue": 23,
    "avg_processing_time_seconds": 34.5,
    "throughput_docs_per_hour": 165.2
  }
}
```

**Error Responses**:

`503 Service Unavailable`: One or more components unhealthy
```json
{
  "status": "degraded",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api": {"status": "up"},
    "redis": {"status": "down", "error": "Connection timeout"},
    "workers": {"status": "up"},
    "storage": {"status": "up"}
  }
}
```

#### 6. Metrics Endpoint

**Endpoint**: `GET /api/v1/metrics`

**Authentication**: Bearer token required

**Response**: `200 OK` (Prometheus format)
```
# HELP decalogo_batch_jobs_total Total number of batch jobs
# TYPE decalogo_batch_jobs_total counter
decalogo_batch_jobs_total{status="completed"} 1523
decalogo_batch_jobs_total{status="failed"} 12
decalogo_batch_jobs_total{status="cancelled"} 5

# HELP decalogo_batch_processing_time_seconds Processing time per document
# TYPE decalogo_batch_processing_time_seconds histogram
decalogo_batch_processing_time_seconds_bucket{le="10"} 234
decalogo_batch_processing_time_seconds_bucket{le="30"} 892
decalogo_batch_processing_time_seconds_bucket{le="60"} 1480
decalogo_batch_processing_time_seconds_bucket{le="+Inf"} 1523

# HELP decalogo_batch_queue_length Current queue length
# TYPE decalogo_batch_queue_length gauge
decalogo_batch_queue_length 23

# HELP decalogo_batch_active_workers Number of active workers
# TYPE decalogo_batch_active_workers gauge
decalogo_batch_active_workers 5
```

### Performance Targets

**Throughput**: 170 documents per hour (5 workers × 4 processes × 8.5 docs/hour)

**Latency**:
- Small batch (1-10 docs): < 5 minutes
- Medium batch (11-50 docs): < 20 minutes
- Large batch (51-100 docs): < 40 minutes

**Availability**: 99.5% uptime

**Resource Allocation per Worker**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB (logs + temp artifacts)

### Scaling Configuration

```
Production Deployment (5 Workers)
├─ Load Balancer (Nginx)
│  └─ 2 FastAPI instances (active-active)
│     ├─ Instance 1: 4 uvicorn workers
│     └─ Instance 2: 4 uvicorn workers
│
├─ Redis Cluster (High Availability)
│  ├─ Master: Write operations
│  ├─ Replica 1: Read operations
│  └─ Sentinel: Automatic failover
│
├─ Celery Worker Pool
│  ├─ Worker 1: 4 processes, 8GB RAM
│  ├─ Worker 2: 4 processes, 8GB RAM
│  ├─ Worker 3: 4 processes, 8GB RAM
│  ├─ Worker 4: 4 processes, 8GB RAM
│  └─ Worker 5: 4 processes, 8GB RAM
│
└─ MinIO/S3 Storage
   ├─ Bucket: decalogo-batch-artifacts
   └─ Replication: Cross-region
```

### Monitoring and Observability

**Metrics Collection**:
- Prometheus scraping `/api/v1/metrics` every 15s
- Grafana dashboards for real-time monitoring
- Alert rules for SLO violations

**Key Metrics**:
- Job submission rate (jobs/minute)
- Queue depth (jobs waiting)
- Processing time distribution (p50, p95, p99)
- Worker utilization (active/idle ratio)
- Error rate (failures/total jobs)
- Throughput (documents/hour)

**Logging**:
- Structured JSON logs via OpenTelemetry
- Centralized log aggregation (ELK/Loki)
- Trace correlation with job IDs
- Log retention: 30 days

**Alerting Rules**:
- Queue depth > 100 for > 10 minutes
- Worker failure rate > 5%
- Processing time p95 > 60 seconds
- Redis connection failures
- Storage capacity < 20%

See `BATCH_PROCESSING_GUIDE.md` for detailed setup and operational procedures.

## References

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [Canary Deployment Best Practices](https://cloud.google.com/architecture/application-deployment-and-testing-strategies)
- [SLO Implementation Guide](https://sre.google/workbook/implementing-slos/)
- [Celery Best Practices](https://docs.celeryq.dev/en/stable/userguide/optimizing.html)
- [Redis Persistence](https://redis.io/topics/persistence)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
