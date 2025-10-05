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

## References

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [Canary Deployment Best Practices](https://cloud.google.com/architecture/application-deployment-and-testing-strategies)
- [SLO Implementation Guide](https://sre.google/workbook/implementing-slos/)
