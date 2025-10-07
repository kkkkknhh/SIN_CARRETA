# Flux Diagnostic Report Generator

## Overview

The Flux Diagnostic Report Generator (`generate_flux_diagnostic_report.py`) produces human-readable Markdown reports from structured JSON metrics collected during pipeline diagnostic runs. The generator transforms raw performance data into actionable insights with six comprehensive sections.

## Report Structure

### 1. Executive Summary (≤200 words)
- **Pipeline health status:** HEALTHY, DEGRADED, or CRITICAL
- **Node pass/fail statistics:** X/15 passing nodes
- **Connection stability:** Y/14 stable data flows (≥95% reliability)
- **Output quality verification:** Determinism, coverage, rubric alignment, gate passage
- **Critical findings:** High-level summary of issues or confirmation of nominal operation

### 2. Node-by-Node Performance Table
Displays all 15 pipeline stages with:
- **Latency:** Formatted with appropriate units (μs/ms/s)
- **Peak Memory:** Formatted in B/KB/MB/GB
- **Throughput:** Items/s or k items/s
- **Status:** ✓ PASS, ⚠ WARN, or ✗ FAIL
- **Notes:** Specific reason for status (e.g., "High latency: 2.35 s")

**Assessment Thresholds:**
- **FAIL:** Error rate >5%, latency >5s, memory >2GB
- **WARN:** Latency >2s, memory >1GB, throughput <1 item/s

### 3. Connection Assessment
Evaluates inter-node data flows with:
- **Stability percentage:** 0-100% reliability metric
- **Verdict:** ✓ EXCELLENT (≥99%), ✓ GOOD (≥95%), ⚠ ACCEPTABLE (≥85%), ✗ UNSTABLE (<85%)
- **Type mismatch examples:** Concrete field names showing contract violations

### 4. Final Output Quality
Verifies four key acceptance criteria:
- **Determinism:** Multiple runs produce identical SHA-256 hashes
- **Question coverage:** 300/300 questions required
- **Rubric alignment:** Exit code 0 from `tools/rubric_check.py`
- **Gate passage:** All 6 acceptance gates must pass

### 5. Top 5 Risks
Critical bottlenecks ranked by severity (0-100):
- **100:** Critical error rate (>10%)
- **90:** Excessive latency (>10s)
- **85:** Memory exhaustion (>4GB)
- **80:** Unstable data flow (<85% stability)
- **75:** Type contract violations (>5 mismatches)
- **70:** Performance degradation (other FAIL conditions)
- **50:** Performance warnings

Each risk includes:
- Location (specific node/connection name)
- Severity score
- Detailed description with metrics

### 6. Top 5 Recommended Fixes
Actionable recommendations for each identified risk, including:
- **Error Rate Fixes:** Circuit breakers, input validation, health checks, automatic failover
- **Latency Fixes:** Profiling, caching layers (LRU, TTL=1800s), batching/parallelization, timeouts
- **Memory Fixes:** Streaming/chunked processing, memory profiling, resource constraints, disk offloading
- **Data Flow Fixes:** Retry logic, message queues (Celery/RabbitMQ), dead letter queues, connection monitoring
- **Type Contract Fixes:** Schema validation, Pydantic models, CI/CD type checking (mypy --strict), contract tests

## Usage

### Manual Invocation

```bash
# Basic usage (defaults to reports/flux_diagnostic.json)
python3 generate_flux_diagnostic_report.py

# Custom input/output paths
python3 generate_flux_diagnostic_report.py path/to/input.json path/to/output.md
```

### Automatic Integration

Invoke after diagnostic execution completes:

```bash
# Example diagnostic workflow
python3 run_flux_diagnostic.py > reports/flux_diagnostic.json
python3 generate_flux_diagnostic_report.py reports/flux_diagnostic.json reports/flux_diagnostic.md
```

### Programmatic Usage

```python
from pathlib import Path
from generate_flux_diagnostic_report import generate_report

json_path = Path("reports/flux_diagnostic.json")
output_path = Path("reports/flux_diagnostic.md")

success = generate_report(json_path, output_path)
if success:
    print(f"Report generated: {output_path}")
else:
    print("Report generation failed")
```

## Input JSON Schema

The generator expects the following JSON structure:

```json
{
  "nodes": {
    "node_name": {
      "latency_ms": 123.4,
      "peak_memory_mb": 256.5,
      "throughput": 10.5,
      "error_rate": 0.005,
      "operations": 1000
    }
  },
  "connections": {
    "source->target": {
      "stability": 0.99,
      "type_mismatches": ["field1", "field2"],
      "throughput": 10.5
    }
  },
  "output_quality": {
    "determinism_verified": true,
    "determinism_runs": {
      "run_1": "hash1",
      "run_2": "hash2"
    },
    "question_coverage": 300,
    "rubric_check_exit_code": 0,
    "rubric_aligned": true,
    "all_gates_passed": true,
    "failed_gates": []
  }
}
```

### Required Fields

**nodes (dict):**
- `latency_ms` (float): Execution time in milliseconds
- `peak_memory_mb` (float): Peak memory usage in megabytes
- `throughput` (float): Items processed per second
- `error_rate` (float): Error rate as decimal (0.0 - 1.0)

**connections (dict):**
- `stability` (float): Connection stability as decimal (0.0 - 1.0)
- `type_mismatches` (list): Field names with type violations

**output_quality (dict):**
- `determinism_verified` (bool): Whether determinism check passed
- `question_coverage` (int): Number of questions covered (target: 300)
- `rubric_check_exit_code` (int): Exit code from rubric_check.py
- `rubric_aligned` (bool): Whether rubric alignment passed
- `all_gates_passed` (bool): Whether all acceptance gates passed
- `failed_gates` (list): Names of failed gates (empty if all passed)

## Testing

Run the test suite:

```bash
python3 -m pytest test_flux_diagnostic_report.py -v
```

Test coverage includes:
- Formatting functions (bytes, latency, throughput)
- Node health assessment logic
- Risk identification and ranking
- Full report generation (success and failure cases)
- Error handling (missing files, invalid JSON)

## Example Output

See `reports/sample_flux_diagnostic.md` for a complete example report generated from `reports/sample_flux_diagnostic.json`.

Key features demonstrated:
- All 15 pipeline stages with performance metrics
- 14 inter-node connections with stability assessment
- Output quality verification across all 4 criteria
- Risk identification with severity ranking
- Actionable fix recommendations with concrete implementation details

## Integration Points

### With Diagnostic Runners

```python
# Example: Integrate with existing diagnostic runner
import json
from generate_flux_diagnostic_report import generate_report

def run_diagnostic_with_report(plan_text):
    # Run diagnostic
    metrics = collect_pipeline_metrics(plan_text)
    
    # Save JSON
    json_path = Path("reports/flux_diagnostic.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate report
    output_path = Path("reports/flux_diagnostic.md")
    generate_report(json_path, output_path)
    
    return metrics, output_path
```

### With CI/CD Pipelines

```yaml
# Example: GitHub Actions workflow
- name: Run Diagnostic
  run: python3 run_flux_diagnostic.py > reports/flux_diagnostic.json

- name: Generate Report
  run: python3 generate_flux_diagnostic_report.py

- name: Upload Artifacts
  uses: actions/upload-artifact@v3
  with:
    name: diagnostic-reports
    path: reports/flux_diagnostic.*
```

### With Monitoring Systems

```python
# Example: Parse report for alerting
import json

with open("reports/flux_diagnostic.json") as f:
    data = json.load(f)

# Check for critical failures
critical_nodes = [
    name for name, metrics in data["nodes"].items()
    if metrics.get("error_rate", 0) > 0.05
]

if critical_nodes:
    send_alert(f"Critical failures in: {', '.join(critical_nodes)}")
```

## Performance

- **Report generation time:** <100ms for typical diagnostic output
- **Memory usage:** <50MB (JSON parsing + report assembly)
- **Dependencies:** Python 3.7+ standard library only (no external packages)

## Maintenance

### Adding New Metrics

To add new metrics to the report:

1. Update input JSON schema in diagnostic collector
2. Add metric extraction in relevant generator function
3. Update formatting functions if needed
4. Add test cases for new metrics
5. Update this documentation

### Modifying Thresholds

Threshold constants are defined in `assess_node_health()`:

```python
# Current thresholds
ERROR_RATE_FAIL = 0.05  # 5%
ERROR_RATE_WARN = 0.02  # 2%
LATENCY_FAIL_MS = 5000  # 5 seconds
LATENCY_WARN_MS = 2000  # 2 seconds
MEMORY_FAIL_MB = 2048   # 2GB
MEMORY_WARN_MB = 1024   # 1GB
```

Update these values based on operational experience.

## Troubleshooting

### Report Generation Fails

**Error: JSON file not found**
```bash
✗ Error: JSON file not found: reports/flux_diagnostic.json
```
**Solution:** Ensure diagnostic runner completed successfully and output file exists.

**Error: Invalid JSON format**
```bash
✗ Error: Invalid JSON format: Expecting ',' delimiter
```
**Solution:** Validate JSON syntax using `python3 -m json.tool flux_diagnostic.json`

**Error: Missing required fields**
```bash
KeyError: 'nodes'
```
**Solution:** Ensure input JSON contains all required top-level keys (nodes, connections, output_quality).

### Empty or Incomplete Reports

**Symptom:** Report generated but sections are empty
**Cause:** Input JSON has empty collections or missing nested fields
**Solution:** Verify diagnostic collector populates all required fields per schema

## See Also

- **AGENTS.md:** Build, lint, and test commands
- **ARCHITECTURE.md:** Pipeline stage descriptions
- **DEPLOYMENT_INFRASTRUCTURE.md:** Monitoring and alerting integration
- **tools/rubric_check.py:** Rubric alignment verification tool
