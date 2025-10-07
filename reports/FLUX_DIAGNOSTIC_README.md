# Flux Diagnostic Report Generator

## Overview

The Flux Diagnostic Report Generator (`flux_diagnostic_generator.py`) produces structured, machine-readable JSON diagnostic reports by consuming instrumentation data from pipeline monitoring components. The generated report (`flux_diagnostic.json`) provides comprehensive visibility into pipeline health, performance, and contract compliance.

## Data Sources

### Input Files (Expected in `artifacts/`)

1. **diagnostic_data.json** - Per-stage pipeline metrics from `diagnostic_runner.py`
2. **connection_data.json** - Inter-node flow analysis from `connection_stability_analyzer.py`
3. **determinism_data.json** - Verification results from `determinism_verifier.py`

### External Integrations

- **rubric_check.py** - Exit code captured for rubric alignment status
- **git** - Commit hash retrieved via `git rev-parse HEAD`
- **Python stdlib** - Environment metadata (version, platform, libraries)

## Report Structure

The generated `flux_diagnostic.json` contains four top-level sections:

### 1. pipeline_stages (array)

Per-stage metrics for all 15 documented pipeline stages:

```json
{
  "stage_name": "document_segmentation",
  "latency_ms": 120.5,
  "cpu_ms": 95.2,
  "peak_memory_mb": 256.8,
  "throughput": 8.3,
  "cache_hits": 42,
  "contract_checks": {
    "pass": 5,
    "fail": 0
  }
}
```

**Required Stages (validated):**
- document_segmentation
- embedding_generation
- semantic_search
- evidence_extraction
- responsibility_detection
- temporal_analysis
- causal_graph_construction
- intervention_identification
- outcome_mapping
- rubric_alignment
- scoring_computation
- contract_validation
- determinism_verification
- coverage_validation
- report_generation

### 2. connections (array)

Inter-node data flow analysis for 72 documented flow contracts:

```json
{
  "from_node": "embedding_generation",
  "to_node": "semantic_search",
  "contract_type": "data",
  "interface_check": {
    "passed": true,
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "stability_rate": 98.5,
  "suitability_verdict": "SUITABLE",
  "mismatch_examples": []
}
```

**Verdict Enums:**
- `SUITABLE` - Contract fully satisfied
- `UNSTABLE` - Intermittent contract violations
- `INCOMPATIBLE` - Persistent contract failures
- `DEGRADED` - Partial contract compliance
- `UNKNOWN` - Insufficient data

**Mismatch Examples** (populated on failure):
```json
"mismatch_examples": [
  {
    "expected": "dict",
    "actual": "list",
    "location": "embedding_generation.output"
  }
]
```

### 3. final_output (object)

Aggregated validation results:

```json
{
  "determinism_verified": true,
  "coverage_300": {
    "met": true,
    "actual_count": 315,
    "required_count": 300
  },
  "rubric_alignment": "ALIGNED",
  "quality_metrics": {
    "confidence_scores": {
      "overall": 0.87,
      "evidence_extraction": 0.92,
      "responsibility_detection": 0.85,
      "intervention_mapping": 0.84
    },
    "evidence_completeness": {
      "percentage": 94.5,
      "missing_categories": ["long_term_impacts"]
    }
  }
}
```

**Rubric Alignment Status** (from rubric_check.py exit code):
- `ALIGNED` (exit 0) - 1:1 alignment verified
- `ERROR` (exit 1) - Execution error
- `MISSING_FILES` (exit 2) - Required files not found
- `MISALIGNED` (exit 3) - Alignment validation failed
- `UNKNOWN` (exit -1) - rubric_check.py not available

### 4. environment (object)

Execution metadata for reproducibility:

```json
{
  "repo_commit_hash": "6f6f138c49e1850ff0245d5654705a17e814ed51",
  "python_version": "3.13.7",
  "os_platform": "macOS-15.4-arm64-arm-64bit-Mach-O",
  "library_versions": {
    "sentence-transformers": "2.2.0",
    "scikit-learn": "1.3.0",
    "numpy": "2.3.3",
    "spacy": "3.7.2",
    "pytest": "8.4.2",
    "torch": "2.1.0",
    "transformers": "4.35.0"
  },
  "execution_timestamp": "2025-10-07T17:22:34.281182+00:00"
}
```

## Usage

### Basic Execution

```bash
python3 reports/flux_diagnostic_generator.py
```

**Output:**
```json
{
  "status": "success",
  "output_file": "/path/to/reports/flux_diagnostic.json",
  "timestamp": "2025-10-07T17:22:34.283484+00:00"
}
```

### Programmatic Usage

```python
from reports.flux_diagnostic_generator import FluxDiagnosticGenerator

# Load instrumentation data
diagnostic_data = {...}  # From diagnostic_runner.py
connection_data = {...}  # From connection_stability_analyzer.py
determinism_data = {...}  # From determinism_verifier.py

# Generate report
generator = FluxDiagnosticGenerator(
    diagnostic_data=diagnostic_data,
    connection_data=connection_data,
    determinism_data=determinism_data,
    rubric_exit_code=0
)

report = generator.generate_report()
generator.save_report(Path("output/report.json"))
```

## Validation

### Stage Validation

The generator **fails with exit code 2** if:
- Any of the 15 expected stages is missing
- Stage objects lack required fields (stage_name, latency_ms, cpu_ms, peak_memory_mb, throughput, cache_hits, contract_checks)
- contract_checks missing `pass` or `fail` keys

**Error Example:**
```json
{
  "status": "validation_failed",
  "error": "Instrumentation data validation failed: Missing expected stages: ['rubric_alignment']"
}
```

### Connection Validation

The generator **fails with exit code 2** if:
- Fewer than 72 connections documented
- Connection objects lack required fields (from_node, to_node, interface_check, stability_rate, suitability_verdict, mismatch_examples)
- suitability_verdict not in valid enum values

**Error Example:**
```json
{
  "status": "validation_failed",
  "error": "Instrumentation data validation failed: Expected 72 flow contracts, found 50"
}
```

### Data Integrity

All required fields validated for correct types:
- Numeric fields: latency_ms, cpu_ms, peak_memory_mb, throughput, cache_hits, stability_rate
- Boolean: determinism_verified
- Objects: contract_checks, interface_check, coverage_300, quality_metrics
- Arrays: pipeline_stages, connections, mismatch_examples

## Exit Codes

- **0**: Report generated successfully
- **1**: Execution error (environment detection failed, file I/O error)
- **2**: Validation failed (incomplete/malformed instrumentation data)

## Testing

Comprehensive test suite in `test_flux_diagnostic_generator.py`:

```bash
python3 -m pytest test_flux_diagnostic_generator.py -v
```

**Test Coverage:**
- Valid data report generation (15 stages, 72 connections)
- Missing/incomplete stage validation
- Connection contract validation
- Verdict enum validation
- Environment metadata collection
- Final output aggregation
- Rubric exit code mapping
- JSON file persistence

## Integration with Pipeline

### 1. Generate Instrumentation Data

```bash
# Run pipeline with instrumentation enabled
python3 diagnostic_runner.py > artifacts/diagnostic_data.json
python3 connection_stability_analyzer.py > artifacts/connection_data.json
python3 determinism_verifier.py > artifacts/determinism_data.json
```

### 2. Generate Diagnostic Report

```bash
python3 reports/flux_diagnostic_generator.py
```

### 3. Analyze Results

```bash
# Pretty-print report
python3 -m json.tool reports/flux_diagnostic.json

# Extract specific metrics
jq '.pipeline_stages[] | select(.contract_checks.fail > 0)' reports/flux_diagnostic.json
jq '.connections[] | select(.suitability_verdict != "SUITABLE")' reports/flux_diagnostic.json
```

## Flow Contract Mapping

The 72 documented flow contracts map to 5 contract categories per flow pair:

1. **data** - Data format and schema contracts
2. **interface** - API interface contracts
3. **stability** - Temporal stability contracts
4. **performance** - Latency and throughput contracts
5. **determinism** - Reproducibility contracts

Plus 2 cross-stage contracts:
- **cache** - Embedding generation → Causal graph construction
- **bypass** - Evidence extraction → Scoring computation

## Dependencies

- Python 3.7+
- Standard library: json, sys, subprocess, platform, pathlib, datetime
- Testing: pytest

## Future Enhancements

- [ ] Real-time streaming report updates
- [ ] Historical trend analysis (compare reports over time)
- [ ] Anomaly detection for metric deviations
- [ ] Integration with monitoring dashboards (Grafana, DataDog)
- [ ] Contract violation alerting
- [ ] Performance regression detection
