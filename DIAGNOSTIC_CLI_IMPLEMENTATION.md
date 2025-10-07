# Diagnostic CLI Implementation

## Overview

Extended `miniminimoon_cli.py` with a new `diagnostic` command that provides comprehensive instrumentation of the canonical pipeline execution with full performance metrics, error tracking, and determinism validation.

## Implementation Details

### 1. New Files Created

#### `diagnostic_runner.py`
- **InstrumentedOrchestrator**: Extends `MiniminiMoonOrchestrator` with node-level instrumentation
- **NodeMetrics**: Dataclass for capturing timing, status, and error information per pipeline node
- **DiagnosticReport**: Complete report structure with connection stability, output quality, and determinism checks
- **run_diagnostic()**: Main function that executes instrumented pipeline with comprehensive error handling
- **generate_reports()**: Generates both JSON and Markdown reports from diagnostic data

Key Features:
- Full try-except wrapping at each node with immediate error capture
- Stack traces written to both log and stderr with context (node name, input state)
- Performance metrics captured at microsecond precision
- Connection stability analysis during warmup phase
- Output quality assessment (segment counts, embeddings, detectors)
- Determinism verification (hash-based with extensibility for multi-run checks)

#### `test_diagnostic_runner.py`
- Unit tests for `NodeMetrics` and `DiagnosticReport` dataclasses
- Report generation tests (success and failure scenarios)
- JSON and Markdown format validation
- Temporary file handling for test isolation

#### `validate_diagnostic_cli.py`
- Integration validation script
- Tests CLI argument parsing and help output
- Syntax validation for all new files
- File structure verification

### 2. Modified Files

#### `miniminimoon_cli.py`
Extended with new `cmd_diagnostic()` function:
- **Logging Configuration**: Dedicated FileHandler for `reports/diagnostic.log` with INFO level
- **Format**: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
- **Multi-logger Setup**: Captures output from:
  - `diagnostic_runner`
  - `connection_stability_analyzer` (future integration)
  - `output_quality_assessor` (future integration)
  - `determinism_verifier` (future integration)
  - `miniminimoon_orchestrator`
- **Error Handling**: 
  - Try-except wraps entire execution
  - Errors logged with full stack trace and context
  - Stack traces written to both log file and stderr
  - Exit code 1 on any exception
- **Report Output**:
  - Markdown report printed to stdout
  - Summary with total execution time
  - Top-3 performance bottlenecks extracted from JSON
  - Non-deterministic behavior warnings
- **Directory Management**: Ensures `reports/` directory exists via `pathlib.Path`

### 3. Instrumentation Points

The diagnostic runner instruments these pipeline nodes:
1. `compute_plan_hash` - Document fingerprinting
2. `sanitize_plan` - Text normalization
3. `segment_document` - Document segmentation
4. `generate_embeddings` - Embedding model inference
5. `detect_responsibilities` - Responsibility detection
6. `detect_contradictions` - Contradiction analysis
7. `detect_monetary` - Monetary entity extraction
8. `score_feasibility` - Feasibility scoring
9. `detect_causal_patterns` - Causal pattern detection
10. `validate_teoria_cambio` - Theory of change validation
11. `validate_dag` - DAG validation
12. `evaluate_questionnaire` - Questionnaire engine execution

### 4. Output Artifacts

#### `reports/diagnostic.log`
- Timestamped execution log with module names
- Node-level start/completion messages
- Error traces with full context
- Connection stability metrics
- Run summary with status

#### `reports/flux_diagnostic.json`
```json
{
  "total_execution_time_ms": 5000.0,
  "node_metrics": [
    {
      "node_name": "segment_document",
      "start_time": 1234567890.123,
      "end_time": 1234567891.456,
      "duration_ms": 1333.0,
      "status": "success",
      "error_msg": "",
      "input_state": {...},
      "output_state": {...}
    }
  ],
  "connection_stability": {
    "warmup_duration_ms": 1200.0,
    "models_loaded": true,
    "status": "stable"
  },
  "output_quality": {
    "segments_count": 42,
    "embeddings_count": 42,
    "responsibilities_count": 15,
    "quality_status": "passed"
  },
  "determinism_check": {
    "plan_hash": "abc123...",
    "deterministic": true,
    "notes": "Single run - determinism requires multiple executions"
  },
  "status": "success"
}
```

#### `reports/flux_diagnostic.md`
- Human-readable diagnostic report
- Section for connection stability
- Section for output quality assessment
- Section for determinism check
- Table of node execution metrics
- Failed node details with error messages

### 5. CLI Usage

```bash
# Basic diagnostic run
python3 miniminimoon_cli.py diagnostic plan.txt

# With custom repo and rubric
python3 miniminimoon_cli.py diagnostic plan.txt --repo /path/to/repo --rubric custom_rubric.json

# View help
python3 miniminimoon_cli.py diagnostic --help
```

### 6. Error Handling Strategy

#### Node-Level Error Capture
```python
try:
    result = func(*args, **kwargs)
    # Log success metrics
except Exception as e:
    # Capture error with context
    error_msg = str(e)
    stack_trace = traceback.format_exc()
    
    # Log to file with context
    logger.error(f"Failed node: {node_name} - {error_msg}")
    logger.error(f"Input state: {input_state}")
    logger.error(stack_trace)
    
    # Re-raise for top-level handler
    raise
```

#### Top-Level Error Capture
- Catches all exceptions from diagnostic execution
- Logs error with full context to log file
- Prints error summary to stderr
- Includes plan path, repo, and log file location
- Returns exit code 1

### 7. Performance Bottleneck Identification

The CLI automatically:
1. Sorts nodes by `duration_ms` (descending)
2. Extracts top-3 slowest operations
3. Displays in summary:
   ```
   Top 3 Performance Bottlenecks:
     1. generate_embeddings: 2500.00ms (success)
     2. evaluate_questionnaire: 1800.00ms (success)
     3. segment_document: 1200.00ms (success)
   ```

### 8. Determinism Validation

Current Implementation:
- Single-run SHA-256 hash of sanitized plan text
- Captures run timestamp
- Includes note about multi-run requirement

Future Extension Points:
- `determinism_verifier.py` integration
- Multi-run comparison with variance analysis
- Seed reproducibility validation
- Output artifact comparison

### 9. Validation Results

```bash
$ python3 validate_diagnostic_cli.py
================================================================================
DIAGNOSTIC CLI VALIDATION
================================================================================

✓ miniminimoon_cli.py exists
✓ diagnostic_runner.py exists
✓ test_diagnostic_runner.py exists

✓ miniminimoon_cli.py has valid syntax
✓ diagnostic_runner.py has valid syntax
✓ test_diagnostic_runner.py has valid syntax

✓ CLI help includes diagnostic command

✓ Diagnostic command help works correctly

================================================================================
✅ ALL VALIDATION TESTS PASSED
================================================================================
```

### 10. Integration Points

The diagnostic runner is designed to integrate with:
- **connection_stability_analyzer.py**: Model loading and warmup analysis
- **output_quality_assessor.py**: Quality metrics validation
- **determinism_verifier.py**: Multi-run comparison and reproducibility checks

These components can be added by:
1. Creating the respective modules
2. Importing in `diagnostic_runner.py`
3. Calling their analysis functions
4. Including results in the diagnostic report
5. Logs automatically captured via configured loggers

## Architecture Decisions

1. **Inheritance over Composition**: `InstrumentedOrchestrator` extends `MiniminiMoonOrchestrator` for minimal code duplication
2. **Dataclasses for Reports**: Type-safe, serializable, easy to extend
3. **Dual Output Format**: JSON for programmatic access, Markdown for human readability
4. **Comprehensive Logging**: Separate logger per component with centralized file handler
5. **Fail-Fast with Context**: Immediate error capture with full context preservation
6. **Exit Code Convention**: 0=success, 1=error (consistent with CLI standards)

## Testing Strategy

1. **Unit Tests**: `test_diagnostic_runner.py` validates core data structures
2. **Integration Tests**: `validate_diagnostic_cli.py` validates CLI integration
3. **Syntax Validation**: All files pass `py_compile`
4. **Manual Testing**: Requires full dependency installation for end-to-end validation

## Limitations & Future Work

1. **Dependencies**: Full execution requires all MINIMINIMOON dependencies (networkx, spacy, torch, etc.)
2. **Multi-Run Determinism**: Current implementation only captures single-run hash
3. **Performance Baselines**: No threshold-based alerts for performance regressions
4. **Memory Profiling**: Not yet integrated (memory-profiler available in requirements.txt)
5. **Distributed Tracing**: OpenTelemetry integration mentioned but not yet implemented

## Conclusion

The diagnostic CLI command provides comprehensive instrumentation of the canonical pipeline with:
- ✅ Full node-level timing metrics
- ✅ Step-level error capture with context
- ✅ Dual-format reporting (JSON + Markdown)
- ✅ Performance bottleneck identification
- ✅ Connection stability analysis
- ✅ Output quality assessment
- ✅ Determinism validation hooks
- ✅ Comprehensive logging with timestamps and module names
- ✅ Proper error handling with exit codes
- ✅ Directory management (reports/ auto-creation)

The implementation follows best practices for CLI design, error handling, and observability while maintaining compatibility with the existing MINIMINIMOON architecture.
