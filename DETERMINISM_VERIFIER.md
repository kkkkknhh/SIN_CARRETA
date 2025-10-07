# Determinism Verifier

Standalone utility for verifying deterministic execution of `miniminimoon_orchestrator.py`.

## Overview

The Determinism Verifier executes the MINIMINIMOON orchestrator twice on the same input PDF and performs comprehensive comparison of all artifacts to ensure perfect reproducibility. This is critical for:

- **Regression Testing**: Verify that code changes don't introduce non-deterministic behavior
- **Forensic Analysis**: Preserve both run outputs for detailed investigation
- **Compliance Validation**: Ensure evaluation results are reproducible for audit purposes
- **CI/CD Integration**: Automated determinism checks in build pipelines

## Features

### Comprehensive Artifact Comparison

1. **Evidence Registry State**
   - SHA-256 hash of deterministic evidence state
   - Evidence count and provenance validation
   - Component index verification

2. **Flow Runtime Validation**
   - Execution order verification
   - Stage-level trace comparison
   - Canonical flow hash matching

3. **Answer Report Comparison**
   - 300-question evaluation consistency
   - Score and confidence validation
   - Evidence linkage verification

4. **Coverage Report Validation**
   - Dimension-level coverage metrics
   - Evidence distribution analysis

### Advanced Comparison Logic

- **JSON Normalization**: Sorted key ordering ensures identical content is recognized regardless of serialization format
- **Non-Deterministic Field Removal**: Automatically strips timestamps, durations, and runtime-specific metadata
- **Byte-Level Hashing**: SHA-256 hashes computed on normalized JSON for precise comparison
- **Line-Level Diffs**: Unified diff output for human-readable discrepancy analysis

### Forensic Preservation

Both run outputs are preserved in timestamped directories:
```
artifacts/
  determinism_run_20240101_120000/
    run1/
      evidence_registry.json
      flow_runtime.json
      answers_report.json
      coverage_report.json
    run2/
      evidence_registry.json
      flow_runtime.json
      answers_report.json
      coverage_report.json
    determinism_report.json
    determinism_report.txt
```

## Usage

### Basic Usage

```bash
# Verify determinism for a PDF
python3 determinism_verifier.py path/to/plan.pdf

# Output: artifacts/determinism_run_<timestamp>/
```

### Custom Output Directory

```bash
# Specify custom output directory
python3 determinism_verifier.py plan.pdf --output-dir /tmp/determinism_test
```

### Exit Codes

- **0**: Perfect reproducibility (all artifacts match)
- **4**: Determinism violations detected (discrepancies found)
- **1**: Execution errors (orchestrator failures, missing artifacts, etc.)

### CI/CD Integration

```bash
#!/bin/bash
# ci_determinism_check.sh

python3 determinism_verifier.py test_fixtures/sample_plan.pdf

if [ $? -eq 0 ]; then
    echo "✓ Determinism check passed"
    exit 0
elif [ $? -eq 4 ]; then
    echo "✗ Determinism violations detected"
    exit 1
else
    echo "✗ Execution errors"
    exit 1
fi
```

## Output Reports

### JSON Report (`determinism_report.json`)

Structured machine-readable report with:

```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "input_pdf": "/path/to/plan.pdf",
  "perfect_match": true,
  "evidence_hash_run1": "abc123...",
  "evidence_hash_run2": "abc123...",
  "evidence_hash_match": true,
  "flow_hash_run1": "def456...",
  "flow_hash_run2": "def456...",
  "flow_hash_match": true,
  "artifact_comparisons": [
    {
      "artifact_name": "evidence_registry.json",
      "run1_hash": "hash1",
      "run2_hash": "hash1",
      "match": true,
      "run1_size": 12345,
      "run2_size": 12345,
      "diff_lines": []
    }
  ],
  "execution_errors": []
}
```

### Text Report (`determinism_report.txt`)

Human-readable report with:

```
================================================================================
DETERMINISM VERIFICATION REPORT
================================================================================

Timestamp: 2024-01-01T12:00:00.000000
Input PDF: /path/to/plan.pdf
Run 1 Directory: /path/to/run1
Run 2 Directory: /path/to/run2

================================================================================
RESULT: ✓ PERFECT REPRODUCIBILITY
================================================================================

EVIDENCE HASH COMPARISON:
  Run 1: abc123def456...
  Run 2: abc123def456...
  Status: ✓ MATCH

FLOW HASH COMPARISON:
  Run 1: def456abc789...
  Run 2: def456abc789...
  Status: ✓ MATCH

ARTIFACT COMPARISONS:
--------------------------------------------------------------------------------

Artifact: evidence_registry.json
  Run 1 Hash: hash1...
  Run 2 Hash: hash1...
  Run 1 Size: 12345 bytes
  Run 2 Size: 12345 bytes
  Status: ✓ MATCH

[... more artifacts ...]

================================================================================
END OF REPORT
================================================================================
```

### Discrepancy Reporting

When determinism violations are detected:

```
Artifact: answers_report.json
  Run 1 Hash: a1b2c3d4...
  Run 2 Hash: e5f6g7h8...
  Run 1 Size: 50000 bytes
  Run 2 Size: 50123 bytes
  Status: ✗ MISMATCH

  Diff (first 100 lines):
    --- run1/answers_report.json
    +++ run2/answers_report.json
    @@ -45,7 +45,7 @@
         "question_id": "D2-Q5",
    -    "score": 0.85,
    +    "score": 0.87,
         "confidence": 0.9
```

## Implementation Details

### Non-Deterministic Field Removal

The following fields are automatically removed before hash computation:

- `timestamp`, `start_time`, `end_time`
- `execution_time`, `duration_seconds`
- `stage_timestamps`, `creation_time`
- `absolute_path`

This ensures that:
1. Runtime-specific metadata doesn't cause false positives
2. Structural determinism is validated (content, not execution timing)
3. Evidence linkages and answer content are fully verified

### JSON Normalization Algorithm

```python
def normalize_json(json_path):
    # Load JSON
    data = json.load(json_path)
    
    # Remove non-deterministic fields
    remove_nondeterministic_fields(data)
    
    # Serialize with deterministic ordering
    canonical = json.dumps(
        data,
        sort_keys=True,          # Deterministic key order
        ensure_ascii=True,       # Consistent encoding
        separators=(',', ':')    # No whitespace
    )
    
    return canonical.encode('utf-8')
```

### Hash Computation

```python
# SHA-256 of normalized JSON
normalized_bytes = normalize_json(artifact_path)
artifact_hash = hashlib.sha256(normalized_bytes).hexdigest()
```

## Integration with Existing Infrastructure

### Evidence Registry Integration

The verifier uses the `deterministic_hash()` method from `EvidenceRegistry`:

```python
# From evidence_registry.py
def deterministic_hash(self) -> str:
    """Compute deterministic hash of entire registry."""
    ordered_evidence = sorted(self.store.keys())
    # ... hash computation
    return sha256_hex_digest
```

### Orchestrator Integration

The orchestrator's `export_artifacts()` method produces all required artifacts:

1. `evidence_registry.json` - Evidence state with deterministic hash
2. `flow_runtime.json` - Execution trace with flow hash
3. `answers_report.json` - Complete 300-question evaluation
4. `coverage_report.json` - Dimension-level coverage analysis

## Testing

Run the test suite:

```bash
python3 -m pytest test_determinism_verifier.py -v
```

Test coverage:
- JSON normalization (key ordering, whitespace handling)
- Non-deterministic field removal (timestamps, durations, nested fields)
- Artifact comparison (matching/mismatching scenarios)
- Hash extraction (evidence, flow)
- Diff generation (value changes, nested changes)
- Report generation (perfect match, mismatches, execution errors)
- Report export (JSON, text)

## Limitations

### Known Non-Deterministic Sources

The verifier **cannot** detect determinism violations caused by:

1. **External Model Randomness**: If embedding models or NER models have internal randomness
2. **System-Level Randomness**: OS-level entropy sources
3. **Concurrency Issues**: Race conditions in parallel evaluation (though the orchestrator uses locks)

These must be controlled via:
- Fixed random seeds (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`)
- Deterministic model configurations
- Thread-safe shared resources

### Performance Considerations

- Each run executes the full orchestrator (can take several minutes per run)
- 10-minute timeout per run (configurable in code)
- Artifacts are preserved on disk (can use significant space for large evaluations)

## Troubleshooting

### "Orchestrator execution timed out"

Increase timeout in `determinism_verifier.py`:

```python
result = subprocess.run(
    cmd,
    timeout=1200,  # Increase to 20 minutes
    ...
)
```

### "Missing required artifacts"

Verify orchestrator configuration:

```python
# In miniminimoon_orchestrator.py
orchestrator.export_artifacts(output_dir, pipeline_results=results)
```

### False Positives

If you see determinism violations but believe the results are correct:

1. Check if new non-deterministic fields were added (update `_remove_nondeterministic_fields()`)
2. Verify that random seeds are set consistently
3. Review the diff output to understand the discrepancy source

## Future Enhancements

Potential improvements:

1. **Parallel Execution**: Run both orchestrator instances concurrently
2. **Incremental Comparison**: Stream comparison during execution (don't wait for both runs)
3. **Statistical Validation**: Multiple runs with statistical analysis
4. **GPU Determinism**: CUDA determinism verification for GPU-accelerated models
5. **Distributed Tracing**: OpenTelemetry span comparison across runs

## Related Documentation

- `AGENTS.md` - Overall system architecture and commands
- `DEPLOYMENT_INFRASTRUCTURE.md` - Canary deployment and monitoring
- `miniminimoon_orchestrator.py` - Core orchestrator implementation
- `evidence_registry.py` - Evidence registry with deterministic hashing
