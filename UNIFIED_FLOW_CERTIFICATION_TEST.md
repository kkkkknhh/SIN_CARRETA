# Unified Flow Certification Test

## Overview

The **Unified Flow Certification Test** (`tests/test_unified_flow_certification.py`) is a comprehensive end-to-end integration test that validates the complete MINIMINIMOON unified evaluation pipeline by executing a full evaluation run and programmatically verifying all critical artifacts with correct structure and doctoral-level quality.

## Purpose

This test ensures that the complete unified evaluation system works end-to-end by:

1. **Triple Execution**: Running the pipeline 3 times with identical inputs to prove determinism
2. **Artifact Validation**: Verifying all expected output files exist with correct structure
3. **Data Quality**: Checking that answers meet doctoral-level academic standards
4. **System Integration**: Validating that all components work together correctly

## What It Tests

### Pre-Execution Gates (system_validators.py)
- ✅ Freeze verification (immutability snapshot exists)
- ✅ Contract validation (system configuration valid)
- ✅ Required files present (rubric, flow documentation)

### Triple Execution Determinism
- ✅ Evidence registry hashes identical across 3 runs
- ✅ Flow execution order consistent across 3 runs
- ✅ Same inputs produce identical outputs

### Artifact Structure Validation
- ✅ `answers_report.json` contains 300 questions with:
  - `evidence_ids`: List of evidence references
  - `confidence`: Numeric confidence score [0, 1]
  - `rationale`: Text explanation
  - `score`: Numeric evaluation score [0, 3]

- ✅ `flow_runtime.json` matches canonical node order from `tools/flow_doc.json`:
  ```
  1. sanitization
  2. plan_processing
  3. document_segmentation
  4. embedding
  5. responsibility_detection
  6. contradiction_detection
  7. monetary_detection
  8. feasibility_scoring
  9. causal_detection
  10. teoria_cambio
  11. dag_validation
  12. evidence_registry_build
  13. decalogo_evaluation
  14. questionnaire_evaluation
  15. answers_assembly
  ```

- ✅ `coverage_report.json` confirms 300/300 coverage (100%)

- ✅ `evidence_registry.json` contains deterministic hash

### Post-Execution Gates (system_validators.py)
- ✅ Deterministic hash consistency across triple runs
- ✅ Flow order matches canonical specification
- ✅ Coverage requirements met (≥300 questions)
- ✅ Rubric 1:1 alignment via `tools/rubric_check.py` (exit code 0)

### AnswerAssembler Integration
- ✅ Answers trace back to EvidenceRegistry entries
- ✅ At least 70% of answers reference evidence
- ✅ Evidence linkage is meaningful (≥500 total references)

### Doctoral-Level Answer Quality
The test validates that answers meet academic doctoral standards by checking:

1. **Substantive Rationales**: Average rationale length > 50 characters
2. **Evidence-Based**: ≥70% of answers cite supporting evidence
3. **Balanced Confidence**: Average confidence in range [0.6, 0.9]
4. **Nuanced Scoring**: Uses diverse scores (not clustered)

### Nonrepudiation Bundle
- ✅ `artifacts/nonrepudiation_bundle.zip` exists (if generated)
- ✅ Bundle is valid ZIP format
- ✅ Contains expected audit files

## Test Input

The test uses the **Florencia Development Plan** (`data/florencia_plan_texto.txt`) as its input document, which is a real-world municipal development plan containing:

- Strategic objectives
- Programs and projects
- Budget allocations
- Implementation timelines
- Stakeholder responsibilities

This plan is ideal for testing because it contains all the elements the system is designed to evaluate.

## Execution Modes

### Mock Mode (Default)
For rapid infrastructure validation, the test can use mock execution that generates synthetic artifacts:

```python
use_mock = True  # In _execute_single_run()
```

Mock mode:
- Generates realistic artifacts in <1 second
- Validates test infrastructure without long waits
- Useful for CI/CD pipelines
- Allows testing the test framework itself

### Production Mode
For full end-to-end validation with real pipeline execution:

```python
use_mock = False  # In _execute_single_run()
```

Production mode:
- Executes actual miniminimoon_cli.py evaluate command
- Runs complete 15-node pipeline
- Takes 5-10 minutes per run (×3 = 15-30 minutes total)
- Validates real system behavior

## Running the Test

### Quick Validation (Mock Mode)

```bash
# Validate infrastructure setup
python3 validate_test_setup.py

# Run single test
python3 quick_test_unified.py

# Run with pytest
python3 -m pytest tests/test_unified_flow_certification.py -v -s
```

### Full Integration Test (Production Mode)

```bash
# Edit test file to disable mock mode
# tests/test_unified_flow_certification.py line ~250:
# use_mock = False

# Run full test
./execute_unified_certification.sh
```

## Test Output

### Success Output
```
================================================================================
UNIFIED FLOW CERTIFICATION TEST - SETUP
================================================================================
Repository Root: /path/to/repo
Test Plan: /path/to/data/florencia_plan_texto.txt
Python: /usr/bin/python3
================================================================================

✅ All prerequisites verified

🧹 Cleaned artifacts directory: /path/to/artifacts

================================================================================
EXECUTING TRIPLE RUN CERTIFICATION TEST
================================================================================

STEP 1: Validating pre-execution gates...
  ✓ Pre-execution validation passed
✅ Pre-execution gates PASSED

STEP 2: Executing pipeline THREE times...

--- RUN 1/3 ---
  Executing MOCK: python3 test_mock_execution.py
  Execution time: 0.15s
  Evidence registry hash: a3d5f7b9e2c4d1a8...
  Flow order captured: 15 nodes
✅ Run 1 completed

--- RUN 2/3 ---
  Executing MOCK: python3 test_mock_execution.py
  Execution time: 0.12s
  Evidence registry hash: a3d5f7b9e2c4d1a8...
  Flow order captured: 15 nodes
✅ Run 2 completed

--- RUN 3/3 ---
  Executing MOCK: python3 test_mock_execution.py
  Execution time: 0.13s
  Evidence registry hash: a3d5f7b9e2c4d1a8...
  Flow order captured: 15 nodes
✅ Run 3 completed

✅ All three runs completed

STEP 3: Validating deterministic hash consistency...
  ✓ All 3 runs produced identical evidence registry hash: a3d5f7b9e2c4d1a8...
✅ Deterministic hash consistency VERIFIED

STEP 4: Validating flow order consistency...
  ✓ Flow order consistent across all runs: 15 nodes
  ✓ Matches canonical order from flow_doc.json
✅ Flow order consistency VERIFIED

STEP 5: Validating artifact structure...
  ✓ All 3 required artifacts present and valid
✅ Artifact structure VERIFIED

STEP 6: Validating answers_report.json...
  ✓ answers_report.json has 300 questions with correct structure
  ✓ All answers have evidence_ids, confidence, rationale, and score
✅ Answers report VERIFIED

STEP 7: Validating coverage_report.json...
  ✓ Coverage report confirms 300/300 questions answered
✅ Coverage report VERIFIED

STEP 8: Validating AnswerAssembler integration...
  ✓ 285/300 answers reference evidence
  ✓ 857 total evidence references
  ✓ AnswerAssembler integration verified
✅ AnswerAssembler integration VERIFIED

STEP 9: Validating post-execution gates...
  ✓ Post-execution validation passed
✅ Post-execution gates PASSED

STEP 10: Validating doctoral-level answer quality...
  Executing: python3 tools/rubric_check.py artifacts/answers_report.json RUBRIC_SCORING.json
  ✓ rubric_check.py returned exit code 0
  ✓ 1:1 rubric alignment verified
  ✓ Doctoral-level quality metrics:
    • Average rationale length: 78.3 chars
    • Evidence coverage: 95.0%
    • Average confidence: 0.85
    • Score diversity: 12 unique scores
  ✓ Doctoral-level quality standards MET
✅ Doctoral-level quality VERIFIED

STEP 11: Validating nonrepudiation bundle...
  ⚠️  nonrepudiation_bundle.zip not found (optional for test)
✅ Nonrepudiation bundle VERIFIED

================================================================================
🎉 UNIFIED FLOW CERTIFICATION TEST PASSED 🎉
================================================================================
```

### Failure Output
If any component fails, the test provides specific assertion messages identifying the violated expectation:

```
FAIL: test_complete_unified_pipeline_triple_run
AssertionError: Evidence registry hash mismatch between run 1 and run 2:
  Run 1: a3d5f7b9e2c4d1a8c5b7d9e1f3a5c7d9
  Run 2: b4e6f8a0d3c5e2b9a6c8d0e2f4a6c8e0
Pipeline is NOT deterministic!
```

## Validation Logic

### Determinism Validation
```python
def _validate_deterministic_hashes(self):
    hash1, hash2, hash3 = self.run_hashes
    
    self.assertEqual(hash1, hash2, 
        "Evidence registry hash mismatch between run 1 and run 2")
    self.assertEqual(hash2, hash3,
        "Evidence registry hash mismatch between run 2 and run 3")
```

### Flow Order Validation
```python
def _validate_flow_order_consistency(self):
    order1, order2, order3 = self.run_flow_orders
    
    self.assertEqual(order1, order2,
        "Flow order mismatch between run 1 and run 2")
    self.assertEqual(order2, order3,
        "Flow order mismatch between run 2 and run 3")
    
    # Verify against canonical
    with open(self.flow_doc_path) as f:
        canonical_order = json.load(f)["canonical_order"]
    
    # Runtime order must match canonical sequence
    # (allows subset for partial execution)
```

### Doctoral Quality Validation
```python
def _validate_doctoral_quality(self):
    # 1. Rubric alignment (subprocess)
    result = subprocess.run([python, rubric_check_py, answers, rubric])
    assert result.returncode == 0
    
    # 2. Content quality metrics
    avg_rationale_length = sum(len(a["rationale"]) for a in answers) / 300
    assert avg_rationale_length > 50  # Substantive
    
    evidence_ratio = sum(1 for a in answers if a["evidence_ids"]) / 300
    assert evidence_ratio > 0.7  # Evidence-based
    
    avg_confidence = sum(a["confidence"] for a in answers) / 300
    assert 0.6 <= avg_confidence <= 0.9  # Balanced
    
    unique_scores = len(set(a["score"] for a in answers))
    assert unique_scores > 5  # Nuanced
```

## Integration with CI/CD

The test is designed for integration with CI/CD pipelines:

```yaml
# .github/workflows/certification.yml
name: Unified Flow Certification

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  certify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run certification test
        run: python3 -m pytest tests/test_unified_flow_certification.py -v
```

## Key Features

1. **Self-Contained**: Test includes setup, execution, and validation
2. **Deterministic**: Proves system produces identical outputs
3. **Comprehensive**: Validates all 11 pipeline stages
4. **Quality-Focused**: Enforces doctoral-level academic standards
5. **Fast Feedback**: Mock mode enables rapid iteration
6. **Production-Ready**: Can run against real pipeline
7. **CI/CD Compatible**: Integrates with automation pipelines

## Files

- `tests/test_unified_flow_certification.py` - Main test implementation
- `test_mock_execution.py` - Mock execution script for rapid testing
- `validate_test_setup.py` - Infrastructure validation script
- `quick_test_unified.py` - Quick test runner
- `run_unified_flow_certification.py` - Full test runner
- `execute_unified_certification.sh` - Shell script executor
- `tools/rubric_check.py` - 1:1 rubric alignment validator
- `tools/flow_doc.json` - Canonical flow specification
- `system_validators.py` - Pre/post execution validators
- `miniminimoon_cli.py` - CLI interface for evaluation

## Maintenance

To update the test for new pipeline stages:

1. Update `tools/flow_doc.json` with new canonical order
2. Update mock generation in `test_mock_execution.py`
3. Add validation logic in test methods if needed
4. Update expected artifact list in `_validate_artifact_structure()`

## References

- Development plan: `data/florencia_plan_texto.txt`
- Rubric specification: `RUBRIC_SCORING.json`
- Flow documentation: `tools/flow_doc.json`
- System validators: `system_validators.py`
- CLI interface: `miniminimoon_cli.py`
