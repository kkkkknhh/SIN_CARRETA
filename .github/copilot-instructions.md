# GitHub Copilot Instructions - MINIMINIMOON

This document provides guidance for GitHub Copilot when working with the MINIMINIMOON Sistema Canónico de Evaluación de PDM (Municipal Development Plan Evaluation System).

## Repository Overview

MINIMINIMOON is a deterministic and reproducible evaluation system for Municipal Development Plans (PDM) based on:
- **300 structured questions** organized in 10 thematic sections × 30 questions each
- **Canonical flow** with 72 critical flows verified
- **6 acceptance gates** mandatory for quality assurance
- **Single Evidence Registry** as the single source of truth
- **Complete traceability** from evidence to final answer

## Critical Constraints

### 1. Python Version Requirement

**CRITICAL: This system requires Python 3.10 exactly.** Other versions are not supported due to:
- NumPy compatibility requirements (>=1.21.0, <1.25.0 for Python 3.10 support)
- Dependency version constraints for embedding models
- OpenTelemetry instrumentation compatibility

Always verify Python version in any changes:
```python
from version_validator import validate_python_310
validate_python_310()  # Raises RuntimeError if not Python 3.10
```

### 2. Determinism and Reproducibility

**NEVER introduce non-deterministic behavior:**
- All random number generators must use fixed seeds
- No timestamps in deterministic outputs
- No floating-point arithmetic without normalization
- All file operations must maintain canonical order

Verify reproducibility:
```bash
python verify_reproducibility.py
python determinism_verifier.py <input_pdf>
```

### 3. Canonical Flow Order

**NEVER modify the canonical order** of the 15 pipeline stages without updating `tools/flow_doc.json`:

1. Sanitization (`plan_sanitizer`)
2. Plan Processing (`plan_processor`)
3. Document Segmentation (`document_segmenter`)
4. Embeddings (`embedding_model`)
5. Responsibility Detection (`responsibility_detector`)
6. Contradiction Detection (`contradiction_detector`)
7. Monetary Detection (`monetary_detector`)
8. Feasibility Scoring (`feasibility_scorer`)
9. Causal Pattern Detection (`causal_pattern_detector`)
10. Theory of Change (`teoria_cambio`)
11. DAG Validation (`dag_validation`)
12. Evidence Registry Construction (`evidence_registry`)
13. Question Engine (`questionnaire_engine`)
14. Answer Assembly (`answer_assembler`)
15. Trace Matrix Generation

Verify canonical flow:
```bash
python verify_critical_flows.py
```

### 4. Deprecated Modules

**NEVER import or use deprecated modules:**
- `decalogo_pipeline_orchestrator.py` - Deprecated, use `miniminimoon_orchestrator.py`
- Any module marked as deprecated in `DEPRECATIONS.md`

The system will raise `RuntimeError` if deprecated modules are imported.

## Acceptance Gates (Mandatory)

All code changes must respect the 6 acceptance gates:

### Gate #1: Immutable Configuration ✅
```bash
python miniminimoon_orchestrator.py freeze ./config/
```
- Verifies: SHA-256 snapshot of configuration files
- Block: Fails if configuration changes during execution

### Gate #2: Canonical Flow Order ✅
```bash
python verify_critical_flows.py
```
- Verifies: 15 stages in canonical order (matches `tools/flow_doc.json`)
- Block: Fails if flow order deviates

### Gate #3: Deterministic Hash ✅
```bash
python verify_reproducibility.py
```
- Verifies: Evidence registry hash reproducible across runs
- Block: Exit code 4 if hash mismatch

### Gate #4: Complete Coverage ✅
- Verifies: `answers_report.summary.total_questions ≥ 300`
- Block: Fails if not all 300 questions answered

### Gate #5: Rubric Alignment ✅
```bash
python rubric_check.py
```
- Verifies: 1:1 mapping questions ↔ weights (no missing/extra)
- Block: Exit code 3 if misalignment

### Gate #6: No Deprecated Orchestrator ✅
- Verifies: `decalogo_pipeline_orchestrator` NOT used
- Block: RuntimeError on import of deprecated module

## Project Structure

### Entry Point (ONLY USE THIS)
```python
from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator

orchestrator = CanonicalDeterministicOrchestrator(
    config_dir="./config",
    enable_validation=True,
    flow_doc_path="tools/flow_doc.json"
)
results = orchestrator.process_plan_deterministic(plan_path)
```

### Core Configuration Files (3 required)
- `decalogo_industrial.json` - 300 questions for evaluation
- `dnp-standards.latest.clean.json` - DNP standards
- `RUBRIC_SCORING.json` - Scoring system and weights

### Core Code Files
- `miniminimoon_orchestrator.py` - Canonical orchestrator (ONLY entry point)
- `unified_evaluation_pipeline.py` - Unified facade with pre/post validation
- `answer_assembler.py` - Final answer assembler
- `evidence_registry.py` - Single evidence registry
- `system_validators.py` - Pre/post execution validators

## Code Style and Conventions

### Python Style
- Follows PEP 8 conventions with Python 3.10 features
- Use Python 3.10 specific typing: `list[str]` instead of `List[str]`
- Comprehensive docstrings with examples
- Type hints using modern Python 3.10 syntax
- No `SystemExit` calls - graceful error handling with degraded mode fallback

### Testing Requirements
```bash
# Lint
python -m py_compile <module>.py

# Test
python -m pytest test_<module>.py -v

# Integration test
python test_determinism_verifier_integration.py
```

### Atomic File Operations
Always use atomic file operations for safety:
```python
import tempfile
import os

# Write to temporary file first, then rename
with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.path.dirname(path)) as tmp:
    json.dump(data, tmp, indent=2, ensure_ascii=False)
    tmp_path = tmp.name
os.replace(tmp_path, path)  # Atomic rename
```

## CI/CD Pipeline

The system uses a comprehensive CI/CD pipeline:

```yaml
on: [pull_request]
jobs:
  validate:
    - freeze_configuration          # Gate #1
    - verify_pre_execution          # System validators
    - run_evaluation_triple         # 3-run reproducibility test
    - verify_post_execution         # System validators
    - rubric_check                  # Gate #5
    - trace_matrix_generation       # Provenance tracking
    - performance_gate              # p95 latency < budget + 10%
```

### Performance Requirements
- Contract validation caching: <5ms
- Mathematical invariant optimizations: 43% improvement
- Budget monotonicity: <0.15ms
- CI/CD performance gate: Blocks PRs exceeding budget >10%

## Common Tasks

### Adding a New Pipeline Component

1. **Never** modify pipeline order without updating `tools/flow_doc.json`
2. Implement deterministic processing (fixed seeds, no timestamps)
3. Define clear I/O contract with type hints
4. Add comprehensive tests with reproducibility checks
5. Update `FLUJOS_CRITICOS_GARANTIZADOS.md` with new flow
6. Run all gates before committing:
   ```bash
   python verify_critical_flows.py
   python verify_reproducibility.py
   python rubric_check.py
   ```

### Modifying Existing Components

1. **Never** break determinism (check seeds, timestamps, order)
2. Run 3-run reproducibility test to verify no changes
3. Verify all gates pass
4. Check performance budgets are not exceeded

### Adding Tests

1. Follow existing test structure in `tests/` directory
2. Use pytest for unit tests
3. Include determinism verification in integration tests
4. Test with Python 3.10 exactly

## Documentation References

For detailed information, consult these key documents:

- **`AGENTS.md`** - Comprehensive technical setup and architecture
- **`README.md`** - System overview and usage guide
- **`FLUJOS_CRITICOS_GARANTIZADOS.md`** - 72 critical flows documentation
- **`ARCHITECTURE.md`** - System architecture details
- **`DEPLOYMENT_INFRASTRUCTURE.md`** - Canary deployment, tracing, SLO monitoring
- **`DETERMINISM_VERIFIER.md`** - Reproducibility verification tool
- **`MODULE_AUDIT.md`** - Module inventory and cleanup plan
- **`INSTALACION_COMPLETADA.md`** - Installation and setup guide

## Deployment Infrastructure

The system includes comprehensive deployment infrastructure:

### Canary Deployment
Progressive traffic routing (5%→25%→100%) with automated rollback on:
- Contract violations
- Error rate exceeding 10%
- P95 latency exceeding 500ms

### OpenTelemetry Tracing
Instrumentation for 28 critical flows and 11 pipeline components with context propagation.

### SLO Monitoring
Real-time monitoring with thresholds:
- Availability: 99.5%
- P95 Latency: 200ms
- Error Rate: 0.1%

See `DEPLOYMENT_INFRASTRUCTURE.md` for complete documentation.

## Important Notes

- **Always use Python 3.10** - No exceptions
- **Never break determinism** - All outputs must be reproducible
- **Never modify canonical flow order** without updating flow_doc.json
- **Never import deprecated modules** - System will raise RuntimeError
- **Always verify all 6 gates** before committing
- **Always use atomic file operations** for safety
- **Never use eval()** or other dynamic code execution for security

## Getting Help

If unsure about any changes:
1. Check the relevant documentation in the list above
2. Review existing code patterns in similar modules
3. Run the diagnostic CLI: `python diagnostic_runner.py`
4. Verify changes don't break any of the 6 acceptance gates
