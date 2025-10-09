# AGENTS.md

## Python Version Requirement

**IMPORTANT: This system requires Python 3.10 exactly.** Other versions are not supported due to:
- NumPy compatibility requirements (>=1.21.0 for Python 3.10 support)
- Dependency version constraints for embedding models
- OpenTelemetry instrumentation compatibility

### Version Validation

```bash
# Check Python version and compatibility
python3.10 --version  # Should show Python 3.10.x
python3.10 python_310_compatibility_checker.py  # Run full compatibility check
```

## Commands

### Setup

```bash
# Ensure Python 3.10 is used - CRITICAL for NumPy compatibility
python3.10 -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS  
source venv/bin/activate

# Verify Python 3.10 is active
python --version  # Must show Python 3.10.x

# Install dependencies with Python 3.10 compatible versions
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download spaCy Spanish model for responsibility detection
python -m spacy download es_core_news_sm

# Run compatibility check
python python_310_compatibility_checker.py
```

### Build

```bash
# Version validation first
python -c "from version_validator import validate_python_310; validate_python_310(); print('✓ Python 3.10 validated')"

# Text processing components
python -c "import text_processor, utils; print('Text processing build successful')" || echo "Text processing components not available"
# DAG validation components  
python -c "import dag_validation; print('DAG validation build successful')" || echo "DAG validation components not available"
# Embedding model components with NumPy compatibility check
python -c "from version_validator import validate_numpy_compatibility; validate_numpy_compatibility(); import embedding_model; print('Embedding model build successful')"
# TeoriaCambio class
python -m py_compile teoria_cambio.py
# Responsibility detection components
python -m py_compile responsibility_detector.py
# DECALOGO_INDUSTRIAL loader components
python -m py_compile decalogo_loader.py
# SpaCy model loader components
python -m py_compile spacy_loader.py
# Deployment infrastructure components
python -m py_compile canary_deployment.py opentelemetry_instrumentation.py slo_monitoring.py
# Determinism verifier components
python -m py_compile determinism_verifier.py
```

### Lint

```bash
# Version validation
python -c "from version_validator import validate_python_310; validate_python_310()"

# Code follows PEP 8 conventions with Python 3.10 features
python -m py_compile embedding_model.py test_embedding_model.py example_usage.py version_validator.py python_310_compatibility_checker.py
# Additional components if available
python -m py_compile text_processor.py utils.py test_unicode_normalization.py demo_unicode_comparison.py dag_validation.py test_dag_validation.py verify_reproducibility.py validate.py 2>/dev/null || echo "Additional components not available for linting"
# TeoriaCambio class
python -m py_compile teoria_cambio.py
# Responsibility detection components  
python -m py_compile responsibility_detector.py test_responsibility_detector.py
# DECALOGO_INDUSTRIAL loader components
python -m py_compile decalogo_loader.py test_decalogo_loader.py
# SpaCy model loader components
python -m py_compile spacy_loader.py test_spacy_loader.py
# Deployment infrastructure components
python -m py_compile canary_deployment.py opentelemetry_instrumentation.py slo_monitoring.py
python -m py_compile test_canary_deployment.py test_opentelemetry_instrumentation.py test_slo_monitoring.py
# Determinism verifier components
python -m py_compile determinism_verifier.py test_determinism_verifier.py test_determinism_verifier_integration.py
```

### Inspections

```bash
# Run all code quality inspections (bytecode, flake8, mypy, ruff)
python run_inspections.py

# Individual inspections
python -m compileall -q .  # Bytecode compilation
flake8 .                    # PEP 8 style checking
mypy . --config-file pyproject.toml  # Type checking
ruff check .                # Fast linting (optional)
```

### Test

```bash
# Compatibility check before tests
python python_310_compatibility_checker.py

# Embedding model tests with NumPy validation
python -m pytest test_embedding_model.py -v
# Responsibility detection tests
python -m pytest test_responsibility_detector.py -v
# DECALOGO_INDUSTRIAL loader tests
python -m pytest test_decalogo_loader.py -v
# SpaCy model loader tests
python -m pytest test_spacy_loader.py -v
# Deployment infrastructure tests
python -m pytest test_canary_deployment.py -v
python -m pytest test_opentelemetry_instrumentation.py -v
python -m pytest test_slo_monitoring.py -v
# Determinism verifier tests
python test_determinism_verifier_integration.py
# Version validation tests
python -m pytest test_version_validator.py -v
# Additional tests if available
python test_unicode_normalization.py 2>/dev/null || echo "Text processing tests not available"
python test_dag_validation.py 2>/dev/null || echo "DAG validation tests not available"
python validate.py 2>/dev/null || echo "Full validation suite not available"
```

### Dev Server

```bash
# Compatibility validation
python python_310_compatibility_checker.py

# Embedding model demo
python example_usage.py
# Responsibility detection demo
python responsibility_detector.py
# DECALOGO_INDUSTRIAL loader demo
python -c "from decalogo_loader import get_decalogo_industrial; print(get_decalogo_industrial())"
# SpaCy model loader demo
python spacy_loader.py
# Deployment infrastructure demos
python deployment_example.py
python test_deployment_integration.py
# Determinism verifier demo
# NOTE: Requires an actual PDF input file - replace with your test PDF
# python determinism_verifier.py test_fixtures/sample_plan.pdf
echo "Determinism verifier requires PDF input: python determinism_verifier.py <input_pdf>"
# Additional demos if available
python demo_unicode_comparison.py 2>/dev/null || echo "Text processing demo not available"
python dag_validation.py 2>/dev/null || echo "DAG validation demo not available"
```

## Tech Stack

- **Language**: Python 3.10 (REQUIRED - exact version for dependency compatibility)
- **Framework**: sentence-transformers, scikit-learn, numpy for embedding models; spaCy for NER; standard library for
  text processing and atomic file operations
- **Package Manager**: pip with Python 3.10 compatible versions
- **Testing**: pytest and unittest
- **NumPy**: >=1.21.0,<1.25.0 (Python 3.10 compatible range)

## Architecture

```
# Version Validation Components
version_validator.py           # Python 3.10 enforcement and NumPy compatibility validation
├── validate_python_310()     # Strict Python 3.10 requirement
├── validate_numpy_compatibility() # NumPy version compatibility for Python 3.10
├── get_python_version_info()  # Version information utility
└── Automatic validation      # Validates on import (can be disabled)

python_310_compatibility_checker.py # Comprehensive compatibility testing
├── Python310CompatibilityChecker # Main checker class
├── ImportResult              # Import test result structure
├── CRITICAL_MODULES          # Essential dependencies (NumPy, scipy, torch, etc.)
├── INCOMPATIBLE_PATTERNS     # Known version compatibility issues
├── check_numpy_compatibility() # Special NumPy testing for Python 3.10
├── run_compatibility_check() # Full system compatibility validation
└── generate_report()         # Human-readable compatibility report

# Embedding Model Components
embedding_model.py              # Core embedding model with fallback mechanism
├── EmbeddingModel             # Main model class with MPNet->MiniLM fallback
├── create_embedding_model()   # Factory function
├── exception handling         # Robust error handling for model loading
└── batch optimization         # Model-specific batch size optimization

test_embedding_model.py         # Comprehensive test suite for embedding model
example_usage.py               # Demo and usage examples

# Responsibility Detection Components
responsibility_detector.py      # spaCy NER + lexical pattern matching for responsibility detection
├── ResponsibilityDetector     # Main detector class
├── ResponsibilityEntity       # Entity data structure
├── EntityType                 # Entity type enumeration
├── NER integration            # spaCy PERSON/ORG detection
├── Government patterns        # High-priority institutional patterns
├── Position patterns          # Official role detection
├── Institutional patterns     # Generic fallback patterns
├── Entity merging            # Overlap handling
└── Confidence scoring        # Hierarchical scoring system

test_responsibility_detector.py # Comprehensive test suite for responsibility detection

# DECALOGO_INDUSTRIAL Template Loading Components
decalogo_loader.py             # Atomic file operations with fallback template loading
├── load_decalogo_industrial() # Core loading function with atomic write + fallback
├── get_decalogo_industrial()  # Convenience wrapper with caching
├── DECALOGO_INDUSTRIAL_TEMPLATE # Hardcoded template for fallback
├── Atomic file operations     # Temporary file + rename for safety
├── Exception handling         # PermissionError, OSError, IOError handling
├── Fallback mechanism         # In-memory template on write failures
└── Logging integration        # Debug info for deployment issues

test_decalogo_loader.py        # Comprehensive test suite for template loading

# SpaCy Model Loading Components
spacy_loader.py                # Robust spaCy model loader with automatic download and fallback
├── SpacyModelLoader           # Main loader class with retry logic
├── SafeSpacyProcessor         # Example processor with graceful degradation
├── Automatic model download   # Uses spacy.cli.download with retry logic
├── Degraded mode fallback     # Basic text processing when models unavailable
├── Model caching             # Prevents redundant loading
├── Error handling            # No SystemExit calls, graceful error handling
└── Logging integration       # Comprehensive error and warning logging

test_spacy_loader.py           # Comprehensive test suite for spaCy model loading

# Text Processing Components (if available)
text_processor.py              # Core text processing with Unicode normalization
utils.py                       # Utility classes and functions
test_unicode_normalization.py  # Text processing test suite
demo_unicode_comparison.py     # Text processing demo

# DAG Validation Components (if available)
dag_validation.py              # Deterministic Monte Carlo DAG validation
test_dag_validation.py         # DAG validation test suite
verify_reproducibility.py      # DAG reproducibility verification
validate.py                    # Complete validation orchestrator

# Theory of Change Components
teoria_cambio.py               # TeoriaCambio class with cached causal graph construction
├── TeoriaCambio               # Main class with caching mechanism
├── construir_grafo_causal()   # Cached graph construction method
├── _crear_grafo_causal()      # Private graph creation method
└── invalidar_cache_grafo()    # Cache invalidation method

# Deployment Infrastructure Components
canary_deployment.py           # Canary deployment with progressive traffic routing
├── CanaryDeploymentController # Main deployment controller
├── TrafficRouter              # Routes traffic between baseline and canary (5%→25%→100%)
├── MetricsCollector           # Real-time metrics collection and analysis
├── RollbackThresholds         # Automated rollback triggers (error rate, latency, contracts)
├── DeploymentResult           # Deployment result tracking
└── create_canary_controller() # Factory function for easy setup

test_canary_deployment.py      # Comprehensive test suite for canary deployment

opentelemetry_instrumentation.py # OpenTelemetry distributed tracing
├── TracingManager             # Manages tracing initialization and span creation
├── FlowType (28 flows)        # Critical flow enumeration (document, evidence, evaluation, validation, infrastructure)
├── ComponentType (11 comps)   # Pipeline component enumeration
├── @trace_flow decorator      # Decorator for tracing critical flows
├── @trace_component decorator # Decorator for tracing pipeline components
├── SpanLogger                 # Logger with trace context correlation (Phase 0 integration)
└── Context propagation        # Inject/extract trace context across service boundaries

test_opentelemetry_instrumentation.py # Comprehensive test suite for OpenTelemetry

slo_monitoring.py              # SLO monitoring and alerting system
├── SLOMonitor                 # Main monitoring class
├── MetricsAggregator          # Aggregates metrics across time windows
├── SLOThresholds              # Threshold configuration (99.5% availability, 200ms p95, 0.1% error)
├── AlertType                  # Contract violations, performance regressions, fault recovery
├── DashboardDataGenerator     # Generates dashboard data with visual indicators
├── FlowMetrics                # Per-flow metrics (availability, latency, error rate)
└── create_slo_monitor()       # Factory function with phase integration

test_slo_monitoring.py         # Comprehensive test suite for SLO monitoring

# Integration Components
test_deployment_integration.py # Integration test for canary + tracing + monitoring
deployment_example.py          # Example deployment with monitoring

# Determinism Verifier Components
determinism_verifier.py        # Standalone utility for reproducibility verification
├── DeterminismVerifier        # Main verifier class with dual-run orchestration
├── ArtifactComparison         # Comparison result for single artifact
├── DeterminismReport          # Complete verification report
├── JSON normalization         # Sorted key normalization with non-deterministic field removal
├── SHA-256 hashing            # Artifact and evidence state hashing
├── Byte-level comparison      # Precise artifact matching
├── Line-level diffs           # Human-readable discrepancy reports
├── Exit codes                 # 0=perfect, 4=violations, 1=errors
└── Forensic preservation      # Both run outputs saved for analysis

test_determinism_verifier.py   # Comprehensive test suite for verifier
test_determinism_verifier_integration.py # Integration tests with mock orchestrator
example_determinism_check.sh   # Example shell script for CI/CD integration
```

## Code Style

- Follows PEP 8 Python conventions with Python 3.10 features
- Uses Python 3.10 specific typing improvements (e.g., `list[str]` instead of `List[str]`)
- Comprehensive docstrings with examples
- Type hints using modern Python 3.10 syntax
- Robust exception handling for model initialization
- Automatic fallback mechanisms for reliability
- Statistical interpretation warnings to prevent misuse
- Atomic file operations for deployment safety
- No SystemExit calls - graceful error handling with degraded mode fallback
- **Strict Python 3.10 version validation** in all critical modules

## Python 3.10 Compatibility Notes

### NumPy Compatibility
- **Required**: NumPy >= 1.21.0 for Python 3.10 support
- **Recommended**: NumPy < 1.25.0 to avoid breaking changes
- **Critical**: Matrix operations and random number generation APIs

### Key Dependencies
- **PyTorch**: >= 1.12.0 for Python 3.10 support
- **scikit-learn**: >= 1.0.0 for modern API compatibility
- **Transformers**: >= 4.20.0 for recent model support
- **spaCy**: >= 3.4.0 for Python 3.10 compatibility

### Validation Commands
```bash
# Quick version check
python --version  # Must show Python 3.10.x

# Full compatibility validation
python python_310_compatibility_checker.py

# Import version validator in your modules
from version_validator import validate_python_310
validate_python_310()  # Raises RuntimeError if not Python 3.10
```

## Deployment Infrastructure

The system includes comprehensive canary deployment, distributed tracing, and SLO monitoring infrastructure with **Python 3.10 enforcement**:

### Key Features

1. **Canary Deployment** - Progressive traffic routing (5%→25%→100%) with automated rollback on:
   - Contract violations (Phase 6 integration)
   - Error rate exceeding 10%
   - P95 latency exceeding 500ms

2. **OpenTelemetry Tracing** - Instrumentation for:
   - 28 critical flows (document processing, evidence extraction, evaluation, validation, infrastructure)
   - 11 pipeline components (segmenter, embedding model, detectors, evaluators, orchestrator)
   - Context propagation across service boundaries
   - Trace ID correlation with Phase 0 structured logging

3. **SLO Monitoring** - Real-time monitoring with thresholds:
   - Availability: 99.5%
   - P95 Latency: 200ms
   - Error Rate: 0.1%
   - Performance regression: 10% (Phase 3 integration)
   - Fault recovery: 1.5s p99 (Phase 4 integration)

### Usage Example

```python
from canary_deployment import create_canary_controller
from slo_monitoring import create_slo_monitor
from opentelemetry_instrumentation import initialize_tracing, trace_flow, FlowType

# Initialize tracing
initialize_tracing(service_name="decalogo-evaluation-system")

# Create SLO monitor
slo_monitor = create_slo_monitor()

# Create canary controller
controller = create_canary_controller("deployment-v2.0")

# Trace critical flows
@trace_flow(FlowType.DECALOGO_EVALUATION)
def evaluate_plan(plan_text):
    # Evaluation logic
    return results

# Execute deployment with monitoring
result = controller.execute_deployment(request_generator)
```

See `DEPLOYMENT_INFRASTRUCTURE.md` for complete documentation.
