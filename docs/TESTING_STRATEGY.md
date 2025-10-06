# MINIMINIMOON Testing Strategy

## Overview

Comprehensive testing strategy implementing a 70-20-10 testing pyramid:
- **70% Unit Tests**: Fast, isolated tests of individual functions/classes
- **20% Integration Tests**: Component interaction testing
- **10% E2E Tests**: Full system workflow validation

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests targeting single functions or classes.

**Coverage Target**: 85% line coverage for all non-test modules

**Key Areas**:
- Embedding model initialization and encoding
- Responsibility detection patterns
- DAG validation algorithms
- Feasibility scoring logic
- Contradiction detection patterns
- Text processing utilities

### Integration Tests (`tests/integration/`)
Multi-component workflow tests.

**Coverage Target**: 95% of critical flows

**Critical Flows Identified**:
1. Embedding Model + Responsibility Detection
2. TeoriaCambio + DAG Validation
3. Feasibility Scoring + Contradiction Detection
4. Document Segmentation + Embedding Generation
5. SpaCy Loader + Responsibility Detection
6. Decalogo Loading + Template Validation
7. Plan Processor + Feasibility Analysis
8. Questionnaire Engine + Evaluation Pipeline
9. Evidence Registry + Scoring Aggregation
10. Causal Pattern Detection + Graph Construction
11. [... additional flows to be documented]

### E2E Tests (`tests/e2e/`)
Full system tests from document ingestion to report generation.

**Coverage Target**: 10% of test suite

**Workflows**:
- Complete PDM evaluation workflow
- Full teoria cambio construction and validation
- End-to-end document processing pipeline

### Contract Tests (`tests/contracts/`)
Interface contract validation to ensure API stability.

**Contracts Tested**:
1. EmbeddingModel interface
2. ResponsibilityDetector interface
3. TeoriaCambio interface
4. DAGValidator interface
5. FeasibilityScorer interface
6. ContradictionDetector interface
7. DecalogoLoader interface
8. SpacyLoader interface
9. DocumentSegmenter interface

**Contract Enforcement**: All contract tests must pass for PRs to merge.

### Chaos Engineering Tests (`tests/chaos/`)
Resilience validation under failure conditions.

**Fault Injection Scenarios**:
- Network failures (timeouts, connection errors)
- File system errors (permissions, disk full, corruption)
- Memory pressure (OOM conditions)
- Model loading failures
- Invalid input handling

## Coverage Enforcement

### Current Code
- **Minimum Coverage**: 70% overall
- **Target Coverage**: 85% for core modules
- **Branch Coverage**: Required for critical paths

### New Code (PRs)
- **Required Coverage**: 100% for all new code
- **CI/CD Enforcement**: PRs blocked if new code coverage < 100%
- **Tool**: diff-cover

## Mutation Testing

**Tool**: mutmut

**Target Modules** (Critical Paths):
- embedding_model.py
- responsibility_detector.py
- teoria_cambio.py
- dag_validation.py
- feasibility_scorer.py
- contradiction_detector.py

**Target Mutation Score**: 80%

**Schedule**: Weekly on main branch

## Test Execution

### Local Development
```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term-missing

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m contract
pytest -m chaos

# Run critical path tests only
pytest -m critical_path

# Skip slow tests
pytest -m "not slow"
```

### CI/CD Pipeline

**On Pull Request**:
1. Run unit + integration tests (exclude slow, chaos)
2. Generate coverage report
3. Check minimum 70% overall coverage
4. Enforce 100% coverage on new code
5. Run contract tests (blocking)
6. Report results as PR comment

**On Merge to Main**:
1. Run full test suite (including E2E)
2. Generate comprehensive coverage report
3. Upload to Codecov
4. Archive test artifacts

**Weekly Scheduled**:
1. Run mutation testing
2. Run chaos engineering tests
3. Generate resilience report

## Test Organization

```
tests/
├── unit/                  # 70% of tests
│   ├── test_embedding_*.py
│   ├── test_responsibility_*.py
│   └── ...
├── integration/           # 20% of tests
│   ├── test_*_flow.py
│   └── ...
├── e2e/                   # 10% of tests
│   ├── test_pdm_evaluation_e2e.py
│   └── ...
├── contracts/             # API contracts
│   ├── test_*_contract.py
│   └── ...
├── chaos/                 # Resilience tests
│   ├── test_*_resilience.py
│   └── ...
└── conftest.py           # Shared fixtures
```

## Test Data

**Location**: `tests/fixtures/`

**Categories**:
- Sample PDM documents
- Test embeddings
- Mock decalogos
- Causal graph examples
- Contradiction patterns

## Continuous Improvement

### Metrics Tracked
- Code coverage percentage
- Mutation score
- Test execution time
- Flaky test rate
- Contract compliance

### Review Cadence
- **Weekly**: Review coverage trends
- **Monthly**: Analyze slow tests for optimization
- **Quarterly**: Audit test pyramid distribution

## Future Enhancements

1. Property-based testing with Hypothesis
2. Performance benchmarking tests
3. Snapshot testing for reports
4. Visual regression testing for charts
5. Load testing for batch processing
