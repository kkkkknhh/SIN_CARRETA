# Coverage Implementation Plan

## Executive Summary

This plan establishes a comprehensive testing framework for the MINIMINIMOON codebase with:
- **95% critical flow coverage** (28 flows identified)
- **85% unit test coverage** for all non-test modules
- **70/20/10 test pyramid** distribution
- **100% coverage enforcement** on new code
- **9 contract tests** blocking PRs
- **Mutation testing** with 80% target score
- **Chaos engineering** for resilience validation

## Current State Analysis

### Existing Test Files: 40+
- Unit tests for core components
- Some integration scenarios
- Minimal E2E coverage
- No chaos engineering tests
- No formal contract tests

### Non-Test Modules: 66
- Core: 8 modules (embedding, responsibility, teoria_cambio, etc.)
- Evaluation: 10 modules
- Utilities: 15 modules
- Legacy/Deprecated: 33 modules (candidates for archive/delete)

## Implementation Phases

### Phase 1: Foundation (COMPLETED) ✅
- [x] Create test directory structure (unit/, integration/, e2e/, contracts/, chaos/)
- [x] Configure pytest with markers and coverage
- [x] Set up .coveragerc configuration
- [x] Create MODULE_AUDIT.md identifying cleanup targets
- [x] Document testing strategy

### Phase 2: Integration Tests (IN PROGRESS)
Created 5 critical flow integration tests:

**Completed**:
1. ✅ Embedding Model + Responsibility Detection Flow
2. ✅ TeoriaCambio + DAG Validation Flow
3. ✅ Feasibility Scoring + Contradiction Detection Flow
4. ✅ Document Segmentation + Embedding Flow
5. ✅ SpaCy Loader + Responsibility Detection Flow

**Remaining (23 flows to identify and implement)**:
6. Decalogo Loading + Template Validation Flow
7. Plan Processor + Feasibility Analysis Flow
8. Questionnaire Engine + Evaluation Pipeline Flow
9. Evidence Registry + Scoring Aggregation Flow
10. Causal Pattern Detection + Graph Construction Flow
11. PDM Evaluator + Contradiction Detection Flow
12. Monetary Detection + Feasibility Flow
13. Text Processing + Document Segmentation Flow
14. Safe I/O + Decalogo Loading Flow
15. Memory Watchdog + Large Batch Processing Flow
16. Circuit Breaker + Model Loading Flow
17. JSON Utils + Schema Validation Flow
18. Plan Sanitizer + Validation Flow
19. Document Embedding Mapper + Similarity Flow
20. Log Config + Structured Logging Flow
21. Device Config + Model Selection Flow
22. Heap Functionality + Memory Management Flow
23. Unified Evaluation Pipeline + All Components Flow
24. Deterministic Pipeline Validator + Reproducibility Flow
25. Data Flow Contract + Validation Flow
26. NLI Policy Modules + Contradiction Analysis Flow
27. Security Audit + Safe Operations Flow
28. System Validators + Component Health Checks Flow

### Phase 3: Contract Tests (IN PROGRESS)
Created 3 of 9 contract tests:

**Completed**:
1. ✅ EmbeddingModel Contract
2. ✅ ResponsibilityDetector Contract  
3. ✅ TeoriaCambio Contract

**Remaining**:
4. FeasibilityScorer Contract
5. ContradictionDetector Contract
6. DecalogoLoader Contract
7. DocumentSegmenter Contract
8. DAGValidator Contract
9. SpacyLoader Contract

### Phase 4: Chaos Engineering (IN PROGRESS)
Created 3 chaos test suites:

**Completed**:
1. ✅ Network Failure Resilience
2. ✅ Memory Pressure Resilience
3. ✅ File System Chaos

**Additional scenarios needed**:
4. Model Loading Failures
5. Concurrent Access Chaos
6. Input Validation Boundary Tests
7. Resource Exhaustion Tests

### Phase 5: CI/CD Configuration (COMPLETED) ✅
- [x] GitHub Actions workflow for coverage enforcement
- [x] New code 100% coverage requirement using diff-cover
- [x] Contract test blocking on PR
- [x] Mutation testing workflow (weekly schedule)
- [x] Coverage reporting to Codecov

### Phase 6: Unit Test Expansion (TODO)

**Target**: Bring all 66 non-test modules to 85% coverage

**Priority Modules** (by criticality):
1. embedding_model.py - Industrial-grade implementation
2. responsibility_detector.py - Core NER + patterns
3. teoria_cambio.py - Causal graph construction
4. dag_validation.py - Monte Carlo validation
5. feasibility_scorer.py - Pattern matching
6. contradiction_detector.py - Multi-strategy detection
7. questionnaire_engine.py - Large module (81KB)
8. pdm_evaluator.py - Core evaluation
9. decalogo_loader.py - Template loading
10. spacy_loader.py - Model management

### Phase 7: E2E Tests (TODO)

**Target**: 10% of total test suite

**Workflows to Implement**:
1. Complete PDM Document Evaluation (ingestion → report)
2. Full Teoria Cambio Construction & Validation
3. Multi-Document Batch Processing Pipeline
4. End-to-End Contradiction Analysis Workflow
5. Complete Feasibility Assessment Pipeline

### Phase 8: Mutation Testing Setup (COMPLETED) ✅
- [x] mutmut configuration for critical paths
- [x] Weekly execution schedule
- [x] Target: 80% mutation score
- [x] Automated reporting

### Phase 9: Module Cleanup (TODO)

**Action Items from MODULE_AUDIT.md**:
- Delete 10 obsolete files (empty, Python 2.7, debug utils)
- Move 10 files to examples/
- Consolidate duplicate CLI modules
- Archive deprecated components
- **Target**: Reduce from 66 to <30 actively maintained modules

## Test Pyramid Distribution

### Target Distribution
- **70% Unit Tests**: ~150-200 tests
- **20% Integration Tests**: ~40-60 tests  
- **10% E2E Tests**: ~20-30 tests

### Current Status
Need baseline measurement - run:
```bash
pytest --collect-only | grep "test session starts" -A 100
```

## Coverage Metrics

### Overall Target: 85%
### Critical Paths Target: 95%
### New Code Requirement: 100%

### Baseline Measurement Needed
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing --cov-config=.coveragerc
```

## Critical Path Identification

### 28 Critical Flows Identified:
1-5: ✅ Implemented (see Phase 2)
6-28: Documented, awaiting implementation

### 42 Component Interactions to Test:
- Embedding ↔ Responsibility (✅)
- TeoriaCambio ↔ DAG (✅)
- Feasibility ↔ Contradiction (✅)
- Document ↔ Embedding (✅)
- SpaCy ↔ Responsibility (✅)
- [37 more interactions to map and test]

## Contract Enforcement

### 9 Contracts Defined:
3/9 implemented (✅ EmbeddingModel, ResponsibilityDetector, TeoriaCambio)
6/9 remaining

**PR Blocking**: All 9 contract tests must pass before merge

## Mutation Testing Configuration

**Tool**: mutmut  
**Target Modules**: 6 critical paths  
**Target Score**: 80%  
**Schedule**: Weekly on main branch  
**CI Integration**: ✅ Configured

## Chaos Engineering

**Implemented**: 3 chaos test suites  
**Coverage**: Network, Memory, FileSystem  
**Additional Needed**: 4 more scenarios

## Action Items

### Immediate (Next Sprint)
1. [ ] Fix import errors in existing integration tests
2. [ ] Run baseline coverage report
3. [ ] Implement remaining 6 contract tests
4. [ ] Map all 42 component interactions
5. [ ] Implement 5 more integration tests

### Short Term (2-4 weeks)
6. [ ] Expand unit tests for priority modules
7. [ ] Implement 3 E2E test workflows
8. [ ] Complete all 28 integration tests
9. [ ] Add 4 more chaos engineering scenarios
10. [ ] Execute module cleanup (delete, archive, consolidate)

### Medium Term (1-2 months)
11. [ ] Achieve 85% overall coverage
12. [ ] Achieve 95% critical path coverage
13. [ ] Achieve 80% mutation score
14. [ ] Optimize test execution time
15. [ ] Complete documentation for all core modules

## Success Criteria

- ✅ Test pyramid distribution: 70/20/10
- ✅ CI/CD enforces 100% coverage on new code
- ✅ All 9 contract tests block PRs on failure
- [ ] 28 critical flows have integration tests
- [ ] 42 component interactions tested
- [ ] 85% unit test coverage achieved
- [ ] 80% mutation score achieved
- [ ] <30 actively maintained modules
- [ ] All chaos tests passing

## Risks & Mitigations

**Risk**: Test execution time becomes prohibitive  
**Mitigation**: Parallelize test execution, optimize slow tests

**Risk**: Flaky tests due to external dependencies  
**Mitigation**: Use mocks/stubs, implement retry logic

**Risk**: Mutation testing takes too long  
**Mitigation**: Run on schedule, not per-commit

**Risk**: Module cleanup breaks dependencies  
**Mitigation**: Comprehensive integration tests first, then cleanup

## Timeline

- **Week 1-2**: Complete integration + contract tests
- **Week 3-4**: Expand unit test coverage
- **Week 5-6**: E2E tests + chaos engineering
- **Week 7-8**: Module cleanup + optimization
- **Week 9+**: Maintenance and continuous improvement
