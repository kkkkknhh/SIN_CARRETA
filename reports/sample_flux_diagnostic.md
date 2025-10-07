# Pipeline Flux Diagnostic Report
**Generated:** 2025-10-07 12:23:09
**Source:** `reports/sample_flux_diagnostic.json`

---

## Executive Summary

**Pipeline Health:** HEALTHY  
**Timestamp:** 2025-10-07 12:23:09

The evaluation pipeline processed 15 stages with 13/15 passing nodes and 0 failures. Inter-node connectivity shows 14/14 stable data flows (≥95% reliability).

**Output Quality:** ✓ Deterministic execution, 300/300 questions covered, ✓ rubric alignment verified, ✓ all acceptance gates passed.

**Critical Findings:** All systems nominal. No critical issues detected.
---

## Node-by-Node Performance Analysis

| Stage | Latency | Peak Memory | Throughput | Status | Notes |
|-------|---------|-------------|------------|--------|-------|
| answer_assembler | 234.8 ms | 156.7 MB | 42.6 items/s | ✓ PASS | Nominal |
| causal_pattern_detector | 789.2 ms | 567.8 MB | 12.7 items/s | ✓ PASS | Nominal |
| contradiction_detector | 567.3 ms | 345.2 MB | 17.6 items/s | ✓ PASS | Nominal |
| dag_validator | 345.6 ms | 234.1 MB | 28.9 items/s | ✓ PASS | Nominal |
| document_segmenter | 890.5 ms | 412.7 MB | 11.2 items/s | ✓ PASS | Nominal |
| embedding_model | 1.52 s | 1.00 GB | 6.6 items/s | ⚠ WARN | High memory: 1025 MB |
| feasibility_scorer | 456.7 ms | 234.5 MB | 21.9 items/s | ✓ PASS | Nominal |
| gate_validator | 123.4 ms | 89.2 MB | 81.0 items/s | ✓ PASS | Nominal |
| monetary_detector | 123.6 ms | 98.4 MB | 81.0 items/s | ✓ PASS | Nominal |
| plan_processor | 120.8 ms | 256.3 MB | 82.3 items/s | ✓ PASS | Nominal |
| plan_sanitizer | 45.2 ms | 89.5 MB | 220.5 items/s | ✓ PASS | Nominal |
| questionnaire_engine | 2.35 s | 890.3 MB | 4.3 items/s | ⚠ WARN | High latency: 2.35 s |
| responsibility_detector | 234.1 ms | 178.9 MB | 42.7 items/s | ✓ PASS | Nominal |
| rubric_scorer | 567.9 ms | 312.4 MB | 17.6 items/s | ✓ PASS | Nominal |
| teoria_cambio_validator | 1.23 s | 678.9 MB | 8.1 items/s | ✓ PASS | Nominal |

---

## Inter-Node Connection Assessment

Evaluates data flow stability and type compatibility between pipeline stages.

| Connection | Stability | Verdict | Notes |
|------------|-----------|---------|-------|
| assembler->rubric | 99.8% | ✓ EXCELLENT | No type mismatches detected |
| causal->teoria | 99.4% | ✓ EXCELLENT | No type mismatches detected |
| dag->questionnaire | 98.7% | ✓ GOOD | Type mismatches: question_context |
| embedder->causal | 98.9% | ✓ GOOD | Type mismatches: edge_weights, node_metadata |
| embedder->contradiction | 99.3% | ✓ EXCELLENT | Type mismatches: confidence_score |
| embedder->feasibility | 99.6% | ✓ EXCELLENT | No type mismatches detected |
| embedder->monetary | 99.9% | ✓ EXCELLENT | No type mismatches detected |
| embedder->responsibility | 99.7% | ✓ EXCELLENT | No type mismatches detected |
| processor->segmenter | 99.5% | ✓ EXCELLENT | Type mismatches: field_metadata |
| questionnaire->assembler | 99.6% | ✓ EXCELLENT | No type mismatches detected |
| rubric->gate | 99.9% | ✓ EXCELLENT | No type mismatches detected |
| sanitizer->processor | 99.8% | ✓ EXCELLENT | No type mismatches detected |
| segmenter->embedder | 99.2% | ✓ EXCELLENT | No type mismatches detected |
| teoria->dag | 99.1% | ✓ EXCELLENT | Type mismatches: validation_metadata |

---

## Final Output Quality Assessment

Verifies pipeline output integrity and acceptance criteria.

### Determinism Verification
**Status:** ✓ PASS

Multiple runs produced identical outputs:
- run_1: `a3f8d9c2b1e0f4a6d8c7b5e3f2a1d9c8`
- run_2: `a3f8d9c2b1e0f4a6d8c7b5e3f2a1d9c8`
- run_3: `a3f8d9c2b1e0f4a6d8c7b5e3f2a1d9c8`

### Question Coverage
**Status:** 300/300 questions ✓ COMPLETE

### Rubric Alignment
**Status:** ✓ PASS
**Exit Code:** 0 (tools/rubric_check.py)
1:1 alignment verified between answers and rubric scoring.

### Acceptance Gates
**Status:** ✓ ALL GATES PASSED

All 6 acceptance gates passed successfully.

---

## Top 5 Risks

Critical bottlenecks and failures ranked by severity.

### 1. Performance Warning — MEDIUM
**Location:** `embedding_model`  
**Severity Score:** 50/100  
**Description:** High memory: 1025 MB

### 2. Performance Warning — MEDIUM
**Location:** `questionnaire_engine`  
**Severity Score:** 50/100  
**Description:** High latency: 2.35 s


---

## Top 5 Recommended Fixes

Actionable recommendations to address identified risks.

### 1. Fix Performance Warning
**Target:** `embedding_model`

**Recommended Actions:**
- Investigate embedding_model performance metrics
- Add monitoring and alerting for threshold violations
- Review component configuration and scaling parameters
- Implement graceful degradation fallback

### 2. Fix Performance Warning
**Target:** `questionnaire_engine`

**Recommended Actions:**
- Investigate questionnaire_engine performance metrics
- Add monitoring and alerting for threshold violations
- Review component configuration and scaling parameters
- Implement graceful degradation fallback

