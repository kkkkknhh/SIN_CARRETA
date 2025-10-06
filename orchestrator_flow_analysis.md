# MINIMINIMOON Orchestrator Flow Analysis
## Canonical Flow Integration Assessment

**Analysis Date**: 2024  
**Target File**: `miniminimoon_orchestrator.py`  
**Reference Contract**: `data_flow_contract.py` (11-node canonical order)

---

## Executive Summary

**Status**: ⚠️ **PARTIAL INTEGRATION** - 2 of 5 canonical modules missing

The orchestrator implements a **12-stage pipeline** that includes 9 of the 11 canonical nodes defined in the data flow contract, but is **missing direct imports** for 2 of the 5 specified canonical flow modules:
- ❌ **Decatalogo_principal.py** - NOT imported or invoked
- ❌ **validate_teoria_cambio.py** - NOT imported (functionality implemented inline)

---

## 1. Canonical Module Import Analysis

### Five Specified Canonical Flow Modules

| Module | File Status | Import Status | Invocation Status | Notes |
|--------|-------------|---------------|-------------------|-------|
| **Decatalogo_principal.py** | ✅ Exists | ❌ NOT imported | ❌ NOT invoked | Missing integration |
| **dag_validation.py** | ✅ Exists | ✅ Imported (line 35) | ✅ Invoked (node 11) | Fully integrated |
| **embedding_model.py** | ✅ Exists | ✅ Imported (line 26)* | ✅ Invoked (node 4) | Uses `IndustrialEmbeddingModel` |
| **plan_processor.py** | ✅ Exists | ✅ Imported (line 24) | ✅ Invoked (node 2) | Fully integrated |
| **validate_teoria_cambio.py** | ✅ Exists | ❌ NOT imported | ⚠️ Inline impl. | Logic exists in orchestrator |

\* Note: Orchestrator imports `IndustrialEmbeddingModel` instead of base `EmbeddingModel`

### Actual Imports in Orchestrator

```python
# Core processing components (lines 23-27)
from plan_sanitizer import PlanSanitizer
from plan_processor import PlanProcessor              # ✅ CANONICAL
from document_segmenter import DocumentSegmenter
from embedding_model import IndustrialEmbeddingModel  # ✅ CANONICAL (variant)
from spacy_loader import SpacyModelLoader, SafeSpacyProcessor

# Analysis components (lines 30-36)
from responsibility_detector import ResponsibilityDetector
from contradiction_detector import ContradictionDetector
from monetary_detector import MonetaryDetector
from feasibility_scorer import FeasibilityScorer
from teoria_cambio import TeoriaCambio
from dag_validation import AdvancedDAGValidator       # ✅ CANONICAL
from causal_pattern_detector import PDETCausalPatternDetector

# System components (lines 39-41)
from evidence_registry import EvidenceRegistry
from data_flow_contract import CanonicalFlowValidator
from miniminimoon_immutability import ImmutabilityContract

# Questionnaire engine (line 44)
from questionnaire_engine import QuestionnaireEngine
```

---

## 2. Execution Flow Analysis

### 11-Node Canonical Order (from data_flow_contract.py)

```python
execution_order = [
    "sanitization",              # Node 1
    "plan_processing",           # Node 2  ✅ Uses plan_processor.py
    "document_segmentation",     # Node 3
    "embedding",                 # Node 4  ✅ Uses embedding_model.py
    "responsibility_detection",  # Node 5
    "contradiction_detection",   # Node 6
    "monetary_detection",        # Node 7
    "monetary_detection",        # Node 8
    "causal_detection",          # Node 9
    "teoria_cambio",             # Node 10 ⚠️ validate_teoria_cambio.py not used
    "dag_validation"             # Node 11 ✅ Uses dag_validation.py
]
```

### Actual Orchestrator Execution (12 stages)

```python
# From process_plan() method (lines 285-428)

# 1. Sanitization (Node 1)
sanitized_text = self._execute_sanitization(plan_text)
results["executed_nodes"].append("sanitization")

# 2. Plan Processing (Node 2) ✅ CANONICAL MODULE
processed_plan = self._execute_plan_processing(sanitized_text)
# Invokes: self.processor.process(text) → plan_processor.PlanProcessor

# 3. Document Segmentation (Node 3)
segments = self._execute_segmentation(sanitized_text)
results["executed_nodes"].append("document_segmentation")

# 4. Embedding (Node 4) ✅ CANONICAL MODULE
embeddings = self._execute_embedding(segments)
# Invokes: self.embedding_model.embed(segments) → embedding_model.IndustrialEmbeddingModel

# 5. Responsibility Detection (Node 5)
responsibilities = self._execute_responsibility_detection(sanitized_text)
results["executed_nodes"].append("responsibility_detection")

# 6. Contradiction Detection (Node 6)
contradictions = self._execute_contradiction_detection(sanitized_text)
results["executed_nodes"].append("contradiction_detection")

# 7. Monetary Detection (Node 7)
monetary = self._execute_monetary_detection(sanitized_text)
results["executed_nodes"].append("monetary_detection")

# 8. Feasibility Scoring (Node 8)
feasibility = self._execute_feasibility_scoring(sanitized_text)
results["executed_nodes"].append("feasibility_scoring")

# 9. Causal Pattern Detection (Node 9)
causal_patterns = self._execute_causal_detection(sanitized_text)
results["executed_nodes"].append("causal_detection")

# 10. Theory of Change (Node 10) ⚠️ INLINE IMPLEMENTATION
teoria_cambio = self._create_teoria_cambio(...)
teoria_validation = self._validate_teoria_cambio(teoria_cambio)
# NOTE: validate_teoria_cambio.py NOT used - logic inline (lines 608-632)
results["executed_nodes"].append("teoria_cambio")

# 11. DAG Validation (Node 11) ✅ CANONICAL MODULE
dag_results = {...}  # Uses self.dag_validator
# Invokes: self.dag_validator.is_acyclic() → dag_validation.AdvancedDAGValidator
results["executed_nodes"].append("dag_validation")

# 12. Questionnaire Evaluation (EXTRA NODE - not in canonical 11)
questionnaire_results = self._execute_questionnaire_evaluation(...)
results["executed_nodes"].append("questionnaire_evaluation")
```

---

## 3. Data Flow Trace

### Canonical Data Type Transformations

| Stage | Node | Input DataType | Output DataType | Module Used | Contract Match |
|-------|------|----------------|-----------------|-------------|----------------|
| 1 | sanitization | RAW_TEXT | SANITIZED_TEXT | plan_sanitizer | ✅ |
| 2 | plan_processing | SANITIZED_TEXT | METADATA | **plan_processor** | ✅ CANONICAL |
| 3 | document_segmentation | SANITIZED_TEXT | SEGMENTS | document_segmenter | ✅ |
| 4 | embedding | SEGMENTS | EMBEDDINGS | **embedding_model** | ✅ CANONICAL |
| 5 | responsibility_detection | SANITIZED_TEXT | ENTITIES | responsibility_detector | ✅ |
| 6 | contradiction_detection | SANITIZED_TEXT | CONTRADICTIONS | contradiction_detector | ✅ |
| 7 | monetary_detection | SANITIZED_TEXT | MONETARY_VALUES | monetary_detector | ✅ |
| 8 | feasibility_scoring | SANITIZED_TEXT | FEASIBILITY_SCORES | feasibility_scorer | ✅ |
| 9 | causal_detection | SANITIZED_TEXT | CAUSAL_PATTERNS | causal_pattern_detector | ✅ |
| 10 | teoria_cambio | ENTITIES + CAUSAL_PATTERNS + MONETARY_VALUES | TEORIA_CAMBIO | teoria_cambio ⚠️ | ⚠️ INLINE |
| 11 | dag_validation | TEORIA_CAMBIO | DAG_STRUCTURE | **dag_validation** | ✅ CANONICAL |

### Data Flow Dependencies (Actual)

```
RAW_TEXT (file input)
    │
    ├─[1]─→ SANITIZED_TEXT
    │         │
    │         ├─[2]──→ METADATA (plan_processor.py ✅)
    │         │
    │         ├─[3]──→ SEGMENTS
    │         │         │
    │         │         └─[4]──→ EMBEDDINGS (embedding_model.py ✅)
    │         │
    │         ├─[5]──→ ENTITIES (responsibilities)
    │         │         │
    │         ├─[6]──→ CONTRADICTIONS
    │         │         │
    │         ├─[7]──→ MONETARY_VALUES
    │         │         │
    │         ├─[8]──→ FEASIBILITY_SCORES
    │         │         │
    │         └─[9]──→ CAUSAL_PATTERNS
    │                   │
    │                   ├──[Combined]─→ [10] TEORIA_CAMBIO
    │                   │   (ENTITIES + CAUSAL_PATTERNS + MONETARY_VALUES)
    │                   │   ⚠️ validate_teoria_cambio.py NOT used
    │                   │
    │                   └──────────────→ [11] DAG_VALIDATION
    │                                      (dag_validation.py ✅)
    │
    └─[12]──→ QUESTIONNAIRE_EVALUATION (extra node)
```

---

## 4. Contract Compliance Analysis

### Node Dependency Validation

| Node | Expected Dependencies | Actual Dependencies | Status |
|------|----------------------|---------------------|--------|
| sanitization | [] | [] | ✅ |
| plan_processing | [sanitization] | [sanitization] | ✅ |
| document_segmentation | [sanitization] | [sanitization] | ✅ |
| embedding | [document_segmentation] | [document_segmentation] | ✅ |
| responsibility_detection | [sanitization] | [sanitization] | ✅ |
| contradiction_detection | [sanitization] | [sanitization] | ✅ |
| monetary_detection | [sanitization] | [sanitization] | ✅ |
| feasibility_scoring | [sanitization] | [sanitization] | ✅ |
| causal_detection | [sanitization] | [sanitization] | ✅ |
| teoria_cambio | [responsibility_detection, causal_detection, monetary_detection] | [all present] | ✅ |
| dag_validation | [teoria_cambio] | [teoria_cambio] | ✅ |

**Result**: ✅ **All dependency contracts satisfied**

### Execution Order Validation

**Expected**: `[1→2→3→4→5→6→7→8→9→10→11]`  
**Actual**: `[1→2→3→4→5→6→7→8→9→10→11→12]`

**Findings**:
- ✅ First 11 nodes execute in correct canonical order
- ⚠️ Node 12 (questionnaire_evaluation) is **extra** - not part of canonical 11
- ✅ No sequence violations or out-of-order execution detected

---

## 5. Missing Integration Points

### Critical Gaps

#### ❌ **Gap 1: Decatalogo_principal.py**
- **Status**: File exists but **NOT imported or invoked**
- **Impact**: HIGH - This is one of the five specified canonical modules
- **Location**: Should potentially integrate before/during plan_processing
- **Recommendation**: Determine if Decatalogo_principal should:
  - Replace/augment plan_processor.py functionality
  - Serve as preprocessing before sanitization
  - Provide template/validation logic for plan structure

#### ⚠️ **Gap 2: validate_teoria_cambio.py**
- **Status**: File exists but **NOT imported** (functionality implemented inline)
- **Impact**: MEDIUM - Logic exists but not using canonical module
- **Location**: `_validate_teoria_cambio()` method (lines 608-632)
- **Current Implementation**:
  ```python
  def _validate_teoria_cambio(self, teoria_cambio: TeoriaCambio) -> Dict[str, Any]:
      """Validate a Theory of Change"""
      graph = teoria_cambio.construir_grafo_causal()
      orden_result = teoria_cambio.validar_orden_causal(graph)
      caminos_result = teoria_cambio.detectar_caminos_completos(graph)
      sugerencias_result = teoria_cambio.generar_sugerencias(graph)
      # ... inline validation logic
  ```
- **Recommendation**: Import and delegate to `validate_teoria_cambio.py` if it contains canonical validation logic

---

## 6. Sequence Violations

### Assessment: ✅ **NO VIOLATIONS DETECTED**

All executed nodes follow the canonical dependency order:
1. Sanitization runs first (no dependencies)
2. Plan processing depends on sanitization ✅
3. Segmentation depends on sanitization ✅
4. Embedding depends on segmentation ✅
5. Detection modules (5-9) depend on sanitization ✅
6. Theory of Change depends on detection outputs ✅
7. DAG validation depends on Theory of Change ✅

### Extra Node: questionnaire_evaluation
- **Position**: After dag_validation (node 12)
- **Dependencies**: Uses all orchestrator results
- **Impact**: Not part of canonical 11, but does not violate sequence
- **Status**: ✅ Safe extension (does not break canonical flow)

---

## 7. Dependency Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MINIMINIMOON Orchestrator Flow                    │
│                         (12-Stage Pipeline)                          │
└─────────────────────────────────────────────────────────────────────┘

 INPUT: plan_path (file)
    │
    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ NODE 1: Sanitization                                                  ║
║ ├─ Module: plan_sanitizer.PlanSanitizer                               ║
║ ├─ Input: RAW_TEXT                                                    ║
║ ├─ Output: SANITIZED_TEXT                                             ║
║ └─ Status: ✅ Integrated                                              ║
╚═══════════════════════════════════════════════════════════════════════╝
    │
    ├─────────────────────────────────────────┐
    │                                         │
    ▼                                         │
╔═══════════════════════════════════════╗   │
║ NODE 2: Plan Processing               ║   │
║ ├─ Module: plan_processor             ║   │
║ │  🎯 CANONICAL MODULE                ║   │
║ ├─ Input: SANITIZED_TEXT               ║   │
║ ├─ Output: METADATA                    ║   │
║ └─ Status: ✅ Fully Integrated         ║   │
╚═══════════════════════════════════════╝   │
                                            │
    ┌───────────────────────────────────────┘
    │
    ├──────────────────────┬───────────────────────┬──────────────────────┐
    │                      │                       │                      │
    ▼                      ▼                       ▼                      ▼
╔═══════════════════╗  ╔═══════════════════╗  ╔═══════════════════╗  ╔═══════════════════╗
║ NODE 3:           ║  ║ NODE 5:           ║  ║ NODE 6:           ║  ║ NODE 7:           ║
║ Segmentation      ║  ║ Responsibility    ║  ║ Contradiction     ║  ║ Monetary          ║
║ ─────────────     ║  ║ Detection         ║  ║ Detection         ║  ║ Detection         ║
║ Input:            ║  ║ ─────────────     ║  ║ ─────────────     ║  ║ ─────────────     ║
║ SANITIZED_TEXT    ║  ║ Input:            ║  ║ Input:            ║  ║ Input:            ║
║                   ║  ║ SANITIZED_TEXT    ║  ║ SANITIZED_TEXT    ║  ║ SANITIZED_TEXT    ║
║ Output:           ║  ║                   ║  ║                   ║  ║                   ║
║ SEGMENTS          ║  ║ Output:           ║  ║ Output:           ║  ║ Output:           ║
║                   ║  ║ ENTITIES          ║  ║ CONTRADICTIONS    ║  ║ MONETARY_VALUES   ║
║ Status: ✅        ║  ║                   ║  ║                   ║  ║                   ║
╚═══════════════════╝  ║ Status: ✅        ║  ║ Status: ✅        ║  ║ Status: ✅        ║
    │                  ╚═══════════════════╝  ╚═══════════════════╝  ╚═══════════════════╝
    │                      │                       │                      │
    ▼                      │                       │                      │
╔═══════════════════╗      │                       │                      │
║ NODE 4: Embedding ║      │                       │                      │
║ ───────────────── ║      │                       │                      │
║ Module:           ║      │                       │                      │
║ embedding_model   ║      │                       │                      │
║ 🎯 CANONICAL      ║      │                       │                      │
║                   ║      │                       │                      │
║ Input: SEGMENTS   ║      │                       │                      │
║ Output:           ║      │                       │                      │
║ EMBEDDINGS        ║      │                       │                      │
║                   ║      │                       │                      │
║ Status: ✅        ║      │                       │                      │
╚═══════════════════╝      │                       │                      │
                           │                       │                      │
    ┌──────────────────────┴───────────────────────┴──────────────────────┘
    │                      ▼
    │                  ╔═══════════════════╗
    │                  ║ NODE 8:           ║
    │                  ║ Feasibility       ║
    │                  ║ Scoring           ║
    │                  ║ ─────────────     ║
    │                  ║ Input:            ║
    │                  ║ SANITIZED_TEXT    ║
    │                  ║                   ║
    │                  ║ Output:           ║
    │                  ║ FEASIBILITY       ║
    │                  ║ _SCORES           ║
    │                  ║                   ║
    │                  ║ Status: ✅        ║
    │                  ╚═══════════════════╝
    │                      │
    ▼                      ▼
╔═══════════════════╗  ╔═══════════════════╗
║ NODE 9:           ║  ║                   ║
║ Causal Pattern    ║  ║                   ║
║ Detection         ║  ║                   ║
║ ─────────────     ║  ║                   ║
║ Input:            ║  ║                   ║
║ SANITIZED_TEXT    ║  ║                   ║
║                   ║  ║                   ║
║ Output:           ║  ║                   ║
║ CAUSAL_PATTERNS   ║  ║                   ║
║                   ║  ║                   ║
║ Status: ✅        ║  ║                   ║
╚═══════════════════╝  ╚═══════════════════╝
    │
    └──────────────────┐
                       │
    ┌──────────────────┴────────────────────────────────────┐
    │                                                        │
    ▼                                                        │
╔══════════════════════════════════════════════════════════════╗
║ NODE 10: Theory of Change Creation & Validation             ║
║ ──────────────────────────────────────────────────────────   ║
║ Module: teoria_cambio.TeoriaCambio                           ║
║ ⚠️  validate_teoria_cambio.py NOT USED (inline logic)       ║
║                                                              ║
║ Inputs (Combined):                                           ║
║   • ENTITIES (from Node 5)                                   ║
║   • CAUSAL_PATTERNS (from Node 9)                            ║
║   • MONETARY_VALUES (from Node 7)                            ║
║                                                              ║
║ Output: TEORIA_CAMBIO                                        ║
║                                                              ║
║ Methods:                                                     ║
║   • _create_teoria_cambio() (line 580)                       ║
║   • _validate_teoria_cambio() (line 608)                     ║
║                                                              ║
║ Status: ⚠️  Partial Integration (missing canonical module)   ║
╚══════════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════════╗
║ NODE 11: DAG Validation                                      ║
║ ──────────────────────────────────────────────────────────   ║
║ Module: dag_validation.AdvancedDAGValidator                  ║
║ 🎯 CANONICAL MODULE                                          ║
║                                                              ║
║ Input: TEORIA_CAMBIO                                         ║
║ Output: DAG_STRUCTURE                                        ║
║                                                              ║
║ Methods:                                                     ║
║   • _build_dag_from_teoria_cambio() (line 634)               ║
║   • is_acyclic()                                             ║
║   • calculate_acyclicity_pvalue()                            ║
║                                                              ║
║ Status: ✅ Fully Integrated                                  ║
╚══════════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════════╗
║ NODE 12: Questionnaire Evaluation (EXTRA - Not Canonical)   ║
║ ──────────────────────────────────────────────────────────   ║
║ Module: questionnaire_engine.QuestionnaireEngine            ║
║ ⚠️  NOT part of canonical 11-node flow                       ║
║                                                              ║
║ Input: ALL orchestrator results                              ║
║ Output: 300-question evaluation                              ║
║                                                              ║
║ Status: ✅ Integrated (extended functionality)               ║
╚══════════════════════════════════════════════════════════════╝
    │
    ▼
  FINAL RESULTS
  (with immutability proof)
```

---

## 8. Missing Canonical Module: Decatalogo_principal.py

### Analysis

**File Status**: ✅ Exists in repository  
**Import Status**: ❌ **NOT imported in orchestrator**  
**Invocation Status**: ❌ **NOT invoked anywhere**

### Potential Integration Points

Based on typical "Decalogo" (ten commandments/principles) patterns, this module likely provides:
1. **Structural validation** against 10 canonical principles
2. **Pre-processing templates** for industrial development plans
3. **Schema validation** for plan documents

### Recommended Integration Strategy

**Option A**: Pre-sanitization validation
```python
# Before Node 1
decatalogo_results = self._execute_decatalogo_validation(plan_text)
if not decatalogo_results["is_compliant"]:
    # Log warnings or reject
```

**Option B**: Post-processing validation
```python
# After Node 2 (plan_processing)
decatalogo_validation = self._execute_decatalogo_validation(
    sanitized_text, processed_plan
)
```

**Option C**: Parallel validation track
```python
# Alongside other detection nodes (5-9)
decatalogo_score = self._execute_decatalogo_scoring(sanitized_text)
```

---

## 9. Summary Table: Canonical Module Integration

| Module | Import | Init | Invoke | Node | Status |
|--------|--------|------|--------|------|--------|
| **Decatalogo_principal.py** | ❌ | ❌ | ❌ | N/A | ❌ **MISSING** |
| **dag_validation.py** | ✅ | ✅ | ✅ | 11 | ✅ **COMPLETE** |
| **embedding_model.py** | ✅ | ✅ | ✅ | 4 | ✅ **COMPLETE** |
| **plan_processor.py** | ✅ | ✅ | ✅ | 2 | ✅ **COMPLETE** |
| **validate_teoria_cambio.py** | ❌ | ❌ | ⚠️ | 10 | ⚠️ **INLINE** |

**Integration Score**: **3/5 canonical modules fully integrated (60%)**

---

## 10. Recommendations

### Priority 1: Critical
1. **Integrate Decatalogo_principal.py**
   - Import module in orchestrator
   - Determine appropriate execution stage
   - Add to canonical flow contract

### Priority 2: High
2. **Externalize teoria_cambio validation**
   - Import `validate_teoria_cambio.py`
   - Replace inline `_validate_teoria_cambio()` logic
   - Ensure canonical module is source of truth

### Priority 3: Medium
3. **Update data_flow_contract.py**
   - Add contract for questionnaire_evaluation (Node 12)
   - Document extended flow as 12-node pipeline

### Priority 4: Low
4. **Document Decatalogo integration design**
   - Clarify intended role in pipeline
   - Update AGENTS.md with integration plan

---

## 11. Conclusion

The **miniminimoon_orchestrator.py** successfully implements a robust 12-stage pipeline that:
- ✅ Follows the canonical 11-node execution order without violations
- ✅ Properly chains data transformations between stages
- ✅ Integrates 3 of 5 specified canonical modules
- ⚠️ Missing integration for **Decatalogo_principal.py** (critical gap)
- ⚠️ Uses inline logic instead of **validate_teoria_cambio.py** module

**Overall Assessment**: The orchestrator demonstrates strong architectural alignment with the canonical flow contract, but requires integration of the two missing canonical modules to achieve full compliance with the five-module specification.
