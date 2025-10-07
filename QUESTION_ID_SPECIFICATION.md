# Question ID Generation Flow - Technical Specification

## Executive Summary

This document traces the complete question ID generation flow in `questionnaire_engine.py` from base question definitions through thematic point multiplication to final evaluation results.

**KEY FINDING**: There is a **CRITICAL MISMATCH** between the implementation format and RUBRIC_SCORING.json format:
- **Implementation**: `P{1-10}-D{1-6}-Q{1-30}` (300 unique IDs, with thematic point prefix)
- **RUBRIC_SCORING.json**: `D{1-6}-Q{1-300}` (330 entries, sequential numbering 1-300, NO thematic point encoding)

---

## 1. Question ID Format Patterns

### 1.1 BaseQuestion ID Format
**Location**: Line 113, Line 190-962 (30 definitions)
**Format**: `D{1-6}-Q{1-30}`
**Example**: `D1-Q1`, `D2-Q6`, `D6-Q30`

```python
@dataclass
class BaseQuestion:
    id: str  # D1-Q1, D1-Q2, etc.
    dimension: str  # D1, D2, D3, D4, D5, D6
    question_no: int  # 1-30
```

**Pattern Distribution**:
- D1: Q1-Q5 (5 questions)
- D2: Q6-Q10 (5 questions)
- D3: Q11-Q15 (5 questions)
- D4: Q16-Q20 (5 questions)
- D5: Q21-Q25 (5 questions)
- D6: Q26-Q30 (5 questions)

**Total**: 30 base questions with sequential numbering 1-30 across all dimensions.

### 1.2 EvaluationResult ID Format (Implementation)
**Location**: Line 124 (docstring), Line 1284, 1607, 1675 (generation)
**Format**: `P{1-10}-D{1-6}-Q{1-30}`
**Example**: `P1-D1-Q1`, `P2-D1-Q1`, `P10-D6-Q30`

```python
@dataclass
class EvaluationResult:
    question_id: str  # P1-D1-Q1, P2-D1-Q1, etc.
    point_code: str  # P1, P2, etc.
```

**Generation Locations**:
1. **Line 1284**: `question_point_id = f"{point.id}-{question.id}"`
2. **Line 1607**: `question_id = f"{point.id}-{base_q.id}"`
3. **Line 1675**: `question_id = f"{thematic_point.id}-{base_question.id}"`

**All three locations use identical pattern**: `{point.id}-{base_question.id}`

---

## 2. Cardinality Transformation: 30 → 300

### 2.1 Multiplication Mechanism
**Location**: Lines 1268-1305 (execute_full_evaluation), Lines 1585-1620 (evaluate_with_evidence_registry)

```python
# Outer loop: 10 thematic points (P1-P10)
for point in self.thematic_points:
    # Inner loop: 30 base questions (D1-Q1 to D6-Q30)
    for question in self.base_questions:
        # Generate unique ID: P{n}-D{d}-Q{q}
        question_point_id = f"{point.id}-{question.id}"
```

**Result**: 10 points × 30 questions = **300 unique evaluations**

### 2.2 Thematic Point Encoding
**Location**: Lines 1193-1211 (default thematic points)

```python
ThematicPoint(id="P1", title="Derechos de las mujeres e igualdad de género")
ThematicPoint(id="P2", title="Prevención de la violencia y protección frente al conflicto")
...
ThematicPoint(id="P10", title="Migración transfronteriza")
```

**Encoding Scheme**: Thematic context is encoded in the **ID prefix** (`P1-`, `P2-`, etc.), creating 10 distinct evaluation contexts for each of the 30 base questions.

### 2.3 Question Parametrization
**Location**: Lines 1278, 1677

```python
# Template with placeholder
template = "¿El diagnóstico presenta líneas base... para {PUNTO_TEMATICO}?"

# Parametrization
prompt = base_question.template.replace("{PUNTO_TEMATICO}", thematic_point.title)
```

**Example**:
- Base: `D1-Q1` with template containing `{PUNTO_TEMATICO}`
- P1 instance: `P1-D1-Q1` with "Derechos de las mujeres..."
- P2 instance: `P2-D1-Q1` with "Prevención de la violencia..."

---

## 3. Code Locations Analysis

### 3.1 Question ID Creation Points

| Location | Context | Format | Purpose |
|----------|---------|--------|---------|
| Line 113 | BaseQuestion dataclass | `D{d}-Q{q}` | Define base question identity |
| Line 124 | EvaluationResult dataclass | `P{p}-D{d}-Q{q}` | Document expected format |
| Line 1284 | execute_full_evaluation() | `f"{point.id}-{question.id}"` | Generate evaluation IDs |
| Line 1294 | execute_full_evaluation() | Assignment to result | Assign generated ID |
| Line 1607 | evaluate_with_evidence_registry() | `f"{point.id}-{base_q.id}"` | Generate for evidence lookup |
| Line 1675 | _evaluate_question_with_evidence() | `f"{thematic_point.id}-{base_question.id}"` | Generate for result creation |

### 3.2 Question ID Assignment Points

| Location | Context | Operation |
|----------|---------|-----------|
| Line 1294 | `evaluation_result.question_id = question_point_id` | Direct assignment |
| Line 1722 | `EvaluationResult(question_id=question_id, ...)` | Constructor parameter |

### 3.3 Question ID Usage Points

| Location | Context | Purpose |
|----------|---------|---------|
| Line 1304 | `logger.debug(f"✓ {question_point_id}: ...")` | Logging |
| Line 1608 | `evidence_registry.for_question(question_id)` | Evidence retrieval |
| Line 1620 | `all_results.sort(key=lambda r: r.question_id)` | Deterministic ordering |

---

## 4. RUBRIC_SCORING.json Format Analysis

### 4.1 File Format
**Location**: `rubric_scoring.json`
**Format**: `D{1-6}-Q{1-300}`

```json
{
  "weights": {
    "D1-Q1": 0.0033333333333333335,
    "D1-Q2": 0.0033333333333333335,
    ...
    "D1-Q30": 0.0033333333333333335,
    "D1-Q31": 0.0033333333333333335,  // ❌ Beyond base questions
    ...
    "D6-Q50": 0.0033333333333333335   // ❌ Q50 doesn't exist in base
  }
}
```

**Total Entries**: 330 (verified with `grep -c`)

**Weight Value**: `1/300 = 0.00333...` (uniform weighting)

### 4.2 Sequential Numbering Pattern
- D1: Q1-Q55
- D2: Q1-Q55
- D3: Q1-Q55
- D4: Q1-Q55
- D5: Q1-Q55
- D6: Q1-Q55

**Apparent Intent**: Each dimension gets 55 questions (330 ÷ 6 = 55), suggesting an attempt to map 30 base questions × 10 points → 300 questions, but with incorrect arithmetic (50 per dimension would be 300 total).

---

## 5. Format Discrepancies

### 5.1 Critical Mismatch Summary

| Aspect | Implementation | RUBRIC_SCORING.json | Impact |
|--------|----------------|---------------------|--------|
| **Format** | `P{p}-D{d}-Q{q}` | `D{d}-Q{q}` | ❌ Keys won't match |
| **Thematic Encoding** | Prefix (`P1-`, `P2-`) | None | ❌ Context lost |
| **Question Range** | Q1-Q30 (per dimension group) | Q1-Q50+ (per dimension) | ❌ Range mismatch |
| **Total Keys** | 300 unique | 330 entries | ❌ Count mismatch |
| **Dimension Distribution** | 5 questions each | 55 questions each | ❌ Structure mismatch |

### 5.2 Specific Issues

#### Issue 1: Thematic Point Prefix Missing
**Implementation generates**: `P1-D1-Q1`, `P2-D1-Q1`, ..., `P10-D1-Q1`
**RUBRIC_SCORING.json expects**: `D1-Q1` (single entry)

**Consequence**: No way to apply weights to thematic point-specific evaluations.

#### Issue 2: Question Numbering Scheme
**Implementation uses**: Sequential 1-30 across all dimensions
- D1: Q1-Q5
- D2: Q6-Q10
- ...
- D6: Q26-Q30

**RUBRIC_SCORING.json uses**: Sequential 1-50+ within each dimension
- D1: Q1-Q55
- D2: Q1-Q55
- ...
- D6: Q1-Q55

**Consequence**: Same question number (e.g., Q1) appears in multiple dimensions in rubric but only in D1 in implementation.

#### Issue 3: Cardinality
**Implementation**: 300 unique IDs (10 points × 30 questions)
**RUBRIC_SCORING.json**: 330 entries (6 dimensions × 55 questions)

**Consequence**: 30 extra entries in rubric that have no corresponding evaluations.

---

## 6. Evidence Registry Integration

### 6.1 Evidence Lookup Pattern
**Location**: Lines 1607-1608

```python
question_id = f"{point.id}-{base_q.id}"  # e.g., "P1-D1-Q1"
evidence_list = evidence_registry.for_question(question_id)
```

**Format**: Matches implementation format `P{p}-D{d}-Q{q}`

**Implication**: Evidence registry uses the same 3-part ID scheme with thematic point prefix.

### 6.2 Evidence Registry Consistency
✅ **Consistent**: Evidence retrieval and evaluation use identical ID format
❌ **Inconsistent with rubric**: Evidence keys won't match RUBRIC_SCORING.json keys

---

## 7. Aggregation Patterns

### 7.1 Dimension Aggregation
**Location**: Lines 1774-1803 (_aggregate_by_dimension)

Groups by `result.dimension` (D1-D6), aggregating across all thematic points.

### 7.2 Point Aggregation
**Location**: Lines 1805-1848 (_aggregate_by_point)

Groups by `result.point_code` (P1-P10), creating nested dimension scores within each point.

### 7.3 Result Structure
```python
{
  "results": {
    "all_questions": [...],  # 300 items with P{p}-D{d}-Q{q} IDs
    "by_dimension": {...},   # Grouped by D1-D6
    "by_point": {...},       # Grouped by P1-P10
    "global": {...}          # Overall score
  }
}
```

---

## 8. Standardization Recommendations

### 8.1 Option A: Align RUBRIC_SCORING.json to Implementation
**Action**: Regenerate rubric with 300 entries using `P{p}-D{d}-Q{q}` format

```json
{
  "weights": {
    "P1-D1-Q1": 0.0033333333333333335,
    "P1-D1-Q2": 0.0033333333333333335,
    ...
    "P10-D6-Q30": 0.0033333333333333335
  }
}
```

**Pros**: Matches implementation, preserves thematic context
**Cons**: Requires rubric regeneration

### 8.2 Option B: Remove Thematic Prefix from Implementation
**Action**: Change ID generation to use only `D{d}-Q{seq}` with sequential numbering

```python
# Map base question to sequential number 1-300
question_seq = (thematic_point_index * 30) + base_question.question_no
question_id = f"{base_question.dimension}-Q{question_seq}"
```

**Pros**: Matches rubric format
**Cons**: Loses explicit thematic encoding, requires question_no recalculation

### 8.3 Option C: Hybrid Approach (RECOMMENDED)
**Action**: Maintain implementation format, add rubric mapping layer

```python
def map_to_rubric_key(evaluation_id: str) -> str:
    # P1-D1-Q1 → D1-Q1 (first occurrence)
    # P2-D1-Q1 → D1-Q31 (second occurrence)
    # ...
    point_num = int(evaluation_id[1:].split('-')[0])
    base_id = '-'.join(evaluation_id.split('-')[1:])  # D1-Q1
    dim, q = base_id.split('-')
    q_num = int(q[1:])
    rubric_q = (point_num - 1) * 30 + q_num
    return f"{dim}-Q{rubric_q}"
```

**Pros**: Preserves both formats, maintains thematic context in evaluation
**Cons**: Adds mapping complexity

---

## 9. Validation Points

### 9.1 Structure Validation
**Location**: Lines 32-38 (QuestionnaireStructure.validate_structure)

```python
def validate_structure(self) -> bool:
    return (self.DOMAINS * self.QUESTIONS_PER_DOMAIN == self.TOTAL_QUESTIONS and
            self.DIMENSIONS * self.QUESTIONS_PER_DIMENSION == self.QUESTIONS_PER_DOMAIN)
# 10 × 30 = 300 ✓
# 6 × 5 = 30 ✓
```

### 9.2 Count Validation
**Location**: Line 1328

```python
if evaluation_count != self.structure.TOTAL_QUESTIONS:
    raise RuntimeError(f"Expected {self.structure.TOTAL_QUESTIONS}, executed {evaluation_count}")
```

---

## 10. Summary Table

| Component | Format | Count | Thematic Encoding | Status |
|-----------|--------|-------|-------------------|--------|
| BaseQuestion.id | `D{d}-Q{q}` | 30 | None | ✅ Correct |
| EvaluationResult.question_id | `P{p}-D{d}-Q{q}` | 300 | Prefix | ✅ Correct |
| Evidence Registry Keys | `P{p}-D{d}-Q{q}` | 300 | Prefix | ✅ Consistent |
| RUBRIC_SCORING.json Keys | `D{d}-Q{q}` | 330 | None | ❌ Mismatch |
| question_no field | 1-30 | 30 | None | ✅ Correct |

---

## 11. Action Items for Standardization

1. **Immediate**: Document the discrepancy (this document)
2. **Phase 1**: Decide on canonical format (recommend Option C - hybrid)
3. **Phase 2**: Regenerate or create mapping for RUBRIC_SCORING.json
4. **Phase 3**: Add validation that checks rubric keys match evaluation IDs
5. **Phase 4**: Update any downstream consumers to handle chosen format
6. **Phase 5**: Add integration tests verifying format consistency

---

## Appendix A: Question Distribution

```
D1 (Diagnóstico y Recursos): Q1-Q5
  - D1-Q1: Líneas base
  - D1-Q2: Magnitud del problema
  - D1-Q3: Recursos PPI
  - D1-Q4: Capacidades institucionales
  - D1-Q5: Coherencia objetivos-recursos

D2 (Diseño de Intervención): Q6-Q10
  - D2-Q6: Actividades formalizadas
  - D2-Q7: Población diana
  - D2-Q8: Problema-actividad matching
  - D2-Q9: Riesgos identificados
  - D2-Q10: Teoría de intervención

D3 (Productos): Q11-Q15
  - D3-Q11: Estándares técnicos
  - D3-Q12: Monitoreo y evaluación
  - D3-Q13: Calidad de productos
  - D3-Q14: Cobertura territorial
  - D3-Q15: Sostenibilidad

D4 (Resultados): Q16-Q20
  - D4-Q16: Indicadores de resultado
  - D4-Q17: Metas cuantificadas
  - D4-Q18: Plazos definidos
  - D4-Q19: Responsables identificados
  - D4-Q20: Sistema de seguimiento

D5 (Efectos): Q21-Q25
  - D5-Q21: Cambios esperados
  - D5-Q22: Población beneficiaria
  - D5-Q23: Cadena causal
  - D5-Q24: Supuestos explícitos
  - D5-Q25: Efectos no deseados

D6 (Impacto): Q26-Q30
  - D6-Q26: Impacto de largo plazo
  - D6-Q27: Alineación con ODS
  - D6-Q28: Contribución al desarrollo
  - D6-Q29: Transformaciones estructurales
  - D6-Q30: Evaluación de impacto
```

Each question × 10 thematic points = 300 total evaluations

---

**Document Version**: 1.0
**Generated**: 2025-01-XX
**Author**: Code Audit System
