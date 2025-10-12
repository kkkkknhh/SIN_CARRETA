# P-D-Q Canonical Notation System

## Overview

The MINIMINIMOON Sistema Canónico de Evaluación de PDM uses a standardized canonical notation for all question identifiers and rubric keys. This ensures consistency, traceability, and deterministic evaluation across the entire system.

## Canonical Format

### Components

- **P#** = Policy Point (Punto del Decálogo)
  - Range: `P1` through `P10`
  - Represents one of 10 thematic policy areas in Colombian Municipal Development Plans

- **D#** = Analytical Dimension (Dimensión analítica)
  - Range: `D1` through `D6`
  - Represents evaluation dimensions (Diagnóstico, Diseño, Productos, Resultados, Impactos, Teoría de Cambio)

- **Q#** = Question Number
  - Range: `Q1` and up (positive integers)
  - Unique question identifier within a dimension

### Identifiers

#### question_unique_id
**Format:** `P#-D#-Q#`

**Examples:**
- `P4-D2-Q3` - Policy 4, Dimension 2, Question 3
- `P1-D1-Q1` - Policy 1, Dimension 1, Question 1
- `P10-D6-Q30` - Policy 10, Dimension 6, Question 30

**Regex:** `^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$`

#### rubric_key
**Format:** `D#-Q#`

**Purpose:** Used in `RUBRIC_SCORING.json` for scoring weights

**Examples:**
- `D2-Q3` - Dimension 2, Question 3
- `D1-Q1` - Dimension 1, Question 1
- `D6-Q30` - Dimension 6, Question 30

**Regex:** `^D[1-6]-Q[1-9][0-9]*$`

**Derivation:** Extract `D#-Q#` from `P#-D#-Q#`
```python
# Example: "P4-D2-Q3" → "D2-Q3"
parts = question_unique_id.split("-")
rubric_key = f"{parts[1]}-{parts[2]}"
```

## System Structure

### Default Configuration

By default, the system supports:
- **10 policies** × **6 dimensions** × **5 questions** = **300 total questions**

This can be customized via `config/QUESTIONNAIRE_MANIFEST.yaml` overrides.

### Policy Areas (P1-P10)

| ID | Title |
|----|-------|
| P1 | Derechos de las mujeres e igualdad de género |
| P2 | Prevención de la violencia y protección frente al conflicto |
| P3 | Ambiente sano, cambio climático, prevención y atención a desastres |
| P4 | Derechos económicos, sociales y culturales |
| P5 | Derechos de las víctimas y construcción de paz |
| P6 | Derecho al buen futuro de la niñez, adolescencia, juventud |
| P7 | Tierras y territorios |
| P8 | Líderes y defensores de derechos humanos |
| P9 | Crisis de derechos de personas privadas de la libertad |
| P10 | Migración transfronteriza |

### Analytical Dimensions (D1-D6)

| ID | Name | Focus |
|----|------|-------|
| D1 | Diagnóstico y Recursos | Baseline, problem magnitude, resources, institutional capacity |
| D2 | Diseño de Intervención | Activities, target population, intervention design |
| D3 | Productos y Outputs | Technical standards, proportionality, quantification, accountability |
| D4 | Resultados y Outcomes | Result indicators, differentiation, magnitude of change, attribution |
| D5 | Impactos y Efectos de Largo Plazo | Impact indicators, temporal horizons, systemic effects, sustainability |
| D6 | Teoría de Cambio y Coherencia Causal | Theory of change, assumptions, logical framework, monitoring |

## Evidence Format

All evidence entries must follow this canonical structure:

```json
{
  "evidence_id": "toc_P7-D3-Q5",
  "question_unique_id": "P7-D3-Q5",
  "content": {
    "policy": "P7",
    "dimension": "D3",
    "question": 5,
    "score": 0.82,
    "rubric_key": "D3-Q5"
  },
  "confidence": 0.82,
  "stage": "teoria_cambio"
}
```

### Required Fields

- `evidence_id`: Unique identifier for the evidence entry
- `question_unique_id`: Must match pattern `P#-D#-Q#`
- `content.policy`: Must match pattern `P#`
- `content.dimension`: Must match pattern `D#`
- `content.question`: Positive integer
- `content.score`: Float between 0 and 1
- `content.rubric_key`: Must match pattern `D#-Q#`
- `confidence`: Float between 0 and 1
- `stage`: String identifying the processing stage

## Migration from Legacy Formats

The system supports automatic migration from three legacy ID formats:

### Case A: `D#-Q#` (no policy)

**Strategy:** Infer `P#` from document section or context

**Example:**
```python
legacy_id = "D4-Q3"
context = {"section": "P8"}
# Result: "P8-D4-Q3"
```

### Case B: `P#-Q#` (no dimension)

**Strategy:** Infer `D#` from question number range or context

**Example:**
```python
legacy_id = "P2-Q5"
# Q5 falls in D1 range (Q1-Q5)
# Result: "P2-D1-Q5"
```

### Case C: `Q#` (neither policy nor dimension)

**Strategy:** Infer both `P#` and `D#` from context

**Example:**
```python
legacy_id = "Q12"
context = {"section": "P6"}
# Q12 falls in D3 range (Q11-Q15)
# Result: "P6-D3-Q12"
```

### Confidence Thresholds

Migration requires minimum confidence of **0.80** (configurable in `QUESTIONNAIRE_MANIFEST.yaml`).

If confidence is below threshold, migration fails with:
```
ERROR_QID_NORMALIZATION: cannot infer policy for 'D4-Q3' (confidence 0.60 < 0.80)
```

## File Locations

### Core Artifacts

- **Schemas:** `/schemas/`
  - `evidence.schema.json` - Evidence entry validation
  - `rubric.schema.json` - Rubric weights validation
  - `decalogo_bundle.schema.json` - Bundle validation

- **Configuration:** `/config/`
  - `QUESTIONNAIRE_MANIFEST.yaml` - Canonical configuration
  - `RUBRIC_SCORING.json` - Scoring weights (keys: `D#-Q#`)

- **Bundles:** `/bundles/`
  - `decalogo_bundle.json` - Canonical question bundle

- **Migration:** `/migration/`
  - `migrate_legacy_ids.py` - Legacy ID migration tool

### Validation Tools

- `/tools/validate_canonical_schemas.py` - Schema validation
- `/tools/validate_pdq_notation.py` - P-D-Q format validation
- `/tools/generate_canonical_bundle.py` - Bundle generator

### Golden Tests

- `/tests/golden/test_pdq_migration.py` - Migration test cases

## Validation

### Pre-execution Validation

Run before any processing:

```bash
# Validate schemas
python tools/validate_canonical_schemas.py

# Validate P-D-Q notation
python tools/validate_pdq_notation.py

# Run golden tests
python tests/golden/test_pdq_migration.py
```

### CI/CD Integration

Automatic validation runs on every commit via GitHub Actions:

```yaml
# .github/workflows/pdq-validation.yml
- Validates JSON schemas
- Checks P-D-Q notation compliance
- Runs golden test cases
```

### Manual Validation

```python
from migration.migrate_legacy_ids import LegacyIDMigrator

migrator = LegacyIDMigrator()

# Migrate legacy ID
try:
    canonical_id, rubric_key, confidence = migrator.migrate(
        "D4-Q3",
        context={"section": "P8"}
    )
    print(f"Migrated to: {canonical_id}")
except ValueError as e:
    print(f"Migration failed: {e}")
```

## Error Messages

The system provides human-friendly error messages:

| Error Code | Description | Example |
|------------|-------------|---------|
| `ERROR_QID_FORMAT` | Invalid ID format | `invalid id 'D7-Q1' — expected 'P#-D#-Q#'` |
| `ERROR_QID_NORMALIZATION` | Cannot standardize legacy ID | `cannot infer policy for 'D4-Q3'` |
| `ERROR_RUBRIC_MISS` | Missing rubric weight | `missing weight for rubric key 'D3-Q7'` |
| `ERROR_BUNDLE_SCHEMA` | Bundle schema violation | `violates decalogo_bundle.schema.json` |

## Code Examples

### Using in Answer Assembler

```python
from answer_assembler import AnswerAssembler

assembler = AnswerAssembler(
    rubric_path="config/RUBRIC_SCORING.json",
    evidence_registry=registry
)

# Assemble answer for P4-D2-Q3
answer = assembler.assemble(
    question_uid="P4-D2-Q3",
    question_text="..."
)

# answer.rubric_key = "D2-Q3"
# answer.weight = (from rubric weights["D2-Q3"])
```

### Using in Teoria de Cambio

```python
from teoria_cambio import make_question_uid, to_rubric_key

# Generate canonical UID
uid = make_question_uid("P7", "D3", 5)
# Result: "P7-D3-Q5"

# Extract rubric key
rubric_key = to_rubric_key(uid)
# Result: "D3-Q5"
```

## Determinism and Immutability

All artifacts include immutability metadata:

```json
{
  "immutability": {
    "version": "5.0.0",
    "timestamp_utc": "2025-10-12T03:04:00.000Z",
    "hash": "sha256_hex_64_chars"
  }
}
```

Hash calculation:
- Serialization: `json.dumps(payload, sort_keys=True, separators=(",",":"))`
- Algorithm: SHA-256
- Excludes: Volatile fields (timestamps, non-canonical data)

## Acceptance Criteria

A valid P-D-Q implementation must:

1. ✅ All JSON files validate against schemas
2. ✅ No invalid ID patterns in code or config
3. ✅ `answers_report.json` uses `question_unique_id = "P#-D#-Q#"`
4. ✅ All rubric keys follow `D#-Q#` format
5. ✅ Evidence entries include both `question_unique_id` and `rubric_key`
6. ✅ Migration log generated for any legacy IDs
7. ✅ All golden tests pass (17/17)
8. ✅ Deterministic hashes present in exports

## References

- JSON Schema specification: https://json-schema.org/draft/2020-12/schema
- Repository: `/schemas/`, `/config/`, `/bundles/`, `/migration/`
- Related: `ARCHITECTURE.md`, `INTEGRATION_FLOW.md`, `DATA_CONTRACTS.md`
