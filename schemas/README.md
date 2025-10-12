# JSON Schemas for P-D-Q Canonical Notation

This directory contains JSON Schema definitions for validating artifacts in the MINIMINIMOON system.

## Schemas

### evidence.schema.json
Validates evidence entries with P-D-Q canonical notation.

**Required fields:**
- `evidence_id`: Unique identifier
- `question_unique_id`: Format `P#-D#-Q#` (e.g., `P4-D2-Q3`)
- `content.policy`: Format `P#` (e.g., `P4`)
- `content.dimension`: Format `D#` (e.g., `D2`)
- `content.question`: Positive integer
- `content.score`: Float 0-1
- `content.rubric_key`: Format `D#-Q#` (e.g., `D2-Q3`)
- `confidence`: Float 0-1
- `stage`: Processing stage identifier

### rubric.schema.json
Validates rubric scoring weights.

**Required fields:**
- `weights`: Object with keys matching pattern `D#-Q#`
  - Each value must be a non-negative number

**Expected keys:** D1-Q1 through D6-Q50 (300 total for default configuration)

### decalogo_bundle.schema.json
Validates the canonical question bundle.

**Required fields:**
- `version`: Version string
- `policies`: Array of 10 policies (P1-P10)
- `dimensions`: Array of 6 dimensions (D1-D6)
- `questions`: Array of question objects
  - Each question must have:
    - `question_unique_id`: Format `P#-D#-Q#`
    - `rubric_key`: Format `D#-Q#`
    - `text`: Question text
- `lexicon`: Policy-specific terminology

## Validation

Validate files against these schemas:

```bash
# Validate all schemas
python tools/validate_canonical_schemas.py

# Validate P-D-Q notation compliance
python tools/validate_pdq_notation.py
```

## References

- Complete specification: [`docs/PDQ_CANONICAL_NOTATION.md`](../docs/PDQ_CANONICAL_NOTATION.md)
- JSON Schema standard: https://json-schema.org/draft/2020-12/schema
