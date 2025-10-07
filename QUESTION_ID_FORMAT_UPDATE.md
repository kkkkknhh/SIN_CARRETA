# Question ID Format Standardization Update

## Overview
All test files have been updated to generate question IDs in the standardized **D{N}-Q{N}** format (e.g., D1-Q1, D2-Q15, D6-Q50) to match the 300-question structure documented in RUBRIC_SCORING.json.

## Format Specification
- **Pattern**: `D{dimension}-Q{question}`
- **Range**: D1-Q1 through D6-Q50
- **Total Questions**: 300 (6 dimensions × 50 questions each)
- **Formula**: For index i (0-299): `D{i//50 + 1}-Q{i%50 + 1}`

## Files Updated

### 1. test_rubric_subprocess_audit.py
**Changes:**
- Updated `create_test_environment()` to generate mock `answers_report.json` with D{N}-Q{N} format
- Updated mock `RUBRIC_SCORING.json` weights to use D{N}-Q{N} keys
- Updated `test_exit_code_3_mismatch()` to create mismatched rubric with D{N}-Q{N} format

**Before:**
```python
"answers": [{"question_id": f"q{i}", "answer": "test"} for i in range(300)]
"weights": {f"q{i}": 1.0 for i in range(300)}
```

**After:**
```python
"answers": [{"question_id": f"D{i//50 + 1}-Q{i%50 + 1}", "answer": "test"} for i in range(300)]
"weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 1.0 for i in range(300)}
```

### 2. test_system_validators.py
**Changes:**
- Updated `temp_repo` fixture to create RUBRIC_SCORING.json with D{N}-Q{N} format weights
- Updated `test_post_execution_validation_success()` to generate answers with D{N}-Q{N} IDs
- Updated `test_post_execution_insufficient_coverage()` to use D{N}-Q{N} format
- Updated `test_post_execution_all_documents_success()` to include D{N}-Q{N} question IDs in coverage reports
- Updated `test_post_execution_insufficient_coverage()` batch test with D{N}-Q{N} format

**Before:**
```python
"answers": [{"question_id": f"Q{i}"} for i in range(300)]
"weights": {}
```

**After:**
```python
"answers": [{"question_id": f"D{i//50 + 1}-Q{i%50 + 1}"} for i in range(300)]
"weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 0.0033333333333333335 for i in range(300)}
```

### 3. tests/test_unified_flow_certification.py
**Changes:**
- Updated `setUpClass()` to create minimal rubric with D{N}-Q{N} format weights (matching actual weight value from RUBRIC_SCORING.json)
- Updated `_validate_answers_report()` to validate question_id format using regex pattern
- Added regex validation: `r'^D[1-6]-Q([1-9]|[1-4][0-9]|50)$'`
- Enhanced validation messages to confirm D{N}-Q{N} format compliance

**Before:**
```python
"weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 1.0 for i in range(300)}
# No format validation
```

**After:**
```python
"weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 0.0033333333333333335 for i in range(300)}
question_id_pattern = re.compile(r'^D[1-6]-Q([1-9]|[1-4][0-9]|50)$')
self.assertRegex(question_id, question_id_pattern, ...)
```

## Cardinality Verification

### Distribution Across Dimensions
- **D1 (Diagnóstico y Recursos)**: Q1-Q50 (50 questions)
- **D2 (Diseño de Intervención)**: Q1-Q50 (50 questions)
- **D3 (Productos y Outputs)**: Q1-Q50 (50 questions)
- **D4 (Resultados y Outcomes)**: Q1-Q50 (50 questions)
- **D5 (Impactos y Efectos de Largo Plazo)**: Q1-Q50 (50 questions)
- **D6 (Teoría de Cambio y Coherencia Causal)**: Q1-Q50 (50 questions)

**Total**: 6 dimensions × 50 questions = **300 questions**

## Rubric Weight Alignment

### RUBRIC_SCORING.json Structure
```json
{
  "metadata": {
    "total_questions": 300,
    "dimensions": 6
  },
  "weights": {
    "D1-Q1": 0.0033333333333333335,
    "D1-Q2": 0.0033333333333333335,
    ...
    "D6-Q50": 0.0033333333333333335
  }
}
```

All 300 question IDs now have matching weight entries, enabling successful `rubric_check.py` validation in test scenarios.

## Alternative ID Format Removal

### Removed Formats
- ❌ `q{i}` (e.g., q0, q1, q299)
- ❌ `Q{i}` (e.g., Q0, Q1, Q299)
- ❌ `question_{i}` (e.g., question_0, question_1)

### Standardized Format
- ✅ `D{N}-Q{N}` (e.g., D1-Q1, D2-Q15, D6-Q50)

## Test Validation

### Formula Verification
The formula `D{i//50 + 1}-Q{i%50 + 1}` correctly maps:
- i=0 → D1-Q1
- i=49 → D1-Q50
- i=50 → D2-Q1
- i=299 → D6-Q50

### Test Coverage
All test files now:
1. ✅ Generate question IDs in D{N}-Q{N} format
2. ✅ Create mock RUBRIC_SCORING.json with matching weights
3. ✅ Validate question ID format using regex patterns
4. ✅ Ensure 300-question cardinality across all test scenarios
5. ✅ Support rubric_check.py validation in test environments

## Compilation Status
All test files compile successfully:
```bash
python3 -m py_compile test_rubric_subprocess_audit.py test_system_validators.py tests/test_unified_flow_certification.py
✓ No syntax errors
```

## Notes
- No test_answer_assembler_refactor.py file was found in the repository
- All test data generation now matches the actual 300-question cardinality
- Mock rubric weights use the actual value from RUBRIC_SCORING.json (0.0033333333333333335 ≈ 1/300)
- Regex validation in test_unified_flow_certification.py ensures strict format compliance
