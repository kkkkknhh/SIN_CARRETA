# Answer Assembler Refactoring - Weight Integration

## Summary

Refactored `answer_assembler.py` to load scoring modalities, expected elements, and weight mappings directly from configuration files (`RUBRIC_SCORING.json` and `dnp-standards.latest.clean.json`), eliminating hardcoded weight calculations and duplicate scoring logic.

## Key Changes

### 1. Added `rubric_weight` Field to `QuestionAnswer` Dataclass

```python
@dataclass
class QuestionAnswer:
    # ... existing fields ...
    scoring_modality: str
    rubric_weight: float  # NEW FIELD
    evidence_ids: List[str]
    # ... rest of fields ...
```

This field stores the dimension-specific weight for each question, loaded from the weights configuration file.

### 2. New Weight Loading Method: `_load_and_validate_weights()`

This method:
- Reads the `decalogo_dimension_mapping` section from `dnp-standards.latest.clean.json`
- Extracts dimension weights for each of the 10 thematic points (P1-P10)
- Creates a lookup dictionary with keys in format `{point_code}_{dimension_id}` (e.g., "P1_D1", "P2_D3")
- **Validates 1:1 constraint**: Ensures every question has exactly one corresponding weight entry
- Raises `ValueError` if any weight is missing or validation fails

Example weights structure:
```json
{
  "decalogo_dimension_mapping": {
    "P1": {
      "D1_weight": 0.20,
      "D2_weight": 0.20,
      "D3_weight": 0.15,
      "D4_weight": 0.20,
      "D5_weight": 0.15,
      "D6_weight": 0.10
    }
  }
}
```

### 3. Weight Population in `_assemble_single_answer()`

Modified to extract dimension and point code from `question_unique_id` and populate `rubric_weight`:

```python
parts = question_unique_id.split("-")  # e.g., "D1-Q1-P1" -> ["D1", "Q1", "P1"]
if len(parts) >= 3:
    dim_id = parts[0]       # "D1"
    point_code = parts[2]   # "P1"
    weight_key = f"{point_code}_{dim_id}"  # "P1_D1"
    rubric_weight = self.weights_lookup.get(weight_key, 0.0)
```

### 4. Weighted Aggregation in `_aggregate_to_point()`

Modified point-level aggregation to use loaded weights instead of simple averaging:

```python
point_weights = self.weights_config.get("decalogo_dimension_mapping", {}).get(point_code, {})

weighted_percentages = []
for ds in dimension_summaries:
    weight_key = f"{ds.dimension_id}_weight"
    weight = point_weights.get(weight_key, 1.0 / len(dimension_summaries))
    weighted_percentages.append(ds.percentage * weight)

weighted_avg_percentage = sum(weighted_percentages)
```

This ensures that dimensions with higher weights contribute more to the point's overall score.

### 5. Constructor Changes

Added `weights_path` parameter with default value:

```python
def __init__(
    self,
    rubric_path: str = "rubric_scoring.json",
    decalogo_path: str = "DECALOGO_FULL.json",
    weights_path: str = "dnp-standards.latest.clean.json"  # NEW PARAMETER
) -> None:
```

### 6. Validation Logic

The `_load_and_validate_weights()` method implements two validation checks:

**Check 1: Every dimension-point pair has a weight**
```python
for point_code, point_weights in dimension_mapping.items():
    for dim_id in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        weight_key = f"{dim_id}_weight"
        if weight_key not in point_weights:
            raise ValueError(
                f"❌ Missing weight for {dim_id} in point {point_code}. "
                f"Each point must have weights for all 6 dimensions."
            )
```

**Check 2: Every question has a corresponding weight**
```python
for question_unique_id in self.questions_by_unique_id.keys():
    parts = question_unique_id.split("-")
    if len(parts) >= 3:
        dim_id = parts[0]
        point_code = parts[2]
        weight_key = f"{point_code}_{dim_id}"
        if weight_key not in weights_lookup:
            raise ValueError(
                f"❌ Weight constraint violation: Question '{question_unique_id}' "
                f"requires weight '{weight_key}' which is not defined in weights config."
            )
```

## Data Sources

### RUBRIC_SCORING.json
- **scoring_modalities**: TYPE_A through TYPE_F with formulas and conversion tables
- **questions**: 30 question templates with expected_elements, search_patterns, scoring_modality
- **dimensions**: D1-D6 metadata (names, descriptions, max scores)
- **score_bands**: EXCELENTE, BUENO, SATISFACTORIO, INSUFICIENTE, DEFICIENTE

### dnp-standards.latest.clean.json
- **decalogo_dimension_mapping**: Weight mappings for all 10 points × 6 dimensions = 60 weights
- Each point (P1-P10) has 6 dimension weights (D1_weight through D6_weight)
- Weights sum to 1.0 per point
- Critical dimensions and minimum thresholds also specified

## Benefits

1. **Single Source of Truth**: All scoring configuration comes from JSON files, no hardcoded values
2. **Maintainability**: Weights can be updated in configuration without code changes
3. **Validation**: Enforces 1:1 constraint between questions and weights at initialization
4. **Transparency**: Weight values are explicitly stored in each QuestionAnswer for audit trails
5. **Flexibility**: Different weight schemes can be tested by swapping configuration files

## Testing

Created comprehensive test suite (`test_answer_assembler_refactor.py`) covering:

1. ✅ Weight loading from dnp-standards.latest.clean.json
2. ✅ 1:1 constraint validation (raises error on missing weights)
3. ✅ rubric_weight field population in QuestionAnswer
4. ✅ Weighted aggregation at point level
5. ✅ Scoring modalities loaded from RUBRIC_SCORING.json
6. ✅ Expected elements loaded from RUBRIC_SCORING.json

All tests pass successfully.

## Migration Notes

**Breaking Changes:**
- Constructor now accepts optional `weights_path` parameter (defaults to "dnp-standards.latest.clean.json")
- `QuestionAnswer` dataclass now has `rubric_weight` field
- Initialization will fail with `ValueError` if weight configuration is incomplete

**Backward Compatibility:**
- Existing code using default paths will work without changes
- Old reports without `rubric_weight` field will not be affected

## Example Usage

```python
from answer_assembler import AnswerAssembler

# Initialize with default paths (recommended)
assembler = AnswerAssembler()

# Or specify custom paths
assembler = AnswerAssembler(
    rubric_path="RUBRIC_SCORING.json",
    decalogo_path="DECALOGO_FULL.json",
    weights_path="dnp-standards.latest.clean.json"
)

# Assemble report (same API as before)
report = assembler.assemble(evidence_registry, evaluation_results)

# Access rubric_weight in question answers
for qa in report["question_answers"]:
    print(f"Question {qa['question_id']} has weight {qa['rubric_weight']}")
```

## Files Modified

- `answer_assembler.py`: Core refactoring with weight integration
- `test_answer_assembler_refactor.py`: Comprehensive test suite (NEW)

## Validation Results

```
✅ Loaded 60 weights (10 points × 6 dimensions)
✅ Weight validation passed: all 300 questions have corresponding weights
✅ rubric_weight correctly populated in QuestionAnswer
✅ Weighted aggregation working correctly at point level
✅ All 6 scoring modalities loaded from RUBRIC_SCORING.json
✅ Expected elements loaded correctly from RUBRIC_SCORING.json
```
