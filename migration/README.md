# Legacy ID Migration

This directory contains tools for migrating legacy ID formats to the canonical P-D-Q notation.

## Migration Tool

### migrate_legacy_ids.py

Migrates legacy question identifiers to canonical P-D-Q format.

**Supported legacy formats:**

1. **D#-Q#** (no policy) - Infers P# from context
   ```python
   "D4-Q3" + {"section": "P8"} → "P8-D4-Q3"
   ```

2. **P#-Q#** (no dimension) - Infers D# from question range
   ```python
   "P2-Q5" → "P2-D1-Q5"  # Q5 is in D1 range (Q1-Q5)
   ```

3. **Q#** (neither policy nor dimension) - Infers both
   ```python
   "Q12" + {"section": "P6"} → "P6-D3-Q12"  # Q12 is in D3 range (Q11-Q15)
   ```

## Usage

### Command Line

```bash
# Run with test cases
python migration/migrate_legacy_ids.py

# Output: migration_log.json in output/ directory
```

### Programmatic

```python
from migration.migrate_legacy_ids import LegacyIDMigrator

migrator = LegacyIDMigrator(
    manifest_path="config/QUESTIONNAIRE_MANIFEST.yaml",
    bundle_path="bundles/decalogo_bundle.json"
)

# Migrate a legacy ID
try:
    canonical_id, rubric_key, confidence = migrator.migrate(
        legacy_id="D4-Q3",
        context={"section": "P8"}
    )
    print(f"Migrated: {canonical_id}")
    print(f"Rubric key: {rubric_key}")
    print(f"Confidence: {confidence:.2f}")
except ValueError as e:
    print(f"Migration failed: {e}")

# Save migration log
migrator.log.save("output/migration_log.json")
```

## Inference Strategies

1. **section** - Extract policy from document section headers
   - Confidence: 0.95
   - Pattern: Looks for "P#" or "PUNTO #" in section text

2. **question_range** - Map question number to dimension
   - Confidence: 0.85
   - Logic: Q1-Q5→D1, Q6-Q10→D2, Q11-Q15→D3, Q16-Q20→D4, Q21-Q25→D5, Q26-Q30→D6

3. **bundle_lookup** - Search canonical bundle for matching question
   - Confidence: 0.90
   - Requires: Valid bundle with question_unique_id entries

4. **fallback** - Use default from manifest
   - Confidence: 0.50-0.60
   - Default policy: P4 (configurable)

## Confidence Thresholds

Minimum confidence for auto-inference: **0.80** (configurable in `QUESTIONNAIRE_MANIFEST.yaml`)

If confidence is below threshold, migration fails with:
```
ERROR_QID_NORMALIZATION: cannot infer policy for 'D4-Q3' (confidence 0.60 < 0.80)
```

## Migration Log Format

```json
{
  "total_migrations": 3,
  "migrations": [
    {
      "original_id": "D4-Q3",
      "normalized_id": "P8-D4-Q3",
      "rubric_key": "D4-Q3",
      "strategy": "section",
      "confidence": 0.95,
      "notes": "Inferred policy from section"
    }
  ]
}
```

## Testing

Run golden test cases:

```bash
python tests/golden/test_pdq_migration.py
```

## References

- Complete specification: [`docs/PDQ_CANONICAL_NOTATION.md`](../docs/PDQ_CANONICAL_NOTATION.md)
- Configuration: [`config/QUESTIONNAIRE_MANIFEST.yaml`](../config/QUESTIONNAIRE_MANIFEST.yaml)
- Canonical bundle: [`bundles/decalogo_bundle.json`](../bundles/decalogo_bundle.json)
