# Evidence Validation Layer

## Overview

The Evidence Validation Layer is a comprehensive tracking and validation system built into the `EvidenceRegistry` that ensures all 300 questions receive sufficient evidence from detector stages 1-12. It enforces minimum evidence thresholds and provides complete provenance traceability for audit and debugging.

## Key Features

### 1. Evidence Count Tracking
- Tracks evidence counts per question across all detector stages (1-12)
- Enforces configurable minimum threshold (default: 3 evidence sources)
- Identifies questions that fall below the threshold

### 2. Provenance Metadata
Each evidence item includes complete provenance tracking:
- **Detector Type**: Type of detector that produced evidence (e.g., 'monetary', 'responsibility')
- **Stage Number**: Detector stage number (1-12)
- **Confidence Score**: Evidence confidence score (0.0-1.0)
- **Source Text Location**: Page, line, character offsets in source document
- **Execution Timestamp**: ISO 8601 timestamp of evidence generation
- **Quality Metrics**: Precision, recall, F1 scores for evidence quality

### 3. Post-Execution Validation
The `validate_evidence_counts()` method provides comprehensive validation:
- Iterates through all 300 questions
- Checks evidence count against threshold
- Logs questions below threshold with contributing/missing stages
- Generates detailed validation report

### 4. Traceability Chains
Complete evidence traceability showing:
- Question ID → Evidence Sources
- Evidence ID → Detector Module
- Detector Module → Stage Number
- Stage Number → Execution Timestamp
- Quality metrics for each evidence item

## API Reference

### EvidenceProvenance

```python
@dataclass(frozen=True)
class EvidenceProvenance:
    detector_type: str              # Type of detector
    stage_number: int               # Stage number (1-12)
    source_text_location: Dict      # Location in source text
    execution_timestamp: str        # ISO 8601 timestamp
    quality_metrics: Dict[str, float]  # Quality scores
```

**Example:**
```python
provenance = EvidenceProvenance(
    detector_type="responsibility",
    stage_number=2,
    source_text_location={
        "page": 5,
        "line": 42,
        "char_start": 100,
        "char_end": 250
    },
    execution_timestamp="2024-01-15T10:30:00Z",
    quality_metrics={
        "precision": 0.92,
        "recall": 0.88,
        "f1": 0.90
    }
)
```

### Registering Evidence with Provenance

```python
evidence_id = registry.register(
    source_component="responsibility_detector",
    evidence_type="entity_detection",
    content={"entity": "Ministerio de Educación"},
    confidence=0.92,
    applicable_questions=["D1-Q1"],
    provenance=provenance  # Optional provenance metadata
)
```

### Validating Evidence Counts

```python
# Generate all 300 question IDs
all_questions = [f"D{d}-Q{q}" for d in range(1, 11) for q in range(1, 31)]

# Perform validation
validation_result = registry.validate_evidence_counts(
    all_question_ids=all_questions,
    min_evidence_threshold=3
)
```

### Validation Result Structure

```python
{
    "valid": bool,                    # True if all questions meet threshold
    "total_questions": int,           # Total number of questions validated
    "questions_meeting_threshold": int,  # Number meeting threshold
    "questions_below_threshold": List[str],  # Question IDs below threshold
    "validation_timestamp": str,      # ISO 8601 timestamp
    "min_evidence_threshold": int,    # Minimum evidence required
    "evidence_summary": {             # Per-question detailed summary
        "D1-Q1": {
            "evidence_count": int,
            "meets_threshold": bool,
            "stage_contributions": {  # Evidence by stage
                "1": [...],           # List of evidence from stage 1
                "2": [...],           # List of evidence from stage 2
                ...
            },
            "missing_stages": List[int],  # Stages without evidence
            "evidence_sources": [    # Full evidence details
                {
                    "evidence_id": str,
                    "source_component": str,
                    "evidence_type": str,
                    "confidence": float,
                    "detector_type": str,
                    "stage_number": int,
                    "execution_timestamp": str,
                    "quality_metrics": {...}
                },
                ...
            ]
        },
        ...
    },
    "stage_coverage_summary": {      # Stage coverage statistics
        "evidence_count_per_stage": {1: 75, 2: 75, ...},
        "questions_per_stage": {1: [...], 2: [...], ...},
        "stages_with_evidence": [1, 2, 3, ...],
        "stages_without_evidence": []
    }
}
```

### Exporting Validation Results

```python
registry.export_validation_results(
    validation_result,
    "evidence_validation_results.json"
)
```

## Usage Examples

### Basic Validation Workflow

```python
from evidence_registry import EvidenceRegistry, EvidenceProvenance
from datetime import datetime

# Create registry
registry = EvidenceRegistry()

# Register evidence with provenance
for stage in range(1, 13):
    provenance = EvidenceProvenance(
        detector_type="monetary",
        stage_number=stage,
        source_text_location={"page": 1, "line": stage * 10},
        execution_timestamp=datetime.utcnow().isoformat() + "Z",
        quality_metrics={"precision": 0.85, "recall": 0.80, "f1": 0.82}
    )
    
    registry.register(
        source_component=f"detector_stage_{stage}",
        evidence_type="monetary_evidence",
        content={"amount": 1000000},
        confidence=0.85,
        applicable_questions=["D2-Q3"],
        provenance=provenance
    )

# Generate 300 question IDs
all_questions = [f"D{d}-Q{q}" for d in range(1, 11) for q in range(1, 31)]

# Validate
result = registry.validate_evidence_counts(
    all_question_ids=all_questions,
    min_evidence_threshold=3
)

# Check results
if not result["valid"]:
    print(f"Validation failed: {len(result['questions_below_threshold'])} questions below threshold")
    for qid in result["questions_below_threshold"][:5]:
        summary = result["evidence_summary"][qid]
        print(f"  {qid}: {summary['evidence_count']} evidence")
        print(f"    Contributing stages: {sorted(summary['stage_contributions'].keys())}")
        print(f"    Missing stages: {summary['missing_stages'][:5]}...")

# Export results
registry.export_validation_results(result, "validation_results.json")
```

### Analyzing Stage Coverage

```python
# Get stage coverage statistics
stage_coverage = validation_result["stage_coverage_summary"]

# Check which stages produced evidence
print("Evidence count per stage:")
for stage in range(1, 13):
    count = stage_coverage["evidence_count_per_stage"][stage]
    print(f"  Stage {stage:2d}: {count:4d} evidence items")

# Identify stages without evidence
if stage_coverage["stages_without_evidence"]:
    print(f"\nWarning: Stages {stage_coverage['stages_without_evidence']} produced no evidence")
```

### Querying Traceability Chains

```python
# Get evidence for a specific question
question_summary = validation_result["evidence_summary"]["D1-Q1"]

# Display complete traceability chain
print(f"Question D1-Q1 has {question_summary['evidence_count']} evidence sources:")
for source in question_summary["evidence_sources"]:
    print(f"\n  Evidence ID: {source['evidence_id']}")
    print(f"  Detector: {source['detector_type']}, Stage: {source['stage_number']}")
    print(f"  Component: {source['source_component']}")
    print(f"  Confidence: {source['confidence']:.2%}")
    print(f"  Timestamp: {source['execution_timestamp']}")
    print(f"  Quality: P={source['quality_metrics']['precision']:.2%}, "
          f"R={source['quality_metrics']['recall']:.2%}, "
          f"F1={source['quality_metrics']['f1']:.2%}")
```

## Integration with Detectors

### Detector Integration Pattern

Detectors should register evidence with provenance metadata:

```python
class MonetaryDetector:
    def __init__(self, registry: EvidenceRegistry):
        self.registry = registry
        self.stage_number = 1  # Configured stage number
        self.detector_type = "monetary"
    
    def detect_monetary_values(self, text: str, page_num: int, line_num: int) -> List[str]:
        # Perform detection
        monetary_values = self._extract_values(text)
        
        # Register each finding as evidence
        for value in monetary_values:
            provenance = EvidenceProvenance(
                detector_type=self.detector_type,
                stage_number=self.stage_number,
                source_text_location={
                    "page": page_num,
                    "line": line_num,
                    "char_start": value["start"],
                    "char_end": value["end"]
                },
                execution_timestamp=datetime.utcnow().isoformat() + "Z",
                quality_metrics={
                    "precision": 0.85,
                    "recall": 0.80,
                    "f1": 0.82
                }
            )
            
            self.registry.register(
                source_component=f"{self.detector_type}_stage_{self.stage_number}",
                evidence_type="monetary_value",
                content={"amount": value["amount"], "currency": value["currency"]},
                confidence=value["confidence"],
                applicable_questions=self._get_applicable_questions(value),
                provenance=provenance
            )
```

## Logging and Debugging

The validation layer provides detailed logging:

```python
# Questions below threshold are logged with details
logger.warning(
    "Question %s has only %d evidence sources (threshold: %d). "
    "Contributing stages: %s. Missing stages: %s",
    question_id,
    evidence_count,
    min_evidence_threshold,
    sorted(stage_contributions.keys()),
    sorted(missing_stages)
)
```

**Example log output:**
```
WARNING:evidence_registry:Question D1-Q2 has only 2 evidence sources (threshold: 3). 
Contributing stages: [1, 2]. Missing stages: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

## Performance Considerations

- **Validation Complexity**: O(N × M) where N = questions (300), M = avg evidence per question
- **Memory**: Full validation results stored in memory (~5-10 MB for 300 questions)
- **Export**: JSON export is atomic and deterministic

## Testing

Comprehensive test suite in `test_evidence_registry_determinism.py`:

```bash
# Run all validation tests
python3.10 test_evidence_registry_determinism.py

# Run specific validation test
python3.10 -m unittest test_evidence_registry_determinism.TestEvidenceValidationLayer
```

Test coverage includes:
- ✓ Validation with sufficient evidence
- ✓ Detection of insufficient evidence
- ✓ Stage contribution tracking
- ✓ Provenance metadata preservation
- ✓ Stage coverage statistics
- ✓ JSON export/import
- ✓ 300-question scalability

## Demonstration

Run the complete demonstration:

```bash
python3.10 demo_evidence_validation.py
```

This demonstrates:
- Evidence collection from 12 detector stages
- Validation across 300 questions
- Stage coverage analysis
- Traceability chain display
- Validation result export

## Best Practices

1. **Always Include Provenance**: Register evidence with complete provenance metadata
2. **Validate Post-Execution**: Run validation after all detectors complete
3. **Log Threshold Violations**: Review questions below threshold for debugging
4. **Export Results**: Save validation results for audit trail
5. **Monitor Stage Coverage**: Ensure all stages contribute evidence
6. **Track Quality Metrics**: Include precision/recall/F1 for evidence quality assessment

## Future Enhancements

Potential improvements:
- Weighted thresholds based on evidence quality
- Dynamic threshold adjustment based on question complexity
- Real-time validation during evidence collection
- Integration with alerting systems for threshold violations
- Stage dependency tracking (stage N requires stage N-1)
