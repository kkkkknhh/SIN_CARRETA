# Answer Assembler Doctoral Argumentation Integration

## Summary

Modified `answer_assembler.py` to integrate with `doctoral_argumentation_engine.py` for generating rigorous, doctoral-level justifications for each evaluation question with complete Toulmin argumentation structure and quality validation.

## Key Enhancements

### 1. Doctoral Argumentation Engine Integration
- **Calls `generate_argument()`** for each question during answer assembly
- **Passes accumulated evidence** from `evidence_registry` to the doctoral engine
- **Converts evidence format** to `StructuredEvidence` for engine compatibility
- **Creates Bayesian posterior** from confidence scores for statistical rigor

### 2. Evidence Validation (≥3 Sources Requirement)
- **GATE 1: Pre-generation validation** checks each question has ≥3 evidence sources
- **Rejects with status `insufficient_evidence`** if evidence count < 3
- **Adds caveat** to question answer explaining insufficient evidence
- **Tracks failures** in `argumentation_failures` list with detailed reason

### 3. Complete Toulmin Structure Validation
- **GATE 2: Structural completeness check** after generation
- **Validates all 6 components present**:
  - `claim` - Falsifiable statement addressing question
  - `ground` - Primary evidence citation
  - `warrant` - Logical bridge connecting ground to claim
  - `backing` - Additional supporting sources (≥2)
  - `qualifier` - Confidence quantification
  - `rebuttal` - Addresses counterclaims
- **Rejects with `validation_failed`** if any component missing or empty

### 4. Quality Threshold Enforcement
- **GATE 3: Coherence score validation** (≥0.85)
  - Checks `logical_coherence_score` from doctoral engine
  - Validates logical structure and absence of fallacies
  - Rejects arguments with circular reasoning, non-sequiturs
- **GATE 4: Quality score validation** (≥0.80)
  - Checks `academic_quality_scores.overall_score`
  - Validates precision, objectivity, hedging, citations, coherence, sophistication
  - Rejects arguments failing academic writing standards

### 5. Comprehensive Output Serialization
Each successfully generated argument includes:

```python
{
    "question_id": "P1-D1-Q1",
    "argumentation_status": "success",
    "doctoral_justification": {
        "paragraphs": [...],  # 3 paragraphs
        "claim": "...",
        "ground": "...",
        "warrant": "...",
        "backing": [...],     # List of supporting statements
        "qualifier": "...",
        "rebuttal": "..."
    },
    "toulmin_structure": {
        "claim": "...",
        "ground": "...",
        "warrant": "...",
        "backing": [...],
        "qualifier": "...",
        "rebuttal": "...",
        "evidence_sources": [...],
        "confidence_lower": 0.75,
        "confidence_upper": 0.95,
        "logical_coherence_score": 0.92
    },
    "argument_quality": {
        "coherence_score": 0.92,
        "quality_score": 0.87,
        "academic_quality_breakdown": {
            "precision": 0.90,
            "objectivity": 0.88,
            "hedging": 0.85,
            "citations": 0.91,
            "coherence": 0.86,
            "sophistication": 0.82,
            "overall_score": 0.87
        },
        "evidence_sources": [...],
        "evidence_synthesis_map": {...},
        "meets_doctoral_standards": true,
        "confidence_alignment_error": 0.023,
        "validation_timestamp": "2024-01-15T10:30:45.123456"
    }
}
```

### 6. Status Tracking and Failure Reporting
All questions receive an `argumentation_status`:
- `success` - All validations passed, doctoral argument generated
- `insufficient_evidence` - < 3 evidence sources
- `validation_failed` - Failed coherence or quality thresholds
- `generation_failed` - Engine error during generation
- `not_attempted` - Engine unavailable or not applicable

Global summary includes:
```python
{
    "doctoral_argumentation": {
        "coverage": 245,  # Questions with doctoral arguments
        "coverage_percentage": 81.7,
        "high_quality_count": 238,
        "high_quality_percentage": 97.1,
        "status_breakdown": {
            "success": 245,
            "insufficient_evidence": 42,
            "validation_failed": 8,
            "generation_failed": 3,
            "not_attempted": 2
        },
        "average_coherence_score": 0.894,
        "average_quality_score": 0.856,
        "validation_thresholds": {
            "min_evidence_sources": 3,
            "min_coherence_score": 0.85,
            "min_quality_score": 0.80
        },
        "failures": [...]  # Detailed failure records
    }
}
```

## Implementation Details

### Code Structure
```python
# answer_assembler.py modifications

def assemble(self, evaluation_inputs):
    """Enhanced with doctoral argumentation integration"""
    
    # Import doctoral engine
    from doctoral_argumentation_engine import (
        DoctoralArgumentationEngine,
        StructuredEvidence
    )
    
    # Initialize engine
    doctoral_engine = DoctoralArgumentationEngine(
        evidence_registry=self.evidence_registry
    )
    
    for question_result in question_results:
        evidence_list = self._get_evidence_for_question(question_id)
        
        # GATE 1: Evidence count validation
        if len(evidence_list) < 3:
            argumentation_status = "insufficient_evidence"
            continue
        
        # Convert to StructuredEvidence
        structured_evidence = [
            StructuredEvidence(
                source_module=f"detector_{idx}",
                evidence_type="textual",
                content=ev.text,
                confidence=ev.confidence,
                applicable_questions=[question_id],
                metadata={}
            )
            for idx, ev in enumerate(evidence_list)
        ]
        
        # Generate argument
        try:
            argument_result = doctoral_engine.generate_argument(
                question_id=question_id,
                score=score,
                evidence_list=structured_evidence,
                bayesian_posterior={
                    "posterior_mean": confidence,
                    "credible_interval_95": (lower, upper)
                }
            )
            
            toulmin_structure = argument_result["toulmin_structure"]
            
            # GATE 2: Toulmin completeness validation
            required_fields = ["claim", "ground", "warrant", "backing", "qualifier", "rebuttal"]
            missing_fields = [f for f in required_fields if not toulmin_structure.get(f)]
            if missing_fields:
                raise ValueError(f"Incomplete Toulmin: {missing_fields}")
            
            # GATE 3: Coherence threshold
            coherence_score = argument_result["logical_coherence_score"]
            if coherence_score < 0.85:
                raise ValueError(f"Coherence {coherence_score} < 0.85")
            
            # GATE 4: Quality threshold
            quality_score = argument_result["academic_quality_scores"]["overall_score"]
            if quality_score < 0.80:
                raise ValueError(f"Quality {quality_score} < 0.80")
            
            # Success - assemble output
            argumentation_status = "success"
            doctoral_argument = {...}  # Full structure
            
        except ValueError as e:
            argumentation_status = "validation_failed"
            # Log failure details
        except Exception as e:
            argumentation_status = "generation_failed"
            # Log error
```

## Testing

### Integration Test (`test_answer_assembler_integration.py`)
- **Tests insufficient evidence rejection** (< 3 sources)
- **Validates Toulmin completeness** (all 6 components)
- **Checks quality thresholds** (coherence ≥ 0.85, quality ≥ 0.80)
- **Verifies status tracking** (success, insufficient, validation_failed)
- **Validates serialization** (all fields present and properly formatted)

### Validation Script (`validate_answer_assembler_enhancements.py`)
- **Code pattern validation** (checks for all required patterns)
- **Structure validation** (Toulmin fields, thresholds, error handling)
- **Integration validation** (test file exists and comprehensive)
- **Usage demonstration** (example code for downstream consumers)

## Quality Gates Summary

| Gate | Check | Threshold | Action on Failure |
|------|-------|-----------|-------------------|
| 1 | Evidence count | ≥ 3 sources | Reject: `insufficient_evidence` |
| 2 | Toulmin completeness | All 6 components non-empty | Reject: `validation_failed` |
| 3 | Logical coherence | ≥ 0.85 | Reject: `validation_failed` |
| 4 | Academic quality | ≥ 0.80 | Reject: `validation_failed` |

## Downstream Traceability

The enhanced output provides complete traceability:
1. **Evidence sources** - List of all detectors/modules contributing evidence
2. **Evidence synthesis map** - Which evidence supports which Toulmin component
3. **Quality metrics** - All 6 academic quality dimensions + overall score
4. **Validation timestamp** - When argument was generated and validated
5. **Confidence alignment** - Error between stated and Bayesian confidence
6. **Argumentation status** - Success/failure reason for each question
7. **Failure records** - Detailed error messages for debugging

## Compatibility

- **Backward compatible** - Falls back to basic rationale if doctoral engine unavailable
- **Graceful degradation** - Logs warnings but continues processing
- **Optional dependency** - Works without doctoral_argumentation_engine.py
- **Evidence registry agnostic** - Supports multiple evidence registry interfaces

## Files Modified

1. **answer_assembler.py** - Main implementation with all enhancements
2. **test_answer_assembler_integration.py** - Comprehensive integration test (NEW)
3. **validate_answer_assembler_enhancements.py** - Validation script (NEW)
4. **ANSWER_ASSEMBLER_ENHANCEMENTS.md** - This documentation (NEW)

## Verification Commands

```bash
# Compile check
python3.10 -m py_compile answer_assembler.py

# Run integration test
python3.10 test_answer_assembler_integration.py

# Validate enhancements (if execution enabled)
python3.10 validate_answer_assembler_enhancements.py
```

## Next Steps

1. **Enable doctoral_argumentation_engine** - Ensure module is importable in deployment
2. **Configure evidence registry** - Provide proper evidence source integration
3. **Adjust thresholds** - Fine-tune coherence/quality thresholds based on domain
4. **Monitor failures** - Track `argumentation_failures` for systematic issues
5. **Validate outputs** - Verify downstream systems can consume enhanced structure
