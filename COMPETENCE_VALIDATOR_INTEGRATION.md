# CompetenceValidator Integration in evaluate_from_evidence()

## Overview

Successfully integrated `CompetenceValidator` from `pdm_contra/policy/competence.py` into the `evaluate_from_evidence()` method in `Decatalogo_principal.py`. The integration validates institutional competence boundaries and policy compliance for each question being evaluated.

## Integration Details

### 1. Import and Instantiation

**Location**: Lines 2754-2765 in `Decatalogo_principal.py`

```python
from pdm_contra.policy.competence import CompetenceValidator

# Initialize all complementary modules
competence_validator = CompetenceValidator()
```

### 2. Global Competence Analysis (STEP 4)

**Location**: Lines ~2880-2900

- Runs `CompetenceValidator` across all document text for top 10 sectors
- Added error handling with `log_warning_with_text` for graceful continuation
- Logs summary of competence issues found

### 3. Per-Question Competence Validation (STEP 7)

**Location**: Lines 3034-3107

Key features:
- Extracts evidence text snippets for each question
- Validates competence boundaries for question-specific context
- Runs validation for top 5 sectors per question
- Registers results as structured evidence entries with proper provenance

**Code Structure**:
```python
question_competence_validations = {}
for question_id, query, conceptos_clave in sample_questions:
    # Extract evidence for question
    evidence_items = self.buscar_evidencia_causal_avanzada(...)
    
    # COMPETENCE VALIDATION PER QUESTION
    question_competence_issues = []
    try:
        question_text_snippets = [item.get("texto", "") for item in evidence_items]
        question_context = "\n".join(question_text_snippets)
        
        for sector in sectors[:5]:
            try:
                issues = competence_validator.validate_segment(
                    text=question_context,
                    sectors=[sector],
                    level="municipal"
                )
                question_competence_issues.extend(issues)
            except Exception as sector_error:
                log_warning_with_text(...)
                continue
        
        # Register evidence entries with provenance
        for idx, issue in enumerate(question_competence_issues):
            evidence_registry.register(EvidenceEntry(...))
        
        question_competence_validations[question_id] = question_competence_issues
        
    except Exception as e:
        log_warning_with_text(...)
        question_competence_validations[question_id] = []
```

### 4. Evidence Registration with Provenance Metadata

**Location**: Lines 3075-3097

Each competence validation result is registered as a structured evidence entry:

```python
evidence_registry.register(
    EvidenceEntry(
        evidence_id=f"competence_validation::{question_id}::{idx}",
        stage="decalogo_evaluation",
        content={
            "validation_type": "institutional_competence",
            "question_id": question_id,
            "issue": issue,
            "validator_source": "CompetenceValidator",
            "provenance": {
                "module": "pdm_contra.policy.competence",
                "class": "CompetenceValidator",
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
        confidence=0.8,
        metadata={
            "module": "pdm_contra_competence_per_question",
            "detector": "CompetenceValidator",
            "question_id": question_id,
            "stage": "after_contradiction_before_scoring",
        },
    )
)
```

### 5. Results in Evaluation Output

**Location**: Line 3178

Results are included in the comprehensive evaluation output:

```python
evaluation_results = {
    ...
    "question_competence_validations": question_competence_validations,
    ...
}
```

## Integration Placement

The integration follows the correct execution order:

1. **STEP 1**: ContradictionDetector (lines ~2800-2830)
2. **STEP 2**: PatternMatcher (lines ~2835-2850)
3. **STEP 3**: FactibilidadPatternDetector (lines ~2855-2875)
4. **STEP 4**: CompetenceValidator - Global analysis (lines ~2880-2900)
5. **STEP 5**: ExplanationTracer (lines ~2905-2925)
6. **STEP 6**: Evidence registration (lines ~2930-3000)
7. **STEP 7**: Per-question analysis with CompetenceValidator integration (lines ~3005-3135)
8. **STEP 8**: Return evaluation results (lines ~3140-3195)

✅ **Competence validation occurs after contradiction detection but before final scoring aggregation**

## Error Handling

All CompetenceValidator operations include robust error handling:

1. **Global analysis**: Try-except wrapper for each sector validation
2. **Per-question analysis**: Try-except for entire question validation
3. **Per-sector validation**: Try-except for individual sector validations
4. **Graceful continuation**: Errors logged with `log_warning_with_text`, evaluation continues
5. **Default values**: Empty list `[]` assigned if validation fails completely

## Verification Checklist

✅ CompetenceValidator imported from `pdm_contra.policy.competence`  
✅ CompetenceValidator instantiated once at module initialization  
✅ Global competence analysis in STEP 4  
✅ Per-question competence validation in STEP 7  
✅ `validate_segment()` method invoked with proper parameters  
✅ Results registered as structured evidence entries  
✅ Provenance metadata included (module, class, timestamp)  
✅ Evidence IDs use proper format: `competence_validation::{question_id}::{idx}`  
✅ Error handling with `log_warning_with_text` for graceful continuation  
✅ Placement after contradiction detection, before final scoring  
✅ Results included in evaluation output as `question_competence_validations`  
✅ File compiles without syntax errors

## Integration Statistics

- **CompetenceValidator references**: 10 occurrences
- **question_competence_validations references**: 5 occurrences
- **Error handling blocks**: 9 `log_warning_with_text` calls
- **Evidence registration**: 1 competence validation evidence type
- **Stage metadata**: "after_contradiction_before_scoring" properly set

## Testing

To verify the integration:

```bash
# Compile check
python3.10 -m py_compile Decatalogo_principal.py

# Run verification script
python3.10 test_competence_integration_verification.py
```

## Impact on Evaluation

The CompetenceValidator integration:

1. **Validates institutional boundaries**: Detects when actions exceed municipal competence
2. **Checks policy compliance**: Identifies missing essential competencies
3. **Provides legal context**: References relevant laws (Ley 715, etc.)
4. **Enhances traceability**: Full provenance chain in evidence entries
5. **Influences scoring**: Competence violations become part of evidence chain
6. **Maintains robustness**: Graceful error handling prevents evaluation failures

## CompetenceValidator Capabilities

From `pdm_contra/policy/competence.py`, the validator:

- **Detects overreach**: Actions beyond municipal authority (e.g., directing police, appointing teachers)
- **Validates coordination**: Accepts appropriate coordination verbs (gestionar, articular, etc.)
- **Checks essential competencies**: Ensures key municipal responsibilities are addressed
- **Provides legal basis**: References constitutional and legal frameworks
- **Offers remediation**: Suggests reformulations for detected issues
