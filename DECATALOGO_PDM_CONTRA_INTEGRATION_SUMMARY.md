# Decatalogo_principal.py - PDM Contra Integration Summary

## Overview

Successfully modified the `evaluate_from_evidence()` method in `Decatalogo_principal.py` to integrate the complete pdm_contra detection pipeline for Stage 14 contradiction analysis. The method now instantiates and invokes all required detectors before scoring each decalogue principle, enriching evidence with comprehensive contradiction analysis results.

## Key Changes

### 1. PDM Contra Detectors Integration (Lines 2747-2781)

The method now instantiates all required pdm_contra detectors:

```python
# Initialize all complementary modules
contradiction_detector = ContradictionDetector(mode_light=True)
risk_scorer = RiskScorer(alpha=0.1)
pattern_matcher = PatternMatcher(language="es")
nli_detector = SpanishNLIDetector(light_mode=True)
competence_validator = CompetenceValidator()
explanation_tracer = ExplanationTracer(language="es")
factibilidad_detector = FactibilidadPatternDetector()
```

### 2. Detector Execution Pipeline (Lines 2787-2871)

Each detector is invoked sequentially with proper evidence extraction:

- **ContradictionDetector**: Full contradiction analysis with sector filtering
- **PatternMatcher**: Adversative linguistic pattern detection
- **FactibilidadPatternDetector**: Baseline, target, and timeframe pattern detection
- **CompetenceValidator**: Institutional competence validation across sectors
- **ExplanationTracer**: Human-readable explanations generation
- **NLIDetector**: Natural language inference (integrated within ContradictionDetector)

### 3. Evidence Enrichment Dictionary (Lines 3112-3145)

Built comprehensive enriched evidence dictionary containing:

```python
enriched_evidence_dict = {
    "contradictions": [...],           # From ContradictionDetector
    "risk_scores": {...},              # From RiskScorer
    "patterns": {...},                 # From PatternMatcher & FactibilidadDetector
    "nli_results": {...},              # From NLIDetector
    "trace_info": {...},               # From ExplanationTracer
    "competence_issues": [...],        # From CompetenceValidator
    "causal_evidence": {...},          # From buscar_evidencia_causal_avanzada
}
```

### 4. Evidence Registry Integration (Lines 2900-2997)

All detector outputs are registered with proper Stage 14 provenance:

```python
evidence_registry.register(
    EvidenceEntry(
        evidence_id=f"pdm_contra_contradiction::{idx}",
        stage="decalogo_evaluation",
        content={...},
        confidence=getattr(contradiction, "confidence", 0.5),
        metadata={
            "module": "pdm_contra_contradiction",
            "detector": "ContradictionDetector",
        },
    )
)
```

Evidence types registered:
- Contradictions (with confidence, risk level, explanations)
- Competence issues (sector-specific)
- Adversative patterns (linguistic markers)
- Factibilidad patterns (baseline, target, timeframe)
- Risk analysis (global risk score with confidence intervals)
- Causal evidence (per-question enriched evidence)

### 5. Dimension Scoring with Enriched Evidence (Lines 3058-3197)

New **STEP 8** introduced to score each of the 6 decalogue dimensions (D1-D6) using enriched evidence:

```python
dimensiones_decalogo = {
    "D1": {"id": "D1", "nombre": "INSUMOS", ...},
    "D2": {"id": "D2", "nombre": "ACTIVIDADES", ...},
    "D3": {"id": "D3", "nombre": "PRODUCTOS", ...},
    "D4": {"id": "D4", "nombre": "RESULTADOS", ...},
    "D5": {"id": "D5", "nombre": "IMPACTOS", ...},
    "D6": {"id": "D6", "nombre": "CAUSALIDAD", ...},
}

for dim_id, dimension_info in dimensiones_decalogo.items():
    # Extract dimension-specific evidence
    dimension_evidence = self._extract_dimension_evidence(...)
    
    # Calculate score with enrichment impact
    dimension_score = self._calculate_dimension_score(...)
    
    # Store with Stage 14 provenance
    evidence_registry.register(EvidenceEntry(
        evidence_id=f"stage_14_dimension_score::{dim_id}",
        stage="stage_14_contradiction_analysis",
        metadata={"provenance": "Stage 14 - Contradiction Analysis Pipeline"},
        ...
    ))
```

### 6. New Helper Methods

#### `_extract_dimension_evidence()` (Lines 3279-3324)

Extracts evidence relevant to a specific dimension from enriched evidence:

- Filters contradictions by dimension concepts
- Filters adversative patterns by relevance
- Filters competence issues by dimension scope
- Retrieves causal evidence for dimension questions

**Returns:**
```python
{
    "dimension_id": dim_id,
    "contradictions": [...],
    "patterns": [...],
    "competence_issues": [...],
    "causal_evidence": {...},
    "risk_level": "...",
    "overall_risk": 0.0-1.0,
}
```

#### `_calculate_dimension_score()` (Lines 3326-3392)

Calculates dimension score using enriched evidence from pdm_contra pipeline:

**Scoring Formula:**
```python
final_score = (
    base_score                  # Evidence presence (0-1)
    + pattern_bonus             # Detailed analysis indicator (max 0.2)
    - contradiction_penalty     # Contradiction count (max -0.5)
    - competence_penalty        # Competence issues (max -0.3)
    - risk_penalty              # Risk level penalty (0-0.4)
)
```

**Returns:**
```python
{
    "score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "evidence_count": int,
    "raw_scores": {...},
    "enrichment_impact": {...},
}
```

### 7. Enhanced Return Format (Lines 3199-3274)

Updated return structure to include dimension scores while maintaining Stage 15 compatibility:

```python
evaluation_results = {
    "status": "completed",
    "timestamp": datetime.utcnow().isoformat(),
    "summary": {
        "total_contradictions": int,
        "total_competence_issues": int,
        "total_agenda_gaps": int,
        "adversative_patterns": int,
        "factibilidad_patterns": {...},
        "factibilidad_clusters": int,
        "risk_score": float,
        "risk_level": str,
        "dimensions_evaluated": 6,                    # NEW
        "average_dimension_score": float,             # NEW
    },
    "detailed_analysis": {...},
    "dimension_scores": {...},                         # NEW
    "explanations": [...],
    "trace_report": str,
    "question_enriched_evidence": {...},
    "reliability_calibrators": {...},
    "modules_executed": [...],
    "enriched_evidence_summary": {...},               # NEW
}
```

## Stage 14 Provenance Metadata

All enriched evidence entries include Stage 14 provenance:

```python
metadata={
    "stage": "stage_14_contradiction_analysis",
    "module": "DecatalogoEvaluator",
    "detector": "pdm_contra_pipeline",
    "dimension": dim_id,
    "provenance": "Stage 14 - Contradiction Analysis Pipeline",
}
```

## Downstream Compatibility (Stage 15)

The return format maintains full backward compatibility:
- Original keys preserved (`status`, `timestamp`, `summary`, `detailed_analysis`)
- New keys added non-disruptively (`dimension_scores`, `enriched_evidence_summary`)
- Evidence registry properly populated for Stage 15 consumption
- All detector outputs traceable via `modules_executed` list

## Detectors Instantiated and Invoked

1. **ContradictionDetector** (pdm_contra/core.py)
   - Hybrid heuristics + NLI detection
   - Sector-specific analysis
   - Risk level assignment

2. **RiskScorer** (pdm_contra/scoring/risk.py)
   - Aggregate risk scoring
   - Confidence interval calculation
   - Risk level stratification

3. **PatternMatcher** (pdm_contra/nlp/patterns.py)
   - Adversative linguistic patterns
   - Context window extraction
   - Pattern confidence scoring

4. **SpanishNLIDetector** (pdm_contra/nlp/nli.py)
   - Natural language inference
   - Contradiction detection
   - Lightweight lexical heuristics

5. **CompetenceValidator** (pdm_contra/policy/competence.py)
   - Institutional competence validation
   - Sector-level analysis
   - Competence mismatch detection

6. **ExplanationTracer** (pdm_contra/explain/tracer.py)
   - Human-readable explanations
   - Trace report generation
   - Action logging

7. **FactibilidadPatternDetector** (factibilidad/pattern_detector.py)
   - Baseline pattern detection
   - Target pattern extraction
   - Timeframe identification

8. **ReliabilityCalibrator** (evaluation/reliability_calibration.py)
   - Bayesian calibration
   - Precision/recall tracking
   - F1 score computation

## Evidence Flow

```
Raw Evidence (Stages 1-12)
    ↓
evaluate_from_evidence()
    ↓
Detector Instantiation
    ↓
Full Text Analysis (ContradictionDetector, PatternMatcher, etc.)
    ↓
Enriched Evidence Dictionary
    ↓
Per-Dimension Evidence Extraction
    ↓
Dimension Score Calculation
    ↓
Evidence Registry (Stage 14 Provenance)
    ↓
Evaluation Results (Stage 15 Compatible)
```

## Validation

- **Bytecode Compilation**: ✓ Successful
- **Method Signatures**: ✓ Validated
- **Evidence Registry Integration**: ✓ 7 registration points
- **Stage 14 Provenance**: ✓ Proper metadata
- **Downstream Compatibility**: ✓ Stage 15 format maintained
- **Detector Count**: ✓ 8 detectors integrated

## Files Modified

- `Decatalogo_principal.py` (265 lines added/modified)
  - `evaluate_from_evidence()` method enhanced
  - `_extract_dimension_evidence()` method added
  - `_calculate_dimension_score()` method added

## Files Created

- `test_decatalogo_pdm_contra_integration.py` (integration test suite)
- `DECATALOGO_PDM_CONTRA_INTEGRATION_SUMMARY.md` (this document)

## Summary

The `evaluate_from_evidence()` method now provides a complete Stage 14 contradiction analysis pipeline that:

1. ✅ Instantiates all pdm_contra detectors
2. ✅ Invokes detectors on full document text and sectors
3. ✅ Enriches evidence with contradictions, risk scores, patterns, NLI results, and traces
4. ✅ Extracts dimension-specific evidence from enriched results
5. ✅ Calculates dimension scores with enrichment impact metrics
6. ✅ Stores all evidence with Stage 14 provenance metadata
7. ✅ Returns Stage 15-compatible evaluation results with dimension scores
8. ✅ Maintains existing evidence registry integration
9. ✅ Provides full traceability via ExplanationTracer
10. ✅ Supports downstream consumption with proper metadata
