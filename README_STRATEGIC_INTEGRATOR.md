# Strategic Decalogo Integrator - Quick Start Guide

## Overview

The Strategic Decalogo Integrator is a doctoral-level evidence analysis system that integrates evidence from multiple sources across 300 evaluation questions organized in 6 dimensions (D1-D6) for Municipal Development Plan (PDM) assessment.

## Installation

### Prerequisites
- Python 3.8+ (tested on 3.10, 3.12)
- pip package manager

### Install Dependencies

```bash
pip install networkx scipy sentence-transformers torch numpy
```

**Note**: The sentence-transformers package will download the Multi-QA model (~500MB) on first use.

## Quick Start

### 1. Basic Usage

```python
from strategic_decalogo_integrator import StrategicDecalogoIntegrator
from evidence_registry import EvidenceRegistry
from pathlib import Path

# Step 1: Create and populate evidence registry
registry = EvidenceRegistry()

# Register evidence from your pipeline components
registry.register(
    source_component="feasibility_scorer",
    evidence_type="baseline_presence",
    content={"score": 0.85, "text": "Línea base documentada"},
    confidence=0.90,
    applicable_questions=["D1-Q1", "D1-Q2"]
)

# Step 2: Create integrator
integrator = StrategicDecalogoIntegrator(
    evidence_registry=registry,
    documento_plan="Plan de Desarrollo Municipal text...",
    nombre_plan="PDM_Florencia_2024",
    mapping_config_path=Path("integration_mapping.json")
)

# Step 3: Execute complete analysis
results = integrator.execute_complete_analysis()

# Step 4: Check results
print(f"Dimensions analyzed: {len(results['dimensions'])}")
print(f"Quality gates passed: {all(results['quality_gates_passed'].values())}")
```

### 2. Generate Reports

```python
from strategic_decalogo_integrator import (
    generate_metrics_report,
    export_results_to_json
)

# Generate human-readable report
report = generate_metrics_report(results)
print(report)

# Export to JSON
export_results_to_json(results, Path("integration_results.json"))
```

### 3. Using Individual Components

#### Semantic Extraction

```python
from strategic_decalogo_integrator import SemanticExtractor

extractor = SemanticExtractor()

# Extract relevant evidence
segments = [
    "El presupuesto total asciende a $500 millones",
    "Se identifican fuentes de financiación"
]

results = extractor.extract_evidence(
    query="¿Se identifican recursos financieros?",
    segments=segments,
    top_k=10
)

# Results contain only segments with similarity >= 0.75
for text, score in results:
    print(f"Score: {score:.3f} - {text}")
```

#### Causal Graph Analysis

```python
from strategic_decalogo_integrator import CausalGraphAnalyzer
import networkx as nx

analyzer = CausalGraphAnalyzer()

# Create causal graph
G = nx.DiGraph()
G.add_edges_from([
    ('Insumos', 'Actividades', {'weight': 0.8}),
    ('Actividades', 'Productos', {'weight': 0.7}),
    ('Productos', 'Resultados', {'weight': 0.9})
])

# Analyze
result = analyzer.analyze_dimension(G, 'D1')

if result.valid:
    print(f"Valid DAG with p-value: {result.acyclicity_pvalue:.3f}")
    print(f"Average causal strength: {result.avg_causal_strength:.3f}")
else:
    print(f"Invalid: {result.rejection_reason}")
```

#### Bayesian Evidence Integration

```python
from strategic_decalogo_integrator import (
    BayesianEvidenceIntegrator,
    StructuredEvidence
)

integrator = BayesianEvidenceIntegrator()

# Create evidence list
evidence = [
    StructuredEvidence(
        question_id="D1-Q1",
        dimension="D1",
        evidence_type="quantitative",
        raw_evidence={"score": 0.85},
        processed_content={"score": 0.85},
        confidence=0.90,
        source_module="feasibility_scorer"
    ),
    StructuredEvidence(
        question_id="D1-Q1",
        dimension="D1",
        evidence_type="qualitative",
        raw_evidence={"present": True},
        processed_content={"score": 0.80},
        confidence=0.85,
        source_module="plan_processor"
    )
]

# Integrate
result = integrator.integrate_evidence(evidence, "D1-Q1")

print(f"Posterior mean: {result['posterior_mean']:.3f}")
print(f"95% Credible interval: {result['credible_interval_95']}")
print(f"Conflict detected: {result['evidence_conflict_detected']}")
```

## Understanding Results

### Result Structure

```python
{
    "plan_name": "PDM_Municipality_2024",
    "analysis_timestamp": "2024-10-12T12:34:56.789Z",
    "dimensions": {
        "D1": {
            "dimension_score": 0.72,
            "questions_analyzed": 50,
            "question_evidence": {
                "D1-Q1": {
                    "evidence_count": 2,
                    "bayesian_integration": {
                        "posterior_mean": 0.85,
                        "credible_interval_95": [0.68, 0.95],
                        "evidence_conflict_detected": false
                    }
                },
                ...
            }
        },
        ...
    },
    "performance_metrics": {
        "total_questions": 300,
        "questions_with_evidence": 285,
        "avg_evidence_confidence": 0.78,
        "pct_evidence_conflicts": 0.12
    },
    "quality_gates_passed": {
        "all_300_questions_mapped": true,
        "all_6_dimensions_analyzed": true,
        "avg_confidence_acceptable": true,
        ...
    }
}
```

### Quality Gates

The system enforces 5 quality gates:

1. **Gate 1**: Semantic similarity ≥ 0.75 (BEIR-validated)
2. **Gate 2**: Graph acyclicity p-value > 0.95
3. **Gate 3**: Posterior confidence with credible intervals
4. **Gate 4**: All 6 dimensions scored
5. **Gate 5**: Complete provenance tracking

All gates must pass for acceptance.

## Configuration

### Integration Mapping

The `integration_mapping.json` file maps 300 questions to evidence modules:

```json
{
  "version": "1.0",
  "total_questions": 300,
  "dimensions": ["D1", "D2", "D3", "D4", "D5", "D6"],
  "questions": {
    "D1-Q1": {
      "dimension": "D1",
      "primary_modules": ["feasibility_scorer", "monetary_detector"],
      "evidence_types": ["baseline_presence", "diagnostic_data"]
    },
    ...
  }
}
```

You can customize this mapping for your specific pipeline.

## Testing

### Run Test Suite

```bash
# Run all tests
python test_decalogo_integrator.py

# Run specific test class
python -m unittest test_decalogo_integrator.TestSemanticExtractionThresholdEnforcement -v
```

### Expected Output

```
test_threshold_enforcement ... ok
test_irrelevant_segment_exclusion ... ok
test_deterministic_ordering ... ok
...
----------------------------------------------------------------------
Ran 40 tests in 2.345s

OK
```

## Troubleshooting

### Model Download Issues

If you get network errors downloading the semantic model:

```python
# Option 1: Pre-download the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Option 2: Use cached model
# Set environment variable
export SENTENCE_TRANSFORMERS_HOME=/path/to/cache
```

### Memory Issues

For large documents, increase memory limits:

```python
# Process document in chunks
chunk_size = 100  # segments per chunk
for i in range(0, len(segments), chunk_size):
    chunk = segments[i:i+chunk_size]
    results = extractor.extract_evidence(query, chunk)
```

### Offline Mode

For testing without network access:

```python
# Skip semantic extraction (other components work offline)
from strategic_decalogo_integrator import (
    CausalGraphAnalyzer,
    BayesianEvidenceIntegrator,
    DecalogoEvidenceExtractor
)

# Use only offline components
analyzer = CausalGraphAnalyzer()
integrator = BayesianEvidenceIntegrator()
extractor = DecalogoEvidenceExtractor(registry)
```

## Performance Optimization

### Caching

```python
# Cache semantic model
from sentence_transformers import SentenceTransformer

# Load once, reuse
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
extractor = SemanticExtractor()
extractor.model = model  # Reuse loaded model
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_dimension(dim_id):
    # Extract and analyze dimension
    ...
    return results

# Process dimensions in parallel
with ThreadPoolExecutor(max_workers=6) as executor:
    dimension_results = list(executor.map(
        analyze_dimension,
        ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    ))
```

## Academic References

For detailed methodology, see `DESIGN_RATIONALE.md`:

1. **Semantic Similarity**: Thakur et al. (2021), BEIR benchmark
2. **Causal Inference**: Pearl (2009), "Causality"
3. **Bayesian Analysis**: Gelman et al. (2013), "Bayesian Data Analysis"

## Support

For issues or questions:
- Check `DESIGN_RATIONALE.md` for algorithm details
- Review `test_decalogo_integrator.py` for usage examples
- See `STRATEGIC_INTEGRATOR_SUMMARY.md` for implementation summary

## License

Part of the MINIMINIMOON PDM evaluation system.

---

**Version**: 1.0.0  
**Last Updated**: October 12, 2025  
**Status**: Production Ready ✅
