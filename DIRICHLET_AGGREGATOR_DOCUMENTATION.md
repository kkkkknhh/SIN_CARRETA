# Dirichlet-Multinomial Evidence Aggregator

## Overview

This implementation adds Bayesian evidence aggregation to `evidence_registry.py` using Dirichlet-Multinomial conjugate posteriors. It enables multiple detectors to vote on categorical classifications (e.g., dimensions D1-D10) with proper uncertainty quantification.

## Key Components

### 1. DirichletAggregator Class

**Location**: `evidence_registry.py`

**Purpose**: Fuses categorical votes from multiple sources using Bayesian inference.

**Features**:
- Maintains Dirichlet posterior parameters (α)
- Updates with discrete labels or continuous weights
- Computes posterior statistics (mean, mode, credible intervals)
- Quantifies uncertainty via entropy
- Supports k categories (typically k=10 for D1-D10)

### 2. EvidenceRegistry Extensions

**New Attributes**:
- `dimension_aggregators`: Dict of DirichletAggregator per evidence_id (k=10)
- `content_type_aggregator`: Global aggregator for content types (k=5)
- `risk_level_aggregator`: Global aggregator for risk levels (k=3)

**New Methods**:
- `register_evidence()`: Register categorical votes with confidence weighting
- `get_dimension_distribution()`: Query posterior distribution for evidence
- `get_consensus_dimension()`: Check if consensus threshold is met

## API Reference

### DirichletAggregator

```python
# Initialize
agg = DirichletAggregator(k=10, alpha0=0.5)  # Jeffreys prior

# Update with discrete votes
agg.update_from_labels(np.array([3, 3, 4]))  # Votes for categories 3 and 4

# Update with continuous weights (confidence-weighted)
weights = np.zeros(10)
weights[3] = 0.85  # 85% confidence for category 3
agg.update_from_weights(weights)

# Query posterior
mean = agg.posterior_mean()  # E[θ]
mode = agg.posterior_mode()  # MAP estimate
intervals = agg.credible_interval(level=0.95)  # 95% credible intervals
entropy = agg.entropy()  # Uncertainty
max_cat, max_prob = agg.max_probability()  # Winner

# Reset to prior
agg.reset()
```

### EvidenceRegistry Integration

```python
registry = EvidenceRegistry()

# Register evidence with categorical vote
registry.register_evidence(
    evidence_id="evidence_001",
    source="responsibility_detector",
    dimension_vote=3,      # D4 (index 3)
    content_type=0,        # Type 0-4
    risk_level=1,          # 0=low, 1=medium, 2=high
    confidence=0.90        # Detector confidence
)

# Get posterior distribution
dist = registry.get_dimension_distribution("evidence_001")
# Returns: {
#   'mean': np.array([...]),
#   'credible_interval': np.array([[lo, hi], ...]),
#   'max_category': (idx, prob),
#   'entropy': float,
#   'n_votes': int
# }

# Check consensus
consensus = registry.get_consensus_dimension("evidence_001", threshold=0.6)
if consensus is not None:
    print(f"Consensus on D{consensus+1}")
```

## Integration with Detectors

### Example: responsibility_detector.py

```python
def detect_responsibilities(text, evidence_registry):
    entities = extract_entities(text)
    
    for entity in entities:
        evidence_id = f"resp_{hash(entity['text'])}"
        
        # Vote for D4 (responsibilities)
        evidence_registry.register_evidence(
            evidence_id=evidence_id,
            source="responsibility_detector",
            dimension_vote=3,  # D4 (index 3)
            content_type=0,
            risk_level=1,
            confidence=entity['confidence']
        )
```

### Example: Orchestrator Consolidation

```python
def consolidate_evidence(evidence_registry):
    for evidence_id in evidence_registry.dimension_aggregators:
        dist = evidence_registry.get_dimension_distribution(evidence_id)
        
        max_cat, max_prob = dist['max_category']
        print(f"Evidence {evidence_id}: D{max_cat+1} ({max_prob:.1%})")
        print(f"  Uncertainty: {dist['entropy']:.3f}")
        
        # Check consensus
        consensus = evidence_registry.get_consensus_dimension(
            evidence_id, threshold=0.6
        )
        
        if consensus is not None:
            print(f"  ✅ CONSENSUS on D{consensus+1}")
        else:
            print(f"  ⚠️  No clear consensus")
```

## Mathematical Details

### Dirichlet-Multinomial Model

**Prior**: Dir(α₀, ..., α₀) with α₀ = 0.5 (Jeffreys prior)

**Likelihood**: Multinomial(θ)

**Posterior**: Dir(α₀ + n₁, ..., α₀ + nₖ) where nᵢ = vote count for category i

**Properties**:
- Conjugate: Posterior is also Dirichlet
- Sequential updates: Can update incrementally as evidence arrives
- Uncertainty quantification: Entropy and credible intervals

### Posterior Statistics

- **Mean**: E[θᵢ] = αᵢ / Σⱼαⱼ
- **Mode** (MAP): (αᵢ - 1) / (Σⱼαⱼ - k) for αᵢ > 1
- **Credible Interval**: Monte Carlo sampling from Dirichlet posterior
- **Entropy**: -Σᵢ E[θᵢ] log E[θᵢ]

## Validation Results

### Tests
- **33 unit tests** covering all functionality
- All existing `evidence_registry` tests pass (18 tests)
- Comprehensive edge case coverage

### Performance
- Average aggregation time: ~16ms per evidence (including credible intervals)
- Scales well to 100+ evidences
- Thread-safe with Lock protection

### Metrics (from validation)
- ✓ Unanimous votes: Category dominance > 80%
- ✓ Split votes: Entropy > 0.5 (high uncertainty)
- ✓ Credible intervals: Proper shape and ordering
- ✓ Consensus detection: Works with threshold-based filtering

## Files Added

1. **evidence_registry.py** (modified)
   - Added DirichletAggregator class
   - Extended EvidenceRegistry with aggregators
   - ~150 lines of new code

2. **test_dirichlet_aggregator.py** (new)
   - 33 comprehensive tests
   - Tests for all major functionality

3. **example_dirichlet_usage.py** (new)
   - 5 usage examples
   - Demonstrates basic to advanced features

4. **validate_dirichlet_implementation.py** (new)
   - Validates requirements from problem statement
   - Performance metrics

5. **example_detector_integration.py** (new)
   - Integration patterns for detectors
   - Orchestrator consolidation example

## Usage in Pipeline

1. **Evidence Registration** (by detectors):
   ```python
   registry.register_evidence(evidence_id, source, dimension_vote, 
                              content_type, risk_level, confidence)
   ```

2. **Aggregation** (automatic):
   - Dirichlet posterior updated incrementally
   - Thread-safe accumulation

3. **Consolidation** (by orchestrator):
   ```python
   dist = registry.get_dimension_distribution(evidence_id)
   consensus = registry.get_consensus_dimension(evidence_id, threshold=0.6)
   ```

4. **Evaluation** (by answer_assembler):
   - Use consensus dimension if available
   - Use max_probability dimension otherwise
   - Mark as uncertain if entropy > 1.0

## Advantages over Simple Voting

1. **Uncertainty Quantification**: Entropy and credible intervals
2. **Confidence Weighting**: Votes weighted by detector confidence
3. **Prior Knowledge**: Incorporates Jeffreys prior (α₀ = 0.5)
4. **Algebraic Coherence**: Conjugate updates preserve posterior form
5. **Sequential Updates**: Can update incrementally as evidence arrives
6. **Probabilistic Interpretation**: Proper probability distributions

## Dependencies

- `numpy >= 2.1.0`
- `scipy >= 1.7.0`

Both already in `requirements.txt`.

## Backward Compatibility

- All existing `EvidenceRegistry` methods unchanged
- New methods are additions only
- Existing code continues to work without modification
- Optional integration with new Bayesian features

## Future Enhancements

Potential extensions (not implemented):
- Hierarchical Dirichlet Process for automatic category discovery
- Beta-Binomial for binary consensus (alternative to Dirichlet-Multinomial)
- Variational inference for faster credible intervals
- Online learning with forgetting factor for concept drift

## References

- Conjugate Bayesian analysis of the Dirichlet distribution (Gelman et al., 2013)
- Jeffreys prior: α₀ = 0.5 for multinomial (Jeffreys, 1946)
- Dirichlet-Multinomial model (Murphy, 2012, Machine Learning: A Probabilistic Perspective)
