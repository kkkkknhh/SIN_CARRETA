# Bayesian Causal Effect Estimator

## Overview

The Bayesian Causal Effect Estimator provides a statistically rigorous framework for quantifying causal relationships between variables in public policy evaluation. It uses **conjugate Normal-InverseGamma priors** for Bayesian linear regression, enabling closed-form posterior inference with full uncertainty quantification.

## Key Features

### 1. **Bayesian Linear Regression** (`BayesLinearEffect`)
- **Conjugate Prior**: Normal-InverseGamma distribution for efficient computation
- **Closed-Form Posterior**: No MCMC sampling required
- **Credible Intervals**: t-student marginals for coefficient uncertainty
- **Prediction with Uncertainty**: Full predictive distribution
- **Significance Testing**: Bayesian hypothesis testing for causal effects

### 2. **Effects Logger** (`EffectsLogger`)
- **Data Accumulation**: Collects observations across multiple policy plans
- **Persistence**: Save/load functionality for long-term data storage
- **Threshold Checking**: Ensures minimum observations before estimation
- **Statistics**: Real-time summary statistics for accumulated data

### 3. **Causal Effects Manager** (`CausalEffectsManager`)
- **Conditional Activation**: Only estimates when sufficient data available
- **Multi-Effect Management**: Handles multiple causal relationships simultaneously
- **Batch Analysis**: Periodic causal analysis across all tracked effects

## Mathematical Foundation

### Model Specification

We model causal relationships using simple linear regression:

```
Y = β₀ + β₁X + ε,  ε ~ N(0, σ²)
```

Where:
- **Y**: Outcome variable (e.g., feasibility score)
- **X**: Predictor variable (e.g., responsibility score)
- **β₀**: Intercept
- **β₁**: **Causal effect** of interest
- **σ²**: Noise variance

### Prior Distribution

We use conjugate Normal-InverseGamma priors:

```
β | σ² ~ N(μ₀, σ²V₀)
σ² ~ InvGamma(a₀, b₀)
```

Default hyperparameters:
- **μ₀ = [0, 0]**: Vague prior on coefficients
- **V₀ = I × 1000**: Large prior variance (uninformative)
- **a₀ = 1**: InverseGamma shape
- **b₀ = 1**: InverseGamma scale

### Posterior Distribution

The posterior is also Normal-InverseGamma (conjugacy property):

```
β | σ², data ~ N(μₙ, σ²Vₙ)
σ² | data ~ InvGamma(aₙ, bₙ)
```

Where:
- **Vₙ = (V₀⁻¹ + X'X)⁻¹**
- **μₙ = Vₙ(V₀⁻¹μ₀ + X'y)**
- **aₙ = a₀ + n/2**
- **bₙ = b₀ + ½(residual'residual + prior_term)**

### Marginal Distribution for β₁

The marginal posterior for β₁ follows a **t-distribution**:

```
(β₁ - μₙ[1]) / √(bₙ/aₙ × Vₙ[1,1]) ~ t(2aₙ)
```

This enables exact credible intervals and hypothesis testing.

## Usage Examples

### Example 1: Basic Bayesian Regression

```python
import numpy as np
from evaluation.causal_effect_estimator import BayesLinearEffect

# Generate data: y = 2.0 + 3.5*x + noise
np.random.seed(42)
x = np.random.uniform(0, 1, 100)
y = 2.0 + 3.5 * x + np.random.normal(0, 0.5, 100)

# Fit model
model = BayesLinearEffect()
model.fit(x, y)

# Get posterior mean
beta_mean = model.beta_posterior_mean()
print(f"β₀ = {beta_mean[0]:.3f}, β₁ = {beta_mean[1]:.3f}")

# Get 95% credible interval for causal effect
interval = model.beta1_credible_interval(level=0.95)
print(f"95% CI for β₁: [{interval[0]:.3f}, {interval[1]:.3f}]")

# Test significance
is_sig, prob = model.effect_is_significant(threshold=0.0)
print(f"Effect significant: {is_sig}, P(β₁ > 0) = {prob:.3f}")
```

### Example 2: Data Accumulation Across Plans

```python
from pathlib import Path
from orchestrator.effects_logger import EffectsLogger

# Initialize logger
logger = EffectsLogger(Path("./causal_effects_data"))

# Process multiple plans
for plan_id in range(50):
    # ... evaluate plan and get scores ...
    responsibility_score = ...
    feasibility_score = ...
    
    # Log observation
    logger.log_effect(
        effect_name="responsibility_to_feasibility",
        x_value=responsibility_score,
        y_value=feasibility_score,
        plan_id=f"plan_{plan_id:03d}"
    )

# Save accumulated data
logger.save()

# Check if ready for estimation
if logger.has_sufficient_data("responsibility_to_feasibility", min_obs=30):
    print("Ready for causal estimation!")
```

### Example 3: Causal Effects Management

```python
from orchestrator.effects_logger import CausalEffectsManager

# Create manager with minimum observation threshold
manager = CausalEffectsManager(logger, min_obs=30)

# Estimate effect (returns None if insufficient data)
model = manager.estimate_effect("responsibility_to_feasibility")

if model is not None:
    summary = model.get_summary()
    print(f"Causal Effect: β₁ = {summary['beta1_mean']:.3f}")
    print(f"95% CI: {summary['beta1_interval_95']}")
    print(f"Significant: {summary['effect_significant']}")
```

### Example 4: Integration with FeasibilityScorer

```python
from example_feasibility_causal_integration import FeasibilityScorerWithCausalLogging

# Create scorer with effects logging
scorer = FeasibilityScorerWithCausalLogging(
    effects_logger=logger,
    plan_id="plan_123"
)

# Evaluate indicators (automatically logs relationships)
score = scorer.evaluate_indicator(indicator_text)

# Causal relationships are logged for future analysis
```

## Design Principles

### 1. **Non-Invasive Integration**
- The causal estimator is designed as a separate module
- Minimal changes to existing code required
- Optional integration - system works without it

### 2. **Threshold-Based Activation**
- Estimation only occurs when sufficient data available (default: 30 observations)
- Prevents unreliable estimates from small samples
- Graceful degradation when data insufficient

### 3. **Statistical Rigor**
- Uses established Bayesian methods
- Full uncertainty quantification
- Proper significance testing

### 4. **Long-Term Learning**
- Accumulates data across multiple plan evaluations
- Persistent storage for continuous improvement
- Enables meta-analysis of policy effectiveness

## Critical Note: Data Requirements

⚠️ **IMPORTANT**: This module requires data from **multiple plans** (≥30 recommended).

- For **single plan** analysis, the module can be implemented but should NOT be activated
- The system logs data but defers estimation until sufficient corpus available
- This is an **observational learning** component, not single-plan analysis

## Performance Characteristics

### Computational Complexity
- **Fitting**: O(n) for n observations (closed-form solution)
- **Prediction**: O(1) per prediction
- **Storage**: O(n) observations per effect type

### Statistical Properties
- **Consistency**: Posterior converges to true parameters as n → ∞
- **Credible Interval Coverage**: 95% intervals have proper frequentist coverage
- **Robustness**: Vague priors minimize prior influence with sufficient data

## File Structure

```
evaluation/
  └── causal_effect_estimator.py    # Core Bayesian regression class

orchestrator/
  └── effects_logger.py              # Data accumulation and management

test_causal_effect_estimator.py      # Comprehensive test suite

example_causal_analysis.py           # Standalone usage examples
example_feasibility_causal_integration.py  # Integration examples
```

## Testing

Run the comprehensive test suite:

```bash
python3 -m pytest test_causal_effect_estimator.py -v
```

Tests cover:
- Bayesian regression correctness
- Credible interval coverage
- Significance testing
- Save/load functionality
- Effects logger data accumulation
- Threshold-based activation
- Integration workflow

## Future Extensions

Possible enhancements:
1. **Multivariate Effects**: Multiple predictors (β₀ + β₁X₁ + β₂X₂ + ...)
2. **Hierarchical Models**: Effects varying by department/region
3. **Time-Varying Effects**: Track how relationships change over time
4. **Instrumental Variables**: Address confounding and selection bias
5. **Visualization**: Posterior density plots, effect trajectories

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.)
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*
- Conjugate Priors: [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior)

## Authors

MINIMINIMOON Team, 2025

## License

Part of the MINIMINIMOON public policy evaluation system.
