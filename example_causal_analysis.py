"""
Example Usage: Bayesian Causal Effect Estimator

This script demonstrates how to use the Bayesian causal effect estimator
to analyze relationships between variables in public policy plans.

Features demonstrated:
1. Synthetic data generation
2. Effect logging across multiple plans
3. Bayesian linear regression fitting
4. Credible interval computation
5. Significance testing
6. Periodic causal analysis

Author: MINIMINIMOON Team
Date: 2025
"""

import numpy as np
import tempfile
from pathlib import Path

from evaluation.causal_effect_estimator import BayesLinearEffect
from orchestrator.effects_logger import EffectsLogger, CausalEffectsManager, periodic_causal_analysis


def demo_basic_usage():
    """Demo 1: Basic usage of BayesLinearEffect."""
    print("=" * 70)
    print("DEMO 1: Basic Bayesian Linear Regression")
    print("=" * 70)
    
    # Generate synthetic data: y = 2.0 + 3.5*x + noise
    np.random.seed(42)
    n_samples = 100
    x = np.random.uniform(0, 1, n_samples)
    y = 2.0 + 3.5 * x + np.random.normal(0, 0.5, n_samples)
    
    print(f"\nGenerated {n_samples} synthetic observations")
    print(f"True relationship: y = 2.0 + 3.5*x + noise")
    
    # Fit Bayesian model
    model = BayesLinearEffect()
    model.fit(x, y)
    
    # Get results
    beta_mean = model.beta_posterior_mean()
    interval = model.beta1_credible_interval(level=0.95)
    is_sig, prob = model.effect_is_significant(threshold=0.0)
    
    print(f"\nPosterior Results:")
    print(f"  β₀ (intercept): {beta_mean[0]:.3f}")
    print(f"  β₁ (causal effect): {beta_mean[1]:.3f}")
    print(f"  95% Credible Interval for β₁: [{interval[0]:.3f}, {interval[1]:.3f}]")
    print(f"  Effect is significant: {is_sig}")
    print(f"  P(β₁ > 0): {prob:.4f}")
    
    # Make predictions
    x_new = np.array([0.0, 0.5, 1.0])
    y_pred, y_std = model.predict(x_new, return_std=True)
    
    print(f"\nPredictions with Uncertainty:")
    for _, (x_val, y_val, std) in enumerate(zip(x_new, y_pred, y_std)):
        print(f"  x={x_val:.1f}: y={y_val:.3f} ± {std:.3f}")


def demo_effects_logger():
    """Demo 2: Using EffectsLogger to accumulate data."""
    print("\n" + "=" * 70)
    print("DEMO 2: Effects Logger and Data Accumulation")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EffectsLogger(Path(tmpdir))
        
        # Simulate processing multiple plans
        print("\nSimulating 50 policy plans...")
        np.random.seed(42)
        
        for plan_idx in range(50):
            # Simulate: responsibility_score → feasibility_score
            responsibility = np.random.uniform(0.3, 0.9)
            feasibility = 0.2 + 0.6 * responsibility + np.random.normal(0, 0.05)
            feasibility = np.clip(feasibility, 0, 1)
            
            logger.log_effect(
                effect_name="responsibility_to_feasibility",
                x_value=responsibility,
                y_value=feasibility,
                plan_id=f"plan_{plan_idx:03d}"
            )
            
            # Simulate: monetary_score → feasibility_score
            monetary = np.random.uniform(0.2, 0.8)
            feasibility2 = 0.3 + 0.5 * monetary + np.random.normal(0, 0.08)
            feasibility2 = np.clip(feasibility2, 0, 1)
            
            logger.log_effect(
                effect_name="monetary_to_feasibility",
                x_value=monetary,
                y_value=feasibility2,
                plan_id=f"plan_{plan_idx:03d}"
            )
        
        # Show statistics
        stats = logger.get_statistics()
        print(f"\nAccumulated Effects:")
        for effect_name, stat in stats.items():
            print(f"\n  {effect_name}:")
            print(f"    Observations: {stat['n_observations']}")
            print(f"    X: mean={stat['x_mean']:.3f}, std={stat['x_std']:.3f}")
            print(f"    Y: mean={stat['y_mean']:.3f}, std={stat['y_std']:.3f}")
            print(f"    Correlation: {stat['correlation']:.3f}")
        
        # Test persistence
        logger.save()
        print(f"\n  Data saved to: {tmpdir}")
        
        # Reload
        new_logger = EffectsLogger(Path(tmpdir))
        new_logger.load()
        print(f"  Data reloaded: {len(new_logger.get_all_effects())} effect types")


def demo_causal_manager():
    """Demo 3: Causal Effects Manager with threshold-based estimation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Causal Effects Manager")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EffectsLogger(Path(tmpdir))
        
        # Add insufficient data for one effect
        print("\nTesting threshold-based activation...")
        for i in range(20):
            logger.log_effect("effect_A", float(i)/20, float(i)/10, f"plan_{i}")
        
        # Add sufficient data for another effect
        np.random.seed(42)
        for i in range(50):
            x = i / 50.0
            y = 0.5 + 1.5 * x + np.random.normal(0, 0.1)
            logger.log_effect("effect_B", x, y, f"plan_{i}")
        
        # Create manager with min_obs=30
        manager = CausalEffectsManager(logger, min_obs=30)
        
        # Try to estimate both effects
        print("\nAttempting to estimate effects with min_obs=30:")
        
        model_A = manager.estimate_effect("effect_A")
        if model_A is None:
            print(f"  effect_A: Insufficient data ({logger.get_observation_count('effect_A')}/30)")
        else:
            print(f"  effect_A: Estimated successfully")
        
        model_B = manager.estimate_effect("effect_B")
        if model_B is not None:
            summary = model_B.get_summary()
            print(f"  effect_B: Estimated successfully")
            print(f"    β₁ = {summary['beta1_mean']:.3f} {summary['beta1_interval_95']}")
            print(f"    Significant: {summary['effect_significant']}")
        else:
            print(f"  effect_B: Failed to estimate")


def demo_periodic_analysis():
    """Demo 4: Periodic causal analysis workflow."""
    print("\n" + "=" * 70)
    print("DEMO 4: Periodic Causal Analysis")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EffectsLogger(Path(tmpdir))
        
        # Simulate accumulating data over time
        print("\nSimulating data accumulation from 60 policy plans...")
        np.random.seed(42)
        
        for plan_idx in range(60):
            # responsibility → feasibility (strong positive effect)
            responsibility = np.random.uniform(0.3, 0.9)
            feasibility = 0.2 + 0.7 * responsibility + np.random.normal(0, 0.05)
            feasibility = np.clip(feasibility, 0, 1)
            logger.log_effect("responsibility_to_feasibility", responsibility, feasibility, f"plan_{plan_idx}")
            
            # monetary → feasibility (moderate positive effect)
            monetary = np.random.uniform(0.2, 0.8)
            feasibility2 = 0.3 + 0.4 * monetary + np.random.normal(0, 0.08)
            feasibility2 = np.clip(feasibility2, 0, 1)
            logger.log_effect("monetary_to_feasibility", monetary, feasibility2, f"plan_{plan_idx}")
            
            # teoria_cambio → kpi (weak effect - insufficient data)
            teoria = np.random.uniform(0.4, 0.9)
            kpi = 0.5 + 0.2 * teoria + np.random.normal(0, 0.15)
            kpi = np.clip(kpi, 0, 1)
            logger.log_effect("teoria_cambio_to_kpi", teoria, kpi, f"plan_{plan_idx}")
        
        # Run periodic analysis
        print("\nRunning periodic causal analysis (min_obs=30)...")
        results = periodic_causal_analysis(logger, min_obs=30)
        
        print(f"\n{len(results)} effects successfully estimated")


def demo_integration_with_policy_analysis():
    """Demo 5: Integration scenario with policy evaluation."""
    print("\n" + "=" * 70)
    print("DEMO 5: Integration with Policy Evaluation")
    print("=" * 70)
    
    print("\nSimulating real-world policy evaluation workflow...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize effects logger (would persist across runs)
        effects_storage = Path(tmpdir) / "causal_effects"
        logger = EffectsLogger(effects_storage)
        
        # Simulate processing a new plan
        print("\nProcessing Plan #123...")
        
        # Simulate component scores (would come from actual analysis)
        responsibility_score = 0.75
        monetary_score = 0.65
        feasibility_score = 0.70
        
        # Log the relationships for future causal analysis
        logger.log_effect(
            "responsibility_to_feasibility",
            x_value=responsibility_score,
            y_value=feasibility_score,
            plan_id="plan_123"
        )
        
        logger.log_effect(
            "monetary_to_feasibility",
            x_value=monetary_score,
            y_value=feasibility_score,
            plan_id="plan_123"
        )
        
        print(f"  Logged causal observations for plan_123")
        
        # Check if we have enough data to estimate effects
        n_obs = logger.get_observation_count("responsibility_to_feasibility")
        print(f"  Total observations for responsibility→feasibility: {n_obs}")
        
        if n_obs >= 30:
            print("  ✓ Sufficient data for causal estimation")
            manager = CausalEffectsManager(logger, min_obs=30)
            model = manager.estimate_effect("responsibility_to_feasibility")
            if model:
                summary = model.get_summary()
                print(f"  Estimated causal effect: β₁ = {summary['beta1_mean']:.3f}")
        else:
            print(f"  ⚠ Insufficient data for estimation (need {30-n_obs} more observations)")
        
        # Save for next run
        logger.save()
        print(f"\n  Data persisted to: {effects_storage}")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "BAYESIAN CAUSAL EFFECT ESTIMATOR EXAMPLES" + " " * 16 + "║")
    print("║" + " " * 20 + "MINIMINIMOON System" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    demo_basic_usage()
    demo_effects_logger()
    demo_causal_manager()
    demo_periodic_analysis()
    demo_integration_with_policy_analysis()
    
    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. BayesLinearEffect provides closed-form Bayesian regression")
    print("  2. EffectsLogger accumulates data across multiple plans")
    print("  3. CausalEffectsManager only estimates when data is sufficient")
    print("  4. System designed for long-term observational learning")
    print("  5. Integrates seamlessly with existing evaluation pipeline")
    print()
