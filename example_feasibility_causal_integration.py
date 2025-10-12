"""
Integration Example: FeasibilityScorer with Causal Effects Logging

This example demonstrates how to integrate the Bayesian causal effect estimator
with the existing FeasibilityScorer to log and analyze causal relationships
between different scoring components.

Features:
1. Extended FeasibilityScorer with effects logging
2. Automatic logging of component relationships
3. Periodic causal analysis
4. Long-term observational learning

Author: MINIMINIMOON Team
Date: 2025
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from orchestrator.effects_logger import CausalEffectsManager, EffectsLogger

from feasibility_scorer import FeasibilityScorer, IndicatorScore


class FeasibilityScorerWithCausalLogging(FeasibilityScorer):
    """
    Extended FeasibilityScorer that logs causal relationships for future analysis.

    This class extends the base FeasibilityScorer to automatically log
    relationships between component scores (e.g., responsibility → feasibility)
    for long-term causal effect estimation.
    """

    def __init__(
        self,
        enable_parallel: bool = True,
        seed: int = 42,
        evidence_registry=None,
        effects_logger: Optional[EffectsLogger] = None,
        plan_id: Optional[str] = None,
    ):
        """
        Initialize scorer with causal effects logging.

        Args:
            enable_parallel: Enable parallel processing
            seed: Random seed for reproducibility
            evidence_registry: Evidence registry for tracking
            effects_logger: EffectsLogger instance for causal data accumulation
            plan_id: Identifier for the current plan being processed
        """
        super().__init__(enable_parallel, seed, evidence_registry)
        self.effects_logger = effects_logger
        self.plan_id = plan_id or "unknown"

    def evaluate_indicator(self, text: str) -> IndicatorScore:
        """
        Evaluate indicator and log causal relationships.

        Args:
            text: Indicator text to evaluate

        Returns:
            IndicatorScore with comprehensive evaluation
        """
        # Get base evaluation
        score = super().evaluate_indicator(text)

        # Log causal relationships if logger is available
        if self.effects_logger is not None:
            self._log_causal_relationships(score)

        return score

    def _log_causal_relationships(self, score: IndicatorScore):
        """
        Log relationships between components for causal analysis.

        Args:
            score: IndicatorScore containing component evaluations
        """
        # Extract component presence scores (0 or 1)
        has_baseline = 1.0 if score.has_baseline else 0.0
        has_target = 1.0 if score.has_target else 0.0
        has_timeframe = 1.0 if score.has_timeframe else 0.0
        has_responsible = 1.0 if score.has_responsible else 0.0
        has_unit = 1.0 if score.has_unit else 0.0

        # Extract quantitative scores
        has_quantitative_target = 1.0 if score.has_quantitative_target else 0.0

        # Feasibility score as outcome
        feasibility = score.feasibility_score

        # Log various causal relationships

        # 1. Baseline presence → Feasibility
        if has_baseline > 0:
            self.effects_logger.log_effect(
                "baseline_to_feasibility",
                x_value=has_baseline,
                y_value=feasibility,
                plan_id=self.plan_id,
            )

        # 2. Target presence → Feasibility
        if has_target > 0:
            self.effects_logger.log_effect(
                "target_to_feasibility",
                x_value=has_target,
                y_value=feasibility,
                plan_id=self.plan_id,
            )

        # 3. Timeframe presence → Feasibility
        if has_timeframe > 0:
            self.effects_logger.log_effect(
                "timeframe_to_feasibility",
                x_value=has_timeframe,
                y_value=feasibility,
                plan_id=self.plan_id,
            )

        # 4. Quantitative target → Feasibility
        if has_quantitative_target > 0:
            self.effects_logger.log_effect(
                "quantitative_target_to_feasibility",
                x_value=has_quantitative_target,
                y_value=feasibility,
                plan_id=self.plan_id,
            )

        # 5. Unit specification → Feasibility
        if has_unit > 0:
            self.effects_logger.log_effect(
                "unit_to_feasibility",
                x_value=has_unit,
                y_value=feasibility,
                plan_id=self.plan_id,
            )

        # 6. Responsible entity → Feasibility
        if has_responsible > 0:
            self.effects_logger.log_effect(
                "responsible_to_feasibility",
                x_value=has_responsible,
                y_value=feasibility,
                plan_id=self.plan_id,
            )

        # 7. SMART score → Feasibility
        self.effects_logger.log_effect(
            "smart_score_to_feasibility",
            x_value=score.smart_score,
            y_value=feasibility,
            plan_id=self.plan_id,
        )


def demo_integration():
    """Demonstrate integration with FeasibilityScorer."""
    print("=" * 70)
    print("DEMO: FeasibilityScorer with Causal Effects Logging")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize effects logger
        effects_storage = Path(tmpdir) / "causal_effects"
        logger = EffectsLogger(effects_storage)

        # Sample indicators (Spanish policy plan text)
        indicators = [
            "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025, bajo la responsabilidad de la Secretaría de Educación",
            "Reducir la tasa de desempleo desde el valor actual de 15.3% hasta alcanzar el 8% en el periodo 2024-2028",
            "Aumentar el acceso a servicios de salud con cobertura del 70% al 90% durante el cuatrienio",
            "Mejorar la infraestructura vial construyendo 50 km de nuevas carreteras para 2026",
            "Reducir la pobreza extrema desde 12.5% hasta 7% en el horizonte 2024-2028",
            "Incrementar cobertura de alcantarillado del 45% actual al 75% objetivo en 4 años",
            "Mejorar calidad educativa mediante capacitación docente",  # Low quality - vague
            "Incrementar indicadores de desarrollo",  # Very low quality - no specifics
            "Elevar el índice de desarrollo humano desde 0.65 hasta 0.78 para el año 2027",
            "Reducir mortalidad infantil de 18 por mil a 10 por mil en el periodo 2024-2028",
        ]

        print(f"\nProcessing {len(indicators)} indicators from multiple plans...")

        # Simulate processing indicators from different plans
        for idx, indicator_text in enumerate(indicators):
            plan_id = f"plan_{idx // 2:03d}"  # Group indicators by plan

            # Create scorer for this plan
            scorer = FeasibilityScorerWithCausalLogging(
                enable_parallel=False, effects_logger=logger, plan_id=plan_id
            )

            # Evaluate indicator (automatically logs causal relationships)
            score = scorer.evaluate_indicator(indicator_text)

            print(f"\n  Plan {plan_id}, Indicator {idx % 2 + 1}:")
            print(f"    Feasibility: {score.feasibility_score:.3f}")
            print(f"    SMART: {score.smart_score:.3f}")
            print(
                f"    Components: baseline={score.has_baseline}, "
                f"target={score.has_target}, timeframe={score.has_timeframe}"
            )

        # Save accumulated data
        logger.save()
        print(f"\n  Causal effects data saved to: {effects_storage}")

        # Show statistics
        stats = logger.get_statistics()
        print("\n  Accumulated causal relationships:")
        for effect_name, stat in sorted(stats.items()):
            print(
                f"    {effect_name}: {stat['n_observations']} observations, "
                f"corr={stat['correlation']:.3f}"
            )

        # Check if we have enough data for causal estimation
        print("\n  Checking data sufficiency (min_obs=30):")
        for effect_name in logger.get_all_effects():
            n_obs = logger.get_observation_count(effect_name)
            sufficient = logger.has_sufficient_data(effect_name, min_obs=30)
            status = "✓ Ready" if sufficient else f"⚠ Need {30 - n_obs} more"
            print(f"    {effect_name}: {n_obs}/30 observations - {status}")


def demo_causal_estimation_after_accumulation():
    """Demonstrate causal estimation after accumulating sufficient data."""
    print("\n" + "=" * 70)
    print("DEMO: Causal Estimation After Data Accumulation")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        effects_storage = Path(tmpdir) / "causal_effects"
        logger = EffectsLogger(effects_storage)

        # Simulate processing many indicators to accumulate sufficient data
        print("\nSimulating processing of 50 policy plans (100 indicators)...")

        np.random.seed(42)
        for plan_idx in range(50):
            plan_id = f"plan_{plan_idx:03d}"

            # Create scorer for this plan
            scorer = FeasibilityScorerWithCausalLogging(
                enable_parallel=False, effects_logger=logger, plan_id=plan_id
            )

            # Simulate 2 indicators per plan with varying quality
            for _ in range(2):
                # Generate synthetic scores that mimic real relationships
                has_baseline = np.random.rand() > 0.3
                has_target = np.random.rand() > 0.2
                has_timeframe = np.random.rand() > 0.4
                has_quantitative = np.random.rand() > 0.5

                # Feasibility depends on components (with noise)
                feasibility = 0.3
                if has_baseline:
                    feasibility += 0.15
                if has_target:
                    feasibility += 0.15
                if has_timeframe:
                    feasibility += 0.10
                if has_quantitative:
                    feasibility += 0.20
                feasibility += np.random.normal(0, 0.05)
                feasibility = np.clip(feasibility, 0, 1)

                smart_score = (
                    has_baseline + has_target + has_timeframe + has_quantitative
                ) / 4.0

                # Log relationships manually (simulating what scorer would do)
                if has_baseline:
                    logger.log_effect(
                        "baseline_to_feasibility", 1.0, feasibility, plan_id
                    )
                if has_target:
                    logger.log_effect(
                        "target_to_feasibility", 1.0, feasibility, plan_id
                    )
                if has_timeframe:
                    logger.log_effect(
                        "timeframe_to_feasibility", 1.0, feasibility, plan_id
                    )
                logger.log_effect(
                    "smart_score_to_feasibility", smart_score, feasibility, plan_id
                )

        print("  Processed indicators from 50 plans")

        # Estimate causal effects
        print("\n  Estimating causal effects (min_obs=30)...")
        manager = CausalEffectsManager(logger, min_obs=30)

        effects_to_analyze = [
            "baseline_to_feasibility",
            "target_to_feasibility",
            "timeframe_to_feasibility",
            "smart_score_to_feasibility",
        ]

        print("\n  Causal Effect Estimates:")
        for effect_name in effects_to_analyze:
            if effect_name in logger.get_all_effects():
                model = manager.estimate_effect(effect_name)
                if model:
                    summary = model.get_summary()
                    print(f"\n    {effect_name}:")
                    print(f"      β₁ (causal effect): {summary['beta1_mean']:.3f}")
                    print(
                        f"      95% CI: [{summary['beta1_interval_95'][0]:.3f}, "
                        f"{summary['beta1_interval_95'][1]:.3f}]"
                    )
                    print(f"      Significant: {summary['effect_significant']}")
                    print(
                        f"      P(positive effect): {summary['prob_positive_effect']:.1%}"
                    )


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print(
        "║"
        + " " * 8
        + "FEASIBILITY SCORER + CAUSAL EFFECTS INTEGRATION"
        + " " * 12
        + "║"
    )
    print("║" + " " * 20 + "MINIMINIMOON System" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")

    demo_integration()
    demo_causal_estimation_after_accumulation()

    print("\n" + "=" * 70)
    print("Integration demos completed successfully!")
    print("=" * 70)
    print("\nKey Integration Points:")
    print("  1. FeasibilityScorerWithCausalLogging extends base scorer")
    print("  2. Automatically logs component relationships during evaluation")
    print("  3. Data accumulates across multiple plan evaluations")
    print("  4. Causal effects estimated when sufficient data available")
    print("  5. Non-invasive integration - no changes to core scorer logic")
    print()
