"""
Comprehensive test suite for Bayesian Causal Effect Estimator

Tests:
- Bayesian linear regression with known effects
- Credible interval coverage
- Significance testing
- Prediction with uncertainty
- Save/load functionality
- Effects logger data accumulation
- Threshold-based activation
- Integration with multiple plans

Author: MINIMINIMOON Team
Date: 2025
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from orchestrator.effects_logger import (
    CausalEffectsManager,
    EffectsLogger,
    periodic_causal_analysis,
)

from evaluation.causal_effect_estimator import BayesLinearEffect


class TestBayesLinearEffect:
    """Test suite for BayesLinearEffect class."""

    @staticmethod
    def test_initialization_defaults():
        """Test that model initializes with default priors."""
        model = BayesLinearEffect()

        assert model.mu0 is not None
        assert model.V0 is not None
        assert model.a0 == 1.0
        assert model.b0 == 1.0
        assert not model.fitted
        assert model.n_obs == 0

    @staticmethod
    def test_fit_with_synthetic_data():
        """Test fitting with synthetic data with known effect."""
        # Generate data: y = 2.0 + 3.5*x + noise
        np.random.seed(42)
        x = np.random.uniform(0, 1, 100)
        y = 2.0 + 3.5 * x + np.random.normal(0, 0.5, 100)

        model = BayesLinearEffect()
        model.fit(x, y)

        assert model.fitted
        assert model.n_obs == 100

        # Check that β₁ is close to 3.5
        beta1_mean = model.beta_posterior_mean()[1]
        assert 3.0 < beta1_mean < 4.0, f"Expected β₁ ≈ 3.5, got {beta1_mean}"

    @staticmethod
    def test_credible_interval_coverage():
        """Test that credible interval contains true parameter."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 100)
        y = 2.0 + 3.5 * x + np.random.normal(0, 0.5, 100)

        model = BayesLinearEffect()
        model.fit(x, y)

        interval = model.beta1_credible_interval(level=0.95)

        # 95% interval should contain true value 3.5
        assert interval[0] < 3.5 < interval[1], (
            f"95% interval {interval} should contain true value 3.5"
        )

        # Interval should be finite and reasonable
        assert np.isfinite(interval[0]) and np.isfinite(interval[1])
        assert interval[1] > interval[0]

    @staticmethod
    def test_effect_significance():
        """Test significance testing for positive effect."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 100)
        y = 2.0 + 3.5 * x + np.random.normal(0, 0.5, 100)

        model = BayesLinearEffect()
        model.fit(x, y)

        is_sig, prob = model.effect_is_significant(threshold=0.0)

        assert is_sig, "Effect should be significant"
        assert prob > 0.99, f"Probability of positive effect should be high, got {prob}"

    @staticmethod
    def test_prediction_without_uncertainty():
        """Test prediction without uncertainty."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 50)
        y = 1.0 + 2.0 * x + np.random.normal(0, 0.3, 50)

        model = BayesLinearEffect()
        model.fit(x, y)

        x_new = np.array([0.0, 0.5, 1.0])
        y_pred = model.predict(x_new, return_std=False)

        assert y_pred.shape == (3,)
        assert np.all(np.isfinite(y_pred))

    @staticmethod
    def test_prediction_with_uncertainty():
        """Test prediction with uncertainty quantification."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 50)
        y = 1.0 + 2.0 * x + np.random.normal(0, 0.3, 50)

        model = BayesLinearEffect()
        model.fit(x, y)

        x_new = np.array([0.0, 0.5, 1.0])
        y_pred, y_std = model.predict(x_new, return_std=True)

        assert y_pred.shape == (3,)
        assert y_std.shape == (3,)
        assert np.all(y_std > 0)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    @staticmethod
    def test_get_summary():
        """Test summary generation."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 50)
        y = 1.5 + 2.5 * x + np.random.normal(0, 0.4, 50)

        model = BayesLinearEffect()
        model.fit(x, y)

        summary = model.get_summary()

        assert summary["fitted"]
        assert summary["n_obs"] == 50
        assert "beta0_mean" in summary
        assert "beta1_mean" in summary
        assert "beta1_interval_95" in summary
        assert "sigma2_mean" in summary
        assert "effect_significant" in summary
        assert "prob_positive_effect" in summary

    @staticmethod
    def test_unfitted_model_raises_error():
        """Test that unfitted model raises appropriate errors."""
        model = BayesLinearEffect()

        with pytest.raises(RuntimeError):
            model.beta_posterior_mean()

        with pytest.raises(RuntimeError):
            model.beta1_credible_interval()

        with pytest.raises(RuntimeError):
            model.predict(np.array([0.5]))

    @staticmethod
    def test_mismatched_input_sizes():
        """Test that mismatched x and y sizes raise error."""
        model = BayesLinearEffect()
        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(ValueError):
            model.fit(x, y)

    @staticmethod
    def test_save_and_load():
        """Test model persistence."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 50)
        y = 1.0 + 2.0 * x + np.random.normal(0, 0.3, 50)

        model = BayesLinearEffect()
        model.fit(x, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)

            # Load model
            loaded_model = BayesLinearEffect.load(path)

            assert loaded_model.fitted
            assert loaded_model.n_obs == model.n_obs
            np.testing.assert_array_almost_equal(loaded_model.mun, model.mun)
            np.testing.assert_array_almost_equal(loaded_model.Vn, model.Vn)
            assert loaded_model.an == model.an
            assert loaded_model.bn == model.bn


class TestEffectsLogger:
    """Test suite for EffectsLogger class."""

    @staticmethod
    def test_initialization():
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            assert logger.storage_path.exists()
            assert len(logger.effects_db) == 0

    @staticmethod
    def test_log_single_effect():
        """Test logging a single effect observation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            logger.log_effect(
                effect_name="test_effect", x_value=0.5, y_value=0.8, plan_id="plan_001"
            )

            assert "test_effect" in logger.effects_db
            assert len(logger.effects_db["test_effect"]) == 1
            assert logger.get_observation_count("test_effect") == 1

    @staticmethod
    def test_log_multiple_effects():
        """Test logging multiple observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            for i in range(10):
                logger.log_effect(
                    effect_name="test_effect",
                    x_value=float(i) / 10,
                    y_value=float(i) / 5,
                    plan_id=f"plan_{i:03d}",
                )

            assert logger.get_observation_count("test_effect") == 10
            x, y = logger.get_effect_data("test_effect")
            assert len(x) == 10
            assert len(y) == 10

    @staticmethod
    def test_get_effect_data():
        """Test retrieving effect data as arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            x_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            y_values = [0.2, 0.4, 0.6, 0.8, 1.0]

            for i, (x, y) in enumerate(zip(x_values, y_values)):
                logger.log_effect("test", x, y, f"plan_{i}")

            x, y = logger.get_effect_data("test")

            np.testing.assert_array_almost_equal(x, x_values)
            np.testing.assert_array_almost_equal(y, y_values)

    @staticmethod
    def test_has_sufficient_data():
        """Test threshold checking for minimum observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Add 25 observations
            for i in range(25):
                logger.log_effect("test", float(i), float(i), f"plan_{i}")

            assert not logger.has_sufficient_data("test", min_obs=30)
            assert logger.has_sufficient_data("test", min_obs=20)
            assert logger.has_sufficient_data("test", min_obs=25)

    @staticmethod
    def test_save_and_load():
        """Test persistence of effects data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Log some data
            for i in range(5):
                logger.log_effect("effect_a", float(i), float(i * 2), f"plan_{i}")
                logger.log_effect("effect_b", float(i), float(i * 3), f"plan_{i}")

            logger.save()

            # Create new logger and load
            new_logger = EffectsLogger(Path(tmpdir))
            new_logger.load()

            assert "effect_a" in new_logger.effects_db
            assert "effect_b" in new_logger.effects_db
            assert len(new_logger.effects_db["effect_a"]) == 5
            assert len(new_logger.effects_db["effect_b"]) == 5

    @staticmethod
    def test_get_statistics():
        """Test statistics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Add data with known statistics
            x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
            y_vals = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2*x

            for x, y in zip(x_vals, y_vals):
                logger.log_effect("linear", x, y, "plan")

            stats = logger.get_statistics()

            assert "linear" in stats
            assert stats["linear"]["n_observations"] == 5
            assert abs(stats["linear"]["x_mean"] - 3.0) < 0.01
            assert abs(stats["linear"]["y_mean"] - 6.0) < 0.01
            assert abs(stats["linear"]["correlation"] - 1.0) < 0.01


class TestCausalEffectsManager:
    """Test suite for CausalEffectsManager class."""

    @staticmethod
    def test_initialization():
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))
            manager = CausalEffectsManager(logger, min_obs=30)

            assert manager.min_obs == 30
            assert len(manager.estimators) == 0

    @staticmethod
    def test_estimate_with_insufficient_data():
        """Test that estimation returns None with insufficient data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Add only 10 observations
            for i in range(10):
                logger.log_effect("test", float(i), float(i), f"plan_{i}")

            manager = CausalEffectsManager(logger, min_obs=30)
            model = manager.estimate_effect("test")

            assert model is None

    @staticmethod
    def test_estimate_with_sufficient_data():
        """Test successful estimation with sufficient data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Add 50 observations with known relationship
            np.random.seed(42)
            for i in range(50):
                x = i / 50.0
                y = 1.0 + 2.0 * x + np.random.normal(0, 0.1)
                logger.log_effect("causal", x, y, f"plan_{i}")

            manager = CausalEffectsManager(logger, min_obs=30)
            model = manager.estimate_effect("causal")

            assert model is not None
            assert model.fitted
            assert model.n_obs == 50

    @staticmethod
    def test_get_all_effects():
        """Test getting summaries of all estimated effects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Add data for two effects
            np.random.seed(42)
            for i in range(40):
                x = i / 40.0
                logger.log_effect("effect_1", x, 1.0 + 1.5 * x, f"plan_{i}")
                logger.log_effect("effect_2", x, 0.5 + 2.5 * x, f"plan_{i}")

            manager = CausalEffectsManager(logger, min_obs=30)
            manager.estimate_effect("effect_1")
            manager.estimate_effect("effect_2")

            summaries = manager.get_all_effects()

            assert "effect_1" in summaries
            assert "effect_2" in summaries
            assert summaries["effect_1"]["fitted"]
            assert summaries["effect_2"]["fitted"]


class TestIntegration:
    """Integration tests for complete workflow."""

    @staticmethod
    def test_complete_workflow():
        """Test complete workflow from data logging to estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Simulate processing multiple plans
            np.random.seed(42)
            for plan_idx in range(50):
                # Simulate responsibility score → feasibility
                responsibility = np.random.uniform(0.3, 0.9)
                feasibility = 0.2 + 0.6 * responsibility + np.random.normal(0, 0.05)
                feasibility = np.clip(feasibility, 0, 1)

                logger.log_effect(
                    "responsibility_to_feasibility",
                    responsibility,
                    feasibility,
                    f"plan_{plan_idx:03d}",
                )

            # Save and reload
            logger.save()
            new_logger = EffectsLogger(Path(tmpdir))
            new_logger.load()

            # Estimate effect
            manager = CausalEffectsManager(new_logger, min_obs=30)
            model = manager.estimate_effect("responsibility_to_feasibility")

            assert model is not None
            summary = model.get_summary()

            # Check that estimated effect is positive (as expected)
            assert summary["beta1_mean"] > 0
            assert summary["effect_significant"]
            assert summary["prob_positive_effect"] > 0.95

    @staticmethod
    def test_periodic_analysis():
        """Test periodic analysis function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EffectsLogger(Path(tmpdir))

            # Add sufficient data for one effect
            np.random.seed(42)
            for i in range(40):
                x = i / 40.0
                y = 0.5 + 1.5 * x + np.random.normal(0, 0.1)
                logger.log_effect("responsibility_to_feasibility", x, y, f"plan_{i}")

            # Run periodic analysis
            results = periodic_causal_analysis(logger, min_obs=30)

            assert "responsibility_to_feasibility" in results
            assert results["responsibility_to_feasibility"]["fitted"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
