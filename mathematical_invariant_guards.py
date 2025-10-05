#!/usr/bin/env python3
"""
Mathematical Invariant Guards with Precision Monitoring
=======================================================
Implements numerical stability checks and precision guards for floating-point operations.
Addresses the identified violations:
- Transport doubly stochastic violation (3.79% row, 2.53% column deviation)
- Mass conservation violation (0.4% deviation)
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToleranceLevel(Enum):
    """Acceptable tolerance thresholds for different invariants"""
    STRICT = 1e-10      # For critical operations
    STANDARD = 1e-7     # Default tolerance
    RELAXED = 1e-5      # For noisy operations
    PERMISSIVE = 1e-3   # Maximum acceptable


@dataclass
class InvariantViolation:
    """Details of an invariant violation"""
    invariant_name: str
    expected_value: float
    actual_value: float
    deviation: float
    tolerance: float
    severity: str
    timestamp: str


class MathematicalInvariantGuard:
    """Guards for mathematical invariants with automatic alerting"""

    def __init__(self, tolerance: ToleranceLevel = ToleranceLevel.STANDARD):
        self.tolerance = tolerance.value
        self.violations: list[InvariantViolation] = []
        self.alert_callbacks = []

    def register_alert_callback(self, callback):
        """Register callback for invariant violations"""
        self.alert_callbacks.append(callback)

    def _trigger_alert(self, violation: InvariantViolation):
        """Trigger all registered alert callbacks"""
        logger.warning(f"Invariant violation detected: {violation.invariant_name}")
        for callback in self.alert_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def check_transport_doubly_stochastic(
        self,
        transport_plan: np.ndarray,
        tolerance: Optional[float] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Verify transport plan is doubly stochastic with enhanced precision.

        A doubly stochastic matrix has all rows and columns sum to 1.

        Args:
            transport_plan: NxN transport matrix
            tolerance: Override default tolerance

        Returns:
            (is_valid, evidence_dict)
        """
        if tolerance is None:
            tolerance = self.tolerance

        # Numerical stability: use Kahan summation for better precision
        row_sums = self._stable_sum(transport_plan, axis=1)
        col_sums = self._stable_sum(transport_plan, axis=0)

        # Calculate deviations from ideal value of 1.0
        row_deviations = np.abs(row_sums - 1.0)
        col_deviations = np.abs(col_sums - 1.0)

        max_row_dev = np.max(row_deviations)
        max_col_dev = np.max(col_deviations)

        is_valid = (max_row_dev <= tolerance) and (max_col_dev <= tolerance)

        evidence = {
            "max_row_deviation": float(max_row_dev),
            "max_col_deviation": float(max_col_dev),
            "mean_row_deviation": float(np.mean(row_deviations)),
            "mean_col_deviation": float(np.mean(col_deviations)),
            "tolerance": tolerance,
            "is_valid": is_valid
        }

        if not is_valid:
            violation = InvariantViolation(
                invariant_name="transport_doubly_stochastic",
                expected_value=1.0,
                actual_value=float(max(max_row_dev, max_col_dev)),
                deviation=float(max(max_row_dev, max_col_dev)),
                tolerance=tolerance,
                severity="HIGH" if max(max_row_dev, max_col_dev) > 0.01 else "MEDIUM",
                timestamp=self._get_timestamp()
            )
            self.violations.append(violation)
            self._trigger_alert(violation)

        return is_valid, evidence

    def check_mass_conservation(
        self,
        initial_mass: np.ndarray,
        final_mass: np.ndarray,
        tolerance: Optional[float] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Verify mass conservation with high precision.

        Args:
            initial_mass: Initial mass distribution
            final_mass: Final mass distribution after transformation
            tolerance: Override default tolerance

        Returns:
            (is_valid, evidence_dict)
        """
        if tolerance is None:
            tolerance = self.tolerance

        # Use compensated summation for better precision
        initial_total = self._stable_sum(initial_mass)
        final_total = self._stable_sum(final_mass)

        deviation = np.abs(initial_total - final_total)
        relative_deviation = deviation / initial_total if initial_total > 0 else float('inf')

        is_valid = deviation <= tolerance

        evidence = {
            "initial_total": float(initial_total),
            "final_total": float(final_total),
            "absolute_deviation": float(deviation),
            "relative_deviation": float(relative_deviation),
            "tolerance": tolerance,
            "is_valid": is_valid
        }

        if not is_valid:
            violation = InvariantViolation(
                invariant_name="mass_conservation",
                expected_value=float(initial_total),
                actual_value=float(final_total),
                deviation=float(deviation),
                tolerance=tolerance,
                severity="CRITICAL" if relative_deviation > 0.01 else "HIGH",
                timestamp=self._get_timestamp()
            )
            self.violations.append(violation)
            self._trigger_alert(violation)

        return is_valid, evidence

    def normalize_transport_plan(
        self,
        transport_plan: np.ndarray,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize transport plan to be doubly stochastic using Sinkhorn algorithm.

        Args:
            transport_plan: Initial transport matrix
            max_iterations: Maximum Sinkhorn iterations

        Returns:
            (normalized_plan, convergence_info)
        """
        plan = transport_plan.copy()

        # Avoid division by zero
        plan = np.maximum(plan, 1e-300)

        convergence_history = []

        for iteration in range(max_iterations):
            # Normalize rows
            row_sums = self._stable_sum(plan, axis=1, keepdims=True)
            plan = plan / np.maximum(row_sums, 1e-300)

            # Normalize columns
            col_sums = self._stable_sum(plan, axis=0, keepdims=True)
            plan = plan / np.maximum(col_sums, 1e-300)

            # Check convergence
            is_valid, evidence = self.check_transport_doubly_stochastic(
                plan,
                tolerance=1e-6
            )

            convergence_history.append({
                "iteration": iteration,
                "max_row_dev": evidence["max_row_deviation"],
                "max_col_dev": evidence["max_col_deviation"]
            })

            if is_valid:
                break

        convergence_info = {
            "converged": is_valid,
            "iterations": iteration + 1,
            "final_evidence": evidence,
            "history": convergence_history
        }

        return plan, convergence_info

    def _stable_sum(
        self,
        array: np.ndarray,
        axis: Optional[int] = None,
        keepdims: bool = False
    ) -> np.ndarray:
        """
        Kahan compensated summation for better numerical precision.

        This reduces floating-point accumulation errors.
        """
        if axis is None:
            # Flatten and sum
            arr_flat = array.flatten()
            total = 0.0
            compensation = 0.0

            for value in arr_flat:
                y = value - compensation
                t = total + y
                compensation = (t - total) - y
                total = t

            return total
        else:
            # Sum along axis
            result_shape = list(array.shape)
            if keepdims:
                result_shape[axis] = 1
            else:
                result_shape.pop(axis)

            # Move axis to end for easier iteration
            arr_moved = np.moveaxis(array, axis, -1)
            result = np.zeros(arr_moved.shape[:-1])

            # Iterate over all elements except the summation axis
            for idx in np.ndindex(result.shape):
                total = 0.0
                compensation = 0.0

                for value in arr_moved[idx]:
                    y = value - compensation
                    t = total + y
                    compensation = (t - total) - y
                    total = t

                result[idx] = total

            if keepdims:
                result = np.expand_dims(result, axis=axis)

            return result

    def apply_precision_guard(
        self,
        operation_func,
        *args,
        pre_check=None,
        post_check=None,
        **kwargs
    ):
        """
        Wrapper that applies precision guards before and after operations.

        Args:
            operation_func: Function to execute
            pre_check: Optional pre-condition check function
            post_check: Optional post-condition check function

        Returns:
            Result of operation_func with validation
        """
        # Pre-condition check
        if pre_check is not None:
            is_valid, evidence = pre_check(*args, **kwargs)
            if not is_valid:
                logger.warning(f"Pre-condition check failed: {evidence}")

        # Execute operation
        result = operation_func(*args, **kwargs)

        # Post-condition check
        if post_check is not None:
            is_valid, evidence = post_check(result)
            if not is_valid:
                logger.error(f"Post-condition check failed: {evidence}")
                raise ValueError(f"Invariant violation after operation: {evidence}")

        return result

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_violation_report(self) -> Dict[str, Any]:
        """Generate report of all detected violations"""
        return {
            "total_violations": len(self.violations),
            "by_severity": {
                "CRITICAL": len([v for v in self.violations if v.severity == "CRITICAL"]),
                "HIGH": len([v for v in self.violations if v.severity == "HIGH"]),
                "MEDIUM": len([v for v in self.violations if v.severity == "MEDIUM"]),
            },
            "violations": [
                {
                    "invariant": v.invariant_name,
                    "deviation": v.deviation,
                    "severity": v.severity,
                    "timestamp": v.timestamp
                }
                for v in self.violations
            ]
        }


# Regression tests
def test_transport_doubly_stochastic():
    """Regression test for transport doubly stochastic invariant"""
    guard = MathematicalInvariantGuard(ToleranceLevel.STANDARD)

    # Test 1: Perfect doubly stochastic matrix
    n = 10
    perfect_matrix = np.ones((n, n)) / n
    is_valid, evidence = guard.check_transport_doubly_stochastic(perfect_matrix)
    assert is_valid, f"Perfect matrix failed: {evidence}"
    print("✅ Test 1 passed: Perfect doubly stochastic matrix")

    # Test 2: Matrix needing normalization
    random_matrix = np.random.rand(10, 10)
    normalized, info = guard.normalize_transport_plan(random_matrix)
    is_valid, evidence = guard.check_transport_doubly_stochastic(normalized)
    assert is_valid, f"Normalized matrix failed: {evidence}"
    print(f"✅ Test 2 passed: Normalization converged in {info['iterations']} iterations")

    # Test 3: Detect violations
    bad_matrix = np.random.rand(10, 10) * 2.0  # Won't sum to 1
    is_valid, evidence = guard.check_transport_doubly_stochastic(bad_matrix)
    assert not is_valid, "Should detect violation"
    print(f"✅ Test 3 passed: Violation detected - row dev: {evidence['max_row_deviation']:.4f}, col dev: {evidence['max_col_deviation']:.4f}")


def test_mass_conservation():
    """Regression test for mass conservation invariant"""
    guard = MathematicalInvariantGuard(ToleranceLevel.STANDARD)

    # Test 1: Perfect conservation
    initial = np.ones(100)
    final = initial.copy()
    is_valid, evidence = guard.check_mass_conservation(initial, final)
    assert is_valid, f"Perfect conservation failed: {evidence}"
    print("✅ Test 1 passed: Perfect mass conservation")

    # Test 2: Permutation preserves mass
    initial = np.random.rand(100)
    final = np.random.permutation(initial)
    is_valid, evidence = guard.check_mass_conservation(initial, final)
    assert is_valid, f"Permutation conservation failed: {evidence}"
    print("✅ Test 2 passed: Permutation preserves mass")

    # Test 3: Detect violations
    initial = np.ones(100)
    final = initial * 1.01  # 1% increase
    is_valid, evidence = guard.check_mass_conservation(initial, final, tolerance=1e-3)
    assert not is_valid, "Should detect violation"
    print(f"✅ Test 3 passed: Violation detected - deviation: {evidence['relative_deviation']:.4%}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MATHEMATICAL INVARIANT GUARD REGRESSION TESTS")
    print("="*80 + "\n")

    print("Testing Transport Doubly Stochastic Invariant...")
    test_transport_doubly_stochastic()

    print("\nTesting Mass Conservation Invariant...")
    test_mass_conservation()

    print("\n" + "="*80)
    print("ALL REGRESSION TESTS PASSED ✅")
    print("="*80)

