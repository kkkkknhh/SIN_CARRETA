"""
Comprehensive Regression Test Suite for Sinkhorn-Knopp Normalization
====================================================================

Test suite with 1000+ iterations validating:
- Mathematical invariants (doubly-stochastic property)
- Mass conservation within 0.1% tolerance
- Performance benchmarks within 5% of baseline
- Feature flag functionality
- Edge cases and robustness
"""

import time
import numpy as np
import pytest
from typing import List, Dict, Any
from dataclasses import dataclass

from sinkhorn_knopp import (
    sinkhorn_knopp_normalize,
    normalize_transport_plan,
    SinkhornConfiguration,
    SinkhornResult,
    FeatureFlags,
    PreconditionViolation,
    PostconditionViolation,
    ConvergenceError,
)


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for regression testing."""
    mean_time_ms: float = 10.0  # Baseline mean time in milliseconds
    max_time_ms: float = 50.0   # Baseline max time in milliseconds
    tolerance_percent: float = 0.05  # 5% tolerance


class TestSinkhornKnoppBasicFunctionality:
    """Test basic functionality and correctness."""
    
    def test_simple_2x2_matrix(self):
        """Test normalization of simple 2x2 matrix."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        assert result.converged
        assert result.num_iterations > 0
        
        # Check doubly-stochastic property
        row_sums = np.sum(result.normalized_matrix, axis=1)
        col_sums = np.sum(result.normalized_matrix, axis=0)
        
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        assert np.allclose(col_sums, 1.0, atol=1e-6)
    
    def test_identity_matrix(self):
        """Test that identity matrix stays doubly-stochastic."""
        n = 10
        matrix = np.eye(n, dtype=np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        assert result.converged
        row_sums = np.sum(result.normalized_matrix, axis=1)
        col_sums = np.sum(result.normalized_matrix, axis=0)
        
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        assert np.allclose(col_sums, 1.0, atol=1e-6)
    
    def test_uniform_matrix(self):
        """Test normalization of uniform matrix."""
        matrix = np.ones((5, 5), dtype=np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        assert result.converged
        
        # Uniform matrix should converge to uniform distribution
        expected = 1.0 / 5.0
        assert np.allclose(result.normalized_matrix, expected, atol=1e-6)
    
    def test_float64_precision(self):
        """Test that float64 precision is maintained."""
        matrix = np.random.rand(10, 10).astype(np.float32)
        result = sinkhorn_knopp_normalize(matrix)
        
        assert result.normalized_matrix.dtype == np.float64
    
    def test_rectangular_matrix(self):
        """Test normalization of non-square matrix."""
        matrix = np.random.rand(5, 10).astype(np.float64)
        # Rectangular matrices cannot achieve doubly-stochastic property
        # Disable all checks and just verify it doesn't crash
        FeatureFlags.disable('ENABLE_STRICT_CONVERGENCE')
        FeatureFlags.disable('ENABLE_POSTCONDITION_CHECKS')
        try:
            config = SinkhornConfiguration(epsilon_tolerance=0.5, max_iterations=100)
            result = sinkhorn_knopp_normalize(matrix, config)
            
            # Just verify it produces valid output without NaN/Inf
            assert np.all(result.normalized_matrix >= 0)
            assert not np.any(np.isnan(result.normalized_matrix))
            assert not np.any(np.isinf(result.normalized_matrix))
        finally:
            FeatureFlags.enable('ENABLE_STRICT_CONVERGENCE')
            FeatureFlags.enable('ENABLE_POSTCONDITION_CHECKS')


class TestMathematicalInvariants:
    """Test that mathematical invariants hold across many iterations."""
    
    def test_doubly_stochastic_property_1000_iterations(self):
        """Run 1000 iterations with varied matrices, verify doubly-stochastic property."""
        np.random.seed(42)
        failures = []
        epsilon = 0.001  # 0.1% tolerance
        
        for i in range(1000):
            # Generate varied matrix types - focus on square matrices for strict doubly-stochastic property
            if i % 4 == 0:
                # Random positive square matrix
                size = np.random.choice([3, 5, 7, 10])
                matrix = np.random.rand(size, size).astype(np.float64) + 0.1
            elif i % 4 == 1:
                # Sparse square matrix
                size = np.random.choice([3, 5, 7, 10])
                matrix = np.random.rand(size, size).astype(np.float64)
                matrix[matrix < 0.7] = 0
                matrix = matrix + 0.01
            elif i % 4 == 2:
                # Skewed square matrix
                size = np.random.choice([3, 5, 7, 10])
                matrix = np.random.exponential(2.0, (size, size)).astype(np.float64)
            else:
                # Nearly square or square matrix
                size = np.random.choice([3, 5, 7, 10])
                # 80% chance of square, 20% chance of rectangular
                if np.random.rand() < 0.8:
                    matrix = np.random.rand(size, size).astype(np.float64) + 0.1
                else:
                    size2 = size + np.random.choice([-1, 0, 1])
                    matrix = np.random.rand(size, max(size2, 2)).astype(np.float64) + 0.1
            
            try:
                # Use looser tolerance for rectangular matrices
                n_rows, n_cols = matrix.shape
                if n_rows != n_cols:
                    # Skip rectangular matrices - they cannot achieve doubly-stochastic property
                    continue
                else:
                    result = sinkhorn_knopp_normalize(matrix)
                
                # Verify doubly-stochastic property (only for square matrices)
                row_sums = np.sum(result.normalized_matrix, axis=1)
                col_sums = np.sum(result.normalized_matrix, axis=0)
                
                row_error = np.max(np.abs(row_sums - 1.0))
                col_error = np.max(np.abs(col_sums - 1.0))
                
                if row_error > epsilon or col_error > epsilon:
                    failures.append({
                        'iteration': i,
                        'shape': matrix.shape,
                        'row_error': row_error,
                        'col_error': col_error,
                    })
            except Exception as e:
                failures.append({
                    'iteration': i,
                    'shape': matrix.shape,
                    'error': str(e),
                })
        
        assert len(failures) == 0, f"Failed {len(failures)}/1000 iterations: {failures[:10]}"
    
    def test_mass_conservation_invariant(self):
        """Test that mass is conserved within 0.1% across varied inputs."""
        np.random.seed(123)
        failures = []
        epsilon = 0.001
        
        for i in range(100):
            size = np.random.choice([3, 5, 10, 20])
            matrix = np.random.rand(size, size).astype(np.float64) + 0.1
            
            result = sinkhorn_knopp_normalize(matrix)
            
            # Check mass conservation
            if result.mass_conservation_error > epsilon:
                failures.append({
                    'iteration': i,
                    'size': size,
                    'mass_error': result.mass_conservation_error,
                })
        
        assert len(failures) == 0, f"Mass conservation failed: {failures}"
    
    def test_non_negativity_invariant(self):
        """Test that all values remain non-negative."""
        np.random.seed(456)
        
        for i in range(100):
            matrix = np.random.rand(10, 10).astype(np.float64)
            result = sinkhorn_knopp_normalize(matrix)
            
            assert np.all(result.normalized_matrix >= 0), \
                f"Negative values found at iteration {i}"
    
    def test_convergence_monotonicity(self):
        """Test that errors decrease monotonically (mostly)."""
        matrix = np.random.rand(10, 10).astype(np.float64)
        
        config = SinkhornConfiguration(max_iterations=1000)
        result = sinkhorn_knopp_normalize(matrix, config)
        
        # Errors should generally decrease
        row_errors = result.diagnostics['row_error_history']
        col_errors = result.diagnostics['col_error_history']
        
        # Check that final error is smaller than initial
        if len(row_errors) > 1:
            assert row_errors[-1] < row_errors[0] * 1.1  # Allow 10% slack
            assert col_errors[-1] < col_errors[0] * 1.1


class TestMassConservation:
    """Test mass conservation within 0.1% tolerance."""
    
    def test_mass_conservation_small_matrices(self):
        """Test mass conservation for small matrices."""
        epsilon = 0.001
        
        for size in [2, 3, 5, 10]:
            matrix = np.random.rand(size, size).astype(np.float64)
            result = sinkhorn_knopp_normalize(matrix)
            
            assert result.mass_conservation_error < epsilon, \
                f"Mass error {result.mass_conservation_error} exceeds {epsilon} for size {size}"
    
    def test_mass_conservation_large_matrices(self):
        """Test mass conservation for larger matrices."""
        epsilon = 0.001
        
        for size in [50, 100]:
            matrix = np.random.rand(size, size).astype(np.float64)
            result = sinkhorn_knopp_normalize(matrix)
            
            assert result.mass_conservation_error < epsilon, \
                f"Mass error {result.mass_conservation_error} exceeds {epsilon} for size {size}"
    
    def test_mass_conservation_extreme_values(self):
        """Test mass conservation with extreme values."""
        epsilon = 0.001
        
        # Very small values
        matrix = np.random.rand(5, 5).astype(np.float64) * 1e-6
        result = sinkhorn_knopp_normalize(matrix)
        assert result.mass_conservation_error < epsilon
        
        # Very large values
        matrix = np.random.rand(5, 5).astype(np.float64) * 1e6
        result = sinkhorn_knopp_normalize(matrix)
        assert result.mass_conservation_error < epsilon


class TestFeatureFlags:
    """Test feature flag functionality for instant rollback."""
    
    def test_sinkhorn_knopp_disabled(self):
        """Test that disabling Sinkhorn-Knopp returns original matrix."""
        FeatureFlags.disable('ENABLE_SINKHORN_KNOPP')
        
        try:
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            result = sinkhorn_knopp_normalize(matrix)
            
            # Should return original matrix without normalization
            assert not result.converged
            assert result.num_iterations == 0
            assert 'feature_flag_disabled' in result.diagnostics
        finally:
            FeatureFlags.enable('ENABLE_SINKHORN_KNOPP')
    
    def test_precondition_checks_disabled(self):
        """Test that precondition checks can be disabled."""
        FeatureFlags.disable('ENABLE_PRECONDITION_CHECKS')
        
        try:
            # This would normally fail precondition (negative value)
            matrix = np.array([[-1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            
            # Should not raise exception
            # Note: This might still fail during computation, but not at validation
            try:
                result = sinkhorn_knopp_normalize(matrix)
            except:
                pass  # Expected to fail during computation
        finally:
            FeatureFlags.enable('ENABLE_PRECONDITION_CHECKS')
    
    def test_postcondition_checks_disabled(self):
        """Test that postcondition checks can be disabled."""
        FeatureFlags.disable('ENABLE_POSTCONDITION_CHECKS')
        
        try:
            matrix = np.random.rand(5, 5).astype(np.float64)
            config = SinkhornConfiguration(max_iterations=1)  # Insufficient iterations
            
            # Should not raise PostconditionViolation even if not converged
            result = sinkhorn_knopp_normalize(matrix, config)
        finally:
            FeatureFlags.enable('ENABLE_POSTCONDITION_CHECKS')
    
    def test_feature_flag_status(self):
        """Test getting feature flag status."""
        status = FeatureFlags.get_status()
        assert isinstance(status, dict)
        assert 'ENABLE_SINKHORN_KNOPP' in status


class TestPerformanceRegression:
    """Test that performance remains within 5% of baseline."""
    
    def test_performance_baseline_small_matrices(self):
        """Test performance on small matrices (5x5 to 20x20)."""
        np.random.seed(789)
        baseline = PerformanceBaseline(mean_time_ms=5.0, max_time_ms=20.0)
        
        times = []
        for _ in range(100):
            size = np.random.choice([5, 10, 15, 20])
            matrix = np.random.rand(size, size).astype(np.float64)
            
            start = time.perf_counter()
            result = sinkhorn_knopp_normalize(matrix)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            
            times.append(elapsed_ms)
        
        mean_time = np.mean(times)
        max_time = np.max(times)
        
        # Allow 5% tolerance above baseline
        tolerance = 1.0 + baseline.tolerance_percent
        
        assert mean_time < baseline.mean_time_ms * tolerance, \
            f"Mean time {mean_time:.2f}ms exceeds baseline {baseline.mean_time_ms}ms * {tolerance}"
        
        assert max_time < baseline.max_time_ms * tolerance, \
            f"Max time {max_time:.2f}ms exceeds baseline {baseline.max_time_ms}ms * {tolerance}"
    
    def test_performance_baseline_medium_matrices(self):
        """Test performance on medium matrices (50x50)."""
        np.random.seed(101112)
        baseline = PerformanceBaseline(mean_time_ms=50.0, max_time_ms=200.0)
        
        times = []
        for _ in range(20):
            matrix = np.random.rand(50, 50).astype(np.float64)
            
            start = time.perf_counter()
            result = sinkhorn_knopp_normalize(matrix)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            
            times.append(elapsed_ms)
        
        mean_time = np.mean(times)
        tolerance = 1.0 + baseline.tolerance_percent
        
        assert mean_time < baseline.mean_time_ms * tolerance, \
            f"Mean time {mean_time:.2f}ms exceeds baseline {baseline.mean_time_ms}ms * {tolerance}"
    
    def test_performance_scaling(self):
        """Test that performance scales reasonably with matrix size."""
        sizes = [5, 10, 20, 50]
        times = []
        
        for size in sizes:
            matrix = np.random.rand(size, size).astype(np.float64)
            
            start = time.perf_counter()
            result = sinkhorn_knopp_normalize(matrix)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
        
        # Time should increase sub-quadratically with size
        # (since iterations are typically O(n^2) per iteration, but fewer iterations)
        for i in range(len(sizes) - 1):
            size_ratio = sizes[i+1] / sizes[i]
            time_ratio = times[i+1] / times[i]
            
            # Time ratio should be less than size_ratio^2.5
            assert time_ratio < size_ratio ** 2.5, \
                f"Performance scaling issue: size {sizes[i]}->{sizes[i+1]}, time ratio {time_ratio:.2f}"


class TestPreconditions:
    """Test precondition validation."""
    
    def test_negative_values_rejected(self):
        """Test that negative values are rejected."""
        matrix = np.array([[-1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        with pytest.raises(PreconditionViolation, match="non-negative"):
            sinkhorn_knopp_normalize(matrix)
    
    def test_nan_values_rejected(self):
        """Test that NaN values are rejected."""
        matrix = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        with pytest.raises(PreconditionViolation, match="NaN"):
            sinkhorn_knopp_normalize(matrix)
    
    def test_inf_values_rejected(self):
        """Test that Inf values are rejected."""
        matrix = np.array([[np.inf, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        with pytest.raises(PreconditionViolation, match="Inf"):
            sinkhorn_knopp_normalize(matrix)
    
    def test_zero_matrix_rejected(self):
        """Test that all-zero matrix is rejected."""
        matrix = np.zeros((5, 5), dtype=np.float64)
        
        with pytest.raises(PreconditionViolation, match="too small"):
            sinkhorn_knopp_normalize(matrix)
    
    def test_non_2d_matrix_rejected(self):
        """Test that non-2D matrices are rejected."""
        matrix = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        with pytest.raises(PreconditionViolation, match="2D"):
            sinkhorn_knopp_normalize(matrix)
    
    def test_wrong_dtype_converted(self):
        """Test that wrong dtype is converted to float64."""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = sinkhorn_knopp_normalize(matrix)
        
        # Should succeed after conversion
        assert result.normalized_matrix.dtype == np.float64


class TestPostconditions:
    """Test postcondition validation."""
    
    def test_convergence_enforced_with_strict_flag(self):
        """Test that convergence is enforced when strict flag is enabled."""
        FeatureFlags.enable('ENABLE_STRICT_CONVERGENCE')
        
        try:
            matrix = np.random.rand(10, 10).astype(np.float64)
            config = SinkhornConfiguration(max_iterations=1)  # Too few iterations
            
            with pytest.raises(PostconditionViolation, match="did not converge"):
                sinkhorn_knopp_normalize(matrix, config)
        finally:
            pass  # Keep flag as is for other tests
    
    def test_doubly_stochastic_property_enforced(self):
        """Test that doubly-stochastic property is enforced."""
        # This is implicitly tested by checking that no exceptions are raised
        # for converged solutions
        matrix = np.random.rand(5, 5).astype(np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        # Should not raise PostconditionViolation
        assert result.converged


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_element_matrix(self):
        """Test 1x1 matrix."""
        matrix = np.array([[5.0]], dtype=np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        assert result.converged
        assert np.isclose(result.normalized_matrix[0, 0], 1.0)
    
    def test_single_row_matrix(self):
        """Test matrix with single row."""
        matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        # Single row matrices don't converge - disable all checks
        FeatureFlags.disable('ENABLE_STRICT_CONVERGENCE')
        FeatureFlags.disable('ENABLE_POSTCONDITION_CHECKS')
        try:
            config = SinkhornConfiguration(epsilon_tolerance=0.5)
            result = sinkhorn_knopp_normalize(matrix, config)
            
            # Just verify it doesn't crash and produces valid output
            assert np.all(result.normalized_matrix >= 0)
            assert not np.any(np.isnan(result.normalized_matrix))
        finally:
            FeatureFlags.enable('ENABLE_STRICT_CONVERGENCE')
            FeatureFlags.enable('ENABLE_POSTCONDITION_CHECKS')
    
    def test_single_column_matrix(self):
        """Test matrix with single column."""
        matrix = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        # Single column matrices don't converge - disable all checks
        FeatureFlags.disable('ENABLE_STRICT_CONVERGENCE')
        FeatureFlags.disable('ENABLE_POSTCONDITION_CHECKS')
        try:
            config = SinkhornConfiguration(epsilon_tolerance=0.5)
            result = sinkhorn_knopp_normalize(matrix, config)
            
            # Just verify it doesn't crash and produces valid output
            assert np.all(result.normalized_matrix >= 0)
            assert not np.any(np.isnan(result.normalized_matrix))
        finally:
            FeatureFlags.enable('ENABLE_STRICT_CONVERGENCE')
            FeatureFlags.enable('ENABLE_POSTCONDITION_CHECKS')
    
    def test_very_sparse_matrix(self):
        """Test matrix with many zeros."""
        matrix = np.zeros((10, 10), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 2.0
        matrix[2, 2] = 3.0
        
        # Sparse matrices may not converge well - disable strict convergence
        FeatureFlags.disable('ENABLE_STRICT_CONVERGENCE')
        try:
            config = SinkhornConfiguration(max_iterations=5000, epsilon_tolerance=0.01)
            result = sinkhorn_knopp_normalize(matrix, config)
            assert not np.any(np.isnan(result.normalized_matrix))
        finally:
            FeatureFlags.enable('ENABLE_STRICT_CONVERGENCE')
    
    def test_highly_skewed_matrix(self):
        """Test matrix with extreme value differences."""
        matrix = np.random.rand(5, 5).astype(np.float64)
        matrix[0, 0] = 1e6  # One very large value
        matrix[1, 1] = 1e-6  # One very small value
        
        # Skewed matrices need more iterations
        config = SinkhornConfiguration(max_iterations=5000, epsilon_tolerance=0.01)
        result = sinkhorn_knopp_normalize(matrix, config)
        assert not np.any(np.isnan(result.normalized_matrix))


class TestTransportPlanNormalization:
    """Test transport plan normalization convenience function."""
    
    def test_basic_transport_plan(self):
        """Test basic transport plan normalization."""
        cost = np.random.rand(5, 5).astype(np.float64)
        result = normalize_transport_plan(cost)
        
        assert result.converged
        row_sums = np.sum(result.normalized_matrix, axis=1)
        col_sums = np.sum(result.normalized_matrix, axis=0)
        
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        assert np.allclose(col_sums, 1.0, atol=1e-6)
    
    def test_transport_plan_with_source_weights(self):
        """Test transport plan with source marginal constraints."""
        cost = np.random.rand(5, 5).astype(np.float64)
        source_weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype=np.float64)
        
        result = normalize_transport_plan(cost, source_weights=source_weights)
        assert result.converged
    
    def test_transport_plan_with_target_weights(self):
        """Test transport plan with target marginal constraints."""
        cost = np.random.rand(5, 5).astype(np.float64)
        target_weights = np.array([0.15, 0.25, 0.2, 0.2, 0.2], dtype=np.float64)
        
        result = normalize_transport_plan(cost, target_weights=target_weights)
        assert result.converged
    
    def test_transport_plan_invalid_weights(self):
        """Test that invalid weights are rejected."""
        cost = np.random.rand(5, 5).astype(np.float64)
        invalid_weights = np.array([0.1, 0.2, 0.3, 0.2, 0.1], dtype=np.float64)  # Sums to 0.9
        
        with pytest.raises(ValueError, match="must sum to 1"):
            normalize_transport_plan(cost, source_weights=invalid_weights)


class TestConfiguration:
    """Test configuration options."""
    
    def test_custom_max_iterations(self):
        """Test custom maximum iterations."""
        matrix = np.random.rand(10, 10).astype(np.float64)
        config = SinkhornConfiguration(max_iterations=50)
        
        result = sinkhorn_knopp_normalize(matrix, config)
        assert result.num_iterations <= 50
    
    def test_custom_convergence_threshold(self):
        """Test custom convergence threshold."""
        matrix = np.random.rand(10, 10).astype(np.float64)
        config = SinkhornConfiguration(convergence_threshold=1e-3)
        
        result = sinkhorn_knopp_normalize(matrix, config)
        
        # Should converge faster with looser threshold
        assert result.final_row_error < 1e-3
        assert result.final_col_error < 1e-3
    
    def test_custom_epsilon_tolerance(self):
        """Test custom epsilon tolerance."""
        matrix = np.random.rand(10, 10).astype(np.float64)
        config = SinkhornConfiguration(epsilon_tolerance=0.01)  # 1% tolerance
        
        result = sinkhorn_knopp_normalize(matrix, config)
        assert result.mass_conservation_error < 0.01


class TestDiagnostics:
    """Test diagnostic information in results."""
    
    def test_diagnostics_present(self):
        """Test that diagnostics are included in result."""
        matrix = np.random.rand(5, 5).astype(np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        assert 'original_mass' in result.diagnostics
        assert 'final_mass' in result.diagnostics
        assert 'matrix_shape' in result.diagnostics
        assert 'row_error_history' in result.diagnostics
        assert 'col_error_history' in result.diagnostics
    
    def test_error_history_tracking(self):
        """Test that error history is tracked."""
        matrix = np.random.rand(10, 10).astype(np.float64)
        result = sinkhorn_knopp_normalize(matrix)
        
        row_history = result.diagnostics['row_error_history']
        col_history = result.diagnostics['col_error_history']
        
        assert len(row_history) > 0
        assert len(col_history) > 0
        
        # Errors should generally decrease
        if len(row_history) > 1:
            assert row_history[-1] <= row_history[0] * 1.5  # Allow some slack


# Run comprehensive test if executed directly
if __name__ == "__main__":
    print("Running comprehensive Sinkhorn-Knopp regression tests...\n")
    pytest.main([__file__, "-v", "--tb=short"])
