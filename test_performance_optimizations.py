#!/usr/bin/env python3
"""
Tests for Performance Optimizations
===================================

Tests for:
1. Contract validation caching layer
2. PERMUTATION_INVARIANCE optimization
3. BUDGET_MONOTONICITY optimization
4. Performance test suite infrastructure
"""

import pytest
import numpy as np
import time
from data_flow_contract import CanonicalFlowValidator, ValidationCache
from mathematical_invariant_guards import MathematicalInvariantGuard
from performance_test_suite import PerformanceBenchmark, PerformanceBudget


class TestValidationCache:
    """Test validation caching layer"""
    
    def test_cache_basic_operations(self):
        """Test basic cache get/put operations"""
        cache = ValidationCache(max_size=10)
        
        test_data = {"raw_text": "test"}
        node_name = "sanitization"
        
        # Cache miss
        result = cache.get(test_data, node_name)
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0
        
        # Store result
        cache.put(test_data, node_name, True, {"valid": True})
        
        # Cache hit
        result = cache.get(test_data, node_name)
        assert result is not None
        assert cache.hits == 1
        assert cache.misses == 1
        
        is_valid, report = result
        assert is_valid is True
        assert report["valid"] is True
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = ValidationCache(max_size=3)
        
        # Fill cache
        for i in range(3):
            data = {"key": f"value{i}"}
            cache.put(data, f"node{i}", True, {"valid": True})
        
        assert len(cache.cache) == 3
        
        # Add one more - should evict oldest
        data = {"key": "value3"}
        cache.put(data, "node3", True, {"valid": True})
        
        assert len(cache.cache) == 3
        
        # First entry should be evicted
        first_data = {"key": "value0"}
        result = cache.get(first_data, "node0")
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = ValidationCache(max_size=10)
        
        # Generate some cache activity
        for i in range(5):
            data = {"key": f"value{i}"}
            cache.get(data, "node")  # miss
            cache.put(data, "node", True, {"valid": True})
            cache.get(data, "node")  # hit
        
        stats = cache.get_stats()
        assert stats["hits"] == 5
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 5


class TestCanonicalFlowValidatorCaching:
    """Test contract validation with caching"""
    
    def test_validator_with_cache(self):
        """Test validator uses cache correctly"""
        validator = CanonicalFlowValidator(enable_cache=True, cache_size=100)
        
        test_data = {
            "raw_text": "Test document with sufficient length " * 10,
            "sanitized_text": "Test document " * 10,
        }
        
        # First validation - cache miss
        is_valid1, report1 = validator.validate_node_execution("sanitization", test_data)
        assert report1.get("cached") is False
        
        # Second validation - cache hit
        is_valid2, report2 = validator.validate_node_execution("sanitization", test_data)
        assert report2.get("cached") is True
        assert is_valid1 == is_valid2
        
        # Check cache stats
        stats = validator.get_cache_stats()
        assert stats is not None
        assert stats["hits"] >= 1
    
    def test_validator_cache_disabled(self):
        """Test validator works without cache"""
        validator = CanonicalFlowValidator(enable_cache=False)
        
        test_data = {
            "raw_text": "Test document " * 10,
        }
        
        is_valid, report = validator.validate_node_execution("sanitization", test_data)
        assert report.get("cached") is False
        
        # Cache stats should be None
        stats = validator.get_cache_stats()
        assert stats is None
    
    def test_cache_performance_improvement(self):
        """Test that caching improves performance"""
        validator_cached = CanonicalFlowValidator(enable_cache=True)
        validator_uncached = CanonicalFlowValidator(enable_cache=False)
        
        test_data = {
            "raw_text": "Test document " * 100,
            "sanitized_text": "Test document " * 100,
        }
        
        # Warm up
        validator_cached.validate_node_execution("sanitization", test_data)
        
        # Time cached operations
        iterations = 50
        
        start = time.perf_counter()
        for _ in range(iterations):
            validator_cached.validate_node_execution("sanitization", test_data)
        cached_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(iterations):
            validator_uncached.validate_node_execution("sanitization", test_data)
        uncached_time = time.perf_counter() - start
        
        # Cached should be significantly faster
        speedup = uncached_time / cached_time
        print(f"Cache speedup: {speedup:.2f}x")
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"


class TestMathematicalInvariantOptimizations:
    """Test optimizations for mathematical invariant checks"""
    
    def test_permutation_invariance_performance(self):
        """Test PERMUTATION_INVARIANCE is fast"""
        guard = MathematicalInvariantGuard()
        
        # Test with various sizes
        initial = np.random.rand(1000)
        final = np.random.permutation(initial)
        
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            guard.check_mass_conservation(initial, final)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / iterations
        
        print(f"PERMUTATION_INVARIANCE: {avg_ms:.3f}ms average")
        assert avg_ms < 0.5, f"Expected <0.5ms, got {avg_ms:.3f}ms"
    
    def test_budget_monotonicity_performance(self):
        """Test BUDGET_MONOTONICITY is fast"""
        guard = MathematicalInvariantGuard()
        
        # Small matrix for fast operations
        matrix = np.random.rand(50, 50)
        
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            guard.check_transport_doubly_stochastic(matrix)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / iterations
        
        print(f"BUDGET_MONOTONICITY: {avg_ms:.3f}ms average")
        assert avg_ms < 0.15, f"Expected <0.15ms, got {avg_ms:.3f}ms"
    
    def test_small_vs_large_array_optimization(self):
        """Test that small arrays use fast path"""
        guard = MathematicalInvariantGuard()
        
        # Small array (should use np.sum)
        small = np.random.rand(100)
        start = time.perf_counter()
        for _ in range(100):
            guard.check_mass_conservation(small, small)
        small_time = time.perf_counter() - start
        
        # Large array (should use Kahan sum)
        large = np.random.rand(20000)
        start = time.perf_counter()
        for _ in range(100):
            guard.check_mass_conservation(large, large)
        large_time = time.perf_counter() - start
        
        # Small should be much faster
        print(f"Small array: {small_time*1000:.2f}ms, Large array: {large_time*1000:.2f}ms")
        assert small_time < large_time


class TestPerformanceBudgets:
    """Test performance budget infrastructure"""
    
    def test_budget_check_pass(self):
        """Test budget check when within limits"""
        budget = PerformanceBudget("test_component", 10.0, tolerance_pct=10.0)
        
        # Within budget
        passed, msg = budget.check_budget(9.0)
        assert passed is True
        assert "PASS" in msg
        
        # At budget
        passed, msg = budget.check_budget(10.0)
        assert passed is True
        
        # Within tolerance (10%)
        passed, msg = budget.check_budget(10.5)
        assert passed is True
    
    def test_budget_check_fail(self):
        """Test budget check when exceeding limits"""
        budget = PerformanceBudget("test_component", 10.0, tolerance_pct=10.0)
        
        # Exceeds tolerance
        passed, msg = budget.check_budget(12.0)
        assert passed is False
        assert "FAIL" in msg
    
    def test_performance_benchmark_basic(self):
        """Test performance benchmark basic functionality"""
        benchmark = PerformanceBenchmark()
        
        # Define simple test function
        def simple_func(duration_ms=0.01):
            time.sleep(duration_ms / 1000)
        
        # Add budget for test
        benchmark.BUDGETS["test_func"] = PerformanceBudget("test_func", 1.0)
        
        # Run benchmark
        result = benchmark.measure_latency(
            "test_func",
            simple_func,
            iterations=10,
            warmup=2,
            duration_ms=0.01
        )
        
        assert result.iterations == 10
        assert result.p95_ms > 0
        assert result.p50_ms > 0
        assert len(result.latencies_ms) == 10


class TestPerformanceRegressions:
    """Integration tests to prevent performance regressions"""
    
    def test_contract_validation_under_5ms(self):
        """Test that contract validation is under 5ms p95"""
        validator = CanonicalFlowValidator(enable_cache=True)
        
        test_data = {
            "raw_text": "Test document " * 100,
            "sanitized_text": "Test document " * 100,
        }
        
        # Warmup
        for _ in range(10):
            validator.validate_node_execution("sanitization", test_data)
        
        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            validator.validate_node_execution("sanitization", test_data)
            latencies.append((time.perf_counter() - start) * 1000)
        
        p95 = sorted(latencies)[95]
        print(f"Contract validation p95: {p95:.2f}ms")
        assert p95 < 5.0, f"p95 latency {p95:.2f}ms exceeds 5ms budget"
    
    def test_all_optimizations_meet_budgets(self):
        """Test that all optimizations meet their performance budgets"""
        from performance_test_suite import (
            benchmark_contract_validation,
            benchmark_permutation_invariance,
            benchmark_budget_monotonicity
        )
        
        benchmark = PerformanceBenchmark()
        
        # Run all benchmarks
        benchmark.measure_latency(
            "contract_validation_ROUTING",
            benchmark_contract_validation,
            iterations=50
        )
        
        benchmark.measure_latency(
            "PERMUTATION_INVARIANCE",
            benchmark_permutation_invariance,
            iterations=50
        )
        
        benchmark.measure_latency(
            "BUDGET_MONOTONICITY",
            benchmark_budget_monotonicity,
            iterations=50
        )
        
        # All should pass budgets
        all_passed = benchmark.check_all_budgets()
        
        if not all_passed:
            for name, result in benchmark.results.items():
                if not result.budget_passed:
                    print(f"FAILED: {name} - {result.budget_message}")
        
        assert all_passed, "Some performance budgets were not met"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
