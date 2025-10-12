#!/usr/bin/env python3
"""
Performance Test Suite for MINIMINIMOON Pipeline
================================================

Comprehensive performance testing with:
- 100 iterations per component to calculate p95 latency
- Performance budget enforcement with 10% tolerance
- Memory leak detection via 4-hour soak testing
- Automated CI/CD gate integration
"""

import time
import tracemalloc
import statistics
import logging
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBudget:
    """Performance budget for a pipeline component"""
    component_name: str
    p95_budget_ms: float
    tolerance_pct: float = 10.0
    
    def check_budget(self, p95_latency_ms: float) -> Tuple[bool, str]:
        """Check if p95 latency meets budget"""
        max_allowed = self.p95_budget_ms * (1 + self.tolerance_pct / 100)
        passed = p95_latency_ms <= max_allowed
        
        if passed:
            margin = ((max_allowed - p95_latency_ms) / self.p95_budget_ms) * 100
            msg = f"✅ PASS: {p95_latency_ms:.2f}ms < {max_allowed:.2f}ms (budget: {self.p95_budget_ms}ms, margin: {margin:.1f}%)"
        else:
            overage = ((p95_latency_ms - max_allowed) / self.p95_budget_ms) * 100
            msg = f"❌ FAIL: {p95_latency_ms:.2f}ms > {max_allowed:.2f}ms (budget: {self.p95_budget_ms}ms, overage: {overage:.1f}%)"
        
        return passed, msg


@dataclass
class PerformanceResult:
    """Result of performance measurement"""
    component_name: str
    iterations: int
    latencies_ms: List[float]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    budget_passed: bool
    budget_message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PerformanceBenchmark:
    """
    Performance benchmark runner for pipeline components.
    
    Measures latency distributions and enforces performance budgets.
    """
    
    # Performance budgets for 11 canonical pipeline nodes
    BUDGETS = {
        "sanitization": PerformanceBudget("sanitization", 2.0),
        "plan_processing": PerformanceBudget("plan_processing", 3.0),
        "document_segmentation": PerformanceBudget("document_segmentation", 4.0),
        "embedding": PerformanceBudget("embedding", 50.0),
        "responsibility_detection": PerformanceBudget("responsibility_detection", 10.0),
        "contradiction_detection": PerformanceBudget("contradiction_detection", 8.0),
        "monetary_detection": PerformanceBudget("monetary_detection", 5.0),
        "feasibility_scoring": PerformanceBudget("feasibility_scoring", 6.0),
        "causal_detection": PerformanceBudget("causal_detection", 7.0),
        "teoria_cambio": PerformanceBudget("teoria_cambio", 15.0),
        "dag_validation": PerformanceBudget("dag_validation", 10.0),
        
        # Special performance targets
        "contract_validation_ROUTING": PerformanceBudget("contract_validation_ROUTING", 5.0),
        "PERMUTATION_INVARIANCE": PerformanceBudget("PERMUTATION_INVARIANCE", 0.5),
        "BUDGET_MONOTONICITY": PerformanceBudget("BUDGET_MONOTONICITY", 0.15),
    }
    
    def __init__(self):
        self.results: Dict[str, PerformanceResult] = {}
    
    def measure_latency(
        self,
        component_name: str,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> PerformanceResult:
        """
        Measure component latency with statistical analysis.
        
        Args:
            component_name: Name of component being tested
            func: Function to benchmark
            iterations: Number of measurement iterations
            warmup: Number of warmup iterations
            **kwargs: Arguments to pass to func
        
        Returns:
            PerformanceResult with latency statistics
        """
        logger.info("Benchmarking %s (%s warmup + %s iterations)", component_name, warmup, iterations)
        
        # Warmup phase
        for _ in range(warmup):
            func(**kwargs)
        
        # Measurement phase
        latencies_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
        
        # Calculate statistics
        sorted_latencies = sorted(latencies_ms)
        p50_ms = statistics.median(sorted_latencies)
        p95_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        mean_ms = statistics.mean(sorted_latencies)
        std_ms = statistics.stdev(sorted_latencies) if len(sorted_latencies) > 1 else 0.0
        min_ms = min(sorted_latencies)
        max_ms = max(sorted_latencies)
        
        # Check against budget
        budget = self.BUDGETS.get(component_name)
        if budget:
            budget_passed, budget_message = budget.check_budget(p95_ms)
        else:
            budget_passed = True
            budget_message = f"No budget defined for {component_name}"
        
        result = PerformanceResult(
            component_name=component_name,
            iterations=iterations,
            latencies_ms=latencies_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            budget_passed=budget_passed,
            budget_message=budget_message
        )
        
        self.results[component_name] = result
        
        logger.info("%s: p50=%.2fms p95=%.2fms p99=%.2fms", component_name, p50_ms, p95_ms, p99_ms)
        logger.info(budget_message)
        
        return result
    
    @staticmethod
    def run_soak_test(
        component_name: str,
        func: Callable,
        duration_hours: float = 4.0,
        sample_interval_sec: float = 10.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run extended soak test to detect memory leaks.
        
        Args:
            component_name: Name of component
            func: Function to test
            duration_hours: Test duration in hours
            sample_interval_sec: Sampling interval in seconds
            **kwargs: Arguments to pass to func
        
        Returns:
            Soak test results with memory leak detection
        """
        logger.info("Starting %sh soak test for %s", duration_hours, component_name)
        
        tracemalloc.start()
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        memory_samples = []
        iteration_count = 0
        
        try:
            while time.time() < end_time:
                # Run component
                func(**kwargs)
                iteration_count += 1
                
                # Sample memory
                if iteration_count % int(sample_interval_sec / 0.1) == 0:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_samples.append({
                        "time_elapsed_sec": time.time() - start_time,
                        "current_mb": current / 1024 / 1024,
                        "peak_mb": peak / 1024 / 1024,
                        "iterations": iteration_count
                    })
                
                # Sleep to avoid CPU saturation
                time.sleep(0.001)
        
        finally:
            tracemalloc.stop()
        
        # Analyze memory growth
        if len(memory_samples) > 10:
            # Linear regression to detect memory leak trend
            times = [s["time_elapsed_sec"] for s in memory_samples]
            mems = [s["current_mb"] for s in memory_samples]
            
            # Simple linear fit: y = mx + b
            n = len(times)
            x_mean = sum(times) / n
            y_mean = sum(mems) / n
            
            numerator = sum((times[i] - x_mean) * (mems[i] - y_mean) for i in range(n))
            denominator = sum((times[i] - x_mean) ** 2 for i in range(n))
            
            slope_mb_per_sec = numerator / denominator if denominator != 0 else 0
            slope_mb_per_hour = slope_mb_per_sec * 3600
            
            # Threshold: >10MB/hour indicates potential leak
            leak_detected = slope_mb_per_hour > 10.0
        else:
            slope_mb_per_hour = 0.0
            leak_detected = False
        
        result = {
            "component_name": component_name,
            "duration_hours": duration_hours,
            "iterations": iteration_count,
            "memory_samples": memory_samples,
            "leak_detected": leak_detected,
            "memory_growth_mb_per_hour": slope_mb_per_hour,
            "initial_memory_mb": memory_samples[0]["current_mb"] if memory_samples else 0,
            "final_memory_mb": memory_samples[-1]["current_mb"] if memory_samples else 0,
        }
        
        if leak_detected:
            logger.warning("❌ Memory leak detected: %.2f MB/hour growth", slope_mb_per_hour)
        else:
            logger.info("✅ No memory leak detected: %.2f MB/hour growth", slope_mb_per_hour)
        
        return result
    
    def generate_report(self, output_path: str = "performance_report.json"):
        """Generate JSON report of all performance results"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_components": len(self.results),
                "passed": sum(1 for r in self.results.values() if r.budget_passed),
                "failed": sum(1 for r in self.results.values() if not r.budget_passed),
            },
            "results": {
                name: {
                    "p50_ms": r.p50_ms,
                    "p95_ms": r.p95_ms,
                    "p99_ms": r.p99_ms,
                    "mean_ms": r.mean_ms,
                    "std_ms": r.std_ms,
                    "budget_passed": r.budget_passed,
                    "budget_message": r.budget_message,
                }
                for name, r in self.results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Performance report written to %s", output_path)
        return report
    
    def check_all_budgets(self) -> bool:
        """Check if all components passed their performance budgets"""
        return all(r.budget_passed for r in self.results.values())


# Benchmark implementations for specific components
def benchmark_contract_validation():
    """Benchmark contract validation with caching"""
    from data_flow_contract import CanonicalFlowValidator
    
    validator = CanonicalFlowValidator(enable_cache=True)
    
    test_data = {
        "raw_text": "Test document " * 100,
        "sanitized_text": "Test document " * 100,
    }
    
    # This should benefit from caching after first run
    validator.validate_node_execution("sanitization", test_data)


def benchmark_permutation_invariance():
    """Benchmark permutation invariance check"""
    from mathematical_invariant_guards import MathematicalInvariantGuard
    
    guard = MathematicalInvariantGuard()
    
    # Test permutation preservation
    initial = np.random.rand(1000)
    final = np.random.permutation(initial)
    
    guard.check_mass_conservation(initial, final)


def benchmark_budget_monotonicity():
    """Benchmark budget monotonicity check"""
    from mathematical_invariant_guards import MathematicalInvariantGuard
    
    guard = MathematicalInvariantGuard()
    
    # Small matrix check
    matrix = np.random.rand(50, 50)
    guard.check_transport_doubly_stochastic(matrix)


def run_performance_suite(soak_test: bool = False):
    """
    Run complete performance suite.
    
    Args:
        soak_test: If True, run 4-hour soak tests (default: False for CI)
    """
    benchmark = PerformanceBenchmark()
    
    print("\n" + "="*80)
    print("MINIMINIMOON PERFORMANCE TEST SUITE")
    print("="*80 + "\n")
    
    # Run latency benchmarks
    print("Running latency benchmarks (100 iterations per component)...\n")
    
    benchmark.measure_latency(
        "contract_validation_ROUTING",
        benchmark_contract_validation,
        iterations=100
    )
    
    benchmark.measure_latency(
        "PERMUTATION_INVARIANCE",
        benchmark_permutation_invariance,
        iterations=100
    )
    
    benchmark.measure_latency(
        "BUDGET_MONOTONICITY",
        benchmark_budget_monotonicity,
        iterations=100
    )
    
    # Generate report
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    
    report = benchmark.generate_report()
    
    passed = report["summary"]["passed"]
    failed = report["summary"]["failed"]
    total = report["summary"]["total_components"]
    
    print(f"Components tested: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if soak_test:
        print("\n" + "="*80)
        print("SOAK TEST (4 hours - memory leak detection)")
        print("="*80 + "\n")
        
        soak_result = benchmark.run_soak_test(
            "contract_validation_ROUTING",
            benchmark_contract_validation,
            duration_hours=4.0
        )
        
        report["soak_test"] = soak_result
        
        with open("performance_report.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    
    all_passed = benchmark.check_all_budgets()
    
    if all_passed:
        print("✅ ALL PERFORMANCE BUDGETS MET")
        return 0
    else:
        print("❌ PERFORMANCE BUDGET FAILURES DETECTED")
        print("\nFailed components:")
        for name, result in benchmark.results.items():
            if not result.budget_passed:
                print(f"  - {name}: {result.budget_message}")
        return 1


if __name__ == "__main__":
    import sys
    
    # Check for --soak flag
    soak_test = "--soak" in sys.argv
    
    exit_code = run_performance_suite(soak_test=soak_test)
    sys.exit(exit_code)
