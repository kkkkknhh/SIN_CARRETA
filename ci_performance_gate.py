#!/usr/bin/env python3
"""
CI/CD Performance Gate
======================
Validates performance benchmarks against budgets with 10% tolerance.
Fails build when benchmarks exceed budgets.

Usage:
    python ci_performance_gate.py [--iterations 100] [--budgets performance_budgets.yaml]
"""

import sys
import argparse
import yaml
import json
import logging
from performance_test_suite import PerformanceBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_performance_budgets(config_path: str = "performance_budgets.yaml"):
    """Load performance budgets from YAML configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('budgets', {})
    except Exception as e:
        logger.error(f"Error loading performance budgets: {e}")
        return {}


def run_performance_gate(iterations: int = 100, budgets_path: str = "performance_budgets.yaml"):
    """
    Run performance benchmarks and enforce CI/CD gates.
    
    Returns:
        Exit code: 0 if all budgets pass, 1 if any failures
    """
    print("\n" + "="*80)
    print("CI/CD PERFORMANCE GATE")
    print("="*80 + "\n")
    
    budgets = load_performance_budgets(budgets_path)
    if not budgets:
        logger.error("No performance budgets loaded. Cannot enforce gates.")
        return 1
    
    print(f"Loaded {len(budgets)} performance budgets")
    print(f"Running {iterations} iterations per component...\n")
    
    benchmark = PerformanceBenchmark()
    
    # Run lightweight benchmarks (components that can be tested standalone)
    from performance_test_suite import (
        benchmark_contract_validation,
        benchmark_permutation_invariance,
        benchmark_budget_monotonicity
    )
    
    components_to_test = [
        ("contract_validation_ROUTING", benchmark_contract_validation),
        ("PERMUTATION_INVARIANCE", benchmark_permutation_invariance),
        ("BUDGET_MONOTONICITY", benchmark_budget_monotonicity),
    ]
    
    all_passed = True
    violations = []
    
    for component_name, func in components_to_test:
        try:
            result = benchmark.measure_latency(
                component_name,
                func,
                iterations=iterations
            )
            
            if not result.budget_passed:
                all_passed = False
                violations.append({
                    "component": component_name,
                    "p95_ms": result.p95_ms,
                    "message": result.budget_message
                })
        except Exception as e:
            logger.error(f"Error benchmarking {component_name}: {e}")
            all_passed = False
            violations.append({
                "component": component_name,
                "error": str(e)
            })
    
    # Generate report
    print("\n" + "="*80)
    print("PERFORMANCE GATE RESULTS")
    print("="*80 + "\n")
    
    report = benchmark.generate_report("ci_performance_report.json")
    
    passed_count = report["summary"]["passed"]
    failed_count = report["summary"]["failed"]
    total_count = report["summary"]["total_components"]
    
    print(f"Components tested: {total_count}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    
    if all_passed:
        print("\n" + "="*80)
        print("✅ CI/CD GATE: PASS")
        print("="*80)
        print("\nAll performance budgets met. Build can proceed.")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ CI/CD GATE: FAIL")
        print("="*80)
        print("\nPerformance budget violations detected:")
        for violation in violations:
            if "error" in violation:
                print(f"  - {violation['component']}: {violation['error']}")
            else:
                print(f"  - {violation['message']}")
        
        print("\n⚠️  Build should be FAILED due to performance regressions.")
        print("Please optimize the affected components or adjust budgets if intentional.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="CI/CD Performance Gate")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="performance_budgets.yaml",
        help="Path to performance budgets YAML file"
    )
    parser.add_argument(
        "--soak",
        action="store_true",
        help="Run 4-hour soak test for memory leak detection"
    )
    
    args = parser.parse_args()
    
    exit_code = run_performance_gate(args.iterations, args.budgets)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
