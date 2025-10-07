#!/usr/bin/env python3
"""
Contract Validation ROUTING Profiler
====================================
Profiles contract_validation_ROUTING execution using cProfile and line_profiler
to pinpoint which specific operations within the 7.9ms runtime are consuming the most time.
"""

import cProfile
import pstats
import io
import time
from typing import Dict, Any
from deterministic_pipeline_validator import (
    DeterministicPipelineValidator, ContractType
)


def profile_routing_contract_with_cprofile():
    """Profile using cProfile"""
    print("\n" + "="*80)
    print("CPROFILE ANALYSIS: contract_validation_ROUTING")
    print("="*80 + "\n")
    
    validator = DeterministicPipelineValidator()
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the validation
    profiler.enable()
    result = validator.validate_contract(ContractType.ROUTING)
    profiler.disable()
    
    # Get statistics
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("Top 30 functions by cumulative time:")
    print("─" * 80)
    stats.print_stats(30)
    
    print(s.getvalue())
    
    # Also sort by total time
    s2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=s2)
    stats2.strip_dirs()
    stats2.sort_stats('tottime')
    
    print("\n" + "="*80)
    print("Top 30 functions by total time:")
    print("─" * 80)
    stats2.print_stats(30)
    
    print(s2.getvalue())
    
    # Save to file
    profiler.dump_stats('routing_contract_profile.prof')
    print("\n✅ Profile saved to routing_contract_profile.prof")
    print("   View with: python -m pstats routing_contract_profile.prof")
    
    return result


def manual_timing_breakdown():
    """Manual timing breakdown of routing contract validation"""
    print("\n" + "="*80)
    print("MANUAL TIMING BREAKDOWN: contract_validation_ROUTING")
    print("="*80 + "\n")
    
    validator = DeterministicPipelineValidator()
    
    # Time overall
    start_overall = time.perf_counter()
    
    # Time initialization
    start_init = time.perf_counter()
    test_input = {"query": "test", "params": {"seed": 42}}
    end_init = time.perf_counter()
    init_time_ms = (end_init - start_init) * 1000
    
    # Time route selection test
    start_route = time.perf_counter()
    route1 = validator._simulate_routing(test_input)
    route2 = validator._simulate_routing(test_input)
    route_equal = (route1 == route2)
    end_route = time.perf_counter()
    route_time_ms = (end_route - start_route) * 1000
    
    # Time tie-breaking test
    start_tie = time.perf_counter()
    ties = [{"score": 1.0, "id": "a"}, {"score": 1.0, "id": "b"}]
    sorted1 = validator._simulate_tie_breaking(ties)
    sorted2 = validator._simulate_tie_breaking(ties)
    tie_equal = (sorted1 == sorted2)
    end_tie = time.perf_counter()
    tie_time_ms = (end_tie - start_tie) * 1000
    
    # Time result construction
    start_result = time.perf_counter()
    result = {
        "tests_passed": 2 if route_equal and tie_equal else 0,
        "tests_failed": 0 if route_equal and tie_equal else 2,
        "violations": []
    }
    end_result = time.perf_counter()
    result_time_ms = (end_result - start_result) * 1000
    
    end_overall = time.perf_counter()
    overall_time_ms = (end_overall - start_overall) * 1000
    
    # Print breakdown
    print("Phase Breakdown:")
    print("─" * 60)
    print(f"  Initialization:          {init_time_ms:8.4f} ms  ({init_time_ms/overall_time_ms*100:5.1f}%)")
    print(f"  Route selection test:    {route_time_ms:8.4f} ms  ({route_time_ms/overall_time_ms*100:5.1f}%)")
    print(f"  Tie-breaking test:       {tie_time_ms:8.4f} ms  ({tie_time_ms/overall_time_ms*100:5.1f}%)")
    print(f"  Result construction:     {result_time_ms:8.4f} ms  ({result_time_ms/overall_time_ms*100:5.1f}%)")
    print("─" * 60)
    print(f"  TOTAL:                   {overall_time_ms:8.4f} ms")
    print()
    
    return {
        "total_ms": overall_time_ms,
        "breakdown": {
            "initialization_ms": init_time_ms,
            "route_selection_ms": route_time_ms,
            "tie_breaking_ms": tie_time_ms,
            "result_construction_ms": result_time_ms
        },
        "percentages": {
            "initialization": init_time_ms/overall_time_ms*100,
            "route_selection": route_time_ms/overall_time_ms*100,
            "tie_breaking": tie_time_ms/overall_time_ms*100,
            "result_construction": result_time_ms/overall_time_ms*100
        }
    }


def run_multiple_iterations():
    """Run multiple iterations to get stable timing"""
    print("\n" + "="*80)
    print("ITERATION ANALYSIS: contract_validation_ROUTING")
    print("="*80 + "\n")
    
    validator = DeterministicPipelineValidator()
    iterations = 100
    times = []
    
    print(f"Running {iterations} iterations...")
    for i in range(iterations):
        start = time.perf_counter()
        validator.validate_contract(ContractType.ROUTING)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate median
    sorted_times = sorted(times)
    median_time = sorted_times[len(sorted_times) // 2]
    
    # Calculate standard deviation
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5
    
    print(f"\nStatistics over {iterations} iterations:")
    print("─" * 60)
    print(f"  Average:    {avg_time:8.4f} ms")
    print(f"  Median:     {median_time:8.4f} ms")
    print(f"  Min:        {min_time:8.4f} ms")
    print(f"  Max:        {max_time:8.4f} ms")
    print(f"  Std Dev:    {std_dev:8.4f} ms")
    print(f"  Range:      {max_time - min_time:8.4f} ms")
    print()
    
    return {
        "iterations": iterations,
        "average_ms": avg_time,
        "median_ms": median_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "std_dev_ms": std_dev
    }


def main():
    """Run all profiling analyses"""
    results = {}
    
    # 1. cProfile analysis
    print("\n" + "="*80)
    print("STARTING CONTRACT VALIDATION ROUTING PROFILING")
    print("="*80)
    
    try:
        routing_result = profile_routing_contract_with_cprofile()
        print(f"\nRouting contract validation result: {routing_result.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️  cProfile analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Manual timing breakdown
    try:
        timing_breakdown = manual_timing_breakdown()
        results["timing_breakdown"] = timing_breakdown
    except Exception as e:
        print(f"⚠️  Manual timing analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Multiple iterations
    try:
        iteration_stats = run_multiple_iterations()
        results["iteration_statistics"] = iteration_stats
    except Exception as e:
        print(f"⚠️  Iteration analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    import json
    with open("contract_validation_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - routing_contract_profile.prof (cProfile data)")
    print("  - contract_validation_profile_results.json (timing breakdown)")
    print("\n✅ Analysis complete")


if __name__ == "__main__":
    main()
