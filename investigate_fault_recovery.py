#!/usr/bin/env python3
"""
Fault Recovery Investigation Script
===================================
Instruments fault recovery code paths for network_failure, disk_full, and cpu_throttling
to identify timing delays and partial recovery causes.
"""

import time
import logging
from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, 
    FaultRecoveryManager
)
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TimingBreakdown:
    """Detailed timing for fault recovery phases"""
    phase: str
    start_time: float
    end_time: float
    duration_ms: float
    state_before: str
    state_after: str
    metadata: Dict[str, Any]


class InstrumentedCircuitBreaker(CircuitBreaker):
    """Circuit breaker with detailed timing instrumentation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        super().__init__(name, config)
        self.timing_log: List[TimingBreakdown] = []
        self.state_transitions_detailed: List[Dict[str, Any]] = []
    
    def _log_timing(self, phase: str, start: float, end: float, metadata: Dict = None):
        """Log timing for a specific phase"""
        breakdown = TimingBreakdown(
            phase=phase,
            start_time=start,
            end_time=end,
            duration_ms=(end - start) * 1000,
            state_before=self.state.value,
            state_after=self.state.value,
            metadata=metadata or {}
        )
        self.timing_log.append(breakdown)
        
        logger.info(
            f"[{self.name}] Phase: {phase}, Duration: {breakdown.duration_ms:.3f}ms, "
            f"State: {self.state.value}"
        )
    
    def call(self, func, *args, **kwargs):
        """Instrumented call with timing breakdowns"""
        call_start = time.time()
        
        # Phase 1: State check and validation
        check_start = time.time()
        with self.lock:
            self.metrics.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    reset_start = time.time()
                    self._transition_to(CircuitState.HALF_OPEN)
                    reset_end = time.time()
                    self._log_timing(
                        "state_transition_open_to_halfopen",
                        reset_start, reset_end,
                        {"reason": "timeout_expired"}
                    )
                else:
                    self.metrics.rejected_calls += 1
                    reject_time = time.time()
                    self._log_timing(
                        "request_rejection",
                        check_start, reject_time,
                        {"reason": "circuit_open", "opened_at": str(self.opened_at)}
                    )
                    from circuit_breaker import CircuitBreakerError
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry after {self.config.timeout_seconds}s"
                    )
        
        check_end = time.time()
        self._log_timing("state_check", check_start, check_end, {})
        
        # Phase 2: Function execution
        exec_start = time.time()
        try:
            result = func(*args, **kwargs)
            exec_end = time.time()
            self._log_timing(
                "function_execution", exec_start, exec_end,
                {"status": "success"}
            )
            
            # Phase 3: Success handling
            success_start = time.time()
            with self.lock:
                self._on_success((exec_end - exec_start))
            success_end = time.time()
            self._log_timing(
                "success_handling", success_start, success_end,
                {"consecutive_successes": self.metrics.consecutive_successes}
            )
            
            call_end = time.time()
            self._log_timing(
                "total_call", call_start, call_end,
                {"overall_status": "success"}
            )
            
            return result
            
        except Exception as e:
            exec_end = time.time()
            self._log_timing(
                "function_execution", exec_start, exec_end,
                {"status": "failed", "error": str(e)}
            )
            
            # Phase 4: Failure handling
            failure_start = time.time()
            with self.lock:
                self._on_failure(e, (exec_end - exec_start))
            failure_end = time.time()
            self._log_timing(
                "failure_handling", failure_start, failure_end,
                {"consecutive_failures": self.metrics.consecutive_failures}
            )
            
            call_end = time.time()
            self._log_timing(
                "total_call", call_start, call_end,
                {"overall_status": "failed", "error": str(e)}
            )
            
            raise
    
    def _transition_to(self, new_state: CircuitState):
        """Instrumented state transition"""
        transition_start = time.time()
        old_state = self.state
        
        # Record detailed transition
        transition = {
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now().isoformat(),
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
        }
        self.state_transitions_detailed.append(transition)
        
        # Call parent transition
        super()._transition_to(new_state)
        
        transition_end = time.time()
        self._log_timing(
            "state_transition",
            transition_start, transition_end,
            transition
        )
    
    def get_timing_report(self) -> Dict[str, Any]:
        """Generate detailed timing report"""
        if not self.timing_log:
            return {"message": "No timing data collected"}
        
        # Aggregate by phase
        phase_stats = {}
        for timing in self.timing_log:
            if timing.phase not in phase_stats:
                phase_stats[timing.phase] = {
                    "count": 0,
                    "total_ms": 0,
                    "min_ms": float('inf'),
                    "max_ms": 0,
                    "samples": []
                }
            
            stats = phase_stats[timing.phase]
            stats["count"] += 1
            stats["total_ms"] += timing.duration_ms
            stats["min_ms"] = min(stats["min_ms"], timing.duration_ms)
            stats["max_ms"] = max(stats["max_ms"], timing.duration_ms)
            stats["samples"].append(timing.duration_ms)
        
        # Calculate averages
        for phase, stats in phase_stats.items():
            stats["avg_ms"] = stats["total_ms"] / stats["count"]
            del stats["samples"]  # Remove raw samples from report
        
        return {
            "circuit_name": self.name,
            "total_timings": len(self.timing_log),
            "phase_breakdown": phase_stats,
            "state_transitions": self.state_transitions_detailed,
            "recovery_times": self.metrics.recovery_times
        }


def simulate_network_failure(duration_sec: float = 0.5):
    """Simulate network failure with configurable duration"""
    time.sleep(duration_sec)
    raise ConnectionError("Network timeout")


def simulate_disk_full():
    """Simulate disk full condition"""
    time.sleep(0.3)  # Simulate disk check time
    raise OSError("No space left on device")


def simulate_cpu_throttling():
    """Simulate CPU throttling"""
    time.sleep(0.2)  # Simulate throttled computation
    raise RuntimeError("CPU throttling detected")


def run_fault_injection_analysis():
    """Run comprehensive fault injection analysis with timing"""
    print("\n" + "="*80)
    print("FAULT RECOVERY TIMING ANALYSIS")
    print("="*80 + "\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    scenarios = [
        {
            "name": "network_failure",
            "circuit_name": "network_client",
            "fault_func": lambda: simulate_network_failure(0.5),
            "failure_threshold": 3,
            "timeout_seconds": 2.0
        },
        {
            "name": "disk_full",
            "circuit_name": "disk_operations",
            "fault_func": simulate_disk_full,
            "failure_threshold": 3,
            "timeout_seconds": 2.0
        },
        {
            "name": "cpu_throttling",
            "circuit_name": "cpu_operations",
            "fault_func": simulate_cpu_throttling,
            "failure_threshold": 3,
            "timeout_seconds": 2.0
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"Scenario: {scenario['name'].upper()}")
        print(f"{'─'*80}\n")
        
        # Create instrumented circuit
        config = CircuitBreakerConfig(
            failure_threshold=scenario["failure_threshold"],
            timeout_seconds=scenario["timeout_seconds"],
            success_threshold=2
        )
        circuit = InstrumentedCircuitBreaker(scenario["circuit_name"], config)
        
        # Phase 1: Trigger failures to open circuit
        print(f"Phase 1: Triggering {scenario['failure_threshold']} failures...")
        for i in range(scenario["failure_threshold"]):
            try:
                circuit.call(scenario["fault_func"])
            except Exception as e:
                print(f"  Attempt {i+1}: {type(e).__name__}: {e}")
        
        print(f"\nCircuit state: {circuit.state.value}")
        print(f"Consecutive failures: {circuit.metrics.consecutive_failures}")
        
        # Phase 2: Wait for timeout
        print(f"\nPhase 2: Waiting {scenario['timeout_seconds']}s for timeout...")
        time.sleep(scenario["timeout_seconds"] + 0.1)
        
        # Phase 3: Attempt recovery with success
        print("\nPhase 3: Attempting recovery...")
        success_func = lambda: "Success!"
        
        recovery_start = time.time()
        try:
            result1 = circuit.call(success_func)
            print(f"  Recovery attempt 1: {result1} (state: {circuit.state.value})")
            
            result2 = circuit.call(success_func)
            print(f"  Recovery attempt 2: {result2} (state: {circuit.state.value})")
        except Exception as e:
            print(f"  Recovery failed: {e}")
        
        recovery_end = time.time()
        recovery_time = (recovery_end - recovery_start) * 1000
        
        # Get timing report
        timing_report = circuit.get_timing_report()
        results[scenario["name"]] = {
            "scenario": scenario["name"],
            "recovery_time_ms": recovery_time,
            "final_state": circuit.state.value,
            "timing_report": timing_report
        }
        
        # Print detailed breakdown
        print(f"\n{'─'*40}")
        print("TIMING BREAKDOWN:")
        print(f"{'─'*40}")
        print(f"Total recovery time: {recovery_time:.3f}ms")
        print(f"\nPhase statistics:")
        for phase, stats in timing_report["phase_breakdown"].items():
            print(f"  {phase}:")
            print(f"    Count: {stats['count']}")
            print(f"    Average: {stats['avg_ms']:.3f}ms")
            print(f"    Min: {stats['min_ms']:.3f}ms")
            print(f"    Max: {stats['max_ms']:.3f}ms")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Recovery time: {result['recovery_time_ms']:.3f}ms")
        print(f"  Final state: {result['final_state']}")
    
    return results


if __name__ == "__main__":
    results = run_fault_injection_analysis()
    
    # Save results
    import json
    with open("fault_recovery_timing_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Analysis complete. Results saved to fault_recovery_timing_analysis.json")
