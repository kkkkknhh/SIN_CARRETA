#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Canary Deployment with Monitoring

Demonstrates progressive canary deployment with real-time monitoring.
"""

from canary_deployment import create_canary_controller
from slo_monitoring import create_slo_monitor
import random


def main():
    """Example canary deployment"""
    print("="*80)
    print("CANARY DEPLOYMENT EXAMPLE")
    print("="*80)
    
    # Create deployment controller with short hold times for demo
    controller = create_canary_controller(
        deployment_id="example-deployment-v1.0",
        canary_5_hold_seconds=2,
        canary_25_hold_seconds=2,
        full_rollout_hold_seconds=2,
        max_error_rate=10.0,
        max_p95_latency_ms=500.0
    )
    
    print("\nConfiguration:")
    print(f"  - 5% canary: 2 seconds")
    print(f"  - 25% canary: 2 seconds")
    print(f"  - 100% rollout: 2 seconds")
    print(f"  - Max error rate: 10.0%")
    print(f"  - Max p95 latency: 500ms")
    
    # Define request generator
    request_count = [0]
    
    def request_generator():
        """Generate simulated requests"""
        request_count[0] += 1
        request_id = f"req_{request_count[0]}"
        
        # Simulate healthy service
        latency_ms = random.uniform(80, 150)
        success = random.random() > 0.01  # 1% error rate
        contract_valid = True
        
        return (request_id, latency_ms, success, contract_valid)
    
    # Execute deployment
    print("\nStarting deployment...")
    result = controller.execute_deployment(request_generator)
    
    # Display results
    print("\n" + "="*80)
    print("DEPLOYMENT RESULTS")
    print("="*80)
    print(f"Success: {result.success}")
    print(f"Final Stage: {result.final_stage.value}")
    print(f"Stages: {' → '.join(s.value for s in result.stages_completed)}")
    
    if result.rollback_reason:
        print(f"Rollback: {result.rollback_reason.value}")
    
    print(f"\nTotal Requests Processed: {request_count[0]}")
    
    # Export metrics
    controller.export_metrics("output/example_metrics.json")
    print("\nMetrics exported to output/example_metrics.json")


if __name__ == "__main__":
    main()
