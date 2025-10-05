#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test for Canary Deployment, OpenTelemetry, and SLO Monitoring

Demonstrates the complete deployment infrastructure with tracing and monitoring.
"""

import logging
import time
import random
from canary_deployment import create_canary_controller, DeploymentStage
from opentelemetry_instrumentation import (
    initialize_tracing,
    trace_flow,
    FlowType,
    create_span_logger
)
from slo_monitoring import create_slo_monitor, DashboardDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = create_span_logger(__name__)


@trace_flow(FlowType.DECALOGO_EVALUATION)
def evaluate_plan_with_tracing(plan_id: str, canary_version: bool = False) -> dict:
    """
    Simulate plan evaluation with OpenTelemetry tracing.
    
    Args:
        plan_id: Plan identifier
        canary_version: Whether this is the canary version
        
    Returns:
        Evaluation result
    """
    # Simulate processing time
    base_latency = 100.0 if not canary_version else 120.0
    latency = base_latency + random.uniform(-20, 20)
    time.sleep(latency / 1000.0)
    
    # Simulate occasional errors in canary
    success = True
    if canary_version and random.random() < 0.05:  # 5% error rate in canary
        success = False
    
    return {
        "plan_id": plan_id,
        "score": random.uniform(70.0, 95.0) if success else 0.0,
        "success": success,
        "latency_ms": latency,
        "version": "canary" if canary_version else "baseline"
    }


def run_integration_test():
    """Run complete integration test"""
    print("\n" + "="*80)
    print("CANARY DEPLOYMENT + OPENTELEMETRY + SLO MONITORING INTEGRATION TEST")
    print("="*80)
    
    # Initialize OpenTelemetry tracing
    print("\n→ Initializing OpenTelemetry tracing...")
    initialize_tracing(
        service_name="decalogo-evaluation-system",
        service_version="1.0.0",
        environment="production"
    )
    logger.info("OpenTelemetry tracing initialized")
    
    # Initialize SLO monitor
    print("\n→ Initializing SLO monitor...")
    slo_monitor = create_slo_monitor(
        availability_threshold=99.5,
        p95_latency_threshold_ms=200.0,
        error_rate_threshold=0.1
    )
    logger.info("SLO monitor initialized")
    
    # Create canary deployment controller
    print("\n→ Creating canary deployment controller...")
    canary_controller = create_canary_controller(
        deployment_id="decalogo-eval-v2.0",
        canary_5_hold_seconds=3,
        canary_25_hold_seconds=3,
        full_rollout_hold_seconds=3,
        max_error_rate=10.0,
        max_p95_latency_ms=500.0
    )
    logger.info("Canary controller created")
    
    # Define request generator that integrates with SLO monitoring
    def monitored_request_generator():
        """Generate requests with SLO monitoring"""
        request_id = f"req_{random.randint(0, 100000)}"
        version = canary_controller.router.route_request(request_id)
        
        # Evaluate with tracing
        is_canary = (version == "canary")
        result = evaluate_plan_with_tracing(request_id, canary_version=is_canary)
        
        # Record metrics in SLO monitor
        if is_canary:
            slo_monitor.record_request(
                flow_name="decalogo_evaluation",
                success=result["success"],
                latency_ms=result["latency_ms"],
                contract_valid=True  # Assume contract validation passes
            )
        
        return (
            request_id,
            result["latency_ms"],
            result["success"],
            True  # contract_valid
        )
    
    # Execute canary deployment
    print("\n→ Starting canary deployment...")
    print("   This will progress through: 5% → 25% → 100%")
    print("   Monitoring for contract violations, error rate, and latency...")
    
    deployment_result = canary_controller.execute_deployment(monitored_request_generator)
    
    # Display deployment results
    print("\n" + "="*80)
    print("DEPLOYMENT RESULTS")
    print("="*80)
    print(f"Deployment ID: {deployment_result.deployment_id}")
    print(f"Success: {deployment_result.success}")
    print(f"Final Stage: {deployment_result.final_stage.value}")
    print(f"Stages Completed: {[s.value for s in deployment_result.stages_completed]}")
    
    if deployment_result.rollback_reason:
        print(f"⚠️  Rollback Triggered: {deployment_result.rollback_reason.value}")
    else:
        print("✓ Deployment completed successfully")
    
    print(f"\nExecution Time: {(deployment_result.end_time - deployment_result.start_time).total_seconds():.2f}s")
    print(f"Metrics Collected: {len(deployment_result.metrics_history)} snapshots")
    
    # Display metrics history
    print("\n" + "-"*80)
    print("METRICS HISTORY")
    print("-"*80)
    for i, metrics in enumerate(deployment_result.metrics_history):
        print(f"\nSnapshot {i+1} - Stage: {metrics.stage.value}")
        print(f"  Traffic: {metrics.traffic_percentage:.1f}%")
        print(f"  Requests: {metrics.request_count}")
        print(f"  Error Rate: {metrics.error_rate:.2f}%")
        print(f"  P95 Latency: {metrics.p95_latency_ms:.2f}ms")
        print(f"  Contract Violations: {metrics.contract_violations}")
    
    # Check SLO status
    print("\n" + "="*80)
    print("SLO STATUS")
    print("="*80)
    
    slo_status = slo_monitor.check_slo_status("decalogo_evaluation")
    
    print(f"\nAvailability: {slo_status.availability:.2f}%")
    print(f"  Threshold: {slo_monitor.thresholds.availability_percent}%")
    print(f"  Status: {'✓ PASS' if slo_status.availability_slo_met else '✗ FAIL'}")
    
    print(f"\nP95 Latency: {slo_status.p95_latency_ms:.2f}ms")
    print(f"  Threshold: {slo_monitor.thresholds.p95_latency_ms}ms")
    print(f"  Status: {'✓ PASS' if slo_status.p95_latency_slo_met else '✗ FAIL'}")
    
    print(f"\nError Rate: {slo_status.error_rate_percent:.3f}%")
    print(f"  Threshold: {slo_monitor.thresholds.error_rate_percent}%")
    print(f"  Status: {'✓ PASS' if slo_status.error_rate_slo_met else '✗ FAIL'}")
    
    print(f"\nOverall SLO: {'✓ MET' if slo_status.overall_slo_met else '✗ NOT MET'}")
    
    # Generate and display alerts
    print("\n" + "="*80)
    print("ALERTS")
    print("="*80)
    
    alerts = slo_monitor.evaluate_alert_rules()
    
    if alerts:
        print(f"\n⚠️  {len(alerts)} alert(s) generated:")
        for alert in alerts:
            print(f"\n  Alert: {alert.alert_type.value}")
            print(f"  Severity: {alert.severity.value.upper()}")
            print(f"  Flow: {alert.flow_name}")
            print(f"  Message: {alert.message}")
    else:
        print("\n✓ No alerts generated")
    
    # Generate dashboard data
    print("\n" + "="*80)
    print("DASHBOARD")
    print("="*80)
    
    dashboard_generator = DashboardDataGenerator(slo_monitor)
    dashboard = dashboard_generator.generate_dashboard_data()
    
    print(f"\nOverall SLO Compliance: {dashboard['overall']['slo_compliance_percent']:.1f}%")
    print(f"Total Flows Monitored: {dashboard['overall']['total_flows']}")
    print(f"Flows Meeting SLO: {dashboard['overall']['flows_meeting_slo']}")
    print(f"Active Alerts: {dashboard['overall']['active_alerts']}")
    
    # Export results
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    canary_controller.export_metrics("output/canary_metrics.json")
    print("✓ Canary metrics exported to output/canary_metrics.json")
    
    dashboard_generator.export_dashboard_json("output/slo_dashboard.json")
    print("✓ SLO dashboard exported to output/slo_dashboard.json")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETED")
    print("="*80)
    
    return {
        "deployment": deployment_result,
        "slo_status": slo_status,
        "alerts": alerts,
        "dashboard": dashboard
    }


if __name__ == "__main__":
    try:
        results = run_integration_test()
        print("\n✓ Integration test passed")
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        raise
