#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Canary Deployment Infrastructure
"""

import pytest
import time
import random
from datetime import datetime
from canary_deployment import (
    CanaryDeploymentController,
    TrafficRoutingConfig,
    RollbackThresholds,
    DeploymentStage,
    RollbackReason,
    create_canary_controller
)


class TestTrafficRouting:
    """Test traffic routing functionality"""

    def test_progressive_traffic_routing(self):
        """Test that traffic routing progresses through stages correctly"""
        config = TrafficRoutingConfig(
            canary_5_hold_duration_seconds=1,
            canary_25_hold_duration_seconds=1,
            full_rollout_hold_duration_seconds=1
        )
        
        controller = CanaryDeploymentController(
            deployment_id="test_routing",
            traffic_config=config
        )

        # Initial state - baseline
        assert controller.router.get_current_percentage() == 0.0

        # Set to 5% canary
        controller.router.set_stage(DeploymentStage.CANARY_5)
        assert controller.router.get_current_percentage() == 5.0

        # Set to 25% canary
        controller.router.set_stage(DeploymentStage.CANARY_25)
        assert controller.router.get_current_percentage() == 25.0

        # Set to full rollout
        controller.router.set_stage(DeploymentStage.FULL_ROLLOUT)
        assert controller.router.get_current_percentage() == 100.0

    def test_deterministic_routing(self):
        """Test that routing is deterministic based on request ID"""
        controller = create_canary_controller("test_deterministic")
        controller.router.set_stage(DeploymentStage.CANARY_5)

        # Same request ID should always route to same version
        request_id = "test_request_123"
        version1 = controller.router.route_request(request_id)
        version2 = controller.router.route_request(request_id)
        
        assert version1 == version2


class TestMetricsCollection:
    """Test metrics collection and analysis"""

    def test_metrics_recording(self):
        """Test that metrics are recorded correctly"""
        controller = create_canary_controller("test_metrics")
        
        # Record some metrics
        controller.metrics_collector.record_request(
            request_id="req1",
            version="canary",
            latency_ms=100.0,
            success=True,
            contract_valid=True
        )
        
        controller.metrics_collector.record_request(
            request_id="req2",
            version="canary",
            latency_ms=150.0,
            success=False,
            contract_valid=True
        )

        # Analyze metrics
        metrics, should_rollback, reason = controller.metrics_collector.analyze_canary_metrics(
            DeploymentStage.CANARY_5,
            5.0
        )

        # Should have recorded 2 requests
        assert metrics.request_count == 2

    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        controller = create_canary_controller("test_error_rate")
        
        # Record requests with 20% error rate
        for i in range(20):
            success = (i % 5) != 0  # 20% failure rate
            controller.metrics_collector.record_request(
                request_id=f"req{i}",
                version="canary",
                latency_ms=100.0,
                success=success,
                contract_valid=True
            )

        metrics, should_rollback, reason = controller.metrics_collector.analyze_canary_metrics(
            DeploymentStage.CANARY_5,
            5.0
        )

        # Error rate should be approximately 20%
        assert 18.0 <= metrics.error_rate <= 22.0
        
        # Should trigger rollback (> 10% threshold)
        assert should_rollback
        assert reason == RollbackReason.ERROR_RATE_EXCEEDED

    def test_latency_calculation(self):
        """Test p95 latency calculation"""
        controller = create_canary_controller("test_latency")
        
        # Record requests with varying latency
        latencies = [100.0 + i * 10 for i in range(20)]
        for i, latency in enumerate(latencies):
            controller.metrics_collector.record_request(
                request_id=f"req{i}",
                version="canary",
                latency_ms=latency,
                success=True,
                contract_valid=True
            )

        metrics, should_rollback, reason = controller.metrics_collector.analyze_canary_metrics(
            DeploymentStage.CANARY_5,
            5.0
        )

        # P95 should be around 290ms (95th percentile of 100-290ms range)
        assert 280.0 <= metrics.p95_latency_ms <= 300.0


class TestRollbackTriggers:
    """Test automated rollback triggers"""

    def test_contract_violation_rollback(self):
        """Test rollback on contract violation"""
        controller = create_canary_controller("test_contract_rollback")
        
        # Record requests with contract violations
        for i in range(15):
            controller.metrics_collector.record_request(
                request_id=f"req{i}",
                version="canary",
                latency_ms=100.0,
                success=True,
                contract_valid=(i != 10)  # One violation
            )

        metrics, should_rollback, reason = controller.metrics_collector.analyze_canary_metrics(
            DeploymentStage.CANARY_5,
            5.0
        )

        # Should trigger rollback on contract violation
        assert should_rollback
        assert reason == RollbackReason.CONTRACT_VIOLATION
        assert metrics.contract_violations == 1

    def test_error_rate_rollback(self):
        """Test rollback when error rate exceeds threshold"""
        thresholds = RollbackThresholds(max_error_rate=10.0)
        controller = CanaryDeploymentController(
            deployment_id="test_error_rollback",
            rollback_thresholds=thresholds
        )
        
        # Record requests with 15% error rate (exceeds 10% threshold)
        for i in range(20):
            success = (i % 7) != 0  # ~14% failure rate
            controller.metrics_collector.record_request(
                request_id=f"req{i}",
                version="canary",
                latency_ms=100.0,
                success=success,
                contract_valid=True
            )

        metrics, should_rollback, reason = controller.metrics_collector.analyze_canary_metrics(
            DeploymentStage.CANARY_5,
            5.0
        )

        # Should trigger rollback
        assert should_rollback
        assert reason == RollbackReason.ERROR_RATE_EXCEEDED

    def test_latency_rollback(self):
        """Test rollback when p95 latency exceeds threshold"""
        thresholds = RollbackThresholds(max_p95_latency_ms=500.0)
        controller = CanaryDeploymentController(
            deployment_id="test_latency_rollback",
            rollback_thresholds=thresholds
        )
        
        # Record requests with high latency
        for i in range(20):
            latency = 600.0 if i >= 18 else 200.0  # P95 will be 600ms
            controller.metrics_collector.record_request(
                request_id=f"req{i}",
                version="canary",
                latency_ms=latency,
                success=True,
                contract_valid=True
            )

        metrics, should_rollback, reason = controller.metrics_collector.analyze_canary_metrics(
            DeploymentStage.CANARY_5,
            5.0
        )

        # Should trigger rollback
        assert should_rollback
        assert reason == RollbackReason.LATENCY_EXCEEDED


class TestDeploymentExecution:
    """Test full deployment execution"""

    def test_successful_deployment(self):
        """Test successful deployment through all stages"""
        config = TrafficRoutingConfig(
            canary_5_hold_duration_seconds=1,
            canary_25_hold_duration_seconds=1,
            full_rollout_hold_duration_seconds=1
        )
        
        controller = CanaryDeploymentController(
            deployment_id="test_success",
            traffic_config=config
        )

        # Simulate healthy requests
        def healthy_request_generator():
            return (
                f"req_{random.randint(0, 10000)}",
                random.uniform(50.0, 150.0),  # Low latency
                True,  # Success
                True   # Contract valid
            )

        result = controller.execute_deployment(healthy_request_generator)

        # Should complete successfully
        assert result.success
        assert result.final_stage == DeploymentStage.FULL_ROLLOUT
        assert len(result.stages_completed) == 3
        assert result.rollback_reason is None

    def test_deployment_with_rollback(self):
        """Test deployment that triggers rollback"""
        config = TrafficRoutingConfig(
            canary_5_hold_duration_seconds=1,
            canary_25_hold_duration_seconds=1,
            full_rollout_hold_duration_seconds=1
        )
        
        thresholds = RollbackThresholds(max_error_rate=10.0)
        
        controller = CanaryDeploymentController(
            deployment_id="test_rollback",
            traffic_config=config,
            rollback_thresholds=thresholds
        )

        # Simulate unhealthy requests with high error rate
        def unhealthy_request_generator():
            success = random.random() > 0.2  # 20% error rate
            return (
                f"req_{random.randint(0, 10000)}",
                random.uniform(50.0, 150.0),
                success,
                True
            )

        result = controller.execute_deployment(unhealthy_request_generator)

        # Should trigger rollback
        assert not result.success
        assert result.final_stage == DeploymentStage.ROLLBACK
        assert result.rollback_reason == RollbackReason.ERROR_RATE_EXCEEDED


class TestFactoryFunction:
    """Test factory function"""

    def test_create_canary_controller(self):
        """Test factory function creates controller correctly"""
        controller = create_canary_controller(
            deployment_id="test_factory",
            canary_5_hold_seconds=10,
            canary_25_hold_seconds=20,
            full_rollout_hold_seconds=30,
            max_error_rate=5.0,
            max_p95_latency_ms=300.0
        )

        assert controller.deployment_id == "test_factory"
        assert controller.traffic_config.canary_5_hold_duration_seconds == 10
        assert controller.traffic_config.canary_25_hold_duration_seconds == 20
        assert controller.traffic_config.full_rollout_hold_duration_seconds == 30
        assert controller.rollback_thresholds.max_error_rate == 5.0
        assert controller.rollback_thresholds.max_p95_latency_ms == 300.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
