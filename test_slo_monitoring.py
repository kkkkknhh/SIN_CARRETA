#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SLO Monitoring and Alerting System
"""

import pytest
import time
from datetime import datetime, timedelta
from slo_monitoring import (
    SLOMonitor,
    SLOThresholds,
    MetricsAggregator,
    DashboardDataGenerator,
    AlertType,
    AlertSeverity,
    create_slo_monitor
)


class TestMetricsAggregator:
    """Test metrics aggregation"""

    def test_record_request(self):
        """Test recording request metrics"""
        aggregator = MetricsAggregator(window_size_seconds=60)
        
        aggregator.record_request(
            flow_name="test_flow",
            success=True,
            latency_ms=100.0,
            contract_valid=True
        )
        
        metrics = aggregator.get_flow_metrics("test_flow")
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert len(metrics.latencies) == 1

    def test_record_recovery(self):
        """Test recording recovery metrics"""
        aggregator = MetricsAggregator(window_size_seconds=60)
        
        aggregator.record_recovery(
            flow_name="test_flow",
            success=True,
            recovery_time_ms=1200.0
        )
        
        metrics = aggregator.get_flow_metrics("test_flow")
        assert metrics.recovery_attempts == 1
        assert metrics.recovery_successes == 1
        assert len(metrics.recovery_times) == 1

    def test_get_all_flows(self):
        """Test getting all tracked flows"""
        aggregator = MetricsAggregator()
        
        aggregator.record_request("flow1", True, 100.0)
        aggregator.record_request("flow2", True, 150.0)
        aggregator.record_request("flow3", False, 200.0)
        
        flows = aggregator.get_all_flows()
        assert len(flows) == 3
        assert "flow1" in flows
        assert "flow2" in flows
        assert "flow3" in flows


class TestFlowMetrics:
    """Test flow metrics calculations"""

    def test_availability_calculation(self):
        """Test availability percentage calculation"""
        aggregator = MetricsAggregator()
        
        # Record 90% success rate
        for i in range(10):
            aggregator.record_request(
                flow_name="test_flow",
                success=(i != 9),  # 1 failure
                latency_ms=100.0
            )
        
        metrics = aggregator.get_flow_metrics("test_flow")
        assert metrics.availability == 90.0

    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        aggregator = MetricsAggregator()
        
        # Record 20% error rate
        for i in range(10):
            aggregator.record_request(
                flow_name="test_flow",
                success=(i % 5 != 0),  # 2 failures
                latency_ms=100.0
            )
        
        metrics = aggregator.get_flow_metrics("test_flow")
        assert metrics.error_rate == 20.0

    def test_p95_latency_calculation(self):
        """Test p95 latency calculation"""
        aggregator = MetricsAggregator()
        
        # Record latencies from 100ms to 200ms
        for i in range(20):
            aggregator.record_request(
                flow_name="test_flow",
                success=True,
                latency_ms=100.0 + (i * 5)
            )
        
        metrics = aggregator.get_flow_metrics("test_flow")
        
        # P95 should be around 195ms
        assert 190.0 <= metrics.p95_latency <= 200.0

    def test_recovery_success_rate(self):
        """Test recovery success rate calculation"""
        aggregator = MetricsAggregator()
        
        # Record 80% recovery success rate
        for i in range(10):
            aggregator.record_recovery(
                flow_name="test_flow",
                success=(i % 5 != 0),  # 2 failures
                recovery_time_ms=1000.0
            )
        
        metrics = aggregator.get_flow_metrics("test_flow")
        assert metrics.recovery_success_rate == 80.0


class TestSLOMonitor:
    """Test SLO monitoring"""

    def test_initialization(self):
        """Test SLO monitor initialization"""
        thresholds = SLOThresholds(
            availability_percent=99.5,
            p95_latency_ms=200.0,
            error_rate_percent=0.1
        )
        
        monitor = SLOMonitor(thresholds=thresholds)
        
        assert monitor.thresholds.availability_percent == 99.5
        assert monitor.thresholds.p95_latency_ms == 200.0

    def test_check_slo_status_meeting_all(self):
        """Test SLO status when all metrics meet thresholds"""
        monitor = create_slo_monitor()
        
        # Record healthy requests
        for i in range(100):
            monitor.record_request(
                flow_name="healthy_flow",
                success=True,
                latency_ms=100.0,
                contract_valid=True
            )
        
        status = monitor.check_slo_status("healthy_flow")
        
        assert status.availability_slo_met
        assert status.p95_latency_slo_met
        assert status.error_rate_slo_met
        assert status.overall_slo_met

    def test_check_slo_status_availability_breach(self):
        """Test SLO status when availability breaches threshold"""
        monitor = create_slo_monitor(availability_threshold=99.5)
        
        # Record 98% success rate (breaches 99.5% threshold)
        for i in range(100):
            monitor.record_request(
                flow_name="degraded_flow",
                success=(i % 50 != 0),  # 2% failure rate
                latency_ms=100.0
            )
        
        status = monitor.check_slo_status("degraded_flow")
        
        assert not status.availability_slo_met
        assert not status.overall_slo_met
        assert status.availability < 99.5

    def test_check_slo_status_latency_breach(self):
        """Test SLO status when p95 latency breaches threshold"""
        monitor = create_slo_monitor(p95_latency_threshold_ms=200.0)
        
        # Record requests with high p95 latency
        for i in range(100):
            latency = 300.0 if i >= 95 else 100.0
            monitor.record_request(
                flow_name="slow_flow",
                success=True,
                latency_ms=latency
            )
        
        status = monitor.check_slo_status("slow_flow")
        
        assert not status.p95_latency_slo_met
        assert not status.overall_slo_met
        assert status.p95_latency_ms > 200.0

    def test_check_slo_status_error_rate_breach(self):
        """Test SLO status when error rate breaches threshold"""
        monitor = create_slo_monitor(error_rate_threshold=0.1)
        
        # Record 5% error rate (breaches 0.1% threshold)
        for i in range(100):
            monitor.record_request(
                flow_name="error_flow",
                success=(i % 20 != 0),  # 5% failure rate
                latency_ms=100.0
            )
        
        status = monitor.check_slo_status("error_flow")
        
        assert not status.error_rate_slo_met
        assert not status.overall_slo_met
        assert status.error_rate_percent > 0.1


class TestAlertGeneration:
    """Test alert generation"""

    def test_availability_alert(self):
        """Test alert generation for availability breach"""
        monitor = create_slo_monitor(availability_threshold=99.5)
        
        # Create availability breach
        for i in range(100):
            monitor.record_request(
                flow_name="degraded_flow",
                success=(i % 10 != 0),  # 10% failure rate
                latency_ms=100.0
            )
        
        alerts = monitor.evaluate_alert_rules()
        
        # Should generate availability alert
        availability_alerts = [a for a in alerts if a.alert_type == AlertType.AVAILABILITY_DEGRADED]
        assert len(availability_alerts) > 0
        assert availability_alerts[0].severity == AlertSeverity.CRITICAL

    def test_latency_alert(self):
        """Test alert generation for latency breach"""
        monitor = create_slo_monitor(p95_latency_threshold_ms=200.0)
        
        # Create latency breach
        for i in range(100):
            monitor.record_request(
                flow_name="slow_flow",
                success=True,
                latency_ms=300.0
            )
        
        alerts = monitor.evaluate_alert_rules()
        
        # Should generate latency alert
        latency_alerts = [a for a in alerts if a.alert_type == AlertType.SLO_BREACH]
        assert len(latency_alerts) > 0

    def test_contract_violation_alert(self):
        """Test alert generation for contract violations"""
        monitor = create_slo_monitor()
        
        # Record requests with contract violations
        for i in range(50):
            monitor.record_request(
                flow_name="contract_flow",
                success=True,
                latency_ms=100.0,
                contract_valid=(i != 10)  # One violation
            )
        
        alerts = monitor.evaluate_alert_rules()
        
        # Should generate contract violation alert
        contract_alerts = [a for a in alerts if a.alert_type == AlertType.CONTRACT_VIOLATION]
        assert len(contract_alerts) > 0
        assert contract_alerts[0].severity == AlertSeverity.CRITICAL

    def test_performance_regression_alert(self):
        """Test alert generation for performance regression"""
        monitor = create_slo_monitor(performance_regression_threshold=10.0)
        
        # Set baseline with 100ms p95 latency
        for i in range(50):
            monitor.record_request("test_flow", True, 100.0)
        monitor.set_baseline("test_flow")
        
        # Clear metrics and record new slower requests (150ms = 50% regression)
        monitor.aggregator.clear_metrics()
        for i in range(50):
            monitor.record_request("test_flow", True, 150.0)
        
        alerts = monitor.evaluate_alert_rules()
        
        # Should generate performance regression alert
        regression_alerts = [a for a in alerts if a.alert_type == AlertType.PERFORMANCE_REGRESSION]
        assert len(regression_alerts) > 0

    def test_fault_recovery_alert(self):
        """Test alert generation for fault recovery failure"""
        monitor = create_slo_monitor(p99_recovery_time_threshold_ms=1500.0)
        
        # Record slow recovery times
        for i in range(50):
            monitor.record_recovery(
                flow_name="recovery_flow",
                success=True,
                recovery_time_ms=2000.0  # Exceeds 1500ms threshold
            )
        
        alerts = monitor.evaluate_alert_rules()
        
        # Should generate fault recovery alert
        recovery_alerts = [a for a in alerts if a.alert_type == AlertType.FAULT_RECOVERY_FAILURE]
        assert len(recovery_alerts) > 0
        assert recovery_alerts[0].severity == AlertSeverity.CRITICAL


class TestDashboardDataGenerator:
    """Test dashboard data generation"""

    def test_generate_dashboard_data(self):
        """Test dashboard data generation"""
        monitor = create_slo_monitor()
        
        # Record some metrics
        for i in range(100):
            monitor.record_request("flow1", True, 100.0)
            monitor.record_request("flow2", True, 150.0)
        
        generator = DashboardDataGenerator(monitor)
        dashboard = generator.generate_dashboard_data()
        
        assert "timestamp" in dashboard
        assert "overall" in dashboard
        assert "thresholds" in dashboard
        assert "flows" in dashboard
        assert "alerts" in dashboard

    def test_dashboard_flow_data(self):
        """Test dashboard flow data structure"""
        monitor = create_slo_monitor()
        
        # Record metrics for one flow
        for i in range(100):
            monitor.record_request("test_flow", True, 100.0)
        
        generator = DashboardDataGenerator(monitor)
        dashboard = generator.generate_dashboard_data()
        
        assert "test_flow" in dashboard["flows"]
        
        flow_data = dashboard["flows"]["test_flow"]
        assert "availability" in flow_data
        assert "p95_latency" in flow_data
        assert "error_rate" in flow_data
        assert "overall_slo_met" in flow_data
        
        # Check status indicators
        assert flow_data["availability"]["status_indicator"] in ["green", "yellow", "red"]

    def test_dashboard_overall_statistics(self):
        """Test dashboard overall statistics"""
        monitor = create_slo_monitor()
        
        # Record metrics for multiple flows
        for i in range(100):
            monitor.record_request("flow1", True, 100.0)
            monitor.record_request("flow2", True, 150.0)
            monitor.record_request("flow3", False, 200.0)  # Failing flow
        
        generator = DashboardDataGenerator(monitor)
        dashboard = generator.generate_dashboard_data()
        
        overall = dashboard["overall"]
        assert overall["total_flows"] == 3
        assert "flows_meeting_slo" in overall
        assert "slo_compliance_percent" in overall

    def test_dashboard_alert_data(self):
        """Test dashboard alert data"""
        monitor = create_slo_monitor(availability_threshold=99.5)
        
        # Create availability breach
        for i in range(100):
            monitor.record_request(
                flow_name="degraded_flow",
                success=(i % 10 != 0),  # 10% failure
                latency_ms=100.0
            )
        
        # Generate alerts
        monitor.evaluate_alert_rules()
        
        generator = DashboardDataGenerator(monitor)
        dashboard = generator.generate_dashboard_data()
        
        assert len(dashboard["alerts"]) > 0
        
        alert_data = dashboard["alerts"][0]
        assert "alert_id" in alert_data
        assert "alert_type" in alert_data
        assert "severity" in alert_data
        assert "flow_name" in alert_data
        assert "message" in alert_data


class TestFactoryFunction:
    """Test factory function"""

    def test_create_slo_monitor(self):
        """Test factory function creates monitor correctly"""
        monitor = create_slo_monitor(
            availability_threshold=99.0,
            p95_latency_threshold_ms=300.0,
            error_rate_threshold=1.0,
            performance_regression_threshold=15.0,
            p99_recovery_time_threshold_ms=2000.0
        )

        assert monitor.thresholds.availability_percent == 99.0
        assert monitor.thresholds.p95_latency_ms == 300.0
        assert monitor.thresholds.error_rate_percent == 1.0
        assert monitor.thresholds.performance_regression_percent == 15.0
        assert monitor.thresholds.p99_recovery_time_ms == 2000.0


class TestAlertResolution:
    """Test alert resolution"""

    def test_resolve_alert(self):
        """Test resolving an alert"""
        monitor = create_slo_monitor()
        
        # Generate an alert
        for i in range(100):
            monitor.record_request("test_flow", False, 100.0)
        
        alerts = monitor.evaluate_alert_rules()
        assert len(alerts) > 0
        
        # Resolve the alert
        alert_id = alerts[0].alert_id
        monitor.resolve_alert(alert_id)
        
        # Check it's resolved
        resolved_alert = next(a for a in monitor._alerts if a.alert_id == alert_id)
        assert resolved_alert.resolved
        assert resolved_alert.resolved_at is not None

    def test_active_alerts_filter(self):
        """Test filtering for active alerts only"""
        monitor = create_slo_monitor()
        
        # Generate alerts
        for i in range(100):
            monitor.record_request("test_flow", False, 100.0)
        
        alerts = monitor.evaluate_alert_rules()
        assert len(alerts) > 0
        
        # All should be active initially
        active = monitor.get_active_alerts()
        assert len(active) == len(alerts)
        
        # Resolve one
        monitor.resolve_alert(alerts[0].alert_id)
        
        # Active count should decrease
        active_after = monitor.get_active_alerts()
        assert len(active_after) == len(alerts) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
