#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLO Monitoring and Alerting System

Monitors Service Level Objectives (SLOs) for all critical flows:
- Availability: 99.5% threshold
- P95 Latency: 200ms threshold  
- Error Rate: 0.1% threshold

Features:
- Real-time metric collection and aggregation
- SLO dashboard data generation
- Alert rule evaluation
- Integration with Phase 3, 4, and 6 validation systems
"""

import logging
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertType(Enum):
    """Types of alerts"""
    CONTRACT_VIOLATION = "contract_violation"
    PERFORMANCE_REGRESSION = "performance_regression"
    FAULT_RECOVERY_FAILURE = "fault_recovery_failure"
    SLO_BREACH = "slo_breach"
    AVAILABILITY_DEGRADED = "availability_degraded"


@dataclass
class SLOThresholds:
    """SLO threshold definitions"""
    availability_percent: float = 99.5
    p95_latency_ms: float = 200.0
    error_rate_percent: float = 0.1
    
    # Phase 3: Performance regression threshold
    performance_regression_percent: float = 10.0
    
    # Phase 4: Fault recovery threshold
    p99_recovery_time_ms: float = 1500.0


@dataclass
class FlowMetrics:
    """Metrics for a single flow"""
    flow_name: str
    timestamp: datetime
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Latency metrics (milliseconds)
    latencies: List[float] = field(default_factory=list)
    
    # Contract validation
    contract_violations: int = 0
    
    # Recovery metrics
    recovery_attempts: int = 0
    recovery_successes: int = 0
    recovery_times: List[float] = field(default_factory=list)

    @property
    def availability(self) -> float:
        """Calculate availability percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def p50_latency(self) -> float:
        """Calculate p50 latency"""
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency(self) -> float:
        """Calculate p95 latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]

    @property
    def p99_latency(self) -> float:
        """Calculate p99 latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]

    @property
    def p99_recovery_time(self) -> float:
        """Calculate p99 recovery time"""
        if not self.recovery_times:
            return 0.0
        sorted_times = sorted(self.recovery_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]

    @property
    def recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        if self.recovery_attempts == 0:
            return 100.0
        return (self.recovery_successes / self.recovery_attempts) * 100


@dataclass
class Alert:
    """Alert triggered by SLO breach or validation failure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    flow_name: str
    timestamp: datetime
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class SLOStatus:
    """Current SLO status for a flow"""
    flow_name: str
    timestamp: datetime
    
    availability: float
    availability_slo_met: bool
    
    p95_latency_ms: float
    p95_latency_slo_met: bool
    
    error_rate_percent: float
    error_rate_slo_met: bool
    
    overall_slo_met: bool
    
    metrics: FlowMetrics


class MetricsAggregator:
    """
    Aggregates metrics across time windows for SLO calculation.
    """

    def __init__(self, window_size_seconds: int = 300):
        """
        Initialize metrics aggregator.
        
        Args:
            window_size_seconds: Time window for metric aggregation (default 5 minutes)
        """
        self.window_size = timedelta(seconds=window_size_seconds)
        self._metrics_by_flow: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = None

    def record_request(
        self,
        flow_name: str,
        success: bool,
        latency_ms: float,
        contract_valid: bool = True
    ):
        """Record a request metric"""
        metric = {
            'timestamp': datetime.now(),
            'success': success,
            'latency_ms': latency_ms,
            'contract_valid': contract_valid
        }
        self._metrics_by_flow[flow_name].append(metric)

    def record_recovery(
        self,
        flow_name: str,
        success: bool,
        recovery_time_ms: float
    ):
        """Record a fault recovery metric"""
        metric = {
            'timestamp': datetime.now(),
            'recovery_success': success,
            'recovery_time_ms': recovery_time_ms,
            'is_recovery': True
        }
        self._metrics_by_flow[flow_name].append(metric)

    def get_flow_metrics(self, flow_name: str) -> FlowMetrics:
        """
        Get aggregated metrics for a flow within the time window.
        
        Args:
            flow_name: Name of the flow
            
        Returns:
            Aggregated FlowMetrics
        """
        cutoff_time = datetime.now() - self.window_size
        recent_metrics = [
            m for m in self._metrics_by_flow[flow_name]
            if m['timestamp'] >= cutoff_time
        ]

        metrics = FlowMetrics(
            flow_name=flow_name,
            timestamp=datetime.now()
        )

        for m in recent_metrics:
            if m.get('is_recovery'):
                # Recovery metric
                metrics.recovery_attempts += 1
                if m['recovery_success']:
                    metrics.recovery_successes += 1
                metrics.recovery_times.append(m['recovery_time_ms'])
            else:
                # Request metric
                metrics.total_requests += 1
                if m['success']:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1
                
                metrics.latencies.append(m['latency_ms'])
                
                if not m.get('contract_valid', True):
                    metrics.contract_violations += 1

        return metrics

    def get_all_flows(self) -> List[str]:
        """Get list of all tracked flows"""
        return list(self._metrics_by_flow.keys())


class SLOMonitor:
    """
    Monitors SLOs for all critical flows and generates alerts.
    """

    def __init__(
        self,
        thresholds: Optional[SLOThresholds] = None,
        aggregation_window_seconds: int = 300
    ):
        """
        Initialize SLO monitor.
        
        Args:
            thresholds: SLO threshold configuration
            aggregation_window_seconds: Time window for metric aggregation
        """
        self.thresholds = thresholds or SLOThresholds()
        self.aggregator = MetricsAggregator(aggregation_window_seconds)
        
        self._alerts: List[Alert] = []
        self._baseline_metrics: Dict[str, FlowMetrics] = {}

    def record_request(
        self,
        flow_name: str,
        success: bool,
        latency_ms: float,
        contract_valid: bool = True
    ):
        """Record a request for SLO tracking"""
        self.aggregator.record_request(flow_name, success, latency_ms, contract_valid)

    def record_recovery(
        self,
        flow_name: str,
        success: bool,
        recovery_time_ms: float
    ):
        """Record a fault recovery event"""
        self.aggregator.record_recovery(flow_name, success, recovery_time_ms)

    def check_slo_status(self, flow_name: str) -> SLOStatus:
        """
        Check SLO status for a specific flow.
        
        Args:
            flow_name: Name of the flow to check
            
        Returns:
            Current SLO status
        """
        metrics = self.aggregator.get_flow_metrics(flow_name)

        # Check each SLO
        availability_met = metrics.availability >= self.thresholds.availability_percent
        p95_latency_met = metrics.p95_latency <= self.thresholds.p95_latency_ms
        error_rate_met = metrics.error_rate <= self.thresholds.error_rate_percent

        status = SLOStatus(
            flow_name=flow_name,
            timestamp=datetime.now(),
            availability=metrics.availability,
            availability_slo_met=availability_met,
            p95_latency_ms=metrics.p95_latency,
            p95_latency_slo_met=p95_latency_met,
            error_rate_percent=metrics.error_rate,
            error_rate_slo_met=error_rate_met,
            overall_slo_met=(availability_met and p95_latency_met and error_rate_met),
            metrics=metrics
        )

        return status

    def check_all_slos(self) -> Dict[str, SLOStatus]:
        """
        Check SLO status for all tracked flows.
        
        Returns:
            Dictionary mapping flow names to SLO status
        """
        statuses = {}
        for flow_name in self.aggregator.get_all_flows():
            statuses[flow_name] = self.check_slo_status(flow_name)
        return statuses

    def evaluate_alert_rules(self) -> List[Alert]:
        """
        Evaluate all alert rules and generate alerts for violations.
        
        Returns:
            List of newly generated alerts
        """
        new_alerts = []

        for flow_name in self.aggregator.get_all_flows():
            status = self.check_slo_status(flow_name)
            metrics = status.metrics

            # SLO breach alerts
            if not status.availability_slo_met:
                alert = Alert(
                    alert_id=f"{flow_name}_availability_{datetime.now().timestamp()}",
                    alert_type=AlertType.AVAILABILITY_DEGRADED,
                    severity=AlertSeverity.CRITICAL,
                    flow_name=flow_name,
                    timestamp=datetime.now(),
                    message=f"Availability below SLO: {status.availability:.2f}% < {self.thresholds.availability_percent}%",
                    details={
                        "current_availability": status.availability,
                        "threshold": self.thresholds.availability_percent,
                        "total_requests": metrics.total_requests,
                        "failed_requests": metrics.failed_requests
                    }
                )
                new_alerts.append(alert)

            if not status.p95_latency_slo_met:
                alert = Alert(
                    alert_id=f"{flow_name}_latency_{datetime.now().timestamp()}",
                    alert_type=AlertType.SLO_BREACH,
                    severity=AlertSeverity.WARNING,
                    flow_name=flow_name,
                    timestamp=datetime.now(),
                    message=f"P95 latency exceeds SLO: {status.p95_latency_ms:.2f}ms > {self.thresholds.p95_latency_ms}ms",
                    details={
                        "current_p95_latency": status.p95_latency_ms,
                        "threshold": self.thresholds.p95_latency_ms,
                        "p50_latency": metrics.p50_latency,
                        "p99_latency": metrics.p99_latency
                    }
                )
                new_alerts.append(alert)

            if not status.error_rate_slo_met:
                alert = Alert(
                    alert_id=f"{flow_name}_error_rate_{datetime.now().timestamp()}",
                    alert_type=AlertType.SLO_BREACH,
                    severity=AlertSeverity.CRITICAL,
                    flow_name=flow_name,
                    timestamp=datetime.now(),
                    message=f"Error rate exceeds SLO: {status.error_rate_percent:.3f}% > {self.thresholds.error_rate_percent}%",
                    details={
                        "current_error_rate": status.error_rate_percent,
                        "threshold": self.thresholds.error_rate_percent,
                        "total_requests": metrics.total_requests,
                        "failed_requests": metrics.failed_requests
                    }
                )
                new_alerts.append(alert)

            # Contract violation alerts
            if metrics.contract_violations > 0:
                alert = Alert(
                    alert_id=f"{flow_name}_contract_{datetime.now().timestamp()}",
                    alert_type=AlertType.CONTRACT_VIOLATION,
                    severity=AlertSeverity.CRITICAL,
                    flow_name=flow_name,
                    timestamp=datetime.now(),
                    message=f"Contract violations detected: {metrics.contract_violations} violations",
                    details={
                        "violation_count": metrics.contract_violations,
                        "total_requests": metrics.total_requests
                    }
                )
                new_alerts.append(alert)

            # Performance regression alerts (Phase 3)
            if flow_name in self._baseline_metrics:
                baseline = self._baseline_metrics[flow_name]
                if baseline.p95_latency > 0:
                    regression_percent = (
                        (metrics.p95_latency - baseline.p95_latency) / baseline.p95_latency * 100
                    )
                    
                    if regression_percent > self.thresholds.performance_regression_percent:
                        alert = Alert(
                            alert_id=f"{flow_name}_regression_{datetime.now().timestamp()}",
                            alert_type=AlertType.PERFORMANCE_REGRESSION,
                            severity=AlertSeverity.WARNING,
                            flow_name=flow_name,
                            timestamp=datetime.now(),
                            message=f"Performance regression detected: {regression_percent:.1f}% slower than baseline",
                            details={
                                "current_p95_latency": metrics.p95_latency,
                                "baseline_p95_latency": baseline.p95_latency,
                                "regression_percent": regression_percent,
                                "threshold": self.thresholds.performance_regression_percent
                            }
                        )
                        new_alerts.append(alert)

            # Fault recovery alerts (Phase 4)
            if metrics.recovery_attempts > 0:
                if metrics.p99_recovery_time > self.thresholds.p99_recovery_time_ms:
                    alert = Alert(
                        alert_id=f"{flow_name}_recovery_{datetime.now().timestamp()}",
                        alert_type=AlertType.FAULT_RECOVERY_FAILURE,
                        severity=AlertSeverity.CRITICAL,
                        flow_name=flow_name,
                        timestamp=datetime.now(),
                        message=f"P99 recovery time exceeds threshold: {metrics.p99_recovery_time:.2f}ms > {self.thresholds.p99_recovery_time_ms}ms",
                        details={
                            "p99_recovery_time": metrics.p99_recovery_time,
                            "threshold": self.thresholds.p99_recovery_time_ms,
                            "recovery_attempts": metrics.recovery_attempts,
                            "recovery_successes": metrics.recovery_successes,
                            "recovery_success_rate": metrics.recovery_success_rate
                        }
                    )
                    new_alerts.append(alert)

        # Store new alerts
        self._alerts.extend(new_alerts)
        
        if new_alerts:
            logger.warning(f"Generated {len(new_alerts)} new alerts")

        return new_alerts

    def set_baseline(self, flow_name: str):
        """Set current metrics as baseline for performance regression detection"""
        metrics = self.aggregator.get_flow_metrics(flow_name)
        self._baseline_metrics[flow_name] = metrics
        logger.info(f"Baseline set for {flow_name}: p95={metrics.p95_latency:.2f}ms")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [a for a in self._alerts if not a.resolved]

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self._alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                break


class DashboardDataGenerator:
    """
    Generates dashboard data for SLO visualization.
    """

    def __init__(self, monitor: SLOMonitor):
        """Initialize dashboard data generator"""
        self.monitor = monitor

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate complete dashboard data structure.
        
        Returns:
            Dashboard data with all flows and their SLO status
        """
        slo_statuses = self.monitor.check_all_slos()
        active_alerts = self.monitor.get_active_alerts()

        # Aggregate overall statistics
        total_flows = len(slo_statuses)
        flows_meeting_slo = sum(1 for s in slo_statuses.values() if s.overall_slo_met)
        overall_slo_compliance = (flows_meeting_slo / total_flows * 100) if total_flows > 0 else 100.0

        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "total_flows": total_flows,
                "flows_meeting_slo": flows_meeting_slo,
                "slo_compliance_percent": overall_slo_compliance,
                "active_alerts": len(active_alerts)
            },
            "thresholds": {
                "availability_percent": self.monitor.thresholds.availability_percent,
                "p95_latency_ms": self.monitor.thresholds.p95_latency_ms,
                "error_rate_percent": self.monitor.thresholds.error_rate_percent
            },
            "flows": {},
            "alerts": []
        }

        # Add per-flow data
        for flow_name, status in slo_statuses.items():
            dashboard["flows"][flow_name] = {
                "availability": {
                    "value": status.availability,
                    "threshold": self.monitor.thresholds.availability_percent,
                    "slo_met": status.availability_slo_met,
                    "status_indicator": "green" if status.availability_slo_met else "red"
                },
                "p95_latency": {
                    "value_ms": status.p95_latency_ms,
                    "threshold_ms": self.monitor.thresholds.p95_latency_ms,
                    "slo_met": status.p95_latency_slo_met,
                    "status_indicator": "green" if status.p95_latency_slo_met else "yellow"
                },
                "error_rate": {
                    "value_percent": status.error_rate_percent,
                    "threshold_percent": self.monitor.thresholds.error_rate_percent,
                    "slo_met": status.error_rate_slo_met,
                    "status_indicator": "green" if status.error_rate_slo_met else "red"
                },
                "overall_slo_met": status.overall_slo_met,
                "metrics": {
                    "total_requests": status.metrics.total_requests,
                    "successful_requests": status.metrics.successful_requests,
                    "failed_requests": status.metrics.failed_requests,
                    "p50_latency_ms": status.metrics.p50_latency,
                    "p99_latency_ms": status.metrics.p99_latency,
                    "contract_violations": status.metrics.contract_violations
                }
            }

        # Add alert data
        for alert in active_alerts:
            dashboard["alerts"].append({
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "flow_name": alert.flow_name,
                "timestamp": alert.timestamp.isoformat(),
                "message": alert.message,
                "details": alert.details
            })

        return dashboard

    def export_dashboard_json(self, output_path: str):
        """Export dashboard data to JSON file"""
        dashboard = self.generate_dashboard_data()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dashboard data exported to {output_path}")


def create_slo_monitor(
    availability_threshold: float = 99.5,
    p95_latency_threshold_ms: float = 200.0,
    error_rate_threshold: float = 0.1,
    performance_regression_threshold: float = 10.0,
    p99_recovery_time_threshold_ms: float = 1500.0
) -> SLOMonitor:
    """
    Factory function to create an SLO monitor.
    
    Args:
        availability_threshold: Minimum availability percentage (default 99.5%)
        p95_latency_threshold_ms: Maximum p95 latency in milliseconds (default 200ms)
        error_rate_threshold: Maximum error rate percentage (default 0.1%)
        performance_regression_threshold: Maximum performance regression percentage (default 10%)
        p99_recovery_time_threshold_ms: Maximum p99 recovery time in milliseconds (default 1500ms)
    
    Returns:
        Configured SLOMonitor instance
    """
    thresholds = SLOThresholds(
        availability_percent=availability_threshold,
        p95_latency_ms=p95_latency_threshold_ms,
        error_rate_percent=error_rate_threshold,
        performance_regression_percent=performance_regression_threshold,
        p99_recovery_time_ms=p99_recovery_time_threshold_ms
    )

    return SLOMonitor(thresholds=thresholds)
