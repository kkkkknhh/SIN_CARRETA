#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canary Deployment Infrastructure

Progressive traffic routing with automated rollback triggers and real-time metric monitoring.

Features:
- Progressive traffic release: 5% → 25% → 100%
- Configurable hold durations at each stage
- Automated rollback on:
  * Contract violations
  * Error rate > 10%
  * p95 latency > 500ms
- Real-time metric collection and analysis
"""

import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from threading import Lock
import random

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Canary deployment stages"""
    BASELINE = "baseline"
    CANARY_5 = "canary_5"
    CANARY_25 = "canary_25"
    FULL_ROLLOUT = "full_rollout"
    ROLLBACK = "rollback"


class RollbackReason(Enum):
    """Reasons for automated rollback"""
    CONTRACT_VIOLATION = "contract_violation"
    ERROR_RATE_EXCEEDED = "error_rate_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"
    MANUAL = "manual"


@dataclass
class TrafficRoutingConfig:
    """Traffic routing configuration"""
    canary_5_percent: float = 5.0
    canary_25_percent: float = 25.0
    full_rollout_percent: float = 100.0
    canary_5_hold_duration_seconds: int = 300  # 5 minutes
    canary_25_hold_duration_seconds: int = 600  # 10 minutes
    full_rollout_hold_duration_seconds: int = 1800  # 30 minutes


@dataclass
class RollbackThresholds:
    """Rollback trigger thresholds"""
    max_error_rate: float = 10.0  # 10%
    max_p95_latency_ms: float = 500.0  # 500ms
    metric_collection_window_seconds: int = 60  # 1 minute
    min_samples_for_decision: int = 10


@dataclass
class DeploymentMetrics:
    """Metrics collected during deployment"""
    timestamp: datetime
    stage: DeploymentStage
    traffic_percentage: float
    request_count: int
    error_count: int
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    contract_violations: int


@dataclass
class DeploymentResult:
    """Result of a canary deployment"""
    deployment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    final_stage: DeploymentStage
    rollback_reason: Optional[RollbackReason]
    stages_completed: List[DeploymentStage]
    metrics_history: List[DeploymentMetrics]
    success: bool


class TrafficRouter:
    """
    Routes traffic between baseline and canary versions based on deployment stage.
    """

    def __init__(self, config: TrafficRoutingConfig):
        """Initialize traffic router"""
        self.config = config
        self.current_stage = DeploymentStage.BASELINE
        self.canary_percentage = 0.0
        self._lock = Lock()

    def set_stage(self, stage: DeploymentStage):
        """Set current deployment stage and update traffic routing"""
        with self._lock:
            self.current_stage = stage
            
            if stage == DeploymentStage.BASELINE:
                self.canary_percentage = 0.0
            elif stage == DeploymentStage.CANARY_5:
                self.canary_percentage = self.config.canary_5_percent
            elif stage == DeploymentStage.CANARY_25:
                self.canary_percentage = self.config.canary_25_percent
            elif stage == DeploymentStage.FULL_ROLLOUT:
                self.canary_percentage = self.config.full_rollout_percent
            elif stage == DeploymentStage.ROLLBACK:
                self.canary_percentage = 0.0
            
            logger.info(f"Traffic routing updated: stage={stage.value}, canary={self.canary_percentage}%")

    def route_request(self, request_id: str) -> str:
        """
        Route a request to either baseline or canary version.
        
        Args:
            request_id: Unique request identifier
            
        Returns:
            Version to route to: 'baseline' or 'canary'
        """
        with self._lock:
            # Use deterministic routing based on request_id hash
            hash_val = hash(request_id) % 100
            
            if hash_val < self.canary_percentage:
                return "canary"
            else:
                return "baseline"

    def get_current_percentage(self) -> float:
        """Get current canary traffic percentage"""
        with self._lock:
            return self.canary_percentage


class MetricsCollector:
    """
    Collects and aggregates deployment metrics in real-time.
    """

    def __init__(self, thresholds: RollbackThresholds):
        """Initialize metrics collector"""
        self.thresholds = thresholds
        self._metrics: List[Dict[str, Any]] = []
        self._lock = Lock()

    def record_request(
        self,
        request_id: str,
        version: str,
        latency_ms: float,
        success: bool,
        contract_valid: bool
    ):
        """Record metrics for a single request"""
        with self._lock:
            self._metrics.append({
                "timestamp": datetime.now(),
                "request_id": request_id,
                "version": version,
                "latency_ms": latency_ms,
                "success": success,
                "contract_valid": contract_valid
            })

    def analyze_canary_metrics(
        self,
        stage: DeploymentStage,
        traffic_percentage: float
    ) -> Tuple[DeploymentMetrics, bool, Optional[RollbackReason]]:
        """
        Analyze recent canary metrics and determine if rollback is needed.
        
        Args:
            stage: Current deployment stage
            traffic_percentage: Current canary traffic percentage
            
        Returns:
            Tuple of (metrics, should_rollback, rollback_reason)
        """
        with self._lock:
            # Get metrics from the collection window
            cutoff_time = datetime.now() - timedelta(
                seconds=self.thresholds.metric_collection_window_seconds
            )
            recent_metrics = [
                m for m in self._metrics
                if m["timestamp"] >= cutoff_time and m["version"] == "canary"
            ]

            # Not enough samples to make a decision
            if len(recent_metrics) < self.thresholds.min_samples_for_decision:
                return self._create_insufficient_data_metrics(stage, traffic_percentage)

            # Calculate metrics
            request_count = len(recent_metrics)
            error_count = sum(1 for m in recent_metrics if not m["success"])
            error_rate = (error_count / request_count * 100) if request_count > 0 else 0.0
            
            contract_violations = sum(1 for m in recent_metrics if not m["contract_valid"])
            
            latencies = sorted([m["latency_ms"] for m in recent_metrics])
            p50_idx = int(len(latencies) * 0.50)
            p95_idx = int(len(latencies) * 0.95)
            p99_idx = int(len(latencies) * 0.99)
            
            p50_latency = latencies[p50_idx] if latencies else 0.0
            p95_latency = latencies[p95_idx] if latencies else 0.0
            p99_latency = latencies[p99_idx] if latencies else 0.0

            metrics = DeploymentMetrics(
                timestamp=datetime.now(),
                stage=stage,
                traffic_percentage=traffic_percentage,
                request_count=request_count,
                error_count=error_count,
                error_rate=error_rate,
                p50_latency_ms=p50_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                contract_violations=contract_violations
            )

            # Check rollback conditions
            should_rollback = False
            rollback_reason = None

            if contract_violations > 0:
                should_rollback = True
                rollback_reason = RollbackReason.CONTRACT_VIOLATION
                logger.error(f"Contract violation detected: {contract_violations} violations")

            elif error_rate > self.thresholds.max_error_rate:
                should_rollback = True
                rollback_reason = RollbackReason.ERROR_RATE_EXCEEDED
                logger.error(f"Error rate exceeded: {error_rate:.2f}% > {self.thresholds.max_error_rate}%")

            elif p95_latency > self.thresholds.max_p95_latency_ms:
                should_rollback = True
                rollback_reason = RollbackReason.LATENCY_EXCEEDED
                logger.error(f"P95 latency exceeded: {p95_latency:.2f}ms > {self.thresholds.max_p95_latency_ms}ms")

            return metrics, should_rollback, rollback_reason

    def _create_insufficient_data_metrics(
        self,
        stage: DeploymentStage,
        traffic_percentage: float
    ) -> Tuple[DeploymentMetrics, bool, Optional[RollbackReason]]:
        """Create metrics object when insufficient data is available"""
        metrics = DeploymentMetrics(
            timestamp=datetime.now(),
            stage=stage,
            traffic_percentage=traffic_percentage,
            request_count=0,
            error_count=0,
            error_rate=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            contract_violations=0
        )
        return metrics, False, None

    def clear_metrics(self):
        """Clear all collected metrics"""
        with self._lock:
            self._metrics.clear()


class CanaryDeploymentController:
    """
    Controls canary deployment progression with automated rollback.
    """

    def __init__(
        self,
        deployment_id: str,
        traffic_config: Optional[TrafficRoutingConfig] = None,
        rollback_thresholds: Optional[RollbackThresholds] = None
    ):
        """Initialize canary deployment controller"""
        self.deployment_id = deployment_id
        self.traffic_config = traffic_config or TrafficRoutingConfig()
        self.rollback_thresholds = rollback_thresholds or RollbackThresholds()
        
        self.router = TrafficRouter(self.traffic_config)
        self.metrics_collector = MetricsCollector(self.rollback_thresholds)
        
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.stages_completed: List[DeploymentStage] = []
        self.metrics_history: List[DeploymentMetrics] = []
        self.rollback_reason: Optional[RollbackReason] = None

    def execute_deployment(
        self,
        request_generator: Callable[[], Tuple[str, float, bool, bool]]
    ) -> DeploymentResult:
        """
        Execute full canary deployment with progressive traffic routing.
        
        Args:
            request_generator: Function that simulates request handling, returns
                             (request_id, latency_ms, success, contract_valid)
        
        Returns:
            DeploymentResult with final status and metrics
        """
        logger.info(f"Starting canary deployment: {self.deployment_id}")
        logger.info(f"Traffic config: {self.traffic_config}")
        logger.info(f"Rollback thresholds: {self.rollback_thresholds}")

        stages = [
            (DeploymentStage.CANARY_5, self.traffic_config.canary_5_hold_duration_seconds),
            (DeploymentStage.CANARY_25, self.traffic_config.canary_25_hold_duration_seconds),
            (DeploymentStage.FULL_ROLLOUT, self.traffic_config.full_rollout_hold_duration_seconds)
        ]

        for stage, hold_duration in stages:
            logger.info(f"\n{'='*80}")
            logger.info(f"STAGE: {stage.value.upper()}")
            logger.info(f"Hold duration: {hold_duration}s")
            logger.info(f"{'='*80}")

            # Set traffic routing for this stage
            self.router.set_stage(stage)
            self.stages_completed.append(stage)

            # Monitor metrics during hold duration
            success = self._monitor_stage(stage, hold_duration, request_generator)

            if not success:
                # Rollback triggered
                self.router.set_stage(DeploymentStage.ROLLBACK)
                final_stage = DeploymentStage.ROLLBACK
                break
        else:
            # All stages completed successfully
            final_stage = DeploymentStage.FULL_ROLLOUT

        self.end_time = datetime.now()
        execution_time = (self.end_time - self.start_time).total_seconds()

        logger.info(f"\n{'='*80}")
        logger.info(f"DEPLOYMENT COMPLETED: {final_stage.value}")
        logger.info(f"Execution time: {execution_time:.2f}s")
        logger.info(f"Stages completed: {[s.value for s in self.stages_completed]}")
        if self.rollback_reason:
            logger.info(f"Rollback reason: {self.rollback_reason.value}")
        logger.info(f"{'='*80}")

        return DeploymentResult(
            deployment_id=self.deployment_id,
            start_time=self.start_time,
            end_time=self.end_time,
            final_stage=final_stage,
            rollback_reason=self.rollback_reason,
            stages_completed=self.stages_completed,
            metrics_history=self.metrics_history,
            success=(final_stage == DeploymentStage.FULL_ROLLOUT)
        )

    def _monitor_stage(
        self,
        stage: DeploymentStage,
        hold_duration: int,
        request_generator: Callable[[], Tuple[str, float, bool, bool]]
    ) -> bool:
        """
        Monitor metrics during a deployment stage.
        
        Returns:
            True if stage completed successfully, False if rollback triggered
        """
        stage_start = datetime.now()
        stage_end = stage_start + timedelta(seconds=hold_duration)
        
        check_interval = 5  # Check metrics every 5 seconds
        next_check = stage_start + timedelta(seconds=check_interval)

        while datetime.now() < stage_end:
            # Simulate request processing
            request_id, latency_ms, success, contract_valid = request_generator()
            version = self.router.route_request(request_id)
            
            # Only record canary metrics
            if version == "canary":
                self.metrics_collector.record_request(
                    request_id, version, latency_ms, success, contract_valid
                )

            # Periodic metric analysis
            if datetime.now() >= next_check:
                metrics, should_rollback, rollback_reason = \
                    self.metrics_collector.analyze_canary_metrics(
                        stage, self.router.get_current_percentage()
                    )
                
                self.metrics_history.append(metrics)
                
                logger.info(
                    f"Metrics: requests={metrics.request_count}, "
                    f"errors={metrics.error_rate:.2f}%, "
                    f"p95={metrics.p95_latency_ms:.2f}ms, "
                    f"contract_violations={metrics.contract_violations}"
                )

                if should_rollback:
                    logger.error(f"ROLLBACK TRIGGERED: {rollback_reason.value}")
                    self.rollback_reason = rollback_reason
                    return False

                next_check = datetime.now() + timedelta(seconds=check_interval)

            time.sleep(0.1)  # Small sleep to avoid busy loop

        logger.info(f"✅ Stage {stage.value} completed successfully")
        return True

    def export_metrics(self, output_path: str):
        """Export deployment metrics to JSON file"""
        metrics_data = {
            "deployment_id": self.deployment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "stages_completed": [s.value for s in self.stages_completed],
            "rollback_reason": self.rollback_reason.value if self.rollback_reason else None,
            "metrics_history": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "stage": m.stage.value,
                    "traffic_percentage": m.traffic_percentage,
                    "request_count": m.request_count,
                    "error_count": m.error_count,
                    "error_rate": m.error_rate,
                    "p50_latency_ms": m.p50_latency_ms,
                    "p95_latency_ms": m.p95_latency_ms,
                    "p99_latency_ms": m.p99_latency_ms,
                    "contract_violations": m.contract_violations
                }
                for m in self.metrics_history
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics exported to {output_path}")


def create_canary_controller(
    deployment_id: str,
    canary_5_hold_seconds: int = 300,
    canary_25_hold_seconds: int = 600,
    full_rollout_hold_seconds: int = 1800,
    max_error_rate: float = 10.0,
    max_p95_latency_ms: float = 500.0
) -> CanaryDeploymentController:
    """
    Factory function to create a canary deployment controller.
    
    Args:
        deployment_id: Unique deployment identifier
        canary_5_hold_seconds: Hold duration for 5% canary stage
        canary_25_hold_seconds: Hold duration for 25% canary stage
        full_rollout_hold_seconds: Hold duration for full rollout stage
        max_error_rate: Maximum error rate before rollback (percentage)
        max_p95_latency_ms: Maximum p95 latency before rollback (milliseconds)
    
    Returns:
        Configured CanaryDeploymentController
    """
    traffic_config = TrafficRoutingConfig(
        canary_5_hold_duration_seconds=canary_5_hold_seconds,
        canary_25_hold_duration_seconds=canary_25_hold_seconds,
        full_rollout_hold_duration_seconds=full_rollout_hold_seconds
    )

    rollback_thresholds = RollbackThresholds(
        max_error_rate=max_error_rate,
        max_p95_latency_ms=max_p95_latency_ms
    )

    return CanaryDeploymentController(
        deployment_id=deployment_id,
        traffic_config=traffic_config,
        rollback_thresholds=rollback_thresholds
    )
