#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Performance Report Generator
===================================

Queries Prometheus metrics and parses structured logs to generate
comprehensive SLA compliance reports for batch processing:

- Throughput: Actual docs/hr vs 170 target
- Latency: p95 measurements vs 21.2s threshold
- Success rate: Overall and by stage
- Failure categorization
- Worker utilization statistics
- Queue depth analysis
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger(__name__)


@dataclass
class SLAThresholds:
    """SLA threshold configuration."""
    throughput_target: float = 170.0  # docs/hr
    throughput_warning: float = 200.0  # docs/hr (buffer)
    p95_latency_target: float = 21.2  # seconds
    p95_latency_warning: float = 18.0  # seconds
    success_rate_target: float = 0.95  # 95%
    error_rate_threshold: float = 0.05  # 5%
    worker_utilization_min: float = 50.0  # %
    worker_utilization_max: float = 95.0  # %
    queue_depth_warning: int = 1000
    queue_depth_critical: int = 5000


@dataclass
class ThroughputMetrics:
    """Throughput measurement and SLA compliance."""
    current_docs_per_hour: float = 0.0
    target_docs_per_hour: float = 170.0
    sla_compliant: bool = False
    compliance_percentage: float = 0.0
    trend_last_hour: float = 0.0
    
    def calculate_compliance(self):
        """Calculate SLA compliance."""
        self.compliance_percentage = (self.current_docs_per_hour / self.target_docs_per_hour) * 100.0
        self.sla_compliant = self.current_docs_per_hour >= self.target_docs_per_hour


@dataclass
class LatencyMetrics:
    """Latency measurement and SLA compliance."""
    p50_seconds: float = 0.0
    p95_seconds: float = 0.0
    p99_seconds: float = 0.0
    target_p95_seconds: float = 21.2
    sla_compliant: bool = False
    margin_seconds: float = 0.0
    
    def calculate_compliance(self):
        """Calculate SLA compliance."""
        self.margin_seconds = self.target_p95_seconds - self.p95_seconds
        self.sla_compliant = self.p95_seconds <= self.target_p95_seconds


@dataclass
class SuccessRateMetrics:
    """Success rate measurement and failure categorization."""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    success_rate: float = 0.0
    error_rate: float = 0.0
    failure_categories: Dict[str, int] = field(default_factory=dict)
    failure_percentages: Dict[str, float] = field(default_factory=dict)
    sla_compliant: bool = False
    
    def calculate_metrics(self):
        """Calculate success/error rates and categorization."""
        if self.total_documents > 0:
            self.success_rate = self.successful_documents / self.total_documents
            self.error_rate = self.failed_documents / self.total_documents
            self.sla_compliant = self.success_rate >= 0.95
            
            # Calculate failure category percentages
            for category, count in self.failure_categories.items():
                self.failure_percentages[category] = (count / self.total_documents) * 100.0


@dataclass
class WorkerMetrics:
    """Worker utilization statistics."""
    worker_id: str = ""
    utilization_percentage: float = 0.0
    busy_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    documents_processed: int = 0
    healthy: bool = True


@dataclass
class QueueMetrics:
    """Queue depth analysis."""
    current_depth: int = 0
    max_depth: int = 0
    avg_depth: float = 0.0
    growth_rate_per_minute: float = 0.0
    warning_threshold: int = 1000
    critical_threshold: int = 5000
    status: str = "healthy"  # healthy, warning, critical
    
    def calculate_status(self):
        """Determine queue health status."""
        if self.current_depth >= self.critical_threshold:
            self.status = "critical"
        elif self.current_depth >= self.warning_threshold:
            self.status = "warning"
        else:
            self.status = "healthy"


@dataclass
class BatchPerformanceReport:
    """Complete batch processing performance report."""
    report_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    report_period_hours: float = 1.0
    sla_thresholds: SLAThresholds = field(default_factory=SLAThresholds)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    success_rate: SuccessRateMetrics = field(default_factory=SuccessRateMetrics)
    workers: List[WorkerMetrics] = field(default_factory=list)
    queue: QueueMetrics = field(default_factory=QueueMetrics)
    overall_sla_compliant: bool = False
    sla_violations: List[str] = field(default_factory=list)
    
    def calculate_overall_sla(self):
        """Calculate overall SLA compliance and identify violations."""
        self.sla_violations = []
        
        if not self.throughput.sla_compliant:
            self.sla_violations.append(
                f"Throughput below target: {self.throughput.current_docs_per_hour:.1f} < {self.throughput.target_docs_per_hour} docs/hr"
            )
        
        if not self.latency.sla_compliant:
            self.sla_violations.append(
                f"P95 latency exceeds target: {self.latency.p95_seconds:.2f}s > {self.latency.target_p95_seconds}s"
            )
        
        if not self.success_rate.sla_compliant:
            self.sla_violations.append(
                f"Success rate below target: {self.success_rate.success_rate:.1%} < 95%"
            )
        
        if self.queue.status == "critical":
            self.sla_violations.append(
                f"Queue depth critical: {self.queue.current_depth} >= {self.queue.critical_threshold}"
            )
        
        self.overall_sla_compliant = len(self.sla_violations) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_timestamp": self.report_timestamp,
            "report_period_hours": self.report_period_hours,
            "sla_thresholds": asdict(self.sla_thresholds),
            "throughput": asdict(self.throughput),
            "latency": asdict(self.latency),
            "success_rate": asdict(self.success_rate),
            "workers": [asdict(w) for w in self.workers],
            "queue": asdict(self.queue),
            "overall_sla_compliant": self.overall_sla_compliant,
            "sla_violations": self.sla_violations
        }
    
    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(self.to_dict(), indent=2)


class PrometheusClient:
    """Client for querying Prometheus metrics."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url.rstrip("/")
    
    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute instant query against Prometheus."""
        try:
            params = urlencode({"query": query})
            url = f"{self.prometheus_url}/api/v1/query?{params}"
            
            with urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                if data["status"] == "success":
                    return data["data"]
                else:
                    logger.error(f"Prometheus query failed: {data}")
                    return None
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            return None
    
    def query_range(self, query: str, start: datetime, end: datetime, step: str = "1m") -> Optional[Dict[str, Any]]:
        """Execute range query against Prometheus."""
        try:
            params = urlencode({
                "query": query,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
                "step": step
            })
            url = f"{self.prometheus_url}/api/v1/query_range?{params}"
            
            with urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
                if data["status"] == "success":
                    return data["data"]
                else:
                    logger.error(f"Prometheus range query failed: {data}")
                    return None
        except Exception as e:
            logger.error(f"Prometheus range query error: {e}")
            return None
    
    def get_metric_value(self, query: str) -> Optional[float]:
        """Get single metric value."""
        result = self.query(query)
        if result and result["result"]:
            return float(result["result"][0]["value"][1])
        return None


class LogParser:
    """Parser for structured JSON logs."""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
    
    def parse_logs(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Parse structured logs and extract events."""
        events = []
        
        if not self.log_file_path.exists():
            logger.warning(f"Log file not found: {self.log_file_path}")
            return events
        
        with open(self.log_file_path, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    
                    # Filter by timestamp if provided
                    if since:
                        event_time = datetime.fromisoformat(event.get("timestamp", ""))
                        if event_time < since:
                            continue
                    
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        return events
    
    def aggregate_document_processing(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate document processing metrics from logs."""
        total_docs = 0
        success_docs = 0
        error_docs = 0
        error_categories = defaultdict(int)
        latencies = []
        trace_ids = set()
        
        for event in events:
            if event.get("event") == "document_processing_completed":
                trace_id = event.get("trace_id")
                if trace_id:
                    trace_ids.add(trace_id)
                
                total_docs += 1
                status = event.get("status")
                if status == "success":
                    success_docs += 1
                elif status == "error":
                    error_docs += 1
                    category = event.get("error_category", "Unknown")
                    error_categories[category] += 1
                
                latency = event.get("latency_seconds")
                if latency:
                    latencies.append(latency)
        
        return {
            "total_documents": total_docs,
            "successful_documents": success_docs,
            "failed_documents": error_docs,
            "error_categories": dict(error_categories),
            "latencies": latencies,
            "unique_trace_ids": len(trace_ids)
        }


class BatchPerformanceReportGenerator:
    """Main report generator."""
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        log_file_path: str = "/var/log/batch_processor.log",
        sla_thresholds: Optional[SLAThresholds] = None
    ):
        self.prometheus = PrometheusClient(prometheus_url)
        self.log_parser = LogParser(log_file_path)
        self.sla_thresholds = sla_thresholds or SLAThresholds()
    
    def generate_report(
        self,
        period_hours: float = 1.0,
        use_prometheus: bool = True,
        use_logs: bool = True
    ) -> BatchPerformanceReport:
        """Generate comprehensive performance report."""
        report = BatchPerformanceReport(
            report_period_hours=period_hours,
            sla_thresholds=self.sla_thresholds
        )
        
        # Query Prometheus metrics
        if use_prometheus:
            self._collect_prometheus_metrics(report)
        
        # Parse structured logs
        if use_logs:
            since = datetime.utcnow() - timedelta(hours=period_hours)
            self._collect_log_metrics(report, since)
        
        # Calculate SLA compliance
        report.throughput.calculate_compliance()
        report.latency.calculate_compliance()
        report.success_rate.calculate_metrics()
        report.queue.calculate_status()
        report.calculate_overall_sla()
        
        return report
    
    def _collect_prometheus_metrics(self, report: BatchPerformanceReport):
        """Collect metrics from Prometheus."""
        # Throughput
        throughput = self.prometheus.get_metric_value("batch_throughput_per_hour")
        if throughput is not None:
            report.throughput.current_docs_per_hour = throughput
        
        # Latency
        p50 = self.prometheus.get_metric_value('batch_document_processing_latency_seconds{quantile="0.5"}')
        p95 = self.prometheus.get_metric_value('batch_document_processing_latency_seconds{quantile="0.95"}')
        p99 = self.prometheus.get_metric_value('batch_document_processing_latency_seconds{quantile="0.99"}')
        
        if p50 is not None:
            report.latency.p50_seconds = p50
        if p95 is not None:
            report.latency.p95_seconds = p95
        if p99 is not None:
            report.latency.p99_seconds = p99
        
        # Success/Error counts
        success_count = self.prometheus.get_metric_value('batch_documents_processed_total{status="success"}')
        error_count = self.prometheus.get_metric_value('batch_documents_processed_total{status="error"}')
        
        if success_count is not None:
            report.success_rate.successful_documents = int(success_count)
        if error_count is not None:
            report.success_rate.failed_documents = int(error_count)
        
        report.success_rate.total_documents = report.success_rate.successful_documents + report.success_rate.failed_documents
        
        # Worker utilization
        worker_result = self.prometheus.query("worker_utilization_percentage")
        if worker_result and worker_result["result"]:
            for result in worker_result["result"]:
                worker_id = result["metric"].get("worker_id", "unknown")
                utilization = float(result["value"][1])
                
                worker_metrics = WorkerMetrics(
                    worker_id=worker_id,
                    utilization_percentage=utilization,
                    healthy=utilization >= self.sla_thresholds.worker_utilization_min
                )
                report.workers.append(worker_metrics)
        
        # Queue depth
        queue_depth = self.prometheus.get_metric_value("queue_depth")
        if queue_depth is not None:
            report.queue.current_depth = int(queue_depth)
    
    def _collect_log_metrics(self, report: BatchPerformanceReport, since: datetime):
        """Collect metrics from structured logs."""
        events = self.log_parser.parse_logs(since)
        aggregated = self.log_parser.aggregate_document_processing(events)
        
        # Update success rate metrics
        if aggregated["total_documents"] > 0:
            report.success_rate.total_documents = aggregated["total_documents"]
            report.success_rate.successful_documents = aggregated["successful_documents"]
            report.success_rate.failed_documents = aggregated["failed_documents"]
            report.success_rate.failure_categories = aggregated["error_categories"]
        
        # Update latency metrics from logs if Prometheus unavailable
        if aggregated["latencies"] and report.latency.p95_seconds == 0:
            sorted_latencies = sorted(aggregated["latencies"])
            count = len(sorted_latencies)
            
            report.latency.p50_seconds = sorted_latencies[int(count * 0.50)]
            report.latency.p95_seconds = sorted_latencies[int(count * 0.95)]
            report.latency.p99_seconds = sorted_latencies[int(count * 0.99)]
    
    def export_report(self, report: BatchPerformanceReport, output_path: str):
        """Export report to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(report.to_json())
        
        logger.info(f"Report exported to {output_path}")
    
    def print_summary(self, report: BatchPerformanceReport):
        """Print human-readable report summary."""
        print("=" * 80)
        print("BATCH PROCESSING PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Generated: {report.report_timestamp}")
        print(f"Period: {report.report_period_hours:.1f} hours")
        print()
        
        # SLA Status
        status_emoji = "✅" if report.overall_sla_compliant else "❌"
        print(f"{status_emoji} Overall SLA Status: {'COMPLIANT' if report.overall_sla_compliant else 'VIOLATION'}")
        print()
        
        # Throughput
        print("📊 THROUGHPUT")
        print(f"  Current: {report.throughput.current_docs_per_hour:.1f} docs/hr")
        print(f"  Target:  {report.throughput.target_docs_per_hour:.1f} docs/hr")
        print(f"  Status:  {'✅ Compliant' if report.throughput.sla_compliant else '❌ Below target'}")
        print(f"  Compliance: {report.throughput.compliance_percentage:.1f}%")
        print()
        
        # Latency
        print("⏱️  LATENCY")
        print(f"  P50: {report.latency.p50_seconds:.2f}s")
        print(f"  P95: {report.latency.p95_seconds:.2f}s (target: {report.latency.target_p95_seconds}s)")
        print(f"  P99: {report.latency.p99_seconds:.2f}s")
        print(f"  Status: {'✅ Compliant' if report.latency.sla_compliant else '❌ Exceeds target'}")
        print(f"  Margin: {report.latency.margin_seconds:+.2f}s")
        print()
        
        # Success Rate
        print("✔️  SUCCESS RATE")
        print(f"  Total Documents: {report.success_rate.total_documents}")
        print(f"  Successful: {report.success_rate.successful_documents} ({report.success_rate.success_rate:.1%})")
        print(f"  Failed: {report.success_rate.failed_documents} ({report.success_rate.error_rate:.1%})")
        print(f"  Status: {'✅ Compliant' if report.success_rate.sla_compliant else '❌ Below 95%'}")
        
        if report.success_rate.failure_categories:
            print("  Failure Categories:")
            for category, count in sorted(report.success_rate.failure_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = report.success_rate.failure_percentages.get(category, 0)
                print(f"    - {category}: {count} ({percentage:.2f}%)")
        print()
        
        # Workers
        print("👷 WORKERS")
        for worker in report.workers:
            status = "✅" if worker.healthy else "⚠️"
            print(f"  {status} {worker.worker_id}: {worker.utilization_percentage:.1f}% utilization")
        print()
        
        # Queue
        print("📥 QUEUE")
        print(f"  Current Depth: {report.queue.current_depth}")
        print(f"  Status: {report.queue.status.upper()}")
        print()
        
        # Violations
        if report.sla_violations:
            print("⚠️  SLA VIOLATIONS")
            for violation in report.sla_violations:
                print(f"  - {violation}")
            print()
        
        print("=" * 80)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate batch performance report")
    parser.add_argument("--prometheus-url", default="http://localhost:9090", help="Prometheus URL")
    parser.add_argument("--log-file", default="/var/log/batch_processor.log", help="Log file path")
    parser.add_argument("--period", type=float, default=1.0, help="Report period in hours")
    parser.add_argument("--output", default="batch_performance_report.json", help="Output JSON file")
    parser.add_argument("--no-prometheus", action="store_true", help="Skip Prometheus queries")
    parser.add_argument("--no-logs", action="store_true", help="Skip log parsing")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate report
    generator = BatchPerformanceReportGenerator(
        prometheus_url=args.prometheus_url,
        log_file_path=args.log_file
    )
    
    report = generator.generate_report(
        period_hours=args.period,
        use_prometheus=not args.no_prometheus,
        use_logs=not args.no_logs
    )
    
    # Print summary
    generator.print_summary(report)
    
    # Export to JSON
    generator.export_report(report, args.output)
    
    # Exit with error code if SLA violated
    return 0 if report.overall_sla_compliant else 1


if __name__ == "__main__":
    exit(main())
