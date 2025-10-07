"""
Batch Performance Report Generator

Queries Prometheus HTTP API for batch processing metrics over a configurable 
time window, calculates SLA compliance, and generates comprehensive reports 
with trace IDs from structured logs.

SLA Targets:
- Throughput: 170 documents/hour
- P95 Latency: 21.2 seconds
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class PrometheusQueryError(Exception):
    """Raised when Prometheus query fails"""
    pass


class BatchPerformanceReportGenerator:
    """
    Generates comprehensive batch performance reports by querying Prometheus
    and correlating with structured logs for trace IDs.
    """
    
    SLA_THROUGHPUT_TARGET = 170.0
    SLA_P95_LATENCY_TARGET = 21.2
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        artifacts_dir: str = "artifacts",
        log_file: Optional[str] = None
    ):
        """
        Initialize report generator.
        
        Args:
            prometheus_url: Base URL for Prometheus API
            artifacts_dir: Directory to write report artifacts
            log_file: Optional path to structured log file for trace ID extraction
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.log_file = Path(log_file) if log_file else None
    
    def query_prometheus(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "15s"
    ) -> Dict:
        """
        Query Prometheus HTTP API for range data.
        
        Args:
            query: PromQL query expression
            start_time: Query start timestamp
            end_time: Query end timestamp
            step: Query resolution step
            
        Returns:
            Query result dictionary
            
        Raises:
            PrometheusQueryError: If query fails
        """
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "success":
                raise PrometheusQueryError(
                    f"Query failed: {data.get('error', 'Unknown error')}"
                )
            
            return data["data"]["result"]
        except requests.RequestException as e:
            raise PrometheusQueryError(f"Prometheus request failed: {e}")
    
    def query_prometheus_instant(self, query: str, timestamp: datetime) -> Dict:
        """
        Query Prometheus HTTP API for instant data.
        
        Args:
            query: PromQL query expression
            timestamp: Query timestamp
            
        Returns:
            Query result dictionary
            
        Raises:
            PrometheusQueryError: If query fails
        """
        url = f"{self.prometheus_url}/api/v1/query"
        params = {
            "query": query,
            "time": timestamp.timestamp()
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "success":
                raise PrometheusQueryError(
                    f"Query failed: {data.get('error', 'Unknown error')}"
                )
            
            return data["data"]["result"]
        except requests.RequestException as e:
            raise PrometheusQueryError(f"Prometheus request failed: {e}")
    
    def calculate_sla_compliance(
        self,
        throughput_data: List[Tuple[datetime, float]],
        latency_data: List[Tuple[datetime, float]]
    ) -> Dict[str, float]:
        """
        Calculate SLA compliance percentage for throughput and latency.
        
        Args:
            throughput_data: List of (timestamp, throughput) tuples
            latency_data: List of (timestamp, p95_latency) tuples
            
        Returns:
            Dictionary with compliance percentages
        """
        total_throughput_samples = len(throughput_data)
        total_latency_samples = len(latency_data)
        
        if total_throughput_samples == 0 or total_latency_samples == 0:
            return {
                "throughput_compliance_pct": 0.0,
                "latency_compliance_pct": 0.0,
                "overall_compliance_pct": 0.0
            }
        
        throughput_compliant = sum(
            1 for _, value in throughput_data 
            if value >= self.SLA_THROUGHPUT_TARGET
        )
        latency_compliant = sum(
            1 for _, value in latency_data 
            if value <= self.SLA_P95_LATENCY_TARGET
        )
        
        throughput_compliance_pct = (
            throughput_compliant / total_throughput_samples * 100
        )
        latency_compliance_pct = (
            latency_compliant / total_latency_samples * 100
        )
        overall_compliance_pct = (
            throughput_compliance_pct + latency_compliance_pct
        ) / 2
        
        return {
            "throughput_compliance_pct": round(throughput_compliance_pct, 2),
            "latency_compliance_pct": round(latency_compliance_pct, 2),
            "overall_compliance_pct": round(overall_compliance_pct, 2),
            "throughput_samples": total_throughput_samples,
            "latency_samples": total_latency_samples
        }
    
    def extract_trace_ids_from_logs(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, str]]:
        """
        Extract per-document trace IDs from structured logs.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            
        Returns:
            List of dictionaries with trace_id, timestamp, status
        """
        if not self.log_file or not self.log_file.exists():
            logger.warning(
                f"Log file not found: {self.log_file}. Trace IDs not available."
            )
            return []
        
        trace_entries = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        log_timestamp = datetime.fromisoformat(
                            log_entry.get("timestamp", "")
                        )
                        
                        if start_time <= log_timestamp <= end_time:
                            trace_id = log_entry.get("trace_id")
                            if trace_id:
                                trace_entries.append({
                                    "trace_id": trace_id,
                                    "timestamp": log_entry["timestamp"],
                                    "status": log_entry.get("status", "unknown"),
                                    "document_id": log_entry.get("document_id")
                                })
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
        except IOError as e:
            logger.error(f"Failed to read log file: {e}")
        
        return trace_entries
    
    def detect_performance_degradation_incidents(
        self,
        throughput_data: List[Tuple[datetime, float]],
        latency_data: List[Tuple[datetime, float]]
    ) -> List[Dict]:
        """
        Detect performance degradation incidents from time series data.
        
        Args:
            throughput_data: List of (timestamp, throughput) tuples
            latency_data: List of (timestamp, p95_latency) tuples
            
        Returns:
            List of incident dictionaries with start, end, type, severity
        """
        incidents = []
        
        # Throughput degradation: below SLA for 10+ minutes
        throughput_incident_start = None
        for timestamp, value in throughput_data:
            if value < self.SLA_THROUGHPUT_TARGET:
                if throughput_incident_start is None:
                    throughput_incident_start = timestamp
            else:
                if throughput_incident_start is not None:
                    duration = (timestamp - throughput_incident_start).total_seconds()
                    if duration >= 600:  # 10 minutes
                        incidents.append({
                            "type": "throughput_degradation",
                            "severity": "critical",
                            "start": throughput_incident_start.isoformat(),
                            "end": timestamp.isoformat(),
                            "duration_seconds": duration,
                            "min_throughput": min(
                                v for t, v in throughput_data 
                                if throughput_incident_start <= t < timestamp
                            )
                        })
                    throughput_incident_start = None
        
        # Latency degradation: above SLA threshold
        latency_incident_start = None
        for timestamp, value in latency_data:
            if value > self.SLA_P95_LATENCY_TARGET:
                if latency_incident_start is None:
                    latency_incident_start = timestamp
            else:
                if latency_incident_start is not None:
                    duration = (timestamp - latency_incident_start).total_seconds()
                    if duration >= 180:  # 3 minutes
                        incidents.append({
                            "type": "latency_degradation",
                            "severity": "critical",
                            "start": latency_incident_start.isoformat(),
                            "end": timestamp.isoformat(),
                            "duration_seconds": duration,
                            "max_latency": max(
                                v for t, v in latency_data 
                                if latency_incident_start <= t < timestamp
                            )
                        })
                    latency_incident_start = None
        
        return incidents
    
    def generate_report(
        self,
        time_window_hours: int = 1,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """
        Generate comprehensive batch performance report.
        
        Args:
            time_window_hours: Hours to look back from end_time
            end_time: End of report window (default: now)
            
        Returns:
            Complete report dictionary
            
        Raises:
            PrometheusQueryError: If queries fail
        """
        if end_time is None:
            end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        logger.info(
            f"Generating batch performance report for "
            f"{start_time.isoformat()} to {end_time.isoformat()}"
        )
        
        # Query Prometheus for metrics
        throughput_result = self.query_prometheus(
            "batch_throughput_per_hour",
            start_time,
            end_time
        )
        
        latency_result = self.query_prometheus(
            'histogram_quantile(0.95, rate(batch_document_processing_latency_seconds_bucket[5m]))',
            start_time,
            end_time
        )
        
        queue_depth_result = self.query_prometheus(
            "queue_depth",
            start_time,
            end_time
        )
        
        worker_utilization_result = self.query_prometheus(
            "worker_utilization_percentage",
            start_time,
            end_time
        )
        
        # Query for total documents processed
        docs_processed_result = self.query_prometheus_instant(
            f'sum(increase(documents_processed_total[{time_window_hours}h]))',
            end_time
        )
        
        # Query for error rate
        error_rate_result = self.query_prometheus(
            'rate(batch_documents_processed_total{status="error"}[5m]) / rate(batch_documents_processed_total[5m])',
            start_time,
            end_time
        )
        
        # Parse time series data
        throughput_data = [
            (datetime.fromtimestamp(float(ts)), float(val))
            for result in throughput_result
            for ts, val in result.get("values", [])
        ]
        
        latency_data = [
            (datetime.fromtimestamp(float(ts)), float(val))
            for result in latency_result
            for ts, val in result.get("values", [])
        ]
        
        queue_depth_data = [
            (datetime.fromtimestamp(float(ts)), float(val))
            for result in queue_depth_result
            for ts, val in result.get("values", [])
        ]
        
        worker_util_data = [
            (datetime.fromtimestamp(float(ts)), float(val))
            for result in worker_utilization_result
            for ts, val in result.get("values", [])
        ]
        
        error_rate_data = [
            (datetime.fromtimestamp(float(ts)), float(val))
            for result in error_rate_result
            for ts, val in result.get("values", [])
        ]
        
        # Calculate statistics
        total_docs_processed = 0
        if docs_processed_result:
            total_docs_processed = float(
                docs_processed_result[0].get("value", [0, "0"])[1]
            )
        
        # Calculate SLA compliance
        sla_compliance = self.calculate_sla_compliance(
            throughput_data,
            latency_data
        )
        
        # Calculate batch success rate
        avg_error_rate = (
            sum(val for _, val in error_rate_data) / len(error_rate_data)
            if error_rate_data else 0.0
        )
        batch_success_rate = (1.0 - avg_error_rate) * 100
        
        # Detect performance degradation incidents
        incidents = self.detect_performance_degradation_incidents(
            throughput_data,
            latency_data
        )
        
        # Extract trace IDs from logs
        trace_ids = self.extract_trace_ids_from_logs(start_time, end_time)
        
        # Calculate summary statistics
        avg_throughput = (
            sum(val for _, val in throughput_data) / len(throughput_data)
            if throughput_data else 0.0
        )
        avg_p95_latency = (
            sum(val for _, val in latency_data) / len(latency_data)
            if latency_data else 0.0
        )
        max_queue_depth = (
            max(val for _, val in queue_depth_data)
            if queue_depth_data else 0.0
        )
        avg_worker_utilization = (
            sum(val for _, val in worker_util_data) / len(worker_util_data)
            if worker_util_data else 0.0
        )
        
        # Assemble complete report
        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_window": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": time_window_hours
                },
                "prometheus_url": self.prometheus_url
            },
            "sla_targets": {
                "throughput_docs_per_hour": self.SLA_THROUGHPUT_TARGET,
                "p95_latency_seconds": self.SLA_P95_LATENCY_TARGET
            },
            "sla_compliance": sla_compliance,
            "summary_statistics": {
                "total_documents_processed": int(total_docs_processed),
                "avg_throughput_docs_per_hour": round(avg_throughput, 2),
                "avg_p95_latency_seconds": round(avg_p95_latency, 2),
                "max_queue_depth": int(max_queue_depth),
                "avg_worker_utilization_pct": round(avg_worker_utilization, 2),
                "batch_success_rate_pct": round(batch_success_rate, 2),
                "avg_error_rate_pct": round(avg_error_rate * 100, 2)
            },
            "performance_degradation_incidents": {
                "count": len(incidents),
                "incidents": incidents
            },
            "trace_correlation": {
                "total_traces": len(trace_ids),
                "sample_traces": trace_ids[:100]
            },
            "time_series_data": {
                "throughput": [
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in throughput_data
                ],
                "p95_latency": [
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in latency_data
                ],
                "queue_depth": [
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in queue_depth_data
                ],
                "worker_utilization": [
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in worker_util_data
                ],
                "error_rate": [
                    {"timestamp": ts.isoformat(), "value": val}
                    for ts, val in error_rate_data
                ]
            }
        }
        
        return report
    
    def write_report(
        self,
        report: Dict,
        filename_prefix: str = "batch_performance_report"
    ) -> Path:
        """
        Write report to artifacts directory with timestamp.
        
        Args:
            report: Report dictionary from generate_report()
            filename_prefix: Prefix for report filename
            
        Returns:
            Path to written report file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = self.artifacts_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Batch performance report written to {filepath}")
        return filepath
    
    def generate_and_write_report(
        self,
        time_window_hours: int = 1,
        end_time: Optional[datetime] = None,
        filename_prefix: str = "batch_performance_report"
    ) -> Tuple[Dict, Path]:
        """
        Generate report and write to artifacts directory.
        
        Args:
            time_window_hours: Hours to look back from end_time
            end_time: End of report window (default: now)
            filename_prefix: Prefix for report filename
            
        Returns:
            Tuple of (report_dict, filepath)
        """
        report = self.generate_report(time_window_hours, end_time)
        filepath = self.write_report(report, filename_prefix)
        return report, filepath


def create_batch_performance_report(
    prometheus_url: str = "http://localhost:9090",
    time_window_hours: int = 1,
    artifacts_dir: str = "artifacts",
    log_file: Optional[str] = None
) -> Tuple[Dict, Path]:
    """
    Convenience function to create batch performance report.
    
    Args:
        prometheus_url: Base URL for Prometheus API
        time_window_hours: Hours to look back
        artifacts_dir: Directory to write report artifacts
        log_file: Optional path to structured log file for trace ID extraction
        
    Returns:
        Tuple of (report_dict, filepath)
        
    Example:
        >>> report, path = create_batch_performance_report(
        ...     prometheus_url="http://prometheus:9090",
        ...     time_window_hours=6,
        ...     log_file="/var/log/batch_processing.log"
        ... )
        >>> print(f"SLA compliance: {report['sla_compliance']['overall_compliance_pct']}%")
        >>> print(f"Report saved to: {path}")
    """
    generator = BatchPerformanceReportGenerator(
        prometheus_url=prometheus_url,
        artifacts_dir=artifacts_dir,
        log_file=log_file
    )
    return generator.generate_and_write_report(time_window_hours)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        report, filepath = create_batch_performance_report(
            prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
            time_window_hours=int(os.getenv("TIME_WINDOW_HOURS", "1")),
            log_file=os.getenv("LOG_FILE")
        )
        
        print(f"\n{'='*80}")
        print("BATCH PERFORMANCE REPORT SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Report Period: {report['metadata']['time_window']['start']} to "
              f"{report['metadata']['time_window']['end']}")
        print(f"\nSLA Compliance:")
        print(f"  Throughput: {report['sla_compliance']['throughput_compliance_pct']}%")
        print(f"  Latency: {report['sla_compliance']['latency_compliance_pct']}%")
        print(f"  Overall: {report['sla_compliance']['overall_compliance_pct']}%")
        
        print(f"\nSummary Statistics:")
        stats = report['summary_statistics']
        print(f"  Total Documents: {stats['total_documents_processed']}")
        print(f"  Avg Throughput: {stats['avg_throughput_docs_per_hour']} docs/hr")
        print(f"  Avg p95 Latency: {stats['avg_p95_latency_seconds']}s")
        print(f"  Max Queue Depth: {stats['max_queue_depth']}")
        print(f"  Avg Worker Util: {stats['avg_worker_utilization_pct']}%")
        print(f"  Success Rate: {stats['batch_success_rate_pct']}%")
        
        incident_count = report['performance_degradation_incidents']['count']
        print(f"\nPerformance Incidents: {incident_count}")
        if incident_count > 0:
            for incident in report['performance_degradation_incidents']['incidents']:
                print(f"  - {incident['type']} ({incident['severity']}): "
                      f"{incident['duration_seconds']}s")
        
        print(f"\nReport saved to: {filepath}")
        print(f"\n{'='*80}\n")
        
    except PrometheusQueryError as e:
        logger.error(f"Failed to generate report: {e}")
        exit(1)
