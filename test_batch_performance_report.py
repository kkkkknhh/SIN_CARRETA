"""
Tests for Batch Performance Report Generator
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from batch_performance_report import (
    BatchPerformanceReportGenerator,
    PrometheusQueryError,
    create_batch_performance_report
)


class TestBatchPerformanceReportGenerator:
    """Test suite for BatchPerformanceReportGenerator"""
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file with sample structured logs"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            base_time = datetime.utcnow()
            for i in range(10):
                log_entry = {
                    "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                    "trace_id": f"trace-{i:04d}",
                    "document_id": f"doc-{i:04d}",
                    "status": "success" if i % 5 != 0 else "error"
                }
                f.write(json.dumps(log_entry) + '\n')
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink()
    
    @pytest.fixture
    def generator(self, temp_artifacts_dir):
        """Create generator instance"""
        return BatchPerformanceReportGenerator(
            prometheus_url="http://localhost:9090",
            artifacts_dir=temp_artifacts_dir
        )
    
    @pytest.fixture
    def mock_prometheus_response(self):
        """Mock Prometheus API response"""
        def _mock_response(metric_name):
            base_time = datetime.utcnow()
            values = []
            
            if metric_name == "throughput":
                values = [[base_time.timestamp() + i * 15, "180.0"] for i in range(10)]
            elif metric_name == "latency":
                values = [[base_time.timestamp() + i * 15, "18.5"] for i in range(10)]
            elif metric_name == "queue":
                values = [[base_time.timestamp() + i * 15, "50.0"] for i in range(10)]
            elif metric_name == "worker":
                values = [[base_time.timestamp() + i * 15, "75.0"] for i in range(10)]
            elif metric_name == "error":
                values = [[base_time.timestamp() + i * 15, "0.02"] for i in range(10)]
            
            return {
                "status": "success",
                "data": {
                    "result": [
                        {
                            "metric": {},
                            "values": values
                        }
                    ]
                }
            }
        
        return _mock_response
    
    def test_initialization(self, temp_artifacts_dir):
        """Test generator initialization"""
        generator = BatchPerformanceReportGenerator(
            prometheus_url="http://prometheus:9090",
            artifacts_dir=temp_artifacts_dir,
            log_file="/tmp/test.log"
        )
        
        assert generator.prometheus_url == "http://prometheus:9090"
        assert generator.artifacts_dir == Path(temp_artifacts_dir)
        assert generator.log_file == Path("/tmp/test.log")
        assert generator.SLA_THROUGHPUT_TARGET == 170.0
        assert generator.SLA_P95_LATENCY_TARGET == 21.2
    
    @patch('batch_performance_report.requests.get')
    def test_query_prometheus_success(self, mock_get, generator):
        """Test successful Prometheus query"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {},
                        "values": [[1234567890, "100.0"]]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        result = generator.query_prometheus(
            "batch_throughput_per_hour",
            start_time,
            end_time
        )
        
        assert len(result) == 1
        assert result[0]["values"][0][1] == "100.0"
        mock_get.assert_called_once()
    
    @patch('batch_performance_report.requests.get')
    def test_query_prometheus_failure(self, mock_get, generator):
        """Test Prometheus query failure"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "Query timeout"
        }
        mock_get.return_value = mock_response
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        with pytest.raises(PrometheusQueryError, match="Query failed"):
            generator.query_prometheus(
                "invalid_metric",
                start_time,
                end_time
            )
    
    @patch('batch_performance_report.requests.get')
    def test_query_prometheus_instant(self, mock_get, generator):
        """Test instant Prometheus query"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {},
                        "value": [1234567890, "150.0"]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        timestamp = datetime.utcnow()
        result = generator.query_prometheus_instant(
            "documents_processed_total",
            timestamp
        )
        
        assert len(result) == 1
        assert result[0]["value"][1] == "150.0"
    
    def test_calculate_sla_compliance_all_compliant(self, generator):
        """Test SLA compliance calculation with all compliant data"""
        base_time = datetime.utcnow()
        throughput_data = [
            (base_time + timedelta(minutes=i), 180.0)
            for i in range(10)
        ]
        latency_data = [
            (base_time + timedelta(minutes=i), 18.0)
            for i in range(10)
        ]
        
        compliance = generator.calculate_sla_compliance(
            throughput_data,
            latency_data
        )
        
        assert compliance["throughput_compliance_pct"] == 100.0
        assert compliance["latency_compliance_pct"] == 100.0
        assert compliance["overall_compliance_pct"] == 100.0
    
    def test_calculate_sla_compliance_partial(self, generator):
        """Test SLA compliance calculation with partial compliance"""
        base_time = datetime.utcnow()
        throughput_data = [
            (base_time + timedelta(minutes=i), 180.0 if i < 5 else 150.0)
            for i in range(10)
        ]
        latency_data = [
            (base_time + timedelta(minutes=i), 18.0 if i < 7 else 25.0)
            for i in range(10)
        ]
        
        compliance = generator.calculate_sla_compliance(
            throughput_data,
            latency_data
        )
        
        assert compliance["throughput_compliance_pct"] == 50.0
        assert compliance["latency_compliance_pct"] == 70.0
        assert compliance["overall_compliance_pct"] == 60.0
    
    def test_calculate_sla_compliance_empty_data(self, generator):
        """Test SLA compliance with empty data"""
        compliance = generator.calculate_sla_compliance([], [])
        
        assert compliance["throughput_compliance_pct"] == 0.0
        assert compliance["latency_compliance_pct"] == 0.0
        assert compliance["overall_compliance_pct"] == 0.0
    
    def test_extract_trace_ids_from_logs(self, temp_log_file, temp_artifacts_dir):
        """Test trace ID extraction from structured logs"""
        generator = BatchPerformanceReportGenerator(
            artifacts_dir=temp_artifacts_dir,
            log_file=temp_log_file
        )
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow() + timedelta(hours=1)
        
        trace_ids = generator.extract_trace_ids_from_logs(start_time, end_time)
        
        assert len(trace_ids) == 10
        assert trace_ids[0]["trace_id"] == "trace-0000"
        assert trace_ids[0]["document_id"] == "doc-0000"
        assert trace_ids[0]["status"] in ["success", "error"]
    
    def test_extract_trace_ids_missing_log_file(self, generator):
        """Test trace ID extraction with missing log file"""
        generator.log_file = Path("/nonexistent/log.file")
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        trace_ids = generator.extract_trace_ids_from_logs(start_time, end_time)
        
        assert trace_ids == []
    
    def test_detect_performance_degradation_throughput(self, generator):
        """Test detection of throughput degradation incidents"""
        base_time = datetime.utcnow()
        
        throughput_data = [
            (base_time + timedelta(minutes=i), 150.0 if 5 <= i < 20 else 180.0)
            for i in range(30)
        ]
        latency_data = [
            (base_time + timedelta(minutes=i), 18.0)
            for i in range(30)
        ]
        
        incidents = generator.detect_performance_degradation_incidents(
            throughput_data,
            latency_data
        )
        
        throughput_incidents = [
            inc for inc in incidents if inc["type"] == "throughput_degradation"
        ]
        assert len(throughput_incidents) >= 1
        assert throughput_incidents[0]["severity"] == "critical"
        assert throughput_incidents[0]["duration_seconds"] >= 600
    
    def test_detect_performance_degradation_latency(self, generator):
        """Test detection of latency degradation incidents"""
        base_time = datetime.utcnow()
        
        throughput_data = [
            (base_time + timedelta(minutes=i), 180.0)
            for i in range(30)
        ]
        latency_data = [
            (base_time + timedelta(minutes=i), 25.0 if 5 <= i < 10 else 18.0)
            for i in range(30)
        ]
        
        incidents = generator.detect_performance_degradation_incidents(
            throughput_data,
            latency_data
        )
        
        latency_incidents = [
            inc for inc in incidents if inc["type"] == "latency_degradation"
        ]
        assert len(latency_incidents) >= 1
        assert latency_incidents[0]["severity"] == "critical"
        assert latency_incidents[0]["duration_seconds"] >= 180
    
    def test_detect_no_incidents(self, generator):
        """Test no incidents detected with compliant data"""
        base_time = datetime.utcnow()
        
        throughput_data = [
            (base_time + timedelta(minutes=i), 180.0)
            for i in range(30)
        ]
        latency_data = [
            (base_time + timedelta(minutes=i), 18.0)
            for i in range(30)
        ]
        
        incidents = generator.detect_performance_degradation_incidents(
            throughput_data,
            latency_data
        )
        
        assert len(incidents) == 0
    
    @patch.object(BatchPerformanceReportGenerator, 'query_prometheus')
    @patch.object(BatchPerformanceReportGenerator, 'query_prometheus_instant')
    def test_generate_report(
        self,
        mock_instant,
        mock_query,
        generator,
        mock_prometheus_response
    ):
        """Test complete report generation"""
        base_time = datetime.utcnow()
        
        def side_effect_query(query, start, end, step="15s"):
            if "throughput" in query:
                return mock_prometheus_response("throughput")["data"]["result"]
            elif "latency" in query:
                return mock_prometheus_response("latency")["data"]["result"]
            elif "queue" in query:
                return mock_prometheus_response("queue")["data"]["result"]
            elif "worker" in query:
                return mock_prometheus_response("worker")["data"]["result"]
            elif "error" in query:
                return mock_prometheus_response("error")["data"]["result"]
            return []
        
        mock_query.side_effect = side_effect_query
        mock_instant.return_value = [{"value": [base_time.timestamp(), "150"]}]
        
        report = generator.generate_report(time_window_hours=1)
        
        assert "metadata" in report
        assert "sla_targets" in report
        assert "sla_compliance" in report
        assert "summary_statistics" in report
        assert "performance_degradation_incidents" in report
        assert "trace_correlation" in report
        assert "time_series_data" in report
        
        assert report["sla_targets"]["throughput_docs_per_hour"] == 170.0
        assert report["sla_targets"]["p95_latency_seconds"] == 21.2
        assert report["sla_compliance"]["overall_compliance_pct"] == 100.0
    
    @patch.object(BatchPerformanceReportGenerator, 'generate_report')
    def test_write_report(self, mock_generate, generator):
        """Test report writing to artifacts directory"""
        mock_report = {
            "metadata": {"generated_at": datetime.utcnow().isoformat()},
            "summary": "test report"
        }
        
        filepath = generator.write_report(mock_report, "test_report")
        
        assert filepath.exists()
        assert filepath.parent == generator.artifacts_dir
        assert "test_report_" in filepath.name
        assert filepath.suffix == ".json"
        
        with open(filepath, 'r') as f:
            loaded_report = json.load(f)
        
        assert loaded_report == mock_report
    
    @patch.object(BatchPerformanceReportGenerator, 'query_prometheus')
    @patch.object(BatchPerformanceReportGenerator, 'query_prometheus_instant')
    def test_generate_and_write_report(
        self,
        mock_instant,
        mock_query,
        generator,
        mock_prometheus_response
    ):
        """Test complete workflow of generate and write"""
        base_time = datetime.utcnow()
        
        def side_effect_query(query, start, end, step="15s"):
            if "throughput" in query:
                return mock_prometheus_response("throughput")["data"]["result"]
            elif "latency" in query:
                return mock_prometheus_response("latency")["data"]["result"]
            elif "queue" in query:
                return mock_prometheus_response("queue")["data"]["result"]
            elif "worker" in query:
                return mock_prometheus_response("worker")["data"]["result"]
            elif "error" in query:
                return mock_prometheus_response("error")["data"]["result"]
            return []
        
        mock_query.side_effect = side_effect_query
        mock_instant.return_value = [{"value": [base_time.timestamp(), "150"]}]
        
        report, filepath = generator.generate_and_write_report(
            time_window_hours=1,
            filename_prefix="integration_test"
        )
        
        assert filepath.exists()
        assert "integration_test_" in filepath.name
        assert report["sla_compliance"]["overall_compliance_pct"] == 100.0
        
        with open(filepath, 'r') as f:
            loaded_report = json.load(f)
        
        assert loaded_report == report
    
    @patch.object(BatchPerformanceReportGenerator, 'query_prometheus')
    def test_generate_report_query_failure(self, mock_query, generator):
        """Test report generation with Prometheus query failure"""
        mock_query.side_effect = PrometheusQueryError("Connection failed")
        
        with pytest.raises(PrometheusQueryError):
            generator.generate_report(time_window_hours=1)
    
    @patch('batch_performance_report.BatchPerformanceReportGenerator')
    def test_create_batch_performance_report_convenience(self, mock_generator_class):
        """Test convenience function"""
        mock_generator = Mock()
        mock_report = {"test": "report"}
        mock_path = Path("/tmp/report.json")
        mock_generator.generate_and_write_report.return_value = (mock_report, mock_path)
        mock_generator_class.return_value = mock_generator
        
        report, path = create_batch_performance_report(
            prometheus_url="http://test:9090",
            time_window_hours=2,
            artifacts_dir="/tmp",
            log_file="/tmp/test.log"
        )
        
        assert report == mock_report
        assert path == mock_path
        mock_generator_class.assert_called_once_with(
            prometheus_url="http://test:9090",
            artifacts_dir="/tmp",
            log_file="/tmp/test.log"
        )
        mock_generator.generate_and_write_report.assert_called_once_with(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
