#!/usr/bin/env python3
"""
Test suite for Connection Stability Analyzer
"""

import json

import pytest

from connection_stability_analyzer import (
    ConnectionMetrics,
    ConnectionStabilityAnalyzer,
    DataType,
    FlowSpecification,
    create_connection_stability_analyzer,
)


class TestConnectionMetrics:
    """Test ConnectionMetrics dataclass"""

    @staticmethod
    def test_metrics_initialization():
        metrics = ConnectionMetrics(connection_id="test_connection")
        assert metrics.connection_id == "test_connection"
        assert metrics.retry_count == 0
        assert metrics.error_count == 0
        assert metrics.success_count == 0

    @staticmethod
    def test_error_rate_calculation():
        metrics = ConnectionMetrics(connection_id="test")
        metrics.total_attempts = 10
        metrics.error_count = 3
        assert metrics.error_rate == 0.3

    @staticmethod
    def test_error_rate_zero_attempts():
        metrics = ConnectionMetrics(connection_id="test")
        assert metrics.error_rate == 0.0

    @staticmethod
    def test_avg_backoff_delay():
        metrics = ConnectionMetrics(connection_id="test")
        metrics.backoff_delays = [100, 200, 300]
        assert metrics.avg_backoff_delay == 200.0

    @staticmethod
    def test_avg_backoff_delay_empty():
        metrics = ConnectionMetrics(connection_id="test")
        assert metrics.avg_backoff_delay == 0.0

    @staticmethod
    def test_max_backoff_delay():
        metrics = ConnectionMetrics(connection_id="test")
        metrics.backoff_delays = [100, 500, 200]
        assert metrics.max_backoff_delay == 500.0

    @staticmethod
    def test_avg_latency():
        metrics = ConnectionMetrics(connection_id="test")
        metrics.success_count = 5
        metrics.total_latency_ms = 1000.0
        assert metrics.avg_latency_ms == 200.0


class TestFlowSpecification:
    """Test FlowSpecification dataclass"""

    @staticmethod
    def test_flow_spec_creation():
        spec = FlowSpecification(
            flow_id="test_flow",
            source="node_a",
            target="node_b",
            flow_type="data",
            cardinality="1:N",
            input_schema={"data": DataType.RAW_TEXT},
            output_schema={"result": DataType.METADATA},
        )
        assert spec.flow_id == "test_flow"
        assert spec.source == "node_a"
        assert spec.target == "node_b"
        assert spec.cardinality == "1:N"

    @staticmethod
    def test_validate_cardinality_1_to_1():
        spec = FlowSpecification(
            flow_id="test",
            source="a",
            target="b",
            flow_type="data",
            cardinality="1:1",
            input_schema={},
            output_schema={},
        )
        is_valid, msg = spec.validate_cardinality(1)
        assert is_valid is True

        is_valid, msg = spec.validate_cardinality(2)
        assert is_valid is False
        assert "1:1" in msg

    @staticmethod
    def test_validate_cardinality_1_to_n():
        spec = FlowSpecification(
            flow_id="test",
            source="a",
            target="b",
            flow_type="data",
            cardinality="1:N",
            input_schema={},
            output_schema={},
        )
        is_valid, _msg = spec.validate_cardinality(5)
        assert is_valid is True

        is_valid, _msg = spec.validate_cardinality(0)
        assert is_valid is False


class TestConnectionStabilityAnalyzer:
    """Test ConnectionStabilityAnalyzer class"""

    @staticmethod
    def test_analyzer_initialization():
        analyzer = ConnectionStabilityAnalyzer()
        assert analyzer.flow_specifications is not None
        assert analyzer.connection_metrics == {}
        assert analyzer.verdicts == {}

    @staticmethod
    def test_get_or_create_metrics():
        analyzer = ConnectionStabilityAnalyzer()
        metrics = analyzer.get_or_create_metrics("test_connection")
        assert metrics.connection_id == "test_connection"

        same_metrics = analyzer.get_or_create_metrics("test_connection")
        assert same_metrics is metrics

    @staticmethod
    def test_track_retry_attempt():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_retry_attempt("test_conn", 100.0)

        metrics = analyzer.connection_metrics["test_conn"]
        assert metrics.retry_count == 1
        assert metrics.total_attempts == 1
        assert 100.0 in metrics.backoff_delays

    @staticmethod
    def test_track_attempt_success():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("test_conn", success=True, latency_ms=50.0)

        metrics = analyzer.connection_metrics["test_conn"]
        assert metrics.success_count == 1
        assert metrics.error_count == 0
        assert metrics.total_latency_ms == 50.0

    @staticmethod
    def test_track_attempt_failure():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("test_conn", success=False)

        metrics = analyzer.connection_metrics["test_conn"]
        assert metrics.success_count == 0
        assert metrics.error_count == 1

    @staticmethod
    def test_validate_interface_no_spec():
        analyzer = ConnectionStabilityAnalyzer()
        is_valid, errors = analyzer.validate_interface(
            "unknown_source", "unknown_target", {}
        )
        assert is_valid is True
        assert errors == []

    @staticmethod
    def test_validate_type_string():
        analyzer = ConnectionStabilityAnalyzer()
        assert analyzer._validate_type("text", DataType.RAW_TEXT) is True
        assert analyzer._validate_type(123, DataType.RAW_TEXT) is False

    @staticmethod
    def test_validate_type_list():
        analyzer = ConnectionStabilityAnalyzer()
        assert analyzer._validate_type([], DataType.SEGMENTS) is True
        assert analyzer._validate_type("text", DataType.SEGMENTS) is False

    @staticmethod
    def test_validate_type_dict():
        analyzer = ConnectionStabilityAnalyzer()
        assert analyzer._validate_type({}, DataType.METADATA) is True
        assert analyzer._validate_type([], DataType.METADATA) is False

    @staticmethod
    def test_verify_cardinality_no_spec():
        analyzer = ConnectionStabilityAnalyzer()
        is_valid, msg = analyzer.verify_cardinality("unknown_conn", 5)
        assert is_valid is True
        assert msg == ""

    @staticmethod
    def test_capture_schema_mismatch():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.capture_schema_mismatch(
            connection_id="test_conn",
            expected_schema={"field1": "str", "field2": "int"},
            actual_schema={"field1": "str", "field3": "bool"},
            example_data={"field1": "value", "field3": True},
        )

        assert "test_conn" in analyzer.schema_mismatches
        mismatches = analyzer.schema_mismatches["test_conn"]
        assert len(mismatches) == 1

        mismatch = mismatches[0]
        assert "field2" in mismatch.missing_fields
        assert "field3" in mismatch.extra_fields

    @staticmethod
    def test_analyze_connection_stability():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("test_conn", success=True, latency_ms=100.0)
        analyzer.track_attempt("test_conn", success=True, latency_ms=150.0)
        analyzer.track_attempt("test_conn", success=False)

        analysis = analyzer.analyze_connection_stability("test_conn")
        assert analysis["connection_id"] == "test_conn"
        assert analysis["total_attempts"] == 3
        assert analysis["success_count"] == 2
        assert analysis["error_count"] == 1
        assert analysis["error_rate"] == pytest.approx(0.333, rel=0.01)

    @staticmethod
    def test_is_connection_stable_good():
        analyzer = ConnectionStabilityAnalyzer()
        metrics = ConnectionMetrics(connection_id="test")
        metrics.total_attempts = 100
        metrics.success_count = 98
        metrics.error_count = 2

        assert analyzer._is_connection_stable(metrics) is True

    @staticmethod
    def test_is_connection_stable_high_error_rate():
        analyzer = ConnectionStabilityAnalyzer()
        metrics = ConnectionMetrics(connection_id="test")
        metrics.total_attempts = 100
        metrics.success_count = 80
        metrics.error_count = 20

        assert analyzer._is_connection_stable(metrics) is False

    @staticmethod
    def test_is_connection_stable_high_retry_rate():
        analyzer = ConnectionStabilityAnalyzer()
        metrics = ConnectionMetrics(connection_id="test")
        metrics.total_attempts = 100
        metrics.retry_count = 40

        assert analyzer._is_connection_stable(metrics) is False

    @staticmethod
    def test_is_connection_stable_schema_mismatches():
        analyzer = ConnectionStabilityAnalyzer()
        metrics = ConnectionMetrics(connection_id="test")
        metrics.total_attempts = 100
        metrics.schema_mismatches = 5

        assert analyzer._is_connection_stable(metrics) is True

    @staticmethod
    def test_identify_stability_issues():
        analyzer = ConnectionStabilityAnalyzer()
        metrics = ConnectionMetrics(connection_id="test")
        metrics.total_attempts = 100
        metrics.error_count = 15
        metrics.retry_count = 35
        metrics.schema_mismatches = 2
        metrics.backoff_delays = [6000]

        issues = analyzer._identify_stability_issues(metrics)
        assert len(issues) > 0
        assert any("error rate" in issue.lower() for issue in issues)
        assert any("retry rate" in issue.lower() for issue in issues)
        assert any("schema" in issue.lower() for issue in issues)
        assert any("backoff" in issue.lower() for issue in issues)

    @staticmethod
    def test_generate_verdict_stable_suitable():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("test_conn", success=True, latency_ms=100.0)
        analyzer.track_attempt("test_conn", success=True, latency_ms=120.0)

        verdict = analyzer.generate_verdict("test_conn")
        assert verdict.connection_id == "test_conn"
        assert verdict.is_stable is True
        assert verdict.is_suitable is True
        assert verdict.stability_score > 0.9
        assert verdict.suitability_score > 0.9
        assert verdict.verdict_status == "STABLE_SUITABLE"

    @staticmethod
    def test_generate_verdict_unstable():
        analyzer = ConnectionStabilityAnalyzer()
        for _ in range(5):
            analyzer.track_attempt("test_conn", success=False)
        for _ in range(3):
            analyzer.track_attempt("test_conn", success=True)

        verdict = analyzer.generate_verdict("test_conn")
        assert verdict.is_stable is False
        assert len(verdict.violations) > 0

    @staticmethod
    def test_generate_verdict_unsuitable():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("test_conn", success=True)
        metrics = analyzer.get_or_create_metrics("test_conn")
        metrics.schema_mismatches = 5
        metrics.type_incompatibilities = 3

        verdict = analyzer.generate_verdict("test_conn")
        assert verdict.is_suitable is False
        assert len(verdict.violations) > 0
        assert any("schema" in v.lower() for v in verdict.violations)

    @staticmethod
    def test_generate_report_structure():
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("conn1", success=True)
        analyzer.track_attempt("conn2", success=False)

        report = analyzer.generate_report()

        assert "summary" in report
        assert "connection_categories" in report
        assert "verdicts" in report
        assert "flow_specifications" in report
        assert "timestamp" in report

        assert report["summary"]["total_connections"] == 2
        assert "stability_rate" in report["summary"]
        assert "suitability_rate" in report["summary"]

    @staticmethod
    def test_generate_report_categories():
        analyzer = ConnectionStabilityAnalyzer()

        analyzer.track_attempt("good_conn", success=True)
        for _ in range(5):
            analyzer.track_attempt("bad_conn", success=False)

        report = analyzer.generate_report()
        categories = report["connection_categories"]

        assert "stable_suitable" in categories
        assert "stable_unsuitable" in categories
        assert "unstable_suitable" in categories
        assert "unstable_unsuitable" in categories

    @staticmethod
    def test_export_report(tmp_path):
        analyzer = ConnectionStabilityAnalyzer()
        analyzer.track_attempt("test_conn", success=True)

        output_file = tmp_path / "test_report.json"
        _report = analyzer.export_report(str(output_file))

        assert output_file.exists()

        with open(output_file, "r") as f:
            loaded_report = json.load(f)

        assert loaded_report["summary"]["total_connections"] == 1

    @staticmethod
    def test_factory_function():
        analyzer = create_connection_stability_analyzer()
        assert isinstance(analyzer, ConnectionStabilityAnalyzer)


class TestIntegration:
    """Integration tests for full workflow"""

    @staticmethod
    def test_full_validation_workflow():
        analyzer = ConnectionStabilityAnalyzer()

        analyzer.track_attempt("flow1", success=True, latency_ms=100)
        analyzer.track_attempt("flow1", success=True, latency_ms=120)
        analyzer.track_retry_attempt("flow1", backoff_delay=50)

        analyzer.track_attempt("flow2", success=False)
        analyzer.track_attempt("flow2", success=False)
        analyzer.track_retry_attempt("flow2", backoff_delay=200)
        analyzer.track_retry_attempt("flow2", backoff_delay=400)

        analyzer.capture_schema_mismatch(
            "flow2", {"field1": "str"}, {"field2": "int"}, {"field2": 42}
        )

        report = analyzer.generate_report()

        assert report["summary"]["total_connections"] == 2
        assert len(report["verdicts"]) == 2

        flow1_verdict = report["verdicts"]["flow1"]
        assert flow1_verdict["status"] in ["STABLE_SUITABLE", "UNSTABLE_SUITABLE"]

        flow2_verdict = report["verdicts"]["flow2"]
        assert flow2_verdict["status"] == "UNSTABLE_UNSUITABLE"
        assert len(flow2_verdict["violations"]) > 0

    @staticmethod
    def test_multiple_schema_mismatches():
        analyzer = ConnectionStabilityAnalyzer()

        for i in range(3):
            analyzer.capture_schema_mismatch(
                "problematic_conn",
                {"required_field": "str"},
                {"wrong_field": "int"},
                {"wrong_field": i},
            )

        assert len(analyzer.schema_mismatches["problematic_conn"]) == 3

        verdict = analyzer.generate_verdict("problematic_conn")
        assert verdict.is_suitable is False
        assert len(verdict.schema_mismatches) == 3

    @staticmethod
    def test_cardinality_tracking():
        analyzer = ConnectionStabilityAnalyzer()

        flow_spec = FlowSpecification(
            flow_id="test_flow",
            source="source",
            target="target",
            flow_type="data",
            cardinality="1:1",
            input_schema={},
            output_schema={},
        )
        analyzer.flow_specifications["test_flow"] = flow_spec

        is_valid, _msg = analyzer.verify_cardinality("test_flow", 3)
        assert is_valid is False

        metrics = analyzer.get_or_create_metrics("test_flow")
        assert metrics.cardinality_violations == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
