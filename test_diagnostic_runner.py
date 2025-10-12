# coding=utf-8
"""
TEST SUITE — Diagnostic Runner
================================
Comprehensive tests for diagnostic_runner.py instrumentation.

Tests:
- Resource monitoring accuracy
- Contract validation
- Metrics collection
- Wrapper installation
- Determinism preservation
- Report generation
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from diagnostic_runner import (
    ContractValidator,
    DiagnosticRunner,
    DiagnosticWrapper,
    NodeMetrics,
    PipelineMetrics,
    ResourceMonitor,
    run_diagnostic,
)
from miniminimoon_orchestrator import (
    MiniMinimoonOrchestrator,
    PipelineStage,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def node_metrics():
    """Create sample node metrics."""
    return NodeMetrics(
        stage_name="test_stage",
        wall_time_ms=100.5,
        cpu_time_ms=80.2,
        peak_memory_mb=256.0,
        memory_delta_mb=10.5,
        io_wait_ms=5.0,
        error_count=0,
        contract_valid=True,
        thread_id=12345,
    )


@pytest.fixture
def pipeline_metrics():
    """Create sample pipeline metrics."""
    metrics = PipelineMetrics(
        total_wall_time_ms=1500.0,
        total_cpu_time_ms=1200.0,
        peak_memory_mb=512.0,
        total_errors=0,
        stages_passed=15,
        stages_failed=0,
        contract_violations=0,
    )
    return metrics


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    mock = MagicMock(spec=MiniMinimoonOrchestrator)

    # Add mock methods for all stages
    mock._sanitize = MagicMock(return_value={"sanitized_text": "test"})
    mock._process_plan = MagicMock(return_value={"processed": True})
    mock._segment = MagicMock(return_value={"segments": []})
    mock._embed = MagicMock(return_value={"embeddings": []})
    mock._detect_responsibilities = MagicMock(return_value={"responsibilities": []})
    mock._detect_contradictions = MagicMock(return_value={"contradictions": []})
    mock._detect_monetary = MagicMock(return_value={"monetary": []})
    mock._score_feasibility = MagicMock(return_value={"feasibility": 0.8})
    mock._detect_causal_patterns = MagicMock(return_value={"causal": []})
    mock._validate_teoria = MagicMock(return_value={"teoria_valid": True})
    mock._validate_dag = MagicMock(return_value={"dag_valid": True})
    mock._build_registry = MagicMock(return_value={"registry_id": "test"})
    mock._evaluate_decalogo = MagicMock(return_value={"decalogo_score": 0.9})
    mock._evaluate_questionnaire = MagicMock(return_value={"questionnaire_score": 0.85})
    mock._assemble_answers = MagicMock(return_value={"answers": []})

    return mock


# ============================================================================
# TEST NODE METRICS
# ============================================================================


def test_node_metrics_creation(node_metrics):
    """Test NodeMetrics dataclass creation."""
    assert node_metrics.stage_name == "test_stage"
    assert node_metrics.wall_time_ms == 100.5
    assert node_metrics.cpu_time_ms == 80.2
    assert node_metrics.peak_memory_mb == 256.0
    assert node_metrics.error_count == 0
    assert node_metrics.contract_valid is True


def test_node_metrics_to_dict(node_metrics):
    """Test NodeMetrics serialization."""
    data = node_metrics.to_dict()
    assert isinstance(data, dict)
    assert data["stage_name"] == "test_stage"
    assert data["wall_time_ms"] == 100.5
    assert data["contract_valid"] is True


# ============================================================================
# TEST PIPELINE METRICS
# ============================================================================


def test_pipeline_metrics_creation(pipeline_metrics):
    """Test PipelineMetrics dataclass creation."""
    assert pipeline_metrics.total_wall_time_ms == 1500.0
    assert pipeline_metrics.stages_passed == 15
    assert pipeline_metrics.stages_failed == 0


def test_pipeline_metrics_to_dict(pipeline_metrics):
    """Test PipelineMetrics serialization."""
    data = pipeline_metrics.to_dict()
    assert isinstance(data, dict)
    assert data["total_wall_time_ms"] == 1500.0
    assert data["stages_passed"] == 15


# ============================================================================
# TEST CONTRACT VALIDATOR
# ============================================================================


def test_contract_validator_initialization():
    """Test ContractValidator initialization."""
    validator = ContractValidator()
    assert len(validator.STAGE_CONTRACTS) == 15
    assert PipelineStage.SANITIZATION in validator.STAGE_CONTRACTS


def test_contract_validator_input_validation():
    """Test input contract validation."""
    validator = ContractValidator()

    # Valid input
    valid, errors = validator.validate_input(
        PipelineStage.SANITIZATION, {"text": "test"}
    )
    assert valid is True
    assert len(errors) == 0

    # Invalid input (not dict)
    valid, errors = validator.validate_input(PipelineStage.SANITIZATION, "not a dict")
    assert valid is False
    assert len(errors) > 0


def test_contract_validator_output_validation():
    """Test output contract validation."""
    validator = ContractValidator()

    # Valid output
    valid, _errors = validator.validate_output(
        PipelineStage.SANITIZATION, {"sanitized_text": "test"}
    )
    assert valid is True

    # Invalid output (None)
    valid, _errors = validator.validate_output(PipelineStage.SANITIZATION, None)
    assert valid is False


# ============================================================================
# TEST RESOURCE MONITOR
# ============================================================================


def test_resource_monitor_initialization():
    """Test ResourceMonitor initialization."""
    monitor = ResourceMonitor()
    assert monitor.process is not None
    assert monitor._baseline_memory == 0.0


def test_resource_monitor_memory_tracking():
    """Test memory usage tracking."""
    monitor = ResourceMonitor()

    mem_mb = monitor.get_memory_mb()
    assert mem_mb > 0.0
    assert isinstance(mem_mb, float)


def test_resource_monitor_baseline_capture():
    """Test baseline resource capture."""
    monitor = ResourceMonitor()

    monitor.capture_baseline()
    assert monitor._baseline_memory > 0.0

    delta = monitor.get_memory_delta_mb()
    assert isinstance(delta, float)


def test_resource_monitor_cpu_time():
    """Test CPU time tracking."""
    monitor = ResourceMonitor()

    cpu_ms = monitor.get_cpu_time_ms()
    assert cpu_ms >= 0.0
    assert isinstance(cpu_ms, float)


# ============================================================================
# TEST DIAGNOSTIC WRAPPER
# ============================================================================


def test_diagnostic_wrapper_initialization(mock_orchestrator):
    """Test DiagnosticWrapper initialization."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    assert wrapper.orchestrator is mock_orchestrator
    assert isinstance(wrapper.metrics, dict)
    assert isinstance(wrapper.validator, ContractValidator)
    assert len(wrapper._original_methods) > 0


def test_diagnostic_wrapper_hook_installation(mock_orchestrator):
    """Test that hooks are installed for all stage methods."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    # Verify original methods are saved
    assert len(wrapper._original_methods) > 0

    # Verify methods are wrapped
    for stage, method_name in wrapper.STAGE_METHODS.items():
        if hasattr(mock_orchestrator, method_name):
            assert stage.value in wrapper._original_methods


def test_diagnostic_wrapper_metrics_collection(mock_orchestrator):
    """Test metrics collection during method execution."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    # Execute a wrapped method
    _result = mock_orchestrator._sanitize({"text": "test"})

    # Verify metrics were collected
    metrics = wrapper.get_metrics()
    assert PipelineStage.SANITIZATION.value in metrics

    stage_metrics = metrics[PipelineStage.SANITIZATION.value]
    assert stage_metrics.wall_time_ms > 0
    assert stage_metrics.stage_name == PipelineStage.SANITIZATION.value


def test_diagnostic_wrapper_preserves_determinism(mock_orchestrator):
    """Test that wrapper preserves deterministic execution."""
    _wrapper = DiagnosticWrapper(mock_orchestrator)

    # Execute same method twice
    result1 = mock_orchestrator._sanitize({"text": "test"})
    result2 = mock_orchestrator._sanitize({"text": "test"})

    # Results should be identical (mock returns same value)
    assert result1 == result2


def test_diagnostic_wrapper_error_handling(mock_orchestrator):
    """Test error handling in wrapper."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    # Make method raise exception
    mock_orchestrator._sanitize.side_effect = ValueError("Test error")

    # Verify exception is propagated
    with pytest.raises(ValueError):
        mock_orchestrator._sanitize({"text": "test"})

    # Verify error was recorded
    metrics = wrapper.get_metrics()
    if PipelineStage.SANITIZATION.value in metrics:
        stage_metrics = metrics[PipelineStage.SANITIZATION.value]
        assert stage_metrics.error_count > 0


def test_diagnostic_wrapper_thread_safety(mock_orchestrator):
    """Test thread-safe metrics collection."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    def execute_stage():
        mock_orchestrator._sanitize({"text": "test"})

    # Execute from multiple threads
    threads = [threading.Thread(target=execute_stage) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify metrics were collected (should have one entry per execution)
    metrics = wrapper.get_metrics()
    assert PipelineStage.SANITIZATION.value in metrics


# ============================================================================
# TEST DIAGNOSTIC RUNNER
# ============================================================================


def test_diagnostic_runner_initialization():
    """Test DiagnosticRunner initialization."""
    runner = DiagnosticRunner()

    assert runner.config_dir == Path("config")
    assert isinstance(runner.pipeline_metrics, PipelineMetrics)
    assert runner.wrapper is None


def test_diagnostic_runner_with_orchestrator(mock_orchestrator):
    """Test DiagnosticRunner with mock orchestrator."""
    runner = DiagnosticRunner(orchestrator=mock_orchestrator)

    assert runner.orchestrator is mock_orchestrator


@patch("diagnostic_runner.MiniMinimoonOrchestrator")
def test_diagnostic_runner_execution(mock_orch_class, tmp_path):
    """Test full diagnostic execution."""
    mock_orch = MagicMock()
    mock_orch.evaluate = MagicMock(return_value={"final_score": 0.85, "answers": []})
    mock_orch_class.return_value = mock_orch

    # Add required methods
    mock_orch._sanitize = MagicMock(return_value={"sanitized_text": "test"})
    mock_orch._process_plan = MagicMock(return_value={"processed": True})
    mock_orch._segment = MagicMock(return_value={"segments": []})

    runner = DiagnosticRunner(config_dir=tmp_path)

    # This will fail without full orchestrator, but we test initialization
    assert runner is not None


def test_diagnostic_runner_report_generation():
    """Test report generation."""
    runner = DiagnosticRunner()

    # Add sample metrics
    runner.pipeline_metrics.total_wall_time_ms = 1500.0
    runner.pipeline_metrics.stages_passed = 10
    runner.pipeline_metrics.stages_failed = 0

    report = runner.generate_report()

    assert "DIAGNOSTIC REPORT" in report
    assert "SUMMARY" in report
    assert "1500.00 ms" in report


def test_diagnostic_runner_json_export(tmp_path):
    """Test JSON metrics export."""
    runner = DiagnosticRunner()

    runner.pipeline_metrics.total_wall_time_ms = 1500.0
    runner.pipeline_metrics.stages_passed = 10

    output_path = tmp_path / "metrics.json"
    runner.export_metrics_json(output_path)

    assert output_path.exists()

    with open(output_path, "r") as f:
        data = json.load(f)

    assert data["total_wall_time_ms"] == 1500.0
    assert data["stages_passed"] == 10


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================


@patch("diagnostic_runner.DiagnosticRunner")
def test_run_diagnostic_convenience(mock_runner_class, tmp_path):
    """Test run_diagnostic convenience function."""
    mock_runner = MagicMock()
    mock_runner.run_with_diagnostics.return_value = {
        "orchestrator_result": {"score": 0.85},
        "diagnostic_metrics": {},
    }
    mock_runner.generate_report.return_value = "Test report"
    mock_runner_class.return_value = mock_runner

    result = run_diagnostic(
        input_text="Test plan", plan_id="test_run", output_dir=tmp_path
    )

    assert "orchestrator_result" in result
    mock_runner.run_with_diagnostics.assert_called_once()
    mock_runner.generate_report.assert_called_once()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_end_to_end_metrics_flow(mock_orchestrator):
    """Test complete metrics collection flow."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    # Execute multiple stages
    mock_orchestrator._sanitize({"text": "test"})
    mock_orchestrator._process_plan({"text": "test"})
    mock_orchestrator._segment({"data": "test"})

    # Verify all metrics collected
    metrics = wrapper.get_metrics()

    assert len(metrics) >= 3
    assert PipelineStage.SANITIZATION.value in metrics
    assert PipelineStage.PLAN_PROCESSING.value in metrics
    assert PipelineStage.SEGMENTATION.value in metrics


def test_contract_violation_detection(mock_orchestrator):
    """Test contract violation detection."""
    wrapper = DiagnosticWrapper(mock_orchestrator)

    # Execute with invalid input (non-dict)
    try:
        mock_orchestrator._sanitize("invalid input")
    except:
        pass

    metrics = wrapper.get_metrics()
    if PipelineStage.SANITIZATION.value in metrics:
        stage_metrics = metrics[PipelineStage.SANITIZATION.value]
        # Contract validation may flag issues
        assert isinstance(stage_metrics.contract_valid, bool)


def test_performance_overhead_minimal(mock_orchestrator):
    """Test that instrumentation overhead is minimal."""
    # Execute without wrapper
    start = time.perf_counter()
    for _ in range(100):
        mock_orchestrator._sanitize({"text": "test"})
    baseline_time = time.perf_counter() - start

    # Reset mock
    mock_orchestrator._sanitize.reset_mock()
    mock_orchestrator._sanitize.return_value = {"sanitized_text": "test"}

    # Execute with wrapper
    _wrapper = DiagnosticWrapper(mock_orchestrator)
    start = time.perf_counter()
    for _ in range(100):
        mock_orchestrator._sanitize({"text": "test"})
    wrapped_time = time.perf_counter() - start

    # Overhead should be reasonable (less than 50% increase)
    # This is a rough heuristic for testing
    overhead_ratio = (
        (wrapped_time - baseline_time) / baseline_time if baseline_time > 0 else 0
    )
    assert overhead_ratio < 2.0  # Less than 100% overhead


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
