"""
Test suite for memory monitoring watchdog functionality.
"""

import multiprocessing
import os
import time
from unittest import mock

import pytest

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from memory_watchdog import (
    MemoryUsage,
    MemoryWatchdog,
    PlanProcessingWatchdog,
    TerminationReason,
    WatchdogEvent,
)


@pytest.fixture
def mock_psutil():
    """Mock psutil for testing without actual process monitoring."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available")

    with mock.patch("memory_watchdog.psutil") as mock_psutil:
        # Mock process
        mock_process = mock.MagicMock()
        mock_process.is_running.return_value = True
        mock_process.memory_info.return_value = mock.MagicMock(
            rss=50 * 1024 * 1024,
            vms=100 * 1024 * 1024,  # 50MB  # 100MB
        )
        mock_process.memory_percent.return_value = 5.0

        mock_psutil.Process.return_value = mock_process
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.AccessDenied = psutil.AccessDenied
        mock_psutil.TimeoutExpired = psutil.TimeoutExpired

        yield mock_psutil


@pytest.fixture
def memory_watchdog():
    """Create a memory watchdog instance for testing."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available")

    # Use small memory limit for testing
    watchdog = MemoryWatchdog(memory_limit_mb=100, check_interval=0.1)
    yield watchdog

    # Cleanup
    if watchdog.is_monitoring():
        watchdog.stop_monitoring()


@pytest.fixture
def plan_watchdog():
    """Create a plan processing watchdog instance for testing."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available")

    watchdog = PlanProcessingWatchdog(memory_limit_mb=100)
    yield watchdog

    # Cleanup
    watchdog.stop_all_monitoring()


class TestMemoryWatchdog:
    """Test cases for MemoryWatchdog class."""

    @staticmethod
    def test_memory_limit_configuration():
        """Test memory limit configuration with defaults and env vars."""
        # Test explicit limit
        watchdog = MemoryWatchdog(memory_limit_mb=1024)
        assert watchdog.memory_limit_mb == 1024

        # Test environment variable
        with mock.patch.dict(os.environ, {"MEMORY_LIMIT_MB": "512"}):
            watchdog = MemoryWatchdog()
            assert watchdog.memory_limit_mb == 512

        # Test invalid environment variable falls back to default
        with mock.patch.dict(os.environ, {"MEMORY_LIMIT_MB": "invalid"}):
            watchdog = MemoryWatchdog()
            assert watchdog.memory_limit_mb == 2048  # Default

        # Test default when no env var
        with mock.patch.dict(os.environ, {}, clear=True):
            watchdog = MemoryWatchdog()
            assert watchdog.memory_limit_mb == 2048  # Default

    @staticmethod
    def test_psutil_requirement():
        """Test that watchdog requires psutil."""
        with mock.patch("memory_watchdog.PSUTIL_AVAILABLE", False):
            with pytest.raises(ImportError, match="psutil is required"):
                MemoryWatchdog()

    @staticmethod
    def test_process_registration(memory_watchdog, mock_psutil):
        """Test process registration and unregistration."""
        # Test successful registration
        assert memory_watchdog.register_process(1234)
        assert 1234 in memory_watchdog._monitored_processes

        # Test registration of non-existent process
        mock_psutil.Process.side_effect = psutil.NoSuchProcess(1235)
        assert not memory_watchdog.register_process(1235)

        # Test unregistration
        memory_watchdog.unregister_process(1234)
        assert 1234 not in memory_watchdog._monitored_processes

    @staticmethod
    def test_monitoring_lifecycle(memory_watchdog):
        """Test starting and stopping monitoring."""
        # Initial state
        assert not memory_watchdog.is_monitoring()

        # Start monitoring
        assert memory_watchdog.start_monitoring()
        assert memory_watchdog.is_monitoring()

        # Trying to start again should return False
        assert not memory_watchdog.start_monitoring()

        # Stop monitoring
        assert memory_watchdog.stop_monitoring(timeout=2.0)
        assert not memory_watchdog.is_monitoring()

    @staticmethod
    def test_memory_check_normal_usage(memory_watchdog, mock_psutil):
        """Test memory checking with normal usage."""
        # Set up mock for normal memory usage (50MB < 100MB limit)
        mock_process = mock_psutil.Process.return_value
        mock_process.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB

        # Register process and start monitoring
        memory_watchdog.register_process(1234)
        memory_watchdog.start_monitoring()

        # Wait for a few checks
        time.sleep(0.3)

        # Process should still be registered (not terminated)
        assert 1234 in memory_watchdog._monitored_processes
        assert len(memory_watchdog.get_termination_events()) == 0

        memory_watchdog.stop_monitoring()

    @staticmethod
    def test_memory_check_excessive_usage(memory_watchdog, mock_psutil):
        """Test memory checking with excessive usage."""
        # Set up mock for excessive memory usage (200MB > 100MB limit)
        mock_process = mock_psutil.Process.return_value
        mock_process.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB

        # Register process and start monitoring
        memory_watchdog.register_process(1234)
        memory_watchdog.start_monitoring()

        # Wait for memory check to trigger termination
        time.sleep(0.3)

        # Process should be terminated and unregistered
        assert 1234 not in memory_watchdog._monitored_processes
        events = memory_watchdog.get_termination_events()
        assert len(events) == 1
        assert events[0].termination_reason == TerminationReason.MEMORY_EXCEEDED
        assert events[0].pid == 1234

        memory_watchdog.stop_monitoring()

    @staticmethod
    def test_dead_process_cleanup(memory_watchdog, mock_psutil):
        """Test cleanup of dead processes."""
        # Set up mock for dead process
        mock_process = mock_psutil.Process.return_value
        mock_process.is_running.return_value = False

        # Register process and start monitoring
        memory_watchdog.register_process(1234)
        memory_watchdog.start_monitoring()

        # Wait for cleanup
        time.sleep(0.3)

        # Dead process should be automatically unregistered
        assert 1234 not in memory_watchdog._monitored_processes

        memory_watchdog.stop_monitoring()

    @staticmethod
    def test_termination_callback(memory_watchdog, mock_psutil):
        """Test termination callback functionality."""
        callback_events = []

        def test_callback(event):
            callback_events.append(event)

        memory_watchdog.set_termination_callback(test_callback)

        # Set up excessive memory usage
        mock_process = mock_psutil.Process.return_value
        mock_process.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB

        # Register process and start monitoring
        memory_watchdog.register_process(1234)
        memory_watchdog.start_monitoring()

        # Wait for termination
        time.sleep(0.3)

        # Callback should have been called
        assert len(callback_events) == 1
        assert callback_events[0].pid == 1234

        memory_watchdog.stop_monitoring()

    @staticmethod
    def test_context_manager(mock_psutil):
        """Test context manager functionality."""
        with MemoryWatchdog(memory_limit_mb=100, check_interval=0.1) as watchdog:
            assert watchdog.is_monitoring()

        # Should be stopped after context exit
        assert not watchdog.is_monitoring()

    @staticmethod
    def test_get_monitored_processes_info(memory_watchdog, mock_psutil):
        """Test getting information about monitored processes."""
        # Set up mock process
        mock_process = mock_psutil.Process.return_value
        mock_process.memory_info.return_value = mock.MagicMock(
            rss=50 * 1024 * 1024,
            vms=100 * 1024 * 1024,  # 50MB  # 100MB
        )
        mock_process.memory_percent.return_value = 5.0
        mock_process.status.return_value = "running"
        mock_process.create_time.return_value = time.time()

        # Register process
        memory_watchdog.register_process(1234)

        # Get process info
        processes_info = memory_watchdog.get_monitored_processes()

        assert 1234 in processes_info
        assert processes_info[1234]["rss_mb"] == 50.0
        assert processes_info[1234]["memory_percent"] == 5.0
        assert processes_info[1234]["status"] == "running"


class TestPlanProcessingWatchdog:
    """Test cases for PlanProcessingWatchdog class."""

    @staticmethod
    def test_psutil_requirement():
        """Test that plan watchdog requires psutil."""
        with mock.patch("memory_watchdog.PSUTIL_AVAILABLE", False):
            with pytest.raises(ImportError, match="psutil is required"):
                PlanProcessingWatchdog()

    @staticmethod
    def test_worker_termination_handling(plan_watchdog):
        """Test handling of worker process termination."""
        # Create mock termination event
        memory_usage = MemoryUsage(
            rss_mb=200.0, vms_mb=300.0, percent=20.0, timestamp=time.time()
        )

        event = WatchdogEvent(
            pid=1234,
            memory_usage=memory_usage,
            memory_limit_mb=100,
            termination_reason=TerminationReason.MEMORY_EXCEEDED,
            timestamp=time.time(),
        )

        # Handle termination
        plan_watchdog._handle_worker_termination(event)

        # Check that failed plan was recorded
        failed_plans = plan_watchdog.get_failed_plans()
        assert len(failed_plans) == 1
        assert failed_plans[0]["pid"] == 1234
        assert failed_plans[0]["reason"] == "memory_exceeded"

    @staticmethod
    def test_plan_processing_lifecycle(plan_watchdog):
        """Test complete plan processing lifecycle."""
        with mock.patch.object(
            plan_watchdog.watchdog, "register_process", return_value=True
        ):
            with mock.patch.object(
                plan_watchdog.watchdog, "start_monitoring", return_value=True
            ):
                # Start plan processing
                assert plan_watchdog.start_plan_processing(1234)

        # Complete plan processing
        with mock.patch.object(plan_watchdog.watchdog, "unregister_process"):
            plan_watchdog.complete_plan_processing(1234)

    @staticmethod
    def test_monitoring_status(plan_watchdog):
        """Test monitoring status reporting."""
        status = plan_watchdog.get_monitoring_status()

        assert "is_monitoring" in status
        assert "memory_limit_mb" in status
        assert "monitored_processes" in status
        assert "termination_events" in status
        assert "failed_plans" in status
        assert status["memory_limit_mb"] == 100

    @staticmethod
    def test_failed_plans_management(plan_watchdog):
        """Test failed plans tracking and clearing."""
        # Initially empty
        assert len(plan_watchdog.get_failed_plans()) == 0

        # Add a failed plan manually
        failed_plan = {
            "pid": 1234,
            "terminated_at": time.time(),
            "memory_usage_mb": 200.0,
            "memory_limit_mb": 100,
            "reason": "memory_exceeded",
        }
        plan_watchdog.failed_plans.append(failed_plan)

        assert len(plan_watchdog.get_failed_plans()) == 1

        # Clear failed plans
        plan_watchdog.clear_failed_plans()
        assert len(plan_watchdog.get_failed_plans()) == 0

    @staticmethod
    def test_context_manager():
        """Test plan watchdog context manager."""
        with mock.patch("memory_watchdog.PSUTIL_AVAILABLE", True):
            with PlanProcessingWatchdog(memory_limit_mb=100) as watchdog:
                assert watchdog is not None
                # Context manager should handle cleanup


class TestMemoryUsageDataStructures:
    """Test cases for data structures."""

    @staticmethod
    def test_memory_usage_creation():
        """Test MemoryUsage dataclass creation."""
        usage = MemoryUsage(
            rss_mb=100.5, vms_mb=200.0, percent=10.5, timestamp=time.time()
        )

        assert usage.rss_mb == 100.5
        assert usage.vms_mb == 200.0
        assert usage.percent == 10.5
        assert isinstance(usage.timestamp, float)

    @staticmethod
    def test_watchdog_event_creation():
        """Test WatchdogEvent dataclass creation."""
        memory_usage = MemoryUsage(100.0, 200.0, 10.0, time.time())

        event = WatchdogEvent(
            pid=1234,
            memory_usage=memory_usage,
            memory_limit_mb=150,
            termination_reason=TerminationReason.MEMORY_EXCEEDED,
            timestamp=time.time(),
        )

        assert event.pid == 1234
        assert event.memory_usage == memory_usage
        assert event.memory_limit_mb == 150
        assert event.termination_reason == TerminationReason.MEMORY_EXCEEDED


class TestIntegrationScenarios:
    """Integration test scenarios."""

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_real_process_monitoring(self):
        """Test with actual current process (integration test)."""
        current_pid = os.getpid()

        # Use high memory limit so we don't terminate ourselves
        with MemoryWatchdog(memory_limit_mb=4096, check_interval=0.1) as watchdog:
            # Register current process
            assert watchdog.register_process(current_pid)

            # Monitor for a short time
            time.sleep(0.3)

            # Check process info
            processes = watchdog.get_monitored_processes()
            assert current_pid in processes
            assert processes[current_pid]["rss_mb"] > 0

    @staticmethod
    def test_memory_exhaustion_simulation(memory_watchdog, mock_psutil):
        """Test simulation of memory exhaustion scenario."""
        # Set up callback to track terminations
        termination_events = []
        memory_watchdog.set_termination_callback(
            lambda event: termination_events.append(event)
        )

        # Mock process with high memory usage
        mock_process = mock_psutil.Process.return_value
        mock_process.memory_info.return_value.rss = (
            300 * 1024 * 1024
        )  # 300MB > 100MB limit

        # Start monitoring
        memory_watchdog.register_process(1234)
        memory_watchdog.start_monitoring()

        # Wait for termination
        time.sleep(0.3)

        # Verify termination occurred
        assert len(termination_events) == 1
        assert (
            termination_events[0].termination_reason
            == TerminationReason.MEMORY_EXCEEDED
        )
        assert 1234 not in memory_watchdog._monitored_processes

        memory_watchdog.stop_monitoring()


def run_memory_consuming_worker():
    """Worker function that consumes memory for multiprocessing test."""
    # Allocate a large list to consume memory
    large_data = [0] * (50 * 1024 * 1024)  # Approximately 200MB of integers
    time.sleep(2)  # Keep the memory allocated for 2 seconds
    return len(large_data)


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
def test_multiprocessing_memory_monitoring():
    """Test memory monitoring with actual multiprocessing worker."""
    # Use lower memory limit to trigger termination
    with PlanProcessingWatchdog(memory_limit_mb=100) as watchdog:
        # Start a memory-consuming process
        process = multiprocessing.Process(target=run_memory_consuming_worker)
        process.start()

        # Monitor the process
        watchdog.start_plan_processing(process.pid)

        # Wait for process to complete or be terminated
        process.join(timeout=5)

        if process.is_alive():
            process.terminate()
            process.join()

        # Check for termination events or completion
        status = watchdog.get_monitoring_status()
        # Either the process completed normally or was terminated for memory usage


if __name__ == "__main__":
    # Run basic tests if executed directly
    pytest.main([__file__, "-v"])
