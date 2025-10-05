"""
Memory Monitoring Watchdog for Plan Processing Workers

This module implements a memory monitoring system using psutil that tracks RSS memory usage
during plan processing and terminates workers that exceed a configurable memory threshold.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from log_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


class TerminationReason(Enum):
    MEMORY_EXCEEDED = "memory_exceeded"
    MANUAL = "manual"
    COMPLETED = "completed"
    TIMEOUT = "timeout"


@dataclass
class MemoryUsage:
    rss_mb: float
    vms_mb: float
    percent: float
    timestamp: float


@dataclass
class WatchdogEvent:
    pid: int
    memory_usage: MemoryUsage
    memory_limit_mb: int
    termination_reason: TerminationReason
    timestamp: float


class MemoryWatchdog:
    """
    Memory monitoring watchdog that tracks RSS memory usage of worker processes
    and gracefully terminates them when they exceed configurable thresholds.
    """

    def __init__(
        self, memory_limit_mb: Optional[int] = None, check_interval: float = 1.0
    ):
        """
        Initialize memory watchdog.

        Args:
            memory_limit_mb: Memory limit in MB. If None, uses MEMORY_LIMIT_MB env var or 2048 default
            check_interval: Interval between memory checks in seconds
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil is required for memory monitoring")

        # Configure memory limit with fallback hierarchy
        self.memory_limit_mb = self._get_memory_limit(memory_limit_mb)
        self.check_interval = check_interval

        # Monitoring state
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._monitored_processes: Dict[int, psutil.Process] = {}

        # Event tracking
        self._termination_events: list[WatchdogEvent] = []
        self._lock = threading.RLock()

        # Callbacks
        self._termination_callback: Optional[Callable[[WatchdogEvent], None]] = None

        self.logger = LOGGER

    @staticmethod
    def _get_memory_limit(memory_limit_mb: Optional[int]) -> int:
        """Get memory limit with fallback to environment variable and default."""
        if memory_limit_mb is not None:
            return memory_limit_mb

        # Try environment variable
        env_limit = os.environ.get("MEMORY_LIMIT_MB")
        if env_limit:
            try:
                return int(env_limit)
            except ValueError:
                LOGGER.warning(
                    "Invalid MEMORY_LIMIT_MB value: %s, using default", env_limit
                )

        # Conservative default: 2GB per worker process
        return 2048

    def set_termination_callback(self, callback: Callable[[WatchdogEvent], None]):
        """Set callback function to be called when a process is terminated."""
        self._termination_callback = callback

    def register_process(self, pid: int) -> bool:
        """
        Register a worker process for memory monitoring.

        Args:
            pid: Process ID to monitor

        Returns:
            True if process was successfully registered, False otherwise
        """
        try:
            process = psutil.Process(pid)

            with self._lock:
                self._monitored_processes[pid] = process

            self.logger.info(
                "Registered process %s for memory monitoring (limit: %sMB)",
                pid,
                self.memory_limit_mb,
            )
            return True

        except psutil.NoSuchProcess:
            self.logger.warning("Cannot register process %s: process not found", pid)
            return False
        except psutil.AccessDenied:
            self.logger.warning("Cannot register process %s: access denied", pid)
            return False
        except psutil.Error:
            self.logger.exception("Failed to register process %s", pid)
            return False

    def unregister_process(self, pid: int):
        """Unregister a worker process from memory monitoring."""
        with self._lock:
            if pid in self._monitored_processes:
                del self._monitored_processes[pid]
                self.logger.info(f"Unregistered process {pid} from memory monitoring")

    def start_monitoring(self) -> bool:
        """
        Start the memory monitoring thread.

        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            self.logger.warning("Memory monitoring is already running")
            return False

        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="MemoryWatchdog", daemon=True
        )

        try:
            self._monitoring_thread.start()
            self.logger.info(
                "Started memory monitoring (limit: %sMB, check interval: %ss)",
                self.memory_limit_mb,
                self.check_interval,
            )
            return True
        except RuntimeError:
            self.logger.exception("Failed to start memory monitoring")
            return False

    def stop_monitoring(self, timeout: float = 5.0) -> bool:
        """
        Stop the memory monitoring thread.

        Args:
            timeout: Maximum time to wait for thread to stop

        Returns:
            True if monitoring stopped successfully, False if timeout occurred
        """
        if self._monitoring_thread is None:
            return True

        self._stop_event.set()

        try:
            self._monitoring_thread.join(timeout=timeout)
            if self._monitoring_thread.is_alive():
                self.logger.warning(
                    "Memory monitoring thread did not stop within %ss timeout",
                    timeout,
                )
                return False
            else:
                self.logger.info("Memory monitoring stopped")
                return True
        except RuntimeError:
            self.logger.exception("Error stopping memory monitoring")
            return False

    def _monitoring_loop(self):
        """Main monitoring loop that runs in separate thread."""
        self.logger.debug("Memory monitoring loop started")

        while not self._stop_event.is_set():
            try:
                self._check_processes_memory()

                # Wait for next check or stop signal
                if self._stop_event.wait(timeout=self.check_interval):
                    break  # Stop event was set

            except Exception:  # pragma: no cover - defensive logging
                self.logger.exception("Error in memory monitoring loop")
                # Continue monitoring despite errors
                time.sleep(self.check_interval)

        self.logger.debug("Memory monitoring loop stopped")

    def _check_processes_memory(self):
        """Check memory usage of all registered processes."""
        with self._lock:
            # Work with a copy to avoid issues with concurrent modifications
            processes_to_check = self._monitored_processes.copy()

        processes_to_remove = []

        for pid, process in processes_to_check.items():
            try:
                # Check if process is still running
                if not process.is_running():
                    processes_to_remove.append(pid)
                    self.logger.debug("Process %s is no longer running", pid)
                    continue

                # Get memory info
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()

                memory_usage = MemoryUsage(
                    rss_mb=memory_info.rss / (1024 * 1024),  # Convert bytes to MB
                    vms_mb=memory_info.vms / (1024 * 1024),
                    percent=memory_percent,
                    timestamp=time.time(),
                )

                # Check if memory limit exceeded
                if memory_usage.rss_mb > self.memory_limit_mb:
                    self._terminate_process_for_memory(pid, process, memory_usage)
                    processes_to_remove.append(pid)
                else:
                    # Log high memory usage as warning
                    if memory_usage.rss_mb > self.memory_limit_mb * 0.8:
                        self.logger.warning(
                            "Process %s high memory usage: %.1fMB (limit: %sMB)",
                            pid,
                            memory_usage.rss_mb,
                            self.memory_limit_mb,
                        )
                    else:
                        self.logger.debug(
                            "Process %s memory usage: %.1fMB",
                            pid,
                            memory_usage.rss_mb,
                        )

            except psutil.NoSuchProcess:
                processes_to_remove.append(pid)
                self.logger.debug("Process %s no longer exists", pid)
            except psutil.AccessDenied:
                self.logger.warning("Access denied to process %s", pid)
                processes_to_remove.append(pid)
            except psutil.ZombieProcess:
                processes_to_remove.append(pid)
                self.logger.warning("Process %s became a zombie", pid)
            except psutil.Error:
                self.logger.exception("Error checking memory for process %s", pid)

        # Remove dead or inaccessible processes
        if processes_to_remove:
            with self._lock:
                for pid in processes_to_remove:
                    self._monitored_processes.pop(pid, None)

    def _terminate_process_for_memory(
        self, pid: int, process: psutil.Process, memory_usage: MemoryUsage
    ):
        """Gracefully terminate a process that exceeded memory limit."""
        event = WatchdogEvent(
            pid=pid,
            memory_usage=memory_usage,
            memory_limit_mb=self.memory_limit_mb,
            termination_reason=TerminationReason.MEMORY_EXCEEDED,
            timestamp=time.time(),
        )

        self.logger.warning(
            "Process %s exceeded memory limit: %.1fMB > %sMB, terminating",
            pid,
            memory_usage.rss_mb,
            self.memory_limit_mb,
        )

        try:
            # First try graceful termination
            process.terminate()

            # Wait briefly for graceful shutdown
            try:
                process.wait(timeout=3.0)
                self.logger.info("Process %s terminated gracefully", pid)
            except psutil.TimeoutExpired:
                # Force kill if graceful termination failed
                process.kill()
                process.wait(timeout=1.0)
                self.logger.warning(
                    "Process %s force killed after graceful termination timeout",
                    pid,
                )

        except psutil.NoSuchProcess:
            self.logger.info("Process %s already terminated", pid)
        except psutil.Error:
            self.logger.exception("Failed to terminate process %s", pid)

        # Record termination event
        with self._lock:
            self._termination_events.append(event)

        # Call termination callback if set
        if self._termination_callback:
            try:
                self._termination_callback(event)
            except Exception:  # pragma: no cover - callback is user supplied
                self.logger.exception("Error in termination callback")

    def get_monitored_processes(self) -> Dict[int, Dict[str, Any]]:
        """Get information about currently monitored processes."""
        result = {}

        with self._lock:
            for pid, process in self._monitored_processes.items():
                try:
                    if process.is_running():
                        memory_info = process.memory_info()
                        result[pid] = {
                            "rss_mb": memory_info.rss / (1024 * 1024),
                            "vms_mb": memory_info.vms / (1024 * 1024),
                            "memory_percent": process.memory_percent(),
                            "status": process.status(),
                            "create_time": process.create_time(),
                        }
                except psutil.Error as error:
                    result[pid] = {"error": str(error)}

        return result

    def get_termination_events(self) -> list[WatchdogEvent]:
        """Get list of all termination events."""
        with self._lock:
            return self._termination_events.copy()

    def clear_termination_events(self):
        """Clear the termination events history."""
        with self._lock:
            self._termination_events.clear()

    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active."""
        return (
            self._monitoring_thread is not None
            and self._monitoring_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures monitoring is stopped."""
        self.stop_monitoring()


class PlanProcessingWatchdog:
    """
    High-level wrapper for memory monitoring during plan processing.
    Integrates with job schedulers and worker process management.
    """

    def __init__(self, memory_limit_mb: Optional[int] = None):
        """Initialize plan processing watchdog."""
        if not PSUTIL_AVAILABLE:
            raise ImportError(
                "psutil is required for plan processing memory monitoring"
            )

        self.watchdog = MemoryWatchdog(memory_limit_mb=memory_limit_mb)
        self.logger = LOGGER

        # Track failed plans for retry
        self.failed_plans: list[Dict[str, Any]] = []

    def _handle_worker_termination(self, event: WatchdogEvent):
        """Handle worker process termination due to memory limit."""
        if event.termination_reason == TerminationReason.MEMORY_EXCEEDED:
            self.logger.error(
                "Worker process %s terminated for exceeding memory limit (%.1fMB > %sMB)",
                event.pid,
                event.memory_usage.rss_mb,
                event.memory_limit_mb,
            )

            # Record failed plan for retry
            failed_plan = {
                "pid": event.pid,
                "terminated_at": event.timestamp,
                "memory_usage_mb": event.memory_usage.rss_mb,
                "memory_limit_mb": event.memory_limit_mb,
                "reason": "memory_exceeded",
            }
            self.failed_plans.append(failed_plan)

    def start_plan_processing(self, worker_pid: int) -> bool:
        """
        Start memory monitoring for a plan processing worker.

        Args:
            worker_pid: Process ID of worker to monitor

        Returns:
            True if monitoring started successfully
        """
        # Set termination callback
        self.watchdog.set_termination_callback(self._handle_worker_termination)

        # Start monitoring if not already running
        if not self.watchdog.is_monitoring():
            if not self.watchdog.start_monitoring():
                return False

        # Register worker process
        return self.watchdog.register_process(worker_pid)

    def complete_plan_processing(self, worker_pid: int):
        """
        Mark plan processing as completed for a worker.

        Args:
            worker_pid: Process ID of worker that completed processing
        """
        self.watchdog.unregister_process(worker_pid)
        self.logger.info(
            "Plan processing completed successfully for worker %s", worker_pid
        )

    def stop_all_monitoring(self, timeout: float = 5.0) -> bool:
        """
        Stop all memory monitoring and clean up.

        Args:
            timeout: Maximum time to wait for monitoring to stop

        Returns:
            True if stopped successfully
        """
        return self.watchdog.stop_monitoring(timeout=timeout)

    def get_failed_plans(self) -> list[Dict[str, Any]]:
        """Get list of plans that failed due to memory issues."""
        return self.failed_plans.copy()

    def clear_failed_plans(self):
        """Clear the failed plans list."""
        self.failed_plans.clear()

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "is_monitoring": self.watchdog.is_monitoring(),
            "memory_limit_mb": self.watchdog.memory_limit_mb,
            "monitored_processes": self.watchdog.get_monitored_processes(),
            "termination_events": len(self.watchdog.get_termination_events()),
            "failed_plans": len(self.failed_plans),
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all_monitoring()


# Demo function for testing
def demo_memory_watchdog():
    """Demonstrate memory watchdog functionality."""
    if not PSUTIL_AVAILABLE:
        LOGGER.error("psutil not available - cannot run memory watchdog demo")
        return

    LOGGER.info("Memory Watchdog Demo")
    LOGGER.info("%s", "=" * 40)

    # Test basic watchdog functionality
    with MemoryWatchdog(memory_limit_mb=100) as watchdog:  # Very low limit for demo
        current_pid = os.getpid()
        LOGGER.info(
            "Monitoring current process (PID: %s) with 100MB limit", current_pid
        )

        watchdog.register_process(current_pid)

        # Monitor for a few seconds
        time.sleep(3)

        status = watchdog.get_monitored_processes()
        if current_pid in status:
            LOGGER.info("Current memory usage: %.1fMB", status[current_pid]["rss_mb"])

        events = watchdog.get_termination_events()
        LOGGER.info("Termination events: %s", len(events))

    LOGGER.info("Memory watchdog demo completed")


if __name__ == "__main__":
    demo_memory_watchdog()
