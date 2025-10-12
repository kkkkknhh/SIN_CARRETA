"""
Batch processing optimization components.

This module provides components for optimizing batch document processing:
- Document scheduling
- Pre-validation
- Resource monitoring
- Circuit breaker pattern
- Result streaming
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a document to be processed."""

    doc_id: str
    size_bytes: int
    priority: int = 0
    estimated_time_seconds: float = 0.0


class DocumentScheduler:
    """Schedules documents for batch processing based on priority and resources."""

    def __init__(self):
        """Initialize the scheduler."""
        self.queue: List[DocumentMetadata] = []

    def add_document(self, metadata: DocumentMetadata) -> None:
        """Add a document to the scheduling queue."""
        self.queue.append(metadata)
        self.queue.sort(key=lambda d: (-d.priority, d.size_bytes))

    def get_next_batch(self, max_batch_size: int = 10) -> List[DocumentMetadata]:
        """Get the next batch of documents to process."""
        batch = self.queue[:max_batch_size]
        self.queue = self.queue[max_batch_size:]
        return batch

    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return len(self.queue)


class DocumentPreValidator:
    """Pre-validates documents before full processing."""

    def __init__(self):
        """Initialize the pre-validator."""
        pass

    def validate(self, document_path: str) -> Dict[str, Any]:
        """
        Perform basic validation on a document.

        Args:
            document_path: Path to the document file

        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Basic validation checks
        try:
            import os

            if not os.path.exists(document_path):
                result["valid"] = False
                result["errors"].append(f"File not found: {document_path}")
            elif os.path.getsize(document_path) == 0:
                result["valid"] = False
                result["errors"].append("File is empty")
            elif os.path.getsize(document_path) > 100 * 1024 * 1024:
                result["warnings"].append("File is very large (>100MB)")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {str(e)}")

        return result


class ResourceMonitor:
    """Monitors system resources during batch processing."""

    def __init__(self):
        """Initialize the resource monitor."""
        self.start_time = time.time()
        self.documents_processed = 0

    def record_document_processed(self) -> None:
        """Record that a document was processed."""
        self.documents_processed += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        elapsed = time.time() - self.start_time
        throughput = self.documents_processed / elapsed if elapsed > 0 else 0

        return {
            "elapsed_seconds": elapsed,
            "documents_processed": self.documents_processed,
            "throughput_per_second": throughput,
            "throughput_per_hour": throughput * 3600,
        }


class CircuitBreakerWrapper:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 60.0):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # "closed", "open", "half-open"

    def call(self, func, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result or raises exception if circuit is open
        """
        if self.state == "open":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.timeout_seconds
            ):
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

            raise e


class ResultStreamer:
    """Streams processing results to clients."""

    def __init__(self):
        """Initialize the result streamer."""
        self.results: Dict[str, Any] = {}

    def add_result(self, doc_id: str, result: Any) -> None:
        """Add a processing result."""
        self.results[doc_id] = result

    def get_result(self, doc_id: str) -> Optional[Any]:
        """Get a processing result by document ID."""
        return self.results.get(doc_id)

    def get_all_results(self) -> Dict[str, Any]:
        """Get all processing results."""
        return self.results.copy()

    def clear_results(self) -> None:
        """Clear all results."""
        self.results.clear()
