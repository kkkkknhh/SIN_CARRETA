#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Optimization and Resource Management
==========================================

Provides comprehensive batch processing optimization for document evaluation:

1. DocumentScheduler: Groups documents by complexity (file size, page count)
2. CircuitBreakerWrapper: Wraps pipeline stages with failure thresholds
3. ResourceMonitor: Tracks CPU/memory/GPU metrics and adapts concurrency
4. DocumentPreValidator: Fast-fail checks on PDF format, size, encoding
5. ResultStreamer: Yields incremental results with optional Redis integration
"""

import logging
import psutil
import time
import threading
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Iterator, Tuple
from pathlib import Path
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis not available, ResultStreamer will operate without Redis")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PYPDF2_AVAILABLE = True
    except ImportError:
        PYPDF2_AVAILABLE = False
        logging.warning("PDF libraries not available, DocumentPreValidator will have limited validation")

from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError

logger = logging.getLogger(__name__)


# ============================================================================
# DOCUMENT SCHEDULER
# ============================================================================

class DocumentComplexity(Enum):
    """Document complexity levels based on size and page count"""
    SIMPLE = "simple"           # < 1MB, < 10 pages
    MEDIUM = "medium"           # 1-5MB, 10-50 pages
    COMPLEX = "complex"         # 5-20MB, 50-200 pages
    VERY_COMPLEX = "very_complex"  # > 20MB, > 200 pages


@dataclass
class ComplexityMetrics:
    """Metrics for document complexity estimation"""
    file_size_mb: float
    page_count: Optional[int] = None
    estimated_processing_time: float = 0.0
    complexity: DocumentComplexity = DocumentComplexity.SIMPLE


@dataclass
class DocumentBatch:
    """Batch of documents grouped by complexity"""
    complexity: DocumentComplexity
    documents: List[str] = field(default_factory=list)
    estimated_time: float = 0.0
    priority: int = 0


class DocumentScheduler:
    """
    Groups documents by estimated complexity for optimized batch processing.
    """

    def __init__(self):
        self.complexity_cache: Dict[str, ComplexityMetrics] = {}
        self.lock = threading.Lock()

    def estimate_complexity(
        self,
        document_path: str,
        page_count: Optional[int] = None
    ) -> ComplexityMetrics:
        """
        Estimate document complexity based on file size and page count.

        Args:
            document_path: Path to document
            page_count: Optional page count (if already known)

        Returns:
            ComplexityMetrics with estimated complexity
        """
        with self.lock:
            if document_path in self.complexity_cache:
                return self.complexity_cache[document_path]

        try:
            file_size_mb = Path(document_path).stat().st_size / (1024 * 1024)

            if page_count is None and PYPDF2_AVAILABLE:
                page_count = self._estimate_page_count(document_path)

            complexity = self._classify_complexity(file_size_mb, page_count)
            processing_time = self._estimate_processing_time(file_size_mb, page_count)

            metrics = ComplexityMetrics(
                file_size_mb=file_size_mb,
                page_count=page_count,
                complexity=complexity,
                estimated_processing_time=processing_time
            )

            with self.lock:
                self.complexity_cache[document_path] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error estimating complexity for {document_path}: {e}")
            return ComplexityMetrics(
                file_size_mb=0.0,
                complexity=DocumentComplexity.SIMPLE
            )

    def _estimate_page_count(self, document_path: str) -> Optional[int]:
        """Estimate page count from PDF"""
        try:
            with open(document_path, 'rb') as f:
                pdf = PdfReader(f)
                return len(pdf.pages)
        except Exception as e:
            logger.debug(f"Could not extract page count: {e}")
            return None

    def _classify_complexity(
        self,
        file_size_mb: float,
        page_count: Optional[int]
    ) -> DocumentComplexity:
        """Classify document complexity"""
        if file_size_mb > 20 or (page_count and page_count > 200):
            return DocumentComplexity.VERY_COMPLEX
        elif file_size_mb > 5 or (page_count and page_count > 50):
            return DocumentComplexity.COMPLEX
        elif file_size_mb > 1 or (page_count and page_count > 10):
            return DocumentComplexity.MEDIUM
        else:
            return DocumentComplexity.SIMPLE

    def _estimate_processing_time(
        self,
        file_size_mb: float,
        page_count: Optional[int]
    ) -> float:
        """Estimate processing time in seconds"""
        base_time = 5.0
        size_factor = file_size_mb * 2.0
        page_factor = (page_count or 10) * 0.5
        return base_time + size_factor + page_factor

    def group_by_complexity(
        self,
        document_paths: List[str],
        batch_size: int = 10
    ) -> List[DocumentBatch]:
        """
        Group documents into batches by complexity.

        Args:
            document_paths: List of document paths
            batch_size: Maximum documents per batch

        Returns:
            List of DocumentBatch objects
        """
        complexity_groups: Dict[DocumentComplexity, List[str]] = {
            DocumentComplexity.SIMPLE: [],
            DocumentComplexity.MEDIUM: [],
            DocumentComplexity.COMPLEX: [],
            DocumentComplexity.VERY_COMPLEX: []
        }

        for doc_path in document_paths:
            metrics = self.estimate_complexity(doc_path)
            complexity_groups[metrics.complexity].append(doc_path)

        batches = []
        for complexity in DocumentComplexity:
            docs = complexity_groups[complexity]
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                estimated_time = sum(
                    self.estimate_complexity(doc).estimated_processing_time
                    for doc in batch_docs
                )
                batches.append(DocumentBatch(
                    complexity=complexity,
                    documents=batch_docs,
                    estimated_time=estimated_time,
                    priority=self._get_priority(complexity)
                ))

        batches.sort(key=lambda b: (b.priority, -b.estimated_time))
        return batches

    def _get_priority(self, complexity: DocumentComplexity) -> int:
        """Get priority for complexity level (lower is higher priority)"""
        priority_map = {
            DocumentComplexity.SIMPLE: 1,
            DocumentComplexity.MEDIUM: 2,
            DocumentComplexity.COMPLEX: 3,
            DocumentComplexity.VERY_COMPLEX: 4
        }
        return priority_map[complexity]


# ============================================================================
# CIRCUIT BREAKER WRAPPER
# ============================================================================

@dataclass
class StageFailureConfig:
    """Configuration for stage failure thresholds"""
    consecutive_failures: int = 5
    time_window_seconds: float = 60.0
    failure_rate_threshold: float = 0.5


class CircuitBreakerWrapper:
    """
    Wraps unified_evaluation_pipeline stages with circuit breakers.
    """

    def __init__(self):
        self.stage_circuits: Dict[str, CircuitBreaker] = {}
        self.stage_configs: Dict[str, StageFailureConfig] = {}
        self.lock = threading.Lock()

    def register_stage(
        self,
        stage_name: str,
        config: Optional[StageFailureConfig] = None
    ):
        """Register a pipeline stage with circuit breaker"""
        if config is None:
            config = StageFailureConfig()

        circuit_config = CircuitBreakerConfig(
            failure_threshold=config.consecutive_failures,
            timeout_seconds=config.time_window_seconds
        )

        with self.lock:
            self.stage_circuits[stage_name] = CircuitBreaker(
                name=f"stage_{stage_name}",
                config=circuit_config
            )
            self.stage_configs[stage_name] = config

        logger.info(f"Registered circuit breaker for stage: {stage_name}")

    def wrap_stage(
        self,
        stage_name: str,
        stage_func: Callable,
        fallback: Optional[Callable] = None
    ) -> Callable:
        """
        Wrap a pipeline stage with circuit breaker protection.

        Args:
            stage_name: Name of the pipeline stage
            stage_func: Function to wrap
            fallback: Optional fallback function

        Returns:
            Wrapped function with circuit breaker protection
        """
        if stage_name not in self.stage_circuits:
            self.register_stage(stage_name)

        circuit = self.stage_circuits[stage_name]

        def wrapped(*args, **kwargs):
            try:
                return circuit.call(stage_func, *args, **kwargs)
            except CircuitBreakerError as e:
                logger.error(f"Circuit breaker open for stage {stage_name}: {e}")
                if fallback:
                    logger.info(f"Using fallback for stage {stage_name}")
                    return fallback(*args, **kwargs)
                raise

        return wrapped

    def check_thresholds(self, stage_name: str) -> Dict[str, Any]:
        """Check if stage is within acceptable thresholds"""
        if stage_name not in self.stage_circuits:
            return {"status": "unknown", "stage": stage_name}

        circuit = self.stage_circuits[stage_name]
        health = circuit.get_health_status()

        return {
            "stage": stage_name,
            "state": health["state"],
            "success_rate": health["metrics"]["success_rate"],
            "threshold_breached": health["state"] != "closed"
        }

    def get_all_stage_health(self) -> Dict[str, Any]:
        """Get health status for all registered stages"""
        return {
            stage_name: self.check_thresholds(stage_name)
            for stage_name in self.stage_circuits.keys()
        }


# ============================================================================
# RESOURCE MONITOR
# ============================================================================

@dataclass
class ResourceMetrics:
    """Current resource usage metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceThresholds:
    """Thresholds for resource adaptation"""
    cpu_high: float = 80.0
    cpu_critical: float = 95.0
    memory_high: float = 75.0
    memory_critical: float = 90.0
    gpu_high: float = 85.0
    gpu_critical: float = 95.0


class ResourceMonitor:
    """
    Tracks CPU/memory/GPU metrics and adapts batch concurrency dynamically.
    """

    def __init__(self, thresholds: Optional[ResourceThresholds] = None):
        self.thresholds = thresholds or ResourceThresholds()
        self.metrics_history: List[ResourceMetrics] = []
        self.current_concurrency = 4
        self.min_concurrency = 1
        self.max_concurrency = 16
        self.lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def track_metrics(self) -> ResourceMetrics:
        """Track current resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        gpu_percent = None
        gpu_memory_percent = None
        try:
            gpu_percent, gpu_memory_percent = self._get_gpu_metrics()
        except Exception as e:
            logger.debug(f"GPU metrics not available: {e}")

        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent
        )

        with self.lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)

        return metrics

    def _get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU metrics if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_percent = utilization.gpu
            gpu_memory_percent = (memory_info.used / memory_info.total) * 100
            
            pynvml.nvmlShutdown()
            return gpu_percent, gpu_memory_percent
        except Exception:
            return None, None

    def adapt_concurrency(self) -> int:
        """
        Adapt batch concurrency based on current resource usage.

        Returns:
            Recommended concurrency level
        """
        metrics = self.track_metrics()

        cpu_critical = metrics.cpu_percent >= self.thresholds.cpu_critical
        memory_critical = metrics.memory_percent >= self.thresholds.memory_critical
        gpu_critical = (metrics.gpu_percent is not None and 
                       metrics.gpu_percent >= self.thresholds.gpu_critical)

        cpu_high = metrics.cpu_percent >= self.thresholds.cpu_high
        memory_high = metrics.memory_percent >= self.thresholds.memory_high
        gpu_high = (metrics.gpu_percent is not None and 
                   metrics.gpu_percent >= self.thresholds.gpu_high)

        with self.lock:
            if cpu_critical or memory_critical or gpu_critical:
                self.current_concurrency = max(
                    self.min_concurrency,
                    self.current_concurrency - 2
                )
                logger.warning(
                    f"Critical resource usage detected, reducing concurrency to {self.current_concurrency}"
                )
            elif cpu_high or memory_high or gpu_high:
                self.current_concurrency = max(
                    self.min_concurrency,
                    self.current_concurrency - 1
                )
                logger.info(
                    f"High resource usage detected, reducing concurrency to {self.current_concurrency}"
                )
            elif (metrics.cpu_percent < 50 and 
                  metrics.memory_percent < 50 and
                  (metrics.gpu_percent is None or metrics.gpu_percent < 50)):
                self.current_concurrency = min(
                    self.max_concurrency,
                    self.current_concurrency + 1
                )
                logger.info(
                    f"Resources available, increasing concurrency to {self.current_concurrency}"
                )

            return self.current_concurrency

    def start_monitoring(self, interval: float = 5.0):
        """Start background resource monitoring"""
        if self.monitoring:
            logger.warning("Resource monitoring already started")
            return

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    self.adapt_concurrency()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started resource monitoring (interval={interval}s)")

    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped resource monitoring")

    def get_current_metrics(self) -> ResourceMetrics:
        """Get most recent metrics"""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return self.track_metrics()

    def get_average_metrics(self, window: int = 10) -> Dict[str, float]:
        """Get average metrics over recent window"""
        with self.lock:
            recent = self.metrics_history[-window:]

        if not recent:
            return {}

        return {
            "cpu_percent": sum(m.cpu_percent for m in recent) / len(recent),
            "memory_percent": sum(m.memory_percent for m in recent) / len(recent),
            "gpu_percent": sum(m.gpu_percent for m in recent if m.gpu_percent) / 
                          len([m for m in recent if m.gpu_percent]) if any(m.gpu_percent for m in recent) else None
        }


# ============================================================================
# DOCUMENT PRE-VALIDATOR
# ============================================================================

@dataclass
class ValidationError:
    """Structured validation error"""
    error_type: str
    message: str
    document_path: str
    details: Optional[Dict[str, Any]] = None


class DocumentPreValidator:
    """
    Performs fast-fail validation checks before enqueueing documents.
    """

    def __init__(
        self,
        max_size_mb: float = 100.0,
        allowed_encodings: Optional[List[str]] = None
    ):
        self.max_size_mb = max_size_mb
        self.allowed_encodings = allowed_encodings or ['utf-8', 'latin-1', 'cp1252']

    def validate(self, document_path: str) -> Tuple[bool, Optional[ValidationError]]:
        """
        Run all validation checks on document.

        Args:
            document_path: Path to document

        Returns:
            Tuple of (is_valid, error)
        """
        valid, error = self.validate_pdf(document_path)
        if not valid:
            return False, error

        valid, error = self.validate_size(document_path)
        if not valid:
            return False, error

        valid, error = self.validate_encoding(document_path)
        if not valid:
            return False, error

        return True, None

    def validate_pdf(self, document_path: str) -> Tuple[bool, Optional[ValidationError]]:
        """Validate PDF format and readability"""
        if not Path(document_path).exists():
            return False, ValidationError(
                error_type="file_not_found",
                message=f"Document not found: {document_path}",
                document_path=document_path
            )

        if not document_path.lower().endswith('.pdf'):
            return False, ValidationError(
                error_type="invalid_format",
                message=f"Document is not a PDF: {document_path}",
                document_path=document_path
            )

        if not PYPDF2_AVAILABLE:
            logger.debug("PDF validation libraries not available, skipping detailed validation")
            return True, None

        try:
            with open(document_path, 'rb') as f:
                pdf = PdfReader(f)
                if len(pdf.pages) == 0:
                    return False, ValidationError(
                        error_type="empty_pdf",
                        message="PDF contains no pages",
                        document_path=document_path
                    )
            return True, None
        except Exception as e:
            return False, ValidationError(
                error_type="malformed_pdf",
                message=f"PDF validation failed: {str(e)}",
                document_path=document_path,
                details={"exception": str(e)}
            )

    def validate_size(self, document_path: str) -> Tuple[bool, Optional[ValidationError]]:
        """Validate document size is within limits"""
        try:
            size_mb = Path(document_path).stat().st_size / (1024 * 1024)
            if size_mb > self.max_size_mb:
                return False, ValidationError(
                    error_type="size_exceeded",
                    message=f"Document size {size_mb:.2f}MB exceeds limit {self.max_size_mb}MB",
                    document_path=document_path,
                    details={"size_mb": size_mb, "limit_mb": self.max_size_mb}
                )
            return True, None
        except Exception as e:
            return False, ValidationError(
                error_type="size_check_failed",
                message=f"Could not check document size: {str(e)}",
                document_path=document_path,
                details={"exception": str(e)}
            )

    def validate_encoding(self, document_path: str) -> Tuple[bool, Optional[ValidationError]]:
        """Validate document encoding"""
        if not PYPDF2_AVAILABLE:
            logger.debug("PDF libraries not available, skipping encoding validation")
            return True, None

        try:
            with open(document_path, 'rb') as f:
                pdf = PdfReader(f)
                page = pdf.pages[0]
                text = page.extract_text()
                
                if text:
                    for encoding in self.allowed_encodings:
                        try:
                            text.encode(encoding)
                            return True, None
                        except UnicodeEncodeError:
                            continue
                    
                    return False, ValidationError(
                        error_type="encoding_invalid",
                        message=f"Document encoding not in allowed list: {self.allowed_encodings}",
                        document_path=document_path
                    )
                
                return True, None
        except Exception as e:
            logger.warning(f"Encoding validation failed: {e}")
            return True, None


# ============================================================================
# RESULT STREAMER
# ============================================================================

class ResultStreamer:
    """
    Yields evaluation results incrementally as they complete.
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        self.redis_client = None
        if REDIS_AVAILABLE and redis_host:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                self.redis_client = None

    def stream_results(
        self,
        batch_id: str,
        documents: List[str],
        evaluation_func: Callable[[str], Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream evaluation results as they complete.

        Args:
            batch_id: Unique batch identifier
            documents: List of document paths to evaluate
            evaluation_func: Function that evaluates a single document

        Yields:
            Evaluation results as they complete
        """
        total_docs = len(documents)
        completed = 0

        for doc_path in documents:
            try:
                start_time = time.time()
                result = evaluation_func(doc_path)
                execution_time = time.time() - start_time

                result_with_meta = {
                    "batch_id": batch_id,
                    "document_path": doc_path,
                    "result": result,
                    "completed": completed + 1,
                    "total": total_docs,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }

                self._update_redis(batch_id, doc_path, result_with_meta)

                completed += 1
                yield result_with_meta

            except Exception as e:
                logger.error(f"Error evaluating document {doc_path}: {e}")
                error_result = {
                    "batch_id": batch_id,
                    "document_path": doc_path,
                    "error": str(e),
                    "completed": completed + 1,
                    "total": total_docs,
                    "timestamp": datetime.now().isoformat()
                }
                
                self._update_redis(batch_id, doc_path, error_result)
                completed += 1
                yield error_result

    def _update_redis(self, batch_id: str, doc_path: str, result: Dict[str, Any]):
        """Update Redis with partial result"""
        if not self.redis_client:
            return

        try:
            doc_key = f"result:{batch_id}:{Path(doc_path).stem}"
            self.redis_client.set(doc_key, json.dumps(result))
            
            self.redis_client.sadd(f"batch:{batch_id}:documents", doc_path)
            
            self.redis_client.set(
                f"batch:{batch_id}:progress",
                json.dumps({
                    "completed": result.get("completed", 0),
                    "total": result.get("total", 0),
                    "last_update": result.get("timestamp")
                })
            )
            
            self.redis_client.expire(doc_key, 86400)
            
        except Exception as e:
            logger.error(f"Error updating Redis: {e}")

    def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get current batch progress from Redis"""
        if not self.redis_client:
            return None

        try:
            progress = self.redis_client.get(f"batch:{batch_id}:progress")
            if progress:
                return json.loads(progress)
        except Exception as e:
            logger.error(f"Error getting batch progress: {e}")

        return None

    def get_completed_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all completed results for a batch from Redis"""
        if not self.redis_client:
            return []

        try:
            doc_paths = self.redis_client.smembers(f"batch:{batch_id}:documents")
            results = []
            
            for doc_path in doc_paths:
                doc_key = f"result:{batch_id}:{Path(doc_path).stem}"
                result_json = self.redis_client.get(doc_key)
                if result_json:
                    results.append(json.loads(result_json))
            
            return results
        except Exception as e:
            logger.error(f"Error getting completed results: {e}")
            return []


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BATCH OPTIMIZER TESTS")
    print("="*80 + "\n")

    logging.basicConfig(level=logging.INFO)

    # Test DocumentScheduler
    print("1. Testing DocumentScheduler...")
    scheduler = DocumentScheduler()
    
    test_docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    for doc in test_docs:
        Path(doc).touch()
        Path(doc).write_bytes(b"PDF" * (1024 * 1024))
    
    batches = scheduler.group_by_complexity(test_docs, batch_size=2)
    print(f"   Created {len(batches)} batches")
    
    for doc in test_docs:
        Path(doc).unlink()

    # Test CircuitBreakerWrapper
    print("\n2. Testing CircuitBreakerWrapper...")
    wrapper = CircuitBreakerWrapper()
    wrapper.register_stage("test_stage")
    
    def test_func():
        return "success"
    
    wrapped = wrapper.wrap_stage("test_stage", test_func)
    result = wrapped()
    print(f"   Wrapped function result: {result}")

    # Test ResourceMonitor
    print("\n3. Testing ResourceMonitor...")
    monitor = ResourceMonitor()
    metrics = monitor.track_metrics()
    print(f"   CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%")
    concurrency = monitor.adapt_concurrency()
    print(f"   Recommended concurrency: {concurrency}")

    # Test DocumentPreValidator
    print("\n4. Testing DocumentPreValidator...")
    validator = DocumentPreValidator(max_size_mb=10.0)
    
    test_pdf = Path("test.pdf")
    test_pdf.write_bytes(b"%PDF-1.4\n" + b"test" * 100)
    
    valid, error = validator.validate(str(test_pdf))
    print(f"   Validation result: {'✅ VALID' if valid else '❌ INVALID'}")
    if error:
        print(f"   Error: {error.message}")
    
    test_pdf.unlink()

    # Test ResultStreamer
    print("\n5. Testing ResultStreamer...")
    streamer = ResultStreamer()
    
    def mock_eval(doc):
        time.sleep(0.1)
        return {"score": 0.8, "document": doc}
    
    test_docs = ["doc1.pdf", "doc2.pdf"]
    results = list(streamer.stream_results("batch001", test_docs, mock_eval))
    print(f"   Streamed {len(results)} results")

    print("\n" + "="*80)
    print("✅ All batch optimizer tests completed")
    print("="*80 + "\n")
