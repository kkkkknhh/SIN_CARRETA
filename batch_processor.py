#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor with Prometheus Metrics and Structured Logging
================================================================

Batch document processing system with:
- Prometheus metrics export (documents_processed_total, batch_throughput_per_hour, worker_utilization, queue_depth)
- Structured JSON logging with trace_id propagation
- SLA monitoring (170 docs/hr, 21.2s p95 latency)
- Worker pool management
- Queue-based document distribution
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from log_config import JsonFormatter

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Prometheus-compatible batch processing metrics."""
    
    documents_processed_total: int = 0
    documents_success: int = 0
    documents_error: int = 0
    error_counts_by_category: Dict[str, int] = field(default_factory=dict)
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    stage_processing_times: Dict[str, deque] = field(default_factory=dict)
    worker_busy_time: Dict[str, float] = field(default_factory=dict)
    worker_total_time: Dict[str, float] = field(default_factory=dict)
    queue_depths: deque = field(default_factory=lambda: deque(maxlen=100))
    start_time: float = field(default_factory=time.time)
    
    def record_document(self, status: str, latency: float, error_category: Optional[str] = None, trace_id: Optional[str] = None):
        """Record document processing completion."""
        self.documents_processed_total += 1
        if status == "success":
            self.documents_success += 1
        elif status == "error":
            self.documents_error += 1
            if error_category:
                self.error_counts_by_category[error_category] = self.error_counts_by_category.get(error_category, 0) + 1
        
        self.processing_times.append(latency)
        logger.info(json.dumps({
            "event": "document_processed",
            "trace_id": trace_id,
            "status": status,
            "latency_seconds": latency,
            "error_category": error_category,
            "total_processed": self.documents_processed_total
        }))
    
    def record_stage_time(self, stage: str, latency: float, trace_id: Optional[str] = None):
        """Record processing time for a specific stage."""
        if stage not in self.stage_processing_times:
            self.stage_processing_times[stage] = deque(maxlen=1000)
        self.stage_processing_times[stage].append(latency)
        
        logger.debug(json.dumps({
            "event": "stage_completed",
            "trace_id": trace_id,
            "stage": stage,
            "latency_seconds": latency
        }))
    
    def record_worker_activity(self, worker_id: str, busy_time: float, total_time: float):
        """Record worker utilization metrics."""
        self.worker_busy_time[worker_id] = busy_time
        self.worker_total_time[worker_id] = total_time
    
    def record_queue_depth(self, depth: int):
        """Record current queue depth."""
        self.queue_depths.append(depth)
    
    def get_throughput_per_hour(self) -> float:
        """Calculate documents per hour throughput."""
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        if elapsed_hours == 0:
            return 0.0
        return self.documents_processed_total / elapsed_hours
    
    def get_p95_latency(self) -> float:
        """Calculate p95 latency in seconds."""
        if not self.processing_times:
            return 0.0
        sorted_times = sorted(self.processing_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]
    
    def get_worker_utilization(self, worker_id: str) -> float:
        """Calculate worker utilization percentage."""
        if worker_id not in self.worker_total_time or self.worker_total_time[worker_id] == 0:
            return 0.0
        busy = self.worker_busy_time.get(worker_id, 0.0)
        total = self.worker_total_time[worker_id]
        return (busy / total) * 100.0
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counter: documents_processed_total
        lines.append("# HELP batch_documents_processed_total Total documents processed")
        lines.append("# TYPE batch_documents_processed_total counter")
        lines.append(f'batch_documents_processed_total{{status="success"}} {self.documents_success}')
        lines.append(f'batch_documents_processed_total{{status="error"}} {self.documents_error}')
        
        # Error breakdown
        for category, count in self.error_counts_by_category.items():
            lines.append(f'batch_documents_processed_total{{status="error",error_category="{category}"}} {count}')
        
        # Gauge: batch_throughput_per_hour
        lines.append("# HELP batch_throughput_per_hour Current batch processing throughput")
        lines.append("# TYPE batch_throughput_per_hour gauge")
        lines.append(f"batch_throughput_per_hour {self.get_throughput_per_hour():.2f}")
        
        # Gauge: worker_utilization_percentage
        lines.append("# HELP worker_utilization_percentage Worker utilization percentage")
        lines.append("# TYPE worker_utilization_percentage gauge")
        for worker_id in self.worker_total_time.keys():
            utilization = self.get_worker_utilization(worker_id)
            lines.append(f'worker_utilization_percentage{{worker_id="{worker_id}"}} {utilization:.2f}')
        
        # Gauge: queue_depth
        lines.append("# HELP queue_depth Current processing queue depth")
        lines.append("# TYPE queue_depth gauge")
        current_depth = self.queue_depths[-1] if self.queue_depths else 0
        lines.append(f'queue_depth{{queue_name="main"}} {current_depth}')
        
        # Histogram: batch_document_processing_latency_seconds
        lines.append("# HELP batch_document_processing_latency_seconds Document processing latency distribution")
        lines.append("# TYPE batch_document_processing_latency_seconds summary")
        if self.processing_times:
            p50 = sorted(self.processing_times)[int(len(self.processing_times) * 0.50)]
            p95 = self.get_p95_latency()
            p99 = sorted(self.processing_times)[int(len(self.processing_times) * 0.99)] if len(self.processing_times) > 0 else 0
            lines.append(f'batch_document_processing_latency_seconds{{quantile="0.5"}} {p50:.3f}')
            lines.append(f'batch_document_processing_latency_seconds{{quantile="0.95"}} {p95:.3f}')
            lines.append(f'batch_document_processing_latency_seconds{{quantile="0.99"}} {p99:.3f}')
            lines.append(f'batch_document_processing_latency_seconds_count {len(self.processing_times)}')
            lines.append(f'batch_document_processing_latency_seconds_sum {sum(self.processing_times):.3f}')
        
        # Stage processing times
        lines.append("# HELP batch_stage_processing_time_seconds Processing time by stage")
        lines.append("# TYPE batch_stage_processing_time_seconds summary")
        for stage, times in self.stage_processing_times.items():
            if times:
                sorted_times = sorted(times)
                p95_stage = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
                lines.append(f'batch_stage_processing_time_seconds{{stage="{stage}",quantile="0.95"}} {p95_stage:.3f}')
        
        return "\n".join(lines) + "\n"


class BatchWorker:
    """Worker process for batch document processing."""
    
    def __init__(self, worker_id: str, metrics: BatchMetrics):
        self.worker_id = worker_id
        self.metrics = metrics
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time = 0.0
        self.busy_time = 0.0
        
        logger.info(json.dumps({
            "event": "worker_initialized",
            "worker_id": worker_id
        }))
    
    def start(self, work_queue: queue.Queue, results_queue: queue.Queue):
        """Start worker thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, args=(work_queue, results_queue))
        self.thread.daemon = True
        self.thread.start()
        self.start_time = time.time()
        
        logger.info(json.dumps({
            "event": "worker_started",
            "worker_id": self.worker_id
        }))
    
    def stop(self):
        """Stop worker thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        
        # Record final utilization
        total_time = time.time() - self.start_time
        self.metrics.record_worker_activity(self.worker_id, self.busy_time, total_time)
        
        logger.info(json.dumps({
            "event": "worker_stopped",
            "worker_id": self.worker_id,
            "total_time": total_time,
            "busy_time": self.busy_time,
            "utilization": (self.busy_time / total_time * 100) if total_time > 0 else 0
        }))
    
    def _process_loop(self, work_queue: queue.Queue, results_queue: queue.Queue):
        """Main worker processing loop."""
        while self.is_running:
            try:
                # Get work item with timeout
                try:
                    work_item = work_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if work_item is None:  # Poison pill
                    break
                
                # Process document
                task_start = time.time()
                result = self._process_document(work_item)
                task_duration = time.time() - task_start
                
                self.busy_time += task_duration
                
                # Put result
                results_queue.put(result)
                work_queue.task_done()
                
                # Update worker utilization
                total_time = time.time() - self.start_time
                self.metrics.record_worker_activity(self.worker_id, self.busy_time, total_time)
                
            except Exception as e:
                logger.error(json.dumps({
                    "event": "worker_error",
                    "worker_id": self.worker_id,
                    "error": str(e)
                }))
    
    def _process_document(self, work_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document through the pipeline."""
        document_path = work_item["document_path"]
        trace_id = work_item.get("trace_id") or str(uuid.uuid4())
        
        start_time = time.time()
        
        logger.info(json.dumps({
            "event": "document_processing_started",
            "trace_id": trace_id,
            "worker_id": self.worker_id,
            "document_path": document_path
        }))
        
        try:
            # Invoke unified_evaluation_pipeline with trace_id propagation
            result = self._invoke_pipeline(document_path, trace_id)
            
            latency = time.time() - start_time
            status = "success" if result.get("success") else "error"
            error_category = result.get("error_category")
            
            self.metrics.record_document(status, latency, error_category, trace_id)
            
            logger.info(json.dumps({
                "event": "document_processing_completed",
                "trace_id": trace_id,
                "worker_id": self.worker_id,
                "status": status,
                "latency_seconds": latency
            }))
            
            return {
                "trace_id": trace_id,
                "document_path": document_path,
                "status": status,
                "latency": latency,
                "result": result
            }
            
        except Exception as e:
            latency = time.time() - start_time
            error_category = type(e).__name__
            
            self.metrics.record_document("error", latency, error_category, trace_id)
            
            logger.error(json.dumps({
                "event": "document_processing_failed",
                "trace_id": trace_id,
                "worker_id": self.worker_id,
                "error": str(e),
                "error_category": error_category,
                "latency_seconds": latency
            }))
            
            return {
                "trace_id": trace_id,
                "document_path": document_path,
                "status": "error",
                "error": str(e),
                "error_category": error_category,
                "latency": latency
            }
    
    def _invoke_pipeline(self, document_path: str, trace_id: str) -> Dict[str, Any]:
        """Invoke unified_evaluation_pipeline with trace_id propagation."""
        # Stage 1: Unified Evaluation Pipeline
        stage_start = time.time()
        
        env = os.environ.copy()
        env["TRACE_ID"] = trace_id
        
        try:
            result = subprocess.run(
                ["python3", "unified_evaluation_pipeline.py", "--document", document_path, "--trace-id", trace_id],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )
            
            stage_latency = time.time() - stage_start
            self.metrics.record_stage_time("unified_evaluation", stage_latency, trace_id)
            
            if result.returncode != 0:
                raise RuntimeError(f"Pipeline failed: {result.stderr}")
            
            pipeline_result = json.loads(result.stdout) if result.stdout else {}
            
            # Stage 2: MINIMINIMOON Orchestrator (if needed)
            if pipeline_result.get("requires_orchestrator"):
                stage_start = time.time()
                
                orchestrator_result = subprocess.run(
                    ["python3", "miniminimoon_orchestrator.py", "--trace-id", trace_id, "--input", document_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env
                )
                
                stage_latency = time.time() - stage_start
                self.metrics.record_stage_time("miniminimoon_orchestrator", stage_latency, trace_id)
                
                if orchestrator_result.returncode == 0 and orchestrator_result.stdout:
                    orchestrator_data = json.loads(orchestrator_result.stdout)
                    pipeline_result["orchestrator_result"] = orchestrator_data
            
            return {"success": True, "data": pipeline_result}
            
        except subprocess.TimeoutExpired:
            stage_latency = time.time() - stage_start
            self.metrics.record_stage_time("unified_evaluation", stage_latency, trace_id)
            return {"success": False, "error": "Pipeline timeout", "error_category": "TimeoutError"}
        except Exception as e:
            stage_latency = time.time() - stage_start
            self.metrics.record_stage_time("unified_evaluation", stage_latency, trace_id)
            return {"success": False, "error": str(e), "error_category": type(e).__name__}


class BatchProcessor:
    """Main batch processor with worker pool and metrics export."""
    
    def __init__(self, num_workers: int = 4, metrics_export_path: str = "/tmp/batch_metrics.prom"):
        self.num_workers = num_workers
        self.metrics = BatchMetrics()
        self.metrics_export_path = metrics_export_path
        self.workers: List[BatchWorker] = []
        self.work_queue: queue.Queue = queue.Queue()
        self.results_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.metrics_thread: Optional[threading.Thread] = None
        
        logger.info(json.dumps({
            "event": "batch_processor_initialized",
            "num_workers": num_workers,
            "metrics_export_path": metrics_export_path
        }))
    
    def start(self):
        """Start batch processor and workers."""
        self.is_running = True
        
        # Start workers
        for i in range(self.num_workers):
            worker = BatchWorker(f"worker-{i}", self.metrics)
            worker.start(self.work_queue, self.results_queue)
            self.workers.append(worker)
        
        # Start metrics export thread
        self.metrics_thread = threading.Thread(target=self._export_metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        logger.info(json.dumps({
            "event": "batch_processor_started",
            "num_workers": len(self.workers)
        }))
    
    def stop(self):
        """Stop batch processor and workers."""
        self.is_running = False
        
        # Send poison pills to workers
        for _ in self.workers:
            self.work_queue.put(None)
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        # Wait for metrics thread
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5.0)
        
        logger.info(json.dumps({
            "event": "batch_processor_stopped",
            "total_processed": self.metrics.documents_processed_total,
            "throughput_per_hour": self.metrics.get_throughput_per_hour(),
            "p95_latency": self.metrics.get_p95_latency()
        }))
    
    def submit_document(self, document_path: str, trace_id: Optional[str] = None) -> str:
        """Submit a document for processing."""
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        work_item = {
            "document_path": document_path,
            "trace_id": trace_id,
            "submitted_at": time.time()
        }
        
        self.work_queue.put(work_item)
        self.metrics.record_queue_depth(self.work_queue.qsize())
        
        logger.info(json.dumps({
            "event": "document_submitted",
            "trace_id": trace_id,
            "document_path": document_path,
            "queue_depth": self.work_queue.qsize()
        }))
        
        return trace_id
    
    def submit_batch(self, document_paths: List[str]) -> List[str]:
        """Submit multiple documents for processing."""
        trace_ids = []
        for doc_path in document_paths:
            trace_id = self.submit_document(doc_path)
            trace_ids.append(trace_id)
        return trace_ids
    
    def get_results(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all completed results."""
        results = []
        deadline = time.time() + timeout if timeout else None
        
        while True:
            try:
                remaining = None
                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                
                result = self.results_queue.get(timeout=remaining or 0.1)
                results.append(result)
            except queue.Empty:
                if self.work_queue.empty() and self.results_queue.empty():
                    break
        
        return results
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all submitted documents to complete."""
        self.work_queue.join()
    
    def _export_metrics_loop(self):
        """Periodically export metrics to file."""
        while self.is_running:
            try:
                metrics_text = self.metrics.get_prometheus_metrics()
                
                # Write to temp file then rename (atomic)
                temp_path = self.metrics_export_path + ".tmp"
                with open(temp_path, 'w') as f:
                    f.write(metrics_text)
                os.rename(temp_path, self.metrics_export_path)
                
                logger.debug(json.dumps({
                    "event": "metrics_exported",
                    "path": self.metrics_export_path,
                    "throughput": self.metrics.get_throughput_per_hour(),
                    "p95_latency": self.metrics.get_p95_latency()
                }))
                
            except Exception as e:
                logger.error(json.dumps({
                    "event": "metrics_export_failed",
                    "error": str(e)
                }))
            
            time.sleep(10)  # Export every 10 seconds


def main():
    """CLI entry point for batch processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch document processor with metrics")
    parser.add_argument("--documents", nargs="+", help="Document paths to process")
    parser.add_argument("--directory", help="Directory containing documents")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--metrics-path", default="/tmp/batch_metrics.prom", help="Metrics export path")
    args = parser.parse_args()
    
    # Collect documents
    documents = []
    if args.documents:
        documents.extend(args.documents)
    if args.directory:
        doc_dir = Path(args.directory)
        documents.extend([str(p) for p in doc_dir.glob("*.txt")])
    
    if not documents:
        logger.error("No documents specified")
        return 1
    
    # Initialize processor
    processor = BatchProcessor(num_workers=args.workers, metrics_export_path=args.metrics_path)
    processor.start()
    
    try:
        # Submit batch
        logger.info(json.dumps({
            "event": "batch_started",
            "num_documents": len(documents)
        }))
        
        trace_ids = processor.submit_batch(documents)
        
        # Wait for completion
        processor.wait_for_completion(timeout=3600)
        
        # Get results
        results = processor.get_results()
        
        logger.info(json.dumps({
            "event": "batch_completed",
            "total_documents": len(documents),
            "processed": len(results),
            "throughput_per_hour": processor.metrics.get_throughput_per_hour(),
            "p95_latency": processor.metrics.get_p95_latency()
        }))
        
        # Export final metrics
        print(processor.metrics.get_prometheus_metrics())
        
        return 0
        
    finally:
        processor.stop()


if __name__ == "__main__":
    exit(main())
