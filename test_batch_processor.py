#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for batch_processor module."""

import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from batch_processor import (
    BatchMetrics,
    BatchProcessor,
    BatchWorker,
)


class TestBatchMetrics(unittest.TestCase):
    """Test BatchMetrics class."""
    
    def setUp(self):
        self.metrics = BatchMetrics()
    
    def test_record_document_success(self):
        """Test recording successful document processing."""
        self.metrics.record_document("success", 10.5, trace_id="test-trace-1")
        
        self.assertEqual(self.metrics.documents_processed_total, 1)
        self.assertEqual(self.metrics.documents_success, 1)
        self.assertEqual(self.metrics.documents_error, 0)
        self.assertEqual(len(self.metrics.processing_times), 1)
        self.assertEqual(self.metrics.processing_times[0], 10.5)
    
    def test_record_document_error(self):
        """Test recording failed document processing."""
        self.metrics.record_document("error", 5.0, "TimeoutError", "test-trace-2")
        
        self.assertEqual(self.metrics.documents_processed_total, 1)
        self.assertEqual(self.metrics.documents_success, 0)
        self.assertEqual(self.metrics.documents_error, 1)
        self.assertEqual(self.metrics.error_counts_by_category["TimeoutError"], 1)
    
    def test_record_stage_time(self):
        """Test recording stage processing time."""
        self.metrics.record_stage_time("embedding", 0.05, "trace-1")
        self.metrics.record_stage_time("embedding", 0.06, "trace-2")
        
        self.assertIn("embedding", self.metrics.stage_processing_times)
        self.assertEqual(len(self.metrics.stage_processing_times["embedding"]), 2)
    
    def test_record_worker_activity(self):
        """Test recording worker utilization."""
        self.metrics.record_worker_activity("worker-0", 50.0, 100.0)
        
        self.assertEqual(self.metrics.worker_busy_time["worker-0"], 50.0)
        self.assertEqual(self.metrics.worker_total_time["worker-0"], 100.0)
    
    def test_get_throughput_per_hour(self):
        """Test throughput calculation."""
        # Simulate processing documents over time
        self.metrics.start_time = time.time() - 3600  # 1 hour ago
        for i in range(170):
            self.metrics.record_document("success", 10.0, trace_id=f"trace-{i}")
        
        throughput = self.metrics.get_throughput_per_hour()
        self.assertGreaterEqual(throughput, 160)  # Allow some variance
    
    def test_get_p95_latency(self):
        """Test p95 latency calculation."""
        # Add latencies from 1 to 100 seconds
        for i in range(1, 101):
            self.metrics.record_document("success", float(i), trace_id=f"trace-{i}")
        
        p95 = self.metrics.get_p95_latency()
        self.assertAlmostEqual(p95, 95.0, delta=1.0)
    
    def test_get_worker_utilization(self):
        """Test worker utilization calculation."""
        self.metrics.record_worker_activity("worker-0", 75.0, 100.0)
        
        utilization = self.metrics.get_worker_utilization("worker-0")
        self.assertEqual(utilization, 75.0)
    
    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics format."""
        self.metrics.record_document("success", 15.0, trace_id="trace-1")
        self.metrics.record_document("error", 20.0, "ValueError", "trace-2")
        self.metrics.record_worker_activity("worker-0", 50.0, 100.0)
        self.metrics.record_queue_depth(42)
        
        metrics_text = self.metrics.get_prometheus_metrics()
        
        # Verify format
        self.assertIn("batch_documents_processed_total", metrics_text)
        self.assertIn("batch_throughput_per_hour", metrics_text)
        self.assertIn("worker_utilization_percentage", metrics_text)
        self.assertIn("queue_depth", metrics_text)
        self.assertIn('status="success"', metrics_text)
        self.assertIn('status="error"', metrics_text)
        self.assertIn('error_category="ValueError"', metrics_text)


class TestBatchWorker(unittest.TestCase):
    """Test BatchWorker class."""
    
    def setUp(self):
        self.metrics = BatchMetrics()
        self.worker = BatchWorker("test-worker", self.metrics)
    
    def test_worker_initialization(self):
        """Test worker initialization."""
        self.assertEqual(self.worker.worker_id, "test-worker")
        self.assertFalse(self.worker.is_running)
        self.assertEqual(self.worker.busy_time, 0.0)
    
    @patch("subprocess.run")
    def test_process_document_success(self, mock_subprocess):
        """Test successful document processing."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"success": true, "data": {}}',
            stderr=""
        )
        
        work_item = {
            "document_path": "/tmp/test.txt",
            "trace_id": "test-trace"
        }
        
        result = self.worker._process_document(work_item)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["trace_id"], "test-trace")
        self.assertIn("latency", result)
    
    @patch("subprocess.run")
    def test_process_document_error(self, mock_subprocess):
        """Test failed document processing."""
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Pipeline error"
        )
        
        work_item = {
            "document_path": "/tmp/test.txt",
            "trace_id": "test-trace-error"
        }
        
        result = self.worker._process_document(work_item)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_path = os.path.join(self.temp_dir, "metrics.prom")
        self.processor = BatchProcessor(num_workers=2, metrics_export_path=self.metrics_path)
    
    def tearDown(self):
        if self.processor.is_running:
            self.processor.stop()
        
        # Cleanup temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.num_workers, 2)
        self.assertIsInstance(self.processor.metrics, BatchMetrics)
        self.assertIsInstance(self.processor.work_queue, queue.Queue)
        self.assertIsInstance(self.processor.results_queue, queue.Queue)
    
    def test_start_stop(self):
        """Test processor start and stop."""
        self.processor.start()
        self.assertTrue(self.processor.is_running)
        self.assertEqual(len(self.processor.workers), 2)
        
        time.sleep(0.5)  # Let workers initialize
        
        self.processor.stop()
        self.assertFalse(self.processor.is_running)
    
    def test_submit_document(self):
        """Test document submission."""
        self.processor.start()
        
        trace_id = self.processor.submit_document("/tmp/test.txt")
        
        self.assertIsNotNone(trace_id)
        self.assertEqual(len(trace_id), 36)  # UUID length
        self.assertFalse(self.processor.work_queue.empty())
        
        self.processor.stop()
    
    def test_submit_batch(self):
        """Test batch submission."""
        self.processor.start()
        
        documents = ["/tmp/test1.txt", "/tmp/test2.txt", "/tmp/test3.txt"]
        trace_ids = self.processor.submit_batch(documents)
        
        self.assertEqual(len(trace_ids), 3)
        self.assertEqual(self.processor.work_queue.qsize(), 3)
        
        self.processor.stop()
    
    def test_metrics_export(self):
        """Test metrics file export."""
        self.processor.start()
        time.sleep(1.0)  # Wait for metrics export
        
        self.processor.submit_document("/tmp/test.txt")
        time.sleep(11.0)  # Wait for next export cycle
        
        self.processor.stop()
        
        # Check metrics file exists
        self.assertTrue(Path(self.metrics_path).exists())
        
        # Verify metrics content
        with open(self.metrics_path, 'r') as f:
            content = f.read()
            self.assertIn("batch_throughput_per_hour", content)


class TestIntegration(unittest.TestCase):
    """Integration tests for batch processing."""
    
    @patch("subprocess.run")
    def test_end_to_end_processing(self, mock_subprocess):
        """Test end-to-end document processing."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"success": true, "data": {"score": 0.85}}',
            stderr=""
        )
        
        temp_dir = tempfile.mkdtemp()
        metrics_path = os.path.join(temp_dir, "metrics.prom")
        
        processor = BatchProcessor(num_workers=2, metrics_export_path=metrics_path)
        processor.start()
        
        try:
            # Submit documents
            documents = [f"/tmp/test{i}.txt" for i in range(10)]
            trace_ids = processor.submit_batch(documents)
            
            # Wait for completion
            processor.wait_for_completion(timeout=30)
            
            # Check results
            results = processor.get_results(timeout=5)
            
            self.assertEqual(len(results), 10)
            self.assertEqual(len(trace_ids), 10)
            
            # Verify metrics
            self.assertGreater(processor.metrics.documents_processed_total, 0)
            
        finally:
            processor.stop()
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
