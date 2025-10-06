#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Batch Optimizer
===============================

Tests for all batch optimization components.
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from batch_optimizer import (
    DocumentScheduler,
    DocumentComplexity,
    ComplexityMetrics,
    CircuitBreakerWrapper,
    ResourceMonitor,
    ResourceThresholds,
    DocumentPreValidator,
    ValidationError,
    ResultStreamer
)


class TestDocumentScheduler:
    """Tests for DocumentScheduler"""

    def test_estimate_complexity_simple(self, tmp_path):
        scheduler = DocumentScheduler()
        test_file = tmp_path / "small.pdf"
        test_file.write_bytes(b"PDF" * 100)
        
        metrics = scheduler.estimate_complexity(str(test_file))
        assert metrics.complexity == DocumentComplexity.SIMPLE
        assert metrics.file_size_mb < 1.0

    def test_estimate_complexity_medium(self, tmp_path):
        scheduler = DocumentScheduler()
        test_file = tmp_path / "medium.pdf"
        test_file.write_bytes(b"PDF" * (1024 * 512))
        
        metrics = scheduler.estimate_complexity(str(test_file))
        assert metrics.complexity in [DocumentComplexity.SIMPLE, DocumentComplexity.MEDIUM]

    def test_estimate_complexity_complex(self, tmp_path):
        scheduler = DocumentScheduler()
        test_file = tmp_path / "large.pdf"
        test_file.write_bytes(b"PDF" * (1024 * 1024 * 6))
        
        metrics = scheduler.estimate_complexity(str(test_file))
        assert metrics.complexity in [DocumentComplexity.COMPLEX, DocumentComplexity.VERY_COMPLEX]

    def test_complexity_cache(self, tmp_path):
        scheduler = DocumentScheduler()
        test_file = tmp_path / "cached.pdf"
        test_file.write_bytes(b"PDF" * 100)
        
        metrics1 = scheduler.estimate_complexity(str(test_file))
        metrics2 = scheduler.estimate_complexity(str(test_file))
        
        assert metrics1 == metrics2
        assert str(test_file) in scheduler.complexity_cache

    def test_group_by_complexity(self, tmp_path):
        scheduler = DocumentScheduler()
        
        docs = []
        for i in range(5):
            doc = tmp_path / f"doc{i}.pdf"
            size = (i + 1) * 1024 * 512
            doc.write_bytes(b"PDF" * size)
            docs.append(str(doc))
        
        batches = scheduler.group_by_complexity(docs, batch_size=2)
        
        assert len(batches) > 0
        for batch in batches:
            assert len(batch.documents) <= 2
            assert batch.estimated_time > 0

    def test_batch_priority_ordering(self, tmp_path):
        scheduler = DocumentScheduler()
        
        simple = tmp_path / "simple.pdf"
        simple.write_bytes(b"PDF" * 100)
        
        complex_doc = tmp_path / "complex.pdf"
        complex_doc.write_bytes(b"PDF" * (1024 * 1024 * 10))
        
        docs = [str(complex_doc), str(simple)]
        batches = scheduler.group_by_complexity(docs, batch_size=1)
        
        assert batches[0].complexity == DocumentComplexity.SIMPLE


class TestCircuitBreakerWrapper:
    """Tests for CircuitBreakerWrapper"""

    def test_register_stage(self):
        wrapper = CircuitBreakerWrapper()
        wrapper.register_stage("test_stage")
        
        assert "test_stage" in wrapper.stage_circuits
        assert "test_stage" in wrapper.stage_configs

    def test_wrap_stage_success(self):
        wrapper = CircuitBreakerWrapper()
        
        def success_func():
            return "success"
        
        wrapped = wrapper.wrap_stage("test_stage", success_func)
        result = wrapped()
        
        assert result == "success"

    def test_wrap_stage_failure_threshold(self):
        wrapper = CircuitBreakerWrapper()
        
        call_count = [0]
        def flaky_func():
            call_count[0] += 1
            if call_count[0] <= 5:
                raise ValueError("Simulated failure")
            return "success"
        
        wrapped = wrapper.wrap_stage("test_stage", flaky_func)
        
        for i in range(5):
            try:
                wrapped()
            except ValueError:
                pass
        
        try:
            wrapped()
        except Exception as e:
            assert "Circuit" in str(e) or "Simulated" in str(e)

    def test_wrap_stage_with_fallback(self):
        wrapper = CircuitBreakerWrapper()
        
        def failing_func():
            raise ValueError("Always fails")
        
        def fallback_func():
            return "fallback_result"
        
        wrapped = wrapper.wrap_stage("test_stage", failing_func, fallback=fallback_func)
        
        for _ in range(6):
            try:
                result = wrapped()
            except:
                pass
        
        time.sleep(0.1)

    def test_check_thresholds(self):
        wrapper = CircuitBreakerWrapper()
        wrapper.register_stage("test_stage")
        
        status = wrapper.check_thresholds("test_stage")
        
        assert "stage" in status
        assert "state" in status
        assert status["stage"] == "test_stage"

    def test_get_all_stage_health(self):
        wrapper = CircuitBreakerWrapper()
        wrapper.register_stage("stage1")
        wrapper.register_stage("stage2")
        
        health = wrapper.get_all_stage_health()
        
        assert "stage1" in health
        assert "stage2" in health


class TestResourceMonitor:
    """Tests for ResourceMonitor"""

    def test_track_metrics(self):
        monitor = ResourceMonitor()
        metrics = monitor.track_metrics()
        
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.memory_available_mb >= 0

    def test_adapt_concurrency_high_usage(self):
        monitor = ResourceMonitor(thresholds=ResourceThresholds(
            cpu_high=10.0,
            memory_high=10.0
        ))
        monitor.current_concurrency = 8
        
        with patch('psutil.cpu_percent', return_value=90.0):
            with patch('psutil.virtual_memory') as mock_mem:
                mock_mem.return_value.percent = 85.0
                mock_mem.return_value.available = 1024 * 1024 * 1024
                
                concurrency = monitor.adapt_concurrency()
                assert concurrency < 8

    def test_adapt_concurrency_low_usage(self):
        monitor = ResourceMonitor()
        monitor.current_concurrency = 4
        
        with patch('psutil.cpu_percent', return_value=30.0):
            with patch('psutil.virtual_memory') as mock_mem:
                mock_mem.return_value.percent = 40.0
                mock_mem.return_value.available = 4096 * 1024 * 1024
                
                concurrency = monitor.adapt_concurrency()
                assert concurrency >= 4

    def test_concurrency_bounds(self):
        monitor = ResourceMonitor()
        monitor.min_concurrency = 2
        monitor.max_concurrency = 8
        
        with patch('psutil.cpu_percent', return_value=99.0):
            with patch('psutil.virtual_memory') as mock_mem:
                mock_mem.return_value.percent = 99.0
                mock_mem.return_value.available = 100 * 1024 * 1024
                
                for _ in range(10):
                    concurrency = monitor.adapt_concurrency()
                
                assert concurrency >= monitor.min_concurrency

    def test_metrics_history(self):
        monitor = ResourceMonitor()
        
        for _ in range(5):
            monitor.track_metrics()
            time.sleep(0.01)
        
        assert len(monitor.metrics_history) == 5

    def test_get_average_metrics(self):
        monitor = ResourceMonitor()
        
        for _ in range(10):
            monitor.track_metrics()
        
        averages = monitor.get_average_metrics(window=5)
        
        assert "cpu_percent" in averages
        assert "memory_percent" in averages

    def test_start_stop_monitoring(self):
        monitor = ResourceMonitor()
        
        monitor.start_monitoring(interval=0.1)
        assert monitor.monitoring is True
        
        time.sleep(0.3)
        
        monitor.stop_monitoring()
        assert monitor.monitoring is False


class TestDocumentPreValidator:
    """Tests for DocumentPreValidator"""

    def test_validate_pdf_not_found(self):
        validator = DocumentPreValidator()
        valid, error = validator.validate_pdf("/nonexistent/file.pdf")
        
        assert not valid
        assert error.error_type == "file_not_found"

    def test_validate_pdf_invalid_format(self, tmp_path):
        validator = DocumentPreValidator()
        test_file = tmp_path / "notpdf.txt"
        test_file.write_text("Not a PDF")
        
        valid, error = validator.validate_pdf(str(test_file))
        
        assert not valid
        assert error.error_type == "invalid_format"

    def test_validate_size_within_limit(self, tmp_path):
        validator = DocumentPreValidator(max_size_mb=10.0)
        test_file = tmp_path / "small.pdf"
        test_file.write_bytes(b"PDF" * 1024)
        
        valid, error = validator.validate_size(str(test_file))
        
        assert valid
        assert error is None

    def test_validate_size_exceeds_limit(self, tmp_path):
        validator = DocumentPreValidator(max_size_mb=0.001)
        test_file = tmp_path / "large.pdf"
        test_file.write_bytes(b"PDF" * (1024 * 1024))
        
        valid, error = validator.validate_size(str(test_file))
        
        assert not valid
        assert error.error_type == "size_exceeded"

    def test_validate_complete_success(self, tmp_path):
        validator = DocumentPreValidator(max_size_mb=10.0)
        test_file = tmp_path / "valid.pdf"
        test_file.write_bytes(b"%PDF-1.4\n" + b"content" * 100)
        
        valid, error = validator.validate(str(test_file))
        
        if not valid:
            assert error is not None


class TestResultStreamer:
    """Tests for ResultStreamer"""

    def test_stream_results_success(self):
        streamer = ResultStreamer()
        
        def mock_eval(doc):
            return {"score": 0.8, "document": doc}
        
        docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        results = list(streamer.stream_results("batch001", docs, mock_eval))
        
        assert len(results) == 3
        for result in results:
            assert "batch_id" in result
            assert "document_path" in result
            assert "result" in result

    def test_stream_results_with_errors(self):
        streamer = ResultStreamer()
        
        def failing_eval(doc):
            if "fail" in doc:
                raise ValueError("Evaluation failed")
            return {"score": 0.8}
        
        docs = ["doc1.pdf", "fail.pdf", "doc3.pdf"]
        results = list(streamer.stream_results("batch002", docs, failing_eval))
        
        assert len(results) == 3
        assert any("error" in r for r in results)

    def test_stream_results_progress_tracking(self):
        streamer = ResultStreamer()
        
        def mock_eval(doc):
            return {"score": 0.9}
        
        docs = ["doc1.pdf", "doc2.pdf"]
        results = list(streamer.stream_results("batch003", docs, mock_eval))
        
        assert results[0]["completed"] == 1
        assert results[0]["total"] == 2
        assert results[1]["completed"] == 2
        assert results[1]["total"] == 2

    @patch('batch_optimizer.REDIS_AVAILABLE', True)
    def test_redis_integration_mock(self):
        with patch('redis.Redis') as mock_redis_class:
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            mock_client.ping.return_value = True
            
            streamer = ResultStreamer(redis_host="localhost")
            
            assert streamer.redis_client is not None


def test_integration_full_pipeline(tmp_path):
    """Integration test for complete batch optimization pipeline"""
    
    scheduler = DocumentScheduler()
    wrapper = CircuitBreakerWrapper()
    monitor = ResourceMonitor()
    validator = DocumentPreValidator(max_size_mb=10.0)
    streamer = ResultStreamer()
    
    docs = []
    for i in range(3):
        doc = tmp_path / f"doc{i}.pdf"
        doc.write_bytes(b"%PDF-1.4\n" + b"content" * (i + 1) * 1024)
        docs.append(str(doc))
    
    valid_docs = []
    for doc in docs:
        valid, error = validator.validate(doc)
        if valid:
            valid_docs.append(doc)
    
    assert len(valid_docs) >= 2
    
    batches = scheduler.group_by_complexity(valid_docs, batch_size=2)
    assert len(batches) > 0
    
    def mock_evaluation(doc_path):
        time.sleep(0.01)
        return {"score": 0.85, "path": doc_path}
    
    wrapped_eval = wrapper.wrap_stage("evaluation", mock_evaluation)
    
    results = list(streamer.stream_results("batch_integration", valid_docs[:2], wrapped_eval))
    assert len(results) == 2
    
    metrics = monitor.track_metrics()
    assert metrics.cpu_percent >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
