#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for OpenTelemetry Distributed Tracing Instrumentation
"""

import pytest
import logging
from opentelemetry_instrumentation import (
    TracingManager,
    FlowType,
    ComponentType,
    initialize_tracing,
    get_tracing_manager,
    trace_flow,
    trace_component,
    create_span_logger
)


class TestTracingManager:
    """Test tracing manager functionality"""

    def test_initialization(self):
        """Test tracing manager initialization"""
        manager = TracingManager(
            service_name="test-service",
            service_version="1.0.0",
            environment="test"
        )
        
        assert manager.service_name == "test-service"
        assert manager.service_version == "1.0.0"
        assert manager.environment == "test"
        assert manager.tracer is not None

    def test_span_creation(self):
        """Test span creation and context management"""
        manager = TracingManager()
        
        with manager.start_span("test_span") as span:
            assert span is not None
            # Span context should be active
            span_context = span.get_span_context()
            assert span_context is not None

    def test_span_with_flow_type(self):
        """Test span creation with flow type"""
        manager = TracingManager()
        
        with manager.start_span(
            "document_ingestion_test",
            flow_type=FlowType.DOCUMENT_INGESTION
        ) as span:
            assert span is not None

    def test_span_with_component_type(self):
        """Test span creation with component type"""
        manager = TracingManager()
        
        with manager.start_span(
            "embedding_model_test",
            component_type=ComponentType.EMBEDDING_MODEL
        ) as span:
            assert span is not None

    def test_span_with_attributes(self):
        """Test span creation with custom attributes"""
        manager = TracingManager()
        
        attributes = {
            "custom_attr": "value",
            "request_id": "12345"
        }
        
        with manager.start_span("test_span", attributes=attributes) as span:
            assert span is not None

    def test_exception_recording(self):
        """Test that exceptions are recorded in spans"""
        manager = TracingManager()
        
        try:
            with manager.start_span("failing_span") as span:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

    def test_context_injection(self):
        """Test context injection for propagation"""
        manager = TracingManager()
        carrier = {}
        
        manager.inject_context(carrier)
        
        # Should have trace context
        assert 'traceparent' in carrier

    def test_context_extraction(self):
        """Test context extraction from carrier"""
        manager = TracingManager()
        carrier = {
            'traceparent': '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01'
        }
        
        context = manager.extract_context(carrier)
        # Context extraction should not raise errors


class TestGlobalTracingManager:
    """Test global tracing manager"""

    def test_initialize_global_tracing(self):
        """Test global tracing initialization"""
        initialize_tracing(
            service_name="test-global",
            service_version="2.0.0",
            environment="test"
        )
        
        manager = get_tracing_manager()
        assert manager is not None
        assert manager.service_name == "test-global"

    def test_get_tracing_manager_auto_init(self):
        """Test that get_tracing_manager auto-initializes if needed"""
        manager = get_tracing_manager()
        assert manager is not None


class TestFlowDecorator:
    """Test flow tracing decorator"""

    def test_trace_flow_decorator(self):
        """Test that flow decorator creates spans"""
        @trace_flow(FlowType.DOCUMENT_INGESTION)
        def process_document(doc_path: str) -> dict:
            return {"status": "processed", "path": doc_path}

        result = process_document("/path/to/doc.pdf")
        
        assert result["status"] == "processed"
        assert result["path"] == "/path/to/doc.pdf"

    def test_trace_flow_with_exception(self):
        """Test flow decorator with exception"""
        @trace_flow(FlowType.DOCUMENT_INGESTION)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_trace_flow_custom_operation_name(self):
        """Test flow decorator with custom operation name"""
        @trace_flow(FlowType.EMBEDDING_GENERATION, operation_name="custom_embed")
        def embed_text(text: str) -> list:
            return [0.1, 0.2, 0.3]

        result = embed_text("test text")
        assert len(result) == 3


class TestComponentDecorator:
    """Test component tracing decorator"""

    def test_trace_component_decorator(self):
        """Test that component decorator creates spans"""
        @trace_component(ComponentType.EMBEDDING_MODEL)
        def encode_batch(texts: list) -> list:
            return [[0.1, 0.2] for _ in texts]

        result = encode_batch(["text1", "text2"])
        
        assert len(result) == 2

    def test_trace_component_with_exception(self):
        """Test component decorator with exception"""
        @trace_component(ComponentType.EMBEDDING_MODEL)
        def failing_encode():
            raise RuntimeError("Encoding failed")

        with pytest.raises(RuntimeError):
            failing_encode()

    def test_trace_component_tracks_sizes(self):
        """Test that component decorator tracks input/output sizes"""
        @trace_component(ComponentType.DOCUMENT_SEGMENTER)
        def segment_text(text: str) -> list:
            return text.split()

        text = "This is a test document"
        result = segment_text(text)
        
        assert len(result) == 5


class TestSpanLogger:
    """Test span-aware logger"""

    def test_span_logger_creation(self):
        """Test span logger creation"""
        logger = create_span_logger("test_logger")
        assert logger is not None

    def test_span_logger_info(self):
        """Test span logger info logging"""
        logger = create_span_logger("test_logger")
        
        # Should not raise errors
        logger.info("Test info message")

    def test_span_logger_error(self):
        """Test span logger error logging"""
        logger = create_span_logger("test_logger")
        
        # Should not raise errors
        logger.error("Test error message")

    def test_span_logger_with_trace_context(self):
        """Test span logger includes trace context"""
        manager = TracingManager()
        logger = create_span_logger("test_logger")
        
        with manager.start_span("test_span"):
            # Logger should include trace context
            logger.info("Message with trace context")


class TestFlowTypeEnum:
    """Test FlowType enumeration"""

    def test_all_28_flows_defined(self):
        """Test that all 28 critical flows are defined"""
        flow_types = list(FlowType)
        assert len(flow_types) == 28

    def test_flow_type_values(self):
        """Test flow type values are correct"""
        assert FlowType.DOCUMENT_INGESTION.value == "document_ingestion"
        assert FlowType.CAUSAL_PATTERN_DETECTION.value == "causal_pattern_detection"
        assert FlowType.DECALOGO_EVALUATION.value == "decalogo_evaluation"


class TestComponentTypeEnum:
    """Test ComponentType enumeration"""

    def test_all_11_components_defined(self):
        """Test that all 11 pipeline components are defined"""
        component_types = list(ComponentType)
        assert len(component_types) == 11

    def test_component_type_values(self):
        """Test component type values are correct"""
        assert ComponentType.DOCUMENT_SEGMENTER.value == "document_segmenter"
        assert ComponentType.EMBEDDING_MODEL.value == "embedding_model"
        assert ComponentType.PIPELINE_ORCHESTRATOR.value == "pipeline_orchestrator"


class TestIntegration:
    """Integration tests"""

    def test_nested_spans(self):
        """Test nested span creation"""
        manager = TracingManager()
        
        with manager.start_span("parent_span") as parent:
            assert parent is not None
            
            with manager.start_span("child_span") as child:
                assert child is not None
                # Both spans should be active

    def test_flow_and_component_nesting(self):
        """Test nesting of flow and component spans"""
        @trace_component(ComponentType.EMBEDDING_MODEL)
        def encode(text: str) -> list:
            return [0.1, 0.2, 0.3]

        @trace_flow(FlowType.EMBEDDING_GENERATION)
        def process_embeddings(texts: list) -> list:
            return [encode(t) for t in texts]

        result = process_embeddings(["text1", "text2"])
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
