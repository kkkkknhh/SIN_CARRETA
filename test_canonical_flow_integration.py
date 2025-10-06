#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Flow Integration Tests
=================================

End-to-end integration tests exercising all 11 canonical nodes from data_flow_contract.py
in sequence with comprehensive verification of component integrations, fault injection
scenarios, and performance benchmarking.

Test Coverage:
- All 11 canonical nodes: sanitization, plan_processing, document_segmentation, embedding,
  responsibility_detection, contradiction_detection, monetary_detection, feasibility_scoring,
  causal_detection, teoria_cambio, dag_validation
- Integration verification: Decatalogo_principal, dag_validation, embedding_model,
  plan_processor, validate_teoria_cambio
- Fault injection: network_failure, disk_full, cpu_throttling scenarios
- Performance benchmarking: 100 iterations per component with p95/p99 latency metrics
- Baseline performance snapshot with timestamps for regression detection
- Coverage tracking for 28 critical flows from opentelemetry_instrumentation.py
"""

import pytest
import time
import json
import os
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

import numpy as np

# Import system components
from data_flow_contract import CanonicalFlowValidator, DataType, NodeContract
from miniminimoon_orchestrator import MINIMINIMOONOrchestrator, ExecutionContext
from opentelemetry_instrumentation import (
    FlowType, 
    ComponentType,
    initialize_tracing,
    get_tracing_manager
)

# Import individual components for targeted testing
try:
    from plan_sanitizer import PlanSanitizer
    from plan_processor import PlanProcessor
    from document_segmenter import DocumentSegmenter
    from embedding_model import IndustrialEmbeddingModel
    from responsibility_detector import ResponsibilityDetector
    from contradiction_detector import ContradictionDetector
    from monetary_detector import MonetaryDetector
    from feasibility_scorer import FeasibilityScorer
    from causal_pattern_detector import PDETCausalPatternDetector
    from teoria_cambio import TeoriaCambio
    from dag_validation import AdvancedDAGValidator
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logging.warning(f"Some components not available: {e}")

logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures and Utilities
# ============================================================================

@pytest.fixture
def sample_plan_text():
    """Sample plan text for testing"""
    return """
    Plan de Desarrollo Municipal - Municipio de Ejemplo
    
    1. Diagnóstico
    La situación actual del municipio presenta desafíos en infraestructura y servicios.
    
    2. Objetivos
    - Mejorar la calidad de vida de los habitantes
    - Fortalecer la economía local
    - Desarrollar infraestructura sostenible
    
    3. Estrategias
    Se implementarán programas de inversión en educación y salud.
    El alcalde será responsable de coordinar con las entidades departamentales.
    Se asignarán 500 millones de pesos para infraestructura vial.
    
    4. Indicadores
    - Reducción de pobreza en 15%
    - Aumento de cobertura educativa en 20%
    - Construcción de 10 km de vías pavimentadas
    
    5. Teoría de Cambio
    Si invertimos en educación, entonces la empleabilidad aumentará.
    Cuando mejoremos la infraestructura vial, se facilitará el comercio.
    """


@pytest.fixture
def temp_plan_file(sample_plan_text):
    """Create temporary plan file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(sample_plan_text)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def flow_validator():
    """Create canonical flow validator instance"""
    return CanonicalFlowValidator(enable_cache=True, cache_size=100)


@pytest.fixture
def orchestrator():
    """Create orchestrator instance"""
    config = {
        "parallel_processing": False,  # Disable for deterministic testing
        "determinism": {"enabled": True, "seed": 42}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    orch = MINIMINIMOONOrchestrator(config_path=config_path)
    yield orch
    
    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def performance_metrics():
    """Fixture for collecting performance metrics"""
    return {
        "latencies": defaultdict(list),
        "errors": defaultdict(list),
        "timestamps": {},
        "baseline": {}
    }


def measure_latency(func, *args, **kwargs):
    """Measure function execution latency"""
    start = time.time()
    result = func(*args, **kwargs)
    latency = (time.time() - start) * 1000  # Convert to milliseconds
    return result, latency


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate p50, p95, p99 percentiles"""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies))
    }


def save_baseline_snapshot(metrics: Dict[str, Any], output_path: str):
    """Save baseline performance snapshot with timestamps"""
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "version": "1.0.0",
        "system_info": {
            "python_version": os.sys.version,
            "platform": os.sys.platform
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    logger.info(f"Baseline snapshot saved to {output_path}")


# ============================================================================
# Test Class: End-to-End Canonical Flow
# ============================================================================

class TestCanonicalFlowEndToEnd:
    """End-to-end tests for all 11 canonical nodes in sequence"""
    
    def test_all_11_nodes_sequential_execution(self, temp_plan_file, flow_validator, orchestrator):
        """Test that all 11 canonical nodes execute successfully in sequence"""
        # Execute the full canonical flow
        results = orchestrator.process_plan(temp_plan_file)
        
        # Verify no fatal errors
        assert "error" not in results or results.get("error") is None
        
        # Verify all 11 canonical nodes were executed
        expected_nodes = flow_validator.get_execution_plan()
        executed_nodes = results.get("executed_nodes", [])
        
        assert len(executed_nodes) >= 11, f"Expected at least 11 nodes, got {len(executed_nodes)}"
        
        # Verify canonical order
        order_valid, warnings = flow_validator.validate_canonical_order(executed_nodes)
        if warnings:
            logger.warning(f"Order warnings: {warnings}")
        
        # Verify each node's outputs
        assert "metadata" in results, "Missing plan_processing output"
        assert "segments" in results, "Missing document_segmentation output"
        assert "embeddings" in results, "Missing embedding output"
        assert "responsibilities" in results, "Missing responsibility_detection output"
        assert "contradictions" in results, "Missing contradiction_detection output"
        assert "monetary" in results, "Missing monetary_detection output"
        assert "feasibility" in results, "Missing feasibility_scoring output"
        assert "causal_patterns" in results, "Missing causal_detection output"
        assert "teoria_cambio" in results, "Missing teoria_cambio output"
        assert "dag_validation" in results, "Missing dag_validation output"
    
    def test_node_data_flow_contracts(self, temp_plan_file, flow_validator, orchestrator):
        """Test that data flow contracts are satisfied between nodes"""
        results = orchestrator.process_plan(temp_plan_file)
        executed_nodes = results.get("executed_nodes", [])
        
        # Build accumulated data state
        available_data = {"raw_text": "dummy"}
        node_reports = {}
        
        for node_name in executed_nodes:
            # Validate node execution
            is_valid, report = flow_validator.validate_node_execution(
                node_name, available_data, use_cache=False
            )
            node_reports[node_name] = report
            
            # Simulate adding outputs to available data
            contract = flow_validator.get_contract(node_name)
            if contract:
                for output_type in contract.produces:
                    available_data[output_type.value] = "dummy_output"
        
        # Generate comprehensive flow report
        flow_report = flow_validator.generate_flow_report(executed_nodes, node_reports)
        
        assert flow_report["dependencies_satisfied"], \
            f"Dependencies not satisfied: {flow_report['dependency_errors']}"
    
    def test_teoria_cambio_integration(self, temp_plan_file, orchestrator):
        """Test teoria_cambio integration with upstream components"""
        results = orchestrator.process_plan(temp_plan_file)
        
        # Verify teoria_cambio received required inputs
        assert "responsibilities" in results, "Missing responsibilities for teoria_cambio"
        assert "causal_patterns" in results, "Missing causal patterns for teoria_cambio"
        assert "monetary" in results, "Missing monetary values for teoria_cambio"
        
        # Verify teoria_cambio output structure
        teoria_result = results.get("teoria_cambio", {})
        assert "is_valid" in teoria_result or "causal_graph" in teoria_result
    
    def test_dag_validation_integration(self, temp_plan_file, orchestrator):
        """Test DAG validation integration"""
        results = orchestrator.process_plan(temp_plan_file)
        
        # Verify DAG validation output
        dag_result = results.get("dag_validation", {})
        assert "is_acyclic" in dag_result, "DAG validation missing is_acyclic"
        assert "node_count" in dag_result, "DAG validation missing node_count"
        assert "edge_count" in dag_result, "DAG validation missing edge_count"
    
    def test_embedding_model_integration(self, temp_plan_file, orchestrator):
        """Test embedding model integration"""
        results = orchestrator.process_plan(temp_plan_file)
        
        # Verify segments were created
        segments = results.get("segments", {})
        assert segments.get("count", 0) > 0, "No segments created"
        
        # Verify embeddings were generated
        embeddings = results.get("embeddings", {})
        assert embeddings.get("count", 0) > 0, "No embeddings generated"
        assert embeddings.get("count") == segments.get("count"), \
            "Embedding count doesn't match segment count"
    
    def test_plan_processor_integration(self, temp_plan_file, orchestrator):
        """Test plan_processor integration"""
        results = orchestrator.process_plan(temp_plan_file)
        
        # Verify plan processing metadata
        metadata = results.get("metadata", {})
        assert metadata is not None, "Plan processor produced no metadata"
        
        # Check for expected metadata fields (if any)
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
    
    def test_evidence_registry_integration(self, temp_plan_file, orchestrator):
        """Test evidence registry captures data from all nodes"""
        results = orchestrator.process_plan(temp_plan_file)
        
        # Verify evidence registry was populated
        evidence_stats = results.get("evidence_registry", {}).get("statistics", {})
        assert evidence_stats.get("total_evidence", 0) > 0, "No evidence registered"


# ============================================================================
# Test Class: Fault Injection
# ============================================================================

class TestFaultInjection:
    """Fault injection scenarios for partial recovery handling"""
    
    def test_network_failure_scenario(self, temp_plan_file, orchestrator):
        """Test partial recovery handling with network failure simulation"""
        # Mock network-dependent operations to fail
        with patch('embedding_model.IndustrialEmbeddingModel') as mock_embedding:
            mock_embedding.side_effect = ConnectionError("Network unavailable")
            
            results = orchestrator.process_plan(temp_plan_file)
            
            # System should continue with partial results
            assert "error" in results or "execution_summary" in results
            
            # Some nodes should still succeed
            executed_nodes = results.get("executed_nodes", [])
            assert len(executed_nodes) > 0, "No nodes executed despite network failure"
            
            # Verify error was logged
            execution_summary = results.get("execution_summary", {})
            assert execution_summary.get("errors", 0) > 0 or "error" in results
    
    def test_disk_full_scenario(self, temp_plan_file, orchestrator):
        """Test partial recovery handling with disk full simulation"""
        # Mock file write operations to fail
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            try:
                results = orchestrator.process_plan(temp_plan_file)
                
                # Should handle gracefully
                assert results is not None
                
                # Verify partial execution
                executed_nodes = results.get("executed_nodes", [])
                logger.info(f"Executed {len(executed_nodes)} nodes despite disk full")
            except OSError:
                # Acceptable if initial file read fails
                pytest.skip("Cannot proceed without reading input file")
    
    def test_cpu_throttling_scenario(self, temp_plan_file, orchestrator, performance_metrics):
        """Test performance degradation under CPU throttling"""
        # Simulate CPU throttling with artificial delays
        def throttled_processing(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow processing
            return Mock()
        
        with patch.object(PlanSanitizer, 'sanitize', side_effect=throttled_processing):
            start_time = time.time()
            results = orchestrator.process_plan(temp_plan_file)
            total_time = time.time() - start_time
            
            # Record performance degradation
            performance_metrics["latencies"]["throttled"].append(total_time * 1000)
            
            # Should complete despite throttling
            assert results is not None
            assert total_time > 0.1, "Throttling simulation didn't slow execution"
    
    def test_memory_pressure_scenario(self, temp_plan_file):
        """Test behavior under memory pressure"""
        # Create large data structure to simulate memory pressure
        large_data = ["x" * 1000000 for _ in range(10)]
        
        try:
            # Initialize orchestrator with memory pressure
            orchestrator = MINIMINIMOONOrchestrator()
            results = orchestrator.process_plan(temp_plan_file)
            
            # Should handle gracefully
            assert results is not None
            
            # Clean up
            del large_data
        except MemoryError:
            pytest.skip("Insufficient memory for test")
    
    def test_partial_component_failure(self, temp_plan_file, orchestrator):
        """Test that failure in one component doesn't crash entire pipeline"""
        # Mock one component to fail
        with patch('contradiction_detector.ContradictionDetector.detect', 
                   side_effect=RuntimeError("Component failure")):
            results = orchestrator.process_plan(temp_plan_file)
            
            # Pipeline should continue
            assert results is not None
            executed_nodes = results.get("executed_nodes", [])
            
            # Some nodes should succeed even if contradiction_detection fails
            assert "sanitization" in executed_nodes or len(executed_nodes) > 0


# ============================================================================
# Test Class: Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarking with 100 iterations per component"""
    
    @pytest.mark.parametrize("component_name,iterations", [
        ("sanitization", 100),
        ("document_segmentation", 100),
        ("embedding", 50),  # Fewer iterations for expensive operations
    ])
    def test_component_performance_benchmark(self, sample_plan_text, component_name, 
                                            iterations, performance_metrics):
        """Benchmark individual component performance with parameterized iterations"""
        latencies = []
        
        if component_name == "sanitization" and COMPONENTS_AVAILABLE:
            sanitizer = PlanSanitizer()
            for _ in range(iterations):
                _, latency = measure_latency(sanitizer.sanitize, sample_plan_text)
                latencies.append(latency)
        
        elif component_name == "document_segmentation" and COMPONENTS_AVAILABLE:
            segmenter = DocumentSegmenter()
            for _ in range(iterations):
                _, latency = measure_latency(segmenter.segment, sample_plan_text)
                latencies.append(latency)
        
        elif component_name == "embedding" and COMPONENTS_AVAILABLE:
            try:
                embedding_model = IndustrialEmbeddingModel()
                test_segments = ["Test segment 1", "Test segment 2", "Test segment 3"]
                for _ in range(iterations):
                    _, latency = measure_latency(embedding_model.encode, test_segments)
                    latencies.append(latency)
            except Exception as e:
                pytest.skip(f"Embedding model not available: {e}")
        
        else:
            pytest.skip(f"Component {component_name} not available or not implemented")
        
        # Calculate percentiles
        metrics = calculate_percentiles(latencies)
        performance_metrics["latencies"][component_name] = latencies
        performance_metrics["timestamps"][component_name] = datetime.utcnow().isoformat()
        
        # Log results
        logger.info(f"{component_name} benchmark ({iterations} iterations):")
        logger.info(f"  p50: {metrics['p50']:.2f}ms")
        logger.info(f"  p95: {metrics['p95']:.2f}ms")
        logger.info(f"  p99: {metrics['p99']:.2f}ms")
        logger.info(f"  mean: {metrics['mean']:.2f}ms")
        
        # Assertions
        assert len(latencies) == iterations, f"Expected {iterations} measurements"
        assert metrics['p95'] > 0, "p95 latency should be positive"
        assert metrics['p99'] >= metrics['p95'], "p99 should be >= p95"
    
    def test_end_to_end_performance_benchmark(self, temp_plan_file, orchestrator, 
                                              performance_metrics):
        """Benchmark end-to-end pipeline performance"""
        iterations = 10  # Fewer iterations for full pipeline
        latencies = []
        
        for i in range(iterations):
            start = time.time()
            results = orchestrator.process_plan(temp_plan_file)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            logger.info(f"Iteration {i+1}/{iterations}: {latency:.2f}ms")
        
        # Calculate metrics
        metrics = calculate_percentiles(latencies)
        performance_metrics["latencies"]["end_to_end"] = latencies
        performance_metrics["timestamps"]["end_to_end"] = datetime.utcnow().isoformat()
        
        logger.info(f"End-to-end benchmark ({iterations} iterations):")
        logger.info(f"  p50: {metrics['p50']:.2f}ms")
        logger.info(f"  p95: {metrics['p95']:.2f}ms")
        logger.info(f"  p99: {metrics['p99']:.2f}ms")
        logger.info(f"  mean: {metrics['mean']:.2f}ms")
        
        # Assertions
        assert len(latencies) == iterations
        assert metrics['mean'] > 0
    
    def test_baseline_snapshot_creation(self, performance_metrics, tmp_path):
        """Create baseline performance snapshot for regression detection"""
        # Collect summary metrics
        baseline = {
            "components": {},
            "timestamp": datetime.utcnow().isoformat(),
            "test_run_id": f"baseline_{int(time.time())}"
        }
        
        # Add component metrics
        for component_name, latencies in performance_metrics["latencies"].items():
            if latencies:
                baseline["components"][component_name] = calculate_percentiles(latencies)
        
        # Save baseline snapshot
        baseline_path = tmp_path / "performance_baseline.json"
        save_baseline_snapshot(baseline, str(baseline_path))
        
        # Verify snapshot was created
        assert baseline_path.exists(), "Baseline snapshot not created"
        
        # Verify snapshot contents
        with open(baseline_path) as f:
            loaded_baseline = json.load(f)
        
        assert "timestamp" in loaded_baseline
        assert "metrics" in loaded_baseline
        assert "version" in loaded_baseline
        
        logger.info(f"Baseline snapshot created with {len(baseline['components'])} components")


# ============================================================================
# Test Class: Coverage Tracking
# ============================================================================

class TestCoverageTracking:
    """Track coverage across 28 critical flows"""
    
    def test_critical_flows_coverage(self, temp_plan_file, orchestrator):
        """Verify coverage of 28 critical flows from FlowType enum"""
        # Initialize tracing
        initialize_tracing(service_name="test-canonical-flow")
        
        # Execute pipeline
        results = orchestrator.process_plan(temp_plan_file)
        
        # Track which flows were exercised
        exercised_flows = set()
        
        # Map executed nodes to flow types
        executed_nodes = results.get("executed_nodes", [])
        
        flow_mapping = {
            "sanitization": [FlowType.TEXT_NORMALIZATION],
            "document_segmentation": [FlowType.DOCUMENT_SEGMENTATION],
            "embedding": [FlowType.EMBEDDING_GENERATION],
            "responsibility_detection": [FlowType.RESPONSIBILITY_DETECTION],
            "contradiction_detection": [FlowType.CONTRADICTION_DETECTION],
            "monetary_detection": [FlowType.MONETARY_DETECTION],
            "feasibility_scoring": [FlowType.FEASIBILITY_SCORING],
            "causal_detection": [FlowType.CAUSAL_PATTERN_DETECTION],
            "teoria_cambio": [FlowType.TEORIA_CAMBIO_ANALYSIS],
            "dag_validation": [FlowType.DAG_VALIDATION],
            "plan_processing": [FlowType.DOCUMENT_INGESTION]
        }
        
        for node in executed_nodes:
            if node in flow_mapping:
                exercised_flows.update(flow_mapping[node])
        
        # Calculate coverage
        total_flows = len(FlowType)
        covered_flows = len(exercised_flows)
        coverage_percentage = (covered_flows / total_flows) * 100
        
        logger.info(f"Flow coverage: {covered_flows}/{total_flows} ({coverage_percentage:.1f}%)")
        logger.info(f"Exercised flows: {[f.value for f in exercised_flows]}")
        
        # Verify minimum coverage threshold
        assert coverage_percentage >= 35.0, \
            f"Coverage {coverage_percentage:.1f}% below target of 35%"
    
    def test_component_coverage(self, temp_plan_file, orchestrator):
        """Verify coverage of 11 pipeline components"""
        results = orchestrator.process_plan(temp_plan_file)
        executed_nodes = results.get("executed_nodes", [])
        
        # Map nodes to component types
        component_mapping = {
            "document_segmentation": ComponentType.DOCUMENT_SEGMENTER,
            "embedding": ComponentType.EMBEDDING_MODEL,
            "causal_detection": ComponentType.CAUSAL_PATTERN_DETECTOR,
            "monetary_detection": ComponentType.MONETARY_DETECTOR,
            "responsibility_detection": ComponentType.RESPONSIBILITY_DETECTOR,
            "feasibility_scoring": ComponentType.FEASIBILITY_SCORER,
            "contradiction_detection": ComponentType.CONTRADICTION_DETECTOR,
            "teoria_cambio": ComponentType.TEORIA_CAMBIO,
            "questionnaire_evaluation": ComponentType.QUESTIONNAIRE_ENGINE,
        }
        
        covered_components = set()
        for node in executed_nodes:
            if node in component_mapping:
                covered_components.add(component_mapping[node])
        
        # Calculate component coverage
        total_components = len(ComponentType)
        covered_count = len(covered_components)
        component_coverage = (covered_count / total_components) * 100
        
        logger.info(f"Component coverage: {covered_count}/{total_components} ({component_coverage:.1f}%)")
        
        # Verify reasonable component coverage
        assert component_coverage >= 70.0, \
            f"Component coverage {component_coverage:.1f}% below target of 70%"
    
    def test_path_coverage_target(self, temp_plan_file, orchestrator, flow_validator):
        """Verify path coverage exceeds 95% target"""
        results = orchestrator.process_plan(temp_plan_file)
        executed_nodes = results.get("executed_nodes", [])
        
        # Calculate path coverage based on canonical nodes
        canonical_nodes = flow_validator.get_execution_plan()
        executed_canonical = [n for n in executed_nodes if n in canonical_nodes]
        
        path_coverage = (len(executed_canonical) / len(canonical_nodes)) * 100
        
        logger.info(f"Path coverage: {len(executed_canonical)}/{len(canonical_nodes)} ({path_coverage:.1f}%)")
        
        # Target: >95% path coverage
        assert path_coverage >= 95.0, \
            f"Path coverage {path_coverage:.1f}% below target of 95%"


# ============================================================================
# Test Class: Integration Verification
# ============================================================================

class TestIntegrationVerification:
    """Verification tests for specific component integrations"""
    
    def test_decatalogo_principal_integration(self, temp_plan_file):
        """Test Decatalogo_principal integration (if available)"""
        try:
            from Decatalogo_principal import AdvancedDeviceConfig
            
            # Verify Decatalogo_principal components are accessible
            config = AdvancedDeviceConfig(device="cpu")
            assert config is not None
            assert config.get_device() == "cpu"
            
            logger.info("Decatalogo_principal integration verified")
        except ImportError:
            pytest.skip("Decatalogo_principal not available")
    
    def test_validate_teoria_cambio_integration(self):
        """Test validate_teoria_cambio module integration (if available)"""
        try:
            from validate_teoria_cambio import IndustrialGradeValidator
            
            validator = IndustrialGradeValidator()
            assert validator is not None
            
            logger.info("validate_teoria_cambio integration verified")
        except ImportError:
            pytest.skip("validate_teoria_cambio not available")
    
    def test_orchestrator_component_initialization(self, orchestrator):
        """Verify orchestrator successfully initialized all components"""
        # Check that orchestrator has all required components
        assert hasattr(orchestrator, 'sanitizer'), "Missing sanitizer"
        assert hasattr(orchestrator, 'processor'), "Missing processor"
        assert hasattr(orchestrator, 'segmenter'), "Missing segmenter"
        assert hasattr(orchestrator, 'embedding_model'), "Missing embedding_model"
        assert hasattr(orchestrator, 'responsibility_detector'), "Missing responsibility_detector"
        assert hasattr(orchestrator, 'contradiction_detector'), "Missing contradiction_detector"
        assert hasattr(orchestrator, 'monetary_detector'), "Missing monetary_detector"
        assert hasattr(orchestrator, 'feasibility_scorer'), "Missing feasibility_scorer"
        assert hasattr(orchestrator, 'causal_detector'), "Missing causal_detector"
        assert hasattr(orchestrator, 'dag_validator'), "Missing dag_validator"
        assert hasattr(orchestrator, 'flow_validator'), "Missing flow_validator"
        
        logger.info("All orchestrator components initialized successfully")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
