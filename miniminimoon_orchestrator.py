#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINIMINIMOON Orchestrator
=========================

Central orchestrator that coordinates all components in the canonical flow of the MINIMINIMOON system.
This module manages the execution sequence, data flow, and component interactions,
ensuring robust integration across all analytical modules.

REFACTORED: Unified orchestration with EvidenceRegistry and deterministic execution.
"""
import logging
import time
import os
import json
import random
import numpy as np
import yaml
import statistics
from typing import Dict, List, Any, Optional
from functools import wraps
from datetime import datetime

# Import performance monitoring and circuit breaker components
from performance_test_suite import PerformanceBenchmark, PerformanceResult
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError

# Import core processing components
from plan_sanitizer import PlanSanitizer
from plan_processor import PlanProcessor
from document_segmenter import DocumentSegmenter
from embedding_model import IndustrialEmbeddingModel
from spacy_loader import SpacyModelLoader, SafeSpacyProcessor

# Import analysis components
from responsibility_detector import ResponsibilityDetector
from contradiction_detector import ContradictionDetector
from monetary_detector import MonetaryDetector
from feasibility_scorer import FeasibilityScorer
from teoria_cambio import TeoriaCambio
from dag_validation import AdvancedDAGValidator
from causal_pattern_detector import PDETCausalPatternDetector

# Import unified system components
from evidence_registry import EvidenceRegistry
from data_flow_contract import CanonicalFlowValidator
from miniminimoon_immutability import ImmutabilityContract

# Import questionnaire engine for 300-question evaluation
from questionnaire_engine import QuestionnaireEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MINIMINIMOONOrchestrator")


class PerformanceMonitor:
    """
    Real-time performance monitoring with p95/p99 percentile tracking.
    Exports metrics in Prometheus format and enforces performance budgets.
    """
    
    def __init__(self, budgets_config_path: str = "performance_budgets.yaml"):
        self.latencies: Dict[str, List[float]] = {}
        self.budgets = {}
        self.circuit_breaker_events: List[Dict[str, Any]] = []
        self.load_budgets(budgets_config_path)
        
    def load_budgets(self, config_path: str):
        """Load performance budgets from YAML configuration"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.budgets = config.get('budgets', {})
                    logger.info(f"Loaded performance budgets for {len(self.budgets)} nodes")
            else:
                logger.warning(f"Performance budgets file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading performance budgets: {e}")
    
    def record_latency(self, node_name: str, latency_ms: float):
        """Record latency measurement for a node"""
        if node_name not in self.latencies:
            self.latencies[node_name] = []
        self.latencies[node_name].append(latency_ms)
    
    def get_percentiles(self, node_name: str) -> Dict[str, float]:
        """Calculate p50, p95, p99 percentiles for a node"""
        if node_name not in self.latencies or not self.latencies[node_name]:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_latencies = sorted(self.latencies[node_name])
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.50)] if n > 0 else 0.0,
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            "mean": statistics.mean(sorted_latencies),
            "count": n
        }
    
    def check_budget_violation(self, node_name: str) -> tuple[bool, str]:
        """Check if node's p95 latency exceeds budget"""
        if node_name not in self.budgets:
            return True, f"No budget defined for {node_name}"
        
        percentiles = self.get_percentiles(node_name)
        p95_ms = percentiles["p95"]
        
        budget_config = self.budgets[node_name]
        budget_ms = budget_config.get("p95_ms", float('inf'))
        tolerance_pct = budget_config.get("tolerance_pct", 10.0)
        max_allowed_ms = budget_ms * (1 + tolerance_pct / 100)
        
        passed = p95_ms <= max_allowed_ms
        
        if passed:
            margin = ((max_allowed_ms - p95_ms) / budget_ms) * 100
            msg = f"✅ {node_name}: p95={p95_ms:.2f}ms < {max_allowed_ms:.2f}ms (margin: {margin:.1f}%)"
        else:
            overage = ((p95_ms - max_allowed_ms) / budget_ms) * 100
            msg = f"❌ {node_name}: p95={p95_ms:.2f}ms > {max_allowed_ms:.2f}ms (overage: {overage:.1f}%)"
        
        return passed, msg
    
    def record_circuit_event(self, event_type: str, circuit_name: str, data: Dict[str, Any]):
        """Record circuit breaker state transition"""
        self.circuit_breaker_events.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "circuit_name": circuit_name,
            "data": data
        })
    
    def export_prometheus_metrics(self, output_path: str = "performance_metrics.prom"):
        """Export metrics in Prometheus format"""
        lines = [
            "# HELP miniminimoon_pipeline_latency_milliseconds Pipeline node latency",
            "# TYPE miniminimoon_pipeline_latency_milliseconds histogram",
        ]
        
        for node_name, latencies in self.latencies.items():
            if not latencies:
                continue
            
            percentiles = self.get_percentiles(node_name)
            
            # Prometheus histogram format
            lines.append(f'miniminimoon_pipeline_latency_milliseconds{{node="{node_name}",quantile="0.5"}} {percentiles["p50"]:.2f}')
            lines.append(f'miniminimoon_pipeline_latency_milliseconds{{node="{node_name}",quantile="0.95"}} {percentiles["p95"]:.2f}')
            lines.append(f'miniminimoon_pipeline_latency_milliseconds{{node="{node_name}",quantile="0.99"}} {percentiles["p99"]:.2f}')
            lines.append(f'miniminimoon_pipeline_latency_milliseconds_count{{node="{node_name}"}} {percentiles["count"]}')
        
        # Circuit breaker metrics
        lines.append("")
        lines.append("# HELP miniminimoon_circuit_breaker_events_total Circuit breaker state transitions")
        lines.append("# TYPE miniminimoon_circuit_breaker_events_total counter")
        
        event_counts = {}
        for event in self.circuit_breaker_events:
            key = (event["circuit_name"], event["event_type"])
            event_counts[key] = event_counts.get(key, 0) + 1
        
        for (circuit_name, event_type), count in event_counts.items():
            lines.append(f'miniminimoon_circuit_breaker_events_total{{circuit="{circuit_name}",event="{event_type}"}} {count}')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported Prometheus metrics to {output_path}")
    
    def generate_dashboard_html(self, output_path: str = "performance_dashboard.html"):
        """Generate HTML dashboard with real-time metrics"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>MINIMINIMOON Performance Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .metric-card { 
            background: white; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px 0; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .pass { color: #28a745; }
        .fail { color: #dc3545; }
        .warning { color: #ffc107; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #007bff; color: white; }
        .percentile { font-weight: bold; }
    </style>
</head>
<body>
    <h1>🚀 MINIMINIMOON Performance Dashboard</h1>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="metric-card">
        <h2>Pipeline Node Latencies</h2>
        <table>
            <tr>
                <th>Node</th>
                <th>p50 (ms)</th>
                <th>p95 (ms)</th>
                <th>p99 (ms)</th>
                <th>Budget</th>
                <th>Status</th>
            </tr>
"""
        
        for node_name in sorted(self.latencies.keys()):
            percentiles = self.get_percentiles(node_name)
            passed, msg = self.check_budget_violation(node_name)
            
            status_class = "pass" if passed else "fail"
            status_icon = "✅" if passed else "❌"
            
            budget_str = f"{self.budgets.get(node_name, {}).get('p95_ms', 'N/A')} ms" if node_name in self.budgets else "N/A"
            
            html += f"""            <tr>
                <td>{node_name}</td>
                <td>{percentiles['p50']:.2f}</td>
                <td class="percentile">{percentiles['p95']:.2f}</td>
                <td>{percentiles['p99']:.2f}</td>
                <td>{budget_str}</td>
                <td class="{status_class}">{status_icon} {msg.split(':')[0]}</td>
            </tr>
"""
        
        html += """        </table>
    </div>
    
    <div class="metric-card">
        <h2>Circuit Breaker Events</h2>
        <p>Total events: """ + str(len(self.circuit_breaker_events)) + """</p>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated dashboard: {output_path}")


class ExecutionContext:
    """
    Maintains context information during the orchestration process.
    Tracks execution time, component dependencies, and data flow.
    """
    def __init__(self):
        self.start_time = time.time()
        self.execution_times = {}
        self.component_status = {}
        self.data_flow_history = []
        self.error_registry = []
        self.config = {}

    def register_component_execution(self, component_name: str, start_time: float, end_time: float, status: str):
        """Register the execution of a component with timing and status"""
        self.execution_times[component_name] = end_time - start_time
        self.component_status[component_name] = status

    def register_data_flow(self, source: str, destination: str, data_type: str, size: int):
        """Register data flowing between components"""
        self.data_flow_history.append({
            "timestamp": time.time(),
            "source": source,
            "destination": destination,
            "data_type": data_type,
            "size": size
        })

    def register_error(self, component: str, error_type: str, error_message: str, is_fatal: bool = False):
        """Register an error that occurred during execution"""
        self.error_registry.append({
            "timestamp": time.time(),
            "component": component,
            "error_type": error_type,
            "message": error_message,
            "is_fatal": is_fatal
        })

    def get_execution_summary(self) -> Dict[str, Any]:
        """Generate a summary of the orchestration execution"""
        return {
            "total_time": time.time() - self.start_time,
            "component_times": self.execution_times,
            "component_status": self.component_status,
            "data_flows": len(self.data_flow_history),
            "errors": len(self.error_registry),
            "fatal_errors": sum(1 for e in self.error_registry if e["is_fatal"])
        }


def component_execution(component_name: str):
    """
    Decorator to track component execution time and status,
    and handle exceptions gracefully.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            context = kwargs.get('context', self.context)
            start_time = time.time()
            status = "success"

            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_msg = str(e)
                logger.error(f"Error in {component_name}: {error_msg}")
                context.register_error(component_name, type(e).__name__, error_msg)
                return None
            finally:
                end_time = time.time()
                context.register_component_execution(component_name, start_time, end_time, status)

        return wrapper
    return decorator


class MINIMINIMOONOrchestrator:
    """
    Central orchestrator for the MINIMINIMOON system.

    This class coordinates the interactions between all system components,
    manages the canonical flow, and ensures robust error handling and recovery.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MINIMINIMOON orchestrator.

        Args:
            config_path: Optional path to configuration file
        """
        self.context = ExecutionContext()

        # Load configuration
        self.config = self._load_config(config_path)
        self.context.config = self.config

        # Initialize the immutability contract
        self.immutability_contract = ImmutabilityContract()

        # Initialize performance monitoring
        logger.info("Initializing performance monitoring...")
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize circuit breakers for fault-prone operations
        logger.info("Initializing circuit breakers (2.0s recovery threshold)...")
        self.circuit_breakers = self._initialize_circuit_breakers()

        logger.info("Initializing MINIMINIMOON Orchestrator")
        self._initialize_components()
        logger.info("MINIMINIMOON Orchestrator initialized successfully")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "parallel_processing": True,
            "embedding_batch_size": 32,
            "segmentation_strategy": "paragraph",
            "context_window_size": 150,
            "error_tolerance": "medium",
            "log_level": "INFO",
            "cache_embeddings": True,
            "verification_level": "normal",
            "determinism": {
                "enabled": True,
                "seed": 42
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                config = {**default_config, **user_config}
                logger.info(f"Loaded custom configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

        logger.info("Using default configuration")
        return default_config

    def _initialize_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Initialize circuit breakers for fault-prone operations"""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60.0,
            recovery_time_sla_seconds=2.0
        )
        
        fault_prone_operations = [
            "embedding",
            "responsibility_detection",
            "contradiction_detection",
            "causal_detection",
            "teoria_cambio",
            "dag_validation"
        ]
        
        circuits = {}
        for op_name in fault_prone_operations:
            circuit = CircuitBreaker(op_name, circuit_config)
            
            # Register alert callback for monitoring
            def alert_callback(event_type, circuit_name, data):
                self.performance_monitor.record_circuit_event(event_type, circuit_name, data)
                if event_type == "circuit_opened":
                    logger.error(f"🔴 Circuit breaker OPENED: {circuit_name}")
                elif event_type == "state_transition":
                    logger.warning(f"⚠️  Circuit breaker transition: {circuit_name} -> {data.get('to')}")
            
            circuit.register_alert(alert_callback)
            circuits[op_name] = circuit
            
        logger.info(f"Initialized {len(circuits)} circuit breakers")
        return circuits

    def _initialize_components(self):
        """Initialize all system components in the correct order"""
        try:
            # Core processing components
            logger.info("Initializing core processing components...")
            self.sanitizer = PlanSanitizer()
            self.processor = PlanProcessor()
            self.segmenter = DocumentSegmenter()

            # Embedding and NLP components
            logger.info("Initializing embedding and NLP components...")
            self.embedding_model = IndustrialEmbeddingModel()
            self.spacy_loader = SpacyModelLoader()
            self.spacy_processor = SafeSpacyProcessor(self.spacy_loader)

            # Analysis components
            logger.info("Initializing analysis components...")
            self.responsibility_detector = ResponsibilityDetector()
            self.contradiction_detector = ContradictionDetector()
            self.monetary_detector = MonetaryDetector()
            self.feasibility_scorer = FeasibilityScorer(
                enable_parallel=self.config.get("parallel_processing", True)
            )
            # Initialize causal detector with empty list if no PDET municipalities configured
            pdet_municipalities = self.config.get("pdet_municipalities", [])
            self.causal_detector = PDETCausalPatternDetector(pdet_municipalities)
            self.dag_validator = AdvancedDAGValidator()

            # Unified system components
            logger.info("Initializing unified system components...")
            self.evidence_registry = EvidenceRegistry()
            self.flow_validator = CanonicalFlowValidator()

            # Initialize Questionnaire Engine for 300-question evaluation
            logger.info("Initializing Questionnaire Engine (300 questions)...")
            self.questionnaire_engine = QuestionnaireEngine()

            # Verify system components
            verification_level = self.config.get("verification_level", "normal")
            self.immutability_contract.verify_components(verification_level)

            # Register initialization success
            for component_name in [
                "sanitizer", "processor", "segmenter", "embedding_model",
                "spacy_processor", "responsibility_detector",
                "contradiction_detector", "monetary_detector", "feasibility_scorer",
                "causal_detector", "dag_validator", "evidence_registry",
                "flow_validator", "questionnaire_engine"
            ]:
                self.context.component_status[component_name] = "initialized"

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.context.register_error("initialization", type(e).__name__, str(e), is_fatal=True)
            raise RuntimeError(f"Orchestrator initialization failed: {e}")

    def process_plan(self, plan_path: str) -> Dict[str, Any]:
        """
        Process a plan document through the canonical flow.

        Args:
            plan_path: Path to the plan document file

        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Processing plan: {plan_path}")

        # Seed for determinism
        if self.config.get("determinism", {}).get("enabled", False):
            seed = self.config.get("determinism", {}).get("seed", 42)
            random.seed(seed)
            np.random.seed(seed)

        results = {"plan_path": plan_path}

        # Read the document
        try:
            with open(plan_path, 'r', encoding='utf-8') as f:
                plan_text = f.read()
        except Exception as e:
            logger.error(f"Error reading plan file {plan_path}: {e}")
            self.context.register_error("file_reading", type(e).__name__, str(e), is_fatal=True)
            return {"error": f"Could not read file: {str(e)}", "plan_path": plan_path}

        # Extract plan name
        plan_name = os.path.basename(plan_path).split('.')[0]
        results["plan_name"] = plan_name
        results["executed_nodes"] = []

        # Execute the canonical flow with evidence registration
        try:
            # 1. Sanitization
            logger.info("[1/11] Sanitization...")
            sanitized_text = self._execute_sanitization(plan_text)
            results["executed_nodes"].append("sanitization")

            # 2. Plan processing
            logger.info("[2/11] Plan Processing...")
            processed_plan = self._execute_plan_processing(sanitized_text)
            results["metadata"] = processed_plan
            results["executed_nodes"].append("plan_processing")

            # 3. Document segmentation
            logger.info("[3/11] Document Segmentation...")
            segments = self._execute_segmentation(sanitized_text)
            results["segments"] = {"count": len(segments)}
            results["executed_nodes"].append("document_segmentation")

            # 4. Embedding
            logger.info("[4/11] Embedding Generation...")
            embeddings = self._execute_embedding(segments)
            results["embeddings"] = {"count": len(embeddings)}
            results["executed_nodes"].append("embedding")

            # 5. Responsibility detection
            logger.info("[5/11] Responsibility Detection...")
            responsibilities = self._execute_responsibility_detection(sanitized_text)
            results["responsibilities"] = responsibilities
            results["executed_nodes"].append("responsibility_detection")

            # Register evidence for D4 questions
            for resp in responsibilities:
                self.evidence_registry.register(
                    source_component="responsibility_detection",
                    evidence_type="institutional_entity",
                    content=resp,
                    confidence=resp.get("confidence", 0.5),
                    applicable_questions=[f"D4-Q{i}" for i in range(1, 51)]
                )

            # 6. Contradiction detection
            logger.info("[6/11] Contradiction Detection...")
            contradictions = self._execute_contradiction_detection(sanitized_text)
            results["contradictions"] = contradictions
            results["executed_nodes"].append("contradiction_detection")

            # Register evidence for D5 questions
            for contra in contradictions.get("matches", [])[:10]:
                self.evidence_registry.register(
                    source_component="contradiction_detector",
                    evidence_type="contradiction",
                    content=contra,
                    confidence=contra.get("confidence", 0.5),
                    applicable_questions=[f"D5-Q{i}" for i in range(1, 51)]
                )

            # 7. Monetary detection
            logger.info("[7/11] Monetary Detection...")
            monetary = self._execute_monetary_detection(sanitized_text)
            results["monetary"] = monetary
            results["executed_nodes"].append("monetary_detection")

            # Register evidence for D3 questions
            for mon in monetary:
                self.evidence_registry.register(
                    source_component="monetary_detector",
                    evidence_type="monetary_value",
                    content=mon,
                    confidence=mon.get("confidence", 0.8),
                    applicable_questions=[f"D3-Q{i}" for i in range(1, 51)]
                )

            # 8. Feasibility scoring
            logger.info("[8/11] Feasibility Scoring...")
            feasibility = self._execute_feasibility_scoring(sanitized_text)
            results["feasibility"] = feasibility
            results["executed_nodes"].append("feasibility_scoring")

            # Register evidence for D1 questions
            self.evidence_registry.register(
                source_component="feasibility_scorer",
                evidence_type="baseline_presence",
                content={"has_baseline": feasibility.get("has_baseline", False)},
                confidence=0.9 if feasibility.get("has_baseline") else 0.1,
                applicable_questions=[f"D1-Q{i}" for i in range(1, 51)]
            )

            # 9. Causal pattern detection
            logger.info("[9/11] Causal Pattern Detection...")
            causal_patterns = self._execute_causal_detection(sanitized_text)
            results["causal_patterns"] = causal_patterns
            results["executed_nodes"].append("causal_detection")

            # Register evidence for D2 questions
            for pattern in causal_patterns:
                self.evidence_registry.register(
                    source_component="causal_pattern_detector",
                    evidence_type="causal_mechanism",
                    content=pattern,
                    confidence=pattern.get("confidence", 0.6),
                    applicable_questions=[f"D2-Q{i}" for i in range(1, 51)]
                )

            # 10. Theory of Change
            logger.info("[10/11] Theory of Change...")
            teoria_cambio = self._create_teoria_cambio(
                sanitized_text, responsibilities, causal_patterns, monetary
            )
            teoria_validation = self._validate_teoria_cambio(teoria_cambio)
            results["teoria_cambio"] = teoria_validation
            results["executed_nodes"].append("teoria_cambio")

            # Register ToC evidence for D6 questions
            self.evidence_registry.register(
                source_component="teoria_cambio",
                evidence_type="theory_of_change",
                content=teoria_validation,
                confidence=0.8 if teoria_validation.get("is_valid") else 0.3,
                applicable_questions=[f"D6-Q{i}" for i in range(1, 51)]
            )

            # 11. DAG Validation
            logger.info("[11/12] DAG Validation...")
            dag_results = {
                "is_acyclic": self.dag_validator.is_acyclic(),
                "node_count": len(list(self.dag_validator.dag.nodes())),
                "edge_count": len(list(self.dag_validator.dag.edges()))
            }
            results["dag_validation"] = dag_results
            results["executed_nodes"].append("dag_validation")

            # 12. Questionnaire Engine Evaluation (300 questions)
            logger.info("[12/12] Questionnaire Engine - 300 Question Evaluation...")
            logger.info("  → Evaluating 30 questions × 10 thematic points = 300 evaluations")

            # Extract municipality and department if available
            municipality = results.get("metadata", {}).get("municipality", "")
            department = results.get("metadata", {}).get("department", "")

            # Execute full questionnaire evaluation using orchestrator results
            questionnaire_results = self._execute_questionnaire_evaluation(
                results, municipality, department
            )
            results["questionnaire_evaluation"] = questionnaire_results
            results["executed_nodes"].append("questionnaire_evaluation")

            # Register questionnaire evidence
            if questionnaire_results and "point_scores" in questionnaire_results:
                for point_id, point_data in questionnaire_results.get("point_scores", {}).items():
                    self.evidence_registry.register(
                        source_component="questionnaire_engine",
                        evidence_type="structured_evaluation",
                        content={
                            "point_id": point_id,
                            "score": point_data.get("score_percentage", 0),
                            "classification": point_data.get("classification", {}).get("name", "")
                        },
                        confidence=0.95,
                        applicable_questions=[f"{point_id}-D{d}-Q{q}" for d in range(1, 7) for q in range(1, 6)]
                    )

            logger.info(f"  ✅ Questionnaire evaluation completed: {questionnaire_results.get('metadata', {}).get('total_evaluations', 0)} questions evaluated")

            # Freeze evidence registry
            logger.info("Freezing evidence registry...")
            self.evidence_registry.freeze()

            # Get evidence statistics
            results["evidence_registry"] = {
                "statistics": self.evidence_registry.get_statistics()
            }

            # Execution summary
            results["execution_summary"] = self.context.get_execution_summary()

            # Immutability proof
            self.immutability_contract.register_process_execution(results)
            results["immutability_proof"] = {
                "result_hash": self.immutability_contract.generate_result_hash(results),
                "evidence_hash": self.evidence_registry.deterministic_hash()
            }

            logger.info("✅ Canonical flow completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in canonical flow: {e}", exc_info=True)
            self.context.register_error("canonical_flow", type(e).__name__, str(e), is_fatal=True)
            return {
                "error": f"Pipeline error: {str(e)}",
                "plan_path": plan_path,
                "executed_nodes": results.get("executed_nodes", []),
                "execution_summary": self.context.get_execution_summary()
            }

    @component_execution("sanitization")
    def _execute_sanitization(self, text: str) -> str:
        """Execute text sanitization"""
        start_time = time.perf_counter()
        try:
            result = self.sanitizer.sanitize(text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("sanitization", latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("sanitization", latency_ms)
            raise

    @component_execution("plan_processing")
    def _execute_plan_processing(self, text: str) -> Dict[str, Any]:
        """Execute plan processing"""
        start_time = time.perf_counter()
        try:
            result = self.processor.process(text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("plan_processing", latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("plan_processing", latency_ms)
            raise

    @component_execution("segmentation")
    def _execute_segmentation(self, text: str) -> List[str]:
        """Execute document segmentation"""
        start_time = time.perf_counter()
        try:
            result = self.segmenter.segment(text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("document_segmentation", latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("document_segmentation", latency_ms)
            raise

    @component_execution("embedding")
    def _execute_embedding(self, segments: List[str]) -> List[Any]:
        """Execute text embedding with circuit breaker protection"""
        circuit = self.circuit_breakers.get("embedding")
        start_time = time.perf_counter()
        
        def _embed():
            embeddings_array = self.embedding_model.embed(segments)
            if hasattr(embeddings_array, 'tolist'):
                return embeddings_array.tolist()
            return list(embeddings_array)
        
        try:
            if circuit:
                result = circuit.call(_embed)
            else:
                result = _embed()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("embedding", latency_ms)
            return result
        except CircuitBreakerError:
            logger.warning("Embedding circuit breaker is OPEN, returning empty embeddings")
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("embedding", latency_ms)
            return []
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("embedding", latency_ms)
            raise

    @component_execution("responsibility_detection")
    def _execute_responsibility_detection(self, text: str) -> List[Dict[str, Any]]:
        """Execute responsibility entity detection with circuit breaker protection"""
        circuit = self.circuit_breakers.get("responsibility_detection")
        start_time = time.perf_counter()
        
        def _detect():
            entities = self.responsibility_detector.detect_entities(text)
            return [
                {
                    "text": entity.text,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence
                }
                for entity in entities
            ]
        
        try:
            if circuit:
                result = circuit.call(_detect)
            else:
                result = _detect()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("responsibility_detection", latency_ms)
            return result
        except CircuitBreakerError:
            logger.warning("Responsibility detection circuit breaker is OPEN")
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("responsibility_detection", latency_ms)
            return []
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("responsibility_detection", latency_ms)
            raise

    @component_execution("contradiction_detection")
    def _execute_contradiction_detection(self, text: str) -> Dict[str, Any]:
        """Execute contradiction detection with circuit breaker protection"""
        circuit = self.circuit_breakers.get("contradiction_detection")
        start_time = time.perf_counter()
        
        def _detect():
            analysis = self.contradiction_detector.detect_contradictions(text)
            return {
                "total": analysis.total_contradictions,
                "risk_score": analysis.risk_score,
                "risk_level": analysis.risk_level.value,
                "matches": [
                    {
                        "text": c.full_text,
                        "connector": c.adversative_connector,
                        "confidence": c.confidence,
                    }
                    for c in analysis.contradictions
                ]
            }
        
        try:
            if circuit:
                result = circuit.call(_detect)
            else:
                result = _detect()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("contradiction_detection", latency_ms)
            return result
        except CircuitBreakerError:
            logger.warning("Contradiction detection circuit breaker is OPEN")
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("contradiction_detection", latency_ms)
            return {"total": 0, "risk_score": 0.0, "risk_level": "low", "matches": []}
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("contradiction_detection", latency_ms)
            raise

    @component_execution("monetary_detection")
    def _execute_monetary_detection(self, text: str) -> List[Dict[str, Any]]:
        """Execute monetary value detection"""
        start_time = time.perf_counter()
        try:
            monetary_matches = self.monetary_detector.find_monetary_expressions(text)
            result = [
                {
                    "text": match.text,
                    "value": match.value,
                    "currency": match.currency,
                    "confidence": match.confidence,
                }
                for match in monetary_matches
            ]
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("monetary_detection", latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("monetary_detection", latency_ms)
            raise

    @component_execution("feasibility_scoring")
    def _execute_feasibility_scoring(self, text: str) -> Dict[str, Any]:
        """Execute feasibility scoring"""
        start_time = time.perf_counter()
        try:
            feasibility_score = self.feasibility_scorer.score_text(text)
            result = {
                "score": feasibility_score.feasibility_score,
                "has_baseline": feasibility_score.has_baseline,
                "has_target": feasibility_score.has_quantitative_target,
                "has_timeframe": feasibility_score.has_timeframe,
                "detailed_matches": [
                    {
                        "type": match.component_type.value,
                        "text": match.matched_text,
                        "confidence": match.confidence,
                    }
                    for match in feasibility_score.detailed_matches
                ][:10],
            }
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("feasibility_scoring", latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("feasibility_scoring", latency_ms)
            raise

    @component_execution("causal_detection")
    def _execute_causal_detection(self, text: str) -> List[Dict[str, Any]]:
        """Execute causal pattern detection with circuit breaker protection"""
        circuit = self.circuit_breakers.get("causal_detection")
        start_time = time.perf_counter()
        
        def _detect():
            patterns = self.causal_detector.detect_patterns(text)
            return [
                {
                    "text": pattern.text,
                    "cause": pattern.cause,
                    "effect": pattern.effect,
                    "connector": pattern.causal_connector,
                    "confidence": pattern.confidence
                }
                for pattern in patterns
            ]
        
        try:
            if circuit:
                result = circuit.call(_detect)
            else:
                result = _detect()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("causal_detection", latency_ms)
            return result
        except CircuitBreakerError:
            logger.warning("Causal detection circuit breaker is OPEN")
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("causal_detection", latency_ms)
            return []
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("causal_detection", latency_ms)
            raise

    @component_execution("teoria_cambio_creation")
    def _create_teoria_cambio(
        self,
        text: str,
        responsibilities: List[Dict[str, Any]],
        causal_patterns: List[Dict[str, Any]],
        monetary: List[Dict[str, Any]]
    ) -> TeoriaCambio:
        """Create a Theory of Change object"""
        supuestos_causales = [pattern["text"] for pattern in causal_patterns[:5]]

        mediadores = {"institucional": [], "social": []}
        for resp in responsibilities:
            if resp.get("type") in ["institución", "organización"]:
                mediadores["institucional"].append(resp["text"])
            else:
                mediadores["social"].append(resp["text"])

        resultados_intermedios = []
        precondiciones = [item["text"] for item in monetary[:5]]

        return TeoriaCambio(
            supuestos_causales=supuestos_causales,
            mediadores=mediadores,
            resultados_intermedios=resultados_intermedios[:10],
            precondiciones=precondiciones
        )

    @component_execution("teoria_cambio_validation")
    def _validate_teoria_cambio(self, teoria_cambio: TeoriaCambio) -> Dict[str, Any]:
        """Validate a Theory of Change with circuit breaker protection"""
        circuit = self.circuit_breakers.get("teoria_cambio")
        start_time = time.perf_counter()
        
        def _validate():
            graph = teoria_cambio.construir_grafo_causal()
            orden_result = teoria_cambio.validar_orden_causal(graph)
            caminos_result = teoria_cambio.detectar_caminos_completos(graph)
            sugerencias_result = teoria_cambio.generar_sugerencias(graph)

            self._build_dag_from_teoria_cambio(teoria_cambio)

            monte_carlo_result = self.dag_validator.calculate_acyclicity_pvalue(
                "teoria_cambio_validation", iterations=5000
            )

            return {
                "is_valid": orden_result.es_valida and caminos_result.es_valida,
                "order_violations": len(orden_result.violaciones_orden),
                "complete_paths": len(caminos_result.caminos_completos),
                "missing_categories": [cat.name for cat in sugerencias_result.categorias_faltantes],
                "suggestions": sugerencias_result.sugerencias,
                "monte_carlo": {
                    "p_value": monte_carlo_result.p_value,
                    "confidence_interval": monte_carlo_result.confidence_interval,
                },
                "causal_coefficient": teoria_cambio.calcular_coeficiente_causal(),
            }
        
        try:
            if circuit:
                result = circuit.call(_validate)
            else:
                result = _validate()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("teoria_cambio", latency_ms)
            return result
        except CircuitBreakerError:
            logger.warning("Theory of change circuit breaker is OPEN")
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("teoria_cambio", latency_ms)
            return {"is_valid": False, "error": "Circuit breaker open"}
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("teoria_cambio", latency_ms)
            raise

    def _build_dag_from_teoria_cambio(self, teoria_cambio: TeoriaCambio) -> None:
        """Build a DAG validator from a TeoriaCambio object"""
        circuit = self.circuit_breakers.get("dag_validation")
        start_time = time.perf_counter()
        
        def _build():
            self.dag_validator = AdvancedDAGValidator()
            graph = teoria_cambio.construir_grafo_causal()

            for node in graph.nodes():
                self.dag_validator.add_node(node)

            for edge in graph.edges():
                from_node, to_node = edge
                self.dag_validator.add_edge(from_node, to_node)
        
        try:
            if circuit:
                circuit.call(_build)
            else:
                _build()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("dag_validation", latency_ms)
        except CircuitBreakerError:
            logger.warning("DAG validation circuit breaker is OPEN")
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("dag_validation", latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_latency("dag_validation", latency_ms)
            raise

    @component_execution("questionnaire_evaluation")
    def _execute_questionnaire_evaluation(
        self,
        orchestrator_results: Dict[str, Any],
        municipality: str = "",
        department: str = ""
    ) -> Dict[str, Any]:
        """
        Execute 300-question evaluation using QuestionnaireEngine

        Args:
            orchestrator_results: Results from the 11 previous orchestrator steps
            municipality: Municipality name
            department: Department name

        Returns:
            Complete questionnaire evaluation results with 300 questions
        """
        try:
            logger.info("  → Initializing QuestionnaireEngine evaluation...")

            # Execute full evaluation using the questionnaire engine
            questionnaire_results = self.questionnaire_engine.execute_full_evaluation(
                orchestrator_results=orchestrator_results,
                municipality=municipality,
                department=department
            )

            # Extract summary statistics
            total_questions = questionnaire_results.get("metadata", {}).get("total_evaluations", 300)
            global_score = questionnaire_results.get("global_score", {})
            score_percentage = global_score.get("score_percentage", 0.0)
            classification = global_score.get("classification", {})

            logger.info(f"  → Total questions evaluated: {total_questions}")
            logger.info(f"  → Global score: {score_percentage:.1f}%")
            logger.info(f"  → Classification: {classification.get('name', 'N/A')} {classification.get('color', '')}")

            return questionnaire_results

        except Exception as e:
            logger.error(f"Error in questionnaire evaluation: {e}", exc_info=True)
            return {
                "error": f"Questionnaire evaluation failed: {str(e)}",
                "metadata": {
                    "total_evaluations": 0,
                    "status": "error"
                }
            }

    def export_performance_metrics(self, output_dir: str = "."):
        """Export performance metrics and dashboard"""
        prometheus_path = os.path.join(output_dir, "performance_metrics.prom")
        dashboard_path = os.path.join(output_dir, "performance_dashboard.html")
        
        self.performance_monitor.export_prometheus_metrics(prometheus_path)
        self.performance_monitor.generate_dashboard_html(dashboard_path)
        
        logger.info(f"Exported metrics to {prometheus_path}")
        logger.info(f"Generated dashboard at {dashboard_path}")
    
    def check_performance_budgets(self) -> Dict[str, Any]:
        """Check all performance budgets and return violations"""
        violations = []
        passed = []
        
        for node_name in self.performance_monitor.latencies.keys():
            is_passing, message = self.performance_monitor.check_budget_violation(node_name)
            
            if is_passing:
                passed.append({"node": node_name, "message": message})
            else:
                violations.append({"node": node_name, "message": message})
        
        return {
            "total_nodes": len(self.performance_monitor.latencies),
            "passed": len(passed),
            "failed": len(violations),
            "violations": violations,
            "all_passed": passed,
            "ci_gate_status": "PASS" if len(violations) == 0 else "FAIL"
        }
    
    def get_circuit_breaker_health(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers"""
        return {
            circuit_name: circuit.get_health_status()
            for circuit_name, circuit in self.circuit_breakers.items()
        }


def main():
    """Example usage of the MINIMINIMOON orchestrator"""
    import sys

    if len(sys.argv) > 1:
        plan_path = sys.argv[1]
        config_path = sys.argv[2] if len(sys.argv) > 2 else None

        orchestrator = MINIMINIMOONOrchestrator(config_path)
        print(f"Processing plan: {plan_path}")

        results = orchestrator.process_plan(plan_path)

        print("\nProcessing Results:")
        print(f"Plan: {results.get('plan_name', 'Unknown')}")

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"\nNodes executed: {len(results.get('executed_nodes', []))}")
            print(f"Evidence items: {results.get('evidence_registry', {}).get('statistics', {}).get('total_evidence', 0)}")
            print(f"Evidence hash: {results.get('immutability_proof', {}).get('evidence_hash', 'N/A')[:32]}...")
        
        # Export performance metrics
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        orchestrator.export_performance_metrics()
        
        # Check performance budgets
        budget_check = orchestrator.check_performance_budgets()
        print(f"\nBudget Check: {budget_check['ci_gate_status']}")
        print(f"Passed: {budget_check['passed']}/{budget_check['total_nodes']}")
        
        if budget_check['violations']:
            print("\n⚠️  Performance Budget Violations:")
            for violation in budget_check['violations']:
                print(f"  - {violation['message']}")
        
        # Circuit breaker health
        cb_health = orchestrator.get_circuit_breaker_health()
        open_circuits = sum(1 for h in cb_health.values() if h['state'] == 'open')
        if open_circuits > 0:
            print(f"\n🔴 {open_circuits} circuit breaker(s) OPEN")
    else:
        print("Usage: python miniminimoon_orchestrator.py <plan_file_path> [config_path]")


if __name__ == "__main__":
    main()
