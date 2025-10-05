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
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps
from pathlib import Path

# Import core processing components
from plan_sanitizer import PlanSanitizer
from plan_processor import PlanProcessor
from document_segmenter import DocumentSegmenter
from embedding_model import EmbeddingModel, create_embedding_model
from document_embedding_mapper import DocumentEmbeddingMapper
from spacy_loader import SpacyModelLoader, SafeSpacyProcessor

# Import analysis components
from responsibility_detector import ResponsibilityDetector
from contradiction_detector import ContradictionDetector
from monetary_detector import MonetaryDetector
from feasibility_scorer import FeasibilityScorer
from teoria_cambio import TeoriaCambio
from dag_validation import AdvancedDAGValidator
from causal_pattern_detector import CausalPatternDetector

# Import unified system components
from evidence_registry import EvidenceRegistry, CanonicalEvidence
from data_flow_contract import CanonicalFlowValidator, DataType
from system_validators import SystemHealthValidator
from miniminimoon_immutability import ImmutabilityContract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MINIMINIMOONOrchestrator")


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

    def _initialize_components(self):
        """Initialize all system components in the correct order"""
        try:
            # Core processing components
            logger.info("Initializing core processing components...")
            self.sanitizer = PlanSanitizer(self.config.get("sanitization_rules", None))
            self.processor = PlanProcessor()
            self.segmenter = DocumentSegmenter(
                strategy=self.config.get("segmentation_strategy", "paragraph")
            )

            # Embedding and NLP components
            logger.info("Initializing embedding and NLP components...")
            self.embedding_model = create_embedding_model()
            self.spacy_loader = SpacyModelLoader()
            self.spacy_processor = SafeSpacyProcessor(self.spacy_loader)
            self.doc_mapper = DocumentEmbeddingMapper(
                self.embedding_model,
                cache_enabled=self.config.get("cache_embeddings", True)
            )

            # Analysis components
            logger.info("Initializing analysis components...")
            self.responsibility_detector = ResponsibilityDetector()
            self.contradiction_detector = ContradictionDetector()
            self.monetary_detector = MonetaryDetector()
            self.feasibility_scorer = FeasibilityScorer(
                enable_parallel=self.config.get("parallel_processing", True)
            )
            self.causal_detector = CausalPatternDetector()
            self.dag_validator = AdvancedDAGValidator()

            # Unified system components
            logger.info("Initializing unified system components...")
            self.evidence_registry = EvidenceRegistry()
            self.flow_validator = CanonicalFlowValidator()
            self.system_health_validator = SystemHealthValidator()

            # Verify system components
            verification_level = self.config.get("verification_level", "normal")
            self.immutability_contract.verify_components(verification_level)

            # Register initialization success
            for component_name in [
                "sanitizer", "processor", "segmenter", "embedding_model",
                "spacy_processor", "doc_mapper", "responsibility_detector",
                "contradiction_detector", "monetary_detector", "feasibility_scorer",
                "causal_detector", "dag_validator", "evidence_registry",
                "flow_validator", "system_health_validator"
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
            logger.info("[11/11] DAG Validation...")
            dag_results = {
                "is_acyclic": self.dag_validator.is_acyclic(),
                "node_count": len(list(self.dag_validator.dag.nodes())),
                "edge_count": len(list(self.dag_validator.dag.edges()))
            }
            results["dag_validation"] = dag_results
            results["executed_nodes"].append("dag_validation")

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
        return self.sanitizer.sanitize(text)

    @component_execution("plan_processing")
    def _execute_plan_processing(self, text: str) -> Dict[str, Any]:
        """Execute plan processing"""
        return self.processor.process(text)

    @component_execution("segmentation")
    def _execute_segmentation(self, text: str) -> List[str]:
        """Execute document segmentation"""
        return self.segmenter.segment(text)

    @component_execution("embedding")
    def _execute_embedding(self, segments: List[str]) -> List[Any]:
        """Execute text embedding"""
        embeddings_array = self.embedding_model.embed(segments)
        if hasattr(embeddings_array, 'tolist'):
            return embeddings_array.tolist()
        return list(embeddings_array)

    @component_execution("responsibility_detection")
    def _execute_responsibility_detection(self, text: str) -> List[Dict[str, Any]]:
        """Execute responsibility entity detection"""
        entities = self.responsibility_detector.detect_entities(text)
        return [
            {
                "text": entity.text,
                "type": entity.entity_type.value,
                "confidence": entity.confidence
            }
            for entity in entities
        ]

    @component_execution("contradiction_detection")
    def _execute_contradiction_detection(self, text: str) -> Dict[str, Any]:
        """Execute contradiction detection"""
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

    @component_execution("monetary_detection")
    def _execute_monetary_detection(self, text: str) -> List[Dict[str, Any]]:
        """Execute monetary value detection"""
        monetary_matches = self.monetary_detector.find_monetary_expressions(text)
        return [
            {
                "text": match.text,
                "value": match.value,
                "currency": match.currency,
                "confidence": match.confidence,
            }
            for match in monetary_matches
        ]

    @component_execution("feasibility_scoring")
    def _execute_feasibility_scoring(self, text: str) -> Dict[str, Any]:
        """Execute feasibility scoring"""
        feasibility_score = self.feasibility_scorer.score_text(text)
        return {
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

    @component_execution("causal_detection")
    def _execute_causal_detection(self, text: str) -> List[Dict[str, Any]]:
        """Execute causal pattern detection"""
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
        """Validate a Theory of Change"""
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

    def _build_dag_from_teoria_cambio(self, teoria_cambio: TeoriaCambio) -> None:
        """Build a DAG validator from a TeoriaCambio object"""
        self.dag_validator = AdvancedDAGValidator()
        graph = teoria_cambio.construir_grafo_causal()

        for node in graph.nodes():
            self.dag_validator.add_node(node)

        for edge in graph.edges():
            from_node, to_node = edge
            self.dag_validator.add_edge(from_node, to_node)


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
    else:
        print("Usage: python miniminimoon_orchestrator.py <plan_file_path> [config_path]")


if __name__ == "__main__":
    main()

