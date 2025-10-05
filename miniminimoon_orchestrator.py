"""
MINIMINIMOON Orchestrator
=========================

Central orchestrator that coordinates all components in the canonical flow of the MINIMINIMOON system.
This module manages the execution sequence, data flow, and component interactions,
ensuring robust integration across all analytical modules.
"""
import logging
import time
import os
import json
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

# Import evaluation components
from decalogo_loader import get_decalogo_industrial, load_decalogo_industrial
from Decatalogo_principal import SistemaEvaluacionIndustrial, obtener_decalogo_contexto
from Decatalogo_evaluador import IndustrialDecatalogoEvaluatorFull

# Import validation and freezing components
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
                # Return None or appropriate fallback
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
            "decalogo_path": "decalogo_industrial.txt",
            "standards_path": "DNP_STANDARDS.json",
            "error_tolerance": "medium",  # low, medium, high
            "processing_priority": "accuracy",  # speed, balanced, accuracy
            "log_level": "INFO",
            "cache_embeddings": True,
            "verification_level": "normal"  # minimal, normal, strict
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # Merge with default config, user config takes precedence
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
            
            # Evaluation components
            logger.info("Initializing evaluation components...")
            self.decalogo_context = obtener_decalogo_contexto()
            self.decalogo_evaluator = IndustrialDecatalogoEvaluatorFull(self.decalogo_context)
            
            # Verify system components against immutability contract
            verification_level = self.config.get("verification_level", "normal")
            self.immutability_contract.verify_components(verification_level)
            
            # Register the initialization success
            for component_name in [
                "sanitizer", "processor", "segmenter", "embedding_model", 
                "spacy_processor", "doc_mapper", "responsibility_detector",
                "contradiction_detector", "monetary_detector", "feasibility_scorer",
                "causal_detector", "dag_validator", "decalogo_evaluator"
            ]:
                self.context.component_status[component_name] = "initialized"
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.context.register_error("initialization", type(e).__name__, str(e), is_fatal=True)
            raise RuntimeError(f"Orchestrator initialization failed: {e}")
    
    @component_execution("plan_processing")
    def process_plan(self, plan_path: str) -> Dict[str, Any]:
        """
        Process a plan document through the canonical flow.
        
        Args:
            plan_path: Path to the plan document file
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Processing plan: {plan_path}")
        results = {"plan_path": plan_path}
        
        # Read the document
        try:
            with open(plan_path, 'r', encoding='utf-8') as f:
                plan_text = f.read()
        except Exception as e:
            logger.error(f"Error reading plan file {plan_path}: {e}")
            self.context.register_error("file_reading", type(e).__name__, str(e), is_fatal=True)
            return {"error": f"Could not read file: {str(e)}", "plan_path": plan_path}
        
        # Extract plan name from filename
        plan_name = os.path.basename(plan_path).split('.')[0]
        results["plan_name"] = plan_name
        
        # Execute the canonical flow
        try:
            # 1. Plan sanitization
            sanitized_text = self._execute_sanitization(plan_text)
            results["sanitization_status"] = "completed"
            
            # 2. Plan processing and metadata extraction
            processed_plan = self._execute_plan_processing(sanitized_text)
            results["metadata"] = processed_plan.get("metadata", {})
            
            # 3. Document segmentation
            segments = self._execute_segmentation(sanitized_text)
            results["segments"] = {"count": len(segments)}
            
            # 4. Embedding generation
            embeddings = self._execute_embedding(segments)
            results["embeddings"] = {"count": len(embeddings), "dimension": embeddings[0].shape[0] if len(embeddings) > 0 else 0}
            
            # 5. Responsibility detection
            responsibilities = self._execute_responsibility_detection(sanitized_text)
            results["responsibilities"] = responsibilities
            
            # 6. Contradiction detection
            contradictions = self._execute_contradiction_detection(sanitized_text)
            results["contradictions"] = contradictions
            
            # 7. Monetary detection
            monetary = self._execute_monetary_detection(sanitized_text)
            results["monetary"] = monetary
            
            # 8. Feasibility scoring
            feasibility = self._execute_feasibility_scoring(sanitized_text)
            results["feasibility"] = feasibility
            
            # 9. Causal pattern detection
            causal_patterns = self._execute_causal_detection(sanitized_text)
            results["causal_patterns"] = causal_patterns
            
            # 10. Theory of Change creation and validation
            teoria_cambio = self._create_teoria_cambio(
                sanitized_text, 
                responsibilities, 
                causal_patterns,
                monetary
            )
            teoria_validation = self._validate_teoria_cambio(teoria_cambio)
            results["teoria_cambio"] = teoria_validation
            
            # 11. Decalogo evaluation
            evaluation = self._execute_decalogo_evaluation(sanitized_text, plan_name)
            results["evaluation"] = evaluation
            
            # Add execution summary
            results["execution_summary"] = self.context.get_execution_summary()
            
            # Generate immutability proof
            self.immutability_contract.register_process_execution(results)
            results["immutability_hash"] = self.immutability_contract.generate_result_hash(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in canonical flow: {e}")
            self.context.register_error("canonical_flow", type(e).__name__, str(e), is_fatal=True)
            return {
                "error": f"Processing error: {str(e)}",
                "plan_path": plan_path,
                "plan_name": plan_name,
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
        return self.embedding_model.embed(segments).tolist()
    
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
        # Extract causal assumptions from causal patterns
        supuestos_causales = [pattern["text"] for pattern in causal_patterns[:5]]
        
        # Extract mediators from responsibilities
        mediadores = {"institucional": [], "social": []}
        for resp in responsibilities:
            if resp.get("type") in ["institución", "organización"]:
                mediadores["institucional"].append(resp["text"])
            else:
                mediadores["social"].append(resp["text"])
        
        # Extract results from feasibility scoring
        resultados_intermedios = []
        feasibility = self._execute_feasibility_scoring(text)
        for match in feasibility.get("detailed_matches", []):
            if match["type"] in ["INDICATOR", "TARGET"]:
                resultados_intermedios.append(match["text"])
        
        # Extract preconditions from monetary expressions
        precondiciones = [item["text"] for item in monetary[:5]]
        
        # Create the Theory of Change
        return TeoriaCambio(
            supuestos_causales=supuestos_causales,
            mediadores=mediadores,
            resultados_intermedios=resultados_intermedios[:10],
            precondiciones=precondiciones
        )
    
    @component_execution("teoria_cambio_validation")
    def _validate_teoria_cambio(self, teoria_cambio: TeoriaCambio) -> Dict[str, Any]:
        """Validate a Theory of Change"""
        # Build causal graph
        graph = teoria_cambio.construir_grafo_causal()
        
        # Validate order of causality
        orden_result = teoria_cambio.validar_orden_causal(graph)
        
        # Detect complete paths
        caminos_result = teoria_cambio.detectar_caminos_completos(graph)
        
        # Generate suggestions
        sugerencias_result = teoria_cambio.generar_sugerencias(graph)
        
        # Prepare DAG for advanced validation
        self._build_dag_from_teoria_cambio(teoria_cambio)
        
        # Perform Monte Carlo validation
        monte_carlo_result = self.dag_validator.calculate_acyclicity_pvalue(
            "teoria_cambio_validation", iterations=5000
        )
        
        # Return comprehensive results
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
        """Build a DAG validator from a TeoriaCambio object."""
        # Reset the validator
        self.dag_validator = AdvancedDAGValidator()
        
        # Get the graph
        graph = teoria_cambio.construir_grafo_causal()
        
        # Add nodes and edges to DAG validator
        for node in graph.nodes():
            self.dag_validator.add_node(node)
        
        for edge in graph.edges():
            from_node, to_node = edge
            self.dag_validator.add_edge(from_node, to_node)
    
    @component_execution("decalogo_evaluation")
    def _execute_decalogo_evaluation(self, text: str, plan_name: str) -> Dict[str, Any]:
        """Execute Decalogo evaluation"""
        # Extract evidence for all dimensions
        evidencias = {}
        for dim_id in range(1, 11):
            evidencias[dim_id] = self._extract_evidence(text, dim_id)
        
        # Generate evaluation report
        reporte = self.decalogo_evaluator.generar_reporte_final(
            evidencias, plan_name
        )
        
        # Extract key results for the return value
        return {
            "global_score": reporte.resumen_ejecutivo["puntaje_global"],
            "alignment_level": reporte.resumen_ejecutivo["nivel_alineacion"],
            "recommendation": reporte.resumen_ejecutivo["recomendacion_estrategica_global"],
            "confidence": reporte.resumen_ejecutivo["nivel_confianza_evaluacion"],
            "dimension_scores": {
                p.punto_id: p.puntaje_agregado_punto
                for p in reporte.reporte_por_punto
            },
            "global_gaps": reporte.reporte_macro["brechas_globales"][:5],
            "global_recommendations": reporte.reporte_macro["recomendaciones_globales"][:5],
        }
    
    def _extract_evidence(self, text: str, dimension_id: int) -> Dict[str, List[Any]]:
        """Extract structured evidence from text for evaluation"""
        evidencia = {}
        
        # Basic text segments
        text_segments = text.split("\n\n")
        evidencia["texto_completo"] = [text]
        evidencia["segmentos"] = text_segments[:20]
        
        # Extract indicators and targets from feasibility scoring
        feasibility = self._execute_feasibility_scoring(text)
        
        # Organize by type
        indicators = []
        targets = []
        baselines = []
        for match in feasibility.get("detailed_matches", []):
            if match["type"] == "INDICATOR":
                indicators.append(match["text"])
            elif match["type"] == "TARGET":
                targets.append(match["text"])
            elif match["type"] == "BASELINE":
                baselines.append(match["text"])
        
        evidencia["indicadores"] = indicators
        evidencia["metas"] = targets
        evidencia["lineas_base"] = baselines
        
        # Extract responsible entities
        responsibilities = self._execute_responsibility_detection(text)
        evidencia["responsables"] = [r["text"] for r in responsibilities]
        
        # Extract monetary values
        monetary = self._execute_monetary_detection(text)
        evidencia["presupuesto"] = [m["text"] for m in monetary]
        
        # Extract contradictions
        contradictions = self._execute_contradiction_detection(text)
        if "matches" in contradictions:
            evidencia["contradicciones"] = [c["text"] for c in contradictions["matches"]]
        
        # Extract dimension-specific evidence
        keywords = self._get_dimension_keywords(dimension_id)
        if keywords:
            dimension_name = f"dimension_{dimension_id}"
            evidencia[dimension_name] = self._extract_segments_with_keywords(text, keywords)
        
        return evidencia
    
    def _get_dimension_keywords(self, dimension_id: int) -> List[str]:
        """Get keywords relevant to a specific dimension"""
        # Maps dimension IDs to relevant keywords
        dimension_keywords = {
            1: ["género", "mujer", "igualdad", "equidad", "discriminación"],
            2: ["seguridad", "paz", "conflicto", "violencia", "víctimas"],
            3: ["ambiente", "clima", "sostenible", "ecológico", "biodiversidad"],
            4: ["económico", "empleo", "productivo", "ingreso", "pobreza"],
            5: ["víctimas", "reparación", "conflicto", "paz", "reconciliación"],
            6: ["juventud", "niñez", "educación", "escuela", "adolescencia"],
            7: ["tierra", "territorio", "rural", "agro", "campesino"],
            8: ["líderes", "defensor", "derechos", "comunidad", "participación"],
            9: ["libertad", "cárcel", "penitenciario", "delito", "reinserción"],
            10: ["migración", "frontera", "migrante", "desplazamiento", "movilidad"]
        }
        
        return dimension_keywords.get(dimension_id, [])
    
    def _extract_segments_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract text segments containing specific keywords"""
        segments = []
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                segments.append(paragraph)
        
        return segments
    
    def freeze_integration(self) -> Dict[str, Any]:
        """
        Freeze the current integration state, creating an immutability record
        that can be used to verify system integrity.
        
        Returns:
            Dictionary with freeze status and verification data
        """
        return self.immutability_contract.freeze_integration()
    
    def verify_integration(self) -> Dict[str, Any]:
        """
        Verify that the integration hasn't been tampered with.
        
        Returns:
            Dictionary with verification results
        """
        return self.immutability_contract.verify_integration()


def main():
    """Example usage of the MINIMINIMOON orchestrator"""
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1] if len(sys.argv) > 2 else None
        orchestrator = MINIMINIMOONOrchestrator(config_path)
        
        plan_path = sys.argv[1]
        print(f"Processing plan: {plan_path}")
        
        results = orchestrator.process_plan(plan_path)
        
        # Print a summary of the results
        print("\nProcessing Results:")
        print(f"Plan: {results.get('plan_name', 'Unknown')}")
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("\nResponsibilities detected:")
            for resp in results.get("responsibilities", [])[:3]:
                print(f"  - {resp['text']} ({resp['type']})")
                
            print("\nContradictions detected:")
            contradictions = results.get("contradictions", {})
            print(f"  Total: {contradictions.get('total', 0)}")
            print(f"  Risk level: {contradictions.get('risk_level', 'Unknown')}")
            
            print("\nFeasibility score:")
            feasibility = results.get("feasibility", {})
            print(f"  Score: {feasibility.get('score', 0):.2f}")
            
            print("\nEvaluation results:")
            evaluation = results.get("evaluation", {})
            print(f"  Global score: {evaluation.get('global_score', 0):.2f}")
            print(f"  Alignment level: {evaluation.get('alignment_level', 'Unknown')}")
            
            print("\nExecution summary:")
            summary = results.get("execution_summary", {})
            print(f"  Total time: {summary.get('total_time', 0):.2f} seconds")
            print(f"  Components executed: {len(summary.get('component_times', {}))}")
            print(f"  Errors encountered: {summary.get('errors', 0)}")
            
            # Show immutability verification
            print("\nImmutability verification:")
            print(f"  Result hash: {results.get('immutability_hash', 'Not generated')}")
    else:
        print("Usage: python miniminimoon_orchestrator.py <plan_file_path> [config_path]")


if __name__ == "__main__":
    main()
