"""
MINIMINIMOON System Integration
==============================

Central integration module that connects all system components for maximum performance.
This module wires together all detectors, validators, and scoring mechanisms to provide
deep analysis and scoring of questionnaires using DNP standards.

Usage:
    from miniminimoon_system import MINIMINIMOONSystem
    system = MINIMINIMOONSystem()
    results = system.analyze_plan("path/to/plan.txt")
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import all system components
from contradiction_detector import ContradictionDetector
from dag_validation import AdvancedDAGValidator, create_sample_causal_graph
from decalogo_loader import get_decalogo_industrial, load_decalogo_industrial
from embedding_model import EmbeddingModel, create_embedding_model
from feasibility_scorer import FeasibilityScorer
from monetary_detector import MonetaryDetector
from responsibility_detector import ResponsibilityDetector
from spacy_loader import SpacyModelLoader, SafeSpacyProcessor
from teoria_cambio import TeoriaCambio, ValidacionResultado

# Import evaluation system
from Decatalogo_evaluador import IndustrialDecatalogoEvaluatorFull
from Decatalogo_principal import (
    DimensionDecalogo,
    SistemaEvaluacionIndustrial,
    obtener_decalogo_contexto,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MINIMINIMOONSystem")

class MINIMINIMOONSystem:
    """
    Integrated system for deep analysis and scoring using all MINIMINIMOON components.
    
    This class connects all detectors, validators, and analysis tools to provide
    comprehensive evaluation of policy documents and plans using DNP standards.
    
    Attributes:
        embedding_model: Text embedding model with fallback
        contradiction_detector: Text contradiction detector
        responsibility_detector: Entity responsibility detector
        feasibility_scorer: Plan feasibility scoring system
        spacy_processor: Text processor with NLP capabilities
        monetary_detector: Monetary value detection in text
        dag_validator: DAG validation for causal relationships
        decalogo_evaluator: Full decalogo evaluation system
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the integrated MINIMINIMOON system.
        
        Args:
            config_path: Optional path to configuration file
        """
        logger.info("Initializing MINIMINIMOON System...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load DNP standards
        self.standards = self._load_standards()
        
        # Initialize all components with optimal settings
        logger.info("Initializing component: Embedding Model")
        self.embedding_model = create_embedding_model()
        
        logger.info("Initializing component: Contradiction Detector")
        self.contradiction_detector = ContradictionDetector()
        
        logger.info("Initializing component: Spacy Model Loader")
        self.spacy_loader = SpacyModelLoader()
        self.spacy_processor = SafeSpacyProcessor(self.spacy_loader)
        
        logger.info("Initializing component: Responsibility Detector")
        self.responsibility_detector = ResponsibilityDetector()
        
        logger.info("Initializing component: Feasibility Scorer")
        self.feasibility_scorer = FeasibilityScorer(enable_parallel=True)
        
        logger.info("Initializing component: Monetary Detector")
        self.monetary_detector = MonetaryDetector()
        
        logger.info("Initializing component: DAG Validator")
        self.dag_validator = AdvancedDAGValidator()
        
        # Initialize evaluation systems
        logger.info("Initializing component: Decalogo Evaluator")
        self.decalogo_context = obtener_decalogo_contexto()
        self.decalogo_evaluator = IndustrialDecatalogoEvaluatorFull(self.decalogo_context)
        
        logger.info("✅ MINIMINIMOON System initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "parallel_processing": True,
            "embedding_batch_size": 32,
            "context_window_size": 150,
            "decalogo_path": "decalogo_industrial.txt",
            "standards_path": "DNP_STANDARDS.json",
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
    
    def _load_standards(self) -> Dict[str, Any]:
        """Load DNP standards from JSON file."""
        try:
            standards_path = self.config.get("standards_path", "DNP_STANDARDS.json")
            if os.path.exists(standards_path):
                with open(standards_path, 'r', encoding='utf-8') as f:
                    standards = json.load(f)
                logger.info(f"Loaded DNP standards from {standards_path}")
                return standards
            else:
                logger.warning(f"Standards file not found at {standards_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading DNP standards: {e}")
            return {}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis using all available detectors.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info("Starting comprehensive text analysis...")
        results = {}
        
        # Text embeddings
        try:
            embeddings = self.embedding_model.embed([text])
            results["embedding"] = {
                "vector": embeddings[0].tolist() if len(embeddings) > 0 else [],
                "dimension": len(embeddings[0]) if len(embeddings) > 0 else 0,
                "model_info": self.embedding_model.get_model_info(),
            }
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            results["embedding"] = {"error": str(e)}
        
        # Contradiction detection
        try:
            contradiction_analysis = self.contradiction_detector.detect_contradictions(text)
            results["contradictions"] = {
                "total": contradiction_analysis.total_contradictions,
                "risk_score": contradiction_analysis.risk_score,
                "risk_level": contradiction_analysis.risk_level.value,
                "matches": [
                    {
                        "text": c.full_text,
                        "connector": c.adversative_connector,
                        "risk_level": c.risk_level.value,
                        "confidence": c.confidence,
                    }
                    for c in contradiction_analysis.contradictions
                ],
            }
        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")
            results["contradictions"] = {"error": str(e)}
        
        # NLP processing
        try:
            nlp_results = self.spacy_processor.process_text(text)
            results["nlp_analysis"] = {
                "tokens": nlp_results["tokens"][:100],  # Limit for readability
                "entities": nlp_results.get("entities", [])[:20],
                "is_degraded": nlp_results.get("is_degraded", False),
            }
        except Exception as e:
            logger.error(f"Error in NLP processing: {e}")
            results["nlp_analysis"] = {"error": str(e)}
        
        # Responsibility detection
        try:
            responsibility_entities = self.responsibility_detector.detect_entities(text)
            results["responsibilities"] = [
                {
                    "text": entity.text,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "role": getattr(entity, "role", None),
                }
                for entity in responsibility_entities
            ]
        except Exception as e:
            logger.error(f"Error detecting responsibilities: {e}")
            results["responsibilities"] = {"error": str(e)}
        
        # Feasibility scoring
        try:
            feasibility_score = self.feasibility_scorer.score_text(text)
            results["feasibility"] = {
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
        except Exception as e:
            logger.error(f"Error scoring feasibility: {e}")
            results["feasibility"] = {"error": str(e)}
        
        # Monetary detection
        try:
            monetary_matches = self.monetary_detector.find_monetary_expressions(text)
            results["monetary"] = [
                {
                    "text": match.text,
                    "value": match.value,
                    "currency": match.currency,
                    "confidence": match.confidence,
                }
                for match in monetary_matches
            ]
        except Exception as e:
            logger.error(f"Error detecting monetary values: {e}")
            results["monetary"] = {"error": str(e)}
        
        logger.info("Comprehensive text analysis complete")
        return results
    
    def validate_teoria_cambio(self, teoria_cambio: TeoriaCambio) -> Dict[str, Any]:
        """
        Validate a Theory of Change using DAG validation.
        
        Args:
            teoria_cambio: TeoriaCambio object to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating Theory of Change...")
        
        try:
            # Build causal graph from teoria_cambio
            graph = teoria_cambio.construir_grafo_causal()
            
            # Validate order of causality
            orden_result = teoria_cambio.validar_orden_causal(graph)
            
            # Detect complete paths
            caminos_result = teoria_cambio.detectar_caminos_completos(graph)
            
            # Generate suggestions
            sugerencias_result = teoria_cambio.generar_sugerencias(graph)
            
            # Build DAG for advanced validation
            self._build_dag_from_teoria_cambio(teoria_cambio)
            
            # Perform Monte Carlo validation
            monte_carlo_result = self.dag_validator.calculate_acyclicity_pvalue_advanced(
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
                    "bayesian_posterior": monte_carlo_result.bayesian_posterior,
                    "robustness_score": monte_carlo_result.robustness_score,
                },
                "causal_coefficient": teoria_cambio.calcular_coeficiente_causal(),
                "identifiability": teoria_cambio.verificar_identificabilidad(),
            }
        except Exception as e:
            logger.error(f"Error validating Theory of Change: {e}")
            return {"error": str(e)}
    
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
    
    def evaluate_decalogo_dimension(self, text: str, dimension_id: int) -> Dict[str, Any]:
        """
        Evaluate text against a specific Decalogo dimension.
        
        Args:
            text: Text to evaluate
            dimension_id: ID of the dimension to evaluate against (1-10)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating text against Decalogo dimension {dimension_id}...")
        
        try:
            # Create dimension object
            dimension = DimensionDecalogo(
                dimension_id, 
                f"Dimension {dimension_id}", 
                "Evaluación automática"
            )
            
            # Extract evidence from text
            evidencia = self._extract_evidence_from_text(text)
            
            # Evaluate dimension using the evaluator
            evaluacion_punto, analisis, resultado = self.decalogo_evaluator.evaluar_punto_completo(
                evidencia, dimension_id
            )
            
            # Return relevant results
            return {
                "dimension_id": dimension_id,
                "score": evaluacion_punto.puntaje_agregado_punto,
                "classification": evaluacion_punto.clasificacion_cualitativa,
                "explanation": evaluacion_punto.explicacion_extensa,
                "dimensional_scores": {
                    ev.dimension: ev.puntaje_dimension
                    for ev in evaluacion_punto.evaluaciones_dimensiones
                },
                "gaps": evaluacion_punto.brechas_identificadas,
                "recommendations": evaluacion_punto.recomendaciones_especificas,
                "strengths": evaluacion_punto.fortalezas_identificadas,
                "risks": evaluacion_punto.riesgos_detectados,
                "analysis": {
                    "confidence": analisis.confidence_global,
                    "has_baseline": analisis.tiene_linea_base,
                    "has_targets": analisis.tiene_metas,
                    "contradictions": analisis.riesgos,
                }
            }
        except Exception as e:
            logger.error(f"Error evaluating against Decalogo dimension {dimension_id}: {e}")
            return {"error": str(e), "dimension_id": dimension_id}
    
    def evaluate_full_plan(self, plan_text: str, plan_name: str) -> Dict[str, Any]:
        """
        Perform a complete evaluation of a plan against all Decalogo dimensions.
        
        Args:
            plan_text: Full text of the plan
            plan_name: Name of the plan
            
        Returns:
            Dictionary with complete evaluation results
        """
        logger.info(f"Starting full evaluation of plan: {plan_name}")
        
        try:
            # Extract evidence for all dimensions
            evidencias_por_punto = {}
            for dim_id in range(1, 11):
                evidencias_por_punto[dim_id] = self._extract_evidence_from_text(
                    plan_text, dimension_id=dim_id
                )
            
            # Generate complete report
            reporte = self.decalogo_evaluator.generar_reporte_final(
                evidencias_por_punto, plan_name
            )
            
            # Extract key results for the return value
            result = {
                "plan_name": plan_name,
                "global_score": reporte.resumen_ejecutivo["puntaje_global"],
                "alignment_level": reporte.resumen_ejecutivo["nivel_alineacion"],
                "strategic_recommendation": reporte.resumen_ejecutivo["recomendacion_estrategica_global"],
                "confidence": reporte.resumen_ejecutivo["nivel_confianza_evaluacion"],
                "dimensions_evaluated": len(reporte.reporte_por_punto),
                "questions_evaluated": len(reporte.reporte_por_pregunta),
                "clusters_evaluated": len(reporte.reporte_meso_por_cluster),
                "dimension_scores": {
                    p.punto_id: p.puntaje_agregado_punto
                    for p in reporte.reporte_por_punto
                },
                "global_gaps": reporte.reporte_macro["brechas_globales"][:5],
                "global_recommendations": reporte.reporte_macro["recomendaciones_globales"][:5],
                "evaluation_date": reporte.metadata["fecha_evaluacion"],
            }
            
            logger.info(f"Full evaluation complete for plan: {plan_name}")
            return result
        except Exception as e:
            logger.error(f"Error in full plan evaluation: {e}")
            return {"error": str(e), "plan_name": plan_name}
    
    def _extract_evidence_from_text(
        self, text: str, dimension_id: Optional[int] = None
    ) -> Dict[str, List[Any]]:
        """Extract structured evidence from text for evaluation."""
        # Extract all possible evidence types
        evidencia = {}
        
        # Basic text segments
        text_segments = text.split("\n\n")
        evidencia["texto_completo"] = [text]
        evidencia["segmentos"] = text_segments[:20]
        
        # Run analysis
        analysis = self.analyze_text(text)
        
        # Extract indicators and targets
        if "feasibility" in analysis and "detailed_matches" in analysis["feasibility"]:
            indicators = []
            targets = []
            baselines = []
            for match in analysis["feasibility"]["detailed_matches"]:
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
        if "responsibilities" in analysis and isinstance(analysis["responsibilities"], list):
            evidencia["responsables"] = [r["text"] for r in analysis["responsibilities"]]
        
        # Extract monetary values
        if "monetary" in analysis and isinstance(analysis["monetary"], list):
            evidencia["presupuesto"] = [m["text"] for m in analysis["monetary"]]
            evidencia["valores_monetarios"] = analysis["monetary"]
        
        # Extract contradictions
        if "contradictions" in analysis and "matches" in analysis["contradictions"]:
            evidencia["contradicciones"] = [
                c["text"] for c in analysis["contradictions"]["matches"]
            ]
        
        # Additional evidence extraction specific to dimension
        if dimension_id is not None:
            if dimension_id in [1, 2, 8]:  # Security and justice related
                keywords = ["seguridad", "justicia", "paz", "convivencia", "derechos"]
                evidencia["seguridad_justicia"] = self._extract_segments_with_keywords(
                    text, keywords
                )
                
            elif dimension_id in [3, 5]:  # Vulnerable populations
                keywords = ["mujer", "género", "vulnerables", "víctimas", "inclusión"]
                evidencia["poblacion_vulnerable"] = self._extract_segments_with_keywords(
                    text, keywords
                )
                
            elif dimension_id in [6, 7]:  # Education and health
                keywords = ["educación", "salud", "bienestar", "calidad de vida"]
                evidencia["educacion_salud"] = self._extract_segments_with_keywords(
                    text, keywords
                )
                
            elif dimension_id in [4, 9]:  # Economic development
                keywords = ["economía", "desarrollo", "empleo", "trabajo", "ingresos"]
                evidencia["desarrollo_economico"] = self._extract_segments_with_keywords(
                    text, keywords
                )
                
            elif dimension_id in [10]:  # Housing and environment
                keywords = ["vivienda", "hábitat", "ambiente", "sostenible"]
                evidencia["vivienda_ambiente"] = self._extract_segments_with_keywords(
                    text, keywords
                )
        
        return evidencia
    
    def _extract_segments_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract text segments containing specific keywords."""
        segments = []
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                segments.append(paragraph)
        
        return segments
    
    def analyze_plan(self, plan_path: str) -> Dict[str, Any]:
        """
        Analyze a plan document from file with all available tools.
        
        Args:
            plan_path: Path to the plan document
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            # Load plan text
            with open(plan_path, 'r', encoding='utf-8') as f:
                plan_text = f.read()
            
            # Get plan name from filename
            plan_name = os.path.basename(plan_path).split('.')[0]
            
            # Perform comprehensive analysis
            text_analysis = self.analyze_text(plan_text)
            
            # Create a basic Theory of Change
            teoria_cambio = self._create_teoria_cambio_from_text(plan_text)
            teoria_validation = self.validate_teoria_cambio(teoria_cambio)
            
            # Full plan evaluation
            plan_evaluation = self.evaluate_full_plan(plan_text, plan_name)
            
            # Combine all results
            return {
                "plan_name": plan_name,
                "file_path": plan_path,
                "text_analysis": text_analysis,
                "teoria_cambio_validation": teoria_validation,
                "plan_evaluation": plan_evaluation,
            }
        except Exception as e:
            logger.error(f"Error analyzing plan {plan_path}: {e}")
            return {
                "error": str(e),
                "plan_path": plan_path,
            }
    
    def _create_teoria_cambio_from_text(self, text: str) -> TeoriaCambio:
        """Create a Theory of Change object from text."""
        # Extract key elements for Theory of Change
        analysis = self.analyze_text(text)
        
        # Extract inputs
        supuestos_causales = []
        if "nlp_analysis" in analysis and "tokens" in analysis["nlp_analysis"]:
            # Use first 50 tokens as causal assumptions for demonstration
            supuestos_causales = [" ".join(analysis["nlp_analysis"]["tokens"][:50])]
        
        # Extract responsibilities as mediators
        mediadores = {"institucional": [], "social": []}
        if "responsibilities" in analysis and isinstance(analysis["responsibilities"], list):
            for resp in analysis["responsibilities"]:
                if resp.get("type") == "institución" or resp.get("type") == "organización":
                    mediadores["institucional"].append(resp["text"])
                else:
                    mediadores["social"].append(resp["text"])
        
        # Extract feasibility indicators as results
        resultados_intermedios = []
        if "feasibility" in analysis and "detailed_matches" in analysis["feasibility"]:
            for match in analysis["feasibility"]["detailed_matches"]:
                resultados_intermedios.append(match["text"])
        
        # Extract monetary values as preconditions
        precondiciones = []
        if "monetary" in analysis and isinstance(analysis["monetary"], list):
            for monetary in analysis["monetary"]:
                precondiciones.append(f"{monetary['text']}")
        
        # Create the Theory of Change object
        return TeoriaCambio(
            supuestos_causales=supuestos_causales,
            mediadores=mediadores,
            resultados_intermedios=resultados_intermedios[:10],
            precondiciones=precondiciones,
        )

    def batch_process_plans(self, directory_path: str) -> Dict[str, Any]:
        """
        Process multiple plans in a directory.
        
        Args:
            directory_path: Path to directory containing plans
            
        Returns:
            Dictionary with results for each plan
        """
        results = {}
        
        try:
            # Get all text files in directory
            plan_files = [
                os.path.join(directory_path, f) 
                for f in os.listdir(directory_path) 
                if f.endswith(('.txt', '.pdf', '.docx'))
            ]
            
            logger.info(f"Found {len(plan_files)} plans to process")
            
            # Process each plan
            for i, plan_path in enumerate(plan_files):
                logger.info(f"Processing plan {i+1}/{len(plan_files)}: {plan_path}")
                
                try:
                    # Extract text from file
                    if plan_path.endswith('.txt'):
                        with open(plan_path, 'r', encoding='utf-8') as f:
                            plan_text = f.read()
                    elif plan_path.endswith('.pdf'):
                        # Placeholder for PDF extraction
                        logger.warning(f"PDF extraction not implemented: {plan_path}")
                        continue
                    elif plan_path.endswith('.docx'):
                        # Placeholder for DOCX extraction
                        logger.warning(f"DOCX extraction not implemented: {plan_path}")
                        continue
                    
                    # Get plan name from filename
                    plan_name = os.path.basename(plan_path).split('.')[0]
                    
                    # Evaluate plan
                    evaluation = self.evaluate_full_plan(plan_text, plan_name)
                    results[plan_name] = evaluation
                    
                except Exception as e:
                    logger.error(f"Error processing plan {plan_path}: {e}")
                    results[os.path.basename(plan_path)] = {"error": str(e)}
            
            # Add summary statistics
            scores = [
                r["global_score"] for r in results.values() 
                if isinstance(r, dict) and "global_score" in r
            ]
            
            if scores:
                import numpy as np
                results["_summary"] = {
                    "total_plans": len(plan_files),
                    "processed_plans": len(scores),
                    "average_score": np.mean(scores),
                    "min_score": np.min(scores),
                    "max_score": np.max(scores),
                    "std_dev": np.std(scores),
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {"error": str(e)}


# Example usage
def main():
    """Example usage of the integrated MINIMINIMOON system."""
    import sys
    
    # Initialize system
    system = MINIMINIMOONSystem()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Process a single plan file
        if os.path.isfile(sys.argv[1]):
            print(f"Analyzing plan: {sys.argv[1]}")
            results = system.analyze_plan(sys.argv[1])
            print("\nAnalysis Results:")
            print(f"Plan Name: {results.get('plan_name', 'Unknown')}")
            
            if "plan_evaluation" in results:
                eval_results = results["plan_evaluation"]
                print(f"Global Score: {eval_results.get('global_score', 'N/A')}")
                print(f"Alignment Level: {eval_results.get('alignment_level', 'N/A')}")
                print(f"Strategic Recommendation: {eval_results.get('strategic_recommendation', 'N/A')}")
                
                if "dimension_scores" in eval_results:
                    print("\nDimension Scores:")
                    for dim, score in eval_results["dimension_scores"].items():
                        print(f"  Dimension {dim}: {score:.2f}")
                
                if "global_recommendations" in eval_results:
                    print("\nRecommendations:")
                    for i, rec in enumerate(eval_results["global_recommendations"][:3], 1):
                        print(f"  {i}. {rec}")
        
        # Process multiple plans in a directory
        elif os.path.isdir(sys.argv[1]):
            print(f"Batch processing plans in: {sys.argv[1]}")
            results = system.batch_process_plans(sys.argv[1])
            
            if "_summary" in results:
                summary = results["_summary"]
                print("\nBatch Processing Summary:")
                print(f"Total Plans: {summary['total_plans']}")
                print(f"Processed Plans: {summary['processed_plans']}")
                print(f"Average Score: {summary['average_score']:.2f}")
                print(f"Score Range: {summary['min_score']:.2f} - {summary['max_score']:.2f}")
                
                # Print top 3 plans by score
                top_plans = sorted(
                    [(name, r["global_score"]) for name, r in results.items() 
                     if isinstance(r, dict) and "global_score" in r],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                print("\nTop Performing Plans:")
                for i, (name, score) in enumerate(top_plans, 1):
                    print(f"  {i}. {name}: {score:.2f}")
    
    else:
        # Interactive mode with sample text
        print("No file specified. Running with sample text...")
        
        sample_text = """
        El objetivo del Plan Municipal de Desarrollo 2024-2027 es aumentar la cobertura educativa al 95% para 2027.
        Sin embargo, los recursos presupuestales han sido reducidos en un 30% respecto al año anterior.
        
        La Secretaría de Educación Municipal liderará el programa de ampliación de cobertura educativa,
        con un presupuesto de $500 millones COP para infraestructura educativa.
        
        Metas principales:
        1. Reducir la deserción escolar del 12% actual al 5% para 2027
        2. Aumentar el promedio de pruebas SABER de 45.6 a 55.0 puntos
        3. Capacitar a 1,500 docentes en nuevas metodologías pedagógicas
        
        El proyecto contempla la construcción de 5 nuevas escuelas en zonas rurales y
        la contratación de 200 nuevos docentes. Se espera beneficiar a 10,000 estudiantes
        de primaria y secundaria.
        """
        
        # Analyze text
        analysis = system.analyze_text(sample_text)
        
        print("\nText Analysis Results:")
        
        if "contradictions" in analysis:
            contra = analysis["contradictions"]
            print(f"Contradictions: {contra.get('total', 0)}")
            print(f"Risk Level: {contra.get('risk_level', 'Unknown')}")
            
            if contra.get("matches"):
                print("\nContradiction found:")
                match = contra["matches"][0]
                print(f"  Text: {match.get('text', '')}")
                print(f"  Connector: {match.get('connector', '')}")
                print(f"  Risk: {match.get('risk_level', '')}")
        
        if "responsibilities" in analysis:
            print(f"\nResponsible Entities: {len(analysis['responsibilities'])}")
            for i, resp in enumerate(analysis["responsibilities"][:3], 1):
                print(f"  {i}. {resp.get('text', '')} ({resp.get('type', '')})")
        
        if "feasibility" in analysis:
            feas = analysis["feasibility"]
            print(f"\nFeasibility Score: {feas.get('score', 0):.2f}")
            print(f"Has Baseline: {feas.get('has_baseline', False)}")
            print(f"Has Target: {feas.get('has_target', False)}")
            print(f"Has Timeframe: {feas.get('has_timeframe', False)}")
        
        if "monetary" in analysis:
            print(f"\nMonetary Expressions: {len(analysis['monetary'])}")
            for i, mon in enumerate(analysis["monetary"][:3], 1):
                print(f"  {i}. {mon.get('text', '')} = {mon.get('value', '')} {mon.get('currency', '')}")
        
        # Create and validate a Theory of Change
        teoria_cambio = system._create_teoria_cambio_from_text(sample_text)
        validation = system.validate_teoria_cambio(teoria_cambio)
        
        print("\nTheory of Change Validation:")
        print(f"Valid: {validation.get('is_valid', False)}")
        print(f"Complete Paths: {validation.get('complete_paths', 0)}")
        print(f"Order Violations: {validation.get('order_violations', 0)}")
        print(f"Causal Coefficient: {validation.get('causal_coefficient', 0):.2f}")
        
        # Evaluate against a dimension
        evaluation = system.evaluate_decalogo_dimension(sample_text, 6)  # Dimension 6 (Education)
        
        print("\nDecalogo Dimension Evaluation:")
        print(f"Dimension: {evaluation.get('dimension_id', 'Unknown')}")
        print(f"Score: {evaluation.get('score', 0):.2f}")
        print(f"Classification: {evaluation.get('classification', 'Unknown')}")
        
        if "dimensional_scores" in evaluation:
            print("\nDimensional Scores:")
            for dim, score in evaluation["dimensional_scores"].items():
                print(f"  {dim}: {score:.2f}")
        
        if "recommendations" in evaluation:
            print("\nRecommendations:")
            for i, rec in enumerate(evaluation["recommendations"][:3], 1):
                print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
