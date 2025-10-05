"""
DECALOGO Pipeline Orchestrator

Serves as the central coordinator for the knowledge extraction pipeline,
ensuring that each component produces the precise evidence needed to
answer specific DECALOGO questions.

This orchestrator:
1. Maps DECALOGO questions to specific evidence extractors
2. Routes plan sections to appropriate analyzers
3. Collects and standardizes evidence 
4. Aggregates results for comprehensive evaluation

The orchestrator ensures complete coverage of all DECALOGO framework questions
through proper component alignment and evidence routing.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable

# Import knowledge extractors
try:
    from causal_pattern_detector import create_causal_pattern_detector
    from teoria_cambio import TeoriaCambio, evaluar_teoria_cambio
    from monetary_detector import MonetaryDetector, create_monetary_detector
    from feasibility_scorer import FeasibilityScorer
    from responsibility_detector import ResponsibilityDetector
    from document_segmenter import DocumentSegmenter
    from contradiction_detector import ContradictionDetector
except ImportError as e:
    # Log import errors but continue - the orchestrator will handle missing components
    logging.error(f"Failed to import component: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecalogoDimension(Enum):
    """DECALOGO evaluation dimensions."""
    DE1 = "DE-1"  # Strategic Coherence
    DE2 = "DE-2"  # Thematic Inclusion
    DE3 = "DE-3"  # Participatory Process
    DE4 = "DE-4"  # Results Orientation


@dataclass
class DecalogoQuestion:
    """Represents a specific DECALOGO question with component mapping."""
    dimension: DecalogoDimension
    question_id: str
    text: str
    components: List[str]
    priority: int = 1
    
    @property
    def full_id(self) -> str:
        """Get the full question identifier."""
        return f"{self.dimension.value} Q{self.question_id}"


@dataclass
class EvidenceItem:
    """Standardized evidence item produced by knowledge extractors."""
    source_component: str
    question_ids: List[str]
    evidence_type: str
    content: Any
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecalogoEvaluation:
    """Complete evaluation results for a DECALOGO question."""
    question: DecalogoQuestion
    answer: str
    score: float
    confidence: float
    supporting_evidence: List[EvidenceItem]
    justification: str


class PipelineOrchestrator:
    """
    Central orchestrator for the DECALOGO knowledge extraction pipeline.
    
    Coordinates component execution, evidence collection, and result aggregation
    to ensure comprehensive coverage of all DECALOGO questions.
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator with component registry and question mappings."""
        # Initialize component registry
        self.components = {}
        
        # Initialize question mapping
        self.questions = self._build_question_mapping()
        
        # Initialize question-to-component mapping
        self.question_components = self._build_question_component_mapping()
        
        # Load available components
        self._load_components()
    
    def _build_question_mapping(self) -> Dict[str, DecalogoQuestion]:
        """
        Build comprehensive mapping of all DECALOGO questions.
        
        Returns:
            Dictionary mapping question IDs to DecalogoQuestion objects
        """
        questions = []
        
        # DE-1: Strategic Coherence
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE1,
            question_id="1",
            text="¿El diagnóstico identifica brechas y retos?",
            components=["document_segmenter", "gap_analyzer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE1,
            question_id="2",
            text="¿Las estrategias responden a los retos identificados?",
            components=["document_segmenter", "strategy_analyzer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE1,
            question_id="3",
            text="¿Los resultados tienen líneas base y metas?",
            components=["feasibility_scorer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE1,
            question_id="4",
            text="¿Existe un encadenamiento lógico en la cadena de valor?",
            components=["teoria_cambio", "causal_pattern_detector"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE1,
            question_id="5",
            text="¿Los indicadores son relevantes y específicos?",
            components=["feasibility_scorer", "indicator_analyzer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE1,
            question_id="6",
            text="¿Existe un marco lógico completo?",
            components=["teoria_cambio", "causal_pattern_detector"]
        ))
        
        # DE-2: Thematic Inclusion
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE2,
            question_id="1",
            text="¿Se articulan con Plan Nacional de Desarrollo?",
            components=["document_segmenter", "policy_alignment_detector"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE2,
            question_id="2",
            text="¿Se incorpora presupuesto para cada componente?",
            components=["monetary_detector"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE2,
            question_id="3",
            text="¿Se incorporan los ODS en objetivos/indicadores?",
            components=["document_segmenter", "ods_detector"]
        ))
        
        # DE-3: Participatory Process
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE3,
            question_id="1",
            text="¿Hay evidencia de participación ciudadana?",
            components=["document_segmenter", "participation_analyzer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE3,
            question_id="2",
            text="¿Se identifican grupos sociales específicos?",
            components=["document_segmenter", "social_group_detector"]
        ))
        
        # DE-4: Results Orientation
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE4,
            question_id="1",
            text="¿Los productos tienen KPI medibles?",
            components=["feasibility_scorer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE4,
            question_id="2",
            text="¿Los resultados tienen líneas base?",
            components=["feasibility_scorer"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE4,
            question_id="3",
            text="¿Existen entidades responsables por resultado?",
            components=["responsibility_detector"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE4,
            question_id="4",
            text="¿Los recursos son suficientes para resultados?",
            components=["monetary_detector"]
        ))
        
        questions.append(DecalogoQuestion(
            dimension=DecalogoDimension.DE4,
            question_id="5",
            text="¿Se articula con planes de largo plazo?",
            components=["document_segmenter", "long_term_plan_detector"]
        ))
        
        # Build dictionary mapping
        questions_dict = {}
        for q in questions:
            questions_dict[q.full_id] = q
        
        return questions_dict
    
    def _build_question_component_mapping(self) -> Dict[str, Set[str]]:
        """
        Build mapping from components to questions they answer.
        
        Returns:
            Dictionary mapping component names to sets of question IDs
        """
        component_questions = {}
        
        for question_id, question in self.questions.items():
            for component in question.components:
                if component not in component_questions:
                    component_questions[component] = set()
                component_questions[component].add(question_id)
        
        return component_questions
    
    def _load_components(self):
        """Load and initialize all available knowledge extraction components."""
        # Try to load teoria_cambio
        try:
            self.components["teoria_cambio"] = TeoriaCambio()
            logger.info("Loaded teoria_cambio component")
        except Exception as e:
            logger.warning(f"Failed to load teoria_cambio: {e}")
        
        # Try to load causal_pattern_detector
        try:
            self.components["causal_pattern_detector"] = create_causal_pattern_detector()
            logger.info("Loaded causal_pattern_detector component")
        except Exception as e:
            logger.warning(f"Failed to load causal_pattern_detector: {e}")
        
        # Try to load monetary_detector
        try:
            self.components["monetary_detector"] = create_monetary_detector()
            logger.info("Loaded monetary_detector component")
        except Exception as e:
            logger.warning(f"Failed to load monetary_detector: {e}")
        
        # Try to load feasibility_scorer
        try:
            self.components["feasibility_scorer"] = FeasibilityScorer()
            logger.info("Loaded feasibility_scorer component")
        except Exception as e:
            logger.warning(f"Failed to load feasibility_scorer: {e}")
        
        # Try to load document_segmenter
        try:
            self.components["document_segmenter"] = DocumentSegmenter()
            logger.info("Loaded document_segmenter component")
        except Exception as e:
            logger.warning(f"Failed to load document_segmenter: {e}")
        
        # Try to load responsibility_detector
        try:
            self.components["responsibility_detector"] = ResponsibilityDetector()
            logger.info("Loaded responsibility_detector component")
        except Exception as e:
            logger.warning(f"Failed to load responsibility_detector: {e}")
        
        # Try to load contradiction_detector
        try:
            self.components["contradiction_detector"] = ContradictionDetector()
            logger.info("Loaded contradiction_detector component")
        except Exception as e:
            logger.warning(f"Failed to load contradiction_detector: {e}")
        
        # Analyze component coverage
        self._analyze_component_coverage()
    
    def _analyze_component_coverage(self):
        """Analyze and log component coverage of DECALOGO questions."""
        # Check which questions are covered by available components
        covered_questions = set()
        for component_name in self.components:
            if component_name in self.question_components:
                covered_questions.update(self.question_components[component_name])
        
        # Calculate coverage percentage
        total_questions = len(self.questions)
        covered_count = len(covered_questions)
        coverage_pct = (covered_count / total_questions) * 100 if total_questions > 0 else 0
        
        logger.info(f"DECALOGO question coverage: {covered_count}/{total_questions} questions ({coverage_pct:.1f}%)")
        
        # Log uncovered questions
        uncovered = set(self.questions.keys()) - covered_questions
        if uncovered:
            logger.warning(f"Uncovered questions: {', '.join(sorted(uncovered))}")
    
    def extract_evidence(self, plan_text: str, plan_id: str) -> Dict[str, List[EvidenceItem]]:
        """
        Extract all available evidence from plan text using loaded components.
        
        Args:
            plan_text: Full text of the development plan
            plan_id: Unique identifier for the plan
            
        Returns:
            Dictionary mapping question IDs to lists of evidence items
        """
        # Initialize evidence collection
        evidence_by_question = {q_id: [] for q_id in self.questions}
        
        # First, segment the document if segmenter is available
        plan_sections = {}
        if "document_segmenter" in self.components:
            try:
                segmenter = self.components["document_segmenter"]
                plan_sections = segmenter.segment_document(plan_text)
                logger.info(f"Document segmented into {len(plan_sections)} sections")
            except Exception as e:
                logger.error(f"Error segmenting document: {e}")
                # Use full text as fallback
                plan_sections = {"full_text": plan_text}
        else:
            # Use full text if segmenter not available
            plan_sections = {"full_text": plan_text}
        
        # Collect evidence using teoria_cambio
        if "teoria_cambio" in self.components:
            try:
                tc_results = evaluar_teoria_cambio(plan_text)
                
                # DE-1 Q4 evidence
                evidence_item_q4 = EvidenceItem(
                    source_component="teoria_cambio",
                    question_ids=["DE-1 Q4"],
                    evidence_type="value_chain_linkage",
                    content=tc_results["value_chain_evaluation"],
                    confidence=0.7,
                    metadata={"logical_linkage": tc_results["de1_q4"]["answer"] == "Sí"}
                )
                evidence_by_question["DE-1 Q4"].append(evidence_item_q4)
                
                # DE-1 Q6 evidence
                evidence_item_q6 = EvidenceItem(
                    source_component="teoria_cambio",
                    question_ids=["DE-1 Q6"],
                    evidence_type="logical_framework",
                    content=tc_results["logical_framework_evaluation"],
                    confidence=0.7,
                    metadata={"framework_complete": tc_results["de1_q6"]["answer"] == "Sí"}
                )
                evidence_by_question["DE-1 Q6"].append(evidence_item_q6)
                
                logger.info("Collected evidence from teoria_cambio")
            except Exception as e:
                logger.error(f"Error collecting evidence from teoria_cambio: {e}")
        
        # Collect evidence using causal_pattern_detector
        if "causal_pattern_detector" in self.components:
            try:
                detector = self.components["causal_pattern_detector"]
                decalogo_analysis = detector.analyze_document_for_decalogo(plan_text)
                
                # DE-1 Q4 evidence
                evidence_item_q4 = EvidenceItem(
                    source_component="causal_pattern_detector",
                    question_ids=["DE-1 Q4"],
                    evidence_type="causal_patterns",
                    content=decalogo_analysis["causal_analysis"],
                    confidence=decalogo_analysis["de1_q4"]["confidence"],
                    metadata={"has_logical_linkage": decalogo_analysis["de1_q4"]["answer"] == "Sí"}
                )
                evidence_by_question["DE-1 Q4"].append(evidence_item_q4)
                
                # DE-1 Q6 evidence
                evidence_item_q6 = EvidenceItem(
                    source_component="causal_pattern_detector",
                    question_ids=["DE-1 Q6"],
                    evidence_type="causal_framework",
                    content=decalogo_analysis,
                    confidence=decalogo_analysis["de1_q6"]["confidence"],
                    metadata={"has_complete_framework": decalogo_analysis["de1_q6"]["answer"] == "Sí"}
                )
                evidence_by_question["DE-1 Q6"].append(evidence_item_q6)
                
                logger.info("Collected evidence from causal_pattern_detector")
            except Exception as e:
                logger.error(f"Error collecting evidence from causal_pattern_detector: {e}")
        
        # Collect evidence using monetary_detector
        if "monetary_detector" in self.components:
            try:
                detector = self.components["monetary_detector"]
                monetary_analysis = detector.analyze_monetary_coverage(plan_text)
                
                # DE-2 Q2 evidence
                evidence_item_budget = EvidenceItem(
                    source_component="monetary_detector",
                    question_ids=["DE-2 Q2"],
                    evidence_type="budget_coverage",
                    content=monetary_analysis,
                    confidence=0.8,
                    metadata={"budget_coverage": monetary_analysis.budget_coverage}
                )
                evidence_by_question["DE-2 Q2"].append(evidence_item_budget)
                
                # DE-4 Q4 evidence - resource adequacy
                resource_eval = detector.evaluate_resource_adequacy(plan_text)
                evidence_item_resources = EvidenceItem(
                    source_component="monetary_detector",
                    question_ids=["DE-4 Q4"],
                    evidence_type="resource_adequacy",
                    content=resource_eval,
                    confidence=0.8,
                    metadata={
                        "resources_adequate": resource_eval["de4_resource_adequacy"],
                        "resource_score": resource_eval["resource_adequacy_score"]
                    }
                )
                evidence_by_question["DE-4 Q4"].append(evidence_item_resources)
                
                logger.info("Collected evidence from monetary_detector")
            except Exception as e:
                logger.error(f"Error collecting evidence from monetary_detector: {e}")
        
        # Collect evidence using feasibility_scorer
        if "feasibility_scorer" in self.components:
            try:
                scorer = self.components["feasibility_scorer"]
                plan_feasibility = scorer.evaluate_plan_feasibility(plan_text)
                
                # DE-1 Q3 evidence
                de1_q3_result = plan_feasibility["decalogo_answers"]["DE1_Q3"]
                evidence_item_q3 = EvidenceItem(
                    source_component="feasibility_scorer",
                    question_ids=["DE-1 Q3"],
                    evidence_type="baseline_target_evaluation",
                    content=de1_q3_result,
                    confidence=de1_q3_result["confidence"],
                    metadata={"have_baselines_targets": de1_q3_result["answer"] == "Sí"}
                )
                evidence_by_question["DE-1 Q3"].append(evidence_item_q3)
                
                # DE-4 Q1 evidence
                de4_q1_result = plan_feasibility["decalogo_answers"]["DE4_Q1"]
                evidence_item_kpi = EvidenceItem(
                    source_component="feasibility_scorer",
                    question_ids=["DE-4 Q1"],
                    evidence_type="kpi_evaluation",
                    content=de4_q1_result,
                    confidence=de4_q1_result["confidence"],
                    metadata={"have_measurable_kpis": de4_q1_result["answer"] == "Sí"}
                )
                evidence_by_question["DE-4 Q1"].append(evidence_item_kpi)
                
                # DE-4 Q2 evidence
                de4_q2_result = plan_feasibility["decalogo_answers"]["DE4_Q2"]
                evidence_item_baselines = EvidenceItem(
                    source_component="feasibility_scorer",
                    question_ids=["DE-4 Q2"],
                    evidence_type="baseline_evaluation",
                    content=de4_q2_result,
                    confidence=de4_q2_result["confidence"],
                    metadata={"have_baselines": de4_q2_result["answer"] == "Sí"}
                )
                evidence_by_question["DE-4 Q2"].append(evidence_item_baselines)
                
                logger.info("Collected evidence from feasibility_scorer")
            except Exception as e:
                logger.error(f"Error collecting evidence from feasibility_scorer: {e}")
        
        # Collect evidence using responsibility_detector
        if "responsibility_detector" in self.components:
            try:
                detector = self.components["responsibility_detector"]
                responsibility_entities = detector.detect_entities(plan_text)
                
                # Analyze entities for responsible parties
                responsible_count = len([e for e in responsibility_entities if e.confidence >= 0.6])
                has_responsible_entities = responsible_count >= 3
                
                # DE-4 Q3 evidence
                evidence_item = EvidenceItem(
                    source_component="responsibility_detector",
                    question_ids=["DE-4 Q3"],
                    evidence_type="responsibility_detection",
                    content=responsibility_entities,
                    confidence=0.7,
                    metadata={
                        "entity_count": responsible_count,
                        "has_responsible_entities": has_responsible_entities
                    }
                )
                evidence_by_question["DE-4 Q3"].append(evidence_item)
                
                logger.info("Collected evidence from responsibility_detector")
            except Exception as e:
                logger.error(f"Error collecting evidence from responsibility_detector: {e}")
        
        # Add more component evidence extraction as needed...
        
        # Return evidence collection
        return evidence_by_question
    
    def evaluate_plan(self, plan_text: str, plan_id: str) -> Dict[str, DecalogoEvaluation]:
        """
        Conduct comprehensive DECALOGO evaluation of a development plan.
        
        Args:
            plan_text: Full text of the development plan
            plan_id: Unique identifier for the plan
            
        Returns:
            Dictionary mapping question IDs to evaluation results
        """
        # Extract all available evidence
        evidence_collection = self.extract_evidence(plan_text, plan_id)
        
        # Perform evaluation for each question
        evaluations = {}
        
        for question_id, question in self.questions.items():
            # Get evidence for this question
            question_evidence = evidence_collection.get(question_id, [])
            
            if not question_evidence:
                # Skip questions with no evidence
                logger.warning(f"No evidence available for {question_id}: {question.text}")
                continue
            
            # Evaluate the question based on evidence
            evaluation = self._evaluate_question(question, question_evidence)
            evaluations[question_id] = evaluation
        
        return evaluations
    
    def _evaluate_question(self, question: DecalogoQuestion, evidence: List[EvidenceItem]) -> DecalogoEvaluation:
        """
        Evaluate a DECALOGO question based on collected evidence.
        
        Args:
            question: The DECALOGO question to evaluate
            evidence: List of evidence items for this question
            
        Returns:
            Comprehensive evaluation result
        """
        # Default values
        answer = "No determinado"
        score = 0.0
        confidence = 0.0
        justification = "Evidencia insuficiente para determinar."
        
        if not evidence:
            return DecalogoEvaluation(
                question=question,
                answer=answer,
                score=score,
                confidence=confidence,
                supporting_evidence=[],
                justification=justification
            )
        
        # Calculate average evidence confidence
        avg_confidence = sum(item.confidence for item in evidence) / len(evidence)
        
        # Different evaluation strategies based on question
        if question.full_id == "DE-1 Q4":
            # Logical linkage in value chain
            has_linkage = any(
                item.metadata.get("logical_linkage", False) or 
                item.metadata.get("has_logical_linkage", False)
                for item in evidence
            )
            
            linkage_scores = [
                item.content.get("overall_score", 0.0) if isinstance(item.content, dict) else 0.0
                for item in evidence
            ]
            
            avg_score = sum(linkage_scores) / len(linkage_scores) if linkage_scores else 0.0
            
            answer = "Sí" if has_linkage else "No"
            score = avg_score
            confidence = avg_confidence
            justification = f"{'Se encontró' if has_linkage else 'No se encontró'} un encadenamiento lógico en la cadena de valor."
        
        elif question.full_id == "DE-1 Q6":
            # Complete logical framework
            has_framework = any(
                item.metadata.get("framework_complete", False) or 
                item.metadata.get("has_complete_framework", False)
                for item in evidence
            )
            
            framework_scores = [
                item.content.get("completeness_score", 0.0) if isinstance(item.content, dict) else 0.0
                for item in evidence
            ]
            
            avg_score = sum(framework_scores) / len(framework_scores) if framework_scores else 0.0
            
            answer = "Sí" if has_framework else "No"
            score = avg_score
            confidence = avg_confidence
            justification = f"{'Se encontró' if has_framework else 'No se encontró'} un marco lógico completo."
        
        elif question.full_id == "DE-2 Q2":
            # Budget incorporation
            budget_coverage = max([
                item.metadata.get("budget_coverage", 0.0) for item in evidence
            ], default=0.0)
            
            has_budget = budget_coverage >= 0.5
            
            answer = "Sí" if has_budget else "No"
            score = budget_coverage
            confidence = avg_confidence
            justification = f"{'Se incorpora' if has_budget else 'No se incorpora'} presupuesto adecuado para los componentes."
        
        elif question.full_id in ["DE-1 Q3", "DE-4 Q1", "DE-4 Q2"]:
            # Questions about baselines, targets, and KPIs
            positive_answers = [
                item for item in evidence 
                if item.content.get("answer") == "Sí"
            ]
            
            has_positive = len(positive_answers) > 0
            
            answer = "Sí" if has_positive else "No"
            score = sum(item.content.get("coverage_ratio", 0.0) for item in evidence) / len(evidence) if evidence else 0.0
            confidence = avg_confidence
            
            if question.full_id == "DE-1 Q3":
                justification = f"Los resultados {'tienen' if has_positive else 'no tienen'} líneas base y metas adecuadas."
            elif question.full_id == "DE-4 Q1":
                justification = f"Los productos {'tienen' if has_positive else 'no tienen'} KPIs medibles."
            else:
                justification = f"Los resultados {'tienen' if has_positive else 'no tienen'} líneas base."
        
        elif question.full_id == "DE-4 Q3":
            # Responsible entities
            has_responsible = any(
                item.metadata.get("has_responsible_entities", False)
                for item in evidence
            )
            
            entity_counts = [
                item.metadata.get("entity_count", 0) for item in evidence
            ]
            
            total_entities = sum(entity_counts)
            
            answer = "Sí" if has_responsible else "No"
            score = min(1.0, total_entities / 5) if total_entities > 0 else 0.0
            confidence = avg_confidence
            justification = f"{'Se identificaron' if has_responsible else 'No se identificaron'} entidades responsables por resultado."
        
        elif question.full_id == "DE-4 Q4":
            # Resource adequacy
            resources_adequate = any(
                item.metadata.get("resources_adequate", False)
                for item in evidence
            )
            
            adequacy_scores = [
                item.metadata.get("resource_score", 0.0) for item in evidence
            ]
            
            avg_score = sum(adequacy_scores) / len(adequacy_scores) if adequacy_scores else 0.0
            
            answer = "Sí" if resources_adequate else "No"
            score = avg_score
            confidence = avg_confidence
            justification = f"Los recursos {'son' if resources_adequate else 'no son'} suficientes para lograr los resultados."
        
        else:
            # Generic evaluation for other questions
            # This would be expanded in a full implementation with specific logic per question
            positive_evidence = [
                item for item in evidence if 
                any(value == True for key, value in item.metadata.items() if key.startswith("has_"))
            ]
            
            answer = "Sí" if len(positive_evidence) > len(evidence) / 2 else "No"
            score = len(positive_evidence) / len(evidence) if evidence else 0.0
            confidence = avg_confidence
            justification = f"Evaluación basada en {len(evidence)} fuentes de evidencia."
        
        # Create and return evaluation
        return DecalogoEvaluation(
            question=question,
            answer=answer,
            score=score,
            confidence=confidence,
            supporting_evidence=evidence,
            justification=justification
        )
    
    def generate_evaluation_report(self, evaluations: Dict[str, DecalogoEvaluation]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report from individual question evaluations.
        
        Args:
            evaluations: Dictionary of question evaluations
            
        Returns:
            Dictionary with structured evaluation report
        """
        if not evaluations:
            return {
                "status": "error",
                "message": "No evaluations available",
                "dimensions": {},
                "overall_score": 0.0
            }
        
        # Group by dimension
        dimensions = {}
        for dimension in DecalogoDimension:
            dimensions[dimension.value] = {
                "questions": {},
                "score": 0.0,
                "answered_questions": 0
            }
        
        # Populate dimensions with questions
        for question_id, evaluation in evaluations.items():
            dimension_id = evaluation.question.dimension.value
            dimensions[dimension_id]["questions"][question_id] = {
                "text": evaluation.question.text,
                "answer": evaluation.answer,
                "score": evaluation.score,
                "confidence": evaluation.confidence,
                "justification": evaluation.justification
            }
            
            dimensions[dimension_id]["answered_questions"] += 1
        
        # Calculate dimension scores
        for dimension_id, dimension in dimensions.items():
            questions = dimension["questions"]
            if questions:
                dimension["score"] = sum(q["score"] for q in questions.values()) / len(questions)
            else:
                dimension["score"] = 0.0
        
        # Calculate overall score
        dimension_scores = [d["score"] for d in dimensions.values()]
        overall_score = sum(dimension_scores) / len(dimension_scores) if dimension_scores else 0.0
        
        # Generate summary
        summary = {
            "overall_score": overall_score,
            "dimensions": {d: dimensions[d]["score"] for d in dimensions},
            "answered_questions": sum(d["answered_questions"] for d in dimensions.values()),
            "total_questions": len(self.questions)
        }
        
        # Compile report
        report = {
            "status": "success",
            "summary": summary,
            "dimensions": dimensions,
            "overall_score": overall_score,
            "timestamp": str(datetime.datetime.now()),
            "evaluation_coverage": summary["answered_questions"] / summary["total_questions"] if summary["total_questions"] > 0 else 0.0
        }
        
        return report


def create_pipeline_orchestrator() -> PipelineOrchestrator:
    """
    Factory function to create pipeline orchestrator instance.
    
    Returns:
        Initialized PipelineOrchestrator
    """
    return PipelineOrchestrator()


# Example usage
if __name__ == "__main__":
    import datetime
    
    # Create orchestrator
    orchestrator = create_pipeline_orchestrator()
    
    # Sample plan text
    sample_plan = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    
    DIAGNÓSTICO
    Actualmente la cobertura en educación es del 75% y en salud del 65%.
    La tasa de desempleo ha aumentado al 12% en los últimos años.
    
    OBJETIVOS Y METAS
    1. Aumentar la cobertura educativa al 95% para el año 2027.
       Línea base: 75% (2023)
       Meta: 95% (2027)
       Responsable: Secretaría de Educación Municipal
    
    2. Reducir la tasa de desempleo al 8% durante el cuatrienio.
       Línea base: 12% (2023)
       Meta: 8% (2027)
    
    3. Construir 500 viviendas de interés social.
       Plazo: 4 años
       Presupuesto: $50.000 millones COP
       
    ARTICULACIÓN
    Este plan contribuye a los objetivos del Plan Nacional de Desarrollo,
    específicamente a "Convergencia Regional" y "Equidad para la Paz".
    
    También se alinea con los Objetivos de Desarrollo Sostenible (ODS):
    - ODS 4: Educación de calidad
    - ODS 8: Trabajo decente y crecimiento económico
    - ODS 11: Ciudades y comunidades sostenibles
    
    PRESUPUESTO
    El presupuesto total para el plan es de $500.000 millones de pesos para el cuatrienio.
    Fuentes de financiación:
    - Recursos propios: 60%
    - Transferencias nacionales: 30%
    - Cooperación internacional: 10%
    
    PARTICIPACIÓN
    Se realizaron 15 talleres participativos con comunidades de todas las comunas,
    contando con la participación de más de 1,000 ciudadanos.
    """
    
    # Evaluate plan
    evaluations = orchestrator.evaluate_plan(sample_plan, "PLAN_EJEMPLO_001")
    
    # Generate report
    report = orchestrator.generate_evaluation_report(evaluations)
    
    # Display results
    print("=== DECALOGO EVALUATION RESULTS ===")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print(f"Evaluation Coverage: {report['evaluation_coverage']:.1%}")
    print("\nDimension Scores:")
    for dim_id, score in report['summary']['dimensions'].items():
        print(f"- {dim_id}: {score:.2f}")
    
    print("\nQuestion Answers:")
    for question_id, evaluation in evaluations.items():
        print(f"- {question_id}: {evaluation.answer} (Score: {evaluation.score:.2f}, Confidence: {evaluation.confidence:.2f})")
        print(f"  Justification: {evaluation.justification}")
