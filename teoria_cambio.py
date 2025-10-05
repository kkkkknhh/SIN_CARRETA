"""
Teoria de Cambio (Theory of Change) Module

This module implements causal graph construction and validation for development plans,
enabling the evaluation of logical intervention frameworks and causal connections.

Key features:
- Cached causal graph construction
- Logical framework validation
- Value chain linkage analysis
- Integration with DAG validation for statistical testing

This module is essential for answering:
- DE-1 Q4: "Is there logical linkage in the value chain?"
- DE-1 Q6: "Is there a complete logical framework?"
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import networkx as nx
import numpy as np

# Import the DAG validation module for causal graph validation
try:
    from dag_validation import AdvancedDAGValidator, GraphType
except ImportError:
    # Provide placeholder for testing
    class AdvancedDAGValidator:
        def __init__(self, graph_type=None): 
            self.graph_nodes = {}
    class GraphType:
        THEORY_OF_CHANGE = "theory_of_change"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalElementType(Enum):
    """Types of elements in a theory of change."""
    INPUT = "input"                # Resources, investments
    ACTIVITY = "activity"          # Activities, processes
    OUTPUT = "output"              # Direct products, services
    OUTCOME = "outcome"            # Medium-term results
    IMPACT = "impact"              # Long-term changes
    ASSUMPTION = "assumption"      # Critical assumptions
    EXTERNAL_FACTOR = "external_factor"  # External influences


class LogicModelQuality(Enum):
    """Quality levels for logical framework evaluation."""
    COMPLETE = "complete"          # All components present with clear connections
    PARTIAL = "partial"            # Some components missing or connections unclear
    MINIMAL = "minimal"            # Few components, poor connections
    INVALID = "invalid"            # Critical flaws or missing components


@dataclass
class CausalElement:
    """
    Represents an element in the theory of change model.
    
    Attributes:
        id: Unique identifier
        text: Textual description of the element
        element_type: Type of causal element
        preconditions: IDs of elements that are preconditions for this element
        indicators: Measurable indicators for this element
        confidence: Confidence in the element identification (0-1)
    """
    id: str
    text: str
    element_type: CausalElementType
    preconditions: Set[str] = field(default_factory=set)
    indicators: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "element_type": self.element_type.value,
            "preconditions": list(self.preconditions),
            "indicators": self.indicators,
            "confidence": self.confidence
        }


@dataclass
class LogicModelValidationResult:
    """
    Results of validating a logical framework.
    
    Attributes:
        quality: Overall quality assessment
        completeness_score: Score for completeness (0-1)
        coherence_score: Score for logical coherence (0-1)
        missing_components: List of missing component types
        logical_gaps: List of logical gaps identified
        causal_p_value: p-value from DAG acyclicity test
        recommendations: List of recommended improvements
    """
    quality: LogicModelQuality
    completeness_score: float
    coherence_score: float
    missing_components: List[CausalElementType] = field(default_factory=list)
    logical_gaps: List[str] = field(default_factory=list)
    causal_p_value: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "quality": self.quality.value,
            "completeness_score": self.completeness_score,
            "coherence_score": self.coherence_score,
            "missing_components": [comp.value for comp in self.missing_components],
            "logical_gaps": self.logical_gaps,
            "causal_p_value": self.causal_p_value,
            "recommendations": self.recommendations
        }


class TeoriaCambio:
    """
    Theory of Change analysis for development plans.
    
    This class constructs and analyzes causal graphs representing development plan 
    logic models, evaluating logical linkages between inputs, activities, outputs,
    outcomes, and impacts.
    
    Methods:
        construir_grafo_causal: Build causal graph with caching
        evaluar_marco_logico: Evaluate logical framework quality
        verificar_cadena_valor: Verify value chain integrity
        identificar_brechas_logicas: Identify logical gaps
        generar_recomendaciones: Generate improvement recommendations
    """
    
    def __init__(self):
        """Initialize the Theory of Change analyzer."""
        self._cached_graph = None
        self._cached_hash = None
        self._validator = None
    
    def construir_grafo_causal(self, elementos: List[CausalElement]) -> nx.DiGraph:
        """
        Build causal graph from elements with caching.
        
        Args:
            elementos: List of causal elements
            
        Returns:
            NetworkX DiGraph representing the causal model
        """
        # Calculate hash of input elements for cache invalidation
        elementos_hash = self._calculate_elementos_hash(elementos)
        
        # Check cache
        if self._cached_hash == elementos_hash and self._cached_graph is not None:
            return self._cached_graph
        
        # Cache miss, build new graph
        graph = self._crear_grafo_causal(elementos)
        
        # Update cache
        self._cached_graph = graph
        self._cached_hash = elementos_hash
        
        return graph
    
    def invalidar_cache_grafo(self):
        """Invalidate the cached graph."""
        self._cached_graph = None
        self._cached_hash = None
    
    def _calculate_elementos_hash(self, elementos: List[CausalElement]) -> str:
        """Calculate deterministic hash for a list of elements."""
        if not elementos:
            return "empty"
        
        # Sort elements by ID for deterministic hashing
        sorted_elementos = sorted(elementos, key=lambda e: e.id)
        
        # Create hash from concatenated element attributes
        hash_input = "".join(f"{e.id}|{e.text}|{e.element_type.value}|{','.join(sorted(e.preconditions))}" 
                            for e in sorted_elementos)
        
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def _crear_grafo_causal(self, elementos: List[CausalElement]) -> nx.DiGraph:
        """
        Create causal graph from elements.
        
        Args:
            elementos: List of causal elements
            
        Returns:
            NetworkX DiGraph representing the causal model
        """
        G = nx.DiGraph()
        
        # Add nodes for each element
        for elem in elementos:
            G.add_node(
                elem.id,
                text=elem.text,
                element_type=elem.element_type.value,
                indicators=elem.indicators,
                confidence=elem.confidence
            )
        
        # Add edges based on preconditions
        for elem in elementos:
            for precond_id in elem.preconditions:
                if precond_id in G:
                    G.add_edge(precond_id, elem.id)
        
        return G
    
    def evaluar_marco_logico(self, elementos: List[CausalElement]) -> LogicModelValidationResult:
        """
        Evaluate the logical framework represented by the causal elements.
        
        Args:
            elementos: List of causal elements
            
        Returns:
            LogicModelValidationResult with comprehensive evaluation
        """
        if not elementos:
            return LogicModelValidationResult(
                quality=LogicModelQuality.INVALID,
                completeness_score=0.0,
                coherence_score=0.0,
                missing_components=list(CausalElementType),
                logical_gaps=["Marco lógico vacío"],
                recommendations=["Definir todos los componentes del marco lógico"]
            )
        
        # Build causal graph
        G = self.construir_grafo_causal(elementos)
        
        # Check for missing component types
        element_types_present = {elem.element_type for elem in elementos}
        missing_components = [et for et in CausalElementType 
                             if et not in element_types_present and et != CausalElementType.EXTERNAL_FACTOR]
        
        # Calculate completeness score
        critical_types = {CausalElementType.INPUT, CausalElementType.ACTIVITY, 
                         CausalElementType.OUTPUT, CausalElementType.OUTCOME}
        num_critical_present = len(critical_types.intersection(element_types_present))
        completeness_score = num_critical_present / len(critical_types)
        
        # Identify logical gaps
        logical_gaps = self.identificar_brechas_logicas(elementos, G)
        
        # Check coherence using path analysis
        coherence_score = self._calcular_coherencia(G, elementos)
        
        # Validate causal structure using DAG validation
        causal_p_value = self._validar_estructura_causal(elementos)
        
        # Generate recommendations
        recommendations = self.generar_recomendaciones(
            elementos, missing_components, logical_gaps, coherence_score
        )
        
        # Determine overall quality
        quality = self._determinar_calidad(
            completeness_score, coherence_score, causal_p_value, logical_gaps
        )
        
        return LogicModelValidationResult(
            quality=quality,
            completeness_score=completeness_score,
            coherence_score=coherence_score,
            missing_components=missing_components,
            logical_gaps=logical_gaps,
            causal_p_value=causal_p_value,
            recommendations=recommendations
        )
    
    def verificar_cadena_valor(self, elementos: List[CausalElement]) -> Dict[str, Any]:
        """
        Verify the integrity of the value chain (inputs → activities → outputs → outcomes → impacts).
        Specifically answers DE-1 Q4: "Is there logical linkage in the value chain?"
        
        Args:
            elementos: List of causal elements
            
        Returns:
            Dictionary with value chain verification results
        """
        # Build causal graph
        G = self.construir_grafo_causal(elementos)
        
        # Group elements by type
        elements_by_type = {et: [] for et in CausalElementType}
        for elem in elementos:
            elements_by_type[elem.element_type].append(elem)
        
        # Check connections across the value chain
        value_chain = [
            CausalElementType.INPUT,
            CausalElementType.ACTIVITY,
            CausalElementType.OUTPUT,
            CausalElementType.OUTCOME,
            CausalElementType.IMPACT
        ]
        
        linkage_results = {}
        
        # Check each sequential linkage in the value chain
        for i in range(len(value_chain) - 1):
            from_type = value_chain[i]
            to_type = value_chain[i + 1]
            
            from_elements = elements_by_type[from_type]
            to_elements = elements_by_type[to_type]
            
            # Skip if either element type is missing
            if not from_elements or not to_elements:
                linkage_name = f"{from_type.value}_to_{to_type.value}"
                linkage_results[linkage_name] = {
                    "exists": False,
                    "coverage": 0.0,
                    "strength": 0.0,
                    "gaps": [f"Missing {from_type.value if not from_elements else to_type.value} elements"]
                }
                continue
            
            # Calculate linkage statistics
            connected_to_elements = set()
            connection_count = 0
            
            for to_elem in to_elements:
                has_connection = False
                for from_elem in from_elements:
                    if from_elem.id in to_elem.preconditions or nx.has_path(G, from_elem.id, to_elem.id):
                        has_connection = True
                        connection_count += 1
                
                if has_connection:
                    connected_to_elements.add(to_elem.id)
            
            # Calculate metrics
            coverage = len(connected_to_elements) / len(to_elements)
            strength = connection_count / (len(from_elements) * len(to_elements))
            
            # Identify gaps
            gaps = []
            if coverage < 1.0:
                unconnected = [e.id for e in to_elements if e.id not in connected_to_elements]
                gaps.append(f"Unconnected {to_type.value} elements: {', '.join(unconnected[:3])}" + 
                           (f" and {len(unconnected) - 3} more" if len(unconnected) > 3 else ""))
            
            linkage_name = f"{from_type.value}_to_{to_type.value}"
            linkage_results[linkage_name] = {
                "exists": len(connected_to_elements) > 0,
                "coverage": coverage,
                "strength": strength,
                "gaps": gaps
            }
        
        # Assess overall value chain integrity
        complete_chain_exists = all(result["exists"] for result in linkage_results.values())
        average_coverage = sum(result["coverage"] for result in linkage_results.values()) / len(linkage_results)
        average_strength = sum(result["strength"] for result in linkage_results.values()) / len(linkage_results)
        
        overall_score = (average_coverage * 0.6 + average_strength * 0.4)
        
        return {
            "complete_chain_exists": complete_chain_exists,
            "average_coverage": average_coverage,
            "average_strength": average_strength,
            "overall_score": overall_score,
            "linkage_details": linkage_results,
            "has_logical_linkage": overall_score >= 0.5,
            "chain_quality": self._classify_chain_quality(overall_score),
            "de1_q4_answer": "Sí" if overall_score >= 0.5 else "No"
        }
    
    def verificar_marco_logico_completo(self, elementos: List[CausalElement]) -> Dict[str, Any]:
        """
        Verify if a complete logical framework exists (DE-1 Q6).
        
        A complete logical framework has:
        1. All key component types (inputs, activities, outputs, outcomes)
        2. Logical connections between components
        3. No significant gaps in the causal chain
        
        Args:
            elementos: List of causal elements
            
        Returns:
            Dictionary with logical framework verification results
        """
        if not elementos:
            return {
                "is_complete": False,
                "completeness_score": 0.0,
                "missing_components": [et.value for et in CausalElementType 
                                      if et != CausalElementType.EXTERNAL_FACTOR],
                "de1_q6_answer": "No"
            }
        
        # Evaluate logical framework
        validation_result = self.evaluar_marco_logico(elementos)
        
        # Check value chain integrity
        value_chain_results = self.verificar_cadena_valor(elementos)
        
        # A framework is complete if it has:
        # 1. At least "partial" quality (medium-high completeness)
        # 2. Good value chain linkage
        is_complete = (
            validation_result.quality in [LogicModelQuality.COMPLETE, LogicModelQuality.PARTIAL] and
            validation_result.completeness_score >= 0.75 and
            value_chain_results["overall_score"] >= 0.5 and
            len(validation_result.logical_gaps) <= 2
        )
        
        return {
            "is_complete": is_complete,
            "completeness_score": validation_result.completeness_score,
            "coherence_score": validation_result.coherence_score,
            "missing_components": [comp.value for comp in validation_result.missing_components],
            "logical_gaps": validation_result.logical_gaps,
            "value_chain_score": value_chain_results["overall_score"],
            "de1_q6_answer": "Sí" if is_complete else "No"
        }
    
    def identificar_brechas_logicas(self, elementos: List[CausalElement], G: nx.DiGraph = None) -> List[str]:
        """
        Identify logical gaps in the causal model.
        
        Args:
            elementos: List of causal elements
            G: Optional pre-built causal graph
            
        Returns:
            List of logical gap descriptions
        """
        if not G:
            G = self.construir_grafo_causal(elementos)
        
        gaps = []
        
        # Group elements by type
        elements_by_type = {et: [] for et in CausalElementType}
        for elem in elementos:
            elements_by_type[elem.element_type].append(elem)
        
        # Check for isolated elements (no inputs or outputs)
        for elem in elementos:
            predecessors = list(G.predecessors(elem.id))
            successors = list(G.successors(elem.id))
            
            if not predecessors and elem.element_type not in [CausalElementType.INPUT, CausalElementType.ASSUMPTION]:
                gaps.append(f"Elemento aislado sin precondiciones: '{elem.text[:50]}...' ({elem.id})")
            
            if not successors and elem.element_type not in [CausalElementType.IMPACT, CausalElementType.ASSUMPTION]:
                gaps.append(f"Elemento terminal sin efectos: '{elem.text[:50]}...' ({elem.id})")
        
        # Check for typical logical inconsistencies
        if elements_by_type[CausalElementType.OUTPUT]:
            for output in elements_by_type[CausalElementType.OUTPUT]:
                # Outputs should be connected to activities
                connected_to_activity = False
                for activity in elements_by_type[CausalElementType.ACTIVITY]:
                    if nx.has_path(G, activity.id, output.id):
                        connected_to_activity = True
                        break
                
                if not connected_to_activity and elements_by_type[CausalElementType.ACTIVITY]:
                    gaps.append(f"Producto sin actividad asociada: '{output.text[:50]}...' ({output.id})")
        
        if elements_by_type[CausalElementType.OUTCOME]:
            for outcome in elements_by_type[CausalElementType.OUTCOME]:
                # Outcomes should be connected to outputs
                connected_to_output = False
                for output in elements_by_type[CausalElementType.OUTPUT]:
                    if nx.has_path(G, output.id, outcome.id):
                        connected_to_output = True
                        break
                
                if not connected_to_output and elements_by_type[CausalElementType.OUTPUT]:
                    gaps.append(f"Resultado sin productos asociados: '{outcome.text[:50]}...' ({outcome.id})")
        
        # Check for cycles (logical inconsistencies)
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                for cycle in cycles[:3]:  # List up to 3 cycles
                    cycle_text = ' → '.join(cycle)
                    gaps.append(f"Ciclo causal detectado: {cycle_text}")
        except:
            # NetworkX simple_cycles can raise exceptions for some graph structures
            pass
        
        return gaps
    
    def generar_recomendaciones(
        self, 
        elementos: List[CausalElement], 
        missing_components: List[CausalElementType],
        logical_gaps: List[str],
        coherence_score: float
    ) -> List[str]:
        """
        Generate recommendations for improving the logical framework.
        
        Args:
            elementos: List of causal elements
            missing_components: List of missing component types
            logical_gaps: List of identified logical gaps
            coherence_score: Score for logical coherence (0-1)
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Recommendations for missing components
        for comp_type in missing_components:
            if comp_type == CausalElementType.INPUT:
                recommendations.append("Definir los insumos y recursos necesarios para la implementación")
            elif comp_type == CausalElementType.ACTIVITY:
                recommendations.append("Especificar las actividades principales a realizar")
            elif comp_type == CausalElementType.OUTPUT:
                recommendations.append("Definir los productos esperados de las actividades")
            elif comp_type == CausalElementType.OUTCOME:
                recommendations.append("Establecer los resultados que se esperan alcanzar")
            elif comp_type == CausalElementType.IMPACT:
                recommendations.append("Definir los impactos a largo plazo que se busca generar")
            elif comp_type == CausalElementType.ASSUMPTION:
                recommendations.append("Identificar los supuestos críticos del marco lógico")
        
        # Recommendations based on logical gaps
        if logical_gaps:
            if len(logical_gaps) <= 3:
                for gap in logical_gaps:
                    if "aislado" in gap:
                        recommendations.append("Conectar los elementos aislados a la cadena causal")
                    elif "sin precondiciones" in gap:
                        recommendations.append("Definir las precondiciones necesarias para todos los elementos")
                    elif "sin efectos" in gap:
                        recommendations.append("Establecer los efectos esperados de los elementos terminales")
                    elif "Producto sin actividad" in gap:
                        recommendations.append("Asociar cada producto con al menos una actividad que lo genere")
                    elif "Resultado sin productos" in gap:
                        recommendations.append("Vincular cada resultado con los productos que contribuyen a lograrlo")
                    elif "Ciclo causal" in gap:
                        recommendations.append("Eliminar los ciclos causales para mantener una lógica unidireccional")
            else:
                recommendations.append("Revisar y corregir las brechas lógicas identificadas en el marco lógico")
        
        # Recommendations based on coherence score
        if coherence_score < 0.3:
            recommendations.append("Reconstruir el marco lógico con una secuencia causal clara")
        elif coherence_score < 0.6:
            recommendations.append("Fortalecer las conexiones entre los componentes del marco lógico")
        
        # General recommendations if very few elements
        if len(elementos) <= 3:
            recommendations.append("Desarrollar un marco lógico más completo con todos los componentes esenciales")
        
        # Remove duplicates while preserving order
        seen = set()
        recommendations = [rec for rec in recommendations if not (rec in seen or seen.add(rec))]
        
        return recommendations
    
    def _calcular_coherencia(self, G: nx.DiGraph, elementos: List[CausalElement]) -> float:
        """Calculate coherence score of the logical framework."""
        if not elementos:
            return 0.0
        
        # Calculate density of connections between different element types
        element_types = {elem.id: elem.element_type for elem in elementos}
        type_connections = {
            (CausalElementType.INPUT, CausalElementType.ACTIVITY): 0,
            (CausalElementType.ACTIVITY, CausalElementType.OUTPUT): 0,
            (CausalElementType.OUTPUT, CausalElementType.OUTCOME): 0,
            (CausalElementType.OUTCOME, CausalElementType.IMPACT): 0
        }
        
        # Count actual connections
        for edge in G.edges():
            from_id, to_id = edge
            from_type = element_types.get(from_id)
            to_type = element_types.get(to_id)
            if (from_type, to_type) in type_connections:
                type_connections[(from_type, to_type)] += 1
        
        # Count potential connections
        potential_connections = {k: 0 for k in type_connections}
        elements_by_type = {et: [] for et in CausalElementType}
        for elem in elementos:
            elements_by_type[elem.element_type].append(elem)
            
        for from_type, to_type in type_connections:
            from_elements = elements_by_type[from_type]
            to_elements = elements_by_type[to_type]
            potential_connections[(from_type, to_type)] = len(from_elements) * len(to_elements)
        
        # Calculate connection ratios
        connection_ratios = []
        for key in type_connections:
            potential = potential_connections[key]
            if potential > 0:
                ratio = type_connections[key] / potential
                connection_ratios.append(ratio)
        
        # Final coherence score
        if not connection_ratios:
            return 0.0
        
        # Weight complete logical paths more heavily
        coherence_score = sum(connection_ratios) / len(connection_ratios)
        
        # Check for complete paths from inputs to impacts
        if elements_by_type[CausalElementType.INPUT] and elements_by_type[CausalElementType.IMPACT]:
            complete_paths = 0
            for input_elem in elements_by_type[CausalElementType.INPUT]:
                for impact_elem in elements_by_type[CausalElementType.IMPACT]:
                    if nx.has_path(G, input_elem.id, impact_elem.id):
                        complete_paths += 1
            
            max_complete_paths = len(elements_by_type[CausalElementType.INPUT]) * len(elements_by_type[CausalElementType.IMPACT])
            if max_complete_paths > 0:
                path_completeness = complete_paths / max_complete_paths
                coherence_score = 0.7 * coherence_score + 0.3 * path_completeness
        
        return coherence_score
    
    def _validar_estructura_causal(self, elementos: List[CausalElement]) -> Optional[float]:
        """
        Validate causal structure using DAG validation.
        
        Args:
            elementos: List of causal elements
            
        Returns:
            p-value from acyclicity test, or None if validation fails
        """
        try:
            # Create DAG validator
            if not self._validator:
                self._validator = AdvancedDAGValidator(graph_type=GraphType.THEORY_OF_CHANGE)
            
            # Reset validator
            self._validator.graph_nodes = {}
            
            # Add nodes
            for elem in elementos:
                self._validator.add_node(elem.id)
            
            # Add edges
            for elem in elementos:
                for precond_id in elem.preconditions:
                    if precond_id in self._validator.graph_nodes:
                        self._validator.add_edge(precond_id, elem.id)
            
            # Calculate acyclicity p-value
            result = self._validator.calculate_acyclicity_pvalue("teoria_cambio_test", 1000)
            
            return result.p_value
        except Exception as e:
            logger.warning(f"Error validating causal structure: {e}")
            return None
    
    def _determinar_calidad(
        self, completeness_score: float, coherence_score: float, 
        causal_p_value: Optional[float], logical_gaps: List[str]
    ) -> LogicModelQuality:
        """Determine overall logical framework quality."""
        # If causal validation failed badly, quality is invalid
        if causal_p_value is not None and causal_p_value > 0.9:
            return LogicModelQuality.INVALID
            
        # If critical components are missing, quality is at most partial
        if completeness_score < 0.5:
            return LogicModelQuality.MINIMAL
        
        # Calculate weighted score
        weighted_score = 0.5 * completeness_score + 0.5 * coherence_score
        
        # Adjust for logical gaps
        if logical_gaps:
            gap_penalty = min(0.3, 0.1 * len(logical_gaps))
            weighted_score = max(0, weighted_score - gap_penalty)
        
        # Map score to quality level
        if weighted_score >= 0.8:
            return LogicModelQuality.COMPLETE
        elif weighted_score >= 0.5:
            return LogicModelQuality.PARTIAL
        elif weighted_score >= 0.2:
            return LogicModelQuality.MINIMAL
        else:
            return LogicModelQuality.INVALID
    
    def _classify_chain_quality(self, score: float) -> str:
        """Classify value chain quality based on score."""
        if score >= 0.8:
            return "Excelente"
        elif score >= 0.6:
            return "Buena"
        elif score >= 0.4:
            return "Regular"
        elif score >= 0.2:
            return "Débil"
        else:
            return "Deficiente"


def extract_causal_elements_from_text(text: str) -> List[CausalElement]:
    """
    Extract causal elements from plan text (simplified version).
    
    Args:
        text: Plan text to analyze
        
    Returns:
        List of extracted CausalElement objects
    """
    elements = []
    
    # This is a placeholder implementation.
    # In a real implementation, you would use NLP techniques to extract elements.
    
    # Sample detection of inputs
    input_patterns = [
        r'(?i)recursos\s+(?:de|para|:)\s+([^.]+)',
        r'(?i)insumos\s+(?:de|para|:)\s+([^.]+)',
        r'(?i)inversiones\s+(?:de|para|:)\s+([^.]+)'
    ]
    
    for pattern in input_patterns:
        for match in re.finditer(pattern, text):
            element_text = match.group(1).strip()
            element_id = f"input_{len(elements)}"
            elements.append(CausalElement(
                id=element_id,
                text=element_text,
                element_type=CausalElementType.INPUT,
                confidence=0.8
            ))
    
    # Sample detection of activities
    activity_patterns = [
        r'(?i)actividades\s+(?:de|para|:)\s+([^.]+)',
        r'(?i)(?:implementar|ejecutar|desarrollar|realizar)\s+([^.]+)',
        r'(?i)(?:se\s+(?:implementará|ejecutará|desarrollará|realizará))\s+([^.]+)'
    ]
    
    for pattern in activity_patterns:
        for match in re.finditer(pattern, text):
            element_text = match.group(1).strip()
            element_id = f"activity_{len(elements)}"
            elements.append(CausalElement(
                id=element_id,
                text=element_text,
                element_type=CausalElementType.ACTIVITY,
                confidence=0.7
            ))
    
    # Sample detection of outputs
    output_patterns = [
        r'(?i)productos?\s+(?:de|para|:)\s+([^.]+)',
        r'(?i)(?:entregar|producir|generar)\s+([^.]+)',
        r'(?i)(?:se\s+(?:entregará|producirá|generará))\s+([^.]+)'
    ]
    
    for pattern in output_patterns:
        for match in re.finditer(pattern, text):
            element_text = match.group(1).strip()
            element_id = f"output_{len(elements)}"
            elements.append(CausalElement(
                id=element_id,
                text=element_text,
                element_type=CausalElementType.OUTPUT,
                confidence=0.7
            ))
    
    # Sample detection of outcomes
    outcome_patterns = [
        r'(?i)resultados?\s+(?:de|para|:)\s+([^.]+)',
        r'(?i)(?:lograr|alcanzar|conseguir)\s+([^.]+)',
        r'(?i)(?:se\s+(?:logrará|alcanzará|conseguirá))\s+([^.]+)'
    ]
    
    for pattern in outcome_patterns:
        for match in re.finditer(pattern, text):
            element_text = match.group(1).strip()
            element_id = f"outcome_{len(elements)}"
            elements.append(CausalElement(
                id=element_id,
                text=element_text,
                element_type=CausalElementType.OUTCOME,
                confidence=0.6
            ))
    
    # Sample detection of impacts
    impact_patterns = [
        r'(?i)impactos?\s+(?:de|para|:)\s+([^.]+)',
        r'(?i)(?:impactar|transformar|cambiar)\s+([^.]+)',
        r'(?i)(?:a\s+largo\s+plazo)\s+([^.]+)'
    ]
    
    for pattern in impact_patterns:
        for match in re.finditer(pattern, text):
            element_text = match.group(1).strip()
            element_id = f"impact_{len(elements)}"
            elements.append(CausalElement(
                id=element_id,
                text=element_text,
                element_type=CausalElementType.IMPACT,
                confidence=0.5
            ))
    
    # This is a very simplified implementation.
    # In a real system, you would use more sophisticated NLP and
    # also analyze the relationships between elements.
    
    return elements


def evaluar_teoria_cambio(text: str) -> Dict[str, Any]:
    """
    Evaluate theory of change in a plan document.
    
    Args:
        text: Plan text to analyze
        
    Returns:
        Evaluation results dictionary with answers to DE-1 Q4 and Q6
    """
    # Extract causal elements
    elementos = extract_causal_elements_from_text(text)
    
    # Create teoria de cambio analyzer
    teoria = TeoriaCambio()
    
    # Evaluate logical linkage (DE-1 Q4)
    value_chain_results = teoria.verificar_cadena_valor(elementos)
    
    # Evaluate complete logical framework (DE-1 Q6)
    framework_results = teoria.verificar_marco_logico_completo(elementos)
    
    # Combine results
    results = {
        "num_elements": len(elementos),
        "elements_by_type": {et.value: sum(1 for e in elementos if e.element_type == et) for et in CausalElementType},
        "value_chain_evaluation": value_chain_results,
        "logical_framework_evaluation": framework_results,
        "de1_q4": {
            "question": "¿Existe un encadenamiento lógico en la cadena de valor?",
            "answer": value_chain_results["de1_q4_answer"],
            "score": value_chain_results["overall_score"],
            "confidence": 0.7
        },
        "de1_q6": {
            "question": "¿Existe un marco lógico completo?",
            "answer": framework_results["de1_q6_answer"],
            "score": framework_results["completeness_score"],
            "confidence": 0.7
        }
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    sample_text = """
    El programa de desarrollo municipal implementará una serie de actividades para mejorar 
    la calidad de vida de la población. Se invertirán recursos en la construcción de 
    infraestructura educativa y de salud. Las actividades incluyen la construcción de 
    5 centros de salud y 10 escuelas. Como resultado, se espera lograr un aumento en 
    la cobertura educativa y de salud. Los productos incluyen infraestructura nueva 
    y programas de capacitación para personal. A largo plazo, se espera impactar 
    positivamente en los indicadores de desarrollo humano de la región.
    """
    
    # Extract causal elements
    elementos = extract_causal_elements_from_text(sample_text)
    
    print(f"Extracted {len(elementos)} causal elements:")
    for i, elem in enumerate(elementos):
        print(f"{i+1}. {elem.element_type.value}: {elem.text[:50]}...")
    
    # Create teoria de cambio analyzer
    teoria = TeoriaCambio()
    
    # Evaluate logical framework
    evaluation = teoria.evaluar_marco_logico(elementos)
    
    print("\nLogical Framework Evaluation:")
    print(f"Quality: {evaluation.quality.value}")
    print(f"Completeness: {evaluation.completeness_score:.2f}")
    print(f"Coherence: {evaluation.coherence_score:.2f}")
    
    if evaluation.missing_components:
        print(f"Missing components: {[c.value for c in evaluation.missing_components]}")
    
    if evaluation.logical_gaps:
        print(f"Logical gaps: {evaluation.logical_gaps}")
    
    if evaluation.recommendations:
        print("\nRecommendations:")
        for rec in evaluation.recommendations:
            print(f"- {rec}")
    
    # Evaluate value chain
    value_chain = teoria.verificar_cadena_valor(elementos)
    
    print("\nValue Chain Evaluation:")
    print(f"Complete chain exists: {value_chain['complete_chain_exists']}")
    print(f"Overall score: {value_chain['overall_score']:.2f}")
    print(f"DE-1 Q4 answer: {value_chain['de1_q4_answer']}")
    
    # Evaluate complete logical framework
    framework = teoria.verificar_marco_logico_completo(elementos)
    
    print("\nComplete Logical Framework Evaluation:")
    print(f"Is complete: {framework['is_complete']}")
    print(f"DE-1 Q6 answer: {framework['de1_q6_answer']}")
