"""
Causal Pattern Detection Module

Identifies cause-effect relationships in development plans for comprehensive logical 
framework assessment. This module focuses on supporting the evaluation of DE-1 
questions related to logical intervention frameworks, causal connections, and value chain.

Features:
- Identification of causal patterns in text
- Analysis of causal coherence
- Support for logical framework validation
- Integration with teoria_cambio and dag_validation modules
- Extraction of causal elements for evaluation
"""

import logging
import re
import warnings
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Pattern, Match

import numpy as np
import pandas as pd

# Suppress deprecation warnings that might come from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import dcor
    import networkx as nx
    import pygam
    import torch
    import torch.nn as nn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    from econml.dml import CausalForestDML
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    logger.warning("Some dependencies are not available. Using simplified causal detection.")


class CausalRelationType:
    """Types of causal relationships that can be detected in text."""
    DIRECT = "direct"              # Direct causation
    ENABLING = "enabling"          # Enabling condition
    PREVENTING = "preventing"      # Preventing condition
    CONTRIBUTING = "contributing"  # Contributing factor
    NECESSARY = "necessary"        # Necessary condition
    SUFFICIENT = "sufficient"      # Sufficient condition


class CausalPatternDetector:
    """
    Base class for detecting causal patterns in text.
    
    This class provides basic functionality for detecting causal patterns
    using rule-based and lexical approaches. It serves as a fallback when
    full dependencies are not available.
    """
    
    def __init__(self):
        """Initialize the causal pattern detector."""
        # Causal indicators (in Spanish)
        self.causal_patterns = [
            r'(?i)(?:causa|causado por|causando|debido a|gracias a)',
            r'(?i)(?:produce|producido por|produciendo|genera|generado por)',
            r'(?i)(?:resultado de|como resultado|resultando en|deriva en|derivado de)',
            r'(?i)(?:consecuencia de|como consecuencia|conlleva|conllevando)',
            r'(?i)(?:implica|implicando|conduce a|conduce)',
            r'(?i)(?:lleva a|llevando a|lleva)',
            r'(?i)(?:influye en|influenciado por|impacta|impactando)',
            r'(?i)(?:efecto de|efectos de|afecta|afectando)',
            r'(?i)(?:por lo tanto|por lo cual|por ende|por consiguiente)',
            r'(?i)(?:así que|así pues|de modo que|de manera que)',
        ]
        
        # Temporal patterns that may indicate causality
        self.temporal_patterns = [
            r'(?i)(?:después de|luego de|tras|posterior a)',
            r'(?i)(?:antes de|previo a|anteriormente)',
            r'(?i)(?:durante|mientras|al mismo tiempo)',
            r'(?i)(?:inicialmente|finalmente|eventualmente)',
        ]
        
        # Causal relation patterns
        self.relation_patterns = {
            CausalRelationType.DIRECT: [
                r'(?i)(?:causa directa|directamente|explícitamente)',
                r'(?i)(?:produce directamente|genera directamente)',
            ],
            CausalRelationType.ENABLING: [
                r'(?i)(?:permite|posibilita|facilita|habilita)',
                r'(?i)(?:condición para|necesario para)',
            ],
            CausalRelationType.PREVENTING: [
                r'(?i)(?:impide|obstaculiza|previene|evita)',
                r'(?i)(?:restringe|limita|dificulta)',
            ],
            CausalRelationType.CONTRIBUTING: [
                r'(?i)(?:contribuye a|aporta a|ayuda a)',
                r'(?i)(?:colabora con|apoya a|favorece)',
            ],
            CausalRelationType.NECESSARY: [
                r'(?i)(?:requisito|indispensable|esencial)',
                r'(?i)(?:sin el cual no|condición sine qua non)',
            ],
            CausalRelationType.SUFFICIENT: [
                r'(?i)(?:suficiente para|basta para|alcanza para)',
                r'(?i)(?:suficiente para lograr|basta con)',
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_causal_patterns = [re.compile(p) for p in self.causal_patterns]
        self.compiled_temporal_patterns = [re.compile(p) for p in self.temporal_patterns]
        self.compiled_relation_patterns = {
            rel_type: [re.compile(p) for p in patterns] 
            for rel_type, patterns in self.relation_patterns.items()
        }
    
    def detect_causal_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect causal patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected causal patterns with metadata
        """
        if not text:
            return []
        
        results = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            # Check for causal patterns
            causal_matches = self._find_pattern_matches(sentence, self.compiled_causal_patterns)
            
            if causal_matches:
                # This sentence contains causal language
                for match in causal_matches:
                    # Determine the potential cause and effect parts
                    cause_part, effect_part = self._extract_cause_effect(sentence, match)
                    
                    # Determine relation type
                    relation_type = self._determine_relation_type(sentence, match)
                    
                    # Check if temporal patterns support the causality
                    temporal_evidence = any(pattern.search(sentence) for pattern in self.compiled_temporal_patterns)
                    
                    # Add result
                    results.append({
                        'sentence': sentence,
                        'causal_marker': match.group(0),
                        'cause': cause_part.strip() if cause_part else "",
                        'effect': effect_part.strip() if effect_part else "",
                        'relation_type': relation_type,
                        'has_temporal_evidence': temporal_evidence,
                        'confidence': self._calculate_confidence(sentence, match, temporal_evidence),
                        'span': (match.start(), match.end()),
                    })
        
        return results
    
    def analyze_causal_coherence(self, text: str) -> Dict[str, Any]:
        """
        Analyze overall causal coherence in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with causal coherence analysis
        """
        causal_patterns = self.detect_causal_patterns(text)
        
        if not causal_patterns:
            return {
                'has_causal_language': False,
                'causal_density': 0.0,
                'avg_confidence': 0.0,
                'relation_types': {},
                'causal_coherence_score': 0.0,
                'recommendation': "El texto no presenta lenguaje causal explícito. " +
                                 "Considere incorporar explicaciones causales para mejorar el marco lógico."
            }
        
        # Calculate metrics
        word_count = len(text.split())
        causal_density = len(causal_patterns) / (word_count / 100)  # per 100 words
        avg_confidence = sum(p['confidence'] for p in causal_patterns) / len(causal_patterns)
        
        # Count relation types
        relation_types = {}
        for pattern in causal_patterns:
            rel_type = pattern['relation_type']
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        # Calculate coherence score
        complete_patterns = sum(1 for p in causal_patterns if p['cause'] and p['effect'])
        coherence_score = complete_patterns / len(causal_patterns) if causal_patterns else 0.0
        coherence_score = coherence_score * 0.7 + avg_confidence * 0.3
        
        # Generate recommendation
        if coherence_score >= 0.7:
            recommendation = "El texto presenta un marco causal coherente y explícito."
        elif coherence_score >= 0.4:
            recommendation = "El texto presenta cierta estructura causal, pero podría fortalecerse " + \
                            "haciendo más explícitas las relaciones causa-efecto."
        else:
            recommendation = "El texto tiene un marco causal débil. Se recomienda reformular " + \
                            "para establecer claramente relaciones entre causas y efectos."
        
        return {
            'has_causal_language': True,
            'causal_density': causal_density,
            'causal_pattern_count': len(causal_patterns),
            'avg_confidence': avg_confidence,
            'relation_types': relation_types,
            'causal_coherence_score': coherence_score,
            'recommendation': recommendation
        }
    
    def extract_causal_elements_for_teoria_cambio(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract elements that can be used in Teoria de Cambio analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted elements with metadata
        """
        causal_patterns = self.detect_causal_patterns(text)
        elements = []
        
        for pattern in causal_patterns:
            if pattern['cause']:
                elements.append({
                    'text': pattern['cause'],
                    'element_type': 'causal_factor',
                    'confidence': pattern['confidence'],
                    'leads_to': pattern['effect'] if pattern['effect'] else None,
                    'relation_type': pattern['relation_type']
                })
            
            if pattern['effect']:
                elements.append({
                    'text': pattern['effect'],
                    'element_type': 'effect',
                    'confidence': pattern['confidence'],
                    'caused_by': pattern['cause'] if pattern['cause'] else None,
                    'relation_type': pattern['relation_type']
                })
        
        return elements
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting for Spanish
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_pattern_matches(self, text: str, patterns: List[Pattern[str]]) -> List[Match[str]]:
        """Find all matches for a set of patterns in text."""
        matches = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                matches.append(match)
        return matches
    
    def _extract_cause_effect(self, sentence: str, match: Match[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract cause and effect parts from a sentence based on causal marker.
        
        Args:
            sentence: Full sentence text
            match: Match object for the causal marker
            
        Returns:
            Tuple of (cause_text, effect_text), either may be None if unclear
        """
        marker = match.group(0).lower()
        start_pos = match.start()
        end_pos = match.end()
        
        # Split sentence at the marker position
        before_marker = sentence[:start_pos].strip()
        after_marker = sentence[end_pos:].strip()
        
        # Default assignment based on common patterns
        if marker in ['debido a', 'gracias a', 'como resultado', 'consecuencia de', 'resultado de']:
            # Effect before marker, cause after: "X debido a Y"
            return after_marker, before_marker
        else:
            # Cause before marker, effect after: "X causa Y"
            return before_marker, after_marker
    
    def _determine_relation_type(self, sentence: str, match: Match[str]) -> str:
        """Determine the type of causal relation."""
        for rel_type, patterns in self.compiled_relation_patterns.items():
            for pattern in patterns:
                if pattern.search(sentence):
                    return rel_type
        
        # Default to direct causation if no specific type is identified
        return CausalRelationType.DIRECT
    
    def _calculate_confidence(self, sentence: str, match: Match[str], has_temporal: bool) -> float:
        """
        Calculate confidence score for a causal pattern.
        
        Args:
            sentence: Full sentence text
            match: Match object for the causal marker
            has_temporal: Whether temporal evidence is present
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence
        confidence = 0.6
        
        # Adjust for specific causal markers
        marker = match.group(0).lower()
        strong_markers = ['causa', 'causado por', 'debido a', 'como resultado', 'por lo tanto']
        if any(strong in marker for strong in strong_markers):
            confidence += 0.1
        
        # Adjust for temporal evidence
        if has_temporal:
            confidence += 0.1
        
        # Adjust for sentence position
        if match.start() > len(sentence) / 2:
            # Causal marker in second half of sentence is often more reliable
            confidence += 0.05
        
        # Adjust for sentence length (longer sentences may be more ambiguous)
        if len(sentence) < 100:
            confidence += 0.05
        elif len(sentence) > 200:
            confidence -= 0.05
        
        return min(1.0, confidence)


class IndustrialCausalPatternDetector(CausalPatternDetector):
    """
    Advanced causal pattern detector with industrial-grade capabilities.
    
    This class extends the base detector with advanced NLP and statistical
    methods when dependencies are available. It falls back to base functionality
    when dependencies are not available.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the industrial causal pattern detector.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.advanced_mode = HAS_DEPENDENCIES
        if self.advanced_mode:
            self.causal_graph = nx.DiGraph()
            self.models: Dict[str, Any] = {}
            self.scalers: Dict[str, Any] = {}
            self.results: Dict[str, Any] = {}
        else:
            logger.warning("Operating in fallback mode with simplified causal detection")
    
    def detect_causal_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect causal patterns in text using advanced methods when available.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected causal patterns with metadata
        """
        # Use base detection first
        patterns = super().detect_causal_patterns(text)
        
        # Apply advanced processing if available
        if self.advanced_mode and len(patterns) > 0:
            try:
                patterns = self._enhance_patterns_with_nlp(text, patterns)
            except Exception as e:
                logger.warning(f"Error in advanced pattern processing: {e}")
        
        return patterns
    
    def analyze_document_for_decalogo(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive causal analysis for DECALOGO evaluation.
        
        This method focuses on DE-1 questions related to logical frameworks
        and causal connections.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Analysis results with focus on DE-1 evaluation
        """
        # Basic causal analysis
        basic_analysis = self.analyze_causal_coherence(text)
        
        # Extract elements for teoria de cambio
        causal_elements = self.extract_causal_elements_for_teoria_cambio(text)
        
        # Check if we have enough causal elements for a meaningful evaluation
        has_sufficient_causality = len(causal_elements) >= 5 and basic_analysis['causal_coherence_score'] >= 0.4
        
        # Calculate linkage score for DE-1 Q4
        linkage_score = self._calculate_linkage_score(causal_elements)
        has_logical_linkage = linkage_score >= 0.5
        
        # Assess completeness of logical framework for DE-1 Q6
        framework_score = self._assess_logical_framework_completeness(causal_elements)
        has_complete_framework = framework_score >= 0.7
        
        return {
            'causal_analysis': basic_analysis,
            'causal_element_count': len(causal_elements),
            'has_sufficient_causality': has_sufficient_causality,
            'linkage_score': linkage_score,
            'framework_score': framework_score,
            'de1_q4': {
                'question': '¿Existe un encadenamiento lógico en la cadena de valor?',
                'answer': 'Sí' if has_logical_linkage else 'No',
                'score': linkage_score,
                'confidence': 0.6,
                'evidence': f"Se encontraron {len(causal_elements)} elementos causales con una densidad de {basic_analysis['causal_density']:.2f}"
            },
            'de1_q6': {
                'question': '¿Existe un marco lógico completo?',
                'answer': 'Sí' if has_complete_framework else 'No',
                'score': framework_score,
                'confidence': 0.6,
                'evidence': basic_analysis['recommendation']
            }
        }
    
    def extract_causal_network(self, text: str) -> Dict[str, Any]:
        """
        Extract a causal network from text for visualization and analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with network data and metrics
        """
        patterns = self.detect_causal_patterns(text)
        
        if not patterns:
            return {'nodes': [], 'edges': [], 'metrics': {}}
        
        G = nx.DiGraph()
        nodes: Set[str] = set()
        edges: List[Dict[str, Any]] = []
        
        for pattern in patterns:
            if pattern['cause'] and pattern['effect']:
                cause_text = self._clean_for_node_label(pattern['cause'])
                effect_text = self._clean_for_node_label(pattern['effect'])
                nodes.add(cause_text)
                nodes.add(effect_text)
                edges.append({'from': cause_text, 'to': effect_text, 'type': pattern['relation_type'], 'confidence': pattern['confidence']})
                G.add_node(cause_text)
                G.add_node(effect_text)
                G.add_edge(cause_text, effect_text, type=pattern['relation_type'], confidence=pattern['confidence'])
        
        metrics: Dict[str, Any] = {}
        try:
            metrics['node_count'] = G.number_of_nodes()
            metrics['edge_count'] = G.number_of_edges()
            metrics['density'] = nx.density(G)
            try:
                metrics['has_cycles'] = not nx.is_directed_acyclic_graph(G)
            except Exception:
                metrics['has_cycles'] = False
            if G.number_of_nodes() > 2:
                metrics['key_causes'] = list(dict(sorted(
                    nx.out_degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:3]).keys())
                metrics['key_effects'] = list(dict(sorted(
                    nx.in_degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:3]).keys())
        except Exception as e:
            logger.warning(f"Error calculating network metrics: {e}")
        
        return {'nodes': list(nodes), 'edges': edges, 'metrics': metrics}
    
    def _enhance_patterns_with_nlp(self, text: str, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance detected patterns with advanced NLP techniques."""
        if not patterns:
            return patterns
        
        # This would be implemented with NLP models in a full implementation
        return patterns
    
    def _clean_for_node_label(self, text: str) -> str:
        """Clean and shorten text for use as a node label."""
        # Truncate long text
        if len(text) > 50:
            text = text[:47] + "..."
        
        # Remove special characters that might cause issues in graph visualization
        text = re.sub(r'[^\w\s.,;:?!¿¡-]', '', text)
        
        return text.strip()
    
    def _calculate_linkage_score(self, causal_elements: List[Dict[str, Any]]) -> float:
        """
        Calculate score for logical linkage in the value chain (DE-1 Q4).
        
        Args:
            causal_elements: List of extracted causal elements
            
        Returns:
            Linkage score between 0 and 1
        """
        if not causal_elements:
            return 0.0
        
        # Calculate linked elements (elements that have both cause and effect connections)
        linked_elements = 0
        for element in causal_elements:
            has_cause = element.get('caused_by') is not None
            has_effect = element.get('leads_to') is not None
            
            if has_cause and has_effect:
                linked_elements += 1
                
        # Calculate linkage ratio
        linkage_ratio = linked_elements / len(causal_elements) if causal_elements else 0.0
        
        # Calculate weighted score based on number of elements and their confidence
        avg_confidence = sum(element.get('confidence', 0.5) for element in causal_elements) / len(causal_elements)
        
        # Apply logarithmic scaling for element count (more elements is better but with diminishing returns)
        count_factor = min(1.0, np.log(len(causal_elements) + 1) / np.log(21))  # Scales up to 20 elements
        
        linkage_score = (linkage_ratio * 0.6) + (avg_confidence * 0.2) + (count_factor * 0.2)
        
        return linkage_score
    
    def _assess_logical_framework_completeness(self, causal_elements: List[Dict[str, Any]]) -> float:
        """
        Assess the completeness of the logical framework (DE-1 Q6).
        
        Args:
            causal_elements: List of extracted causal elements
            
        Returns:
            Framework completeness score between 0 and 1
        """
        if not causal_elements:
            return 0.0
        
        # Minimum requirements for a complete framework
        min_elements = 5
        min_linkage_ratio = 0.3
        
        # Calculate linked elements
        linked_elements = 0
        for element in causal_elements:
            has_cause = element.get('caused_by') is not None
            has_effect = element.get('leads_to') is not None
            
            if has_cause or has_effect:
                linked_elements += 1
        
        linkage_ratio = linked_elements / len(causal_elements) if causal_elements else 0.0
        
        # Calculate coherence by checking for relation type diversity
        relation_types = set(element.get('relation_type', '') for element in causal_elements)
        relation_diversity = min(1.0, len(relation_types) / 3)  # At least 3 types for full points
        
        # Calculate framework score
        element_count_score = min(1.0, len(causal_elements) / min_elements)
        linkage_score = 1.0 if linkage_ratio >= min_linkage_ratio else linkage_ratio / min_linkage_ratio
        
        framework_score = (
            element_count_score * 0.4 +
            linkage_score * 0.4 +
            relation_diversity * 0.2
        )
        
        return framework_score
    
    def build_causal_graph(self, text: str, method="pattern_based") -> nx.DiGraph:
        """
        Build a causal graph from text.
        
        Args:
            text: Text to analyze
            method: Method to use for graph construction
            
        Returns:
            NetworkX DiGraph representing the causal model
        """
        if method == "pattern_based" or not self.advanced_mode:
            return self._build_pattern_based_graph(text)
        elif method == "statistical" and self.advanced_mode:
            try:
                # Convert text to data for statistical analysis
                data = self._text_to_data(text)
                return self._build_statistical_graph(data)
            except Exception as e:
                logger.warning(f"Statistical graph building failed: {e}")
                return self._build_pattern_based_graph(text)
        else:
            return self._build_pattern_based_graph(text)
    
    def _build_pattern_based_graph(self, text: str) -> nx.DiGraph:
        """Build causal graph based on pattern detection."""
        G = nx.DiGraph()
        patterns = self.detect_causal_patterns(text)
        
        for pattern in patterns:
            if pattern['cause'] and pattern['effect']:
                # Clean text for node labels
                cause_text = self._clean_for_node_label(pattern['cause'])
                effect_text = self._clean_for_node_label(pattern['effect'])
                
                # Add nodes and edge
                G.add_node(cause_text, type='cause')
                G.add_node(effect_text, type='effect')
                G.add_edge(cause_text, effect_text,
                          type=pattern['relation_type'],
                          confidence=pattern['confidence'])
        
        return G
    
    def _text_to_data(self, text: str) -> pd.DataFrame:
        """Convert text to structured data for statistical analysis."""
        # This would be implemented in a full version
        # For now, return an empty DataFrame
        return pd.DataFrame()
    
    def _build_statistical_graph(self, data: pd.DataFrame) -> nx.DiGraph:
        """Build causal graph using statistical methods."""
        # This would be implemented in a full version
        # For now, return an empty graph
        return nx.DiGraph()


def create_causal_pattern_detector(config=None) -> CausalPatternDetector:
    """
    Factory function to create a causal pattern detector.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        CausalPatternDetector instance (basic or industrial)
    """
    if HAS_DEPENDENCIES:
        return IndustrialCausalPatternDetector(config)
    else:
        return CausalPatternDetector()


# Simple usage example
if __name__ == "__main__":
    # Create detector
    detector = create_causal_pattern_detector()
    
    # Example text with causal patterns
    sample_text = """
    El programa de desarrollo municipal implementará una estrategia integral que mejorará 
    la calidad de vida de los ciudadanos. La construcción de nuevas infraestructuras
    educativas generará mayor acceso a la educación. Debido a la mejora en los servicios
    públicos, la satisfacción ciudadana aumentará significativamente.
    
    La inversión en seguridad ciudadana producirá una reducción de la criminalidad del 30%,
    lo cual conllevará a un incremento del turismo y la actividad económica local.
    Por lo tanto, los indicadores de desarrollo humano mostrarán una mejora sustancial
    al finalizar el periodo de gobierno.
    """
    
    # Detect causal patterns
    patterns = detector.detect_causal_patterns(sample_text)
    
    # Print results
    print(f"Found {len(patterns)} causal patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\n{i+1}. {pattern['causal_marker']} (confidence: {pattern['confidence']:.2f})")
        print(f"   Cause: {pattern['cause']}")
        print(f"   Effect: {pattern['effect']}")
        print(f"   Type: {pattern['relation_type']}")
    
    # Analyze causal coherence
    coherence = detector.analyze_causal_coherence(sample_text)
    print(f"\nCausal coherence score: {coherence['causal_coherence_score']:.2f}")
    print(f"Recommendation: {coherence['recommendation']}")
    
    # DECALOGO evaluation
    if isinstance(detector, IndustrialCausalPatternDetector):
        decalogo = detector.analyze_document_for_decalogo(sample_text)
        print("\nDECALOGO DE-1 Evaluation:")
        print(f"DE-1 Q4: {decalogo['de1_q4']['answer']} (score: {decalogo['de1_q4']['score']:.2f})")
        print(f"DE-1 Q6: {decalogo['de1_q6']['answer']} (score: {decalogo['de1_q6']['score']:.2f})")
