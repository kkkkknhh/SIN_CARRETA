import math
from typing import List, Dict, Tuple
from .pattern_detector import PatternDetector, PatternMatch


class FactibilidadScorer:
    """
    Calculates factibilidad scores based on pattern detection and proximity.
    
    Uses a refined scoring formula: 
    score_final = w1 * similarity_score + w2 * (causal_connections / segment_length) + w3 * informative_length_ratio
    
    Where:
    - w1: Weight for similarity score component (default 0.5)
    - w2: Weight for normalized causal density (default 0.3) 
    - w3: Weight for informative content tie-breaker (default 0.2)
    - similarity_score: Semantic similarity score between 0-1
    - causal_connections: Number of detected causal relationships
    - segment_length: Length of text segment in characters
    - informative_length_ratio: Ratio of non-stopword content to total length
    """
    
    def __init__(self, proximity_window: int = 500, base_score: float = 0.0,
                 w1: float = 0.5, w2: float = 0.3, w3: float = 0.2):
        """
        Initialize the scorer with configurable weighting coefficients.
        
        Args:
            proximity_window: Maximum character distance for pattern clustering
            base_score: Base factibilidad score before pattern bonuses
            w1: Weight for similarity score component (default 0.5)
            w2: Weight for normalized causal density (default 0.3)
            w3: Weight for informative content tie-breaker (default 0.2)
            
        Raises:
            ValueError: If weights don't sum to a meaningful range [0.8, 1.2]
        """
        self.pattern_detector = PatternDetector()
        self.proximity_window = proximity_window
        self.base_score = base_score
        
        # Validate and set weighting coefficients
        self._validate_weights(w1, w2, w3)
        self.w1 = w1  # Similarity score weight
        self.w2 = w2  # Normalized causal density weight  
        self.w3 = w3  # Informative content weight
        
        # Scoring weights (legacy)
        self.cluster_bonus = 25.0  # Bonus for finding all three patterns together
        self.proximity_bonus_max = 15.0  # Maximum bonus for close proximity
        self.individual_pattern_bonus = 5.0  # Bonus for each pattern type found
        self.multiple_instance_bonus = 2.0  # Bonus for multiple instances of same type
        
        # Spanish stopwords for informative content calculation
        self.spanish_stopwords = {
            'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 
            'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'no', 'una', 'su', 
            'del', 'las', 'más', 'pero', 'sus', 'me', 'ya', 'muy', 'mi', 'sin',
            'sobre', 'este', 'ser', 'tiene', 'todo', 'esta', 'también', 'hasta',
            'donde', 'cuando', 'como', 'entre', 'durante', 'antes', 'después'
        }
    
    @staticmethod
    def _validate_weights(w1: float, w2: float, w3: float) -> None:
        """
        Validate that weights sum to a meaningful range for score calibration.
        
        Args:
            w1, w2, w3: Weight coefficients to validate
            
        Raises:
            ValueError: If weights sum is outside acceptable range [0.8, 1.2]
        """
        weight_sum = w1 + w2 + w3
        if not (0.8 <= weight_sum <= 1.2):
            raise ValueError(
                f"Weight sum {weight_sum:.3f} outside acceptable range [0.8, 1.2]. "
                f"Weights should sum to approximately 1.0 for proper score calibration."
            )
    
    def update_weights(self, w1: float = None, w2: float = None, w3: float = None) -> None:
        """
        Update weighting coefficients with validation.
        
        Args:
            w1: New similarity score weight (None to keep current)
            w2: New causal density weight (None to keep current) 
            w3: New informative content weight (None to keep current)
            
        Raises:
            ValueError: If new weights don't sum to acceptable range
        """
        new_w1 = w1 if w1 is not None else self.w1
        new_w2 = w2 if w2 is not None else self.w2
        new_w3 = w3 if w3 is not None else self.w3
        
        self._validate_weights(new_w1, new_w2, new_w3)
        
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2
        if w3 is not None:
            self.w3 = w3
    
    def score_text(self, text: str, similarity_score: float = 0.0) -> Dict:
        """
        Calculate factibilidad score using refined scoring formula.
        
        Formula: score_final = w1 * similarity_score + w2 * (causal_connections / segment_length) + w3 * informative_length_ratio
        
        Args:
            text: Text to analyze and score
            similarity_score: Semantic similarity score between 0-1 (default 0.0)
            
        Returns:
            Dictionary containing score breakdown and analysis with refined scoring
        """
        # Detect all patterns (legacy scoring)
        all_matches = self.pattern_detector.detect_patterns(text)
        clusters = self.pattern_detector.find_pattern_clusters(text, self.proximity_window)
        
        # Calculate base individual pattern scores (legacy)
        individual_scores = self._calculate_individual_scores(all_matches)
        
        # Calculate cluster scores (legacy)
        cluster_scores = self._calculate_cluster_scores(clusters)
        
        # Calculate legacy total score
        legacy_total_score = (self.base_score + 
                             individual_scores['total'] + 
                             cluster_scores['total'])
        
        # Calculate refined score components
        segment_length = len(text)
        causal_connections = self._count_causal_connections(all_matches)
        informative_length_ratio = self._calculate_informative_length_ratio(text)
        
        # Apply refined scoring formula
        score_final = self._calculate_refined_score(
            similarity_score, causal_connections, segment_length, informative_length_ratio
        )
        
        return {
            'score_final': score_final,
            'similarity_score': similarity_score,
            'causal_density': causal_connections / max(1, segment_length),
            'normalized_causal_connections': causal_connections / max(1, segment_length),
            'informative_length_ratio': informative_length_ratio,
            'causal_connections': causal_connections,
            'segment_length': segment_length,
            'weights': {'w1': self.w1, 'w2': self.w2, 'w3': self.w3},
            
            # Legacy scores for backward compatibility
            'total_score': min(legacy_total_score, 100.0),  # Cap at 100
            'base_score': self.base_score,
            'individual_pattern_scores': individual_scores,
            'cluster_scores': cluster_scores,
            'pattern_matches': all_matches,
            'clusters': clusters,
            'analysis': self._generate_analysis(all_matches, clusters)
        }
    
    def _calculate_refined_score(self, similarity_score: float, causal_connections: int, 
                               segment_length: int, informative_length_ratio: float) -> float:
        """
        Calculate the refined score using the weighted formula.
        
        Args:
            similarity_score: Semantic similarity score between 0-1
            causal_connections: Number of detected causal relationships
            segment_length: Length of text segment in characters
            informative_length_ratio: Ratio of non-stopword content to total
            
        Returns:
            Refined final score
        """
        # Avoid division by zero
        normalized_causal_density = causal_connections / max(1, segment_length)
        
        score_final = (
            self.w1 * similarity_score +
            self.w2 * normalized_causal_density +
            self.w3 * informative_length_ratio
        )
        
        return score_final
    
    @staticmethod
    def _count_causal_connections(all_matches: Dict[str, List[PatternMatch]]) -> int:
        """
        Count causal connections based on detected patterns.
        
        Args:
            all_matches: Dictionary of pattern matches by type
            
        Returns:
            Number of causal connections detected
        """
        # Count connections between baseline, target, and timeframe patterns
        baseline_count = len(all_matches.get('baseline', []))
        target_count = len(all_matches.get('target', []))
        timeframe_count = len(all_matches.get('timeframe', []))
        
        # Simple causal connection counting: sum of all pattern matches
        # More sophisticated logic could consider proximity between different pattern types
        total_connections = baseline_count + target_count + timeframe_count
        
        # Bonus for having all three pattern types (complete causal chain)
        if baseline_count > 0 and target_count > 0 and timeframe_count > 0:
            total_connections += 2  # Bonus for completeness
            
        return total_connections
    
    def _calculate_informative_length_ratio(self, text: str) -> float:
        """
        Calculate the ratio of informative content to total content.
        
        Informative content is measured by non-stopword tokens and
        excludes common Spanish stopwords.
        
        Args:
            text: Text to analyze
            
        Returns:
            Ratio of informative content (0.0 to 1.0)
        """
        import re
        
        # Tokenize text (simple word splitting)
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
            
        # Count non-stopword tokens
        informative_words = [word for word in words if word not in self.spanish_stopwords]
        
        # Calculate ratio
        informative_ratio = len(informative_words) / len(words)
        
        # Alternative: character-based ratio (excluding whitespace)
        total_chars = len(text)
        non_whitespace_chars = len(text.replace(' ', '').replace('\t', '').replace('\n', ''))
        
        if total_chars == 0:
            char_ratio = 0.0
        else:
            char_ratio = non_whitespace_chars / total_chars
        
        # Use average of token-based and character-based ratios
        return (informative_ratio + char_ratio) / 2.0
    
    def _calculate_individual_scores(self, matches: Dict[str, List[PatternMatch]]) -> Dict:
        """Calculate scores for individual pattern types."""
        scores = {}
        total = 0.0
        
        for pattern_type, pattern_matches in matches.items():
            if pattern_matches:
                # Base bonus for having this pattern type
                type_score = self.individual_pattern_bonus
                
                # Bonus for multiple instances (diminishing returns)
                if len(pattern_matches) > 1:
                    type_score += self.multiple_instance_bonus * math.log(len(pattern_matches))
                
                scores[pattern_type] = {
                    'score': type_score,
                    'count': len(pattern_matches),
                    'matches': pattern_matches
                }
                total += type_score
            else:
                scores[pattern_type] = {
                    'score': 0.0,
                    'count': 0,
                    'matches': []
                }
        
        scores['total'] = total
        return scores
    
    def _calculate_cluster_scores(self, clusters: List[Dict]) -> Dict:
        """Calculate scores for pattern clusters."""
        cluster_scores = []
        total_score = 0.0
        
        for cluster in clusters:
            # Base cluster bonus
            cluster_score = self.cluster_bonus
            
            # Proximity bonus (inverse relationship with span)
            span = cluster['span']
            proximity_factor = max(0, 1 - (span / self.proximity_window))
            proximity_bonus = self.proximity_bonus_max * proximity_factor
            cluster_score += proximity_bonus
            
            # Density bonus (more patterns in smaller space)
            total_matches = (len(cluster['matches']['baseline']) + 
                           len(cluster['matches']['target']) + 
                           len(cluster['matches']['timeframe']))
            density_bonus = min(5.0, total_matches * 1.5)
            cluster_score += density_bonus
            
            cluster_info = {
                'base_bonus': self.cluster_bonus,
                'proximity_bonus': proximity_bonus,
                'density_bonus': density_bonus,
                'total_score': cluster_score,
                'span': span,
                'match_count': total_matches,
                'cluster': cluster
            }
            
            cluster_scores.append(cluster_info)
            total_score += cluster_score
        
        return {
            'clusters': cluster_scores,
            'total': total_score,
            'count': len(clusters)
        }
    
    @staticmethod
    def _generate_analysis(matches: Dict[str, List[PatternMatch]], 
                          clusters: List[Dict]) -> Dict:
        """Generate human-readable analysis of the scoring."""
        analysis = {
            'has_baseline': len(matches['baseline']) > 0,
            'has_target': len(matches['target']) > 0,
            'has_timeframe': len(matches['timeframe']) > 0,
            'has_complete_cluster': len(clusters) > 0,
            'pattern_counts': {
                'baseline': len(matches['baseline']),
                'target': len(matches['target']),
                'timeframe': len(matches['timeframe'])
            },
            'cluster_count': len(clusters),
            'strengths': [],
            'weaknesses': []
        }
        
        # Identify strengths
        if analysis['has_complete_cluster']:
            analysis['strengths'].append("Contiene patrones completos de línea base, metas y plazos")
        
        if len(clusters) > 1:
            analysis['strengths'].append("Múltiples grupos de patrones completos identificados")
        
        if any(len(matches[pt]) > 2 for pt in matches):
            analysis['strengths'].append("Rica en indicadores específicos")
        
        # Identify weaknesses
        if not analysis['has_baseline']:
            analysis['weaknesses'].append("Falta indicadores de línea base o situación inicial")
        
        if not analysis['has_target']:
            analysis['weaknesses'].append("Falta indicadores de metas u objetivos claros")
        
        if not analysis['has_timeframe']:
            analysis['weaknesses'].append("Falta indicadores temporales o plazos específicos")
        
        if not analysis['has_complete_cluster']:
            analysis['weaknesses'].append("Los patrones no aparecen agrupados en el texto")
        
        return analysis
    
    def score_segments(self, text: str, segment_size: int = 1000, 
                      overlap: int = 200) -> List[Dict]:
        """
        Score text in segments for more granular analysis.
        
        Args:
            text: Text to analyze
            segment_size: Size of each segment in characters
            overlap: Overlap between segments in characters
            
        Returns:
            List of segment scores
        """
        segments = []
        start = 0
        segment_id = 0
        
        while start < len(text):
            end = min(start + segment_size, len(text))
            segment_text = text[start:end]
            
            score_result = self.score_text(segment_text)
            score_result.update({
                'segment_id': segment_id,
                'start_pos': start,
                'end_pos': end,
                'text': segment_text
            })
            
            segments.append(score_result)
            
            start += segment_size - overlap
            segment_id += 1
            
            if end >= len(text):
                break
        
        return segments