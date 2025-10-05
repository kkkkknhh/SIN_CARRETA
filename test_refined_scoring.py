#!/usr/bin/env python3
"""
Comprehensive tests for the refined scoring formula implementation.
"""

import unittest
from factibilidad import FactibilidadScorer


class TestRefinedScoring(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = FactibilidadScorer()
        
    def test_default_weights(self):
        """Test default weight values."""
        self.assertEqual(self.scorer.w1, 0.5)
        self.assertEqual(self.scorer.w2, 0.3)
        self.assertEqual(self.scorer.w3, 0.2)
        
    def test_weight_validation_success(self):
        """Test successful weight validation."""
        # These should all pass
        valid_weight_sets = [
            (0.4, 0.4, 0.2),
            (0.6, 0.3, 0.1),
            (0.33, 0.33, 0.34),
            (0.5, 0.25, 0.25),
        ]
        
        for w1, w2, w3 in valid_weight_sets:
            with self.subTest(w1=w1, w2=w2, w3=w3):
                scorer = FactibilidadScorer(w1=w1, w2=w2, w3=w3)
                self.assertEqual(scorer.w1, w1)
                self.assertEqual(scorer.w2, w2)
                self.assertEqual(scorer.w3, w3)
                
    def test_weight_validation_failure(self):
        """Test weight validation failure cases."""
        invalid_weight_sets = [
            (0.1, 0.1, 0.1),  # Sum too low (0.3)
            (0.7, 0.7, 0.7),  # Sum too high (2.1)
            (0.4, 0.2, 0.1),  # Sum too low (0.7)
            (0.5, 0.5, 0.3),  # Sum too high (1.3)
        ]
        
        for w1, w2, w3 in invalid_weight_sets:
            with self.subTest(w1=w1, w2=w2, w3=w3):
                with self.assertRaises(ValueError):
                    FactibilidadScorer(w1=w1, w2=w2, w3=w3)
                    
    def test_weight_update(self):
        """Test weight update functionality."""
        scorer = FactibilidadScorer()
        
        # Update individual weights
        scorer.update_weights(w1=0.4)
        self.assertEqual(scorer.w1, 0.4)
        self.assertEqual(scorer.w2, 0.3)  # Unchanged
        self.assertEqual(scorer.w3, 0.2)  # Unchanged
        
        # Update all weights
        scorer.update_weights(w1=0.6, w2=0.25, w3=0.15)
        self.assertEqual(scorer.w1, 0.6)
        self.assertEqual(scorer.w2, 0.25)
        self.assertEqual(scorer.w3, 0.15)
        
    def test_weight_update_validation_failure(self):
        """Test weight update validation failure."""
        scorer = FactibilidadScorer()
        
        with self.assertRaises(ValueError):
            scorer.update_weights(w1=0.1, w2=0.1, w3=0.1)  # Sum too low
            
    def test_refined_score_calculation(self):
        """Test the refined scoring formula calculation."""
        text = """
        La línea base actual muestra 100 usuarios registrados.
        Nuestro objetivo es alcanzar 500 usuarios para diciembre 2024.
        """
        
        result = self.scorer.score_text(text, similarity_score=0.8)
        
        # Check all required components are present
        self.assertIn('score_final', result)
        self.assertIn('similarity_score', result)
        self.assertIn('causal_density', result)
        self.assertIn('informative_length_ratio', result)
        self.assertIn('causal_connections', result)
        self.assertIn('segment_length', result)
        self.assertIn('weights', result)
        
        # Check data types and ranges
        self.assertIsInstance(result['score_final'], float)
        self.assertIsInstance(result['causal_density'], float)
        self.assertIsInstance(result['informative_length_ratio'], float)
        self.assertIsInstance(result['causal_connections'], int)
        self.assertIsInstance(result['segment_length'], int)
        
        # Check ranges
        self.assertGreaterEqual(result['informative_length_ratio'], 0.0)
        self.assertLessEqual(result['informative_length_ratio'], 1.0)
        self.assertGreaterEqual(result['causal_density'], 0.0)
        self.assertEqual(result['similarity_score'], 0.8)
        
        # Verify formula calculation
        expected_score = (
            self.scorer.w1 * 0.8 +
            self.scorer.w2 * result['causal_density'] +
            self.scorer.w3 * result['informative_length_ratio']
        )
        self.assertAlmostEqual(result['score_final'], expected_score, places=6)
        
    def test_causal_connection_counting(self):
        """Test causal connection counting logic."""
        test_cases = [
            {
                'text': 'línea base actual objetivo meta para 2024',
                'min_connections': 5,  # baseline + target + timeframe + bonus
                'description': 'Complete causal chain'
            },
            {
                'text': 'objetivo principal meta importante',
                'min_connections': 2,  # Only targets
                'description': 'Only target patterns'
            },
            {
                'text': 'línea base situación inicial',
                'min_connections': 2,  # Only baselines
                'description': 'Only baseline patterns'
            },
            {
                'text': 'contenido sin patrones específicos',
                'min_connections': 0,  # No patterns
                'description': 'No patterns'
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case['description']):
                result = self.scorer.score_text(case['text'])
                self.assertGreaterEqual(
                    result['causal_connections'],
                    case['min_connections'],
                    f"Expected at least {case['min_connections']} connections for '{case['text']}'"
                )
                
    def test_informative_content_calculation(self):
        """Test informative content ratio calculation."""
        test_cases = [
            {
                'text': 'objetivo meta alcanzar conseguir lograr',
                'expected_min': 0.8,  # Mostly non-stopwords
                'description': 'High informative content'
            },
            {
                'text': 'el la de que y en un es se no te lo le',
                'expected_max': 0.5,  # Mostly stopwords
                'description': 'Low informative content'
            },
            {
                'text': '',
                'expected': 0.0,  # Empty text
                'description': 'Empty text'
            },
            {
                'text': '   \n\t   ',
                'expected': 0.0,  # Only whitespace
                'description': 'Whitespace only'
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case['description']):
                result = self.scorer.score_text(case['text'])
                
                if 'expected' in case:
                    self.assertAlmostEqual(
                        result['informative_length_ratio'],
                        case['expected'],
                        places=1,
                        msg=f"Expected {case['expected']} for '{case['text']}'"
                    )
                elif 'expected_min' in case:
                    self.assertGreaterEqual(
                        result['informative_length_ratio'],
                        case['expected_min'],
                        f"Expected at least {case['expected_min']} for '{case['text']}'"
                    )
                elif 'expected_max' in case:
                    self.assertLessEqual(
                        result['informative_length_ratio'],
                        case['expected_max'],
                        f"Expected at most {case['expected_max']} for '{case['text']}'"
                    )
                    
    def test_segment_length_normalization(self):
        """Test that causal density is properly normalized by segment length."""
        short_text = "línea base objetivo 2024"
        long_text = short_text + " con contenido adicional" * 50
        
        short_result = self.scorer.score_text(short_text)
        long_result = self.scorer.score_text(long_text)
        
        # Long text should have lower causal density due to normalization
        self.assertGreater(
            short_result['causal_density'],
            long_result['causal_density'],
            "Shorter text should have higher causal density"
        )
        
        # Both should have same number of causal connections
        self.assertEqual(
            short_result['causal_connections'],
            long_result['causal_connections'],
            "Both texts should have same causal connection count"
        )
        
    def test_backward_compatibility(self):
        """Test that legacy scoring fields are still available."""
        text = "línea base objetivo para 2024"
        result = self.scorer.score_text(text)
        
        # Legacy fields should still be present
        legacy_fields = [
            'total_score', 'base_score', 'individual_pattern_scores',
            'cluster_scores', 'pattern_matches', 'clusters', 'analysis'
        ]
        
        for field in legacy_fields:
            self.assertIn(field, result, f"Legacy field '{field}' missing")
            
    def test_different_similarity_scores(self):
        """Test scoring with different similarity score inputs."""
        text = "línea base objetivo meta para 2024"
        similarity_scores = [0.0, 0.3, 0.5, 0.8, 1.0]
        
        results = []
        for sim_score in similarity_scores:
            result = self.scorer.score_text(text, sim_score)
            results.append(result['score_final'])
            
        # Final scores should increase with similarity score
        for i in range(1, len(results)):
            self.assertGreater(
                results[i], results[i-1],
                f"Score with similarity {similarity_scores[i]} should be higher than {similarity_scores[i-1]}"
            )
            
    def test_custom_weight_impact(self):
        """Test that custom weights actually impact the final score."""
        text = "línea base objetivo meta para el año 2024"
        similarity_score = 0.6
        
        # Test with similarity-heavy weights
        sim_heavy_scorer = FactibilidadScorer(w1=0.8, w2=0.1, w3=0.1)
        sim_result = sim_heavy_scorer.score_text(text, similarity_score)
        
        # Test with causal-heavy weights  
        causal_heavy_scorer = FactibilidadScorer(w1=0.1, w2=0.8, w3=0.1)
        causal_result = causal_heavy_scorer.score_text(text, similarity_score)
        
        # Test with informative-heavy weights
        info_heavy_scorer = FactibilidadScorer(w1=0.1, w2=0.1, w3=0.8)
        info_result = info_heavy_scorer.score_text(text, similarity_score)
        
        # Results should be different based on weight emphasis
        scores = [sim_result['score_final'], causal_result['score_final'], info_result['score_final']]
        self.assertEqual(len(set(scores)), 3, "Different weight configurations should produce different scores")


if __name__ == '__main__':
    unittest.main(verbosity=2)