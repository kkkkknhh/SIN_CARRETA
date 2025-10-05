"""
TEST SUITE FOR CAUSAL PATTERN DETECTOR
======================================

Comprehensive test suite for the CausalPatternDetector class, including tests for:
- New causal connector patterns (implica, conduce a, mediante, por medio de, tendencia a)
- Unicode normalization integration
- Pattern weighting and scoring
- False positive detection and mitigation
- Context-aware confidence adjustment
- Edge cases and regression testing
"""

import unittest

from causal_pattern_detector import CausalPatternDetector


class TestCausalPatternDetector(unittest.TestCase):
    """Test suite for causal pattern detection with focus on new patterns."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = CausalPatternDetector()

        # Test texts with new patterns
        self.test_texts = {
            # New pattern: implica
            "implica_causal": "El aumento de temperatura implica mayor evaporación del agua.",
            "implica_logical": "Esta ecuación implica que x debe ser positivo.",
            "implica_negated": "Esto no implica una relación causal directa.",
            "implica_question": "¿Esto implica que debemos cambiar el plan?",
            # New pattern: conduce a
            "conduce_causal": "La falta de inversión conduce a la obsolescencia tecnológica.",
            "conduce_variations": "Este proceso conduce hacia mejores resultados.",
            "conduce_conditional": "Si seguimos así, podría conducir al fracaso.",
            # New pattern: mediante
            "mediante_causal": "La empresa redujo costos mediante la automatización.",
            "mediante_instrumental": "El análisis se realizó mediante una herramienta especializada.",
            "mediante_method": "La medición se efectúa mediante el método estándar.",
            # New pattern: por medio de
            "por_medio_causal": "Se logró el objetivo por medio de la colaboración.",
            "por_medio_instrumental": "La técnica funciona por medio de la resonancia magnética.",
            # New pattern: tendencia a
            "tendencia_causal": "Los individuos muestran tendencia a repetir comportamientos exitosos.",
            "tendencia_statistical": "Los datos muestran una tendencia a la alza en los gráficos.",
            "tendencia_correlation": "La estadística revela tendencia a la correlación positiva.",
            # Multiple patterns in one text
            "multiple_patterns": """
            El cambio climático implica mayores riesgos ambientales. Esto conduce a 
            la necesidad de adaptación mediante nuevas tecnologías. Por medio de la 
            investigación observamos tendencia a mejores resultados.
            """,
            # Existing patterns for regression testing
            "existing_porque": "El proyecto falló porque no hubo suficiente financiación.",
            "existing_debido": "La demora se produjo debido a problemas técnicos.",
            "existing_ya_que": "Se canceló la reunión ya que no había quórum.",
        }

    def test_new_pattern_implica_detection(self):
        """Test detection of 'implica' pattern with context awareness."""
        # Should detect causal usage with high confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["implica_causal"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "implica")
        self.assertGreater(matches[0].confidence, 0.5)

        # Should detect logical usage with lower confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["implica_logical"]
        )
        self.assertEqual(len(matches), 1)
        self.assertLess(
            matches[0].confidence, 0.5
        )  # Reduced due to mathematical context

        # Should detect negated usage with very low confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["implica_negated"]
        )
        self.assertEqual(len(matches), 1)
        self.assertLess(matches[0].confidence, 0.3)  # Reduced due to negation

    def test_new_pattern_conduce_detection(self):
        """Test detection of 'conduce a' pattern and variations."""
        matches = self.detector.detect_causal_patterns(
            self.test_texts["conduce_causal"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "conduce a")
        self.assertGreater(matches[0].confidence, 0.7)

        # Test variations like "conduce hacia"
        matches = self.detector.detect_causal_patterns(
            self.test_texts["conduce_variations"]
        )
        self.assertEqual(len(matches), 1)

    def test_new_pattern_mediante_detection(self):
        """Test detection of 'mediante' pattern with instrumental vs causal distinction."""
        # Causal usage should have moderate confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["mediante_causal"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "mediante")

        # Instrumental usage should have lower confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["mediante_instrumental"]
        )
        self.assertEqual(len(matches), 1)
        instrumental_confidence = matches[0].confidence

        matches = self.detector.detect_causal_patterns(
            self.test_texts["mediante_causal"]
        )
        causal_confidence = matches[0].confidence

        # Causal usage should have higher confidence than instrumental
        self.assertGreater(causal_confidence, instrumental_confidence)

    def test_new_pattern_por_medio_detection(self):
        """Test detection of 'por medio de' pattern."""
        matches = self.detector.detect_causal_patterns(
            self.test_texts["por_medio_causal"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "por medio de")
        self.assertGreater(matches[0].confidence, 0.4)

        # Should handle instrumental usage with appropriate confidence reduction
        matches = self.detector.detect_causal_patterns(
            self.test_texts["por_medio_instrumental"]
        )
        self.assertEqual(len(matches), 1)
        self.assertLess(matches[0].confidence, 0.6)

    def test_new_pattern_tendencia_detection(self):
        """Test detection of 'tendencia a' pattern with statistical vs causal distinction."""
        # Causal usage
        matches = self.detector.detect_causal_patterns(
            self.test_texts["tendencia_causal"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "tendencia a")

        # Statistical usage should have lower confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["tendencia_statistical"]
        )
        self.assertEqual(len(matches), 1)
        statistical_confidence = matches[0].confidence

        matches = self.detector.detect_causal_patterns(
            self.test_texts["tendencia_causal"]
        )
        causal_confidence = matches[0].confidence

        # Causal usage should have higher confidence than statistical
        self.assertGreater(causal_confidence, statistical_confidence)

    def test_multiple_patterns_in_text(self):
        """Test detection of multiple new patterns in a single text."""
        matches = self.detector.detect_causal_patterns(
            self.test_texts["multiple_patterns"]
        )

        # Should detect all new patterns
        detected_connectors = {match.connector for match in matches}
        expected_connectors = {
            "implica",
            "conduce a",
            "mediante",
            "por medio de",
            "tendencia a",
        }

        # All new patterns should be detected
        self.assertTrue(expected_connectors.issubset(detected_connectors))
        self.assertGreaterEqual(len(matches), 5)

    def test_pattern_weighting_system(self):
        """Test that new patterns have appropriate semantic strength weights."""
        weights = self.detector.get_supported_patterns()

        # Check that new patterns are present with expected relative weights
        self.assertIn("implica", weights)
        self.assertIn("conduce a", weights)
        self.assertIn("mediante", weights)
        self.assertIn("por medio de", weights)
        self.assertIn("tendencia a", weights)

        # Check relative weights align with semantic strength expectations
        self.assertLess(
            weights["tendencia a"], weights["conduce a"]
        )  # tendencia weaker than conduce
        self.assertLess(
            weights["mediante"], weights["implica"]
        )  # mediante weaker than implica
        self.assertLess(
            weights["implica"], weights["porque"]
        )  # implica weaker than porque
        self.assertGreater(
            weights["conduce a"], weights["mediante"]
        )  # conduce stronger than mediante

    def test_context_awareness_false_positive_reduction(self):
        """Test that context analysis reduces false positives appropriately."""
        # Question context should reduce confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["implica_question"]
        )
        self.assertEqual(len(matches), 1)
        self.assertLess(
            matches[0].confidence, 0.6
        )  # Should be reduced from question context

        # Conditional context should reduce confidence
        matches = self.detector.detect_causal_patterns(
            self.test_texts["conduce_conditional"]
        )
        self.assertEqual(len(matches), 1)
        self.assertLess(
            matches[0].confidence, 0.7
        )  # Should be reduced from conditional context

    def test_unicode_normalization_integration(self):
        """Test that Unicode normalization works correctly with new patterns."""
        # Test with accented characters and Unicode variations
        unicode_text = "La educación implica mejores oportunidades económicas."
        matches = self.detector.detect_causal_patterns(unicode_text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "implica")

    def test_pattern_classification(self):
        """Test that new patterns are correctly classified by type."""
        matches = self.detector.detect_causal_patterns(
            self.test_texts["implica_causal"]
        )
        self.assertEqual(matches[0].pattern_type, "implication_causation")

        matches = self.detector.detect_causal_patterns(
            self.test_texts["conduce_causal"]
        )
        self.assertEqual(matches[0].pattern_type, "generative_causation")

        matches = self.detector.detect_causal_patterns(
            self.test_texts["mediante_causal"]
        )
        self.assertEqual(matches[0].pattern_type, "instrumental_causation")

        matches = self.detector.detect_causal_patterns(
            self.test_texts["tendencia_causal"]
        )
        self.assertEqual(matches[0].pattern_type, "tendency_causation")

    def test_overlapping_pattern_resolution(self):
        """Test that overlapping patterns are resolved correctly with confidence priority."""
        # Create text with potential overlapping matches
        overlap_text = (
            "El proceso implica y conduce a mejores resultados mediante la innovación."
        )
        matches = self.detector.detect_causal_patterns(overlap_text)

        # Should not have overlapping matches
        for i, match1 in enumerate(matches):
            for j, match2 in enumerate(matches[i + 1:], i + 1):
                self.assertFalse(
                    self.detector._matches_overlap(match1, match2),
                    f"Matches {i} and {j} overlap: {match1.text} and {match2.text}",
                )

    def test_regression_existing_patterns(self):
        """Test that existing patterns still work correctly after new additions."""
        # Test that existing patterns are still detected
        matches = self.detector.detect_causal_patterns(
            self.test_texts["existing_porque"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "porque")
        self.assertGreater(matches[0].confidence, 0.9)

        matches = self.detector.detect_causal_patterns(
            self.test_texts["existing_debido"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "debido a")

        matches = self.detector.detect_causal_patterns(
            self.test_texts["existing_ya_que"]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].connector, "ya que")

    def test_pattern_statistics_calculation(self):
        """Test comprehensive pattern statistics calculation."""
        stats = self.detector.calculate_pattern_statistics(
            self.test_texts["multiple_patterns"]
        )

        self.assertGreater(stats["total_matches"], 0)
        self.assertIn("pattern_types", stats)
        self.assertIn("confidence_distribution", stats)
        self.assertGreater(stats["average_confidence"], 0)

        # Should include new pattern types
        self.assertIn("implication_causation", stats["pattern_types"])
        self.assertIn("generative_causation", stats["pattern_types"])
        self.assertIn("instrumental_causation", stats["pattern_types"])

    def test_empty_and_edge_cases(self):
        """Test edge cases and empty inputs."""
        # Empty text
        matches = self.detector.detect_causal_patterns("")
        self.assertEqual(len(matches), 0)

        # None input should be handled gracefully
        matches = self.detector.detect_causal_patterns(None) if None else []
        self.assertEqual(len(matches), 0)

        # Text without causal patterns
        matches = self.detector.detect_causal_patterns(
            "Este es un texto normal sin conectores."
        )
        self.assertEqual(len(matches), 0)

    def test_confidence_score_ranges(self):
        """Test that confidence scores are within valid ranges (0.0 to 1.0)."""
        all_texts = list(self.test_texts.values())

        for text in all_texts:
            matches = self.detector.detect_causal_patterns(text)
            for match in matches:
                self.assertGreaterEqual(match.confidence, 0.0)
                self.assertLessEqual(match.confidence, 1.0)
                self.assertIsInstance(match.confidence, float)

    def test_new_patterns_semantic_strength_ordering(self):
        """Test that new patterns follow expected semantic strength ordering."""
        weights = self.detector.get_supported_patterns()

        # Expected ordering from strongest to weakest for new patterns
        new_patterns_ordered = [
            "conduce a",  # 0.75 - medium-high
            "implica",  # 0.60 - medium
            "por medio de",  # 0.55 - medium-low
            "mediante",  # 0.50 - medium-low
            "tendencia a",  # 0.45 - lowest
        ]

        # Verify ordering is correct
        for i in range(len(new_patterns_ordered) - 1):
            current = new_patterns_ordered[i]
            next_pattern = new_patterns_ordered[i + 1]
            self.assertGreater(
                weights[current],
                weights[next_pattern],
                f"{current} should have higher weight than {next_pattern}",
            )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
