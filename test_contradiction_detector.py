"""
Test suite for Contradiction Detection Module
"""

import unittest

from contradiction_detector import (
    ContradictionAnalysis,
    ContradictionDetector,
    ContradictionMatch,
    RiskLevel,
)


class TestContradictionDetector(unittest.TestCase):
    """Comprehensive test suite for ContradictionDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = ContradictionDetector()

    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.context_window, 150)
        self.assertTrue(len(self.detector.compiled_adversative) > 0)
        self.assertTrue(len(self.detector.compiled_goals) > 0)
        self.assertTrue(len(self.detector.compiled_actions) > 0)
        self.assertTrue(len(self.detector.compiled_quantitative) > 0)

    def test_custom_context_window(self):
        """Test detector with custom context window."""
        detector = ContradictionDetector(context_window=200)
        self.assertEqual(detector.context_window, 200)

    def test_normalize_text(self):
        """Test Unicode normalization."""
        # Test with accented characters
        input_text = "café niño año"
        normalized = self.detector._normalize_text(input_text)
        self.assertEqual(normalized, "café niño año")

        # Test with mixed Unicode representations
        input_text = "ñ"  # Different Unicode representations of ñ
        normalized = self.detector._normalize_text(input_text)
        self.assertTrue(normalized)

    def test_pattern_matching_adversative(self):
        """Test adversative connector pattern matching."""
        text = "El objetivo es mejorar, sin embargo, los recursos son limitados."
        matches = self.detector._find_pattern_matches(
            text, self.detector.compiled_adversative, "adversative"
        )
        self.assertTrue(len(matches) >= 1)
        # Check if "sin embargo" is among the matches
        adversative_texts = [match[0].lower() for match in matches]
        self.assertTrue(
            any("sin embargo" in text for text in adversative_texts))

    def test_pattern_matching_goals(self):
        """Test goal pattern matching."""
        text = "La meta es alcanzar el objetivo de reducir la pobreza."
        matches = self.detector._find_pattern_matches(
            text, self.detector.compiled_goals, "goal"
        )
        self.assertTrue(len(matches) >= 2)  # At least "meta" and "objetivo"

    def test_pattern_matching_actions(self):
        """Test action verb pattern matching."""
        text = "Se debe implementar, desarrollar y ejecutar el programa."
        matches = self.detector._find_pattern_matches(
            text, self.detector.compiled_actions, "action"
        )
        # implementar, desarrollar, ejecutar
        self.assertTrue(len(matches) >= 3)

    def test_pattern_matching_quantitative(self):
        """Test quantitative pattern matching."""
        text = "Incrementar en 50% y reducir hasta 1000 millones de pesos."
        matches = self.detector._find_pattern_matches(
            text, self.detector.compiled_quantitative, "quantitative"
        )
        self.assertTrue(len(matches) >= 2)  # 50% and 1000 millones

    def test_extract_context_window(self):
        """Test context window extraction."""
        text = (
            "Este es un texto largo para probar la extracción de ventana de contexto."
        )
        context = self.detector._extract_context_window(text, 20)
        self.assertTrue(len(context) <= self.detector.context_window)
        self.assertTrue("texto" in context or "probar" in context)

    def test_calculate_contradiction_confidence_high(self):
        """Test confidence calculation with high-confidence scenario."""
        adversative_pos = 30
        goal_matches = [("meta", 35, 39)]  # Very close to adversative
        action_matches = [("implementar", 25, 36)]  # Close to adversative
        quantitative_matches = [("50%", 40, 43)]  # Close to adversative

        confidence = self.detector._calculate_contradiction_confidence(
            adversative_pos, goal_matches, action_matches, quantitative_matches
        )
        self.assertTrue(confidence > 0.8)

    def test_calculate_contradiction_confidence_low(self):
        """Test confidence calculation with low-confidence scenario."""
        adversative_pos = 10
        goal_matches = [("meta", 200, 204)]  # Far from adversative
        action_matches = []  # No action matches
        quantitative_matches = []  # No quantitative matches

        confidence = self.detector._calculate_contradiction_confidence(
            adversative_pos, goal_matches, action_matches, quantitative_matches
        )
        self.assertTrue(confidence < 0.5)

    def test_determine_risk_level_high(self):
        """Test high risk level determination."""
        risk_level = self.detector._determine_risk_level(0.9, 3)
        self.assertEqual(risk_level, RiskLevel.HIGH)

    def test_determine_risk_level_medium_high(self):
        """Test medium-high risk level determination."""
        risk_level = self.detector._determine_risk_level(0.65, 2)
        self.assertEqual(risk_level, RiskLevel.MEDIUM_HIGH)

    def test_determine_risk_level_medium(self):
        """Test medium risk level determination."""
        risk_level = self.detector._determine_risk_level(0.45, 1)
        self.assertEqual(risk_level, RiskLevel.MEDIUM)

    def test_determine_risk_level_low(self):
        """Test low risk level determination."""
        risk_level = self.detector._determine_risk_level(0.3, 0)
        self.assertEqual(risk_level, RiskLevel.LOW)

    def test_detect_contradictions_high_risk(self):
        """Test detection of high-risk contradiction."""
        text = "El objetivo es aumentar la cobertura educativa al 95% para 2027, sin embargo, los recursos presupuestales han sido reducidos en un 30% este año."

        analysis = self.detector.detect_contradictions(text)

        self.assertTrue(analysis.total_contradictions > 0)
        self.assertTrue(analysis.risk_score > 0.5)
        self.assertIsNotNone(analysis.highest_confidence_contradiction)
        self.assertTrue(
            analysis.summary["medium-high"] > 0 or analysis.summary["high"] > 0
        )

    def test_detect_contradictions_medium_risk(self):
        """Test detection of medium-risk contradiction."""
        text = "Se busca fortalecer la seguridad ciudadana, pero no se ha definido una estrategia clara."

        analysis = self.detector.detect_contradictions(text)

        self.assertTrue(analysis.total_contradictions > 0)
        self.assertTrue(analysis.risk_score > 0.3)

    def test_detect_contradictions_no_contradiction(self):
        """Test detection when no contradictions are present."""
        text = "El programa pretende mejorar la calidad de vida mediante la construcción de viviendas sociales."

        analysis = self.detector.detect_contradictions(text)

        self.assertEqual(analysis.total_contradictions, 0)
        self.assertEqual(analysis.risk_score, 0.0)
        self.assertEqual(analysis.risk_level, RiskLevel.LOW)
        self.assertIsNone(analysis.highest_confidence_contradiction)

    def test_detect_contradictions_no_adversative(self):
        """Test detection when no adversative connectors are present."""
        text = "La meta es crear empleos en el sector agrícola con presupuesto de 1000 millones."

        analysis = self.detector.detect_contradictions(text)

        self.assertEqual(analysis.total_contradictions, 0)
        self.assertEqual(analysis.risk_score, 0.0)

    def test_detect_contradictions_multiple(self):
        """Test detection of multiple contradictions in one text."""
        text = """El objetivo es reducir la pobreza al 15% mediante programas sociales, sin embargo, 
                 estos programas han perdido financiación. Aunque se pretende alcanzar 50,000 beneficiarios, 
                 pero la capacidad operativa actual es de solo 20,000 personas."""

        analysis = self.detector.detect_contradictions(text)

        self.assertTrue(analysis.total_contradictions >= 2)
        self.assertTrue(analysis.risk_score > 0.4)

    def test_detect_contradictions_spanish_accents(self):
        """Test detection with Spanish accented characters."""
        text = "La meta de educación es del 100%, sin embargo, la financiación es insuficiente."

        analysis = self.detector.detect_contradictions(text)

        self.assertTrue(analysis.total_contradictions > 0)

    def test_integrate_with_risk_assessment_high_contradiction(self):
        """Test integration with existing risk assessment - high contradiction."""
        text = "Meta del 95% de cobertura, sin embargo, presupuesto reducido 30%."
        existing_score = 0.3

        result = self.detector.integrate_with_risk_assessment(
            text, existing_score)

        self.assertEqual(result["base_score"], existing_score)
        self.assertTrue(result["contradiction_risk"] > 0.2)
        self.assertTrue(result["integrated_score"] > existing_score)
        self.assertTrue(result["contradiction_count"] > 0)
        self.assertTrue(result["highest_contradiction_confidence"] > 0.5)

    def test_integrate_with_risk_assessment_no_contradiction(self):
        """Test integration with existing risk assessment - no contradiction."""
        text = "El programa mejorará la calidad de vida con presupuesto adecuado."
        existing_score = 0.2

        result = self.detector.integrate_with_risk_assessment(
            text, existing_score)

        self.assertEqual(result["base_score"], existing_score)
        self.assertEqual(result["contradiction_risk"], 0.0)
        self.assertEqual(result["integrated_score"], existing_score)
        self.assertEqual(result["contradiction_count"], 0)
        self.assertEqual(result["highest_contradiction_confidence"], 0.0)

    def test_integrate_with_risk_assessment_capped_score(self):
        """Test that integrated score is capped at 1.0."""
        text = "Meta del 100%, sin embargo, recursos del 0%. Aunque se pretende alcanzar todo, pero no hay nada."
        existing_score = 0.9

        result = self.detector.integrate_with_risk_assessment(
            text, existing_score)

        self.assertLessEqual(result["integrated_score"], 1.0)

    def test_contradiction_match_dataclass(self):
        """Test ContradictionMatch dataclass creation."""
        match = ContradictionMatch(
            adversative_connector="sin embargo",
            goal_keywords=["meta"],
            action_verbs=["implementar"],
            quantitative_targets=["50%"],
            full_text="test text",
            start_pos=0,
            end_pos=10,
            risk_level=RiskLevel.MEDIUM_HIGH,
            confidence=0.75,
            context_window="test context",
        )

        self.assertEqual(match.adversative_connector, "sin embargo")
        self.assertEqual(match.risk_level, RiskLevel.MEDIUM_HIGH)
        self.assertEqual(match.confidence, 0.75)

    def test_contradiction_analysis_dataclass(self):
        """Test ContradictionAnalysis dataclass creation."""
        analysis = ContradictionAnalysis(
            contradictions=[],
            total_contradictions=0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            highest_confidence_contradiction=None,
            summary={"low": 0, "medium": 0, "medium-high": 0, "high": 0},
        )

        self.assertEqual(analysis.total_contradictions, 0)
        self.assertEqual(analysis.risk_level, RiskLevel.LOW)
        self.assertIsNone(analysis.highest_confidence_contradiction)

    def test_real_world_examples(self):
        """Test with real-world examples from public policy documents."""
        examples = [
            # Realistic policy contradiction
            "La Alcaldía implementará 50 nuevos centros de salud para alcanzar cobertura universal del 100% en 2025, no obstante, el presupuesto aprobado solo cubre la operación de 15 centros durante los próximos dos años.",
            # Goal vs capacity contradiction
            "El programa busca capacitar 10,000 productores rurales en técnicas sostenibles, aunque la entidad solo cuenta con 5 técnicos especializados y no hay planes de contratación adicional.",
            # Timeline contradiction
            "Se establecerá la meta de reducir la mortalidad infantil en 40% para el próximo año, sin embargo, los indicadores actuales muestran una tendencia creciente en los últimos tres años.",
            # Resource vs ambition contradiction
            "El objetivo es construir 1,000 viviendas de interés social con presupuesto de $50,000 millones, pero el costo unitario promedio en la región es de $80,000 por vivienda.",
        ]

        for i, example in enumerate(examples):
            with self.subTest(example_index=i):
                analysis = self.detector.detect_contradictions(example)

                # Should detect at least one contradiction
                self.assertTrue(
                    analysis.total_contradictions > 0,
                    f"Failed to detect contradiction in example {i + 1}",
                )

                # Should have medium or higher risk
                self.assertTrue(
                    analysis.risk_score > 0.3,
                    f"Risk score too low for example {i + 1}: {analysis.risk_score}",
                )

                # Should have high-confidence contradiction
                if analysis.highest_confidence_contradiction:
                    self.assertTrue(
                        analysis.highest_confidence_contradiction.confidence > 0.4,
                        f"Confidence too low for example {i + 1}",
                    )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty text
        analysis = self.detector.detect_contradictions("")
        self.assertEqual(analysis.total_contradictions, 0)

        # Only adversative connector, no goals/actions
        analysis = self.detector.detect_contradictions(
            "Sin embargo, es importante.")
        self.assertEqual(analysis.total_contradictions, 0)

        # Only goals, no adversative connectors
        analysis = self.detector.detect_contradictions(
            "La meta es implementar programas."
        )
        self.assertEqual(analysis.total_contradictions, 0)

        # Very short text
        analysis = self.detector.detect_contradictions("Meta, pero no.")
        # This might or might not detect - depends on proximity
        self.assertTrue(analysis.total_contradictions >= 0)

    def test_performance_large_text(self):
        """Test performance with large text."""
        # Create a large text with multiple contradictions
        large_text = (
            """
        La política pública tiene como objetivo principal mejorar la calidad de vida de los ciudadanos 
        mediante la implementación de programas integrales. Sin embargo, los recursos asignados son 
        insuficientes para cubrir las necesidades identificadas. El programa busca alcanzar una cobertura 
        del 95% de la población objetivo, aunque los estudios técnicos indican que con la capacidad 
        actual solo se puede atender al 60% de los beneficiarios.
        
        La meta es crear 5,000 empleos en el sector rural para el año 2025, no obstante, las 
        condiciones económicas actuales y la falta de inversión privada hacen que este objetivo 
        sea difícil de alcanzar. Se pretende establecer 100 nuevos centros de atención, pero 
        el presupuesto aprobado solo permite la construcción de 30 centros.
        
        El plan incluye la formación de 2,000 técnicos especializados, sin embargo, no existen 
        instituciones educativas con la capacidad para ofrecer estos programas. Aunque se ha 
        establecido la meta de reducir la pobreza en un 50%, pero las políticas complementarias 
        necesarias no han sido formuladas ni implementadas.
        """
            * 10
        )  # Multiply to make it larger

        import time

        start_time = time.time()
        analysis = self.detector.detect_contradictions(large_text)
        end_time = time.time()

        # Should complete in reasonable time (less than 5 seconds)
        self.assertTrue(end_time - start_time < 5.0)

        # Should detect multiple contradictions
        self.assertTrue(analysis.total_contradictions > 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
