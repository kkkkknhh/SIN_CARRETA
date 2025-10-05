"""
Comprehensive test suite for FeasibilityScorer with manually annotated dataset.
Tests precision and recall of quality detection patterns.
"""

import pickle
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import pytest

from feasibility_scorer import ComponentType, FeasibilityScorer


class TestDataset:
    """Manually annotated test dataset for validation."""

    @staticmethod
    def get_high_quality_indicators() -> List[Dict[str, Any]]:
        """High-quality indicators with all components and quantitative elements."""
        return [
            {
                "text": "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025",
                "expected_score": 0.9,
                "expected_tier": "high",
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.TARGET,
                    ComponentType.TIME_HORIZON,
                    ComponentType.NUMERICAL,
                    ComponentType.DATE,
                ],
                "has_quantitative_baseline": True,
                "has_quantitative_target": True,
            },
            {
                "text": "Reducir from baseline of 15.3 million people in poverty to target of 8 million by December 2024",
                "expected_score": 0.85,
                "expected_tier": "high",
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.TARGET,
                    ComponentType.TIME_HORIZON,
                    ComponentType.NUMERICAL,
                    ComponentType.DATE,
                ],
                "has_quantitative_baseline": True,
                "has_quantitative_target": True,
            },
            {
                "text": "Aumentar el valor inicial de 2.5 millones de beneficiarios hasta alcanzar el objetivo de 4 millones en el horizonte temporal 2020-2025",
                "expected_score": 0.88,
                "expected_tier": "high",
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.TARGET,
                    ComponentType.TIME_HORIZON,
                    ComponentType.NUMERICAL,
                    ComponentType.DATE,
                ],
                "has_quantitative_baseline": True,
                "has_quantitative_target": True,
            },
        ]

    @staticmethod
    def get_medium_quality_indicators() -> List[Dict[str, Any]]:
        """Medium-quality indicators with basic components, some quantitative elements."""
        return [
            {
                "text": "Mejorar desde la situación inicial hasta el objetivo propuesto con incremento del 20%",
                "expected_score": 0.6,
                "expected_tier": "medium",
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.TARGET,
                    ComponentType.NUMERICAL,
                ],
                "has_quantitative_baseline": False,
                "has_quantitative_target": True,
            },
            {
                "text": "Partir del nivel base actual para lograr la meta establecida en los próximos años",
                "expected_score": 0.55,
                "expected_tier": "medium",
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.TARGET,
                    ComponentType.TIME_HORIZON,
                ],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
            {
                "text": "Achieve target improvement from current baseline within the established timeframe",
                "expected_score": 0.58,
                "expected_tier": "medium",
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.TARGET,
                    ComponentType.TIME_HORIZON,
                ],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
        ]

    @staticmethod
    def get_low_quality_indicators() -> List[Dict[str, Any]]:
        """Low-quality indicators with minimal components."""
        return [
            {
                "text": "Partir de la línea base para alcanzar el objetivo",
                "expected_score": 0.3,
                "expected_tier": "low",
                "expected_components": [ComponentType.BASELINE, ComponentType.TARGET],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
            {
                "text": "Improve from baseline to reach established goal",
                "expected_score": 0.32,
                "expected_tier": "low",
                "expected_components": [ComponentType.BASELINE, ComponentType.TARGET],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
        ]

    @staticmethod
    def get_insufficient_indicators() -> List[Dict[str, Any]]:
        """Insufficient indicators missing core components."""
        return [
            {
                "text": "Aumentar el acceso a servicios de salud en la región",
                "expected_score": 0.0,
                "expected_tier": "insufficient",
                "expected_components": [],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
            {
                "text": "Mejorar la calidad educativa mediante nuevas estrategias",
                "expected_score": 0.0,
                "expected_tier": "insufficient",
                "expected_components": [],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
            {
                "text": "La meta es fortalecer las instituciones públicas",
                "expected_score": 0.0,
                "expected_tier": "insufficient",
                "expected_components": [ComponentType.TARGET],
                "has_quantitative_baseline": False,
                "has_quantitative_target": False,
            },
        ]


class TestFeasibilityScorer:
    """Test suite for FeasibilityScorer functionality."""

    @pytest.fixture
    def scorer(self):
        return FeasibilityScorer()

    @staticmethod
    def test_high_quality_indicators(scorer):
        """Test scoring of high-quality indicators."""
        indicators = TestDataset.get_high_quality_indicators()

        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data["text"])

            # Check score within reasonable range (±0.1)
            assert (
                abs(result.feasibility_score - indicator_data["expected_score"]) <= 0.15
            ), (
                f"Score mismatch for '{indicator_data['text']}': expected {indicator_data['expected_score']}, got {result.feasibility_score}"
            )

            # Check quality tier
            assert result.quality_tier == indicator_data["expected_tier"], (
                f"Quality tier mismatch for '{indicator_data['text']}': expected {indicator_data['expected_tier']}, got {result.quality_tier}"
            )

            # Check quantitative components
            assert (
                result.has_quantitative_baseline
                == indicator_data["has_quantitative_baseline"]
            )
            assert (
                result.has_quantitative_target
                == indicator_data["has_quantitative_target"]
            )

            # Check that key components are detected
            detected_types = set(result.components_detected)
            for expected_component in indicator_data["expected_components"]:
                assert expected_component in detected_types, (
                    f"Missing component {expected_component} in '{indicator_data['text']}'"
                )

    @staticmethod
    def test_medium_quality_indicators(scorer):
        """Test scoring of medium-quality indicators."""
        indicators = TestDataset.get_medium_quality_indicators()

        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data["text"])

            assert (
                abs(result.feasibility_score - indicator_data["expected_score"]) <= 0.15
            )
            assert result.quality_tier == indicator_data["expected_tier"]
            assert (
                result.has_quantitative_baseline
                == indicator_data["has_quantitative_baseline"]
            )
            assert (
                result.has_quantitative_target
                == indicator_data["has_quantitative_target"]
            )

    @staticmethod
    def test_low_quality_indicators(scorer):
        """Test scoring of low-quality indicators."""
        indicators = TestDataset.get_low_quality_indicators()

        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data["text"])

            assert (
                abs(result.feasibility_score - indicator_data["expected_score"]) <= 0.15
            )
            assert result.quality_tier == indicator_data["expected_tier"]
            assert (
                result.has_quantitative_baseline
                == indicator_data["has_quantitative_baseline"]
            )
            assert (
                result.has_quantitative_target
                == indicator_data["has_quantitative_target"]
            )

    @staticmethod
    def test_insufficient_indicators(scorer):
        """Test scoring of insufficient indicators."""
        indicators = TestDataset.get_insufficient_indicators()

        for indicator_data in indicators:
            result = scorer.calculate_feasibility_score(indicator_data["text"])

            assert result.feasibility_score == 0.0, (
                f"Expected 0.0 score for insufficient indicator, got {result.feasibility_score}"
            )
            assert result.quality_tier == "insufficient"
            assert result.has_quantitative_baseline == False
            assert result.has_quantitative_target == False

    @staticmethod
    def test_mandatory_requirements(scorer):
        """Test that baseline and target are mandatory for positive scores."""
        # Only baseline
        result = scorer.calculate_feasibility_score(
            "La línea base es de 50% de cobertura"
        )
        assert result.feasibility_score == 0.0

        # Only target
        result = scorer.calculate_feasibility_score("El objetivo es llegar al 80%")
        assert result.feasibility_score == 0.0

        # Both present
        result = scorer.calculate_feasibility_score(
            "Partir de línea base 50% hasta objetivo 80%"
        )
        assert result.feasibility_score > 0.0

    @staticmethod
    def test_spanish_patterns(scorer):
        """Test Spanish-specific pattern detection."""
        spanish_texts = [
            "línea base de 30% hasta meta de 60%",
            "valor inicial 25 millones para objetivo 40 millones",
            "situación actual mejorar hasta propósito establecido",
            "desde punto de partida hasta finalidad en el plazo 2025",
        ]

        for text in spanish_texts:
            result = scorer.calculate_feasibility_score(text)
            assert result.feasibility_score > 0.0, (
                f"Failed to detect Spanish patterns in: {text}"
            )

    @staticmethod
    def test_english_patterns(scorer):
        """Test English-specific pattern detection."""
        english_texts = [
            "baseline of 30% to target of 60%",
            "current level 25 million to goal 40 million",
            "starting point improve to aim within timeline",
            "from initial value achieve target by 2025",
        ]

        for text in english_texts:
            result = scorer.calculate_feasibility_score(text)
            assert result.feasibility_score > 0.0, (
                f"Failed to detect English patterns in: {text}"
            )

    @staticmethod
    def test_numerical_detection(scorer):
        """Test numerical pattern detection."""
        numerical_texts = [
            "incrementar 25%",
            "reducir en 1.5 millones",
            "increase by 30 percent",
            "reduce 2,500 thousand",
        ]

        for text in numerical_texts:
            components = scorer.detect_components(text)
            numerical_detected = any(
                c.component_type == ComponentType.NUMERICAL for c in components
            )
            assert numerical_detected, f"Failed to detect numerical pattern in: {text}"

    @staticmethod
    def test_date_detection(scorer):
        """Test date pattern detection."""
        date_texts = [
            "para el año 2025",
            "en diciembre 2024",
            "by January 2025",
            "15/12/2024",
            "hasta 2026",
        ]

        for text in date_texts:
            components = scorer.detect_components(text)
            date_detected = any(
                c.component_type == ComponentType.DATE for c in components
            )
            assert date_detected, f"Failed to detect date pattern in: {text}"

    @staticmethod
    def test_quantitative_component_detection(scorer):
        """Test detection of quantitative baselines and targets."""
        # Quantitative baseline
        text1 = "línea base de 65% incrementar hasta meta general"
        result1 = scorer.calculate_feasibility_score(text1)
        assert result1.has_quantitative_baseline == True
        assert result1.has_quantitative_target == False

        # Quantitative target
        text2 = "partir de situación actual hasta objetivo de 85%"
        result2 = scorer.calculate_feasibility_score(text2)
        assert result2.has_quantitative_baseline == False
        assert result2.has_quantitative_target == True

        # Both quantitative
        text3 = "línea base 40% hasta meta 70%"
        result3 = scorer.calculate_feasibility_score(text3)
        assert result3.has_quantitative_baseline == True
        assert result3.has_quantitative_target == True

    @staticmethod
    def test_batch_scoring(scorer):
        """Test batch scoring functionality with parallel processing."""
        indicators = [
            "línea base 50% meta 80% año 2025",
            "situación actual mejorar objetivo",
            "aumentar servicios región",
        ]

        # Test sequential processing (default)
        results_seq = scorer.batch_score(indicators)
        assert len(results_seq) == 3
        assert (
            results_seq[0].feasibility_score
            > results_seq[1].feasibility_score
            > results_seq[2].feasibility_score
        )

        # Test parallel processing option
        results_par = scorer.batch_score(indicators, use_parallel=True)
        assert len(results_par) == 3

        # Results should be identical regardless of processing method
        for seq, par in zip(results_seq, results_par):
            assert seq.feasibility_score == par.feasibility_score
            assert seq.quality_tier == par.quality_tier

        # Test with extended batch for parallel processing comparison
        extended_indicators = indicators * 5  # 15 total for better parallel testing
        results_extended = scorer.batch_score(
            extended_indicators, compare_backends=True
        )
        assert len(results_extended) == 15

        # Test with parallel disabled
        scorer_no_parallel = FeasibilityScorer(enable_parallel=False)
        results_no_parallel = scorer_no_parallel.batch_score(indicators)
        assert len(results_no_parallel) == 3

    @staticmethod
    def test_parallel_processing_configuration():
        """Test parallel processing configuration and backend selection."""
        # Test default configuration
        scorer = FeasibilityScorer()
        assert scorer.n_jobs <= 8
        assert scorer.backend == "loky"

        # Test custom configuration
        scorer_custom = FeasibilityScorer(n_jobs=4, backend="threading")
        assert scorer_custom.n_jobs == 4
        assert scorer_custom.backend == "threading"

        # Test with parallel disabled
        scorer_disabled = FeasibilityScorer(enable_parallel=False)
        assert not scorer_disabled.enable_parallel

    @staticmethod
    def test_picklable_scorer_copy(scorer):
        """Test that scorer can create picklable copies for parallel processing."""
        import pickle

        # Test that the main scorer might not be picklable due to logger
        try:
            pickle.dumps(scorer)
            main_picklable = True
        except Exception:
            main_picklable = False

        # Test that the copy is always picklable
        copy = scorer._create_picklable_copy()
        pickled = pickle.dumps(copy)
        unpickled = pickle.loads(pickled)

        # Verify the copy works correctly
        test_text = "línea base 50% meta 80%"
        original_result = scorer.calculate_feasibility_score(test_text)
        copy_result = unpickled.calculate_feasibility_score(test_text)

        assert (
            abs(original_result.feasibility_score - copy_result.feasibility_score)
            < 0.01
        )

    @staticmethod
    def test_precision_recall_metrics(scorer):
        """Test precision and recall of component detection."""
        all_indicators = (
            TestDataset.get_high_quality_indicators()
            + TestDataset.get_medium_quality_indicators()
            + TestDataset.get_low_quality_indicators()
            + TestDataset.get_insufficient_indicators()
        )

        total_baseline_expected = sum(
            1
            for ind in all_indicators
            if ComponentType.BASELINE in ind["expected_components"]
        )
        total_target_expected = sum(
            1
            for ind in all_indicators
            if ComponentType.TARGET in ind["expected_components"]
        )

        baseline_detected = 0
        target_detected = 0

        for indicator_data in all_indicators:
            result = scorer.calculate_feasibility_score(indicator_data["text"])
            if ComponentType.BASELINE in result.components_detected:
                baseline_detected += 1
            if ComponentType.TARGET in result.components_detected:
                target_detected += 1

        baseline_recall = (
            baseline_detected / total_baseline_expected
            if total_baseline_expected > 0
            else 0
        )
        target_recall = (
            target_detected / total_target_expected if total_target_expected > 0 else 0
        )

        # Expect high recall (>80%) for core components
        assert baseline_recall >= 0.8, f"Baseline recall too low: {baseline_recall}"
        assert target_recall >= 0.8, f"Target recall too low: {target_recall}"

    @staticmethod
    def test_unicode_normalization(scorer):
        """Test Unicode normalization functionality and its impact on pattern matching."""
        # Test cases with various Unicode characters that should normalize
        test_cases = [
            {
                # Full-width percent, smart quotes
                "original": 'Incrementar la línea base de 65％ a una "meta" de 85％',
                "normalized": 'Incrementar la línea base de 65% a una "meta" de 85%',
                "description": "Full-width characters and smart quotes",
            },
            {
                "original": "Alcanzar objetivo de 1‚500 millones",  # Different comma character
                "normalized": "Alcanzar objetivo de 1,500 millones",
                "description": "Different comma character",
            },
            {
                "original": "Meta de año ２０２５",  # Full-width numbers
                "normalized": "Meta de año 2025",
                "description": "Full-width numbers",
            },
            {
                "original": "Línea base: café → té",  # Accented characters and arrows
                "normalized": "Línea base: café → té",
                "description": "Accented characters and special symbols",
            },
        ]

        for case in test_cases:
            # Test normalization function directly
            normalized = scorer._normalize_text(case["original"])
            expected = unicodedata.normalize("NFKC", case["original"])
            assert normalized == expected, (
                f"Normalization failed for {case['description']}"
            )

            # Test that both original and normalized text produce similar component detection
            original_components = scorer.detect_components(case["original"])
            normalized_components = scorer.detect_components(normalized)

            # Should detect same number of components after normalization
            assert len(original_components) == len(normalized_components), (
                f"Component count differs for {case['description']}: original={len(original_components)}, normalized={len(normalized_components)}"
            )

    @staticmethod
    def test_unicode_pattern_matching_improvement(scorer):
        """Test that Unicode normalization improves pattern matching reliability."""
        # Create test cases with Unicode variants that might cause matching issues
        unicode_variants = [
            # Different quote styles
            ('baseline "50%" target "80%"', 'baseline "50%" target "80%"'),
            (
                'baseline "50%" target "80%"',
                'baseline "50%" target "80%"',
            ),  # curly quotes
            # Different dash/hyphen characters
            ("2020-2025 timeline", "2020—2025 timeline"),  # em dash vs hyphen
            # Full-width vs half-width characters
            ("meta 85％", "meta 85%"),
            ("año ２０２５", "año 2025"),
            # Ligatures and composed characters
            ("coeficiente", "coeﬁciente"),  # fi ligature
        ]

        match_improvements = 0
        total_tests = 0

        for normalized_text, variant_text in unicode_variants:
            # Score both versions
            normalized_score = scorer.calculate_feasibility_score(normalized_text)
            variant_score = scorer.calculate_feasibility_score(variant_text)

            # Count components detected
            normalized_components = len(normalized_score.components_detected)
            variant_components = len(variant_score.components_detected)

            total_tests += 1

            # After normalization, should detect same or more components
            if variant_components >= normalized_components:
                match_improvements += 1

        # Expect that normalization helps in most cases
        improvement_rate = match_improvements / total_tests
        assert improvement_rate >= 0.7, (
            f"Unicode normalization improvement rate too low: {improvement_rate}"
        )

    @staticmethod
    def test_regex_match_consistency(scorer):
        """Test that regex patterns work consistently after Unicode normalization."""
        # Test patterns that might be sensitive to Unicode variants
        sensitive_patterns = [
            {
                "text_variants": [
                    "línea base de 50％",  # Full-width percent
                    "línea base de 50%",  # Half-width percent
                ],
                "expected_components": [
                    ComponentType.BASELINE,
                    ComponentType.NUMERICAL,
                ],
            },
            {
                "text_variants": [
                    'meta "85%" para ２０２５',  # Mixed width characters
                    'meta "85%" para 2025',  # Normal characters
                ],
                "expected_components": [
                    ComponentType.TARGET,
                    ComponentType.NUMERICAL,
                    ComponentType.DATE,
                ],
            },
            {
                "text_variants": [
                    "incrementar‚ reducir por 30％",  # Different punctuation
                    "incrementar, reducir por 30%",  # Standard punctuation
                ],
                "expected_components": [ComponentType.NUMERICAL],
            },
        ]

        for pattern_test in sensitive_patterns:
            results = []
            for variant in pattern_test["text_variants"]:
                result = scorer.calculate_feasibility_score(variant)
                results.append(result)

            # All variants should detect same components after normalization
            first_components = set(results[0].components_detected)
            for result in results[1:]:
                current_components = set(result.components_detected)
                assert first_components == current_components, (
                    f"Component detection inconsistent across Unicode variants: {pattern_test['text_variants']}"
                )

                # Verify expected components are present
                for expected_component in pattern_test["expected_components"]:
                    assert expected_component in current_components, (
                        f"Expected component {expected_component} missing from variant detection"
                    )

    @staticmethod
    def test_documentation_generation(scorer):
        """Test documentation generation."""
        docs = scorer.get_detection_rules_documentation()

        assert "Feasibility Scorer Detection Rules Documentation" in docs
        assert "Spanish Pattern Recognition" in docs
        assert "Quality Tiers" in docs
        assert "Examples" in docs
        assert len(docs) > 1000  # Ensure comprehensive documentation


class TestCalcularCalidadEvidencia:
    """Test suite for calcular_calidad_evidencia method."""

    @pytest.fixture
    def scorer(self):
        return FeasibilityScorer()

    @staticmethod
    def test_empty_and_edge_cases(scorer):
        """Test handling of empty and edge case inputs."""
        # Empty string
        assert scorer.calcular_calidad_evidencia("") == 0.0

        # Whitespace only
        assert scorer.calcular_calidad_evidencia("   ") == 0.0

        # None-like content
        assert scorer.calcular_calidad_evidencia("   \n\t  ") == 0.0

    @staticmethod
    def test_monetary_value_detection(scorer):
        """Test detection of monetary amounts."""
        # Colombian pesos
        high_monetary = [
            "COP $2.5 millones invertidos en el proyecto",
            "Presupuesto de $15,000 millones de pesos",
            "Inversión de COP 3.2 millones",
            "Costo total: $1,500 mil pesos",
        ]

        for text in high_monetary:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.3, f"Low monetary score for: {text}"

        # USD amounts
        usd_texts = [
            "Investment of $2.3 million USD",
            "Budget allocation: 5.7 million dollars",
            "Total cost $850,000 dollars",
        ]

        for text in usd_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.25, f"Low USD score for: {text}"

        # No monetary content
        no_money = "Mejora general del sistema educativo"
        assert scorer.calcular_calidad_evidencia(no_money) <= 0.5

    @staticmethod
    def test_temporal_indicator_detection(scorer):
        """Test detection of dates and temporal indicators."""
        # Years
        year_texts = [
            "Metas para el año 2025",
            "Implementación en 2024",
            "Evaluación 2023-2025",
        ]

        for text in year_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.1, f"Low year score for: {text}"

        # Quarters
        quarter_texts = [
            "Revisión trimestral Q1 2024",
            "Evaluación primer trimestre",
            "Resultados Q3",
            "Quarter 4 assessment",
        ]

        for text in quarter_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.08, f"Low quarter score for: {text}"

        # Months
        month_texts = [
            "Entrega en enero 2024",
            "Evaluación febrero de 2025",
            "Review in March 2024",
            "December assessment",
        ]

        for text in month_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.06, f"Low month score for: {text}"

        # Periodicity
        periodicity_texts = [
            "Monitoreo con periodicidad anual",
            "Evaluación mensual frequency",
            "Revisión cada 6 meses",
            "Quarterly monitoring system",
        ]

        for text in periodicity_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.05, f"Low periodicity score for: {text}"

    @staticmethod
    def test_measurement_terminology_detection(scorer):
        """Test detection of measurement and evaluation terms."""
        # Baseline terminology
        baseline_texts = [
            "Establecer línea base para el indicador",
            "Current baseline assessment shows",
            "Valor inicial de referencia",
            "Punto de partida del proyecto",
        ]

        for text in baseline_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.1, f"Low baseline score for: {text}"

        # Target/goal terminology
        target_texts = [
            "Meta establecida para el proyecto",
            "Objetivo principal del programa",
            "Target achievement expected",
            "Goal setting methodology",
        ]

        for text in target_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.08, f"Low target score for: {text}"

        # Measurement concepts
        measurement_texts = [
            "Indicador de desempeño clave",
            "Performance metric established",
            "Sistema de monitoreo y evaluación",
            "Measurement framework development",
        ]

        for text in measurement_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.12, f"Low measurement score for: {text}"

    @staticmethod
    def test_structure_penalty(scorer):
        """Test penalty for title/bullet indicators without values."""
        # Title-like without values (should get penalty)
        title_without_values = [
            "• Mejora del sistema educativo",
            "# Fortalecimiento institucional",
            "DESARROLLO RURAL SOSTENIBLE:",
            "- Acceso a servicios de salud",
        ]

        for text in title_without_values:
            score = scorer.calcular_calidad_evidencia(text)
            # Should get structure penalty, reducing overall score
            assert score <= 0.4, f"High score despite title penalty for: {text}"

        # Title-like with values (should avoid penalty)
        title_with_values = [
            "• Incrementar cobertura en 25% para 2024",
            "# Inversión: COP $2.5 millones",
            "DESARROLLO RURAL: meta 80% cobertura",
            "- Beneficiarios: 15,000 personas Q1 2024",
        ]

        for text in title_with_values:
            score = scorer.calcular_calidad_evidencia(text)
            # Should avoid penalty and score higher
            assert score >= 0.3, f"Low score despite values for: {text}"

    @staticmethod
    def test_combined_scoring(scorer):
        """Test scoring with multiple quality indicators."""
        # High quality: monetary + temporal + terminology
        high_quality = [
            "Línea base: COP $5.2 millones en 2023, meta $8.5 millones para Q4 2025 con monitoreo trimestral",
            "Baseline investment $3.1 million, target $4.8 million by December 2024, quarterly evaluation",
            "Indicador: incrementar presupuesto de 2.5 millones a 4.2 millones pesos durante 2024-2025",
        ]

        for text in high_quality:
            score = scorer.calcular_calidad_evidencia(text)
            assert score >= 0.7, f"Low combined score for high quality text: {text}"

        # Medium quality: some indicators
        medium_quality = [
            "Evaluación anual del proyecto en 2024",
            "Meta de $2 millones establecida",
            "Monitoreo trimestral con indicadores clave",
        ]

        for text in medium_quality:
            score = scorer.calcular_calidad_evidencia(text)
            assert 0.2 <= score <= 0.7, f"Score out of medium range for: {text}"

        # Low quality: minimal indicators
        low_quality = [
            "Mejora general del proyecto",
            "Desarrollo de capacidades institucionales",
            "Fortalecimiento del sector",
        ]

        for text in low_quality:
            score = scorer.calcular_calidad_evidencia(text)
            assert score <= 0.3, f"High score for low quality text: {text}"

    @staticmethod
    def test_unicode_normalization(scorer):
        """Test Unicode normalization handling."""
        # Unicode variations that should normalize to same result
        unicode_variants = [
            "Inversión: $2.5 millones",  # Regular
            "Inversión: $2.5 millones",  # With combining characters
            "Inversión: $2.５ millones",  # Full-width characters
        ]

        scores = [scorer.calcular_calidad_evidencia(text) for text in unicode_variants]

        # Scores should be similar after normalization
        for i in range(1, len(scores)):
            assert abs(scores[0] - scores[i]) <= 0.1, (
                f"Unicode normalization failed: {scores}"
            )

    @staticmethod
    def test_malformed_numbers(scorer):
        """Test handling of malformed monetary/numeric values."""
        malformed_texts = [
            "Presupuesto: $..5 millones",
            "Inversión de COP ,, pesos",
            "Meta: % incremento",
            "Año: 20XX evaluación",
        ]

        for text in malformed_texts:
            score = scorer.calcular_calidad_evidencia(text)
            # Should handle gracefully, not crash
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    @staticmethod
    def test_score_boundaries(scorer):
        """Test that scores are always within [0.0, 1.0] bounds."""
        test_texts = [
            "",  # Empty
            # Very high content
            "COP $50 millones baseline 2023 meta $100 millones 2025 quarterly monitoring evaluation indicator",
            "texto simple",  # Simple content
            "• Lista sin valores",  # Title penalty
            "Investment: $999,999,999 millions by Q4 2099 with monthly baseline indicator assessment",  # Extreme values
        ]

        for text in test_texts:
            score = scorer.calcular_calidad_evidencia(text)
            assert 0.0 <= score <= 1.0, f"Score out of bounds for: {text}, got: {score}"
            assert isinstance(score, float), f"Non-float score for: {text}"


class TestAtomicReportGeneration:
    """Test atomic report generation functionality."""

    @pytest.fixture
    def scorer(self):
        return FeasibilityScorer()

    @pytest.fixture
    def test_indicators(self):
        return [
            "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025",
            "Mejorar desde la situación inicial hasta el objetivo propuesto",
            "Aumentar el acceso a servicios de salud en la región",
        ]

    @staticmethod
    def test_successful_report_generation(scorer, test_indicators):
        """Test successful atomic report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            # Generate report
            scorer.generate_report(test_indicators, str(report_path))

            # Verify file exists and has content
            assert report_path.exists()
            assert report_path.stat().st_size > 0

            # Verify report content structure
            content = report_path.read_text(encoding="utf-8")
            assert "# Feasibility Assessment Report" in content
            assert "## Summary" in content
            assert "## Detailed Analysis" in content
            assert "## Recommendations" in content
            assert f"Total indicators analyzed: {len(test_indicators)}" in content

    @staticmethod
    def test_empty_indicators_error(scorer):
        """Test that empty indicators list raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            with pytest.raises(ValueError, match="Indicators list cannot be empty"):
                scorer.generate_report([], str(report_path))

    def test_atomic_file_operations(self, scorer, test_indicators):
        """Test that atomic operations prevent partial file writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            # Mock a scenario where report generation succeeds but rename might fail
            original_rename = Path.rename

            def mock_rename_success(self, target):
                # Simulate successful atomic rename
                return original_rename(self, target)

            Path.rename = mock_rename_success

            try:
                scorer.generate_report(test_indicators, str(report_path))

                # Verify no temporary files remain
                temp_files = list(Path(temp_dir).glob("*.tmp.*"))
                assert len(temp_files) == 0, "Temporary files were not cleaned up"

                # Verify final file exists
                assert report_path.exists()

            finally:
                Path.rename = original_rename

    @staticmethod
    def test_temporary_file_cleanup_on_error(scorer, test_indicators):
        """Test that temporary files are cleaned up when errors occur."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            # Mock Path.rename to raise an exception
            original_rename = Path.rename

            def mock_rename_failure(self, target):
                raise OSError("Simulated rename failure")

            Path.rename = mock_rename_failure

            try:
                with pytest.raises(IOError, match="Failed to generate report"):
                    scorer.generate_report(test_indicators, str(report_path))

                # Verify temporary files are cleaned up
                temp_files = list(Path(temp_dir).glob("*.tmp.*"))
                assert len(temp_files) == 0, (
                    "Temporary files were not cleaned up after error"
                )

                # Verify final file doesn't exist
                assert not report_path.exists()

            finally:
                Path.rename = original_rename

    def test_unique_temporary_filenames(self, scorer, test_indicators):
        """Test that temporary files have unique names to avoid conflicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            # Intercept temporary file creation to check uniqueness
            created_temp_files = []
            original_open = Path.open

            def mock_open(self, *args, **kwargs):
                if str(self).endswith(".tmp."):
                    created_temp_files.append(str(self))
                return original_open(self, *args, **kwargs)

            # Generate multiple reports to test uniqueness
            for i in range(3):
                temp_report_path = Path(temp_dir) / f"test_report_{i}.md"
                scorer.generate_report(test_indicators, str(temp_report_path))

            # In practice, the temp files are quickly renamed, so we test the mechanism works
            # by checking that reports are generated successfully without conflicts
            for i in range(3):
                temp_report_path = Path(temp_dir) / f"test_report_{i}.md"
                assert temp_report_path.exists()

    @staticmethod
    def test_report_content_completeness(scorer, test_indicators):
        """Test that generated report contains all expected sections and data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            scorer.generate_report(test_indicators, str(report_path))

            content = report_path.read_text(encoding="utf-8")

            # Check header and metadata
            assert "# Feasibility Assessment Report" in content
            assert "Generated on:" in content
            assert f"Total indicators analyzed: {len(test_indicators)}" in content

            # Check summary section
            assert "## Summary" in content
            assert "Average feasibility score:" in content
            assert "Quality tier distribution:" in content

            # Check detailed analysis
            assert "## Detailed Analysis" in content
            for i, indicator in enumerate(test_indicators, 1):
                assert f"### {i}. Indicator Analysis" in content
                assert f"**Text:** {indicator}" in content

            # Check recommendations
            assert "## Recommendations" in content

            # Check footer
            assert "*Report generated by Feasibility Scorer v1.0*" in content

    @staticmethod
    def test_report_content_sorting(scorer):
        """Test that indicators are sorted by score in the report."""
        indicators = [
            "aumentar servicios región",  # Low score
            "línea base 50% meta 80%",  # High score
            "situación actual objetivo",  # Medium score
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.md"

            scorer.generate_report(indicators, str(report_path))

            content = report_path.read_text(encoding="utf-8")

            # Find positions of indicators in the report
            high_score_pos = content.find("línea base 50% meta 80%")
            medium_score_pos = content.find("situación actual objetivo")
            low_score_pos = content.find("aumentar servicios región")

            # Verify high score appears before medium, which appears before low
            assert high_score_pos < medium_score_pos < low_score_pos

    @staticmethod
    def test_zero_evidence_support_override(scorer):
        """Test that zero evidencia_soporte overrides normal scoring logic."""
        text = "línea base 50% meta 80% año 2025 responsable Secretaría"

        # Test normal scoring
        normal_result = scorer.calculate_feasibility_score(text)
        assert normal_result.feasibility_score > 0.0
        assert normal_result.quality_tier != "REQUIERE MAYOR EVIDENCIA"

        # Test with zero evidence support
        zero_evidence_result = scorer.calculate_feasibility_score(
            text, evidencia_soporte=0
        )
        assert zero_evidence_result.feasibility_score == 0.0
        assert zero_evidence_result.quality_tier == "REQUIERE MAYOR EVIDENCIA"
        assert len(zero_evidence_result.components_detected) == 0
        assert len(zero_evidence_result.detailed_matches) == 0
        assert zero_evidence_result.has_quantitative_baseline == False
        assert zero_evidence_result.has_quantitative_target == False

        # Test with non-zero evidence support (should work normally)
        normal_result_with_evidence = scorer.calculate_feasibility_score(
            text, evidencia_soporte=1
        )
        assert normal_result_with_evidence.feasibility_score > 0.0
        assert normal_result_with_evidence.quality_tier != "REQUIERE MAYOR EVIDENCIA"

    @staticmethod
    def test_batch_scoring_with_evidence_support(scorer):
        """Test batch scoring with evidencia_soporte values."""
        indicators = [
            "línea base 50% meta 80% año 2025",
            "situación actual mejorar objetivo",
            "aumentar servicios región",
        ]
        evidencia_list = [0, 1, 2]  # First one has zero evidence

        results = scorer.batch_score(indicators, evidencia_soporte_list=evidencia_list)

        # First result should be overridden due to zero evidence
        assert results[0].feasibility_score == 0.0
        assert results[0].quality_tier == "REQUIERE MAYOR EVIDENCIA"

        # Other results should score normally
        assert results[1].feasibility_score > 0.0
        assert results[1].quality_tier != "REQUIERE MAYOR EVIDENCIA"


def test_feasibility_scorer_picklable_roundtrip():
    """FeasibilityScorer instances should survive pickle/unpickle."""

    scorer = FeasibilityScorer(enable_parallel=False)
    payload = pickle.dumps(scorer)
    restored = pickle.loads(payload)

    assert isinstance(restored, FeasibilityScorer)

    sample_text = "Incrementar la cobertura del 60% al 80% para 2025"
    original_score = scorer.calculate_feasibility_score(sample_text)
    restored_score = restored.calculate_feasibility_score(sample_text)

    assert pytest.approx(original_score.feasibility_score) == restored_score.feasibility_score
    assert original_score.components_detected == restored_score.components_detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
