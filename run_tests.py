#!/usr/bin/env python3
"""
Simple test runner that doesn't require pytest.
Tests the feasibility scorer functionality.
"""

import sys

from feasibility_scorer import ComponentType, FeasibilityScorer


class TestRunner:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0

    @staticmethod
    def assert_equal(actual, expected, message=""):
        if actual != expected:
            raise AssertionError(
                f"Expected {expected}, got {actual}. {message}")

    @staticmethod
    def assert_true(condition, message=""):
        if not condition:
            raise AssertionError(f"Condition was False. {message}")

    @staticmethod
    def assert_false(condition, message=""):
        if condition:
            raise AssertionError(f"Condition was True. {message}")

    @staticmethod
    def assert_in(item, container, message=""):
        if item not in container:
            raise AssertionError(f"{item} not in {container}. {message}")

    def run_test(self, test_func, test_name):
        self.tests_run += 1
        try:
            test_func()
            self.tests_passed += 1
            print(f"✓ {test_name}")
        except Exception as e:
            self.tests_failed += 1
            print(f"✗ {test_name}: {str(e)}")

    def summary(self):
        print("\nTest Summary:")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        return self.tests_failed == 0


def test_high_quality_indicators():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    # High quality indicator with all components
    text = "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025"
    result = scorer.calculate_feasibility_score(text)

    runner.assert_true(
        result.feasibility_score >= 0.8, "High quality score should be >= 0.8"
    )
    runner.assert_equal(result.quality_tier, "high",
                        "Should be high quality tier")
    runner.assert_true(
        result.has_quantitative_baseline, "Should detect quantitative baseline"
    )
    runner.assert_true(
        result.has_quantitative_target, "Should detect quantitative target"
    )
    runner.assert_in(
        ComponentType.BASELINE, result.components_detected, "Should detect baseline"
    )
    runner.assert_in(
        ComponentType.TARGET, result.components_detected, "Should detect target"
    )


def test_mandatory_requirements():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    # Only baseline - should get 0 score
    result = scorer.calculate_feasibility_score(
        "La línea base es de 50% de cobertura")
    runner.assert_equal(
        result.feasibility_score, 0.0, "Missing target should result in 0 score"
    )

    # Only target - should get 0 score
    result = scorer.calculate_feasibility_score("El objetivo es llegar al 80%")
    runner.assert_equal(
        result.feasibility_score, 0.0, "Missing baseline should result in 0 score"
    )

    # Both present - should get positive score
    result = scorer.calculate_feasibility_score(
        "Partir de línea base 50% hasta objetivo 80%"
    )
    runner.assert_true(
        result.feasibility_score > 0.0,
        "Both baseline and target should give positive score",
    )


def test_spanish_patterns():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    # Test with simpler, clearer patterns
    spanish_texts = [
        "línea base de 30% hasta meta de 60%",
        "valor inicial 25 millones para objetivo 40 millones",
    ]

    for text in spanish_texts:
        result = scorer.calculate_feasibility_score(text)
        runner.assert_true(
            result.feasibility_score > 0.0,
            f"Failed to detect Spanish patterns in: {text}",
        )


def test_english_patterns():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    english_texts = [
        "baseline of 30% to target of 60%",
        "current level 25 million to goal 40 million",
    ]

    for text in english_texts:
        result = scorer.calculate_feasibility_score(text)
        runner.assert_true(
            result.feasibility_score > 0.0,
            f"Failed to detect English patterns in: {text}",
        )


def test_numerical_detection():
    scorer = FeasibilityScorer()
    runner = TestRunner()

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
        runner.assert_true(
            numerical_detected, f"Failed to detect numerical pattern in: {text}"
        )


def test_date_detection():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    date_texts = [
        "para el año 2025",
        "en diciembre 2024",
        "by January 2025",
        "15/12/2024",
        "hasta 2026",
    ]

    for text in date_texts:
        components = scorer.detect_components(text)
        date_detected = any(c.component_type ==
                            ComponentType.DATE for c in components)
        runner.assert_true(
            date_detected, f"Failed to detect date pattern in: {text}")


def test_insufficient_indicators():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    insufficient_texts = [
        "Aumentar el acceso a servicios de salud en la región",
        "Mejorar la calidad educativa mediante nuevas estrategias",
        "La meta es fortalecer las instituciones públicas",
    ]

    for text in insufficient_texts:
        result = scorer.calculate_feasibility_score(text)
        runner.assert_equal(
            result.feasibility_score,
            0.0,
            f"Expected 0.0 score for insufficient indicator: {text}",
        )
        runner.assert_equal(
            result.quality_tier, "insufficient", "Should be insufficient quality tier"
        )


def test_batch_scoring():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    indicators = [
        "línea base 50% meta 80% año 2025",  # High quality
        "línea base actual objetivo definido",  # Medium quality
        "aumentar servicios región",  # Insufficient
    ]

    # Test both parallel and sequential modes
    results_sequential = scorer.batch_score(indicators, compare_backends=False)
    runner.assert_equal(len(results_sequential), 3, "Should return 3 results")

    # Test with parallel processing disabled
    scorer_seq = FeasibilityScorer(enable_parallel=False)
    results_disabled = scorer_seq.batch_score(indicators)
    runner.assert_equal(len(results_disabled), 3,
                        "Should work with parallel disabled")

    # First should score higher than others
    runner.assert_true(
        results_sequential[0].feasibility_score
        > results_sequential[1].feasibility_score,
        "First should score higher than second",
    )

    # Third should be 0 (insufficient)
    runner.assert_equal(
        results_sequential[2].feasibility_score, 0.0, "Third should be insufficient"
    )


def test_batch_scoring_with_monitoring():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    indicators = [
        "línea base 50% meta 80% año 2025",
        "baseline 30% target 60% by 2024",
        "aumentar servicios región",
    ]

    result = scorer.batch_score_with_monitoring(indicators)

    # Check result structure
    runner.assert_equal(len(result.scores), 3, "Should return 3 scores")
    runner.assert_equal(result.total_indicators, 3,
                        "Should count 3 indicators")
    runner.assert_true(result.duracion_segundos >= 0,
                       "Duration should be non-negative")
    runner.assert_true(result.planes_por_minuto >= 0,
                       "Rate should be non-negative")
    runner.assert_true(
        isinstance(result.execution_time,
                   str), "Execution time should be string"
    )

    # Check that scores match regular batch_score
    regular_results = scorer.batch_score(indicators)
    for i, (monitoring_score, regular_score) in enumerate(
        zip(result.scores, regular_results)
    ):
        runner.assert_equal(
            monitoring_score.feasibility_score,
            regular_score.feasibility_score,
            f"Score {i} should match between methods",
        )

    # Performance validation - should process at reasonable rate
    if result.duracion_segundos > 0:
        runner.assert_true(
            result.planes_por_minuto > 0, "Should have positive processing rate"
        )


def test_quantitative_components():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    # Test with very clear separation - quantitative baseline only
    text1 = "La línea base muestra 65% de cobertura actual. Por separado, la meta es general y cualitativa sin números"
    result1 = scorer.calculate_feasibility_score(text1)
    runner.assert_true(
        result1.has_quantitative_baseline, "Should detect quantitative baseline"
    )
    runner.assert_false(
        result1.has_quantitative_target, "Should not detect quantitative target"
    )

    # Both quantitative
    text3 = "línea base 40% hasta meta 70%"
    result3 = scorer.calculate_feasibility_score(text3)
    runner.assert_true(
        result3.has_quantitative_baseline, "Should detect quantitative baseline"
    )
    runner.assert_true(
        result3.has_quantitative_target, "Should detect quantitative target"
    )


def test_documentation():
    scorer = FeasibilityScorer()
    runner = TestRunner()

    docs = scorer.get_detection_rules_documentation()
    runner.assert_true(
        "Feasibility Scorer Detection Rules Documentation" in docs,
        "Should contain title",
    )
    runner.assert_true(
        "Spanish Pattern Recognition" in docs, "Should contain Spanish patterns section"
    )
    runner.assert_true("Quality Tiers" in docs,
                       "Should contain quality tiers section")
    runner.assert_true(
        len(docs) > 1000, "Documentation should be comprehensive")


def main():
    print("Running Feasibility Scorer Tests")
    print("=" * 40)

    runner = TestRunner()

    # Run all tests
    runner.run_test(test_high_quality_indicators, "High Quality Indicators")
    runner.run_test(test_mandatory_requirements, "Mandatory Requirements")
    runner.run_test(test_spanish_patterns, "Spanish Patterns")
    runner.run_test(test_english_patterns, "English Patterns")
    runner.run_test(test_numerical_detection, "Numerical Detection")
    runner.run_test(test_date_detection, "Date Detection")
    runner.run_test(test_insufficient_indicators, "Insufficient Indicators")
    runner.run_test(test_batch_scoring, "Batch Scoring")
    runner.run_test(test_batch_scoring_with_monitoring,
                    "Batch Scoring with Monitoring")
    runner.run_test(test_quantitative_components, "Quantitative Components")
    runner.run_test(test_documentation, "Documentation")

    success = runner.summary()

    if success:
        print("\n🎉 All tests passed!")

        # Run a quick demo
        print("\n" + "=" * 40)
        print("QUICK DEMO")
        print("=" * 40)

        scorer = FeasibilityScorer()
        demo_indicators = [
            "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025",
            "Mejorar desde la situación actual hasta el objetivo propuesto",
            "Aumentar el acceso a servicios de salud",
        ]

        for i, indicator in enumerate(demo_indicators, 1):
            result = scorer.calculate_feasibility_score(indicator)
            print(f'\n{i}. "{indicator[:60]}..."')
            print(
                f"   Score: {result.feasibility_score:.2f} | Tier: {result.quality_tier}"
            )
            print(
                f"   Components: {[c.value for c in result.components_detected]}")

        return 0
    else:
        print("\n❌ Some tests failed!")
        print("\nShowing detailed scores for debugging:")

        # Debug the failing tests
        scorer = FeasibilityScorer()

        debug_cases = [
            (
                "Spanish patterns",
                "desde situación actual hasta alcanzar propósito establecido",
            ),
            ("Batch test", "línea base actual objetivo definido"),
            (
                "Quantitative",
                "línea base de 65%. Separado por distancia. meta general establecida",
            ),
        ]

        for name, text in debug_cases:
            result = scorer.calculate_feasibility_score(text)
            print(f'\n{name}: "{text}"')
            print(f"  Score: {result.feasibility_score}")
            print(
                f"  Components detected: {[c.value for c in result.components_detected]}"
            )
            print("  Detailed matches:")
            for match in result.detailed_matches:
                print(
                    f"    {match.component_type.value}: '{match.matched_text}' (conf: {match.confidence:.2f})"
                )

        return 1


if __name__ == "__main__":
    sys.exit(main())
