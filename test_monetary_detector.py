import unittest

from monetary_detector import MonetaryType, create_monetary_detector


class TestMonetaryDetector(unittest.TestCase):
    """Comprehensive test suite for MonetaryDetector."""

    def setUp(self):
        """Setup detector instance for each test."""
        self.detector = create_monetary_detector()

    def test_number_parsing_spanish_conventions(self):
        """Test number parsing with Spanish decimal conventions."""
        # Decimal comma
        assert self.detector._parse_number("1,5") == 1.5
        assert self.detector._parse_number("123,45") == 123.45

        # Thousands separator (period)
        assert self.detector._parse_number("1.234") == 1234.0
        assert self.detector._parse_number("1.234.567") == 1234567.0

        # Mixed: thousands separator + decimal comma
        assert self.detector._parse_number("1.234.567,89") == 1234567.89
        assert self.detector._parse_number("12.345,6") == 12345.6

        # English-style (period as decimal)
        assert self.detector._parse_number("123.45") == 123.45

        # No separators
        assert self.detector._parse_number("1234") == 1234.0

    def test_scale_multipliers(self):
        """Test scale multiplier application."""
        # Spanish terms
        assert self.detector._apply_scale(1.0, "mil") == 1000.0
        assert self.detector._apply_scale(1.0, "millón") == 1000000.0
        assert self.detector._apply_scale(1.0, "millones") == 1000000.0
        assert self.detector._apply_scale(2.5, "millones") == 2500000.0

        # Abbreviations
        assert self.detector._apply_scale(1.0, "K") == 1000.0
        assert self.detector._apply_scale(1.0, "M") == 1000000.0
        assert self.detector._apply_scale(1.0, "MM") == 1000000.0
        assert self.detector._apply_scale(1.0, "B") == 1000000000.0

        # No scale
        assert self.detector._apply_scale(100.0, None) == 100.0

    def test_currency_extraction(self):
        """Test currency extraction from pre/post positions."""
        assert self.detector._extract_currency("$", None) == "USD"
        assert self.detector._extract_currency(None, "COP") == "COP"
        assert (
            self.detector._extract_currency("€", "USD") == "EUR"
        )  # Pre takes precedence
        assert self.detector._extract_currency(None, None) is None

    def test_basic_monetary_detection(self):
        """Test basic monetary amount detection."""
        text = "$100 y €200"
        results = self.detector.detect_monetary_expressions(text)

        assert len(results) == 2

        # First match: $100
        assert results[0].value == 100.0
        assert results[0].type == MonetaryType.CURRENCY
        assert results[0].currency == "USD"

        # Second match: €200
        assert results[1].value == 200.0
        assert results[1].type == MonetaryType.CURRENCY
        assert results[1].currency == "EUR"

    def test_monetary_with_scales(self):
        """Test monetary amounts with scale indicators."""
        test_cases = [
            ("$1.2 millones", 1200000.0, "USD"),
            ("€2,5M", 2500000.0, "EUR"),
            ("1000K COP", 1000000.0, "COP"),
            ("USD 3.5 miles", 3500.0, "USD"),
        ]

        for text, expected_value, expected_currency in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value
            assert results[0].currency == expected_currency

    def test_percentage_detection(self):
        """Test percentage detection and normalization."""
        test_cases = [
            ("50%", 0.5),
            ("12,5%", 0.125),
            ("100%", 1.0),
            ("0,75%", 0.0075),
            ("150%", 1.5),
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value
            assert results[0].type == MonetaryType.PERCENTAGE
            assert results[0].currency is None

    def test_numeric_with_scales(self):
        """Test pure numeric values with scales."""
        test_cases = [
            ("1.5 millones", 1500000.0),
            ("250K", 250000.0),
            ("3,2M", 3200000.0),
            ("500 mil", 500000.0),
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value
            assert results[0].type == MonetaryType.NUMERIC
            assert results[0].currency is None

    def test_complex_text_mixed_expressions(self):
        """Test detection in complex text with mixed expressions."""
        text = (
            "El presupuesto es de $2.5 millones COP, con un aumento del 15% "
            "respecto a los 1,8M del año anterior. Los gastos operativos "
            "representan €750K aproximadamente."
        )

        results = self.detector.detect_monetary_expressions(text)

        assert len(results) == 4

        # $2.5 millones COP - should detect as one expression with scale
        assert results[0].value == 2500000.0
        assert results[0].currency == "USD"  # Pre-currency takes precedence

        # 15%
        assert results[1].value == 0.15
        assert results[1].type == MonetaryType.PERCENTAGE

        # 1,8M - detected as numeric with scale (no currency symbol)
        assert results[2].value == 1800000.0
        assert results[2].type == MonetaryType.NUMERIC

        # €750K
        assert results[3].value == 750000.0
        assert results[3].currency == "EUR"

    def test_edge_cases_decimal_separators(self):
        """Test edge cases with different decimal separator combinations."""
        test_cases = [
            # Spanish format: thousands=period, decimal=comma
            ("$1.234.567,89", 1234567.89),
            ("€12.345,60", 12345.60),
            # English format: thousands=comma, decimal=period
            ("$1,234,567.89", 1234567.89),
            ("£12,345.60", 12345.60),
            # Ambiguous single separator cases
            ("$1.234", 1234.0),  # Treated as thousands separator
            ("$12.34", 12.34),  # Treated as decimal separator
            ("$1,234", 1234.0),  # Treated as thousands separator
            ("$12,34", 12.34),  # Treated as decimal separator
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value

    def test_mixed_currencies_same_expression(self):
        """Test expressions with mixed currency indicators."""
        text = "$2.5 COP millones"
        results = self.detector.detect_monetary_expressions(text)

        assert len(results) == 1
        # No scale applied since regex stops at COP
        assert results[0].value == 2.5
        assert results[0].currency == "USD"  # Pre-currency takes precedence

    def test_abbreviation_conventions(self):
        """Test handling of ambiguous abbreviations M vs MM."""
        test_cases = [
            ("$1M", 1000000.0),  # M = million
            ("$1MM", 1000000.0),  # MM = million (same as M)
            ("€2.5M", 2500000.0),
            ("£3,2MM", 3200000.0),
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value

    def test_overlapping_matches_avoided(self):
        """Test that overlapping matches are properly handled."""
        # This could potentially match both as currency and numeric
        text = "$100M"
        results = self.detector.detect_monetary_expressions(text)

        # Should only match once as currency, not also as numeric
        assert len(results) == 1
        assert results[0].value == 100000000.0
        assert results[0].type == MonetaryType.CURRENCY

    def test_position_tracking(self):
        """Test that match positions are correctly tracked."""
        text = "Inicio $100 medio €200 final"
        results = self.detector.detect_monetary_expressions(text)

        assert len(results) == 2

        # Check positions are reasonable
        assert results[0].start_pos < results[0].end_pos
        assert results[1].start_pos < results[1].end_pos
        assert results[0].end_pos <= results[1].start_pos

        # Check original match text
        assert "$100" in results[0].original_match
        assert "€200" in results[1].original_match

    def test_normalize_monetary_expression_single(self):
        """Test single expression normalization convenience method."""
        assert self.detector.normalize_monetary_expression(
            "$1.5M") == 1500000.0
        assert self.detector.normalize_monetary_expression("25%") == 0.25
        assert self.detector.normalize_monetary_expression("500K") == 500000.0
        assert self.detector.normalize_monetary_expression("invalid") is None
        assert self.detector.normalize_monetary_expression("") is None

    def test_unicode_normalization(self):
        """Test Unicode text normalization."""
        # Test with various Unicode characters
        text_with_unicode = "$1.5\u00a0millones"  # Non-breaking space
        results = self.detector.detect_monetary_expressions(text_with_unicode)

        assert len(results) == 1
        assert results[0].value == 1500000.0

    def test_case_insensitive_matching(self):
        """Test case insensitive pattern matching."""
        test_cases = [
            ("$1.5 MILLONES", 1500000.0),
            ("€2.3 millones", 2300000.0),
            ("1K USD", 1000.0),
            ("2m cop", 2000000.0),
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value

    def test_whitespace_handling(self):
        """Test proper whitespace handling in patterns."""
        test_cases = [
            ("$ 100", 100.0),  # Space after currency
            ("100 $", 100.0),  # Space before currency
            ("1.5  millones", 1500000.0),  # Multiple spaces
            ("25 %", 0.25),  # Space before percentage
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert results[0].value == expected_value

    def test_no_false_positives_pure_numbers(self):
        """Test that pure numbers without currency/scale/% are not detected as monetary."""
        text = "El año 2023 tuvo 365 días y la temperatura fue de 25 grados."
        results = self.detector.detect_monetary_expressions(text)

        # Should not detect any monetary expressions
        assert len(results) == 0

    def test_currency_symbols_mapping(self):
        """Test various currency symbols are properly mapped."""
        test_cases = [
            ("¥1000", "JPY"),
            ("£500", "GBP"),
            ("250 CHF", "CHF"),
            ("1000 ARS", "ARS"),
            ("BRL 500", "BRL"),  # Pre-currency position works better
        ]

        for text, expected_currency in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1, (
                f"Expected 1 result for '{text}', got {len(results)}"
            )
            assert results[0].currency == expected_currency

    def test_large_numbers_with_multiple_scale_indicators(self):
        """Test handling of very large numbers with multiple scale terms."""
        # Note: This tests the regex doesn't match invalid combinations
        text = "$1 millones de miles"  # Should only match the first valid part
        results = self.detector.detect_monetary_expressions(text)

        assert len(results) == 1
        assert results[0].value == 1000000.0  # Only processes first scale

    def test_decimal_precision_preservation(self):
        """Test that decimal precision is preserved in calculations."""
        test_cases = [
            ("$1,23 millones", 1230000.0),
            ("€2,456 miles", 2456000.0),  # "miles" means thousands in Spanish
            ("15,75%", 0.1575),
            ("$0,5M", 500000.0),  # Added currency symbol
        ]

        for text, expected_value in test_cases:
            results = self.detector.detect_monetary_expressions(text)
            assert len(results) == 1
            assert (
                abs(results[0].value - expected_value) < 0.001
            )  # Float precision check


if __name__ == "__main__":
    unittest.main(verbosity=2)
