#!/usr/bin/env python3
"""
Test script specifically for Unicode normalization functionality.
"""

import unicodedata

from feasibility_scorer import FeasibilityScorer


def test_unicode_normalization():
    """Test Unicode normalization functionality and its impact on pattern matching."""
    scorer = FeasibilityScorer()

    # Test cases with various Unicode characters that should normalize
    test_cases = [
        {
            # Full-width percent, smart quotes
            "original": 'Incrementar la línea base de 65％ a una "meta" de 85％',
            "description": "Full-width characters and smart quotes",
        },
        {
            "original": "Alcanzar objetivo de 1‚500 millones",  # Different comma character
            "description": "Different comma character",
        },
        {
            "original": "Meta de año ２０２５",  # Full-width numbers
            "description": "Full-width numbers",
        },
        {
            "original": 'baseline "50%" target "80%"',  # Curly quotes
            "description": "Curly quotation marks",
        },
    ]

    print("UNICODE NORMALIZATION TESTS")
    print("=" * 50)

    all_passed = True

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {case['description']}")
        print(f"   Original: {repr(case['original'])}")

        # Test normalization function directly
        normalized = scorer._normalize_text(case["original"])
        expected = unicodedata.normalize("NFKC", case["original"])

        if normalized == expected:
            print("   ✓ Normalization function works correctly")
        else:
            print("   ✗ Normalization function failed")
            all_passed = False

        print(f"   Normalized: {repr(normalized)}")

        # Test that both original and normalized text produce similar component detection
        original_components = scorer.detect_components(case["original"])
        normalized_components = scorer.detect_components(normalized)

        original_count = len(original_components)
        normalized_count = len(normalized_components)

        print(f"   Components (original): {original_count}")
        print(f"   Components (normalized): {normalized_count}")

        if original_count == normalized_count:
            print("   ✓ Component detection consistent after normalization")
        else:
            print("   ✗ Component detection differs after normalization")
            all_passed = False

    # Test pattern matching improvement
    print(f"\n{'=' * 50}")
    print("PATTERN MATCHING IMPROVEMENT TESTS")
    print("=" * 50)

    unicode_variants = [
        # Different quote styles
        ('baseline "50%" target "80%"', 'baseline "50%" target "80%"'),
        # Different dash/hyphen characters
        ("timeline 2020-2025", "timeline 2020—2025"),  # em dash vs hyphen
        # Full-width vs half-width characters
        ("meta 85％", "meta 85%"),
        ("año ２０２５", "año 2025"),
    ]

    match_improvements = 0
    total_tests = len(unicode_variants)

    for j, (normalized_text, variant_text) in enumerate(unicode_variants, 1):
        print(f"\n{j}. Testing variant matching:")
        print(f"   Standard: {repr(normalized_text)}")
        print(f"   Variant:  {repr(variant_text)}")

        # Score both versions
        normalized_score = scorer.calculate_feasibility_score(normalized_text)
        variant_score = scorer.calculate_feasibility_score(variant_text)

        # Count components detected
        normalized_components = len(normalized_score.components_detected)
        variant_components = len(variant_score.components_detected)

        print(f"   Standard components: {normalized_components}")
        print(f"   Variant components: {variant_components}")

        # After normalization, should detect same or more components
        if variant_components >= normalized_components:
            match_improvements += 1
            print("   ✓ Variant matching maintained or improved")
        else:
            print("   ✗ Variant matching degraded")

    improvement_rate = match_improvements / total_tests
    print(f"\nImprovement rate: {improvement_rate:.1%}")

    if improvement_rate >= 0.7:
        print("✓ Unicode normalization provides adequate improvement")
    else:
        print("✗ Unicode normalization improvement rate too low")
        all_passed = False

    print(f"\n{'=' * 50}")
    if all_passed:
        print("🎉 All Unicode normalization tests PASSED!")
        return True
    else:
        print("❌ Some Unicode normalization tests FAILED!")
        return False


if __name__ == "__main__":
    success = test_unicode_normalization()
    exit(0 if success else 1)
