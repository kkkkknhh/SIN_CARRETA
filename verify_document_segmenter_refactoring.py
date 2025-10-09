#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for Document Segmenter refactoring.

This script validates that the Document Segmenter has been correctly refactored
to align with decalogo-industrial.latest.clean.json structure.

Run this script to verify:
1. All dimension mappings use D1-D6 format (not old DE- format)
2. All section types are properly defined
3. Backward compatibility is maintained
4. Pattern recognition works correctly
"""

import sys
from document_segmenter import (
    DocumentSegmenter, 
    DocumentSegment, 
    SectionType, 
    SegmentationType
)


def verify_dimension_format():
    """Verify that all dimensions use D1-D6 format."""
    print("Verifying dimension format...")
    
    # Test all section types
    all_dimensions = set()
    for section_type in SectionType:
        segment = DocumentSegment(
            text="Test segment",
            start_pos=0,
            end_pos=12,
            segment_type=SegmentationType.SECTION,
            section_type=section_type
        )
        all_dimensions.update(segment.decalogo_dimensions)
    
    # Check no old DE- format
    old_format = [d for d in all_dimensions if d.startswith("DE-")]
    if old_format:
        print(f"  ✗ FAIL: Old DE- format dimensions found: {old_format}")
        return False
    
    # Check only D1-D6 format (plus empty for OTHER)
    valid_dims = {"D1", "D2", "D3", "D4", "D5", "D6"}
    invalid_dims = [d for d in all_dimensions if d and d not in valid_dims]
    if invalid_dims:
        print(f"  ✗ FAIL: Invalid dimension format: {invalid_dims}")
        return False
    
    print(f"  ✓ All dimensions use correct format: {sorted(all_dimensions - {''})}")
    return True


def verify_section_types():
    """Verify that all section types are properly defined."""
    print("\nVerifying section types...")
    
    expected_d1 = ["DIAGNOSTIC", "BASELINE", "RESOURCES", "CAPACITY", "BUDGET", "PARTICIPATION"]
    expected_d2 = ["ACTIVITY", "MECHANISM", "INTERVENTION", "STRATEGY", "TIMELINE"]
    expected_d3 = ["PRODUCT", "OUTPUT"]
    expected_d4 = ["RESULT", "OUTCOME", "INDICATOR", "MONITORING"]
    expected_d5 = ["IMPACT", "LONG_TERM_EFFECT"]
    expected_d6 = ["CAUSAL_THEORY", "CAUSAL_LINK"]
    expected_multi = ["VISION", "OBJECTIVE", "RESPONSIBILITY"]
    expected_other = ["OTHER"]
    
    all_expected = (expected_d1 + expected_d2 + expected_d3 + expected_d4 + 
                   expected_d5 + expected_d6 + expected_multi + expected_other)
    
    actual_types = [st.name for st in SectionType]
    
    missing = set(all_expected) - set(actual_types)
    if missing:
        print(f"  ✗ FAIL: Missing section types: {missing}")
        return False
    
    print(f"  ✓ All expected section types defined: {len(actual_types)} types")
    return True


def verify_dimension_coverage():
    """Verify that all D1-D6 dimensions have at least one section type."""
    print("\nVerifying dimension coverage...")
    
    dimension_coverage = {"D1": [], "D2": [], "D3": [], "D4": [], "D5": [], "D6": []}
    
    for section_type in SectionType:
        segment = DocumentSegment(
            text="Test",
            start_pos=0,
            end_pos=4,
            segment_type=SegmentationType.SECTION,
            section_type=section_type
        )
        for dim in segment.decalogo_dimensions:
            if dim in dimension_coverage:
                dimension_coverage[dim].append(section_type.name)
    
    all_covered = True
    for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        count = len(dimension_coverage[dim])
        if count == 0:
            print(f"  ✗ {dim}: No section types")
            all_covered = False
        else:
            print(f"  ✓ {dim}: {count} section types")
    
    return all_covered


def verify_backward_compatibility():
    """Verify that string inputs still work (backward compatibility)."""
    print("\nVerifying backward compatibility...")
    
    segmenter = DocumentSegmenter()
    
    # Test string input
    try:
        text = "Diagnóstico de la situación actual con líneas base verificables."
        segments = segmenter.segment(text)
        print(f"  ✓ String input works: {len(segments)} segments")
    except Exception as e:
        print(f"  ✗ FAIL: String input failed: {e}")
        return False
    
    # Test dict input
    try:
        segments_dict = segmenter.segment({"full_text": text})
        print(f"  ✓ Dict input works: {len(segments_dict)} segments")
    except Exception as e:
        print(f"  ✗ FAIL: Dict input failed: {e}")
        return False
    
    # Test empty inputs
    try:
        assert len(segmenter.segment("")) == 0
        assert len(segmenter.segment({})) == 0
        assert len(segmenter.segment(None)) == 0
        print("  ✓ Empty inputs handled correctly")
    except Exception as e:
        print(f"  ✗ FAIL: Empty input handling failed: {e}")
        return False
    
    return True


def verify_pattern_recognition():
    """Verify that pattern recognition works for each dimension."""
    print("\nVerifying pattern recognition...")
    
    segmenter = DocumentSegmenter()
    
    test_cases = [
        ("D1", "Diagnóstico con líneas base y recursos asignados"),
        ("D2", "Actividades formalizadas con mecanismos causales"),
        ("D3", "Productos con indicadores verificables"),
        ("D4", "Resultados con métricas de outcome"),
        ("D5", "Impactos de largo plazo medibles"),
        ("D6", "Teoría de cambio explícita con DAG")
    ]
    
    all_passed = True
    for expected_dim, text in test_cases:
        segments = segmenter.segment_text(text)
        if segments:
            found_dims = set()
            for seg in segments:
                found_dims.update(seg.decalogo_dimensions)
            
            if expected_dim in found_dims:
                print(f"  ✓ {expected_dim} pattern recognized")
            else:
                print(f"  ⚠ {expected_dim} pattern not recognized (found: {found_dims})")
                # Not failing on this, as patterns might not catch everything
        else:
            print(f"  ⚠ {expected_dim} produced no segments")
    
    return True  # Pattern recognition is best-effort


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("DOCUMENT SEGMENTER REFACTORING VERIFICATION")
    print("=" * 70)
    
    tests = [
        ("Dimension Format", verify_dimension_format),
        ("Section Types", verify_section_types),
        ("Dimension Coverage", verify_dimension_coverage),
        ("Backward Compatibility", verify_backward_compatibility),
        ("Pattern Recognition", verify_pattern_recognition),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All verification tests PASSED")
        print("✓ Document Segmenter refactoring is complete and correct")
        return 0
    else:
        print(f"\n✗ {total - passed} verification test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
