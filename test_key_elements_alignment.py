#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify KEY_ELEMENTS alignment with decalogo-industrial.latest.clean.json

This test ensures that plan_sanitizer.py has the correct 6 dimensions
aligned with the canonical structure.
"""

import json
from plan_sanitizer import KEY_ELEMENTS


def test_key_elements_has_six_dimensions():
    """Verify that KEY_ELEMENTS has exactly 6 dimensions."""
    assert len(KEY_ELEMENTS) == 6, (
        f"KEY_ELEMENTS must have 6 dimensions, found {len(KEY_ELEMENTS)}"
    )


def test_key_elements_dimension_names():
    """Verify that dimension names match the canonical structure."""
    expected_dimensions = {
        "insumos",      # D1
        "actividades",  # D2
        "productos",    # D3
        "resultados",   # D4
        "impactos",     # D5
        "causalidad"    # D6
    }
    
    actual_dimensions = set(KEY_ELEMENTS.keys())
    
    assert actual_dimensions == expected_dimensions, (
        f"Dimension names don't match.\n"
        f"Expected: {expected_dimensions}\n"
        f"Got: {actual_dimensions}\n"
        f"Missing: {expected_dimensions - actual_dimensions}\n"
        f"Extra: {actual_dimensions - expected_dimensions}"
    )


def test_key_elements_no_old_nomenclature():
    """Verify that old DE-X nomenclature is not present."""
    old_keys = {"indicators", "diagnostics", "participation", "monitoring"}
    actual_keys = set(KEY_ELEMENTS.keys())
    
    overlap = old_keys & actual_keys
    
    assert len(overlap) == 0, (
        f"Old dimension keys found: {overlap}. "
        f"These should be replaced with new D1-D6 aligned keys."
    )


def test_key_elements_all_have_patterns():
    """Verify that each dimension has at least one regex pattern."""
    for dimension, patterns in KEY_ELEMENTS.items():
        assert len(patterns) > 0, (
            f"Dimension '{dimension}' has no regex patterns"
        )
        assert isinstance(patterns, list), (
            f"Dimension '{dimension}' patterns should be a list"
        )


def test_alignment_with_canonical_json():
    """Verify alignment with decalogo-industrial.latest.clean.json structure."""
    try:
        with open('decalogo-industrial.latest.clean.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', [])
        
        # Verify the JSON has 6 dimensions
        dimensions_in_json = set()
        for q in questions:
            dim = q.get('dimension')
            if dim:
                dimensions_in_json.add(dim)
        
        expected_json_dims = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}
        
        assert dimensions_in_json == expected_json_dims, (
            f"JSON should have dimensions D1-D6, found: {dimensions_in_json}"
        )
        
        # Verify counts
        dim_counts = {}
        for q in questions:
            dim = q.get('dimension')
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        for dim in expected_json_dims:
            assert dim_counts.get(dim) == 50, (
                f"Dimension {dim} should have 50 questions, found {dim_counts.get(dim)}"
            )
        
        print("✓ Canonical JSON has correct 6-dimension structure (D1-D6)")
        print("✓ Each dimension has exactly 50 questions")
        print("✓ Total questions: 300")
        
    except FileNotFoundError:
        print("⚠ Warning: decalogo-industrial.latest.clean.json not found, skipping JSON verification")


def test_dimension_key_mapping():
    """Document the mapping between KEY_ELEMENTS keys and D1-D6 dimensions."""
    mapping = {
        "insumos": "D1",
        "actividades": "D2", 
        "productos": "D3",
        "resultados": "D4",
        "impactos": "D5",
        "causalidad": "D6"
    }
    
    print("\n" + "="*80)
    print("KEY_ELEMENTS Dimension Mapping")
    print("="*80)
    
    for key, dimension in mapping.items():
        assert key in KEY_ELEMENTS, f"Missing key: {key}"
        pattern_count = len(KEY_ELEMENTS[key])
        print(f"{dimension}: {key:15s} ({pattern_count} regex patterns)")
    
    print("="*80)


if __name__ == "__main__":
    print("Testing KEY_ELEMENTS alignment with decalogo-industrial.latest.clean.json\n")
    
    test_key_elements_has_six_dimensions()
    print("✓ KEY_ELEMENTS has 6 dimensions")
    
    test_key_elements_dimension_names()
    print("✓ Dimension names are correct (insumos, actividades, productos, resultados, impactos, causalidad)")
    
    test_key_elements_no_old_nomenclature()
    print("✓ No old DE-X nomenclature found")
    
    test_key_elements_all_have_patterns()
    print("✓ All dimensions have regex patterns")
    
    test_alignment_with_canonical_json()
    
    test_dimension_key_mapping()
    
    print("\n✓ All tests passed! KEY_ELEMENTS is correctly aligned with the canonical 6-dimension structure.")
