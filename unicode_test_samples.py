#!/usr/bin/env python3
"""
Test script to demonstrate Unicode normalization effects on text samples.
"""

import unicodedata
import sys
sys.path.append('.')
from feasibility_scorer import FeasibilityScorer

def test_unicode_normalization():
    """Test Unicode normalization on various text samples."""
    scorer = FeasibilityScorer()
    
    # Test samples with various Unicode characters
    test_samples = [
        {
            'text': 'Incrementar la línea base de 65％ a una "meta" de 85％',  # Full-width percent, smart quotes
            'description': 'Full-width percent signs and smart quotes'
        },
        {
            'text': 'Alcanzar objetivo de 1‚500 millones en año ２０２５',        # Different comma, full-width numbers  
            'description': 'Different comma character and full-width numbers'
        },
        {
            'text': 'baseline "50%" target "80%" by 2024',                    # Curly quotes
            'description': 'Curly quotation marks'
        },
        {
            'text': 'Horizonte 2020—2025 para meta',                         # Em dash
            'description': 'Em dash instead of hyphen'
        },
        {
            'text': 'Coeficiente línea base',                                # fi ligature (if present)
            'description': 'Potential ligature characters'
        }
    ]
    
    print('UNICODE NORMALIZATION COMPARISON')
    print('=' * 60)
    
    total_improvements = 0
    
    for i, sample in enumerate(test_samples, 1):
        text = sample['text']
        description = sample['description']
        
        print(f'\n{i}. Test: {description}')
        print(f'   Original:   {repr(text)}')
        
        # Normalize the text
        normalized = unicodedata.normalize('NFKC', text)
        print(f'   Normalized: {repr(normalized)}')
        print(f'   Changed: {text != normalized}')
        
        # Test regex matching before and after normalization
        print('\n   Pattern Matching Comparison:')
        
        # Score original text (scorer will normalize it internally)
        result_original = scorer.calculate_feasibility_score(text)
        result_normalized = scorer.calculate_feasibility_score(normalized)
        
        components_original = len(result_original.components_detected)
        components_normalized = len(result_normalized.components_detected)
        
        print(f'   Components detected (original):   {components_original}')
        print(f'   Components detected (normalized): {components_normalized}')
        print(f'   Score (original):   {result_original.feasibility_score:.3f}')
        print(f'   Score (normalized): {result_normalized.feasibility_score:.3f}')
        
        if components_normalized >= components_original:
            total_improvements += 1
            print('   ✓ Normalization maintained or improved detection')
        else:
            print('   ✗ Normalization reduced detection')
        
        # Show detailed component matches
        if result_original.detailed_matches:
            print('   Detailed matches:')
            for match in result_original.detailed_matches:
                print(f'     - {match.component_type.value}: "{match.matched_text}"')
    
    print(f'\n{"=" * 60}')
    print(f'SUMMARY: {total_improvements}/{len(test_samples)} cases maintained/improved detection')
    improvement_rate = total_improvements / len(test_samples) * 100
    print(f'Improvement rate: {improvement_rate:.1f}%')

if __name__ == '__main__':
    test_unicode_normalization()