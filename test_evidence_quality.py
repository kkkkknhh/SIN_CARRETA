#!/usr/bin/env python3
"""
Simple test script for the calcular_calidad_evidencia method
"""

from feasibility_scorer import FeasibilityScorer

def test_calcular_calidad_evidencia():
    scorer = FeasibilityScorer()
    
    # Test cases with expected approximate scores
    test_cases = [
        # High quality: monetary + temporal + terminology
        ('Línea base: COP $5.2 millones en 2023, meta $8.5 millones para Q4 2025 con monitoreo trimestral', 0.7),
        
        # Medium quality: some indicators
        ('Investment of $2.3 million USD baseline for 2024 target achievement', 0.4),
        
        # Low quality with terminology
        ('Indicador de desempeño con periodicidad anual desde enero 2024', 0.3),
        
        # Title penalty case
        ('• Mejora del sistema educativo', 0.1),
        
        # Empty case
        ('', 0.0),
        
        # Edge case with malformed data
        ('Presupuesto: $..5 millones año 20XX', 0.2)
    ]
    
    print("Testing calcular_calidad_evidencia method:")
    print("=" * 60)
    
    all_passed = True
    for text, expected_min in test_cases:
        score = scorer.calcular_calidad_evidencia(text)
        passed = 0.0 <= score <= 1.0
        
        if expected_min > 0:
            passed = passed and score >= expected_min - 0.2  # Allow some tolerance
        
        status = "✓" if passed else "✗"
        display_text = text[:40] + "..." if len(text) > 40 else text
        print(f"{status} Score: {score:.3f} | Expected: >={expected_min} | {display_text}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All calcular_calidad_evidencia tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_calcular_calidad_evidencia()