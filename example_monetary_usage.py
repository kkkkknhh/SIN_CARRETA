#!/usr/bin/env python3
"""
Example usage of the MonetaryDetector for Spanish text processing.

This module demonstrates how to use the MonetaryDetector to find and normalize
monetary amounts, percentages, and numeric values with scales in Spanish text.
"""

from monetary_detector import create_monetary_detector, MonetaryType

def main():
    """Demonstrate monetary detection capabilities."""
    # Create detector instance
    detector = create_monetary_detector()
    
    print("=== Spanish Monetary Expression Detection Demo ===\n")
    
    # Example Spanish text with various monetary expressions
    sample_texts = [
        "El presupuesto total es de $2.5 millones COP para el próximo año.",
        "La empresa reportó ingresos de €1.234.567,89 con un crecimiento del 15%.",
        "El proyecto requiere una inversión de USD 500K aproximadamente.",
        "Los costos operativos aumentaron a 1,8M respecto a los 1.200 miles del año anterior.",
        "La meta de ventas es alcanzar ¥2.500.000 con un margen del 12,5%.",
        "El fondo tiene assets por £750K y espera un retorno del 8,3%.",
    ]
    
    # Process each text sample
    for i, text in enumerate(sample_texts, 1):
        print(f"Sample {i}: {text}")
        results = detector.detect_monetary_expressions(text)
        
        if results:
            print("  Detected expressions:")
            for j, match in enumerate(results, 1):
                type_name = match.type.value.upper()
                currency_info = f" ({match.currency})" if match.currency else ""
                print(f"    [{j}] '{match.text}' -> {match.value:,.2f}{currency_info} [{type_name}]")
        else:
            print("  No monetary expressions detected")
        print()
    
    print("=== Convention Examples ===\n")
    
    # Demonstrate handling of ambiguous abbreviations
    abbreviation_examples = [
        ("$1M", "M = million"),
        ("€2.5MM", "MM = million (same as M)"),
        ("£500K", "K = thousand"),
        ("¥3B", "B = billion"),
    ]
    
    print("Abbreviation conventions:")
    for expr, explanation in abbreviation_examples:
        result = detector.normalize_monetary_expression(expr)
        print(f"  {expr:8} -> {result:>12,.0f} ({explanation})")
    print()
    
    # Demonstrate decimal separator handling
    decimal_examples = [
        ("$1.234.567,89", "Spanish format: period=thousands, comma=decimal"),
        ("$1,234,567.89", "English format: comma=thousands, period=decimal"),
        ("€12.345,60", "Mixed format: automatic detection"),
    ]
    
    print("Decimal separator conventions:")
    for expr, explanation in decimal_examples:
        result = detector.normalize_monetary_expression(expr)
        print(f"  {expr:15} -> {result:>12,.2f} ({explanation})")
    print()
    
    # Demonstrate percentage conversion
    percentage_examples = ["50%", "12,5%", "0,75%", "150%"]
    
    print("Percentage conversion (to decimal):")
    for expr in percentage_examples:
        result = detector.normalize_monetary_expression(expr)
        print(f"  {expr:6} -> {result:.4f}")
    print()
    
    print("=== Advanced Features ===\n")
    
    # Complex text analysis
    complex_text = """
    El informe financiero muestra que los ingresos alcanzaron $15.5 millones USD,
    un incremento del 23% comparado con los €12M del período anterior. Los gastos
    operativos fueron de 8,2M COP, mientras que la inversión en I+D representó
    el 4,5% del presupuesto total. Se proyecta un crecimiento del 18,7% para el
    siguiente trimestre.
    """
    
    print("Complex text analysis:")
    print(complex_text.strip())
    print("\nExtracted values:")
    
    results = detector.detect_monetary_expressions(complex_text)
    for i, match in enumerate(results, 1):
        type_name = match.type.value.upper()
        currency_info = f" {match.currency}" if match.currency else ""
        position_info = f" at position {match.start_pos}-{match.end_pos}"
        print(f"  [{i}] {match.value:>12,.2f}{currency_info:>5} [{type_name}] - '{match.text}'{position_info}")
    
    print(f"\nTotal expressions found: {len(results)}")
    
    # Summary statistics
    currencies = [r for r in results if r.type == MonetaryType.CURRENCY and r.currency]
    percentages = [r for r in results if r.type == MonetaryType.PERCENTAGE]
    numerics = [r for r in results if r.type == MonetaryType.NUMERIC]
    
    print(f"  - Currency amounts: {len(currencies)}")
    print(f"  - Percentages: {len(percentages)}")
    print(f"  - Numeric with scales: {len(numerics)}")
    
    if currencies:
        unique_currencies = set(r.currency for r in currencies)
        print(f"  - Unique currencies: {', '.join(sorted(unique_currencies))}")


if __name__ == "__main__":
    main()