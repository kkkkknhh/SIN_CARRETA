from causal_pattern_detector import CausalPatternDetector

detector = CausalPatternDetector()

# Debug failing annotated test cases
failing_texts = {
    "implica_logical": "Si x > 5, entonces la ecuación implica que y debe ser negativo.",
    "tendencia_pattern": "Los datos muestran una tendencia a la correlación positiva en el gráfico.",
    "complex_text": """La deforestación implica pérdida de biodiversidad, lo cual conduce a 
                      desequilibrios ecológicos. Estos se estudian mediante técnicas de 
                      monitoreo satelital, observando tendencia a patrones preocupantes.""",
}

for key, text in failing_texts.items():
    print(f"\n=== Testing: {key} ===")
    print(f"Text: {text}")
    matches = detector.detect_causal_patterns(text)
    print(f"Matches found: {len(matches)}")
    for match in matches:
        print(f"  Connector: {match.connector}")
        print(f"  Text: '{match.text}'")
        print(f"  Confidence: {match.confidence}")
        print(f"  Semantic Strength: {match.semantic_strength}")
        print(f"  Context before: '{match.context_before[:50]}...'")
        print(f"  Context after: '{match.context_after[:50]}...'")

# Test specific false positive patterns
print("\n=== Testing False Positive Patterns ===")
for pattern in detector.false_positive_contexts:
    print(f"Pattern: {pattern.pattern}")

    # Test against the logical example
    logical_text = "Si x > 5, entonces la ecuación implica que y debe ser negativo."
    if pattern.search(logical_text):
        print("  Matches logical text: YES")
    else:
        print("  Matches logical text: NO")
