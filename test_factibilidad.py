#!/usr/bin/env python3
"""
Test script for the factibilidad scoring module.
"""

from factibilidad import PatternDetector, FactibilidadScorer


def test_pattern_detection():
    """Test basic pattern detection functionality."""
    detector = PatternDetector()
    
    # Test text with all three pattern types
    test_text = """
    La línea base actual muestra que tenemos 100 usuarios registrados.
    Nuestro objetivo es alcanzar 500 usuarios para diciembre de 2024.
    Esta meta representa un crecimiento del 400% en 6 meses.
    """
    
    matches = detector.detect_patterns(test_text)
    
    print("=== Pattern Detection Test ===")
    for pattern_type, pattern_matches in matches.items():
        print(f"\n{pattern_type.upper()} patterns found: {len(pattern_matches)}")
        for match in pattern_matches:
            print(f"  - '{match.text}' at position {match.start}-{match.end}")
    
    clusters = detector.find_pattern_clusters(test_text)
    print(f"\nClusters found: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: span={cluster['span']} chars")
        print(f"    Text: {cluster['text'][:100]}...")


def test_scoring():
    """Test the factibilidad scoring functionality."""
    scorer = FactibilidadScorer()
    
    # Test texts with different levels of completeness
    test_texts = [
        {
            'name': 'Complete text',
            'text': """
            Actualmente tenemos una línea base de 50 proyectos completados.
            Nuestro objetivo es alcanzar 120 proyectos para el año 2025.
            Esta meta debe lograrse en un plazo de 18 meses.
            """,
            'similarity_score': 0.8
        },
        {
            'name': 'Missing baseline',
            'text': """
            El objetivo principal es conseguir 200 nuevos clientes.
            Esta meta debe cumplirse antes de diciembre de 2024.
            """,
            'similarity_score': 0.6
        },
        {
            'name': 'Missing timeframe',
            'text': """
            Partiendo de la situación inicial de 30% de satisfacción,
            buscamos alcanzar un 85% de satisfacción del cliente.
            """,
            'similarity_score': 0.7
        },
        {
            'name': 'Scattered patterns',
            'text': """
            La empresa tiene como propósito expandir su mercado.
            
            En el estado actual, contamos con 5 oficinas.
            
            Para el próximo año esperamos resultados positivos.
            """,
            'similarity_score': 0.4
        }
    ]
    
    print("\n=== Scoring Test ===")
    for test_case in test_texts:
        print(f"\n--- {test_case['name']} ---")
        result = scorer.score_text(test_case['text'], test_case['similarity_score'])
        
        # Show refined scoring results
        print(f"Refined Score Final: {result['score_final']:.4f}")
        print(f"  - Similarity Score: {result['similarity_score']:.2f} (w1={result['weights']['w1']})")
        print(f"  - Causal Density: {result['causal_density']:.6f} (w2={result['weights']['w2']})")
        print(f"  - Informative Ratio: {result['informative_length_ratio']:.3f} (w3={result['weights']['w3']})")
        print(f"  - Causal Connections: {result['causal_connections']}")
        print(f"  - Segment Length: {result['segment_length']} chars")
        
        # Show legacy scoring for comparison
        print(f"Legacy Total Score: {result['total_score']:.1f}")
        print(f"Clusters found: {result['cluster_scores']['count']}")
        
        analysis = result['analysis']
        print(f"Has baseline: {analysis['has_baseline']}")
        print(f"Has target: {analysis['has_target']}")
        print(f"Has timeframe: {analysis['has_timeframe']}")
        
        if analysis['strengths']:
            print("Strengths:")
            for strength in analysis['strengths']:
                print(f"  - {strength}")
        
        if analysis['weaknesses']:
            print("Weaknesses:")
            for weakness in analysis['weaknesses']:
                print(f"  - {weakness}")


def test_weight_validation():
    """Test weight validation and configuration."""
    print("\n=== Weight Validation Test ===")
    
    # Test valid weights
    try:
        scorer = FactibilidadScorer(w1=0.4, w2=0.4, w3=0.2)
        print("✓ Valid weights accepted: w1=0.4, w2=0.4, w3=0.2")
    except ValueError as e:
        print(f"✗ Valid weights rejected: {e}")
    
    # Test invalid weights (sum too low)
    try:
        scorer = FactibilidadScorer(w1=0.2, w2=0.2, w3=0.2)
        print("✗ Invalid weights accepted: w1=0.2, w2=0.2, w3=0.2")
    except ValueError as e:
        print(f"✓ Invalid weights (sum too low) rejected: {e}")
        
    # Test invalid weights (sum too high)  
    try:
        scorer = FactibilidadScorer(w1=0.6, w2=0.5, w3=0.4)
        print("✗ Invalid weights accepted: w1=0.6, w2=0.5, w3=0.4")
    except ValueError as e:
        print(f"✓ Invalid weights (sum too high) rejected: {e}")
        
    # Test weight updates
    try:
        scorer = FactibilidadScorer()
        scorer.update_weights(w1=0.6, w2=0.3, w3=0.1)
        print(f"✓ Weight update successful: w1={scorer.w1}, w2={scorer.w2}, w3={scorer.w3}")
    except ValueError as e:
        print(f"✗ Weight update failed: {e}")


def test_edge_cases():
    """Test edge cases for refined scoring."""
    print("\n=== Edge Cases Test ===")
    
    scorer = FactibilidadScorer()
    
    edge_cases = [
        ("Empty text", ""),
        ("Only stopwords", "el la de que y en un es se no"),
        ("Only whitespace", "   \n\t   "),
        ("Single word", "objetivo"),
        ("Very long text with patterns", 
         "línea base " + "contenido intermedio " * 100 + "objetivo meta " + "más contenido " * 100 + "para 2025")
    ]
    
    for case_name, text in edge_cases:
        print(f"\n--- {case_name} ---")
        try:
            result = scorer.score_text(text, 0.5)
            print(f"Score final: {result['score_final']:.4f}")
            print(f"Informative ratio: {result['informative_length_ratio']:.3f}")
            print(f"Causal connections: {result['causal_connections']}")
            print(f"Segment length: {result['segment_length']}")
        except Exception as e:
            print(f"Error: {e}")


def test_specific_patterns():
    """Test specific pattern recognition."""
    detector = PatternDetector()
    
    pattern_tests = [
        ("Baseline patterns", [
            "línea base establecida",
            "situación inicial crítica", 
            "punto de partida claro",
            "estado actual preocupante",
            "valor inicial de referencia"
        ]),
        ("Target patterns", [
            "meta ambiciosa",
            "objetivo principal",
            "alcanzar resultados",
            "conseguir la mejora",
            "lograr el cambio"
        ]),
        ("Timeframe patterns", [
            "al 2025",
            "para diciembre de 2024",
            "en 6 meses",
            "primer trimestre",
            "próximo año",
            "2023-2025"
        ])
    ]
    
    print("\n=== Specific Pattern Tests ===")
    for category, test_phrases in pattern_tests:
        print(f"\n--- {category} ---")
        for phrase in test_phrases:
            matches = detector.detect_patterns(phrase)
            found_types = [pt for pt, matches in matches.items() if matches]
            print(f"'{phrase}' -> {found_types}")


if __name__ == '__main__':
    test_pattern_detection()
    test_scoring()
    test_weight_validation()
    test_edge_cases()
    test_specific_patterns()