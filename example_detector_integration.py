#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration example: How to use DirichletAggregator in detectors.

Shows how responsibility_detector.py and other detectors can integrate
with the new Bayesian evidence aggregation system.
"""

from evidence_registry import EvidenceRegistry


def example_responsibility_detector_integration():
    """
    Example: responsibility_detector.py integration
    
    This shows how the existing responsibility_detector can be modified
    to use the new Bayesian aggregation.
    """
    print("=" * 80)
    print("INTEGRATION EXAMPLE: responsibility_detector.py")
    print("=" * 80)
    
    # Initialize registry (normally done by orchestrator)
    evidence_registry = EvidenceRegistry()
    
    # Simulate responsibility detection results
    text = """
    El Alcalde Municipal será responsable de la implementación del 
    proyecto de desarrollo urbano, con un presupuesto asignado de 
    $500,000,000 COP para el año 2024.
    """
    
    # In real responsibility_detector.py:
    # entities = extract_entities(text)
    
    # Simulated entities detected
    entities = [
        {
            'text': 'Alcalde Municipal',
            'confidence': 0.95,
            'type': 'POSITION'
        }
    ]
    
    print(f"\nTexto analizado: {text.strip()}")
    print(f"\nEntidades detectadas: {len(entities)}")
    
    for entity in entities:
        evidence_id = f"resp_{hash(entity['text']) % 10000:04d}"
        
        print(f"\nRegistrando evidencia para: {entity['text']}")
        print(f"  - Evidence ID: {evidence_id}")
        print(f"  - Confianza: {entity['confidence']:.2f}")
        
        # Este detector vota D4 (responsabilidades) - índice 3
        evidence_registry.register_evidence(
            evidence_id=evidence_id,
            source="responsibility_detector",
            dimension_vote=3,  # D4 (índice 3)
            content_type=0,    # Presupuestal
            risk_level=1,      # Medio
            confidence=entity['confidence']
        )
        
        print(f"  ✓ Voto registrado: D4 (Responsabilidades)")
    
    # Later, monetary_detector might also analyze same evidence
    print(f"\nSimulando detector adicional (monetary_detector)...")
    
    # monetary_detector finds monetary amount, also votes D4
    evidence_registry.register_evidence(
        evidence_id=f"resp_{hash('Alcalde Municipal') % 10000:04d}",
        source="monetary_detector",
        dimension_vote=3,  # También vota D4
        content_type=0,
        risk_level=1,
        confidence=0.90
    )
    print(f"  ✓ monetary_detector también vota D4")
    
    # Check consensus
    evidence_id = f"resp_{hash('Alcalde Municipal') % 10000:04d}"
    dist = evidence_registry.get_dimension_distribution(evidence_id)
    
    print(f"\nDistribución posterior para evidencia {evidence_id}:")
    max_cat, max_prob = dist['max_category']
    print(f"  Dimensión más probable: D{max_cat+1} ({max_prob:.1%})")
    print(f"  Entropía (incertidumbre): {dist['entropy']:.3f}")
    print(f"  Número de votos: {dist['n_votes']}")
    
    consensus = evidence_registry.get_consensus_dimension(evidence_id, threshold=0.6)
    if consensus is not None:
        print(f"  ✅ CONSENSO: D{consensus+1}")
    else:
        print(f"  ⚠️  Sin consenso claro")


def example_orchestrator_consolidation():
    """
    Example: Orchestrator consolidation phase
    
    Shows how the orchestrator can consolidate evidence from multiple detectors.
    """
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLE: orchestrator consolidation")
    print("=" * 80)
    
    # Initialize registry
    evidence_registry = EvidenceRegistry()
    
    # Simulate multiple detectors processing same document
    print("\nSimulando detección de múltiples módulos...")
    
    # Evidence 1: Responsibility assignment
    evidence_registry.register_evidence(
        evidence_id="evidence_001",
        source="responsibility_detector",
        dimension_vote=3,  # D4
        content_type=0,
        risk_level=1,
        confidence=0.92
    )
    print("  ✓ responsibility_detector → D4")
    
    evidence_registry.register_evidence(
        evidence_id="evidence_001",
        source="monetary_detector",
        dimension_vote=3,  # D4
        content_type=0,
        risk_level=1,
        confidence=0.88
    )
    print("  ✓ monetary_detector → D4")
    
    evidence_registry.register_evidence(
        evidence_id="evidence_001",
        source="causal_detector",
        dimension_vote=3,  # D4
        content_type=2,
        risk_level=1,
        confidence=0.85
    )
    print("  ✓ causal_detector → D4")
    
    # Evidence 2: Mixed signals
    evidence_registry.register_evidence(
        evidence_id="evidence_002",
        source="feasibility_scorer",
        dimension_vote=2,  # D3
        content_type=1,
        risk_level=0,
        confidence=0.75
    )
    print("  ✓ feasibility_scorer → D3")
    
    evidence_registry.register_evidence(
        evidence_id="evidence_002",
        source="contradiction_detector",
        dimension_vote=5,  # D6
        content_type=3,
        risk_level=2,
        confidence=0.80
    )
    print("  ✓ contradiction_detector → D6")
    
    # Consolidation phase
    print("\nFase de consolidación en orchestrator:")
    print("-" * 80)
    
    for evidence_id in evidence_registry.dimension_aggregators:
        dist = evidence_registry.get_dimension_distribution(evidence_id)
        
        print(f"\n{evidence_id}:")
        max_cat, max_prob = dist['max_category']
        print(f"  Dimensión más probable: D{max_cat+1} ({max_prob:.1%})")
        print(f"  Incertidumbre (entropía): {dist['entropy']:.3f}")
        
        # Verificar consenso
        consensus_dim = evidence_registry.get_consensus_dimension(
            evidence_id, threshold=0.6
        )
        
        if consensus_dim is not None:
            print(f"  ✅ CONSENSO en D{consensus_dim+1}")
            print(f"     → Usar esta dimensión para evaluación")
        else:
            print(f"  ⚠️  Sin consenso claro")
            print(f"     → Usar D{max_cat+1} pero marcar como incierto")
            print(f"     → Considerar revisión manual si es crítico")


def example_api_usage_patterns():
    """Common API usage patterns"""
    print("\n" + "=" * 80)
    print("API USAGE PATTERNS")
    print("=" * 80)
    
    registry = EvidenceRegistry()
    
    print("\nPatrón 1: Registro simple desde detector")
    print("-" * 40)
    print("""
    # En cualquier detector:
    registry.register_evidence(
        evidence_id="unique_id",
        source="my_detector",
        dimension_vote=3,      # D4 (índice 3)
        content_type=0,        # 0-4
        risk_level=1,          # 0-2 (bajo/medio/alto)
        confidence=0.85        # 0.0-1.0
    )
    """)
    
    print("\nPatrón 2: Consultar distribución posterior")
    print("-" * 40)
    print("""
    dist = registry.get_dimension_distribution(evidence_id)
    
    if dist:
        mean = dist['mean']              # Probabilidades por dimensión
        max_cat, max_prob = dist['max_category']
        entropy = dist['entropy']        # Incertidumbre
        intervals = dist['credible_interval']  # Intervalos 95%
    """)
    
    print("\nPatrón 3: Verificar consenso con umbral")
    print("-" * 40)
    print("""
    consensus = registry.get_consensus_dimension(
        evidence_id, 
        threshold=0.6  # 60% de probabilidad
    )
    
    if consensus is not None:
        # Usar consenso
        dimension = consensus
    else:
        # Manejar incertidumbre
        dimension = dist['max_category'][0]
        mark_as_uncertain = True
    """)


if __name__ == "__main__":
    example_responsibility_detector_integration()
    example_orchestrator_consolidation()
    example_api_usage_patterns()
    
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLES COMPLETED")
    print("=" * 80)
