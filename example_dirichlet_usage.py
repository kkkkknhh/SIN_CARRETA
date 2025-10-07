#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of DirichletAggregator for Bayesian evidence consensus.

Demonstrates:
1. Basic DirichletAggregator usage
2. Integration with EvidenceRegistry
3. Consensus determination across multiple detectors
4. Uncertainty quantification
"""

import numpy as np
from evidence_registry import DirichletAggregator, EvidenceRegistry


def example_1_basic_aggregator():
    """Example 1: Basic DirichletAggregator usage"""
    print("=" * 80)
    print("EJEMPLO 1: USO BÁSICO DE DIRICHLET AGGREGATOR")
    print("=" * 80)
    
    # Create aggregator for 3 categories
    agg = DirichletAggregator(k=3, alpha0=0.5)
    print(f"\nAgregador inicializado con k=3, alpha0=0.5")
    print(f"Prior alpha: {agg.alpha}")
    
    # Add unanimous votes for category 0
    print("\n1. Votos unánimes para categoría 0:")
    agg.update_from_labels(np.array([0, 0, 0, 0, 0]))
    
    mean = agg.posterior_mean()
    max_cat, max_prob = agg.max_probability()
    entropy = agg.entropy()
    
    print(f"   Posterior mean: {mean}")
    print(f"   Categoría más probable: {max_cat} con P={max_prob:.2%}")
    print(f"   Entropía (incertidumbre): {entropy:.3f}")
    
    # Add split votes
    print("\n2. Agregar votos divididos:")
    agg.update_from_labels(np.array([1, 2]))
    
    mean = agg.posterior_mean()
    max_cat, max_prob = agg.max_probability()
    entropy = agg.entropy()
    
    print(f"   Posterior mean: {mean}")
    print(f"   Categoría más probable: {max_cat} con P={max_prob:.2%}")
    print(f"   Entropía (incertidumbre): {entropy:.3f}")
    
    # Credible intervals
    print("\n3. Intervalos de credibilidad 95%:")
    intervals = agg.credible_interval(level=0.95)
    for i, (lo, hi) in enumerate(intervals):
        print(f"   Categoría {i}: [{lo:.3f}, {hi:.3f}]")
    
    print()


def example_2_dimension_consensus():
    """Example 2: Consensus across multiple detectors for dimension voting"""
    print("=" * 80)
    print("EJEMPLO 2: CONSENSO DE DIMENSIÓN CON MÚLTIPLES DETECTORES")
    print("=" * 80)
    
    registry = EvidenceRegistry()
    
    # Scenario: 4 detectors analyze same evidence
    evidence_id = "evidencia_presupuesto_001"
    
    print(f"\nEvidencia ID: {evidence_id}")
    print("\nVotos de detectores:")
    
    # Detector 1: responsibility_detector votes D4 (index 3)
    registry.register_evidence(
        evidence_id=evidence_id,
        source="responsibility_detector",
        dimension_vote=3,  # D4
        content_type=0,
        risk_level=1,
        confidence=0.9
    )
    print("  ✓ responsibility_detector → D4 (confianza: 0.90)")
    
    # Detector 2: monetary_detector also votes D4
    registry.register_evidence(
        evidence_id=evidence_id,
        source="monetary_detector",
        dimension_vote=3,  # D4
        content_type=1,
        risk_level=1,
        confidence=0.85
    )
    print("  ✓ monetary_detector → D4 (confianza: 0.85)")
    
    # Detector 3: causal_detector votes D4
    registry.register_evidence(
        evidence_id=evidence_id,
        source="causal_detector",
        dimension_vote=3,  # D4
        content_type=2,
        risk_level=1,
        confidence=0.80
    )
    print("  ✓ causal_detector → D4 (confianza: 0.80)")
    
    # Detector 4: feasibility_scorer votes D5 (dissenting opinion)
    registry.register_evidence(
        evidence_id=evidence_id,
        source="feasibility_scorer",
        dimension_vote=4,  # D5
        content_type=0,
        risk_level=2,
        confidence=0.70
    )
    print("  ✓ feasibility_scorer → D5 (confianza: 0.70)")
    
    # Get distribution
    print("\nDistribución posterior de dimensión:")
    dist = registry.get_dimension_distribution(evidence_id)
    
    # Show top 3 dimensions
    mean = dist['mean']
    sorted_dims = np.argsort(mean)[::-1][:3]
    
    for dim in sorted_dims:
        prob = mean[dim]
        if prob > 0.01:  # Only show if > 1%
            print(f"  D{dim+1}: {prob:.1%}")
    
    # Check consensus
    max_cat, max_prob = dist['max_category']
    print(f"\nCategoría más probable: D{max_cat+1} ({max_prob:.1%})")
    print(f"Entropía (incertidumbre): {dist['entropy']:.3f}")
    print(f"Número de votos: {dist['n_votes']}")
    
    # Consensus check
    consensus = registry.get_consensus_dimension(evidence_id, threshold=0.6)
    if consensus is not None:
        print(f"\n✅ CONSENSO ALCANZADO: D{consensus+1}")
    else:
        print(f"\n⚠️  SIN CONSENSO CLARO (umbral: 60%)")
    
    print()


def example_3_uncertainty_comparison():
    """Example 3: Compare high vs low uncertainty scenarios"""
    print("=" * 80)
    print("EJEMPLO 3: COMPARACIÓN DE ESCENARIOS DE INCERTIDUMBRE")
    print("=" * 80)
    
    registry = EvidenceRegistry()
    
    # Scenario A: Strong consensus (low uncertainty)
    print("\nEscenario A: CONSENSO FUERTE")
    evidence_a = "evidencia_consenso"
    
    for i in range(5):
        registry.register_evidence(
            evidence_id=evidence_a,
            source=f"detector_{i}",
            dimension_vote=2,  # All vote D3
            content_type=0,
            risk_level=1,
            confidence=0.85
        )
    
    dist_a = registry.get_dimension_distribution(evidence_a)
    max_cat_a, max_prob_a = dist_a['max_category']
    
    print(f"  Dimensión dominante: D{max_cat_a+1} ({max_prob_a:.1%})")
    print(f"  Entropía: {dist_a['entropy']:.3f}")
    print(f"  Interpretación: Baja incertidumbre, consenso claro")
    
    # Scenario B: Split votes (high uncertainty)
    print("\nEscenario B: VOTOS DIVIDIDOS")
    evidence_b = "evidencia_dividida"
    
    # Each detector votes for different dimension
    for i in range(5):
        registry.register_evidence(
            evidence_id=evidence_b,
            source=f"detector_{i}",
            dimension_vote=i,  # Different dimension each
            content_type=0,
            risk_level=1,
            confidence=0.80
        )
    
    dist_b = registry.get_dimension_distribution(evidence_b)
    max_cat_b, max_prob_b = dist_b['max_category']
    
    print(f"  Dimensión dominante: D{max_cat_b+1} ({max_prob_b:.1%})")
    print(f"  Entropía: {dist_b['entropy']:.3f}")
    print(f"  Interpretación: Alta incertidumbre, sin consenso")
    
    # Compare
    print("\nComparación:")
    print(f"  Escenario A - Entropía: {dist_a['entropy']:.3f} (consenso)")
    print(f"  Escenario B - Entropía: {dist_b['entropy']:.3f} (división)")
    print(f"  Diferencia: {dist_b['entropy'] - dist_a['entropy']:.3f}")
    
    print()


def example_4_credible_intervals():
    """Example 4: Credible intervals for uncertainty quantification"""
    print("=" * 80)
    print("EJEMPLO 4: INTERVALOS DE CREDIBILIDAD PARA CUANTIFICAR INCERTIDUMBRE")
    print("=" * 80)
    
    # Strong evidence for one dimension
    agg = DirichletAggregator(k=10, alpha0=0.5)
    
    # 15 votes for D4 (index 3), 3 votes for D5 (index 4)
    votes = [3] * 15 + [4] * 3
    agg.update_from_labels(np.array(votes))
    
    print("\nEvidencia: 15 votos para D4, 3 votos para D5")
    
    mean = agg.posterior_mean()
    intervals = agg.credible_interval(level=0.95)
    
    print("\nProbabilidades e intervalos de credibilidad 95%:")
    
    # Show only dimensions with > 1% probability
    for i in range(10):
        if mean[i] > 0.01:
            lo, hi = intervals[i]
            print(f"  D{i+1}: {mean[i]:.1%} [{lo:.3f}, {hi:.3f}]")
    
    # Interpretation
    max_cat, max_prob = agg.max_probability()
    lo, hi = intervals[max_cat]
    
    print(f"\nInterpretación:")
    print(f"  La dimensión más probable es D{max_cat+1}")
    print(f"  Probabilidad puntual: {max_prob:.1%}")
    print(f"  Intervalo de credibilidad 95%: [{lo:.1%}, {hi:.1%}]")
    print(f"  Conclusión: Tenemos 95% de confianza que la verdadera")
    print(f"             probabilidad de D{max_cat+1} está en ese rango")
    
    print()


def example_5_real_world_pipeline():
    """Example 5: Complete pipeline simulation"""
    print("=" * 80)
    print("EJEMPLO 5: SIMULACIÓN DE PIPELINE COMPLETO")
    print("=" * 80)
    
    registry = EvidenceRegistry()
    
    # Simulate processing 3 pieces of evidence
    evidence_items = [
        ("evidencia_001", "Asignación presupuestal para obra pública"),
        ("evidencia_002", "Competencias del alcalde municipal"),
        ("evidencia_003", "Indicadores de seguimiento trimestral")
    ]
    
    # Simulated detector votes (in real system, these come from actual detectors)
    votes_simulation = {
        "evidencia_001": [
            ("responsibility_detector", 3, 0.90),  # D4
            ("monetary_detector", 3, 0.95),         # D4
            ("causal_detector", 3, 0.85),           # D4
            ("feasibility_scorer", 2, 0.75),        # D3
        ],
        "evidencia_002": [
            ("responsibility_detector", 3, 0.95),   # D4
            ("contradiction_detector", 3, 0.80),    # D4
        ],
        "evidencia_003": [
            ("monetary_detector", 5, 0.85),         # D6
            ("feasibility_scorer", 5, 0.90),        # D6
            ("causal_detector", 4, 0.70),           # D5
        ]
    }
    
    print("\nProcesando evidencias del pipeline:\n")
    
    for evidence_id, description in evidence_items:
        print(f"{evidence_id}: {description}")
        
        # Register votes
        votes = votes_simulation[evidence_id]
        for detector, dim, conf in votes:
            registry.register_evidence(
                evidence_id=evidence_id,
                source=detector,
                dimension_vote=dim,
                content_type=0,
                risk_level=1,
                confidence=conf
            )
            print(f"  ← {detector}: D{dim+1} (conf={conf:.2f})")
        
        # Get consensus
        dist = registry.get_dimension_distribution(evidence_id)
        max_cat, max_prob = dist['max_category']
        consensus = registry.get_consensus_dimension(evidence_id, threshold=0.6)
        
        if consensus is not None:
            print(f"  → CONSENSO: D{consensus+1} ({max_prob:.1%}, H={dist['entropy']:.2f})")
        else:
            print(f"  → SIN CONSENSO: D{max_cat+1} mejor ({max_prob:.1%}, H={dist['entropy']:.2f})")
        
        print()
    
    print("Pipeline completo: 3 evidencias procesadas con consenso bayesiano")
    print()


if __name__ == "__main__":
    example_1_basic_aggregator()
    example_2_dimension_consensus()
    example_3_uncertainty_comparison()
    example_4_credible_intervals()
    example_5_real_world_pipeline()
    
    print("=" * 80)
    print("TODOS LOS EJEMPLOS COMPLETADOS")
    print("=" * 80)
