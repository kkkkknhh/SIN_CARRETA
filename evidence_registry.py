#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Registry - Central Immutable Evidence Store
====================================================

Provides a canonical, immutable, deterministic registry for all evidence
produced by MINIMINIMOON pipeline components. Ensures:
- Single source of truth for evidence
- Provenance tracking (evidence → questions)
- Deterministic hashing for reproducibility
- Thread-safe registration
- Bayesian evidence aggregation via Dirichlet posteriors
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from threading import Lock
import logging

import numpy as np
from scipy.stats import dirichlet

logger = logging.getLogger(__name__)


class DirichletAggregator:
    """
    Fusiona votos categóricos con prior Dirichlet.
    Mantiene posterior α y calcula estadísticas.
    
    Uses Dirichlet-Multinomial conjugate posterior for Bayesian
    aggregation of categorical votes from multiple sources.
    """
    
    def __init__(self, k: int, alpha0: float = 0.5):
        """
        Initialize Dirichlet aggregator.
        
        Args:
            k: Número de categorías
            alpha0: Prior Dirichlet (0.5 = Jeffreys, 1.0 = uniforme)
        """
        self.k = k
        self.alpha0 = alpha0
        self.alpha = np.full(k, alpha0, dtype=float)
        self.n_updates = 0
    
    def update_from_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Actualiza posterior con votos categóricos.
        
        Args:
            labels: Array de enteros en [0, k)
        
        Returns:
            Parámetros alpha actualizados
        
        Raises:
            ValueError: Si labels fuera de rango
        """
        labels = np.asarray(labels, dtype=int)
        
        # Validar rango
        if len(labels) > 0 and (labels.min() < 0 or labels.max() >= self.k):
            raise ValueError(f"Labels fuera de rango [0, {self.k})")
        
        # Contar votos
        counts = np.bincount(labels, minlength=self.k)
        
        # Actualizar posterior (conjugado)
        self.alpha += counts
        self.n_updates += len(labels)
        
        logger.debug(f"Agregador actualizado: {len(labels)} votos, total={self.n_updates}")
        return self.alpha.copy()
    
    def update_from_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Actualiza con pesos continuos (votos fraccionarios).
        
        Args:
            weights: Array de shape (k,) con pesos >= 0 que suman ~1
        
        Returns:
            Parámetros alpha actualizados
            
        Raises:
            ValueError: Si weights tiene shape incorrecta
        """
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.k,):
            raise ValueError(f"Weights debe tener shape ({self.k},)")
        
        # Normalizar y escalar
        weights_norm = weights / (weights.sum() + 1e-10)
        pseudo_counts = weights_norm * 10  # Escalar para pseudo-conteos
        
        self.alpha += pseudo_counts
        self.n_updates += 1
        
        return self.alpha.copy()
    
    def posterior_mean(self) -> np.ndarray:
        """
        Media posterior E[θ] = α / Σα
        
        Returns:
            Array de shape (k,) con probabilidades
        """
        return self.alpha / self.alpha.sum()
    
    def posterior_mode(self) -> np.ndarray:
        """
        Moda posterior (MAP): (α - 1) / (Σα - k)
        Solo válida si todos α > 1
        
        Returns:
            Array de shape (k,) con probabilidades (o media si α <= 1)
        """
        if (self.alpha <= 1).any():
            logger.warning("Moda no definida para α <= 1, usando media")
            return self.posterior_mean()
        
        return (self.alpha - 1) / (self.alpha.sum() - self.k)
    
    def credible_interval(self, level: float = 0.95, n_samples: int = 8000) -> np.ndarray:
        """
        Intervalo de credibilidad para cada categoría.
        
        Args:
            level: Nivel de credibilidad (default 0.95)
            n_samples: Muestras para Monte Carlo
        
        Returns:
            Array de shape (k, 2) con [lo, hi] por categoría
        """
        # Muestreo de Dirichlet
        samples = dirichlet(self.alpha).rvs(size=n_samples, random_state=42)
        
        # Cuantiles
        lo = np.quantile(samples, (1 - level) / 2, axis=0)
        hi = np.quantile(samples, 1 - (1 - level) / 2, axis=0)
        
        return np.column_stack([lo, hi])
    
    def entropy(self) -> float:
        """
        Entropía de la distribución posterior (incertidumbre).
        Valores altos = alta incertidumbre.
        
        Returns:
            Entropía en nats
        """
        probs = self.posterior_mean()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def max_probability(self) -> Tuple[int, float]:
        """
        Categoría con máxima probabilidad posterior.
        
        Returns:
            (categoria, probabilidad)
        """
        probs = self.posterior_mean()
        idx = int(np.argmax(probs))
        return (idx, float(probs[idx]))
    
    def reset(self):
        """Reinicia al prior."""
        self.alpha = np.full(self.k, self.alpha0, dtype=float)
        self.n_updates = 0


@dataclass(frozen=True)
class CanonicalEvidence:
    """
    Immutable evidence item produced by a pipeline component.

    Attributes:
        source_component: Component that produced this evidence (e.g., 'feasibility_scorer')
        evidence_type: Type of evidence (e.g., 'baseline_presence', 'monetary_value')
        content: Actual evidence content (must be JSON-serializable)
        confidence: Confidence score [0.0, 1.0]
        applicable_questions: List of question IDs this evidence answers (e.g., ['D1-Q1', 'D2-Q5'])
        metadata: Additional metadata (timestamps, version, etc.)
    """
    source_component: str
    evidence_type: str
    content: Any
    confidence: float
    applicable_questions: tuple  # Tuple for immutability
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate evidence at creation"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if not isinstance(self.applicable_questions, tuple):
            # Convert to tuple if list was passed
            object.__setattr__(self, 'applicable_questions', tuple(self.applicable_questions))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_component": self.source_component,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "confidence": self.confidence,
            "applicable_questions": list(self.applicable_questions),
            "metadata": self.metadata
        }


class EvidenceRegistry:
    """
    Central registry for all evidence produced during PDM evaluation.

    Thread-safe, deterministic, and immutable evidence store that tracks:
    - All evidence items by unique ID
    - Question → evidence mappings (provenance)
    - Component → evidence mappings
    - Deterministic hash for reproducibility verification
    """

    def __init__(self):
        """Initialize empty evidence registry"""
        self.store: Dict[str, CanonicalEvidence] = {}
        self.provenance: Dict[str, List[str]] = {}  # question_id → [evidence_ids]
        self.component_index: Dict[str, List[str]] = {}  # component → [evidence_ids]
        self._lock = Lock()
        self._frozen = False
        self.creation_time = time.time()
        
        # Agregadores Dirichlet para consenso bayesiano
        self.dimension_aggregators: Dict[str, DirichletAggregator] = {}  # Por evidence_id
        self.content_type_aggregator = DirichletAggregator(k=5, alpha0=0.5)  # 5 tipos
        self.risk_level_aggregator = DirichletAggregator(k=3, alpha0=1.0)  # bajo/medio/alto

        logger.info("EvidenceRegistry initialized")

    def register(
        self,
        source_component: str,
        evidence_type: str,
        content: Any,
        confidence: float,
        applicable_questions: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register new evidence in the registry.

        Args:
            source_component: Name of component producing evidence
            evidence_type: Type/category of evidence
            content: Evidence content (must be JSON-serializable)
            confidence: Confidence score [0.0, 1.0]
            applicable_questions: List of question IDs this evidence applies to
            metadata: Optional additional metadata

        Returns:
            Evidence ID (deterministic hash)

        Raises:
            RuntimeError: If registry is frozen
            ValueError: If evidence is invalid
        """
        if self._frozen:
            raise RuntimeError("Cannot register evidence: registry is frozen")

        with self._lock:
            # Create metadata
            meta = metadata or {}
            meta.update({
                "timestamp": time.time(),
                "registry_version": "1.0"
            })

            # Create evidence object
            evidence = CanonicalEvidence(
                source_component=source_component,
                evidence_type=evidence_type,
                content=content,
                confidence=confidence,
                applicable_questions=tuple(applicable_questions),
                metadata=meta
            )

            # Generate deterministic evidence ID
            evidence_id = self._generate_evidence_id(evidence)
            meta["evidence_id"] = evidence_id

            # Re-create with updated metadata (needed for immutability)
            evidence = CanonicalEvidence(
                source_component=source_component,
                evidence_type=evidence_type,
                content=content,
                confidence=confidence,
                applicable_questions=tuple(applicable_questions),
                metadata=meta
            )

            # Store evidence
            self.store[evidence_id] = evidence

            # Update provenance index
            for question_id in applicable_questions:
                if question_id not in self.provenance:
                    self.provenance[question_id] = []
                self.provenance[question_id].append(evidence_id)

            # Update component index
            if source_component not in self.component_index:
                self.component_index[source_component] = []
            self.component_index[source_component].append(evidence_id)

            logger.debug(f"Registered evidence {evidence_id} from {source_component} "
                        f"for {len(applicable_questions)} questions")

            return evidence_id

    def _generate_evidence_id(self, evidence: CanonicalEvidence) -> str:
        """
        Generate deterministic evidence ID.

        Uses SHA-256 hash of canonicalized evidence content.
        """
        # Create canonical representation
        canonical = {
            "source": evidence.source_component,
            "type": evidence.evidence_type,
            "content": evidence.content,
            "questions": sorted(evidence.applicable_questions)
        }

        # Serialize with sorted keys for determinism
        blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(',', ':'))

        # Hash
        digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

        # Create readable ID
        return f"{evidence.source_component}::{evidence.evidence_type}::{digest}"

    def for_question(self, question_id: str) -> List[CanonicalEvidence]:
        """
        Retrieve all evidence applicable to a specific question.

        Args:
            question_id: Question identifier (e.g., 'D1-Q1')

        Returns:
            List of evidence items, sorted by confidence (descending)
        """
        evidence_ids = self.provenance.get(question_id, [])
        evidence_list = [self.store[eid] for eid in evidence_ids if eid in self.store]

        # Sort by confidence (descending) for deterministic ordering
        return sorted(evidence_list, key=lambda e: (-e.confidence, e.metadata.get('evidence_id', '')))

    def for_component(self, component_name: str) -> List[CanonicalEvidence]:
        """
        Retrieve all evidence produced by a specific component.

        Args:
            component_name: Component identifier

        Returns:
            List of evidence items
        """
        evidence_ids = self.component_index.get(component_name, [])
        return [self.store[eid] for eid in evidence_ids if eid in self.store]

    def get_all_questions(self) -> Set[str]:
        """Get set of all questions that have evidence"""
        return set(self.provenance.keys())

    def get_all_components(self) -> Set[str]:
        """Get set of all components that produced evidence"""
        return set(self.component_index.keys())

    def freeze(self):
        """Freeze the registry (no more evidence can be added)"""
        with self._lock:
            self._frozen = True
            logger.info(f"EvidenceRegistry frozen with {len(self.store)} evidence items")

    def is_frozen(self) -> bool:
        """Check if registry is frozen"""
        return self._frozen

    def deterministic_hash(self) -> str:
        """
        Compute deterministic hash of entire registry.

        This hash can be used to verify that two runs with the same input
        produced identical evidence.

        Returns:
            SHA-256 hex digest of registry contents
        """
        # Create ordered list of all evidence
        ordered_evidence = []
        for eid in sorted(self.store.keys()):
            evidence = self.store[eid]
            # Exclude timestamp from hash (keep only structure)
            evidence_dict = evidence.to_dict()
            if 'timestamp' in evidence_dict.get('metadata', {}):
                evidence_dict = evidence_dict.copy()
                evidence_dict['metadata'] = evidence_dict['metadata'].copy()
                del evidence_dict['metadata']['timestamp']
            ordered_evidence.append((eid, evidence_dict))

        # Serialize with deterministic ordering
        blob = json.dumps(ordered_evidence, sort_keys=True, ensure_ascii=True, separators=(',', ':'))

        # Compute hash
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_evidence": len(self.store),
            "total_questions": len(self.provenance),
            "total_components": len(self.component_index),
            "frozen": self._frozen,
            "creation_time": self.creation_time,
            "deterministic_hash": self.deterministic_hash(),
            "evidence_by_component": {
                comp: len(eids) for comp, eids in self.component_index.items()
            },
            "evidence_per_question": {
                qid: len(eids) for qid, eids in self.provenance.items()
            }
        }

    def export_to_json(self, filepath: str):
        """Export registry to JSON file"""
        data = {
            "metadata": {
                "creation_time": self.creation_time,
                "frozen": self._frozen,
                "total_evidence": len(self.store),
                "deterministic_hash": self.deterministic_hash()
            },
            "evidence": {
                eid: evidence.to_dict()
                for eid, evidence in self.store.items()
            },
            "provenance": self.provenance,
            "component_index": self.component_index
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported registry to {filepath}")
    
    def register_evidence(
        self, 
        evidence_id: str, 
        source: str,
        dimension_vote: int, 
        content_type: int,
        risk_level: int, 
        confidence: float
    ):
        """
        Registra evidencia con votos categóricos para agregación Bayesiana.
        
        Args:
            evidence_id: Identificador único
            source: Módulo que genera evidencia
            dimension_vote: Voto de dimensión (0-9 para D1-D10)
            content_type: Tipo de contenido (0-4)
            risk_level: Nivel de riesgo (0-2)
            confidence: Confianza del detector (0-1)
        """
        with self._lock:
            # Crear agregador de dimensión si no existe
            if evidence_id not in self.dimension_aggregators:
                self.dimension_aggregators[evidence_id] = DirichletAggregator(k=10, alpha0=0.5)
            
            # Actualizar con votos ponderados por confianza
            dim_weights = np.zeros(10)
            dim_weights[dimension_vote] = confidence
            self.dimension_aggregators[evidence_id].update_from_weights(dim_weights)
            
            # Actualizar agregadores globales
            self.content_type_aggregator.update_from_labels(np.array([content_type]))
            self.risk_level_aggregator.update_from_labels(np.array([risk_level]))
            
            logger.info(f"Evidencia {evidence_id} registrada desde {source}")
    
    def get_dimension_distribution(self, evidence_id: str) -> Optional[Dict]:
        """
        Obtiene distribución posterior de dimensión para una evidencia.
        
        Args:
            evidence_id: Identificador de evidencia
        
        Returns:
            Dict con {
                'mean': probabilidades,
                'credible_interval': intervalos,
                'max_category': (idx, prob),
                'entropy': incertidumbre,
                'n_votes': número de actualizaciones
            } o None si no existe
        """
        if evidence_id not in self.dimension_aggregators:
            return None
        
        agg = self.dimension_aggregators[evidence_id]
        return {
            'mean': agg.posterior_mean(),
            'credible_interval': agg.credible_interval(level=0.95),
            'max_category': agg.max_probability(),
            'entropy': agg.entropy(),
            'n_votes': agg.n_updates
        }
    
    def get_consensus_dimension(self, evidence_id: str, threshold: float = 0.6) -> Optional[int]:
        """
        Retorna dimensión con consenso si P(D_i) > threshold.
        
        Args:
            evidence_id: Identificador de evidencia
            threshold: Umbral de probabilidad para consenso
        
        Returns:
            Índice de dimensión o None si no hay consenso
        """
        dist = self.get_dimension_distribution(evidence_id)
        if dist is None:
            return None
        
        max_cat, max_prob = dist['max_category']
        if max_prob >= threshold:
            return max_cat
        return None

    def __len__(self) -> int:
        """Return number of evidence items"""
        return len(self.store)

    def __repr__(self) -> str:
        return (f"EvidenceRegistry(evidence={len(self.store)}, "
                f"questions={len(self.provenance)}, "
                f"components={len(self.component_index)}, "
                f"frozen={self._frozen})")

