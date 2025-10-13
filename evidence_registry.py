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
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from threading import Lock
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceProvenance:
    """
    Provenance metadata for evidence tracking across detector stages.
    
    Attributes:
        detector_type: Type of detector that produced evidence (e.g., 'monetary', 'responsibility')
        stage_number: Detector stage number (1-12)
        source_text_location: Location in source text (page, line, character offsets)
        execution_timestamp: ISO 8601 timestamp of evidence generation
        quality_metrics: Evidence quality scores and metadata
    """
    detector_type: str
    stage_number: int
    source_text_location: Dict[str, Any]
    execution_timestamp: str
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate provenance metadata"""
        if not 1 <= self.stage_number <= 12:
            raise ValueError(f"Stage number must be 1-12, got {self.stage_number}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "detector_type": self.detector_type,
            "stage_number": self.stage_number,
            "source_text_location": self.source_text_location,
            "execution_timestamp": self.execution_timestamp,
            "quality_metrics": self.quality_metrics
        }


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
        provenance: Provenance tracking information
    """
    source_component: str
    evidence_type: str
    content: Any
    confidence: float
    applicable_questions: tuple  # Tuple for immutability
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[EvidenceProvenance] = None

    def __post_init__(self):
        """Validate evidence at creation"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if not isinstance(self.applicable_questions, tuple):
            # Convert to tuple if list was passed
            object.__setattr__(self, 'applicable_questions', tuple(self.applicable_questions))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "source_component": self.source_component,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "confidence": self.confidence,
            "applicable_questions": list(self.applicable_questions),
            "metadata": self.metadata
        }
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        return result


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

        logger.info("EvidenceRegistry initialized")

    def register(
        self,
        source_component: str,
        evidence_type: str,
        content: Any,
        confidence: float,
        applicable_questions: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[EvidenceProvenance] = None
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
            provenance: Optional provenance tracking information

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
                metadata=meta,
                provenance=provenance
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
                metadata=meta,
                provenance=provenance
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

            logger.debug(
                "Registered evidence %s from %s for %s questions",
                evidence_id,
                source_component,
                len(applicable_questions)
            )

            return evidence_id

    @staticmethod
    def _generate_evidence_id(evidence: CanonicalEvidence) -> str:
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
            logger.info("EvidenceRegistry frozen with %s evidence items", len(self.store))

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

        logger.info("Exported registry to %s", filepath)

    def __len__(self) -> int:
        """Return number of evidence items"""
        return len(self.store)

    def validate_evidence_counts(
        self,
        all_question_ids: List[str],
        min_evidence_threshold: int = 3
    ) -> Dict[str, Any]:
        """
        Validate that all questions meet minimum evidence count threshold.
        
        Args:
            all_question_ids: List of all question IDs to validate (e.g., 300 questions)
            min_evidence_threshold: Minimum number of evidence sources required (default: 3)
        
        Returns:
            Dictionary containing:
                - valid: Whether all questions meet threshold
                - total_questions: Total number of questions validated
                - questions_below_threshold: List of question IDs that fail validation
                - evidence_summary: Per-question evidence counts and stage breakdown
                - validation_timestamp: ISO 8601 timestamp of validation
        """
        questions_below_threshold = []
        evidence_summary = {}
        
        for question_id in all_question_ids:
            # Get all evidence for this question
            evidence_list = self.for_question(question_id)
            evidence_count = len(evidence_list)
            
            # Track stage contributions
            stage_contributions = {}
            missing_stages = set(range(1, 13))  # Stages 1-12
            
            for evidence in evidence_list:
                if evidence.provenance:
                    stage_num = evidence.provenance.stage_number
                    if stage_num not in stage_contributions:
                        stage_contributions[stage_num] = []
                    stage_contributions[stage_num].append({
                        "evidence_id": evidence.metadata.get("evidence_id"),
                        "detector_type": evidence.provenance.detector_type,
                        "confidence": evidence.confidence,
                        "source_component": evidence.source_component
                    })
                    missing_stages.discard(stage_num)
            
            # Build evidence summary for this question
            evidence_summary[question_id] = {
                "evidence_count": evidence_count,
                "meets_threshold": evidence_count >= min_evidence_threshold,
                "stage_contributions": stage_contributions,
                "missing_stages": sorted(missing_stages),
                "evidence_sources": [
                    {
                        "evidence_id": e.metadata.get("evidence_id"),
                        "source_component": e.source_component,
                        "evidence_type": e.evidence_type,
                        "confidence": e.confidence,
                        "detector_type": e.provenance.detector_type if e.provenance else None,
                        "stage_number": e.provenance.stage_number if e.provenance else None,
                        "execution_timestamp": e.provenance.execution_timestamp if e.provenance else None,
                        "quality_metrics": e.provenance.quality_metrics if e.provenance else {}
                    }
                    for e in evidence_list
                ]
            }
            
            # Track questions below threshold
            if evidence_count < min_evidence_threshold:
                questions_below_threshold.append(question_id)
                logger.warning(
                    "Question %s has only %d evidence sources (threshold: %d). "
                    "Contributing stages: %s. Missing stages: %s",
                    question_id,
                    evidence_count,
                    min_evidence_threshold,
                    sorted(stage_contributions.keys()),
                    sorted(missing_stages)
                )
        
        validation_result = {
            "valid": len(questions_below_threshold) == 0,
            "total_questions": len(all_question_ids),
            "questions_meeting_threshold": len(all_question_ids) - len(questions_below_threshold),
            "questions_below_threshold": questions_below_threshold,
            "evidence_summary": evidence_summary,
            "validation_timestamp": datetime.utcnow().isoformat() + "Z",
            "min_evidence_threshold": min_evidence_threshold,
            "stage_coverage_summary": self._compute_stage_coverage(evidence_summary)
        }
        
        logger.info(
            "Evidence validation complete: %d/%d questions meet threshold (%d evidence minimum)",
            validation_result["questions_meeting_threshold"],
            validation_result["total_questions"],
            min_evidence_threshold
        )
        
        return validation_result
    
    def _compute_stage_coverage(self, evidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute stage coverage statistics across all questions.
        
        Args:
            evidence_summary: Per-question evidence summary
        
        Returns:
            Dictionary with stage coverage statistics
        """
        stage_counts = {stage: 0 for stage in range(1, 13)}
        questions_per_stage = {stage: [] for stage in range(1, 13)}
        
        for question_id, summary in evidence_summary.items():
            for stage_num in summary.get("stage_contributions", {}).keys():
                stage_counts[stage_num] += 1
                questions_per_stage[stage_num].append(question_id)
        
        return {
            "evidence_count_per_stage": stage_counts,
            "questions_per_stage": questions_per_stage,
            "stages_with_evidence": [s for s, count in stage_counts.items() if count > 0],
            "stages_without_evidence": [s for s, count in stage_counts.items() if count == 0]
        }
    
    def export_validation_results(
        self,
        validation_result: Dict[str, Any],
        filepath: str
    ):
        """
        Export validation results to JSON file with full traceability.
        
        Args:
            validation_result: Result from validate_evidence_counts()
            filepath: Path to output JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2, ensure_ascii=False)
        
        logger.info("Exported validation results to %s", filepath)

    def __repr__(self) -> str:
        return (f"EvidenceRegistry(evidence={len(self.store)}, "
                f"questions={len(self.provenance)}, "
                f"components={len(self.component_index)}, "
                f"frozen={self._frozen})")

