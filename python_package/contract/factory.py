#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Packet Factory
========================

Factory functions to create signed EvidencePacket instances with validation.
"""

from typing import Any, Dict, List

from .evidence_proto_gen import EvidencePacketModel, PipelineStage, get_hmac_secret


def create_evidence_packet(
    stage: int,
    source_component: str,
    evidence_type: str,
    content: Dict[str, Any],
    confidence: float,
    applicable_questions: List[str],
    metadata: Dict[str, Any] | None = None,
    schema_version: str = "1.0.0",
    sign: bool = True,
) -> EvidencePacketModel:
    """
    Create a new EvidencePacket with optional signing.

    Args:
        stage: Pipeline stage (0-16)
        source_component: Component that produced evidence
        evidence_type: Type of evidence
        content: Evidence content (must be JSON-serializable)
        confidence: Confidence score [0.0, 1.0]
        applicable_questions: Question IDs this evidence answers
        metadata: Additional metadata (optional)
        schema_version: Schema version (default: "1.0.0")
        sign: Whether to sign the packet (default: True)

    Returns:
        Signed EvidencePacketModel

    Raises:
        ValueError: If validation fails or HMAC secret not set (when sign=True)
    """
    # Create packet without signature
    packet = EvidencePacketModel(
        schema_version=schema_version,
        stage=stage,
        source_component=source_component,
        evidence_type=evidence_type,
        content=content,
        confidence=confidence,
        applicable_questions=applicable_questions,
        metadata=metadata or {},
    )

    # Sign if requested
    if sign:
        secret = get_hmac_secret()
        packet = packet.with_signature(secret)

    return packet


def create_sanitization_evidence(
    content: Dict[str, Any],
    confidence: float,
    applicable_questions: List[str],
    source_component: str = "sanitization",
    evidence_type: str = "sanitized_text",
    metadata: Dict[str, Any] | None = None,
) -> EvidencePacketModel:
    """Create evidence from sanitization stage."""
    return create_evidence_packet(
        stage=PipelineStage.SANITIZATION,
        source_component=source_component,
        evidence_type=evidence_type,
        content=content,
        confidence=confidence,
        applicable_questions=applicable_questions,
        metadata=metadata,
    )


def create_feasibility_evidence(
    content: Dict[str, Any],
    confidence: float,
    applicable_questions: List[str],
    source_component: str = "feasibility_scorer",
    evidence_type: str = "baseline_presence",
    metadata: Dict[str, Any] | None = None,
) -> EvidencePacketModel:
    """Create evidence from feasibility scoring stage."""
    return create_evidence_packet(
        stage=PipelineStage.FEASIBILITY,
        source_component=source_component,
        evidence_type=evidence_type,
        content=content,
        confidence=confidence,
        applicable_questions=applicable_questions,
        metadata=metadata,
    )


def create_decalogo_evidence(
    content: Dict[str, Any],
    confidence: float,
    applicable_questions: List[str],
    source_component: str = "decalogo_evaluator",
    evidence_type: str = "decalogo_score",
    metadata: Dict[str, Any] | None = None,
) -> EvidencePacketModel:
    """Create evidence from decalogo evaluation stage."""
    return create_evidence_packet(
        stage=PipelineStage.DECALOGO_EVAL,
        source_component=source_component,
        evidence_type=evidence_type,
        content=content,
        confidence=confidence,
        applicable_questions=applicable_questions,
        metadata=metadata,
    )


def validate_packet(packet: EvidencePacketModel, secret: str | None = None) -> bool:
    """
    Validate an EvidencePacket.

    Args:
        packet: EvidencePacket to validate
        secret: HMAC secret for signature verification (uses env var if None)

    Returns:
        True if valid, False otherwise
    """
    # Basic validation is done by Pydantic
    # Verify signature if present
    if packet.signature:
        if secret is None:
            try:
                secret = get_hmac_secret()
            except ValueError:
                return False
        return packet.verify_signature(secret)

    return True
