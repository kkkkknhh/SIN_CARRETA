#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvidencePacket Pydantic Wrapper with Canonical JSON and HMAC Signing
=====================================================================

Provides a frozen, immutable Pydantic model wrapping the protobuf EvidencePacket
with canonical JSON serialization and HMAC-SHA256 signing.

Features:
- Immutable (frozen=True)
- Extra fields forbidden (extra="forbid")
- Canonical JSON with sorted keys
- HMAC-SHA256 signing and verification
- Deterministic serialization
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Try to import generated protobuf types (will be available after generate_proto.sh)
try:
    from .evidence_pb2 import EvidencePacket as ProtoEvidencePacket
    from .evidence_pb2 import PipelineStage as ProtoPipelineStage

    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False
    # Fallback for development/testing before proto generation
    ProtoEvidencePacket = None
    ProtoPipelineStage = None


class PipelineStage:
    """Pipeline stage enum matching proto definition."""

    UNSPECIFIED = 0
    SANITIZATION = 1
    PLAN_PROCESSING = 2
    SEGMENTATION = 3
    EMBEDDING = 4
    RESPONSIBILITY = 5
    CONTRADICTION = 6
    MONETARY = 7
    FEASIBILITY = 8
    CAUSAL = 9
    TEORIA = 10
    DAG = 11
    REGISTRY_BUILD = 12
    DECALOGO_LOAD = 13
    DECALOGO_EVAL = 14
    QUESTIONNAIRE_EVAL = 15
    ANSWERS_ASSEMBLY = 16


class EvidencePacketModel(BaseModel):
    """
    Immutable Evidence Packet model with canonical serialization and signing.

    This model wraps the protobuf EvidencePacket with Pydantic validation
    and provides methods for canonical JSON serialization and HMAC signing.

    Attributes:
        schema_version: Schema version for compatibility tracking (e.g., "1.0.0")
        stage: Pipeline stage that produced this evidence
        source_component: Component that produced evidence (e.g., 'feasibility_scorer')
        evidence_type: Type of evidence (e.g., 'baseline_presence')
        content: Evidence content (must be JSON-serializable)
        confidence: Confidence score [0.0, 1.0]
        applicable_questions: Question IDs this evidence answers
        metadata: Additional metadata
        timestamp: ISO 8601 timestamp
        signature: HMAC-SHA256 signature (computed)
        evidence_hash: SHA-256 hash of canonical JSON (computed)
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable
        extra="forbid",  # Reject unknown fields
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    schema_version: str = Field(
        default="1.0.0",
        description="Schema version for compatibility",
        pattern=r"^\d+\.\d+\.\d+$",
    )
    stage: int = Field(description="Pipeline stage (0-16)", ge=0, le=16)
    source_component: str = Field(
        description="Component that produced evidence", min_length=1, max_length=200
    )
    evidence_type: str = Field(
        description="Type of evidence", min_length=1, max_length=200
    )
    content: Dict[str, Any] = Field(description="Evidence content (JSON-serializable)")
    confidence: float = Field(description="Confidence score [0.0, 1.0]", ge=0.0, le=1.0)
    applicable_questions: List[str] = Field(
        description="Question IDs this evidence answers", min_length=0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp",
    )
    signature: Optional[str] = Field(default=None, description="HMAC-SHA256 signature")
    evidence_hash: Optional[str] = Field(
        default=None, description="SHA-256 hash of canonical JSON"
    )

    @field_validator("applicable_questions")
    @classmethod
    def validate_question_ids(cls, v: List[str]) -> List[str]:
        """Validate question ID format (e.g., D1-Q1)."""
        for qid in v:
            if not qid or len(qid) > 50:
                raise ValueError(f"Invalid question ID: {qid}")
        return v

    def canonical_json(self) -> str:
        """
        Generate canonical JSON representation with sorted keys.

        This is the basis for signatures and hashes. It excludes the
        signature and evidence_hash fields to avoid circular dependencies.

        Returns:
            Canonical JSON string with sorted keys and no whitespace variance
        """
        # Create dict excluding signature and hash fields
        data = {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "source_component": self.source_component,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "confidence": self.confidence,
            "applicable_questions": sorted(
                self.applicable_questions
            ),  # Sort for determinism
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

        # Serialize with sorted keys, no whitespace variance
        return json.dumps(
            data, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of canonical JSON representation.

        Returns:
            SHA-256 hex digest
        """
        canonical = self.canonical_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def compute_signature(self, secret: str) -> str:
        """
        Compute HMAC-SHA256 signature of canonical JSON.

        Args:
            secret: HMAC secret key

        Returns:
            HMAC-SHA256 hex digest

        Raises:
            ValueError: If secret is empty
        """
        if not secret:
            raise ValueError("HMAC secret cannot be empty")

        canonical = self.canonical_json()
        return hmac.new(
            secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_signature(self, secret: str) -> bool:
        """
        Verify HMAC-SHA256 signature.

        Args:
            secret: HMAC secret key

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.signature:
            return False

        try:
            expected = self.compute_signature(secret)
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(self.signature, expected)
        except Exception:
            return False

    def with_signature(self, secret: str) -> EvidencePacketModel:
        """
        Create a new packet with signature and hash computed.

        Args:
            secret: HMAC secret key

        Returns:
            New EvidencePacketModel with signature and hash
        """
        # Compute signature and hash
        sig = self.compute_signature(secret)
        hash_val = self.compute_hash()

        # Create new instance with signature and hash
        # We use model_copy with update to create a new frozen instance
        return self.model_copy(update={"signature": sig, "evidence_hash": hash_val})

    def to_proto(self) -> Optional[ProtoEvidencePacket]:
        """
        Convert to protobuf message.

        Returns:
            ProtoEvidencePacket or None if proto not available
        """
        if not PROTO_AVAILABLE or ProtoEvidencePacket is None:
            return None

        packet = ProtoEvidencePacket()
        packet.schema_version = self.schema_version
        packet.stage = self.stage
        packet.source_component = self.source_component
        packet.evidence_type = self.evidence_type
        packet.content_json = json.dumps(self.content, sort_keys=True)
        packet.confidence = self.confidence
        packet.applicable_questions.extend(self.applicable_questions)
        packet.metadata_json = json.dumps(self.metadata, sort_keys=True)
        packet.timestamp = self.timestamp
        if self.signature:
            packet.signature = self.signature
        if self.evidence_hash:
            packet.evidence_hash = self.evidence_hash

        return packet

    @classmethod
    def from_proto(cls, proto: ProtoEvidencePacket) -> EvidencePacketModel:
        """
        Create from protobuf message.

        Args:
            proto: ProtoEvidencePacket

        Returns:
            EvidencePacketModel
        """
        content = json.loads(proto.content_json) if proto.content_json else {}
        metadata = json.loads(proto.metadata_json) if proto.metadata_json else {}

        return cls(
            schema_version=proto.schema_version,
            stage=proto.stage,
            source_component=proto.source_component,
            evidence_type=proto.evidence_type,
            content=content,
            confidence=proto.confidence,
            applicable_questions=list(proto.applicable_questions),
            metadata=metadata,
            timestamp=proto.timestamp,
            signature=proto.signature if proto.signature else None,
            evidence_hash=proto.evidence_hash if proto.evidence_hash else None,
        )


def get_hmac_secret() -> str:
    """
    Get HMAC secret from environment variable.

    Returns:
        HMAC secret from EVIDENCE_HMAC_SECRET env var

    Raises:
        ValueError: If environment variable is not set
    """
    secret = os.environ.get("EVIDENCE_HMAC_SECRET")
    if not secret:
        raise ValueError(
            "EVIDENCE_HMAC_SECRET environment variable not set. "
            "Please set it to a secure random string (minimum 32 characters)."
        )
    if len(secret) < 32:
        raise ValueError(
            f"EVIDENCE_HMAC_SECRET is too short ({len(secret)} chars). "
            "Minimum 32 characters required for security."
        )
    return secret
