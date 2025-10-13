"""
Contract package for EvidencePacket protocol buffer definitions.
"""

from .evidence_pb2 import (
    EvidencePacket,
    EvidencePacketBatch,
    PipelineStage,
    RegistryEntry,
)

__all__ = [
    "EvidencePacket",
    "EvidencePacketBatch",
    "RegistryEntry",
    "PipelineStage",
]
