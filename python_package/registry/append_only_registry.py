#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append-Only Evidence Registry with Chained Hashes
==================================================

Provides a file-based append-only registry with cryptographic chaining
for evidence integrity verification.

Features:
- Append-only operations (no modification or deletion)
- Chained hashes (prev_hash + current canonical JSON -> entry_hash)
- Integrity verification
- CLI for verification
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from contract.evidence_proto_gen import EvidencePacketModel

# Import the evidence model
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RegistryEntry:
    """
    Entry in the append-only registry.

    Attributes:
        prev_hash: Hash of the previous entry (genesis entry has "0000...")
        packet: The evidence packet
        entry_hash: Hash of this entry (prev_hash + packet canonical JSON)
        sequence_number: Entry sequence number (starting from 0)
    """

    prev_hash: str
    packet: EvidencePacketModel
    entry_hash: str
    sequence_number: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prev_hash": self.prev_hash,
            "packet": json.loads(self.packet.model_dump_json()),
            "entry_hash": self.entry_hash,
            "sequence_number": self.sequence_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegistryEntry:
        """Create from dictionary."""
        packet = EvidencePacketModel.model_validate(data["packet"])
        return cls(
            prev_hash=data["prev_hash"],
            packet=packet,
            entry_hash=data["entry_hash"],
            sequence_number=data["sequence_number"],
        )


class AppendOnlyRegistry:
    """
    Append-only evidence registry with chained hashes.

    The registry maintains a chain of evidence entries where each entry
    contains a hash that depends on the previous entry's hash, creating
    a tamper-evident chain similar to a blockchain.
    """

    GENESIS_HASH = "0" * 64  # SHA-256 all zeros

    def __init__(self, registry_path: Path | str):
        """
        Initialize registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.entries: List[RegistryEntry] = []

        # Load existing registry if it exists
        if self.registry_path.exists():
            self._load()

    def _compute_entry_hash(self, prev_hash: str, packet: EvidencePacketModel) -> str:
        """
        Compute hash for a registry entry.

        Args:
            prev_hash: Hash of previous entry
            packet: Evidence packet

        Returns:
            SHA-256 hash of (prev_hash + packet canonical JSON)
        """
        canonical = packet.canonical_json()
        combined = f"{prev_hash}{canonical}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        """Load registry from file."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entries = [RegistryEntry.from_dict(entry) for entry in data["entries"]]

    def _save(self) -> None:
        """Save registry to file."""
        # Ensure directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0.0",
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }

        # Write atomically using temp file
        temp_path = self.registry_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        # Atomic rename
        temp_path.replace(self.registry_path)

    def append(self, packet: EvidencePacketModel) -> RegistryEntry:
        """
        Append a new evidence packet to the registry.

        Args:
            packet: Evidence packet to append

        Returns:
            Created registry entry
        """
        # Get previous hash
        prev_hash = self.entries[-1].entry_hash if self.entries else self.GENESIS_HASH

        # Compute entry hash
        entry_hash = self._compute_entry_hash(prev_hash, packet)

        # Create entry
        entry = RegistryEntry(
            prev_hash=prev_hash,
            packet=packet,
            entry_hash=entry_hash,
            sequence_number=len(self.entries),
        )

        # Append and save
        self.entries.append(entry)
        self._save()

        return entry

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """
        Verify the integrity of the entire chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.entries:
            return True, None

        # Check genesis entry
        first_entry = self.entries[0]
        if first_entry.prev_hash != self.GENESIS_HASH:
            return (
                False,
                f"First entry prev_hash should be genesis hash, got {first_entry.prev_hash}",
            )

        # Verify each entry's hash
        for i, entry in enumerate(self.entries):
            # Check sequence number
            if entry.sequence_number != i:
                return (
                    False,
                    f"Entry {i}: sequence_number mismatch (expected {i}, got {entry.sequence_number})",
                )

            # Recompute hash
            expected_hash = self._compute_entry_hash(entry.prev_hash, entry.packet)
            if entry.entry_hash != expected_hash:
                return (
                    False,
                    f"Entry {i}: hash mismatch (expected {expected_hash}, got {entry.entry_hash})",
                )

            # Check chain linkage (except for first entry)
            if i > 0:
                prev_entry = self.entries[i - 1]
                if entry.prev_hash != prev_entry.entry_hash:
                    return (
                        False,
                        f"Entry {i}: chain broken (prev_hash doesn't match previous entry_hash)",
                    )

        return True, None

    def get_entry(self, sequence_number: int) -> Optional[RegistryEntry]:
        """Get entry by sequence number."""
        if 0 <= sequence_number < len(self.entries):
            return self.entries[sequence_number]
        return None

    def get_all_entries(self) -> List[RegistryEntry]:
        """Get all entries."""
        return self.entries.copy()

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "entry_count": len(self.entries),
            "registry_path": str(self.registry_path),
            "genesis_hash": self.GENESIS_HASH,
            "latest_hash": self.entries[-1].entry_hash if self.entries else None,
        }


def verify_registry(registry_path: str) -> int:
    """
    Verify registry integrity (CLI command).

    Args:
        registry_path: Path to registry JSON file

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    try:
        registry = AppendOnlyRegistry(registry_path)

        if not registry.entries:
            print(f"✓ Registry is empty: {registry_path}")
            return 0

        is_valid, error = registry.verify_chain()

        if is_valid:
            stats = registry.get_stats()
            print(f"✓ Registry verification PASSED")
            print(f"  Entries: {stats['entry_count']}")
            print(f"  Latest hash: {stats['latest_hash'][:16]}...")
            print(f"  File: {registry_path}")
            return 0
        else:
            print(f"✗ Registry verification FAILED")
            print(f"  Error: {error}")
            print(f"  File: {registry_path}")
            return 1

    except Exception as e:
        print(f"✗ Registry verification ERROR: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python append_only_registry.py <registry_path>")
        print(
            "       python -m python_package.registry.append_only_registry <registry_path>"
        )
        sys.exit(1)

    registry_path = sys.argv[1]
    exit_code = verify_registry(registry_path)
    sys.exit(exit_code)
