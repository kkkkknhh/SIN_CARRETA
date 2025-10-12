#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration script for legacy IDs to P-D-Q canonical notation.

Handles three legacy cases:
  A) "D#-Q#" (no policy) → infer P# from context
  B) "P#-Q#" (no dimension) → infer D# from context
  C) "Q#" (neither) → infer both P# and D#

Migration strategies:
  1. section: Extract from document section headers
  2. semantic: Use semantic similarity with policy descriptions
  3. fallback: Use default_policy_if_unknown from manifest
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Canonical patterns
_RX_CANONICAL = re.compile(r"^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$")
_RX_LEGACY_DQ = re.compile(r"^D([1-6])-Q([1-9][0-9]*)$")
_RX_LEGACY_PQ = re.compile(r"^P(10|[1-9])-Q([1-9][0-9]*)$")
_RX_LEGACY_Q = re.compile(r"^Q([1-9][0-9]*)$")


class MigrationLog:
    """Track migration entries."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(
        self,
        original_id: str,
        normalized_id: str,
        rubric_key: str,
        strategy: str,
        confidence: float,
        notes: str = "",
    ) -> None:
        """Add a migration log entry."""
        self.entries.append(
            {
                "original_id": original_id,
                "normalized_id": normalized_id,
                "rubric_key": rubric_key,
                "strategy": strategy,
                "confidence": confidence,
                "notes": notes,
            }
        )

    def save(self, output_path: Path) -> None:
        """Save migration log to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_migrations": len(self.entries), "migrations": self.entries},
                f,
                ensure_ascii=False,
                indent=2,
            )


class LegacyIDMigrator:
    """Migrate legacy IDs to canonical P-D-Q notation."""

    def __init__(
        self, manifest_path: Optional[Path] = None, bundle_path: Optional[Path] = None
    ):
        """Initialize migrator with configuration."""
        self.default_policy = "P4"
        self.min_confidence = 0.80
        self.log = MigrationLog()

        # Load manifest if provided
        if manifest_path and manifest_path.exists():
            self._load_manifest(manifest_path)

        # Load bundle for semantic mapping if provided
        self.bundle_questions: Dict[str, Dict[str, Any]] = {}
        if bundle_path and bundle_path.exists():
            self._load_bundle(bundle_path)

    def _load_manifest(self, path: Path) -> None:
        """Load questionnaire manifest."""
        import yaml

        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            rules = manifest.get("rules", {})
            self.default_policy = rules.get("default_policy_if_unknown", "P4")
            self.min_confidence = rules.get("min_confidence_for_auto_inference", 0.80)
        except Exception as e:
            print(f"WARNING: Could not load manifest: {e}")

    def _load_bundle(self, path: Path) -> None:
        """Load decalogo bundle for semantic mapping."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            for q in bundle.get("questions", []):
                qid = q.get("question_unique_id")
                if qid:
                    self.bundle_questions[qid] = q
        except Exception as e:
            print(f"WARNING: Could not load bundle: {e}")

    def migrate(
        self, legacy_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, float]:
        """
        Migrate a legacy ID to canonical format.

        Returns: (normalized_id, rubric_key, confidence)
        Raises: ValueError if migration fails
        """
        # Already canonical?
        if _RX_CANONICAL.match(legacy_id):
            rubric_key = self._extract_rubric_key(legacy_id)
            return legacy_id, rubric_key, 1.0

        context = context or {}

        # Case A: D#-Q# format
        match_dq = _RX_LEGACY_DQ.match(legacy_id)
        if match_dq:
            return self._migrate_dq(match_dq, context)

        # Case B: P#-Q# format
        match_pq = _RX_LEGACY_PQ.match(legacy_id)
        if match_pq:
            return self._migrate_pq(match_pq, context)

        # Case C: Q# format
        match_q = _RX_LEGACY_Q.match(legacy_id)
        if match_q:
            return self._migrate_q(match_q, context)

        raise ValueError(
            f"ERROR_QID_NORMALIZATION: cannot standardize legacy id '{legacy_id}'"
        )

    def _extract_rubric_key(self, canonical_id: str) -> str:
        """Extract D#-Q# from P#-D#-Q#."""
        parts = canonical_id.split("-")
        if len(parts) == 3:
            return f"{parts[1]}-{parts[2]}"
        raise ValueError(f"Invalid canonical ID: {canonical_id}")

    def _migrate_dq(
        self, match: re.Match, context: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        """Migrate D#-Q# format by inferring P#."""
        dim = f"D{match.group(1)}"
        q_num = match.group(2)
        rubric_key = f"{dim}-Q{q_num}"

        # Try to infer policy from context
        policy, confidence, strategy = self._infer_policy(context)

        if confidence < self.min_confidence:
            raise ValueError(
                f"ERROR_QID_NORMALIZATION: cannot infer policy for '{rubric_key}' "
                f"(confidence {confidence:.2f} < {self.min_confidence})"
            )

        normalized_id = f"{policy}-{dim}-Q{q_num}"
        self.log.add(
            original_id=f"{dim}-Q{q_num}",
            normalized_id=normalized_id,
            rubric_key=rubric_key,
            strategy=strategy,
            confidence=confidence,
            notes=f"Inferred policy from {strategy}",
        )

        return normalized_id, rubric_key, confidence

    def _migrate_pq(
        self, match: re.Match, context: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        """Migrate P#-Q# format by inferring D#."""
        policy = f"P{match.group(1)}"
        q_num = match.group(2)

        # Try to infer dimension from context
        dim, confidence, strategy = self._infer_dimension(policy, q_num, context)

        if confidence < self.min_confidence:
            raise ValueError(
                f"ERROR_QID_NORMALIZATION: cannot infer dimension for '{policy}-Q{q_num}' "
                f"(confidence {confidence:.2f} < {self.min_confidence})"
            )

        normalized_id = f"{policy}-{dim}-Q{q_num}"
        rubric_key = f"{dim}-Q{q_num}"
        self.log.add(
            original_id=f"{policy}-Q{q_num}",
            normalized_id=normalized_id,
            rubric_key=rubric_key,
            strategy=strategy,
            confidence=confidence,
            notes=f"Inferred dimension from {strategy}",
        )

        return normalized_id, rubric_key, confidence

    def _migrate_q(
        self, match: re.Match, context: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        """Migrate Q# format by inferring both P# and D#."""
        q_num = match.group(1)

        # Try to infer policy
        policy, p_conf, p_strategy = self._infer_policy(context)

        # Try to infer dimension
        dim, d_conf, d_strategy = self._infer_dimension(policy, q_num, context)

        # Combined confidence
        confidence = min(p_conf, d_conf)

        if confidence < self.min_confidence:
            raise ValueError(
                f"ERROR_QID_NORMALIZATION: cannot infer policy and dimension for 'Q{q_num}' "
                f"(confidence {confidence:.2f} < {self.min_confidence})"
            )

        normalized_id = f"{policy}-{dim}-Q{q_num}"
        rubric_key = f"{dim}-Q{q_num}"
        self.log.add(
            original_id=f"Q{q_num}",
            normalized_id=normalized_id,
            rubric_key=rubric_key,
            strategy=f"{p_strategy}+{d_strategy}",
            confidence=confidence,
            notes=f"Inferred policy from {p_strategy}, dimension from {d_strategy}",
        )

        return normalized_id, rubric_key, confidence

    def _infer_policy(self, context: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Infer policy from context.
        Returns: (policy, confidence, strategy)
        """
        # Strategy 1: Extract from section header
        section = context.get("section", "")
        if section:
            for p_num in range(1, 11):
                policy_id = f"P{p_num}" if p_num < 10 else "P10"
                if policy_id in section.upper() or f"PUNTO {p_num}" in section.upper():
                    return policy_id, 0.95, "section"

        # Strategy 2: Semantic matching (not implemented here, would need embeddings)
        # For now, skip to fallback

        # Strategy 3: Fallback to default
        return self.default_policy, 0.60, "fallback"

    def _infer_dimension(
        self, policy: str, q_num: str, context: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """
        Infer dimension from context.
        Returns: (dimension, confidence, strategy)
        """
        # Strategy 1: Map by question number range (assuming 5 questions per dimension)
        q_int = int(q_num)
        dim_index = (
            (q_int - 1) % 30
        ) // 5 + 1  # 30 questions per policy, 5 per dimension
        if 1 <= dim_index <= 6:
            return f"D{dim_index}", 0.85, "question_range"

        # Strategy 2: Look up in bundle
        _candidate_id = f"{policy}-D1-Q{q_num}"
        for dim_num in range(1, 7):
            test_id = f"{policy}-D{dim_num}-Q{q_num}"
            if test_id in self.bundle_questions:
                return f"D{dim_num}", 0.90, "bundle_lookup"

        # Fallback: default to D1
        return "D1", 0.50, "fallback"


def main():
    """Main migration routine."""
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / "config" / "QUESTIONNAIRE_MANIFEST.yaml"
    bundle_path = repo_root / "bundles" / "decalogo_bundle.json"
    output_path = repo_root / "output" / "migration_log.json"

    print("=" * 70)
    print("Legacy ID Migration Tool")
    print("=" * 70)

    migrator = LegacyIDMigrator(manifest_path, bundle_path)

    # Test cases from spec
    test_cases = [
        ("D4-Q3", {"section": "P8"}),  # Legacy case A
        ("P3-D2-Q4", {}),  # Already canonical
        ("P2-Q5", {}),  # Legacy case B (should fail without D)
        ("Q12", {"section": "P6"}),  # Legacy case C
    ]

    print("\nTesting migration scenarios:")
    for legacy_id, ctx in test_cases:
        print(f"\n  Input: {legacy_id}")
        print(f"  Context: {ctx}")
        try:
            normalized, rubric_key, conf = migrator.migrate(legacy_id, ctx)
            print(f"  ✓ Output: {normalized}")
            print(f"    Rubric key: {rubric_key}")
            print(f"    Confidence: {conf:.2f}")
        except ValueError as e:
            print(f"  ✗ {e}")

    # Save log
    output_path.parent.mkdir(exist_ok=True)
    migrator.log.save(output_path)
    print(f"\n✓ Migration log saved to: {output_path}")
    print(f"  Total migrations: {len(migrator.log.entries)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
