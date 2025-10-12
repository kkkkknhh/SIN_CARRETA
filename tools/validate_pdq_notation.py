#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical P-D-Q ID Format Validator for CI.

Validates that all IDs in code, config, and bundles follow canonical notation:
  - P# = Policy (P1..P10)
  - D# = Dimension (D1..D6)
  - Q# = Question (Q1+)
  - question_unique_id = "P#-D#-Q#"
  - rubric_key = "D#-Q#"

Exit code 0 = all valid, 1 = violations found
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Canonical patterns
_RX_CANONICAL_UID = re.compile(r"^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$")
_RX_RUBRIC_KEY = re.compile(r"^D[1-6]-Q[1-9][0-9]*$")
_RX_POLICY = re.compile(r"^P(10|[1-9])$")
_RX_DIM = re.compile(r"^D[1-6]$")


class ValidationResult:
    """Tracks validation results."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.files_checked: int = 0
        self.ids_validated: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("Validation Summary")
        print("=" * 70)
        print(f"Files checked: {self.files_checked}")
        print(f"IDs validated: {self.ids_validated}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n❌ ERRORS:")
            for err in self.errors:
                print(f"  - {err}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warn in self.warnings:
                print(f"  - {warn}")

        if self.is_valid():
            print("\n✅ All validations passed!")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} error(s)")


def validate_rubric(path: Path, result: ValidationResult) -> None:
    """Validate RUBRIC_SCORING.json."""
    print(f"\nValidating rubric: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        result.add_error(f"Rubric: Failed to load {path}: {e}")
        return

    result.files_checked += 1

    if "weights" not in data:
        result.add_error("Rubric: Missing 'weights' field")
        return

    weights = data["weights"]
    total_keys = len(weights)
    invalid_keys: List[str] = []

    for key in weights.keys():
        result.ids_validated += 1
        if not _RX_RUBRIC_KEY.match(key):
            invalid_keys.append(key)

    if invalid_keys:
        result.add_error(
            f"Rubric: {len(invalid_keys)}/{total_keys} invalid rubric keys (expected D#-Q#): "
            f"{', '.join(invalid_keys[:10])}{' ...' if len(invalid_keys) > 10 else ''}"
        )

    # Check for expected count (10 policies × 6 dimensions × 5 questions = 300)
    expected = 300
    if total_keys != expected:
        result.add_warning(f"Rubric: Expected {expected} keys, found {total_keys}")

    print(f"  ✓ Validated {total_keys} rubric keys")


def validate_bundle(path: Path, result: ValidationResult) -> None:
    """Validate decalogo_bundle.json."""
    print(f"\nValidating bundle: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        result.add_error(f"Bundle: Failed to load {path}: {e}")
        return

    result.files_checked += 1

    # Validate policies
    if "policies" not in data:
        result.add_error("Bundle: Missing 'policies' field")
    else:
        policies = data["policies"]
        if len(policies) != 10:
            result.add_error(f"Bundle: Expected 10 policies, found {len(policies)}")
        for p in policies:
            result.ids_validated += 1
            if not _RX_POLICY.match(p):
                result.add_error(f"Bundle: Invalid policy format: {p}")

    # Validate dimensions
    if "dimensions" not in data:
        result.add_error("Bundle: Missing 'dimensions' field")
    else:
        dims = data["dimensions"]
        if len(dims) != 6:
            result.add_error(f"Bundle: Expected 6 dimensions, found {len(dims)}")
        for d in dims:
            result.ids_validated += 1
            if not _RX_DIM.match(d):
                result.add_error(f"Bundle: Invalid dimension format: {d}")

    # Validate questions
    if "questions" not in data:
        result.add_error("Bundle: Missing 'questions' field")
    else:
        questions = data["questions"]
        invalid_uids: List[str] = []
        invalid_rubrics: List[str] = []

        for i, q in enumerate(questions):
            # Check question_unique_id
            if "question_unique_id" not in q:
                result.add_error(f"Bundle: Question {i}: missing question_unique_id")
            else:
                uid = q["question_unique_id"]
                result.ids_validated += 1
                if not _RX_CANONICAL_UID.match(uid):
                    invalid_uids.append(uid)

            # Check rubric_key
            if "rubric_key" not in q:
                result.add_error(f"Bundle: Question {i}: missing rubric_key")
            else:
                rk = q["rubric_key"]
                result.ids_validated += 1
                if not _RX_RUBRIC_KEY.match(rk):
                    invalid_rubrics.append(rk)

            # Check consistency between UID and rubric_key
            if "question_unique_id" in q and "rubric_key" in q:
                uid = q["question_unique_id"]
                rk = q["rubric_key"]
                # Extract D#-Q# from P#-D#-Q#
                parts = uid.split("-")
                if len(parts) == 3:
                    expected_rk = f"{parts[1]}-{parts[2]}"
                    if expected_rk != rk:
                        result.add_error(
                            f"Bundle: Question {i}: rubric_key '{rk}' doesn't match "
                            f"question_unique_id '{uid}' (expected '{expected_rk}')"
                        )

        if invalid_uids:
            result.add_error(
                f"Bundle: {len(invalid_uids)} invalid question_unique_id formats: "
                f"{', '.join(invalid_uids[:5])}{' ...' if len(invalid_uids) > 5 else ''}"
            )

        if invalid_rubrics:
            result.add_error(
                f"Bundle: {len(invalid_rubrics)} invalid rubric_key formats: "
                f"{', '.join(invalid_rubrics[:5])}{' ...' if len(invalid_rubrics) > 5 else ''}"
            )

        print(f"  ✓ Validated {len(questions)} questions")


def check_code_for_violations(repo_root: Path, result: ValidationResult) -> None:
    """Check Python code for non-canonical ID patterns."""
    print("\nChecking code for canonical notation violations...")

    # Patterns that should NOT appear (legacy patterns)
    bad_patterns = [
        (
            r'question_id\s*=\s*["\']D\d+-Q\d+["\']',
            "D#-Q# without P# (should be P#-D#-Q#)",
        ),
        (
            r'question_id\s*=\s*["\']P\d+-Q\d+["\']',
            "P#-Q# without D# (should be P#-D#-Q#)",
        ),
        (r'question_id\s*=\s*["\']Q\d+["\']', "Q# alone (should be P#-D#-Q#)"),
    ]

    # Files to check
    py_files = list(repo_root.glob("*.py"))
    py_files.extend(repo_root.glob("**/*.py"))

    # Exclude certain directories
    exclude_dirs = {
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
    }
    py_files = [f for f in py_files if not any(ex in f.parts for ex in exclude_dirs)]

    violations: List[Tuple[Path, int, str, str]] = []

    for py_file in py_files[:100]:  # Limit to first 100 files for performance
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            result.files_checked += 1

            for line_no, line in enumerate(lines, 1):
                for pattern, description in bad_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append((py_file, line_no, description, line.strip()))
        except Exception:
            continue

    if violations:
        result.add_warning(
            f"Code: Found {len(violations)} potential non-canonical ID patterns"
        )
        for fpath, lineno, desc, line in violations[:10]:
            result.add_warning(f"  {fpath.name}:{lineno}: {desc}")
            result.add_warning(f"    > {line[:80]}")


def main():
    """Main validation routine."""
    repo_root = Path(__file__).parent.parent
    result = ValidationResult()

    print("=" * 70)
    print("P-D-Q Canonical Notation Validator")
    print("=" * 70)

    # Validate rubric
    rubric_path = repo_root / "config" / "RUBRIC_SCORING.json"
    if rubric_path.exists():
        validate_rubric(rubric_path, result)
    else:
        result.add_warning(f"Rubric not found: {rubric_path}")

    # Validate bundle
    bundle_path = repo_root / "bundles" / "decalogo_bundle.json"
    if bundle_path.exists():
        validate_bundle(bundle_path, result)
    else:
        result.add_error(f"Bundle not found: {bundle_path}")

    # Check code
    check_code_for_violations(repo_root, result)

    # Print summary
    result.print_summary()

    return 0 if result.is_valid() else 1


if __name__ == "__main__":
    sys.exit(main())
