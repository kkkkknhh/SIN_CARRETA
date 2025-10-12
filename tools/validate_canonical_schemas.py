#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate JSON files against their schemas.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from jsonschema import ValidationError, validate

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("WARNING: jsonschema not installed, using basic validation")


def basic_validate_pattern(value: str, pattern: str) -> bool:
    """Basic pattern validation without jsonschema."""
    import re

    return bool(re.match(pattern, value))


def basic_validate_evidence(data: Dict[str, Any]) -> List[str]:
    """Basic validation for evidence schema."""
    errors = []
    required = ["evidence_id", "question_unique_id", "content", "confidence", "stage"]

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if (
        "question_unique_id" in data
        and not basic_validate_pattern(
        data["question_unique_id"], r"^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$"
    )
    ):
        errors.append(
            f"Invalid question_unique_id format: {data['question_unique_id']}"
        )

    if "content" in data:
        content = data["content"]
        if (
            "rubric_key" in content
            and not basic_validate_pattern(
            content["rubric_key"], r"^D[1-6]-Q[1-9][0-9]*$"
        )
        ):
            errors.append(f"Invalid rubric_key format: {content['rubric_key']}")

    return errors


def basic_validate_rubric(data: Dict[str, Any]) -> List[str]:
    """Basic validation for rubric schema."""
    errors = []

    if "weights" not in data:
        errors.append("Missing required field: weights")
        return errors

    weights = data["weights"]
    for key in weights:
        if not basic_validate_pattern(key, r"^D[1-6]-Q[1-9][0-9]*$"):
            errors.append(f"Invalid rubric key format: {key}")

    return errors


def basic_validate_bundle(data: Dict[str, Any]) -> List[str]:
    """Basic validation for decalogo bundle schema."""
    errors = []
    required = ["version", "policies", "dimensions", "questions", "lexicon"]

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "policies" in data:
        if len(data["policies"]) != 10:
            errors.append(f"Expected 10 policies, got {len(data['policies'])}")
        for p in data["policies"]:
            if not basic_validate_pattern(p, r"^P(10|[1-9])$"):
                errors.append(f"Invalid policy format: {p}")

    if "dimensions" in data:
        if len(data["dimensions"]) != 6:
            errors.append(f"Expected 6 dimensions, got {len(data['dimensions'])}")
        for d in data["dimensions"]:
            if not basic_validate_pattern(d, r"^D[1-6]$"):
                errors.append(f"Invalid dimension format: {d}")

    if "questions" in data:
        for i, q in enumerate(data["questions"]):
            if "question_unique_id" not in q:
                errors.append(f"Question {i}: missing question_unique_id")
            elif not basic_validate_pattern(
                q["question_unique_id"], r"^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$"
            ):
                errors.append(
                    f"Question {i}: invalid question_unique_id format: {q['question_unique_id']}"
                )

            if "rubric_key" not in q:
                errors.append(f"Question {i}: missing rubric_key")
            elif not basic_validate_pattern(q["rubric_key"], r"^D[1-6]-Q[1-9][0-9]*$"):
                errors.append(
                    f"Question {i}: invalid rubric_key format: {q['rubric_key']}"
                )

    return errors


def validate_file(json_path: Path, schema_path: Path) -> bool:
    """Validate a JSON file against its schema."""
    print(f"\nValidating: {json_path.name}")
    print(f"Schema: {schema_path.name}")

    # Load files
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        print(f"  ✗ Error loading files: {e}")
        return False

    # Validate
    try:
        if HAS_JSONSCHEMA:
            validate(instance=data, schema=schema)
            print("  ✓ Valid (jsonschema)")
            return True
        else:
            # Use basic validation
            schema_title = schema.get("title", "")
            if schema_title == "EvidenceEntry":
                errors = basic_validate_evidence(data)
            elif schema_title == "RubricScoring":
                errors = basic_validate_rubric(data)
            elif schema_title == "DecalogoBundle":
                errors = basic_validate_bundle(data)
            else:
                print(f"  ⊘ No basic validator for schema: {schema_title}")
                return True

            if errors:
                print(f"  ✗ Validation errors:")
                for err in errors[:10]:  # Show first 10 errors
                    print(f"    - {err}")
                if len(errors) > 10:
                    print(f"    ... and {len(errors) - 10} more errors")
                return False
            else:
                print("  ✓ Valid (basic validation)")
                return True
    except ValidationError as e:
        print(f"  ✗ Validation error: {e.message}")
        if e.path:
            print(f"    Path: {list(e.path)}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def main():
    """Main validation routine."""
    repo_root = Path(__file__).parent.parent
    schemas_dir = repo_root / "schemas"

    if not schemas_dir.exists():
        print(f"ERROR: Schemas directory not found: {schemas_dir}")
        return 1

    print("=" * 70)
    print("P-D-Q Canonical Notation Validation")
    print("=" * 70)

    validations = [
        (
            repo_root / "config" / "RUBRIC_SCORING.json",
            schemas_dir / "rubric.schema.json",
        ),
        (
            repo_root / "bundles" / "decalogo_bundle.json",
            schemas_dir / "decalogo_bundle.schema.json",
        ),
    ]

    results = []
    for json_file, schema_file in validations:
        if not json_file.exists():
            print(f"\n⊘ Skipping {json_file.name} (not found)")
            continue
        if not schema_file.exists():
            print(f"\n⊘ Skipping validation (schema not found): {schema_file.name}")
            continue

        results.append(validate_file(json_file, schema_file))

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} validations passed")
    print("=" * 70)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
