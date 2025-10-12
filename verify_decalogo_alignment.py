#!/usr/bin/env python
"""
Verification script for decalogo-industrial.latest.clean.json alignment.

This script validates that the decalogo file maintains proper structure:
- Correct number of questions (300 expected)
- No duplicate questions
- All required fields present
- Proper structure matching the schema
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_decalogo(path: Path) -> Dict:
    """Load and parse the decalogo JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {path}: {e}")
        sys.exit(1)


def verify_structure(data: Dict) -> List[str]:
    """Verify the basic structure of the decalogo data."""
    errors = []

    # Check required top-level fields
    required_fields = {"version", "schema", "total", "questions"}
    missing = required_fields - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # Check that questions is a list
    if not isinstance(data.get("questions"), list):
        errors.append("'questions' field must be a list")

    return errors


def verify_questions(questions: List[Dict]) -> Tuple[List[str], Dict]:
    """Verify the questions data structure and content."""
    errors = []
    stats = {
        "total_questions": len(questions),
        "unique_questions": 0,
        "unique_points": set(),
        "unique_dimensions": set(),
        "questions_per_point": {},
    }

    required_fields = {
        "id",
        "dimension",
        "question_no",
        "point_code",
        "point_title",
        "prompt",
        "hints",
    }

    seen_combinations: Set[Tuple] = set()

    for i, question in enumerate(questions):
        # Check required fields
        missing = required_fields - set(question.keys())
        if missing:
            errors.append(f"Question {i}: missing fields {missing}")
            continue

        # Track statistics
        point_code = question.get("point_code")
        dimension = question.get("dimension")
        question_no = question.get("question_no")

        stats["unique_points"].add(point_code)
        stats["unique_dimensions"].add(dimension)

        # Track questions per point
        if point_code:
            stats["questions_per_point"][point_code] = (
                stats["questions_per_point"].get(point_code, 0) + 1
            )

        # Check for duplicates
        combo = (point_code, dimension, question_no)
        if combo in seen_combinations:
            errors.append(
                f"Duplicate question: {dimension}-Q{question_no} "
                f"for {point_code} at index {i}"
            )
        else:
            seen_combinations.add(combo)

    stats["unique_questions"] = len(seen_combinations)

    return errors, stats


def verify_alignment(data: Dict) -> bool:
    """Main verification function."""
    print("=" * 70)
    print("DECALOGO ALIGNMENT VERIFICATION")
    print("=" * 70)

    all_errors = []

    # Verify structure
    print("\n📋 Checking structure...")
    structure_errors = verify_structure(data)
    if structure_errors:
        all_errors.extend(structure_errors)
        for error in structure_errors:
            print(f"  ❌ {error}")
    else:
        print("  ✓ Structure is valid")

    # Verify questions
    print("\n📝 Checking questions...")
    questions = data.get("questions", [])
    question_errors, stats = verify_questions(questions)

    if question_errors:
        all_errors.extend(question_errors)
        # Show first 10 errors only
        for error in question_errors[:10]:
            print(f"  ❌ {error}")
        if len(question_errors) > 10:
            print(f"  ... and {len(question_errors) - 10} more errors")
    else:
        print("  ✓ All questions have required fields")

    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total questions in file: {stats['total_questions']}")
    print(f"  Total field value: {data.get('total', 'N/A')}")
    print(f"  Unique combinations: {stats['unique_questions']}")
    print(f"  Policy points (P1-P10): {len(stats['unique_points'])}")
    print(f"  Dimensions (D1-D6): {len(stats['unique_dimensions'])}")

    # Check expected counts
    print("\n🔍 Validation checks:")

    expected_total = 300
    if stats["total_questions"] == expected_total:
        print(
            f"  ✓ Total questions: {stats['total_questions']} (expected {expected_total})"
        )
    else:
        error = (
            f"Total questions: {stats['total_questions']} (expected {expected_total})"
        )
        all_errors.append(error)
        print(f"  ❌ {error}")

    if data.get("total") == stats["total_questions"]:
        print("  ✓ 'total' field matches actual count")
    else:
        error = f"'total' field ({data.get('total')}) doesn't match actual count ({stats['total_questions']})"
        all_errors.append(error)
        print(f"  ❌ {error}")

    if stats["unique_questions"] == stats["total_questions"]:
        print("  ✓ No duplicate questions")
    else:
        duplicates = stats["total_questions"] - stats["unique_questions"]
        error = f"Found {duplicates} duplicate questions"
        all_errors.append(error)
        print(f"  ❌ {error}")

    expected_points = 10
    if len(stats["unique_points"]) == expected_points:
        print(f"  ✓ All {expected_points} policy points present")
    else:
        error = f"Expected {expected_points} policy points, found {len(stats['unique_points'])}"
        all_errors.append(error)
        print(f"  ❌ {error}")

    # Check questions per point
    print("\n📈 Questions per policy point:")
    expected_per_point = 30
    for point in sorted(stats["questions_per_point"].keys()):
        count = stats["questions_per_point"][point]
        status = "✓" if count == expected_per_point else "❌"
        print(f"  {status} {point}: {count:3d} questions")
        if count != expected_per_point:
            all_errors.append(
                f"{point} has {count} questions (expected {expected_per_point})"
            )

    # Final result
    print("\n" + "=" * 70)
    if all_errors:
        print(f"❌ VALIDATION FAILED - {len(all_errors)} error(s) found")
        print("=" * 70)
        return False
    else:
        print("✅ VALIDATION PASSED - All checks successful!")
        print("=" * 70)
        return True


def main():
    """Main entry point."""
    # Use central path resolver
    from repo_paths import get_decalogo_path

    decalogo_path = get_decalogo_path()

    # Allow override from command line
    if len(sys.argv) > 1:
        decalogo_path = Path(sys.argv[1])

    print(f"\nVerifying: {decalogo_path}")

    data = load_decalogo(decalogo_path)
    success = verify_alignment(data)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
