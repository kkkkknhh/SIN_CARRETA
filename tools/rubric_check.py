#!/usr/bin/env python3
"""
Rubric alignment checker - verifies 1:1 alignment between answers and rubric.

This tool ensures that:
1. All question IDs in answers_report.json exist in RUBRIC_SCORING.json
2. All question IDs in RUBRIC_SCORING.json are used
3. Question IDs follow the correct pattern: P{1-10}-D{1-6}-Q{1-30}
4. Weights sum to exactly 1.0
"""

import json
import re
import sys
from pathlib import Path


def check_rubric_alignment(answers_path=None, rubric_path=None):
    """Check 1:1 alignment between answers and rubric"""
    try:
        repo_root = Path(__file__).parent.parent

        # Use provided paths or default to repo locations
        if answers_path is None:
            answers_path = repo_root / "artifacts" / "answers_report.json"
        else:
            answers_path = Path(answers_path)

        if rubric_path is None:
            rubric_path = repo_root / "RUBRIC_SCORING.json"
        else:
            rubric_path = Path(rubric_path)

        # Check if files exist
        if not rubric_path.exists():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"RUBRIC_SCORING.json not found at {rubric_path}",
                    }
                ),
                file=sys.stderr,
            )
            return 2

        # Load rubric
        with open(rubric_path) as f:
            rubric = json.load(f)

        weights = rubric.get("weights", {})

        # Verify rubric structure
        pattern = re.compile(r"^P([1-9]|10)-D[1-6]-Q([1-9]|[12][0-9]|30)$")
        invalid_ids = [qid for qid in weights.keys() if not pattern.match(qid)]

        if invalid_ids:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "Invalid question ID format in rubric",
                        "invalid_ids": invalid_ids[:10],
                        "expected_format": "P{1-10}-D{1-6}-Q{1-30}",
                    }
                )
            )
            return 3

        # Verify weights sum
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-10:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "Weights do not sum to 1.0",
                        "actual_sum": weight_sum,
                        "difference": abs(weight_sum - 1.0),
                    }
                )
            )
            return 4

        # Verify exactly 300 entries
        if len(weights) != 300:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"Expected 300 weights, found {len(weights)}",
                        "count": len(weights),
                    }
                )
            )
            return 5

        # If answers file doesn't exist, just validate rubric structure
        if not answers_path.exists():
            print(
                json.dumps(
                    {
                        "ok": True,
                        "message": "Rubric structure valid (answers file not yet generated)",
                        "total_questions": len(weights),
                        "weight_sum": weight_sum,
                        "format_valid": True,
                    }
                )
            )
            return 0

        # Load answers and check alignment
        with open(answers_path) as f:
            answers = json.load(f)

        answer_ids = {a["question_id"] for a in answers.get("answers", [])}

        missing = [qid for qid in answer_ids if qid not in weights]
        extra = [qid for qid in weights.keys() if qid not in answer_ids]

        if missing or extra:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "missing_in_rubric": missing[:10],
                        "extra_in_rubric": extra[:10],
                        "message": "1:1 alignment failed between answers and rubric",
                    }
                )
            )
            return 6

        print(
            json.dumps(
                {
                    "ok": True,
                    "message": "1:1 alignment verified",
                    "total_questions": len(weights),
                    "weight_sum": weight_sum,
                    "format_valid": True,
                }
            )
        )
        return 0

    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 3:
        answers_arg = sys.argv[1]
        rubric_arg = sys.argv[2]
        sys.exit(check_rubric_alignment(answers_arg, rubric_arg))
    else:
        sys.exit(check_rubric_alignment())
