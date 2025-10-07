#!/usr/bin/env python3
"""
rubric_check.py - Standalone CLI tool for validating question-weight mapping consistency.

Compares question IDs from answers_report.json against weights in RUBRIC_SCORING.json.
Outputs minimal diff report suitable for CI/CD pipelines and system_validators.py.

Exit codes:
  0 - Question sets match exactly
  3 - Missing or extra weights detected
  2 - Missing input files
  1 - Runtime errors
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, Any


def load_json(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """Load JSON file, return (data, error_message)."""
    if not file_path.exists():
        return {}, f"File not found: {file_path}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), ""
    except json.JSONDecodeError as e:
        return {}, f"JSON decode error in {file_path}: {e}"
    except Exception as e:
        return {}, f"Error reading {file_path}: {e}"


def extract_answers_questions(data: Dict[str, Any]) -> Set[str]:
    """Extract question IDs from answers_report.json."""
    questions = set()
    for answer in data.get("answers", []):
        qid = answer.get("question_id")
        if qid:
            questions.add(str(qid))
    return questions


def extract_rubric_weights(data: Dict[str, Any]) -> Set[str]:
    """Extract question IDs from RUBRIC_SCORING.json weights section."""
    weights = data.get("weights", {})
    return set(str(k) for k in weights.keys())


def format_diff_report(missing: Set[str], extra: Set[str]) -> str:
    """Generate minimal diff report suitable for human reading and programmatic parsing."""
    lines = []
    lines.append("=" * 80)
    lines.append("RUBRIC WEIGHT MISMATCH")
    lines.append("=" * 80)
    
    if missing:
        lines.append(f"\nMISSING_WEIGHTS: {len(missing)}")
        for qid in sorted(missing):
            lines.append(f"  - {qid}")
    
    if extra:
        lines.append(f"\nEXTRA_WEIGHTS: {len(extra)}")
        for qid in sorted(extra):
            lines.append(f"  + {qid}")
    
    lines.append("\n" + "=" * 80)
    lines.append(f"TOTAL_MISMATCHES: {len(missing) + len(extra)}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main(argv: list = None) -> int:
    """Main entry point for CLI."""
    if argv is None:
        argv = sys.argv[1:]
    
    # Parse arguments
    if len(argv) < 2:
        print("Usage: rubric_check.py <answers_report.json> <RUBRIC_SCORING.json>", file=sys.stderr)
        return 1
    
    answers_path = Path(argv[0])
    rubric_path = Path(argv[1])
    
    # Load files
    answers_data, answers_error = load_json(answers_path)
    rubric_data, rubric_error = load_json(rubric_path)
    
    # Check for missing files
    if answers_error or rubric_error:
        if answers_error:
            print(answers_error, file=sys.stderr)
        if rubric_error:
            print(rubric_error, file=sys.stderr)
        return 2
    
    # Extract question sets
    try:
        answers_questions = extract_answers_questions(answers_data)
        rubric_weights = extract_rubric_weights(rubric_data)
    except Exception as e:
        print(f"Runtime error during extraction: {e}", file=sys.stderr)
        return 1
    
    # Perform set comparison
    missing_weights = answers_questions - rubric_weights
    extra_weights = rubric_weights - answers_questions
    
    # Output results
    if not missing_weights and not extra_weights:
        print(f"OK: All {len(answers_questions)} question weights match exactly")
        return 0
    
    # Generate and print diff report
    report = format_diff_report(missing_weights, extra_weights)
    print(report)
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
