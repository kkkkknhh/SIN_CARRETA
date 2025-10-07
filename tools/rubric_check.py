#!/usr/bin/env python3
"""
rubric_check.py - 1:1 Validation for answers_report.json vs RUBRIC_SCORING.json

Validates that the question sets between answers_report.json and RUBRIC_SCORING.json['weights']
are identical (1:1 alignment).

Exit codes:
  0 - Success: perfect 1:1 alignment
  1 - Runtime error (exception, file parsing error)
  2 - Missing input file(s)
  3 - Mismatch: questions in answers not in weights, or vice versa

Output:
  JSON to stdout with minimal diff (missing_weights/extra_weights counts and lists)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any


def load_json_file(path: Path, label: str) -> Dict[str, Any]:
    """Load and parse JSON file with error handling."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(json.dumps({
            "ok": False,
            "error": f"{label} not found",
            "path": str(path)
        }), file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(json.dumps({
            "ok": False,
            "error": f"{label} JSON parsing failed",
            "path": str(path),
            "details": str(e)
        }), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": f"Failed to load {label}",
            "path": str(path),
            "details": str(e)
        }), file=sys.stderr)
        sys.exit(1)


def extract_question_ids_from_answers(answers_data: Dict[str, Any]) -> Set[str]:
    """Extract question IDs from answers_report.json."""
    answers_list = answers_data.get("answers", [])
    if not isinstance(answers_list, list):
        raise ValueError("answers_report.json must have an 'answers' list")
    
    question_ids = set()
    for answer in answers_list:
        if isinstance(answer, dict) and "question_id" in answer:
            question_ids.add(answer["question_id"])
    
    return question_ids


def extract_question_ids_from_rubric(rubric_data: Dict[str, Any]) -> Set[str]:
    """Extract question IDs from RUBRIC_SCORING.json['weights']."""
    weights = rubric_data.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError("RUBRIC_SCORING.json must have a 'weights' dict")
    
    return set(weights.keys())


def check_rubric_alignment(answers_path: Path, rubric_path: Path) -> int:
    """
    Check 1:1 alignment between answers and rubric weights.
    
    Returns:
        0 - Success (perfect alignment)
        1 - Runtime error
        2 - Missing input file
        3 - Mismatch detected
    """
    try:
        # Load files
        answers_data = load_json_file(answers_path, "answers_report.json")
        rubric_data = load_json_file(rubric_path, "RUBRIC_SCORING.json")
        
        # Extract question ID sets
        answer_ids = extract_question_ids_from_answers(answers_data)
        weight_ids = extract_question_ids_from_rubric(rubric_data)
        
        # Compute diffs
        missing_in_weights = sorted(answer_ids - weight_ids)
        extra_in_weights = sorted(weight_ids - answer_ids)
        
        # Build output
        output = {
            "ok": len(missing_in_weights) == 0 and len(extra_in_weights) == 0,
            "answers_count": len(answer_ids),
            "weights_count": len(weight_ids),
            "missing_weights_count": len(missing_in_weights),
            "extra_weights_count": len(extra_in_weights)
        }
        
        if missing_in_weights:
            output["missing_weights"] = missing_in_weights
        
        if extra_in_weights:
            output["extra_weights"] = extra_in_weights
        
        # Print minimal diff output
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
        # Return appropriate exit code
        if output["ok"]:
            return 0
        else:
            return 3
            
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": "Runtime error during validation",
            "details": str(e)
        }), file=sys.stderr)
        return 1


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate 1:1 alignment between answers_report.json and RUBRIC_SCORING.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 - Success: perfect 1:1 alignment
  1 - Runtime error (exception, file parsing error)
  2 - Missing input file(s)
  3 - Mismatch: questions in answers not in weights, or vice versa

Examples:
  # Using default paths (relative to repo root)
  python3 tools/rubric_check.py

  # Using custom paths
  python3 tools/rubric_check.py --answers artifacts/answers_report.json --rubric RUBRIC_SCORING.json
  
  # Explicit positional arguments (legacy compatibility)
  python3 tools/rubric_check.py artifacts/answers_report.json RUBRIC_SCORING.json
        """
    )
    
    parser.add_argument(
        "answers_report",
        nargs="?",
        type=str,
        help="Path to answers_report.json (default: artifacts/answers_report.json)"
    )
    
    parser.add_argument(
        "rubric_scoring",
        nargs="?",
        type=str,
        help="Path to RUBRIC_SCORING.json (default: RUBRIC_SCORING.json)"
    )
    
    parser.add_argument(
        "--answers",
        type=str,
        help="Path to answers_report.json (alternative to positional arg)"
    )
    
    parser.add_argument(
        "--rubric",
        type=str,
        help="Path to RUBRIC_SCORING.json (alternative to positional arg)"
    )
    
    args = parser.parse_args()
    
    # Determine repo root (script is in tools/ subdirectory)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    # Resolve answers path (priority: positional > --answers > default)
    if args.answers_report:
        answers_path = Path(args.answers_report)
    elif args.answers:
        answers_path = Path(args.answers)
    else:
        answers_path = repo_root / "artifacts" / "answers_report.json"
    
    # Resolve rubric path (priority: positional > --rubric > default)
    if args.rubric_scoring:
        rubric_path = Path(args.rubric_scoring)
    elif args.rubric:
        rubric_path = Path(args.rubric)
    else:
        rubric_path = repo_root / "RUBRIC_SCORING.json"
    
    # Make paths absolute if they're relative
    if not answers_path.is_absolute():
        answers_path = repo_root / answers_path
    if not rubric_path.is_absolute():
        rubric_path = repo_root / rubric_path
    
    # Run validation and exit with appropriate code
    exit_code = check_rubric_alignment(answers_path, rubric_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
