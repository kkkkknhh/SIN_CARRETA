#!/usr/bin/env python3
"""
rubric_check.py — Verifies 1:1 correspondence between questions and rubric weights.

Validates that:
1. Every question in answers_report.json has a corresponding weight in RUBRIC_SCORING.json
2. Every weight in RUBRIC_SCORING.json corresponds to a question in answers_report.json

Exit codes:
0: Perfect 1:1 correspondence
3: Mismatch detected (missing or extra weights)
1: Internal error (file not found, parse error, etc.)
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Set


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Cannot parse JSON {path}: {e}")


def verify_rubric_correspondence(
    answers_path: str = "artifacts/answers_report.json",
    rubric_path: str = "RUBRIC_SCORING.json"
) -> Dict[str, Any]:
    """
    Verify 1:1 correspondence between questions and rubric weights.
    
    Returns:
        Dict with keys:
        - ok: bool (True if perfect 1:1 correspondence)
        - missing_weights: List[str] (questions without rubric weights)
        - extra_weights: List[str] (rubric weights without questions)
        - total_questions: int
        - total_weights: int
    """
    answers = load_json(Path(answers_path))
    rubric = load_json(Path(rubric_path))
    
    # Extract question IDs from answers
    question_ids: Set[str] = {
        answer.get("question_id")
        for answer in answers.get("answers", [])
        if answer.get("question_id")
    }
    
    # Extract weight keys from rubric
    weight_keys: Set[str] = set(rubric.get("weights", {}).keys())
    
    # Find mismatches
    missing_weights = sorted(question_ids - weight_keys)
    extra_weights = sorted(weight_keys - question_ids)
    
    ok = len(missing_weights) == 0 and len(extra_weights) == 0
    
    return {
        "ok": ok,
        "missing_weights": missing_weights,
        "extra_weights": extra_weights,
        "total_questions": len(question_ids),
        "total_weights": len(weight_keys),
    }


def main() -> int:
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify 1:1 correspondence between questions and rubric weights"
    )
    parser.add_argument(
        "--answers",
        default="artifacts/answers_report.json",
        help="Path to answers_report.json (default: artifacts/answers_report.json)"
    )
    parser.add_argument(
        "--rubric",
        default="RUBRIC_SCORING.json",
        help="Path to RUBRIC_SCORING.json (default: RUBRIC_SCORING.json)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        result = verify_rubric_correspondence(args.answers, args.rubric)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Total questions: {result['total_questions']}")
            print(f"Total rubric weights: {result['total_weights']}")
            
            if result["missing_weights"]:
                print(f"\n❌ Missing rubric weights for {len(result['missing_weights'])} questions:")
                for qid in result["missing_weights"][:10]:
                    print(f"  - {qid}")
                if len(result["missing_weights"]) > 10:
                    print(f"  ... and {len(result['missing_weights']) - 10} more")
            
            if result["extra_weights"]:
                print(f"\n❌ Extra rubric weights not in answers for {len(result['extra_weights'])} keys:")
                for key in result["extra_weights"][:10]:
                    print(f"  - {key}")
                if len(result["extra_weights"]) > 10:
                    print(f"  ... and {len(result['extra_weights']) - 10} more")
            
            if result["ok"]:
                print("\n✅ Perfect 1:1 correspondence verified")
            else:
                print("\n❌ Mismatch detected - 1:1 correspondence violated")
        
        if result["ok"]:
            return 0
        else:
            return 3  # Special exit code for mismatch
            
    except Exception as e:
        if args.json:
            print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
