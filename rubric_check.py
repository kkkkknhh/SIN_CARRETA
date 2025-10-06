#!/usr/bin/env python3
"""
Rubric Check Tool - Validates question-weight mapping between answers report and rubric scoring files.

Exit codes:
  0 - Success: All weights match
  2 - Missing input files
  3 - Question-weight mismatch detected
  1 - Other errors
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, Any


def load_json_file(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """Load JSON file and return (data, error_message)."""
    if not file_path.exists():
        return {}, f"File not found: {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), ""
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON in {file_path}: {e}"
    except Exception as e:
        return {}, f"Error reading {file_path}: {e}"


def extract_question_ids_from_answers(answers_data: Dict[str, Any]) -> Set[str]:
    """Extract question IDs from answers report."""
    question_ids = set()
    for answer in answers_data.get("answers", []):
        qid = answer.get("question_id")
        if qid:
            question_ids.add(qid)
    return question_ids


def extract_question_ids_from_rubric(rubric_data: Dict[str, Any]) -> Set[str]:
    """Extract question IDs from rubric scoring file."""
    question_ids = set()
    
    # Check weights section
    weights = rubric_data.get("weights", {})
    question_ids.update(weights.keys())
    
    # Also check questions array if present
    for question in rubric_data.get("questions", []):
        qid = question.get("id")
        if qid:
            question_ids.add(qid)
    
    return question_ids


def generate_diff_report(missing: Set[str], extra: Set[str], output_dir: Path) -> str:
    """Generate diff file showing missing and extra weights."""
    diff_file = output_dir / "rubric_weight_diff.txt"
    
    with open(diff_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RUBRIC WEIGHT MISMATCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if missing:
            f.write(f"MISSING WEIGHTS ({len(missing)} questions in answers but not in rubric):\n")
            f.write("-" * 80 + "\n")
            for qid in sorted(missing):
                f.write(f"  - {qid}\n")
            f.write("\n")
        
        if extra:
            f.write(f"EXTRA WEIGHTS ({len(extra)} questions in rubric but not in answers):\n")
            f.write("-" * 80 + "\n")
            for qid in sorted(extra):
                f.write(f"  + {qid}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total mismatches: {len(missing) + len(extra)}\n")
    
    return str(diff_file)


def main():
    parser = argparse.ArgumentParser(
        description="Validate question-weight mapping between answers report and rubric scoring files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 - Success: All weights match
  2 - Missing input files
  3 - Question-weight mismatch detected
  1 - Other errors
        """
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path("artifacts/answers_report.json"),
        help="Path to answers report JSON file (default: artifacts/answers_report.json)"
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("rubric_scoring.json"),
        help="Path to rubric scoring JSON file (default: rubric_scoring.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write diff files (default: artifacts)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rubric Check Tool")
    print(f"=" * 80)
    print(f"Answers file: {args.answers}")
    print(f"Rubric file:  {args.rubric}")
    print(f"Output dir:   {args.output_dir}")
    print()
    
    # Load files
    answers_data, answers_error = load_json_file(args.answers)
    rubric_data, rubric_error = load_json_file(args.rubric)
    
    if answers_error or rubric_error:
        print("ERROR: Missing or invalid input files", file=sys.stderr)
        if answers_error:
            print(f"  - {answers_error}", file=sys.stderr)
        if rubric_error:
            print(f"  - {rubric_error}", file=sys.stderr)
        print()
        return 2
    
    # Extract question IDs
    answers_questions = extract_question_ids_from_answers(answers_data)
    rubric_questions = extract_question_ids_from_rubric(rubric_data)
    
    print(f"Questions in answers report: {len(answers_questions)}")
    print(f"Questions in rubric scoring: {len(rubric_questions)}")
    print()
    
    # Find mismatches
    missing_in_rubric = answers_questions - rubric_questions
    extra_in_rubric = rubric_questions - answers_questions
    
    if not missing_in_rubric and not extra_in_rubric:
        print("✅ SUCCESS: All question weights match perfectly!")
        print(f"   Total questions validated: {len(answers_questions)}")
        return 0
    
    # Generate diff report
    print("❌ MISMATCH DETECTED:")
    if missing_in_rubric:
        print(f"   - {len(missing_in_rubric)} questions in answers but missing weights in rubric")
        print(f"     First 10: {sorted(list(missing_in_rubric))[:10]}")
    if extra_in_rubric:
        print(f"   - {len(extra_in_rubric)} questions have weights in rubric but not in answers")
        print(f"     First 10: {sorted(list(extra_in_rubric))[:10]}")
    print()
    
    diff_file = generate_diff_report(missing_in_rubric, extra_in_rubric, args.output_dir)
    print(f"📄 Detailed diff report written to: {diff_file}")
    print()
    
    return 3


if __name__ == "__main__":
    sys.exit(main())
