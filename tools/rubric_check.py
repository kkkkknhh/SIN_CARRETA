#!/usr/bin/env python3
"""
Rubric Check CLI Tool

Validates 1:1 alignment between answers_report.json and RUBRIC_SCORING.json question IDs.
Exits with specific codes for different failure scenarios.
"""

import json
import sys
from pathlib import Path
from typing import Set, Tuple, Dict, Any


EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_FILE_ERROR = 2
EXIT_MISMATCH = 3


def parse_answers_report(file_path: Path) -> Set[str]:
    """
    Extract all question IDs from answers_report.json.
    
    Expected structure:
    {
        "answers": [
            {"question_id": "DE-1-Q1", ...},
            ...
        ]
    }
    
    Returns:
        Set of question ID strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(json.dumps({
            "error": "file_read_error",
            "file": str(file_path),
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    except json.JSONDecodeError as e:
        print(json.dumps({
            "error": "invalid_json",
            "file": str(file_path),
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    
    if not isinstance(data, dict):
        print(json.dumps({
            "error": "invalid_structure",
            "file": str(file_path),
            "message": "Expected top-level object"
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    
    if 'answers' not in data:
        print(json.dumps({
            "error": "missing_answers_key",
            "file": str(file_path),
            "message": "Expected 'answers' key at top level"
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    
    if not isinstance(data['answers'], list):
        print(json.dumps({
            "error": "invalid_answers_type",
            "file": str(file_path),
            "message": "Expected 'answers' to be a list"
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    
    question_ids = set()
    for idx, answer in enumerate(data['answers']):
        if not isinstance(answer, dict):
            continue
        if 'question_id' in answer:
            question_ids.add(answer['question_id'])
    
    return question_ids


def parse_rubric_scoring(file_path: Path) -> Set[str]:
    """
    Extract all question IDs from RUBRIC_SCORING.json weights section.
    
    Expected structure (option 1 - weights dict):
    {
        "weights": {
            "DE-1-Q1": 0.0033,
            ...
        }
    }
    
    Expected structure (option 2 - questions list):
    {
        "questions": [
            {"id": "D1-Q1", ...},
            ...
        ]
    }
    
    Returns:
        Set of question ID strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(json.dumps({
            "error": "file_read_error",
            "file": str(file_path),
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    except json.JSONDecodeError as e:
        print(json.dumps({
            "error": "invalid_json",
            "file": str(file_path),
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    
    if not isinstance(data, dict):
        print(json.dumps({
            "error": "invalid_structure",
            "file": str(file_path),
            "message": "Expected top-level object"
        }), file=sys.stderr)
        sys.exit(EXIT_FILE_ERROR)
    
    # Try to extract from 'weights' section first
    if 'weights' in data:
        if not isinstance(data['weights'], dict):
            print(json.dumps({
                "error": "invalid_weights_type",
                "file": str(file_path),
                "message": "Expected 'weights' to be a dictionary"
            }), file=sys.stderr)
            sys.exit(EXIT_FILE_ERROR)
        return set(data['weights'].keys())
    
    # Fallback: try to extract from 'questions' list
    if 'questions' in data:
        if not isinstance(data['questions'], list):
            print(json.dumps({
                "error": "invalid_questions_type",
                "file": str(file_path),
                "message": "Expected 'questions' to be a list"
            }), file=sys.stderr)
            sys.exit(EXIT_FILE_ERROR)
        
        question_ids = set()
        for question in data['questions']:
            if isinstance(question, dict) and 'id' in question:
                question_ids.add(question['id'])
        return question_ids
    
    # Neither 'weights' nor 'questions' found
    print(json.dumps({
        "error": "missing_weights_or_questions_key",
        "file": str(file_path),
        "message": "Expected 'weights' dictionary or 'questions' list at top level"
    }), file=sys.stderr)
    sys.exit(EXIT_FILE_ERROR)


def compute_diff(answers_ids: Set[str], rubric_ids: Set[str]) -> Dict[str, Any]:
    """
    Compute the difference between two sets of question IDs.
    
    Returns:
        Dict with counts and sorted lists of mismatches
    """
    missing_weights = sorted(answers_ids - rubric_ids)
    extra_weights = sorted(rubric_ids - answers_ids)
    
    return {
        "match": len(missing_weights) == 0 and len(extra_weights) == 0,
        "answers_count": len(answers_ids),
        "rubric_count": len(rubric_ids),
        "missing_weights_count": len(missing_weights),
        "extra_weights_count": len(extra_weights),
        "missing_weights": missing_weights,
        "extra_weights": extra_weights
    }


def main():
    """
    Main CLI entry point.
    """
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "invalid_arguments",
            "message": "Usage: rubric_check.py <answers_report.json> <RUBRIC_SCORING.json>"
        }), file=sys.stderr)
        sys.exit(EXIT_RUNTIME_ERROR)
    
    answers_path = Path(sys.argv[1])
    rubric_path = Path(sys.argv[2])
    
    try:
        # Parse both files
        answers_ids = parse_answers_report(answers_path)
        rubric_ids = parse_rubric_scoring(rubric_path)
        
        # Compute difference
        diff = compute_diff(answers_ids, rubric_ids)
        
        # Output JSON result
        print(json.dumps(diff, indent=2, ensure_ascii=False))
        
        # Exit with appropriate code
        if diff["match"]:
            sys.exit(EXIT_SUCCESS)
        else:
            sys.exit(EXIT_MISMATCH)
    
    except Exception as e:
        print(json.dumps({
            "error": "runtime_error",
            "message": str(e),
            "type": type(e).__name__
        }), file=sys.stderr)
        sys.exit(EXIT_RUNTIME_ERROR)


if __name__ == "__main__":
    main()
