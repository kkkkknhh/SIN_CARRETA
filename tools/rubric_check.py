#!/usr/bin/env python3
"""
Standalone CLI script for validating consistency between answers_report.json and RUBRIC_SCORING.json.

Usage:
    python rubric_check.py <answers_report.json> <RUBRIC_SCORING.json>

Exit codes:
    0 - Sets match perfectly
    1 - Runtime error during execution
    2 - File not found
    3 - Mismatch detected between question sets
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: rubric_check.py <answers_report.json> <RUBRIC_SCORING.json>", file=sys.stderr)
        sys.exit(1)
    
    answers_path = Path(sys.argv[1])
    rubric_path = Path(sys.argv[2])
    
    # Validate file existence
    if not answers_path.exists():
        print(f"Error: File not found: {answers_path}", file=sys.stderr)
        sys.exit(2)
    
    if not rubric_path.exists():
        print(f"Error: File not found: {rubric_path}", file=sys.stderr)
        sys.exit(2)
    
    try:
        # Load and parse JSON files
        with open(answers_path, 'r', encoding='utf-8') as f:
            answers_data = json.load(f)
        
        with open(rubric_path, 'r', encoding='utf-8') as f:
            rubric_data = json.load(f)
        
        # Extract question ID sets
        answers_questions = set(answers_data.keys())
        rubric_weights = set(rubric_data.get('weights', {}).keys())
        
        # Perform set comparison
        missing_weights = answers_questions - rubric_weights
        extra_weights = rubric_weights - answers_questions
        
        # Output results with deterministic ordering
        missing_weights_sorted = sorted(missing_weights)
        extra_weights_sorted = sorted(extra_weights)
        
        print(f"missing_weights: {len(missing_weights_sorted)}")
        for qid in missing_weights_sorted:
            print(f"  {qid}")
        
        print(f"extra_weights: {len(extra_weights_sorted)}")
        for qid in extra_weights_sorted:
            print(f"  {qid}")
        
        # Exit with appropriate code
        if missing_weights or extra_weights:
            sys.exit(3)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
