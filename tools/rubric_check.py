#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def check_rubric_alignment():
    """Check 1:1 alignment between answers and rubric"""
    try:
        repo_root = Path(__file__).parent.parent
        answers_path = repo_root / "artifacts" / "answers_report.json"
        rubric_path = repo_root / "RUBRIC_SCORING.json"
        
        if not answers_path.exists():
            print(json.dumps({"ok": False, "error": "answers_report.json not found"}))
            return 1
        
        if not rubric_path.exists():
            print(json.dumps({"ok": False, "error": "RUBRIC_SCORING.json not found"}))
            return 1
        
        with open(answers_path) as f:
            answers = json.load(f)
        
        with open(rubric_path) as f:
            rubric = json.load(f)
        
        weights = rubric.get("weights", {})
        answer_ids = {a["question_id"] for a in answers.get("answers", [])}
        
        missing = [qid for qid in answer_ids if qid not in weights]
        extra = [qid for qid in weights.keys() if qid not in answer_ids]
        
        if missing or extra:
            print(json.dumps({
                "ok": False,
                "missing_in_rubric": missing[:10],
                "extra_in_rubric": extra[:10],
                "message": "1:1 alignment failed"
            }))
            return 3
        
        print(json.dumps({"ok": True, "message": "1:1 alignment verified"}))
        return 0
        
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    sys.exit(check_rubric_alignment())
