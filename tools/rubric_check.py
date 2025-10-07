#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def check_rubric_alignment(answers_path=None, rubric_path=None):
    """Check 1:1 alignment between answers and rubric"""
    try:
        if answers_path is None or rubric_path is None:
            repo_root = Path(__file__).parent.parent
            answers_path = repo_root / "artifacts" / "answers_report.json"
            rubric_path = repo_root / "RUBRIC_SCORING.json"
        else:
            answers_path = Path(answers_path)
            rubric_path = Path(rubric_path)
        
        if not answers_path.exists():
            print(json.dumps({"ok": False, "error": "answers_report.json not found"}), file=sys.stderr)
            return 2
        
        if not rubric_path.exists():
            print(json.dumps({"ok": False, "error": "RUBRIC_SCORING.json not found"}), file=sys.stderr)
            return 2
        
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
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        return 1

if __name__ == "__main__":
    if len(sys.argv) == 3:
        sys.exit(check_rubric_alignment(sys.argv[1], sys.argv[2]))
    else:
        sys.exit(check_rubric_alignment())
