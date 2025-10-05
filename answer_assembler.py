# coding=utf-8
from __future__ import annotations
import json, pathlib
from typing import Any, Dict, List
from evidence_registry import EvidenceRegistry

class AnswerAssembler:
    def __init__(self, rubric_path: str = "RUBRIC_SCORING.json") -> None:
        self.rubric_path = rubric_path
        self.rubric = self._load_rubric(rubric_path)

    def _load_rubric(self, path: str) -> Dict[str, Any]:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"RUBRIC file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def assemble(self, evidence: EvidenceRegistry, decalogo_eval: Dict[str, Any], questionnaire_eval: Dict[str, Any]) -> Dict[str, Any]:
        weights: Dict[str, float] = self.rubric.get("weights", {})
        qids: List[str] = sorted(set(list(decalogo_eval.get("questions", {}).keys()) + list(questionnaire_eval.get("questions", {}).keys())))
        answers: List[Dict[str, Any]] = []
        for q in qids:
            evs = evidence.get_evidence_for_question(q)
            ev_ids = [e.metadata.get("evidence_id") for e in evs]
            conf = max([e.confidence for e in evs], default=0.0)
            base_score = 1.0 if evs else 0.0
            w = float(weights.get(q, 1.0))
            answers.append({
                "question_id": q,
                "evidence_ids": ev_ids,
                "confidence": round(conf, 4),
                "score": round(base_score * w, 4),
                "rationale": "Evidencia trazable" if evs else "Sin evidencia trazable"
            })
        return {
            "answers": answers,
            "summary": {
                "total_questions": len(qids),
                "with_evidence": sum(1 for a in answers if a["evidence_ids"]),
                "rubric_file": str(self.rubric_path)
            }
        }

    def save_reports(self, report: Dict[str, Any], out_dir: str) -> Dict[str, str]:
        p = pathlib.Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        main = p / "answers_report.json"
        sample = p / "answers_sample.json"
        with main.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=True)
        # sample: first 10
        sample_obj = dict(report)
        sample_obj["answers"] = report["answers"][:10]
        with sample.open("w", encoding="utf-8") as f:
            json.dump(sample_obj, f, indent=2, sort_keys=True, ensure_ascii=True)
        return {"answers_report": str(main), "answers_sample": str(sample)}

if __name__ == "__main__":
    # tiny self-test (requires a RUBRIC_SCORING.json and minimal eval dicts)
    pass
