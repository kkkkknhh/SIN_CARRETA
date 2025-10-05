"""Lightweight risk scoring utilities."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from pdm_contra.models import RiskLevel


class RiskScorer:
    """Combine detected issues into an overall risk score."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def calculate_risk(
        self,
        contradictions: List[Any],
        competence_issues: List[Any],
        agenda_gaps: List[Any],
    ) -> Dict[str, Any]:
        contradiction_score = self._score_contradictions(contradictions)
        competence_score = self._score_competences(competence_issues)
        agenda_score = self._score_agenda(agenda_gaps)

        overall = 0.4 * contradiction_score + 0.35 * competence_score + 0.25 * agenda_score
        risk_level = self._score_to_level(overall)

        return {
            "overall_risk": round(overall, 3),
            "risk_level": risk_level.value,
            "component_scores": {
                "contradiction": round(contradiction_score, 3),
                "competence": round(competence_score, 3),
                "agenda": round(agenda_score, 3),
            },
            "confidence_intervals": self._confidence_intervals(overall),
            "empirical_coverage": 1 - self.alpha,
        }

    @staticmethod
    def _score_contradictions(contradictions: List[Any]) -> float:
        if not contradictions:
            return 0.0
        confidences = []
        for match in contradictions:
            confidences.append(float(getattr(match, "confidence", 0.5)))
            if getattr(match, "nli_score", None):
                confidences[-1] = min(1.0, confidences[-1] + 0.1)
        return min(1.0, 0.6 * mean(confidences) + 0.1 * len(confidences) / 5)

    @staticmethod
    def _score_competences(issues: List[Any]) -> float:
        if not issues:
            return 0.0
        severities = []
        for issue in issues:
            issue_type = str(issue.get("type") if isinstance(issue, dict) else getattr(issue, "competence_type", ""))
            if "overreach" in issue_type:
                severities.append(0.8)
            elif "missing" in issue_type:
                severities.append(0.6)
            else:
                severities.append(0.4)
        return min(1.0, mean(severities) + 0.05 * len(severities))

    @staticmethod
    def _score_agenda(gaps: List[Any]) -> float:
        if not gaps:
            return 0.0
        scores = []
        for gap in gaps:
            severity = str(gap.severity if hasattr(gap, "severity") else gap.get("severity", "medium"))
            mapping = {"low": 0.3, "medium": 0.5, "high": 0.8}
            scores.append(mapping.get(severity, 0.5))
        return min(1.0, mean(scores) + 0.03 * len(scores))

    def _confidence_intervals(self, score: float) -> Dict[str, List[float]]:
        margin = max(0.05, self.alpha / 2)
        return {
            "overall": [max(0.0, score - margin), min(1.0, score + margin)],
        }

    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        if score >= 0.75:
            return RiskLevel.HIGH
        if score >= 0.55:
            return RiskLevel.MEDIUM_HIGH
        if score >= 0.35:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


__all__ = ["RiskScorer"]
