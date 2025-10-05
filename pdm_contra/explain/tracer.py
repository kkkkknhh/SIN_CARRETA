"""Generates human-readable explanations for detected findings."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List


class ExplanationTracer:
    """Aggregate textual explanations for contradictions and risks."""

    def __init__(self, language: str = "es") -> None:
        self.language = language
        self._trace: List[Dict[str, Any]] = []

    def generate_explanations(
        self,
        contradictions: Iterable[Any],
        competence_issues: Iterable[Any],
        agenda_gaps: Iterable[Any],
    ) -> List[str]:
        explanations: List[str] = []

        contradictions = list(contradictions)
        competence_issues = list(competence_issues)
        agenda_gaps = list(agenda_gaps)

        if contradictions:
            explanations.append(self._explain_contradictions(contradictions))
        if competence_issues:
            explanations.append(self._explain_competences(competence_issues))
        if agenda_gaps:
            explanations.append(self._explain_agenda(agenda_gaps))

        if explanations:
            explanations.append(
                self._summary(len(contradictions), len(competence_issues), len(agenda_gaps))
            )

        return explanations

    def add_trace(self, action: str, details: Dict[str, Any]) -> None:
        self._trace.append(
            {"timestamp": datetime.utcnow().isoformat(), "action": action, "details": details}
        )

    def get_trace_report(self) -> str:
        lines = ["REGISTRO DE TRAZABILIDAD", "=" * 40]
        for entry in self._trace:
            lines.append(f"[{entry['timestamp']}] {entry['action']}")
            for key, value in entry["details"].items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _explain_contradictions(contradictions: List[Any]) -> str:
        high_risk = sum(
            1
            for c in contradictions
            if str(getattr(c, "risk_level", "")).lower().startswith("high")
        )
        text = [f"Se detectaron {len(contradictions)} posibles contradicciones."]
        if high_risk:
            text.append(f"{high_risk} presentan un nivel de riesgo alto.")
        return " ".join(text)

    @staticmethod
    def _explain_competences(issues: List[Any]) -> str:
        sectors: Dict[str, int] = {}
        for issue in issues:
            sector = (
                issue.get("sector") if isinstance(issue, dict) else getattr(issue, "sector", "general")
            )
            sectors[sector] = sectors.get(sector, 0) + 1
        parts = [f"Se identificaron {len(issues)} observaciones de competencia."]
        if sectors:
            sector_text = ", ".join(f"{sec}: {count}" for sec, count in sectors.items())
            parts.append(f"Distribución por sector: {sector_text}.")
        return " ".join(parts)

    @staticmethod
    def _explain_agenda(gaps: List[Any]) -> str:
        severities: Dict[str, int] = {}
        for gap in gaps:
            severity = (
                gap.get("severity") if isinstance(gap, dict) else getattr(gap, "severity", "medium")
            )
            severities[severity] = severities.get(severity, 0) + 1
        parts = [f"Se encontraron {len(gaps)} brechas en la cadena de planeación."]
        if severities:
            parts.append(
                "Severidad: "
                + ", ".join(f"{level}: {count}" for level, count in severities.items())
            )
        return " ".join(parts)

    @staticmethod
    def _summary(n_contra: int, n_comp: int, n_agenda: int) -> str:
        total = n_contra + n_comp + n_agenda
        if total == 0:
            return "No se identificaron hallazgos relevantes."
        if total <= 5:
            status = "Se detectaron algunos hallazgos puntuales."
        elif total <= 15:
            status = "Se identificaron varios hallazgos que requieren seguimiento."
        else:
            status = "El documento presenta numerosos hallazgos críticos."
        return f"Total de hallazgos: {total}. {status}"


__all__ = ["ExplanationTracer"]
