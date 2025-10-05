"""Simplified competence validator for municipal planning documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


class CompetenceValidator:
    """Detects potential overreach or omissions in PDM actions."""

    def __init__(self, matrix_path: Path | None = None) -> None:
        self.matrix_path = matrix_path
        self.overreach_patterns: Dict[str, List[re.Pattern[str]]] = {
            "salud": [
                re.compile(r"construir\s+un\s+hospital\s+de\s+nivel\s*[23]", re.IGNORECASE),
                re.compile(r"administrar\s+hospital(es)?", re.IGNORECASE),
            ],
            "educacion": [
                re.compile(r"nombrar\s+docentes", re.IGNORECASE),
                re.compile(r"administrar\s+instituciones\s+educativas", re.IGNORECASE),
            ],
            "seguridad": [
                re.compile(r"dirigir\s+polic[ií]a", re.IGNORECASE),
                re.compile(r"comandar\s+fuerzas\s+armadas", re.IGNORECASE),
            ],
        }
        self.valid_coordination = re.compile(
            r"\b(gestionar|coordinar|articular|apoyar|promover|facilitar)\b",
            re.IGNORECASE,
        )
        self.essential_competences: Dict[str, List[str]] = {
            "salud": ["promoción", "prevención"],
            "educacion": ["infraestructura", "dotación"],
            "agua": ["acueducto", "saneamiento"],
        }

    def validate_segment(
        self, text: str, sectors: List[str], level: str = "municipal"
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        lowered = text.lower()

        for sector in sectors:
            sector_key = sector.lower()
            for pattern in self.overreach_patterns.get(sector_key, []):
                for match in pattern.finditer(text):
                    context_start = max(0, match.start() - 60)
                    context_end = min(len(text), match.end() + 60)
                    context = text[context_start:context_end]
                    if self.valid_coordination.search(context):
                        continue
                    issues.append(
                        {
                            "type": "competence_overreach",
                            "sector": sector_key,
                            "level": level,
                            "text": match.group(),
                            "position": {"start": match.start(), "end": match.end()},
                            "context": context,
                            "required_level": self._required_level(sector_key),
                            "legal_basis": self._legal_basis(sector_key),
                            "explanation": (
                                "La acción descrita suele corresponder a un nivel"
                                " superior de gobierno en Colombia."
                            ),
                            "suggested_fix": (
                                "Reformular usando verbos de coordinación y apoyo"
                                " propios del nivel municipal."
                            ),
                        }
                    )

            if level == "municipal":
                for competence in self.essential_competences.get(sector_key, []):
                    if competence not in lowered:
                        issues.append(
                            {
                                "type": "missing_essential_competence",
                                "sector": sector_key,
                                "level": level,
                                "text": competence,
                                "explanation": (
                                    f"No se menciona la competencia esencial '{competence}'"
                                    " para el sector analizado."
                                ),
                                "suggested_fix": (
                                    "Incorporar acciones explícitas que garanticen la"
                                    f" {competence} a nivel municipal."
                                ),
                            }
                        )

        return issues

    @staticmethod
    def _required_level(sector: str) -> str:
        return {
            "salud": "departamental",
            "educacion": "departamental",
            "seguridad": "nacional",
        }.get(sector, "departamental")

    @staticmethod
    def _legal_basis(sector: str) -> List[str]:
        base = {
            "salud": ["Ley 715 de 2001", "Ley 1438 de 2011"],
            "educacion": ["Ley 715 de 2001"],
            "seguridad": ["Constitución Política art. 189"],
        }
        return base.get(sector, ["Constitución Política art. 287"])


__all__ = ["CompetenceValidator"]
