"""Rule-based pattern matching helpers used by the detector."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern


class PatternMatcher:
    """Detects linguistic patterns that indicate potential contradictions."""

    def __init__(self, language: str = "es") -> None:
        if language != "es":  # pragma: no cover - defensive branch
            raise ValueError("Solo se admite procesamiento en español")

        flags = re.IGNORECASE | re.UNICODE
        self.adversatives = self._compile(
            [
                r"\bsin\s+embargo\b",
                r"\bpero\b",
                r"\baunque\b",
                r"\bno\s+obstante\b",
                r"\ba\s+pesar\s+de(\s+que)?\b",
                r"\bempero\b",
                r"\bmientras\s+que\b",
                r"\bpor\s+el\s+contrario\b",
                r"\ben\s+cambio\b",
            ],
            flags,
        )
        self.goals = self._compile(
            [
                r"\bmetas?\b",
                r"\bobjetivos?\b",
                r"\bprop[óo]sitos?\b",
                r"\bfinalidad(es)?\b",
                r"\bresultados?\s+esperados?\b",
                r"\balcanzar\b",
                r"\blograr\b",
                r"\bpretende(r|mos)?\b",
                r"\bbusca(r|mos)?\b",
            ],
            flags,
        )
        self.action_verbs = self._compile(
            [
                r"\bimplementar\b",
                r"\bejecutar\b",
                r"\bdesarrollar\b",
                r"\brealizar\b",
                r"\bfortalecer\b",
                r"\bpromover\b",
                r"\bgarantizar\b",
                r"\bcoordinar\b",
                r"\bmonitorear\b",
                r"\bcontratar\b",
                r"\bconstruir\b",
            ],
            flags,
        )
        self.quantitative = self._compile(
            [
                r"\d+(?:[.,]\d+)?\s*(?:%|por\s*ciento|porciento)\b",
                r"\d+(?:[.,]\d+)?\s*(?:millones?|mil(?:es)?)\b",
                r"\b(?:incrementar|aumentar|reducir|disminuir)\s+(?:en\s+|hasta\s+)?\d+(?:[.,]\d+)?",
                r"\bmeta\s+de\s+\d+(?:[.,]\d+)?",
                r"\bhasta\s+el\s+\d{4}\b",
            ],
            flags,
        )
        self.modals = self._compile(
            [
                r"\b(?:podría|debería|tendría)\b",
                r"\bes\s+posible\s+que\b",
                r"\bprobablemente\b",
            ],
            flags,
        )
        self.negations = self._compile(
            [
                r"\bno\b",
                r"\bnunca\b",
                r"\bjamás\b",
                r"\btampoco\b",
            ],
            flags,
        )

    @staticmethod
    def _compile(patterns: List[str], flags: int) -> List[Pattern[str]]:
        return [re.compile(pattern, flags) for pattern in patterns]

    def find_adversatives(
        self, text: str, context_window: int = 200
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for pattern in self.adversatives:
            for match in pattern.finditer(text):
                context_start = max(0, match.start() - context_window // 2)
                context_end = min(len(text), match.end() + context_window // 2)
                context = text[context_start:context_end]

                match_info = {
                    "adversative": match.group(),
                    "position": {"start": match.start(), "end": match.end()},
                    "context": context,
                    "goals": self._find_in_context(context, self.goals),
                    "action_verbs": self._find_in_context(context, self.action_verbs),
                    "quantitative": self._find_in_context(context, self.quantitative),
                    "modals": self._find_in_context(context, self.modals),
                    "negations": self._find_in_context(context, self.negations),
                }
                match_info["complexity"] = sum(
                    len(match_info[key]) for key in ("goals", "action_verbs", "quantitative")
                )
                match_info["has_uncertainty"] = bool(match_info["modals"])
                match_info["has_negation"] = bool(match_info["negations"])
                matches.append(match_info)
        return matches

    @staticmethod
    def _find_in_context(context: str, patterns: List[Pattern[str]]) -> List[str]:
        found: List[str] = []
        for pattern in patterns:
            for match in pattern.finditer(context):
                found.append(match.group().strip())
        return sorted(set(found))


__all__ = ["PatternMatcher"]
