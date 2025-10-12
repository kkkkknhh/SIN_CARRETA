"""Causal pattern detection utilities tailored for unit tests.

This module provides a light-weight implementation of ``CausalPatternDetector``
that focuses on high quality rule based detection for a handful of Spanish
causal connectors.  The original project shipped an extremely feature rich
version whose public API diverged from the one exercised by the regression
suite.  The tests in this kata expect a minimal detector with:

* a parameterless constructor
* detection results exposed through ``CausalPatternMatch`` objects
* context aware confidence scores
* helper utilities such as ``get_supported_patterns`` and
  ``calculate_pattern_statistics``

The implementation below restores that contract while keeping the code easy to
reason about.  Confidence scores are derived from semantic weights that are
slightly adjusted according to the surrounding context of each match.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Dict, Iterable, List, Optional


@dataclass
class CausalPatternMatch:
    """Container that represents a detected causal connector.

    Attributes
    ----------
    connector:
        Normalised connector label (e.g. ``"porque"``).
    pattern_type:
        Semantic category for the connector.
    confidence:
        Confidence score in the ``[0.0, 1.0]`` range.
    start:
        Index of the first character of the match inside the analysed text.
    end:
        Index of the first character after the match.
    text:
        Sentence that contains the connector.  Useful for debugging and for the
        overlap checks performed by the test-suite.
    """

    connector: str
    pattern_type: str
    confidence: float
    start: int
    end: int
    text: str


class CausalPatternDetector:
    """Rule based detector that understands a curated set of connectors."""

    #: Definitions for the connectors supported by the detector.  Each entry is
    #: converted into a compiled regular expression at initialisation time.
    _PATTERN_DEFINITIONS: List[Dict[str, str]] = [
        {
            "connector": "porque",
            "pattern_type": "explanatory_causation",
            "pattern": r"\bporque\b",
            "weight": 0.95,
        },
        {
            "connector": "debido a",
            "pattern_type": "explanatory_causation",
            "pattern": r"\bdebido\s+a\b",
            "weight": 0.9,
        },
        {
            "connector": "ya que",
            "pattern_type": "explanatory_causation",
            "pattern": r"\bya\s+que\b",
            "weight": 0.85,
        },
        {
            "connector": "conduce a",
            "pattern_type": "generative_causation",
            "pattern": r"\bconduc\w*\s+(?:a|al|hacia)\b",
            "weight": 0.75,
        },
        {
            "connector": "implica",
            "pattern_type": "implication_causation",
            "pattern": r"\bimplica(?:r)?\b",
            "weight": 0.6,
        },
        {
            "connector": "por medio de",
            "pattern_type": "instrumental_causation",
            "pattern": r"\bpor\s+medio\s+de\b",
            "weight": 0.55,
        },
        {
            "connector": "mediante",
            "pattern_type": "instrumental_causation",
            "pattern": r"\bmediante\b",
            "weight": 0.5,
        },
        {
            "connector": "tendencia a",
            "pattern_type": "tendency_causation",
            "pattern": r"\btendencia\s+a\b",
            "weight": 0.45,
        },
    ]

    def __init__(self):
        self._pattern_configs = [
            {
                "connector": entry["connector"],
                "pattern_type": entry["pattern_type"],
                "regex": re.compile(entry["pattern"], re.IGNORECASE),
                "weight": float(entry["weight"]),
            }
            for entry in self._PATTERN_DEFINITIONS
        ]
        self._weights = {cfg["connector"]: cfg["weight"] for cfg in self._pattern_configs}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_causal_patterns(self, text: Optional[str]) -> List[CausalPatternMatch]:
        """Detect causal connectors in *text*.

        Parameters
        ----------
        text:
            Text to analyse.  ``None`` or an empty string results in an empty
            list.
        """

        normalised = self._normalise_text(text)
        if not normalised:
            return []

        matches: List[CausalPatternMatch] = []
        for config in self._pattern_configs:
            connector = config["connector"]
            for regex_match in config["regex"].finditer(normalised):
                start, end = regex_match.span()
                sentence = self._extract_sentence(normalised, start, end)
                confidence = self._score_match(connector, sentence)
                new_match = CausalPatternMatch(
                    connector=connector,
                    pattern_type=config["pattern_type"],
                    confidence=confidence,
                    start=start,
                    end=end,
                    text=sentence,
                )

                # Filter overlaps by keeping the highest confidence match.
                overlaps = [m for m in matches if self._matches_overlap(m, new_match)]
                if overlaps:
                    strongest = max(overlaps, key=lambda m: m.confidence)
                    if new_match.confidence > strongest.confidence:
                        matches = [m for m in matches if m not in overlaps]
                        matches.append(new_match)
                    continue

                matches.append(new_match)

        return sorted(matches, key=lambda m: (m.start, -m.confidence))

    def get_supported_patterns(self) -> Dict[str, float]:
        """Return a mapping of connector to its semantic weight."""

        return dict(self._weights)

    def calculate_pattern_statistics(self, text: Optional[str]) -> Dict[str, object]:
        """Aggregate statistics for the detected patterns."""

        matches = self.detect_causal_patterns(text)
        if not matches:
            return {
                "total_matches": 0,
                "pattern_types": {},
                "confidence_distribution": {},
                "average_confidence": 0.0,
            }

        pattern_counts: Dict[str, int] = {}
        confidences: List[float] = []
        for match in matches:
            pattern_counts[match.pattern_type] = pattern_counts.get(match.pattern_type, 0) + 1
            confidences.append(match.confidence)

        distribution = self._build_confidence_distribution(confidences)
        average_confidence = sum(confidences) / len(confidences)

        return {
            "total_matches": len(matches),
            "pattern_types": pattern_counts,
            "confidence_distribution": distribution,
            "average_confidence": average_confidence,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_text(text: Optional[str]) -> str:
        if not text:
            return ""
        normalised = unicodedata.normalize("NFC", text)
        return normalised.strip()

    @staticmethod
    def _extract_sentence(text: str, start: int, end: int) -> str:
        """Return the sentence that contains ``text[start:end]``."""

        left = text.rfind(".", 0, start)
        left = max(left, text.rfind("?", 0, start))
        left = max(left, text.rfind("!", 0, start))
        right_period = text.find(".", end)
        right_question = text.find("?", end)
        right_exclamation = text.find("!", end)
        right_candidates = [pos for pos in (right_period, right_question, right_exclamation) if pos != -1]
        right = min(right_candidates) if right_candidates else len(text)
        sentence = text[left + 1 : right].strip()
        if right < len(text) and text[right] in ".?!":
            sentence = f"{sentence}{text[right]}".strip()
        return sentence if sentence else text[start:end]

    def _score_match(self, connector: str, sentence: str) -> float:
        base = self._weights[connector]
        context = self._analyse_context(sentence)

        confidence = base
        if context["question"]:
            confidence -= 0.3
        if context["conditional"]:
            confidence -= 0.15
        if context["hedging"]:
            confidence -= 0.15
        if context["causal_strength"]:
            confidence += 0.1

        sentence_lower = sentence.lower()
        connector_head = connector.split()[0]
        negation_pattern = rf"\bno\b\s+{re.escape(connector_head)}"
        negation = bool(re.search(negation_pattern, sentence_lower))
        if negation:
            confidence -= 0.4

        if connector == "implica" and context["mathematical"]:
            confidence -= 0.2
        if connector == "mediante" and context["instrumental"]:
            confidence -= 0.15
        if connector == "por medio de" and context["instrumental"]:
            confidence -= 0.2
        if connector == "tendencia a" and context["statistical"]:
            confidence -= 0.15

        return float(max(0.1, min(1.0, confidence)))

    @staticmethod
    def _analyse_context(sentence: str) -> Dict[str, bool]:
        sentence_lower = sentence.lower()
        return {
            "question": "?" in sentence,
            "conditional": sentence_lower.strip().startswith("si ")
            or " en caso " in sentence_lower,
            "hedging": bool(
                re.search(r"podr[ií]a|podr[ií]an|podr[ií]amos|posible|posiblemente|quiz[aá]s|podria", sentence_lower)
            ),
            "instrumental": bool(
                re.search(
                    r"herramienta|instrumento|dispositivo|tecnolog|t[eé]cnica|resonancia|m[eé]todo",
                    sentence_lower,
                )
            ),
            "statistical": bool(
                re.search(r"datos|estad[íi]stica|correlaci[óo]n|gr[aá]fic|promedio", sentence_lower)
            ),
            "mathematical": bool(re.search(r"ecuaci[óo]n|=|variable|funci[óo]n", sentence_lower)),
            "causal_strength": bool(
                re.search(
                    r"provoca|causa|genera|impacta|conduce|aumenta|reduce|incrementa|disminuye",
                    sentence_lower,
                )
            ),
        }

    @staticmethod
    def _build_confidence_distribution(values: Iterable[float]) -> Dict[str, int]:
        buckets = {
            "0.0-0.25": 0,
            "0.25-0.5": 0,
            "0.5-0.75": 0,
            "0.75-1.0": 0,
        }
        for value in values:
            if value < 0.25:
                buckets["0.0-0.25"] += 1
            elif value < 0.5:
                buckets["0.25-0.5"] += 1
            elif value < 0.75:
                buckets["0.5-0.75"] += 1
            else:
                buckets["0.75-1.0"] += 1
        return buckets

    # ------------------------------------------------------------------
    # Utilities exposed for the unit tests
    # ------------------------------------------------------------------
    @staticmethod
    def _matches_overlap(match_a: CausalPatternMatch, match_b: CausalPatternMatch) -> bool:
        """Return ``True`` when two matches overlap in the analysed text."""

        return not (match_a.end <= match_b.start or match_b.end <= match_a.start)


__all__ = ["CausalPatternDetector", "CausalPatternMatch"]
