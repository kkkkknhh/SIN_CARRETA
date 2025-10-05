"""Heuristic Natural Language Inference detector for Spanish text."""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple


class SpanishNLIDetector:
    """Lightweight contradiction detector based on lexical heuristics."""

    NEGATION_MARKERS = {
        "no",
        "nunca",
        "jamás",
        "ningún",
        "ninguna",
        "sin",
        "tampoco",
    }

    POSITIVE_INTENT = {
        "aumentar",
        "incrementar",
        "alcanzar",
        "lograr",
        "garantizar",
        "cumplir",
        "expandir",
        "impulsar",
    }

    NEGATIVE_INTENT = {
        "reducir",
        "disminuir",
        "recortar",
        "limitar",
        "suspender",
        "eliminar",
        "restringir",
    }

    def __init__(self, light_mode: bool = False) -> None:
        self.light_mode = light_mode

    def check_contradiction(
        self, premise: str, hypothesis: str, return_all_scores: bool = False
    ) -> Dict[str, float | str | bool | Dict[str, float]]:
        """Check whether two sentences contradict each other."""

        premise_tokens = self._tokenize(premise)
        hypothesis_tokens = self._tokenize(hypothesis)

        negation_score = self._negation_conflict(premise_tokens, hypothesis_tokens)
        intent_score = self._intent_conflict(premise_tokens, hypothesis_tokens)
        numeric_score = self._numeric_conflict(premise, hypothesis)

        score = max(negation_score, intent_score, numeric_score)
        label = "contradiction" if score >= 0.6 else "neutral"
        result: Dict[str, float | str | bool | Dict[str, float]] = {
            "label": label,
            "score": float(score),
            "is_contradiction": label == "contradiction",
        }

        if return_all_scores:
            result["all_scores"] = {
                "contradiction": float(score),
                "neutral": float(1 - score),
                "entailment": 0.0,
            }

        return result

    def batch_check(
        self, pairs: Iterable[Tuple[str, str]], batch_size: int = 8
    ) -> List[Dict[str, float | str | bool]]:
        return [self.check_contradiction(premise, hypothesis) for premise, hypothesis in pairs]

    def _negation_conflict(self, premise: List[str], hypothesis: List[str]) -> float:
        premise_neg = any(token in self.NEGATION_MARKERS for token in premise)
        hypothesis_neg = any(token in self.NEGATION_MARKERS for token in hypothesis)
        if premise_neg == hypothesis_neg:
            return 0.0
        overlap = len(set(premise) & set(hypothesis))
        return 0.6 + min(0.3, overlap * 0.1)

    def _intent_conflict(self, premise: List[str], hypothesis: List[str]) -> float:
        premise_pos = any(token in self.POSITIVE_INTENT for token in premise)
        premise_neg = any(token in self.NEGATIVE_INTENT for token in premise)
        hypothesis_pos = any(token in self.POSITIVE_INTENT for token in hypothesis)
        hypothesis_neg = any(token in self.NEGATIVE_INTENT for token in hypothesis)
        if (premise_pos and hypothesis_neg) or (premise_neg and hypothesis_pos):
            return 0.7
        return 0.0

    @staticmethod
    def _numeric_conflict(premise: str, hypothesis: str) -> float:
        numbers_premise = SpanishNLIDetector._extract_numbers(premise)
        numbers_hypothesis = SpanishNLIDetector._extract_numbers(hypothesis)
        if not numbers_premise or not numbers_hypothesis:
            return 0.0
        distances = []
        for p in numbers_premise:
            for h in numbers_hypothesis:
                distances.append(abs(p - h))
        if not distances:
            return 0.0
        min_distance = min(distances)
        if min_distance == 0:
            return 0.0
        score = 1 - math.tanh(min_distance / 100)
        return max(0.0, min(0.9, score))

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        matches = re.findall(r"-?\d+(?:[.,]\d+)?", text)
        numbers = []
        for match in matches:
            normalized = match.replace(".", "").replace(",", ".")
            try:
                numbers.append(float(normalized))
            except ValueError:
                continue
        return numbers

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())


__all__ = ["SpanishNLIDetector"]
