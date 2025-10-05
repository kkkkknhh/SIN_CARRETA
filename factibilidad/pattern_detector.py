# coding=utf-8
"""
Pattern Detection Module for Policy Indicator Analysis
======================================================

Advanced pattern detection system for identifying baseline values, targets,
and timeframes in Spanish policy documents using comprehensive regex patterns.

This module provides specialized pattern detection capabilities for policy analysis,
focusing on identification of key components that determine indicator feasibility
and quality. It supports Spanish language patterns with high accuracy and includes
sophisticated overlap resolution and clustering algorithms.

Classes:
    PatternMatch: Represents a pattern match with position and confidence information
    PatternDetector: Main detector for baseline, target, and timeframe patterns

Example:
    >>> detector = PatternDetector()
    >>> patterns = detector.detect_patterns(
    ...     "Reducir la línea base de pobreza del 25% actual a una meta del 15% para el año 2025"
    ... )
    >>> for pattern_type, matches in patterns.items():
    ...     print(f"{pattern_type}: {len(matches)} matches")

Note:
    All patterns are optimized for Spanish policy documents and include
    comprehensive coverage of terminology used in municipal development plans.
"""

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PatternMatch:
    """
    Represents a pattern match with position and type information.

    Comprehensive representation of detected patterns including position,
    confidence, and classification information for advanced analysis.

    Args:
        pattern_type (str): Type of pattern matched ("baseline", "target", "timeframe")
        text (str): Actual matched text content
        start (int): Start position in original text
        end (int): End position in original text
        confidence (float, optional): Match confidence level (0.0-1.0). Defaults to 1.0.
    """

    pattern_type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0


class PatternDetector:
    """
    Advanced detector for baseline, target, and timeframe patterns in Spanish text.

    Comprehensive pattern detection system specifically designed for Spanish policy
    documents and municipal development plans. Includes sophisticated regex patterns,
    overlap resolution, and clustering capabilities for high-accuracy detection.

    Attributes:
        baseline_patterns (List[re.Pattern]): Compiled baseline detection patterns
        target_patterns (List[re.Pattern]): Compiled target/goal detection patterns
        timeframe_patterns (List[re.Pattern]): Compiled temporal framework patterns

    Methods:
        detect_patterns: Detect all pattern types in text
        find_pattern_clusters: Find text segments with all three pattern types

    Example:
        >>> detector = PatternDetector()
        >>> text = "Mejorar la línea base educativa del 70% a una meta del 85% en 2025"
        >>> results = detector.detect_patterns(text)
        >>> clusters = detector.find_pattern_clusters(text)

    Note:
        All patterns are case-insensitive and optimized for Spanish policy terminology.
        Includes comprehensive overlap resolution to prevent duplicate matches.

    """

    def __init__(self):
        """Initialize the pattern detector with compiled regex patterns."""
        self.baseline_patterns = self._compile_baseline_patterns()
        self.target_patterns = self._compile_target_patterns()
        self.timeframe_patterns = self._compile_timeframe_patterns()

    @staticmethod
    def _compile_baseline_patterns() -> List[re.Pattern]:
        """
        Compile comprehensive regex patterns for baseline indicators.

        Creates and compiles regex patterns for detecting baseline references
        in Spanish policy text including various synonyms and expressions.

        Returns:
            List[re.Pattern]: List of compiled regex patterns for baseline detection

        Note:
            Patterns include both formal terms (línea base) and informal expressions
            (situación actual, punto de partida) commonly used in policy documents.

        """
        patterns = [
            r"\b(?:línea\s+base|linea\s+base|línea\s+de\s+base|linea\s+de\s+base)\b",
            r"\b(?:situación\s+inicial|situacion\s+inicial)\b",
            r"\b(?:punto\s+de\s+partida)\b",
            r"\b(?:estado\s+actual)\b",
            r"\b(?:condición\s+inicial|condicion\s+inicial)\b",
            r"\b(?:nivel\s+base)\b",
            r"\b(?:valor\s+inicial)\b",
            r"\b(?:posición\s+inicial|posicion\s+inicial)\b",
            r"\b(?:baseline)\b",
            r"\b(?:actualmente|en\s+la\s+actualidad)\b",
            r"\b(?:al\s+inicio|inicialmente)\b",
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    @staticmethod
    def _compile_target_patterns() -> List[re.Pattern]:
        """
        Compile comprehensive regex patterns for target/goal indicators.

        Creates and compiles regex patterns for detecting target and goal references
        including objectives, aspirations, and expected outcomes.

        Returns:
            List[re.Pattern]: List of compiled regex patterns for target detection

        Note:
            Includes both direct terms (meta, objetivo) and action verbs (alcanzar, lograr)
            commonly used to express targets in policy documents.

        """
        patterns = [
            r"\b(?:meta|metas)\b",
            r"\b(?:objetivo|objetivos)\b",
            r"\b(?:alcanzar|lograr)\b",
            r"\b(?:conseguir|obtener)\b",
            r"\b(?:target|targets)\b",
            r"\b(?:propósito|proposito)\b",
            r"\b(?:finalidad)\b",
            r"\b(?:resultado\s+esperado)\b",
            r"\b(?:expectativa|expectativas)\b",
            r"\b(?:aspiración|aspiracion)\b",
            r"\b(?:pretende|pretender)\b",
            r"\b(?:busca|buscar)\b",
            r"\b(?:se\s+espera)\b",
            r"\b(?:se\s+proyecta)\b",
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    @staticmethod
    def _compile_timeframe_patterns() -> List[re.Pattern]:
        """
        Compile comprehensive regex patterns for timeframe indicators.

        Creates and compiles regex patterns for detecting temporal references
        including absolute dates, relative periods, and administrative timeframes.

        Returns:
            List[re.Pattern]: List of compiled regex patterns for timeframe detection

        Note:
            Covers multiple temporal expression types including specific years (2025),
            relative periods (próximos 3 años), quarters, months, and administrative
            periods (vigencia, PDD periods).

        """
        patterns = [
            # Absolute years
            r"\b(?:20\d{2})\b",
            # Relative time expressions
            r"\b(?:al\s+(?:20\d{2}|año\s+20\d{2}))\b",
            r"\b(?:en\s+(?:\d+\s+(?:años?|meses?|días?)))\b",
            r"\b(?:para\s+(?:el\s+)?(?:20\d{2}|fin\s+de\s+año))\b",
            r"\b(?:hasta\s+(?:el\s+)?20\d{2})\b",
            # Quarters and periods
            r"\b(?:[1-4]º?\s*(?:trimestre|cuatrimestre))\b",
            r"\b(?:primer|segundo|tercer|cuarto)\s+(?:trimestre|cuatrimestre)\b",
            r"\b(?:Q[1-4])\b",
            # Months + year
            r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?20\d{2}\b",
            # Relative temporal expressions
            r"\b(?:en\s+los\s+próximos\s+\d+\s+(?:años?|meses?))\b",
            r"\b(?:dentro\s+de\s+\d+\s+(?:años?|meses?))\b",
            r"\b(?:a\s+(?:corto|mediano|largo)\s+plazo)\b",
            r"\b(?:próximo\s+año|proximo\s+año)\b",
            r"\b(?:año\s+(?:que\s+viene|entrante))\b",
            # Date ranges
            r"\b(?:20\d{2}\s*[-–—]\s*20\d{2})\b",
            # Specific date patterns
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b",
            # Vigencias y periodos administrativos
            r"\b(?:vigencia\s+20\d{2})\b",
            r"\b(?:PDD\s*20\d{2}\s*[-–—]\s*20\d{2})\b",
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def detect_patterns(self, text: str) -> Dict[str, List[PatternMatch]]:
        """
        Detect all pattern types in the given text with comprehensive analysis.

        Performs comprehensive pattern detection across all supported types
        including baseline references, targets/goals, and timeframe indicators
        with sophisticated overlap resolution.

        Args:
            text (str): The text to analyze for pattern detection

        Returns:
            Dict[str, List[PatternMatch]]: Dictionary with pattern types as keys
                                         ('baseline', 'target', 'timeframe') and
                                         lists of PatternMatch objects as values

        Note:
            Automatically removes overlapping matches, keeping the longest/most
            specific match when multiple patterns overlap in the same text region.

        """
        return {
            "baseline": self._find_matches(text, self.baseline_patterns, "baseline"),
            "target": self._find_matches(text, self.target_patterns, "target"),
            "timeframe": self._find_matches(text, self.timeframe_patterns, "timeframe"),
        }

    @staticmethod
    def _find_matches(
        text: str, patterns: List[re.Pattern], pattern_type: str
    ) -> List[PatternMatch]:
        """
        Find all matches for a specific pattern type with overlap resolution.

        Args:
            text (str): Text to search for patterns
            patterns (List[re.Pattern]): Compiled regex patterns to apply
            pattern_type (str): Type identifier for the patterns being matched

        Returns:
            List[PatternMatch]: List of non-overlapping pattern matches, sorted by position
                               with preference for longer matches when overlaps occur

        Note:
            Implements sophisticated overlap resolution that keeps the longest match
            when multiple patterns overlap, ensuring no duplicate or conflicting matches.

        """
        matches: List[PatternMatch] = []
        for pattern in patterns:
            for m in pattern.finditer(text):
                matches.append(
                    PatternMatch(
                        pattern_type=pattern_type,
                        text=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Remove overlapping matches, keeping the longest one
        matches.sort(key=lambda x: (x.start, -(x.end - x.start)))
        filtered: List[PatternMatch] = []
        for m in matches:
            overlaps = any(
                (e.start <= m.start < e.end)
                or (e.start < m.end <= e.end)
                or (m.start <= e.start and m.end >= e.end)
                for e in filtered
            )
            if not overlaps:
                filtered.append(m)
        return filtered

    def find_pattern_clusters(
        self, text: str, proximity_window: int = 500
    ) -> List[Dict]:
        """Find text segments where all three pattern types appear within proximity.

        Identifies regions in the text where baseline, target, and timeframe
        patterns co-occur within a specified character distance, indicating
        comprehensive indicator descriptions.

        Identifies coherent text segments that contain baseline, target, and timeframe
        patterns within the specified proximity window, indicating complete indicator
        specifications that are likely to be high-quality and actionable.

        Args:
            text (str): The text to analyze for pattern clusters
            proximity_window (int, optional): Maximum character distance between patterns
                                            to be considered part of the same cluster.
                                            Defaults to 500 characters.

        Returns:
            List[Dict]: List of dictionaries containing cluster information with keys:
                       - 'start': Start position of cluster in text
                       - 'end': End position of cluster in text
                       - 'text': Actual text content of the cluster
                       - 'matches': Dictionary with pattern matches by type
                       - 'span': Total character span of the cluster

        Note:
            Only returns clusters that contain at least one match from each of the
            three pattern types (baseline, target, timeframe), indicating complete
            indicator specifications with measurable components.

        """
        all_matches = self.detect_patterns(text)
        clusters: List[Dict] = []

        baseline_matches = all_matches["baseline"]
        target_matches = all_matches["target"]
        timeframe_matches = all_matches["timeframe"]

        for baseline in baseline_matches:
            cluster = {"baseline": [baseline], "target": [], "timeframe": []}

            for target in target_matches:
                if PatternDetector._within_proximity(
                    baseline, target, proximity_window
                ):
                    cluster["target"].append(target)

            for timeframe in timeframe_matches:
                if PatternDetector._within_proximity(
                    baseline, timeframe, proximity_window
                ):
                    cluster["timeframe"].append(timeframe)

            if cluster["target"] and cluster["timeframe"]:
                all_m = cluster["baseline"] + \
                    cluster["target"] + cluster["timeframe"]
                start_pos = min(m.start for m in all_m)
                end_pos = max(m.end for m in all_m)

                clusters.append(
                    {
                        "start": start_pos,
                        "end": end_pos,
                        "text": text[start_pos:end_pos],
                        "matches": cluster,
                        "span": end_pos - start_pos,
                    }
                )

        return clusters

    @staticmethod
    def _within_proximity(
        a: PatternMatch, b: PatternMatch, proximity_window: int
    ) -> bool:
        """
        Check if two matches are within the specified proximity window.

        Calculates the minimum distance between two pattern matches to determine
        if they should be considered part of the same indicator cluster.

        Args:
            a (PatternMatch): First pattern match
            b (PatternMatch): Second pattern match
            proximity_window (int): Maximum allowed distance in characters

        Returns:
            bool: True if matches are within proximity window, False otherwise

        Note:
            Uses minimum distance calculation considering all possible position
            combinations (start-to-start, end-to-end, start-to-end, end-to-start)
            to accurately measure pattern proximity regardless of match lengths.

        """
        distance = min(
            abs(a.start - b.end),
            abs(a.end - b.start),
            abs(a.start - b.start),
            abs(a.end - b.end),
        )
        return distance <= proximity_window
