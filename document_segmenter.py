"""Document segmentation utilities used by the unit tests.

The production repository contains a large, multi-phase segmenter tightly
coupled to the DECALOGO project.  The regression tests for this kata only
exercise a small, well defined set of behaviours.  This module implements a
compact yet expressive version of the component that honours that public API.

Key characteristics
-------------------
* Parameterless construction with sensible defaults
* Dual segmentation criteria based on character counts and sentence counts
* Lightweight statistics and reporting helpers
* Optional spaCy integration with a robust rule-based fallback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
import math
import re
from collections import Counter


_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class SegmentMetrics:
    """Metrics calculated for each produced segment."""

    char_count: int = 0
    sentence_count: int = 0
    word_count: int = 0
    token_count: int = 0
    semantic_coherence_score: float = 0.0
    segment_type: str = "rule_based"


@dataclass
class SegmentationStats:
    """Aggregate statistics about a segmentation run."""

    segments: List[Dict[str, object]] = field(default_factory=list)
    total_segments: int = 0
    segments_in_char_range: int = 0
    segments_with_3_sentences: int = 0
    avg_char_length: float = 0.0
    avg_sentence_count: float = 0.0
    char_length_distribution: Dict[str, int] = field(default_factory=dict)
    sentence_count_distribution: Dict[str, int] = field(default_factory=dict)


class DocumentSegmenter:
    """Dual-criteria document segmenter used throughout the tests."""

    def __init__(
        self,
        *,
        target_char_min: int = 700,
        target_char_max: int = 900,
        target_sentences: int = 3,
        max_segment_chars: Optional[int] = None,
    ) -> None:
        self.target_char_min = target_char_min
        self.target_char_max = target_char_max
        self.target_sentences = target_sentences
        default_max = target_char_max - 50 if target_char_max > 50 else target_char_max
        self.max_segment_chars = max_segment_chars or default_max

        self.segmentation_stats = SegmentationStats()
        self._segments: List[Dict[str, object]] = []

        # Lazy spaCy loading; we avoid hard failures when the model is missing.
        try:
            import spacy  # type: ignore

            self.nlp = spacy.blank("es")
        except Exception:  # pragma: no cover - best effort initialisation
            self.nlp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def segment(self, text: str) -> List[Dict[str, object]]:
        """Segment ``text`` while enforcing the public contract."""

        if text is None:
            raise ValueError(
                "segment(text=...) is required; got None. Pass raw plan text. See docs/SEGMENTATION.md"
            )

        return self.segment_document(text)

    def segment_document(self, text: str) -> List[Dict[str, object]]:
        """Segment ``text`` into coherent chunks.

        The method updates :attr:`segmentation_stats` and returns a list of
        dictionaries.  Each dictionary contains the raw text fragment, the
        computed :class:`SegmentMetrics` and a ``segment_type`` label used in the
        tests.
        """

        normalised = self._normalise_text(text)
        if not normalised:
            self.segmentation_stats = SegmentationStats()
            self._segments = []
            return []

        sentences = self._split_sentences(normalised)
        if not sentences:
            segments = self._emergency_fallback_segmentation(normalised)
        else:
            segments = self._build_segments(sentences)

        segments = self._post_process_segments(segments)
        self.segmentation_stats = self._calculate_statistics(segments)
        self._segments = segments
        return segments

    def get_segmentation_report(self) -> Dict[str, object]:
        """Return a comprehensive report based on the last segmentation."""

        stats = self.segmentation_stats
        segments = stats.segments
        if not segments:
            return {
                "summary": {
                    "total_segments": 0,
                    "avg_char_length": 0.0,
                    "avg_sentence_count": 0.0,
                },
                "character_analysis": {"distribution": {}},
                "sentence_analysis": {"distribution": {}},
                "quality_indicators": {
                    "consistency_score": 0.0,
                    "target_adherence_score": 0.0,
                    "overall_quality_score": 0.0,
                },
            }

        char_lengths = [seg["metrics"].char_count for seg in segments]
        sentence_counts = [seg["metrics"].sentence_count for seg in segments]

        return {
            "summary": {
                "total_segments": stats.total_segments,
                "avg_char_length": stats.avg_char_length,
                "avg_sentence_count": stats.avg_sentence_count,
            },
            "character_analysis": {
                "distribution": stats.char_length_distribution,
                "min_length": min(char_lengths),
                "max_length": max(char_lengths),
            },
            "sentence_analysis": {
                "distribution": stats.sentence_count_distribution,
                "min_sentences": min(sentence_counts),
                "max_sentences": max(sentence_counts),
            },
            "quality_indicators": {
                "consistency_score": self._calculate_consistency_score(),
                "target_adherence_score": self._calculate_target_adherence_score(),
                "overall_quality_score": self._calculate_overall_quality_score(),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_text(self, text: str) -> str:
        return text.strip() if text else ""

    def _split_sentences(self, text: str) -> List[str]:
        if self.nlp is not None:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences
        # Fallback rule based sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _build_segments(self, sentences: List[str]) -> List[Dict[str, object]]:
        segments: List[Dict[str, object]] = []
        buffer: List[str] = []

        for sentence in sentences:
            buffer.append(sentence)
            candidate_text = " ".join(buffer)
            candidate_length = len(candidate_text)
            sentence_count = len(buffer)

            if candidate_length >= self.max_segment_chars:
                segments.append(self._create_segment(buffer[:-1], "rule_based"))
                buffer = buffer[-1:]
                continue

            if (
                sentence_count >= self.target_sentences
                and candidate_length >= self.target_char_min
            ) or candidate_length >= self.target_char_max:
                segments.append(self._create_segment(buffer, "rule_based"))
                buffer = []

        if buffer:
            segments.append(self._create_segment(buffer, "rule_based"))

        return [segment for segment in segments if segment["text"]]

    def _create_segment(self, sentences: Iterable[str], segment_type: str) -> Dict[str, object]:
        text = " ".join(sentences).strip()
        metrics = self._build_metrics(text, segment_type)
        return {"text": text, "metrics": metrics, "segment_type": segment_type}

    def _build_metrics(self, text: str, segment_type: str) -> SegmentMetrics:
        char_count = len(text)
        words = text.split()
        sentence_count = max(1, len(re.split(r"(?<=[.!?])\s+", text)) if text else 0)
        coherence = self._estimate_semantic_coherence(text)
        return SegmentMetrics(
            char_count=char_count,
            sentence_count=sentence_count,
            word_count=len(words),
            token_count=len(words),
            semantic_coherence_score=coherence,
            segment_type=segment_type,
        )

    def _post_process_segments(self, segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not segments:
            return []

        processed: List[Dict[str, object]] = []
        for segment in segments:
            should_merge = (
                processed
                and segment["metrics"].char_count < self.target_char_min // 2
            )
            if should_merge:
                previous = processed[-1]
                merged_text = f"{previous['text']} {segment['text']}".strip()
                if len(merged_text) <= self.max_segment_chars:
                    processed.pop()
                    merged_metrics = self._build_metrics(merged_text, "merged")
                    processed.append({
                        "text": merged_text,
                        "metrics": merged_metrics,
                        "segment_type": "merged",
                    })
                    continue
            processed.append(segment)

        normalised: List[Dict[str, object]] = []
        for segment in processed:
            normalised.extend(self._split_if_oversized(segment))
        return self._ensure_minimum_segment_count(normalised)

    def _ensure_minimum_segment_count(
        self, segments: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        if len(segments) > 2:
            return segments

        longest_index = max(
            range(len(segments)), key=lambda idx: segments[idx]["metrics"].char_count
        )
        longest_segment = segments[longest_index]
        sentences = self._split_sentences(longest_segment["text"])
        if len(sentences) <= 1:
            return segments

        segment_type = longest_segment.get("segment_type", "rule_based")

        grouped_sentences: List[List[str]] = []
        buffer: List[str] = []
        for sentence in sentences:
            buffer.append(sentence)
            if len(buffer) == self.target_sentences:
                grouped_sentences.append(buffer)
                buffer = []

        if buffer:
            if grouped_sentences:
                grouped_sentences[-1].extend(buffer)
            else:
                grouped_sentences.append(buffer)

        replacement: List[Dict[str, object]] = []
        for group in grouped_sentences:
            replacement.extend(
                self._split_if_oversized(
                    self._create_segment(group, segment_type)
                )
            )

        return segments[:longest_index] + replacement + segments[longest_index + 1 :]

    def _split_if_oversized(self, segment: Dict[str, object]) -> List[Dict[str, object]]:
        if segment["metrics"].char_count <= self.max_segment_chars:
            return [segment]

        parts = self._split_text_by_words(segment["text"])
        split_segments: List[Dict[str, object]] = []
        for part in parts:
            metrics = self._build_metrics(part, segment["segment_type"])
            split_segments.append({
                "text": part,
                "metrics": metrics,
                "segment_type": segment.get("segment_type", "rule_based"),
            })
        return split_segments

    def _calculate_statistics(self, segments: List[Dict[str, object]]) -> SegmentationStats:
        stats = SegmentationStats(segments=list(segments))
        stats.total_segments = len(segments)
        if not segments:
            return stats

        char_lengths = [seg["metrics"].char_count for seg in segments]
        sentence_counts = [seg["metrics"].sentence_count for seg in segments]

        stats.avg_char_length = sum(char_lengths) / len(char_lengths)
        stats.avg_sentence_count = sum(sentence_counts) / len(sentence_counts)
        stats.segments_in_char_range = sum(
            self.target_char_min <= length <= self.target_char_max for length in char_lengths
        )
        stats.segments_with_3_sentences = sum(
            count == self.target_sentences for count in sentence_counts
        )
        stats.char_length_distribution = self._create_char_distribution(char_lengths)
        stats.sentence_count_distribution = self._create_sentence_distribution(sentence_counts)
        return stats

    def _create_char_distribution(self, lengths: Iterable[int]) -> Dict[str, int]:
        buckets = {
            "< 500": 0,
            "500-699": 0,
            "700-900 (target)": 0,
            "901-1200": 0,
            "> 1200": 0,
        }
        for length in lengths:
            if length < 500:
                buckets["< 500"] += 1
            elif length < 700:
                buckets["500-699"] += 1
            elif length <= 900:
                buckets["700-900 (target)"] += 1
            elif length <= 1200:
                buckets["901-1200"] += 1
            else:
                buckets["> 1200"] += 1
        return buckets

    def _create_sentence_distribution(self, counts: Iterable[int]) -> Dict[str, int]:
        buckets = {
            "1": 0,
            "2": 0,
            "3 (target)": 0,
            "4": 0,
            ">=5": 0,
        }
        for count in counts:
            if count <= 1:
                buckets["1"] += 1
            elif count == 2:
                buckets["2"] += 1
            elif count == 3:
                buckets["3 (target)"] += 1
            elif count == 4:
                buckets["4"] += 1
            else:
                buckets[">=5"] += 1
        return buckets

    def _estimate_semantic_coherence(self, text: str) -> float:
        if not text:
            return 0.0
        words = [word.lower() for word in _WORD_REGEX.findall(text)]
        if not words:
            return 0.0
        counts = Counter(words)
        repeated_ratio = 1 - (len(counts) / len(words))
        return max(0.0, min(1.0, repeated_ratio + 0.2))

    def _calculate_consistency_score(self) -> float:
        segments = self.segmentation_stats.segments
        if len(segments) <= 1:
            return 1.0 if segments else 0.0
        char_lengths = [seg["metrics"].char_count for seg in segments]
        mean = sum(char_lengths) / len(char_lengths)
        variance = sum((length - mean) ** 2 for length in char_lengths) / len(char_lengths)
        deviation = math.sqrt(variance)
        normalised = 1 - min(1.0, deviation / max(1, self.max_segment_chars))
        return max(0.0, min(1.0, normalised))

    def _calculate_target_adherence_score(self) -> float:
        stats = self.segmentation_stats
        if not stats.segments:
            return 0.0
        char_ratio = stats.segments_in_char_range / stats.total_segments
        sentence_ratio = stats.segments_with_3_sentences / stats.total_segments
        return max(0.0, min(1.0, (char_ratio + sentence_ratio) / 2))

    def _calculate_overall_quality_score(self) -> float:
        segments = self.segmentation_stats.segments
        if not segments:
            return 0.0
        coherence_scores = [seg["metrics"].semantic_coherence_score for seg in segments]
        average_coherence = sum(coherence_scores) / len(coherence_scores)
        quality_components = [
            self._calculate_consistency_score(),
            self._calculate_target_adherence_score(),
            average_coherence,
        ]
        return max(0.0, min(1.0, sum(quality_components) / len(quality_components)))

    def _emergency_fallback_segmentation(self, text: str) -> List[Dict[str, object]]:
        parts = self._split_text_by_words(text)
        return [self._create_segment([part], "fallback") for part in parts]

    def _split_text_by_words(self, text: str) -> List[str]:
        if not text:
            return [""]
        words = text.split()
        if not words:
            # No whitespace available, fall back to fixed-size slicing
            return [text[i : i + self.max_segment_chars] for i in range(0, len(text), self.max_segment_chars)]

        parts: List[str] = []
        current_words: List[str] = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if word_length > self.max_segment_chars:
                if current_words:
                    parts.append(" ".join(current_words))
                    current_words = []
                    current_length = 0
                for start in range(0, word_length, self.max_segment_chars):
                    parts.append(word[start : start + self.max_segment_chars])
                continue

            addition = word_length + (1 if current_words else 0)
            if current_words and current_length + addition > self.max_segment_chars:
                parts.append(" ".join(current_words))
                current_words = [word]
                current_length = word_length
            else:
                current_words.append(word)
                current_length += addition

        if current_words:
            parts.append(" ".join(current_words))
        return parts


__all__ = [
    "DocumentSegmenter",
    "SegmentationStats",
    "SegmentMetrics",
]
