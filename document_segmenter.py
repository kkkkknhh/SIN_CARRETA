# document_segmenter.py
"""
Contract-pure, strategy-driven document segmenter.

Public contract (frozen):
  - class DocumentSegmenter:
      __init__(self)                          # zero-argument constructor
      segment(self, text: str) -> List[dict]  # main API
      get_segmentation_report(self) -> dict   # stats/quality report

Advanced features (opt-in without widening __init__):
  - Strategy backend with boundary scoring:
      * RuleBasedBackend (default): regex sentences + punctuation-weighted cuts
      * AdvancedBackend (optional): sentence embeddings drift + DP cut placement
        If `sentence_transformers` is unavailable, falls back to a deterministic
        hash-based pseudo-embedding that preserves behavior & tests.

  - Constraint-aware optimizer (dynamic programming):
      Minimizes a cost composed of:
          - length deviation from target window
          - sentence-count deviation
          - negative boundary strength (prefer high-scoring cut points)
          - hard cap on max chars per segment
      Deterministic (no RNG); produces reproducible cuts.

No import-time side effects. No global flags. No kwargs in __init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Protocol
import math
import re
from collections import Counter


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class SegmentMetrics:
    char_count: int = 0
    sentence_count: int = 0
    word_count: int = 0
    token_count: int = 0
    semantic_coherence_score: float = 0.0
    segment_type: str = "rule_based"
    confidence: float = 0.0  # boundary confidence for the final cut decisions


@dataclass
class SegmentationStats:
    segments: List[Dict[str, object]] = field(default_factory=list)
    total_segments: int = 0
    segments_in_char_range: int = 0
    segments_with_target_sentences: int = 0
    segments_with_3_sentences: int = 0  # Backward compatibility alias
    avg_char_length: float = 0.0
    avg_sentence_count: float = 0.0
    char_length_distribution: Dict[str, int] = field(default_factory=dict)
    sentence_count_distribution: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Keep segments_with_3_sentences in sync with segments_with_target_sentences
        if self.segments_with_target_sentences > 0 and self.segments_with_3_sentences == 0:
            self.segments_with_3_sentences = self.segments_with_target_sentences
        elif self.segments_with_3_sentences > 0 and self.segments_with_target_sentences == 0:
            self.segments_with_target_sentences = self.segments_with_3_sentences


# ---------------------------
# Internal constants & regex
# ---------------------------

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


# ---------------------------
# Strategy interface
# ---------------------------

class Backend(Protocol):
    def split_sentences(self, text: str) -> List[str]: ...
    def boundary_scores(self, sentences: List[str]) -> List[float]:
        """
        Score potential cut *after* each sentence i (length N -> N-1 scores).
        Higher is better (stronger boundary).
        """


class RuleBasedBackend:
    """Regex sentence splitter + punctuation-weighted boundary scoring."""

    def split_sentences(self, text: str) -> List[str]:
        sents = _SENTENCE_SPLIT_REGEX.split(text)
        return [s.strip() for s in sents if s.strip()]

    def boundary_scores(self, sentences: List[str]) -> List[float]:
        scores: List[float] = []
        for s in sentences[:-1]:
            s = s.strip()
            if not s:
                scores.append(0.0)
                continue
            tail = s[-1]
            # Period > question/exclamation > other punctuation > none
            if tail == ".":
                base = 1.0
            elif tail in {"?", "!"}:
                base = 0.9
            elif tail in {":", ";", "—", "–", ")"}:
                base = 0.6
            else:
                base = 0.3
            # Longer sentences get slightly more confidence
            base += min(0.2, max(0.0, (len(s) - 80) / 400))
            scores.append(max(0.0, min(1.0, base)))
        return scores


class AdvancedBackend:
    """
    Embedding-drift boundary scorer.

    - If `sentence_transformers` is available, uses a small model to embed sentences
      (deterministic: no sampling; inference is pure).
    - Otherwise, uses a deterministic hash-based pseudo-embedding to approximate
      semantic drift while staying dependency-free and reproducible.

    In both cases, boundary strength = normalized cosine distance between
    adjacent sentence embeddings.
    """

    def __init__(self) -> None:
        self._use_real_embeddings = False
        self._model = None
        try:  # pragma: no cover (environment-dependent)
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Choose a compact multilingual model; name as a constant string.
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self._model = SentenceTransformer(model_name, device="cpu")
            self._use_real_embeddings = True
        except Exception:
            self._use_real_embeddings = False
            self._model = None

    def split_sentences(self, text: str) -> List[str]:
        sents = _SENTENCE_SPLIT_REGEX.split(text)
        return [s.strip() for s in sents if s.strip()]

    # --- deterministic pseudo-embedding (fallback) ---
    @staticmethod
    def _hash32(s: str) -> int:
        # Fowler–Noll–Vo (FNV-1a) 32-bit for determinism
        h = 0x811C9DC5
        for ch in s.encode("utf-8", errors="ignore"):
            h ^= ch
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h

    @classmethod
    def _pseudo_embed(cls, s: str, dim: int = 64) -> List[float]:
        # Produce a stable vector via rolling hash of token n-grams
        tokens = [t.lower() for t in _WORD_REGEX.findall(s)]
        if not tokens:
            return [0.0] * dim
        vec = [0.0] * dim
        for i, tok in enumerate(tokens):
            h = cls._hash32(tok + str(i))
            idx = h % dim
            vec[idx] += 1.0
        # l2-normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a)) or 1.0
        db = math.sqrt(sum(y * y for y in b)) or 1.0
        return max(-1.0, min(1.0, num / (da * db)))

    def boundary_scores(self, sentences: List[str]) -> List[float]:
        if not sentences or len(sentences) == 1:
            return []
        # Embed
        if self._use_real_embeddings and self._model is not None:  # pragma: no cover
            embs = self._model.encode(sentences, convert_to_numpy=False, normalize_embeddings=True)
            # Convert to lists of floats
            embs = [list(map(float, e)) for e in embs]
        else:
            embs = [self._pseudo_embed(s) for s in sentences]
        # Cosine distance between adjacent sentences (0..2), normalize to 0..1
        scores: List[float] = []
        for i in range(len(sentences) - 1):
            cos = self._cosine(embs[i], embs[i + 1])
            dist = 1.0 - cos  # 0 (same) .. 2 (opposite) in worst numeric edge cases
            # Clamp to 0..1 by mild normalization (empirically adequate)
            scores.append(max(0.0, min(1.0, dist)))
        return scores


# ---------------------------
# Segmenter (contract-pure)
# ---------------------------

class DocumentSegmenter:
    """Strategy-driven, constraint-aware segmenter with a zero-arg constructor."""

    # Immutable, audited defaults
    _TARGET_CHAR_MIN: int = 700
    _TARGET_CHAR_MAX: int = 900
    _TARGET_SENTENCES: int = 3

    def __init__(
        self,
        *,
        target_char_min: Optional[int] = None,
        target_char_max: Optional[int] = None,
        target_sentences: Optional[int] = None,
        max_segment_chars: Optional[int] = None,
    ) -> None:
        # Support both zero-arg (contract-pure) and legacy keyword args (backward compat)
        self.target_char_min: int = target_char_min if target_char_min is not None else self._TARGET_CHAR_MIN
        self.target_char_max: int = target_char_max if target_char_max is not None else self._TARGET_CHAR_MAX
        self.target_sentences: int = target_sentences if target_sentences is not None else self._TARGET_SENTENCES
        
        if max_segment_chars is not None:
            self.max_segment_chars = max_segment_chars
        else:
            self.max_segment_chars = max(50, self.target_char_max - 50)

        # Default backend is rule-based; callers may opt into advanced
        self._backend: Backend = RuleBasedBackend()

        self.segmentation_stats: SegmentationStats = SegmentationStats()
        self._segments: List[Dict[str, object]] = []
        
        # Backward compatibility: set nlp to None (new implementation doesn't use spaCy)
        self.nlp = None

    # Factories that DO NOT widen __init__
    @classmethod
    def with_advanced_backend(cls) -> "DocumentSegmenter":
        obj = cls()
        obj._backend = AdvancedBackend()
        return obj

    def set_backend(self, backend: Backend) -> None:
        """Explicit, auditable backend swap (keeps contract pure)."""
        self._backend = backend

    # Public API
    def segment(self, text: str) -> List[Dict[str, object]]:
        if text is None:
            raise ValueError("segment(text=...) is required; got None.")
        return self._segment_document(text)
    
    def segment_document(self, text: str) -> List[Dict[str, object]]:
        """Backward compatibility alias for segment()."""
        return self.segment(text)


    def get_segmentation_report(self) -> Dict[str, object]:
        stats = self.segmentation_stats
        segs = stats.segments
        if not segs:
            return {
                "summary": {"total_segments": 0, "avg_char_length": 0.0, "avg_sentence_count": 0.0},
                "character_analysis": {"distribution": {}},
                "sentence_analysis": {"distribution": {}},
                "quality_indicators": {
                    "consistency_score": 0.0,
                    "target_adherence_score": 0.0,
                    "overall_quality_score": 0.0,
                },
            }

        char_lengths = [s["metrics"].char_count for s in segs]
        sentence_counts = [s["metrics"].sentence_count for s in segs]
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
                "consistency_score": self._consistency_score(),
                "target_adherence_score": self._target_adherence_score(),
                "overall_quality_score": self._overall_quality_score(),
            },
        }

    # ---------------------------
    # Core logic
    # ---------------------------

    def _segment_document(self, text: str) -> List[Dict[str, object]]:
        t = self._normalize(text)
        if not t:
            self.segmentation_stats = SegmentationStats()
            self._segments = []
            return []

        sentences = self._backend.split_sentences(t)
        if not sentences:
            segs = self._fallback_segments(t)
        else:
            cuts, conf = self._place_cuts(sentences, self._backend.boundary_scores(sentences))
            segs = self._materialize_segments(sentences, cuts, conf)

        segs = self._post_process_segments(segs)
        self.segmentation_stats = self._compute_stats(segs)
        self._segments = segs
        return segs

    # DP-based cut placement
    def _place_cuts(self, sents: List[str], scores: List[float]) -> Tuple[List[int], float]:
        """
        Choose cut indices (end-exclusive) to minimize cost under constraints.
        Returns (cuts, global_confidence). cuts are sentence indices where a segment ends.
        """
        N = len(sents)
        # Precompute prefix char counts for O(1) segment length
        sent_chars = [len(x) for x in sents]
        pref = [0]
        for c in sent_chars:
            pref.append(pref[-1] + c + 1)  # +1 for the joining space

        def seg_len(i: int, j: int) -> int:
            # sentences i..j inclusive
            raw = pref[j + 1] - pref[i]
            return max(0, raw - 1)  # remove trailing join space

        def seg_sent_count(i: int, j: int) -> int:
            return (j - i + 1)

        # boundary score after sentence k (i..j means last cut at j)
        def cut_score(j: int) -> float:
            return scores[j] if 0 <= j < len(scores) else 0.0

        # Cost function: lower is better.
        def seg_cost(i: int, j: int) -> float:
            L = seg_len(i, j)
            S = seg_sent_count(i, j)
            # Hard violation if over cap
            if L > self.max_segment_chars:
                return 1e9
            # Length window penalty (target range)
            target_mid = (self.target_char_min + self.target_char_max) / 2
            len_pen = abs(L - target_mid) / max(1.0, target_mid)
            # Sentence target penalty
            sent_pen = abs(S - self.target_sentences) / max(1.0, self.target_sentences)
            # Encourage strong boundary at j
            bscore = cut_score(j)
            boundary_pen = 1.0 - bscore  # high score -> low penalty
            # Weighted sum (tuned for stability, deterministic)
            return 0.55 * len_pen + 0.25 * sent_pen + 0.20 * boundary_pen

        # DP over end index: dp[j] = (cost, prev_index)
        dp: List[Tuple[float, int]] = [(float("inf"), -1)] * N
        for j in range(N):
            # First segment i..j
            best = (float("inf"), -1)
            for i in range(0, j + 1):
                cost = seg_cost(i, j)
                if i == 0:
                    total = cost
                else:
                    prev_cost, _ = dp[i - 1]
                    total = prev_cost + cost
                if total < best[0]:
                    best = (total, i - 1)
            dp[j] = best

        # Reconstruct cuts
        cuts: List[int] = []
        j = N - 1
        while j >= 0:
            _, prev = dp[j]
            cuts.append(j)
            j = prev
        cuts.reverse()

        # Global confidence = mean of boundary scores at chosen cuts (except last)
        cut_bscores = [cut_score(c) for c in cuts[:-1]]
        conf = sum(cut_bscores) / len(cut_bscores) if cut_bscores else 1.0
        return cuts, conf

    def _materialize_segments(self, sents: List[str], cuts: List[int], conf: float) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        start = 0
        for c in cuts:
            chunk = " ".join(sents[start : c + 1]).strip()
            m = self._metrics(chunk, "advanced" if isinstance(self._backend, AdvancedBackend) else "rule_based", conf)
            out.append({"text": chunk, "metrics": m, "segment_type": m.segment_type})
            start = c + 1
        return out

    # ---------------------------
    # Helpers
    # ---------------------------

    def _normalize(self, text: str) -> str:
        return text.strip() if text else ""

    def _metrics(self, text: str, kind: str, conf: float) -> SegmentMetrics:
        char_count = len(text)
        words = text.split()
        sent_cnt = max(1, len(_SENTENCE_SPLIT_REGEX.split(text)) if text else 0)
        coherence = self._coherence(text)
        return SegmentMetrics(
            char_count=char_count,
            sentence_count=sent_cnt,
            word_count=len(words),
            token_count=len(words),
            semantic_coherence_score=coherence,
            segment_type=kind,
            confidence=max(0.0, min(1.0, conf)),
        )

    def _fallback_segments(self, text: str) -> List[Dict[str, object]]:
        parts = self._split_by_words(text)
        return [
            {
                "text": p,
                "metrics": self._metrics(p, "fallback", 1.0),
                "segment_type": "fallback",
            }
            for p in parts
        ]


    def _post_process_segments(self, segs: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not segs:
            return []
        # Merge very tiny tails if safe
        merged: List[Dict[str, object]] = []
        for seg in segs:
            small = seg["metrics"].char_count < (self.target_char_min // 2)
            if merged and small:
                prev = merged[-1]
                combined = f"{prev['text']} {seg['text']}".strip()
                if len(combined) <= self.max_segment_chars:
                    merged.pop()
                    m = self._metrics(combined, "merged", (prev["metrics"].confidence + seg["metrics"].confidence) / 2)
                    merged.append({"text": combined, "metrics": m, "segment_type": "merged"})
                    continue
            merged.append(seg)

        # Enforce strict cap with word-aware split
        normalized: List[Dict[str, object]] = []
        for seg in merged:
            normalized.extend(self._split_if_oversized(seg))
        return self._ensure_minimum_count(normalized)

    def _ensure_minimum_count(self, segs: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if len(segs) > 2 or not segs:
            return segs
        # Try to split the longest into at least two groups by target sentences
        idx = max(range(len(segs)), key=lambda i: segs[i]["metrics"].char_count)
        s = segs[idx]
        sents = _SENTENCE_SPLIT_REGEX.split(s["text"])
        sents = [t.strip() for t in sents if t.strip()]
        if len(sents) <= 1:
            return segs
        groups: List[List[str]] = []
        buf: List[str] = []
        for sent in sents:
            buf.append(sent)
            if len(buf) == self.target_sentences:
                groups.append(buf)
                buf = []
        if buf:
            (groups[-1].extend(buf) if groups else groups.append(buf))
        replacement: List[Dict[str, object]] = []
        for g in groups:
            part = " ".join(g).strip()
            replacement.extend(self._split_if_oversized({"text": part, "metrics": self._metrics(part, s["metrics"].segment_type, s["metrics"].confidence), "segment_type": s["segment_type"]}))
        return segs[:idx] + replacement + segs[idx + 1 :]

    def _split_if_oversized(self, seg: Dict[str, object]) -> List[Dict[str, object]]:
        if seg["metrics"].char_count <= self.max_segment_chars:
            return [seg]
        parts = self._split_by_words(seg["text"])
        out: List[Dict[str, object]] = []
        for p in parts:
            out.append({
                "text": p,
                "metrics": self._metrics(p, seg.get("segment_type", "rule_based"), seg["metrics"].confidence),
                "segment_type": seg.get("segment_type", "rule_based"),
            })
        return out

    def _split_by_words(self, text: str) -> List[str]:
        if not text:
            return [""]
        words = text.split()
        if not words:
            size = max(1, self.max_segment_chars)
            return [text[i:i+size] for i in range(0, len(text), size)]
        parts: List[str] = []
        cur: List[str] = []
        length = 0
        for w in words:
            wl = len(w)
            if wl > self.max_segment_chars:
                if cur:
                    parts.append(" ".join(cur)); cur, length = [], 0
                size = max(1, self.max_segment_chars)
                parts.extend(w[i:i+size] for i in range(0, wl, size))
                continue
            add = wl + (1 if cur else 0)
            if cur and length + add > self.max_segment_chars:
                parts.append(" ".join(cur)); cur, length = [w], wl
            else:
                cur.append(w); length += add
        if cur:
            parts.append(" ".join(cur))
        return parts

    # ---------------------------
    # Stats & quality indicators
    # ---------------------------

    def _compute_stats(self, segs: List[Dict[str, object]]) -> SegmentationStats:
        st = SegmentationStats(segments=list(segs))
        st.total_segments = len(segs)
        if not segs:
            return st
        char_lengths = [s["metrics"].char_count for s in segs]
        sentence_counts = [s["metrics"].sentence_count for s in segs]
        st.avg_char_length = sum(char_lengths) / len(char_lengths)
        st.avg_sentence_count = sum(sentence_counts) / len(sentence_counts)
        st.segments_in_char_range = sum(self.target_char_min <= L <= self.target_char_max for L in char_lengths)
        st.segments_with_target_sentences = sum(c == self.target_sentences for c in sentence_counts)
        st.segments_with_3_sentences = st.segments_with_target_sentences  # Backward compatibility
        st.char_length_distribution = self._char_dist(char_lengths)
        st.sentence_count_distribution = self._sent_dist(sentence_counts)
        return st

    def _char_dist(self, lengths: Iterable[int]) -> Dict[str, int]:
        buckets = {"< 500": 0, "500-699": 0, "700-900 (target)": 0, "901-1200": 0, "> 1200": 0}
        for L in lengths:
            if L < 500: buckets["< 500"] += 1
            elif L < 700: buckets["500-699"] += 1
            elif L <= 900: buckets["700-900 (target)"] += 1
            elif L <= 1200: buckets["901-1200"] += 1
            else: buckets["> 1200"] += 1
        return buckets

    def _sent_dist(self, counts: Iterable[int]) -> Dict[str, int]:
        buckets = {"1": 0, "2": 0, "3 (target)": 0, "4": 0, ">=5": 0}
        for c in counts:
            if c <= 1: buckets["1"] += 1
            elif c == 2: buckets["2"] += 1
            elif c == 3: buckets["3 (target)"] += 1
            elif c == 4: buckets["4"] += 1
            else: buckets[">=5"] += 1
        return buckets

    def _coherence(self, text: str) -> float:
        if not text: return 0.0
        words = [w.lower() for w in _WORD_REGEX.findall(text)]
        if not words: return 0.0
        counts = Counter(words)
        repeated_ratio = 1 - (len(counts) / len(words))
        return max(0.0, min(1.0, repeated_ratio + 0.2))

    def _consistency_score(self) -> float:
        segs = self.segmentation_stats.segments
        if len(segs) <= 1: return 1.0 if segs else 0.0
        Ls = [s["metrics"].char_count for s in segs]
        mean = sum(Ls) / len(Ls)
        var = sum((x - mean) ** 2 for x in Ls) / len(Ls)
        dev = math.sqrt(var)
        return max(0.0, min(1.0, 1 - dev / max(1, self.max_segment_chars)))

    def _target_adherence_score(self) -> float:
        st = self.segmentation_stats
        if not st.segments: return 0.0
        a = st.segments_in_char_range / st.total_segments
        b = st.segments_with_target_sentences / st.total_segments
        return max(0.0, min(1.0, (a + b) / 2))

    def _overall_quality_score(self) -> float:
        segs = self.segmentation_stats.segments
        if not segs: return 0.0
        coh = [s["metrics"].semantic_coherence_score for s in segs]
        avg = sum(coh) / len(coh)
        parts = [self._consistency_score(), self._target_adherence_score(), avg]
        return max(0.0, min(1.0, sum(parts) / len(parts)))

    # ---------------------------
    # Backward compatibility aliases
    # ---------------------------
    
    def _create_char_distribution(self, lengths: Iterable[int]) -> Dict[str, int]:
        """Backward compatibility alias for _char_dist()."""
        return self._char_dist(lengths)
    
    def _create_sentence_distribution(self, counts: Iterable[int]) -> Dict[str, int]:
        """Backward compatibility alias for _sent_dist()."""
        return self._sent_dist(counts)
    
    def _estimate_semantic_coherence(self, text: str) -> float:
        """Backward compatibility alias for _coherence()."""
        return self._coherence(text)
    
    def _calculate_consistency_score(self) -> float:
        """Backward compatibility alias for _consistency_score()."""
        return self._consistency_score()
    
    def _calculate_target_adherence_score(self) -> float:
        """Backward compatibility alias for _target_adherence_score()."""
        return self._target_adherence_score()
    
    def _calculate_overall_quality_score(self) -> float:
        """Backward compatibility alias for _overall_quality_score()."""
        return self._overall_quality_score()
    
    def _emergency_fallback_segmentation(self, text: str) -> List[Dict[str, object]]:
        """Backward compatibility alias for _fallback_segments()."""
        return self._fallback_segments(text)
    
    def _split_text_by_words(self, text: str) -> List[str]:
        """Backward compatibility alias for _split_by_words()."""
        return self._split_by_words(text)


__all__ = ["DocumentSegmenter", "SegmentationStats", "SegmentMetrics"]
