# coding=utf-8
"""
Plan Sanitizer Module

Provides robust text cleaning and normalization for plan documents,
ensuring that all key elements needed for DECALOGO evaluation are preserved.

Features:
- Unicode normalization
- Structure preservation (headings, sections, lists)
- Special character handling
- Key element identification and tagging
- Policy domain preservation

Contract (frozen):
- class PlanSanitizer:
    __init__(self)                      # ZERO-ARG constructor (no kwargs backdoors)
    sanitize_text(self, text: str) -> str
    sanitize_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str
    get_sanitization_stats(self) -> Dict[str, Any]
- Factory helpers that DO NOT widen __init__:
    PlanSanitizer.from_config(cfg: PlanSanitizerConfig) -> PlanSanitizer
    PlanSanitizer.legacy(**legacy_flags) -> PlanSanitizer
    create_plan_sanitizer(...legacy flags...) -> PlanSanitizer  # routes to .legacy()

Notes:
- No import-time side effects beyond logger configuration.
- Deterministic behavior, fail-closed on unknown legacy flags.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

# External utilities (kept as in original repo layout)
from text_processor import normalize_text, clean_policy_text  # type: ignore
from json_utils import safe_read_text_file  # type: ignore

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Key policy elements to preserve - aligned with DECALOGO questions
# ------------------------------------------------------------------------------
KEY_ELEMENTS: Dict[str, List[str]] = {
    # DE-1: Logical Intervention Framework
    "indicators": [
        r"(?i)indicador(?:es)?",
        r"(?i)meta(?:s)?",
        r"(?i)l[ií]nea(?:s)?\s+base",
        r"(?i)resultado(?:s)?\s+esperado(?:s)?",
    ],
    # DE-2: Thematic Inclusion
    "diagnostics": [
        r"(?i)diagn[óo]stico",
        r"(?i)situaci[óo]n\s+actual",
        r"(?i)problem[áa]tica",
        r"(?i)contexto",
    ],
    # DE-3: Participation and Governance
    "participation": [
        r"(?i)participaci[óo]n",
        r"(?i)consulta(?:s)?",
        r"(?i)mesa(?:s)?\s+t[ée]cnica(?:s)?",
        r"(?i)concertaci[óo]n",
    ],
    # DE-4: Results Orientation
    "monitoring": [
        r"(?i)seguimiento",
        r"(?i)monitoreo",
        r"(?i)evaluaci[óo]n",
        r"(?i)control",
        r"(?i)tablero(?:s)?",
    ],
}

# ------------------------------------------------------------------------------
# Immutable configuration (adapter for legacy flags)
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanSanitizerConfig:
    """Immutable configuration container for :class:`PlanSanitizer`."""
    preserve_structure: bool = True
    tag_key_elements: bool = True
    aggressive_cleaning: bool = False

    # Accepted aliases from historical integrations → canonical field
    LEGACY_ALIASES: ClassVar[Dict[str, str]] = {
        "keep_structure": "preserve_structure",
        "structure_preservation": "preserve_structure",
        "tag_elements": "tag_key_elements",
        "highlight_key_elements": "tag_key_elements",
        "aggressive_mode": "aggressive_cleaning",
    }

    @classmethod
    def from_legacy(cls, **legacy: Any) -> "PlanSanitizerConfig":
        """Create a config from strict legacy keyword inputs; raise on unknowns."""
        allowed: Set[str] = {"preserve_structure", "tag_key_elements", "aggressive_cleaning"}
        normalised: Dict[str, bool] = {}
        unknown: Set[str] = set()

        for key, value in legacy.items():
            canonical = cls.LEGACY_ALIASES.get(key, key)
            if canonical not in allowed:
                unknown.add(key)
                continue
            normalised[canonical] = bool(value)

        if unknown:
            raise ValueError(f"Unknown legacy flags: {sorted(unknown)}")

        default = cls()
        return cls(
            preserve_structure=normalised.get("preserve_structure", default.preserve_structure),
            tag_key_elements=normalised.get("tag_key_elements", default.tag_key_elements),
            aggressive_cleaning=normalised.get("aggressive_cleaning", default.aggressive_cleaning),
        )

    def as_dict(self) -> Dict[str, bool]:
        return {
            "preserve_structure": self.preserve_structure,
            "tag_key_elements": self.tag_key_elements,
            "aggressive_cleaning": self.aggressive_cleaning,
        }

# ------------------------------------------------------------------------------
# Main component (contract-pure)
# ------------------------------------------------------------------------------

class PlanSanitizer:
    """
    Sanitizes plan documents to ensure proper text quality
    while preserving critical elements for DECALOGO evaluation.

    ZERO-ARG CONSTRUCTOR to satisfy public API contract.
    """

    # --- constructor (zero-arg, no kwargs backdoors) ---
    def __init__(self) -> None:
        self._apply_config(PlanSanitizerConfig())

    # --- configuration plumbing (internal) ---
    def _apply_config(self, config: PlanSanitizerConfig) -> None:
        self.preserve_structure: bool = config.preserve_structure
        self.tag_key_elements: bool = config.tag_key_elements
        self.aggressive_cleaning: bool = config.aggressive_cleaning
        self._resolved_config: PlanSanitizerConfig = config

        # Compile patterns for key elements
        self.key_element_patterns: Dict[str, List[re.Pattern[str]]] = {
            kind: [re.compile(pat) for pat in pats] for kind, pats in KEY_ELEMENTS.items()
        }

        # Operational stats
        self.stats: Dict[str, Any] = {
            "total_chars_before": 0,
            "total_chars_after": 0,
            "key_elements_preserved": {},
            "structure_elements_preserved": 0,
        }

    @property
    def resolved_config(self) -> PlanSanitizerConfig:
        """Expose effective configuration (for audits/telemetry)."""
        return self._resolved_config

    @classmethod
    def from_config(cls, config: PlanSanitizerConfig) -> "PlanSanitizer":
        """Instantiate using an explicit config without widening __init__."""
        instance = cls()
        instance._apply_config(config)
        return instance

    @classmethod
    def legacy(cls, **legacy: Any) -> "PlanSanitizer":
        """Instantiate honoring strict legacy keyword arguments; raise on unknowns."""
        cfg = PlanSanitizerConfig.from_legacy(**legacy)
        return cls.from_config(cfg)

    # ------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text while preserving key elements.

        Args:
            text: Raw text to sanitize
        Returns:
            Sanitized text with key elements preserved
        """
        if not text:
            # Clear stats in case caller reuses instance
            self.stats["total_chars_before"] = 0
            self.stats["total_chars_after"] = 0
            self.stats["key_elements_preserved"] = {k: 0 for k in KEY_ELEMENTS}
            self.stats["structure_elements_preserved"] = 0
            return ""

        # Track statistics
        self.stats["total_chars_before"] = len(text)
        self.stats["key_elements_preserved"] = {k: 0 for k in KEY_ELEMENTS}

        # 1) Normalize base text (unicode, whitespace, accents, etc.)
        normalized_text = normalize_text(text)

        # 2) Tag key elements before aggressive cleaning so they survive transforms
        if self.tag_key_elements:
            normalized_text = self._tag_key_elements(normalized_text)

        # 3) Optionally preserve structure (headings, lists) via light-touch cleaning
        if self.preserve_structure:
            normalized_text = self._preserve_structure(normalized_text)

        # 4) Policy-aware cleaning (domain-specific) + special chars handling
        sanitized = clean_policy_text(
            normalized_text,
            aggressive=self.aggressive_cleaning,
        )
        sanitized = self._clean_special_characters(sanitized)

        # 5) Final trim
        sanitized = sanitized.strip()

        # Stats
        self.stats["total_chars_after"] = len(sanitized)
        return sanitized

    def sanitize_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        *,
        encoding: str = "utf-8",
    ) -> str:
        """
        Sanitize a file from disk and optionally write the result.

        Args:
            input_path: Path to the input text file
            output_path: Optional path to write sanitized output
            encoding: File encoding for write path (read uses safe reader)
        Returns:
            Sanitized text
        """
        src = Path(input_path)
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")

        original_text = safe_read_text_file(src)
        sanitized_text = self.sanitize_text(original_text)

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding=encoding) as f:
                f.write(sanitized_text)
            logger.info("Sanitized text saved to %s", out)

        return sanitized_text

    def get_sanitization_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the most recent sanitization operation.

        Returns:
            Dictionary with sanitization statistics
        """
        before = self.stats.get("total_chars_before", 0)
        after = self.stats.get("total_chars_after", 0)
        if before > 0:
            reduction_pct = 100 - (after / before * 100)
        else:
            reduction_pct = 0.0
        self.stats["reduction_percentage"] = round(reduction_pct, 2)
        return self.stats

    # ------------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------------

    def _tag_key_elements(self, text: str) -> str:
        """
        Surround detected key elements with stable tags to survive cleaning passes.
        Tags also allow downstream heuristics to verify preservation.
        """
        def mark(match: re.Match[str], kind: str) -> str:
            found = match.group(0)
            self.stats["key_elements_preserved"][kind] += 1
            return f"<KEY:{kind}>{found}</KEY:{kind}>"

        tagged = text
        for kind, patterns in self.key_element_patterns.items():
            for pat in patterns:
                tagged = pat.sub(lambda m, k=kind: mark(m, k), tagged)
        return tagged

    def _preserve_structure(self, text: str) -> str:
        """
        Lightweight structure preservation:
        - Normalize ordered list markers
        - Preserve headings by ensuring newline separation
        - Keep bullet markers consistent
        """
        # Normalize list bullets
        text = re.sub(r"^\s*[-•·]\s+", "- ", text, flags=re.MULTILINE)

        # Normalize numbered headings: "1) Title" or "1. Title" → ensure newline before
        text = re.sub(r"(?m)(?<!\n)(^\s*\d+[\.\)]\s+)", r"\n\1", text)

        # Collapse excessive blank lines to at most two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Count a rough number of structure elements preserved (for stats)
        headings = len(re.findall(r"(?m)^\s*\d+[\.\)]\s+\S", text))
        bullets = len(re.findall(r"(?m)^-\s+\S", text))
        self.stats["structure_elements_preserved"] = headings + bullets

        return text

    def _clean_special_characters(self, text: str) -> str:
        """
        Remove undesirable control characters, normalize whitespace,
        and fix common OCR artifacts while keeping semantic content.
        """
        # Remove non-printable controls (except newlines and tabs)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\xA0-\uFFFF]", " ", text)

        # Normalize common OCR ligatures
        ligatures = {
            "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
            """: '"', """: '"', "'": "'", "'": "'",
            "–": "-", "—": "-", "•": "-", "·": "-",
        }
        for bad, good in ligatures.items():
            text = text.replace(bad, good)

        # Normalize whitespace around tags
        text = re.sub(r"\s*(</KEY:[a-z_]+>)\s*", r"\1 ", text)
        text = re.sub(r"\s*(<KEY:[a-z_]+>)\s*", r" \1", text)

        # Collapse spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

# ------------------------------------------------------------------------------
# Factory (keeps __init__ contract pure)
# ------------------------------------------------------------------------------

def create_plan_sanitizer(
    preserve_structure: bool = True,
    tag_key_elements: bool = True,
    aggressive_cleaning: bool = False,
) -> PlanSanitizer:
    """
    Create a preconfigured plan sanitizer (legacy flags supported).

    NOTE: This does NOT widen PlanSanitizer.__init__.
    """
    return PlanSanitizer.legacy(
        preserve_structure=preserve_structure,
        tag_key_elements=tag_key_elements,
        aggressive_cleaning=aggressive_cleaning,
    )

# ------------------------------------------------------------------------------
# Simple usage example
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    sanitizer = create_plan_sanitizer()
    sample_text = """
    PLAN DE DESARROLLO MUNICIPAL 2020-2023

    1. DIAGNÓSTICO
    La situación actual del municipio presenta desafíos en educación y salud.

    2. OBJETIVOS ESTRATÉGICOS
    2.1 EDUCACIÓN
    Indicador: Tasa de escolaridad
    Línea Base: 85%
    Meta: 95%
    Responsable: Secretaría de Educación Municipal
    """

    out = sanitizer.sanitize_text(sample_text)
    print(out)
    print(sanitizer.get_sanitization_stats())
