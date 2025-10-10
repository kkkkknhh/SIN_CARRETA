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

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Set, Tuple, Union

# Import file reading utility
from json_utils import safe_read_text_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Protocol
import math
import re
from collections import Counter


# ---------------------------
# Constants and regex patterns
# ---------------------------

# Sentence splitting regex - matches sentence-ending punctuation followed by whitespace
_SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])')

# Word extraction regex - matches word characters including accented characters
_WORD_REGEX = re.compile(r'\b[\w\u00C0-\u017F]+\b')


# ---------------------------
# Enums
# ---------------------------

class SegmentationType(Enum):
    """Segmentation strategy types."""
    SECTION = "section"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SectionType(Enum):
    """Section types aligned with DECALOGO dimensions (D1-D6)."""
    # D1: INSUMOS - Diagnóstico, líneas base, recursos, capacidades
    DIAGNOSTIC = "diagnostic"
    BASELINE = "baseline"
    RESOURCES = "resources"
    CAPACITY = "capacity"
    BUDGET = "budget"
    PARTICIPATION = "participation"
    
    # D2: ACTIVIDADES
    ACTIVITY = "activity"
    MECHANISM = "mechanism"
    INTERVENTION = "intervention"
    STRATEGY = "strategy"
    TIMELINE = "timeline"
    
    # D3: PRODUCTOS
    PRODUCT = "product"
    OUTPUT = "output"
    
    # D4: RESULTADOS
    RESULT = "result"
    OUTCOME = "outcome"
    INDICATOR = "indicator"
    MONITORING = "monitoring"
    
    # D5: IMPACTOS
    IMPACT = "impact"
    LONG_TERM_EFFECT = "long_term_effect"
    
    # D6: CAUSALIDAD
    CAUSAL_THEORY = "causal_theory"
    CAUSAL_LINK = "causal_link"
    
    # Legacy/Multi-dimensional
    VISION = "vision"
    OBJECTIVE = "objective"
    RESPONSIBILITY = "responsibility"


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


@dataclass(frozen=True)
class DocumentSegmenterConfig:
    """Immutable configuration container for :class:`DocumentSegmenter`."""

    target_char_min: int = 700
    target_char_max: int = 900
    target_sentences: int = 3
    segmentation_type: SegmentationType = SegmentationType.SECTION
    min_segment_length: int = 50
    max_segment_length: int = 1000
    preserve_context: bool = True
    min_segment_chars: int = 350
    max_segment_chars: int = 900
    context_window: int = 1

    LEGACY_ALIASES: ClassVar[Dict[str, str]] = {
        "min_chunk_size": "target_char_min",
        "max_chunk_size": "target_char_max",
        "target_min_chars": "target_char_min",
        "target_max_chars": "target_char_max",
        "sentence_target": "target_sentences",
        "segment_type": "segmentation_type",
        "keep_context": "preserve_context",
    }

    ALLOWED_KEYS: ClassVar[Set[str]] = {
        "target_char_min",
        "target_char_max",
        "target_sentences",
        "segmentation_type",
        "min_segment_length",
        "max_segment_length",
        "preserve_context",
        "min_segment_chars",
        "max_segment_chars",
        "context_window",
    }

    @staticmethod
    def _coerce_int(
        value: Any,
        default: int,
        *,
        minimum: Optional[int] = None,
    ) -> int:
        """Coerce a value to ``int`` while applying lower bounds."""

        if value is None:
            result = default
        else:
            try:
                result = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Expected integer-compatible value, received {value!r}") from exc

        if minimum is not None:
            result = max(minimum, result)

        return result

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        """Coerce a value to ``bool`` while tolerating ``None`` inputs."""

        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _coerce_segmentation_type(
        value: Any,
        default: SegmentationType,
    ) -> SegmentationType:
        """Convert legacy values into :class:`SegmentationType`."""

        if value is None:
            return default

        if isinstance(value, SegmentationType):
            return value

        if isinstance(value, str):
            try:
                return SegmentationType(value.lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported segmentation_type: {value!r}") from exc

        raise ValueError(f"Unsupported segmentation_type: {value!r}")

    @classmethod
    def from_legacy(cls, **legacy: Any) -> "DocumentSegmenterConfig":
        """Create a configuration object from strict legacy keyword inputs."""

        unknown: Set[str] = set()
        normalised: Dict[str, Any] = {}

        for key, value in legacy.items():
            canonical = cls.LEGACY_ALIASES.get(key, key)
            if canonical not in cls.ALLOWED_KEYS:
                unknown.add(key)
                continue
            normalised[canonical] = value

        if unknown:
            raise ValueError(f"Unknown legacy flags: {sorted(unknown)}")

        defaults = cls()
        target_char_min = cls._coerce_int(
            normalised.get("target_char_min"),
            defaults.target_char_min,
            minimum=10,
        )
        target_char_max = cls._coerce_int(
            normalised.get("target_char_max"),
            defaults.target_char_max,
            minimum=target_char_min,
        )

        min_segment_length = cls._coerce_int(
            normalised.get("min_segment_length"),
            defaults.min_segment_length,
            minimum=1,
        )
        max_segment_length = cls._coerce_int(
            normalised.get("max_segment_length"),
            defaults.max_segment_length,
            minimum=min_segment_length,
        )

        default_min_segment_chars = max(10, target_char_min // 2)
        min_segment_chars = cls._coerce_int(
            normalised.get("min_segment_chars"),
            default_min_segment_chars,
            minimum=10,
        )

        max_segment_chars = cls._coerce_int(
            normalised.get("max_segment_chars"),
            target_char_max,
            minimum=min_segment_chars,
        )

        context_window = cls._coerce_int(
            normalised.get("context_window"),
            defaults.context_window,
            minimum=0,
        )

        target_sentences = cls._coerce_int(
            normalised.get("target_sentences"),
            defaults.target_sentences,
            minimum=1,
        )

        segmentation_type = cls._coerce_segmentation_type(
            normalised.get("segmentation_type"),
            defaults.segmentation_type,
        )

        preserve_context = cls._coerce_bool(
            normalised.get("preserve_context"),
            defaults.preserve_context,
        )

        return cls(
            target_char_min=target_char_min,
            target_char_max=target_char_max,
            target_sentences=target_sentences,
            segmentation_type=segmentation_type,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            preserve_context=preserve_context,
            min_segment_chars=min_segment_chars,
            max_segment_chars=max_segment_chars,
            context_window=context_window,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Expose the configuration as a plain dictionary for diagnostics."""

        return {
            "target_char_min": self.target_char_min,
            "target_char_max": self.target_char_max,
            "target_sentences": self.target_sentences,
            "segmentation_type": self.segmentation_type,
            "min_segment_length": self.min_segment_length,
            "max_segment_length": self.max_segment_length,
            "preserve_context": self.preserve_context,
            "min_segment_chars": self.min_segment_chars,
            "max_segment_chars": self.max_segment_chars,
            "context_window": self.context_window,
        }


class DocumentSegmenter:
    """Segments a document into logical units for DECALOGO analysis."""

    def __init__(self) -> None:
        """Initialise the document segmenter with immutable defaults."""

        self._apply_config(DocumentSegmenterConfig())
        self._initialise_patterns()

    def _apply_config(self, config: DocumentSegmenterConfig) -> None:
        """Apply the supplied configuration, refreshing derived state."""

        self.target_char_min = config.target_char_min
        self.target_char_max = config.target_char_max
        self.target_sentences = config.target_sentences

        self.segmentation_type = config.segmentation_type
        self.min_segment_length = config.min_segment_length
        self.max_segment_length = config.max_segment_length
        self.preserve_context = config.preserve_context

        self.min_segment_chars = config.min_segment_chars
        self.max_segment_chars = config.max_segment_chars
        self.context_window = config.context_window

        self._resolved_config = config

    @property
    def resolved_config(self) -> DocumentSegmenterConfig:
        """Return the resolved configuration for telemetry and audits."""

        return self._resolved_config

    @classmethod
    def from_config(cls, config: DocumentSegmenterConfig) -> "DocumentSegmenter":
        """Instantiate a :class:`DocumentSegmenter` using an explicit config."""

        instance = cls()
        instance._apply_config(config)
        return instance

    @classmethod
    def legacy(cls, **legacy: Any) -> "DocumentSegmenter":
        """Create an instance honouring strict legacy keyword arguments."""

        config = DocumentSegmenterConfig.from_legacy(**legacy)
        return cls.from_config(config)

    def _initialise_patterns(self) -> None:
        """Prepare regular expression patterns for section identification."""

        # Section identification patterns - aligned with DECALOGO dimensions (D1-D6)
        # Based on decalogo-industrial.latest.clean.json structure
        self.section_patterns = {
            # D1: INSUMOS - Diagnóstico, líneas base, recursos, capacidades
            SectionType.DIAGNOSTIC: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:diagn[óo]stico|antecedentes|contexto|situaci[óo]n actual)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:problem[áa]tica|necesidades|demandas)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:caracterizaci[óo]n|perfil)"
            ],
            SectionType.BASELINE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:l[íi]nea(?:s)? base|datos base|baseline)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:series temporales|medici[óo]n inicial)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:estado actual medido|indicadores iniciales)"
            ],
            SectionType.RESOURCES: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:recursos asignados|asignaci[óo]n de recursos)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:plan plurianual|PPI|plan indicativo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:trazabilidad program[áa]tica)"
            ],
            SectionType.CAPACITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:capacidades institucionales|capacidad institucional)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:talento humano|recurso humano|personal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:procesos institucionales|sistemas de informaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cuellos de botella|restricciones institucionales)"
            ],
            SectionType.BUDGET: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:presupuesto|recursos (?:financieros|econ[óo]micos))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:financiaci[óo]n|inversi[óo]n|gasto(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:costeo|asignaci[óo]n (?:presupuestal|de recursos))"
            ],
            SectionType.PARTICIPATION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:participaci[óo]n|gobernanza|concertaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:mesa(?:s)? (?:t[ée]cnica(?:s)?|participativa(?:s)?))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:di[áa]logo(?:s)?|consulta(?:s)?)"
            ],

            # D2: ACTIVIDADES - Formalización, mecanismos causales
            SectionType.ACTIVITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:actividad(?:es)?|acciones?|intervenciones?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:formalizaci[óo]n de actividades|tabla de actividades)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:responsable.*insumo.*output|cronograma.*costo)"
            ],
            SectionType.MECHANISM: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:mecanismo(?:s)? causal(?:es)?|v[íi]a causal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:poblaci[óo]n diana|grupo objetivo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:causa ra[íi]z|mediador(?:es)?)"
            ],
            SectionType.INTERVENTION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:teor[íi]a de intervenci[óo]n|l[óo]gica de intervenci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:complementariedades|secuenciaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:riesgos de implementaci[óo]n|cu[ñn]as de implementaci[óo]n)"
            ],
            SectionType.STRATEGY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:estrategia(?:s)?|l[íi]nea(?:s)? (?:de)? acci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:programa(?:s)?|proyecto(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:iniciativa(?:s)?)"
            ],
            SectionType.TIMELINE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cronograma|calendario|plazos)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:tiempos|periodicidad|fechas)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:hitos|milestones|fases)"
            ],

            # D3: PRODUCTOS - Outputs con indicadores verificables
            SectionType.PRODUCT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:producto(?:s)?|output(?:s)?|entregable(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:bien(?:es)? y servicio(?:s)?|prestaci[óo]n de servicios)"
            ],
            SectionType.OUTPUT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:output(?:s)? verificable(?:s)?|producto verificable)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cobertura proporcional|suficiencia relativa)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:trazabilidad presupuestal del producto)"
            ],

            # D4: RESULTADOS - Outcomes con métricas
            SectionType.RESULT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:resultado(?:s)?|outcome(?:s)?|logro(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:encadenamiento causal|v[íi]nculo productos.*resultados)"
            ],
            SectionType.OUTCOME: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:outcome(?:s)? con m[ée]trica(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:ventana de maduraci[óo]n|tiempo de efecto)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:nivel de ambici[óo]n|magnitud del cambio)"
            ],
            SectionType.INDICATOR: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:indicador(?:es)? de resultado|medici[óo]n de outcome)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:m[ée]trica(?:s)?|f[óo]rmula de c[áa]lculo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:meta(?:s)? cuantificada(?:s)?)"
            ],
            SectionType.MONITORING: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:seguimiento|monitoreo|evaluaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:control|supervisi[óo]n|vigilancia)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:tablero(?:s)? de (?:control|mando))"
            ],

            # D5: IMPACTOS - Efectos de largo plazo
            SectionType.IMPACT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:impacto(?:s)?|efecto(?:s)? (?:de )?largo plazo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cambio(?:s)? estructural(?:es)?|transformaci[óo]n sostenible)"
            ],
            SectionType.LONG_TERM_EFFECT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:efecto(?:s)? duradero(?:s)?|sostenibilidad)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:alineaci[óo]n (?:con )?(?:PND|ODS|marco(?:s)? internacionales?))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:proxy de impacto|indicador(?:es)? proxy)"
            ],

            # D6: CAUSALIDAD - Teoría de cambio explícita
            SectionType.CAUSAL_THEORY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:teor[íi]a de cambio|marco l[óo]gico causal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:DAG|grafo (?:causal|ac[íi]clico dirigido))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:diagrama causal|modelo causal)"
            ],
            SectionType.CAUSAL_LINK: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:encadenamiento (?:causal|l[óo]gico))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:validaci[óo]n l[óo]gica|consistencia causal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:hip[óo]tesis causales|supuestos cr[íi]ticos)"
            ],

            # Multi-dimensional legacy sections
            SectionType.VISION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:visi[óo]n|misi[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:escenario(?:s)? deseado(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:futuro(?:s)? deseado(?:s)?)"
            ],
            SectionType.OBJECTIVE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:objetivo(?:s)?|prop[óo]sito(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:finalidad(?:es)?|meta(?:s)? general(?:es)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:logro(?:s)? esperado(?:s)?)"
            ],
            SectionType.RESPONSIBILITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:responsable(?:s)?|encargado(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:entidad(?:es)? (?:responsable|ejecutora)(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:actor(?:es)? (?:responsable|institucional)(?:s)?)"
            ],
        }

        # Compile all section patterns for efficiency
        self.compiled_patterns = {
            section_type: [re.compile(pattern) for pattern in patterns]
            for section_type, patterns in self.section_patterns.items()
        }
        
        self.section_patterns = {
            # D1: INSUMOS - Diagnóstico, líneas base, recursos, capacidades
            SectionType.DIAGNOSTIC: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:diagn[óo]stico|antecedentes|contexto|situaci[óo]n actual)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:problem[áa]tica|necesidades|demandas)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:caracterizaci[óo]n|perfil)"
            ],
            SectionType.BASELINE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:l[íi]nea(?:s)? base|datos base|baseline)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:series temporales|medici[óo]n inicial)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:estado actual medido|indicadores iniciales)"
            ],
            SectionType.RESOURCES: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:recursos asignados|asignaci[óo]n de recursos)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:plan plurianual|PPI|plan indicativo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:trazabilidad program[áa]tica)"
            ],
            SectionType.CAPACITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:capacidades institucionales|capacidad institucional)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:talento humano|recurso humano|personal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:procesos institucionales|sistemas de informaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cuellos de botella|restricciones institucionales)"
            ],
            SectionType.BUDGET: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:presupuesto|recursos (?:financieros|econ[óo]micos))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:financiaci[óo]n|inversi[óo]n|gasto(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:costeo|asignaci[óo]n (?:presupuestal|de recursos))"
            ],
            SectionType.PARTICIPATION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:participaci[óo]n|gobernanza|concertaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:mesa(?:s)? (?:t[ée]cnica(?:s)?|participativa(?:s)?))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:di[áa]logo(?:s)?|consulta(?:s)?)"
            ],
            
            # D2: ACTIVIDADES - Formalización, mecanismos causales
            SectionType.ACTIVITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:actividad(?:es)?|acciones?|intervenciones?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:formalizaci[óo]n de actividades|tabla de actividades)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:responsable.*insumo.*output|cronograma.*costo)"
            ],
            SectionType.MECHANISM: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:mecanismo(?:s)? causal(?:es)?|v[íi]a causal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:poblaci[óo]n diana|grupo objetivo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:causa ra[íi]z|mediador(?:es)?)"
            ],
            SectionType.INTERVENTION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:teor[íi]a de intervenci[óo]n|l[óo]gica de intervenci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:complementariedades|secuenciaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:riesgos de implementaci[óo]n|cu[ñn]as de implementaci[óo]n)"
            ],
            SectionType.STRATEGY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:estrategia(?:s)?|l[íi]nea(?:s)? (?:de)? acci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:programa(?:s)?|proyecto(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:iniciativa(?:s)?)"
            ],
            SectionType.TIMELINE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cronograma|calendario|plazos)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:tiempos|periodicidad|fechas)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:hitos|milestones|fases)"
            ],
            
            # D3: PRODUCTOS - Outputs con indicadores verificables
            SectionType.PRODUCT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:producto(?:s)?|output(?:s)?|entregable(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:bien(?:es)? y servicio(?:s)?|prestaci[óo]n de servicios)"
            ],
            SectionType.OUTPUT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:output(?:s)? verificable(?:s)?|producto verificable)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cobertura proporcional|suficiencia relativa)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:trazabilidad presupuestal del producto)"
            ],
            
            # D4: RESULTADOS - Outcomes con métricas
            SectionType.RESULT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:resultado(?:s)?|outcome(?:s)?|logro(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:encadenamiento causal|v[íi]nculo productos.*resultados)"
            ],
            SectionType.OUTCOME: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:outcome(?:s)? con m[ée]trica(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:ventana de maduraci[óo]n|tiempo de efecto)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:nivel de ambici[óo]n|magnitud del cambio)"
            ],
            SectionType.INDICATOR: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:indicador(?:es)? de resultado|medici[óo]n de outcome)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:m[ée]trica(?:s)?|f[óo]rmula de c[áa]lculo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:meta(?:s)? cuantificada(?:s)?)"
            ],
            SectionType.MONITORING: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:seguimiento|monitoreo|evaluaci[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:control|supervisi[óo]n|vigilancia)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:tablero(?:s)? de (?:control|mando))"
            ],
            
            # D5: IMPACTOS - Efectos de largo plazo
            SectionType.IMPACT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:impacto(?:s)?|efecto(?:s)? (?:de )?largo plazo)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:cambio(?:s)? estructural(?:es)?|transformaci[óo]n sostenible)"
            ],
            SectionType.LONG_TERM_EFFECT: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:efecto(?:s)? duradero(?:s)?|sostenibilidad)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:alineaci[óo]n (?:con )?(?:PND|ODS|marco(?:s)? internacionales?))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:proxy de impacto|indicador(?:es)? proxy)"
            ],
            
            # D6: CAUSALIDAD - Teoría de cambio explícita
            SectionType.CAUSAL_THEORY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:teor[íi]a de cambio|marco l[óo]gico causal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:DAG|grafo (?:causal|ac[íi]clico dirigido))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:diagrama causal|modelo causal)"
            ],
            SectionType.CAUSAL_LINK: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:encadenamiento (?:causal|l[óo]gico))",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:validaci[óo]n l[óo]gica|consistencia causal)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:hip[óo]tesis causales|supuestos cr[íi]ticos)"
            ],
            
            # Multi-dimensional legacy sections
            SectionType.VISION: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:visi[óo]n|misi[óo]n)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:escenario(?:s)? deseado(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:futuro(?:s)? deseado(?:s)?)"
            ],
            SectionType.OBJECTIVE: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:objetivo(?:s)?|prop[óo]sito(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:finalidad(?:es)?|meta(?:s)? general(?:es)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:logro(?:s)? esperado(?:s)?)"
            ],
            SectionType.RESPONSIBILITY: [
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:responsable(?:s)?|encargado(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:entidad(?:es)? (?:responsable|ejecutora)(?:s)?)",
                r"(?i)(?:^|\n)(?:\d+[\.\)]\s*)?(?:actor(?:es)? (?:responsable|institucional)(?:s)?)"
            ],
        }
        
        # Compile all section patterns for efficiency
        self.compiled_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_patterns[section_type] = [re.compile(pattern) for pattern in patterns]


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
