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
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Set, Union

# Import utility functions from text_processor
from text_processor import (
    normalize_text,
    clean_policy_text,
    )

# Import file reading utility
from json_utils import safe_read_text_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Key policy elements to preserve - aligned with DECALOGO 6 dimensions
# Based on decalogo-industrial.latest.clean.json canonical structure
KEY_ELEMENTS = {
    # D1: INSUMOS (diagnóstico, líneas base, recursos, capacidades institucionales)
    "insumos": [
        r"(?i)diagn[óo]stico",
        r"(?i)l[ií]nea(?:s)?\s+base",
        r"(?i)recurso(?:s)?",
        r"(?i)capacidad(?:es)?\s+institucional(?:es)?",
        r"(?i)situaci[óo]n\s+actual",
        r"(?i)problem[áa]tica",
        r"(?i)brecha(?:s)?",
        r"(?i)coherencia",
    ],
    # D2: ACTIVIDADES (formalización, mecanismos causales, teoría de intervención)
    "actividades": [
        r"(?i)actividad(?:es)?",
        r"(?i)mecanismo(?:s)?\s+causal(?:es)?",
        r"(?i)intervenci[óo]n",
        r"(?i)responsable(?:s)?",
        r"(?i)instrumento(?:s)?",
        r"(?i)poblaci[óo]n\s+diana",
        r"(?i)riesgo(?:s)?\s+de\s+implementaci[óo]n",
    ],
    # D3: PRODUCTOS (outputs con indicadores verificables, trazabilidad)
    "productos": [
        r"(?i)producto(?:s)?",
        r"(?i)output(?:s)?",
        r"(?i)indicador(?:es)?\s+verificable(?:s)?",
        r"(?i)trazabilidad",
        r"(?i)cobertura",
        r"(?i)dosificaci[óo]n",
        r"(?i)entregable(?:s)?",
    ],
    # D4: RESULTADOS (outcomes con métricas, encadenamiento causal)
    "resultados": [
        r"(?i)resultado(?:s)?",
        r"(?i)outcome(?:s)?",
        r"(?i)m[ée]trica(?:s)?",
        r"(?i)meta(?:s)?",
        r"(?i)encadenamiento\s+causal",
        r"(?i)ventana\s+de\s+maduraci[óo]n",
        r"(?i)nivel\s+de\s+ambici[óo]n",
    ],
    # D5: IMPACTOS (efectos largo plazo, proxies, alineación marcos)
    "impactos": [
        r"(?i)impacto(?:s)?",
        r"(?i)efecto(?:s)?\s+(?:de\s+)?largo\s+plazo",
        r"(?i)prox(?:y|ies)",
        r"(?i)transmisi[óo]n",
        r"(?i)rezago(?:s)?",
        r"(?i)PND",
        r"(?i)ODS",
        r"(?i)marco(?:s)?\s+(?:nacional|global)(?:es)?",
    ],
    # D6: CAUSALIDAD (teoría de cambio explícita, DAG, validación lógica)
    "causalidad": [
        r"(?i)teor[ií]a\s+de\s+cambio",
        r"(?i)diagrama\s+causal",
        r"(?i)DAG",
        r"(?i)cadena\s+causal",
        r"(?i)l[óo]gica\s+causal",
        r"(?i)supuesto(?:s)?\s+verificable(?:s)?",
        r"(?i)mediador(?:es)?",
        r"(?i)moderador(?:es)?",
        r"(?i)validaci[óo]n\s+l[óo]gica",
        r"(?i)seguimiento",
        r"(?i)monitoreo",
        r"(?i)evaluaci[óo]n",
    ],
}


@dataclass(frozen=True)
class PlanSanitizerConfig:
    """Immutable configuration container for :class:`PlanSanitizer`."""

    preserve_structure: bool = True
    tag_key_elements: bool = True
    aggressive_cleaning: bool = False

    #: Accepted aliases originating from historical integrations.  Each alias
    #: is mapped to the canonical attribute name so that ``create_plan_sanitizer``
    #: and older orchestrators keep working without code changes.
    LEGACY_ALIASES: ClassVar[Dict[str, str]] = {
        "keep_structure": "preserve_structure",
        "structure_preservation": "preserve_structure",
        "tag_elements": "tag_key_elements",
        "highlight_key_elements": "tag_key_elements",
        "aggressive_mode": "aggressive_cleaning",
    }

    @classmethod
    def from_legacy(cls, **legacy: Any) -> "PlanSanitizerConfig":
        """Create a configuration object from strict legacy keyword inputs."""

        allowed: Set[str] = {"preserve_structure", "tag_key_elements", "aggressive_cleaning"}
        normalised: Dict[str, bool] = {}
        unknown: Set[str] = set()

        for key, value in legacy.items():
            canonical_key = cls.LEGACY_ALIASES.get(key, key)
            if canonical_key not in allowed:
                unknown.add(key)
                continue
            normalised[canonical_key] = bool(value)

        if unknown:
            raise ValueError(f"Unknown legacy flags: {sorted(unknown)}")

        default = cls()
        return cls(
            preserve_structure=normalised.get("preserve_structure", default.preserve_structure),
            tag_key_elements=normalised.get("tag_key_elements", default.tag_key_elements),
            aggressive_cleaning=normalised.get("aggressive_cleaning", default.aggressive_cleaning),
        )

    def as_dict(self) -> Dict[str, bool]:
        """Expose the configuration as a plain dictionary for convenience."""

        return {
            "preserve_structure": self.preserve_structure,
            "tag_key_elements": self.tag_key_elements,
            "aggressive_cleaning": self.aggressive_cleaning,
        }


class PlanSanitizer:
    """
    Sanitizes plan documents to ensure proper text quality
    while preserving critical elements for DECALOGO evaluation.

    Attributes:
        preserve_structure: Whether to preserve document structure
        tag_key_elements: Whether to tag key policy elements for easier detection
        aggressive_cleaning: Level of aggressiveness in text cleaning
    """

    def __init__(self) -> None:
        """Initialise the plan sanitizer with the immutable default config."""

        self._apply_config(PlanSanitizerConfig())

    def _apply_config(self, config: PlanSanitizerConfig) -> None:
        """Apply the supplied configuration, refreshing derived state."""

        self.preserve_structure = config.preserve_structure
        self.tag_key_elements = config.tag_key_elements
        self.aggressive_cleaning = config.aggressive_cleaning

        # Persist the resolved configuration to support advanced debugging and
        # reproducibility tooling that inspects runtime state.
        self._resolved_config = config

        # Compile patterns for key elements
        self.key_element_patterns = {
            element_type: [re.compile(pattern) for pattern in patterns]
            for element_type, patterns in KEY_ELEMENTS.items()
        }

        # Initialize counters for reporting
        self.stats = {
            "total_chars_before": 0,
            "total_chars_after": 0,
            "key_elements_preserved": {},
            "structure_elements_preserved": 0,
        }

    @property
    def resolved_config(self) -> PlanSanitizerConfig:
        """Expose the effective configuration for observability tooling."""

        return self._resolved_config

    @classmethod
    def from_config(cls, config: PlanSanitizerConfig) -> "PlanSanitizer":
        """Instantiate a :class:`PlanSanitizer` using an explicit configuration."""

        instance = cls()
        instance._apply_config(config)
        return instance

    @classmethod
    def legacy(cls, **legacy: Any) -> "PlanSanitizer":
        """Create an instance honouring legacy keyword arguments strictly."""

        config = PlanSanitizerConfig.from_legacy(**legacy)
        return cls.from_config(config)

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text while preserving key elements.
        
        Args:
            text: Raw text to sanitize
            
        Returns:
            Sanitized text with key elements preserved
        """
        if not text:
            return ""
        
        # Track statistics
        self.stats["total_chars_before"] = len(text)
        self.stats["key_elements_preserved"] = {k: 0 for k in KEY_ELEMENTS.keys()}
        
        # First, normalize text
        normalized_text = normalize_text(text)
        
        # Detect and mark key elements before cleaning
        if self.tag_key_elements:
            normalized_text = self._tag_key_elements(normalized_text)
            
        # Preserve structure if needed
        if self.preserve_structure:
            normalized_text = self._preserve_structure(normalized_text)
        
        # Clean the text while preserving marked elements
        cleaned_text = clean_policy_text(normalized_text)

        # Update statistics
        self.stats["total_chars_after"] = len(cleaned_text)
        
        return cleaned_text

    def _tag_key_elements(self, text: str) -> str:
        """
        Tag key policy elements to ensure they're preserved during cleaning.
        
        Args:
            text: Text to process
            
        Returns:
            Text with key elements tagged
        """
        processed_text = text
        
        for element_type, patterns in self.key_element_patterns.items():
            for pattern in patterns:
                matches = list(pattern.finditer(processed_text))
                
                # Track matches for statistics
                self.stats["key_elements_preserved"][element_type] = len(matches)
                
                # Process matches from end to start to avoid position shifts
                for match in reversed(matches):
                    start, end = match.span()
                    # Extract the context around the match
                    context_start = max(0, start - 100)
                    context_end = min(len(processed_text), end + 300)
                    context = processed_text[context_start:context_end]
                    
                    # Keep this context block protected from aggressive cleaning
                    if self.aggressive_cleaning:
                        # Mark the context with special tags
                        processed_text = (
                            processed_text[:context_start] +
                            f"__PRESERVE_START_{element_type}__" +
                            context +
                            f"__PRESERVE_END_{element_type}__" +
                            processed_text[context_end:]
                        )
        
        return processed_text

    def _preserve_structure(self, text: str) -> str:
        """
        Preserve document structure elements like headings, lists, and tables.
        
        Args:
            text: Text to process
            
        Returns:
            Text with structure elements preserved
        """
        # Identify and mark headings
        heading_pattern = re.compile(r"(?m)^(?:[\d.]+\s+)?([A-Z][A-Za-z\s]{3,60})$")
        matches = list(heading_pattern.finditer(text))
        self.stats["structure_elements_preserved"] = len(matches)
        
        # Mark headings from end to start to avoid position shifts
        processed_text = text
        for match in reversed(matches):
            start, end = match.span()
            heading_text = match.group(1)
            # Mark as heading
            processed_text = (
                processed_text[:start] +
                f"\n__HEADING_START__{heading_text}__HEADING_END__\n" +
                processed_text[end:]
            )
        
        return processed_text
    
    def sanitize_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Sanitize a file containing plan text.
        
        Args:
            input_path: Path to input file
            output_path: Optional path to output file (if None, returns text without saving)
            
        Returns:
            Sanitized text
        """
        input_path = Path(input_path)
        
        # Use utility function for safe file reading with encoding fallback
        text = safe_read_text_file(input_path)
        
        sanitized_text = self.sanitize_text(text)
        
        if output_path:
            output_path = Path(output_path)
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(sanitized_text)
                
            logger.info("Sanitized text saved to %s", output_path)
        
        return sanitized_text
    
    def get_sanitization_stats(self) -> Dict[str, any]:
        """
        Get statistics from the most recent sanitization operation.
        
        Returns:
            Dictionary with sanitization statistics
        """
        if self.stats["total_chars_before"] > 0:
            reduction_pct = 100 - (self.stats["total_chars_after"] / self.stats["total_chars_before"] * 100)
        else:
            reduction_pct = 0
        
        self.stats["reduction_percentage"] = round(reduction_pct, 2)
        return self.stats


# Factory function to create a preconfigured sanitizer
def create_plan_sanitizer(
    preserve_structure: bool = True,
    tag_key_elements: bool = True,
    aggressive_cleaning: bool = False,
) -> PlanSanitizer:
    """
    Create a preconfigured plan sanitizer.
    
    Args:
        preserve_structure: Whether to preserve document structure
        tag_key_elements: Whether to tag key elements
        
    Returns:
        Configured PlanSanitizer instance
    """
    return PlanSanitizer.legacy()


# Simple usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sanitizer = create_plan_sanitizer()
    
    # Example text with key elements that should be preserved
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
    
    3. MECANISMOS DE PARTICIPACIÓN
    
    Se realizaron 5 mesas técnicas con la comunidad.
    
    4. SEGUIMIENTO Y MONITOREO
    
    El plan contará con un tablero de control para seguimiento trimestral.
    """
    
    sanitized = sanitizer.sanitize_text(sample_text)
    print("\nSanitized text:")
    print(sanitized)
    
    stats = sanitizer.get_sanitization_stats()
    print("\nSanitization statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


# Convenience functions for common use cases
def sanitize_plan_name(plan_name: str, max_length: int = 255) -> str:
    """Convenience function for plan name sanitization."""
    return PlanSanitizer.sanitize_plan_name(plan_name, max_length)


def standardize_json_keys(
    json_obj: Dict[str, Any], preserve_display: bool = True
) -> Dict[str, Any]:
    """Convenience function for JSON key standardization."""
    return PlanSanitizer.standardize_json_object(json_obj, preserve_display)


def create_plan_directory(plan_name: str, base_path: str = ".") -> str:
    """Convenience function to create a safe directory for a plan."""
    return PlanSanitizer.create_safe_directory(plan_name, base_path, create=True)
