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
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Import utility functions from text_processor
from text_processor import (
    normalize_text,
    clean_policy_text,
    remove_unwanted_characters,
    extract_paragraphs,
    standardize_accents,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Key policy elements to preserve - aligned with DECALOGO questions
KEY_ELEMENTS = {
    # DE-1: Logical Intervention Framework
    "indicators": [
        r"(?i)indicador(?:es)?",
        r"(?i)meta(?:s)?",
        r"(?i)l[ií]nea(?:s)?\s+base",
        r"(?i)resultado(?:s)?\s+esperado(?:s)?",
        r"(?i)producto(?:s)?\s+esperado(?:s)?",
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


class PlanSanitizer:
    """
    Sanitizes plan documents to ensure proper text quality
    while preserving critical elements for DECALOGO evaluation.

    Attributes:
        preserve_structure: Whether to preserve document structure
        tag_key_elements: Whether to tag key policy elements for easier detection
        aggressive_cleaning: Level of aggressiveness in text cleaning
    """

    def __init__(
        self,
        preserve_structure: bool = True,
        tag_key_elements: bool = True,
        aggressive_cleaning: bool = False,
    ):
        """
        Initialize the plan sanitizer with configuration options.

        Args:
            preserve_structure: Whether to preserve document structure
            tag_key_elements: Whether to tag key policy elements
            aggressive_cleaning: Use more aggressive cleaning methods
        """
        self.preserve_structure = preserve_structure
        self.tag_key_elements = tag_key_elements
        self.aggressive_cleaning = aggressive_cleaning
        
        # Compile patterns for key elements
        self.key_element_patterns = {}
        for element_type, patterns in KEY_ELEMENTS.items():
            self.key_element_patterns[element_type] = [
                re.compile(pattern) for pattern in patterns
            ]
            
        # Initialize counters for reporting
        self.stats = {
            "total_chars_before": 0,
            "total_chars_after": 0,
            "key_elements_preserved": {},
            "structure_elements_preserved": 0,
        }

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
        cleaned_text = clean_policy_text(
            normalized_text,
            remove_accents=False,
            keep_line_breaks=self.preserve_structure,
        )
        
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
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try alternate encodings if utf-8 fails
            try:
                with open(input_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"Failed to read file with latin-1 encoding: {e}")
                raise
        
        sanitized_text = self.sanitize_text(text)
        
        if output_path:
            output_path = Path(output_path)
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(sanitized_text)
                
            logger.info(f"Sanitized text saved to {output_path}")
        
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
def create_plan_sanitizer(preserve_structure: bool = True, tag_key_elements: bool = True) -> PlanSanitizer:
    """
    Create a preconfigured plan sanitizer.
    
    Args:
        preserve_structure: Whether to preserve document structure
        tag_key_elements: Whether to tag key elements
        
    Returns:
        Configured PlanSanitizer instance
    """
    return PlanSanitizer(
        preserve_structure=preserve_structure,
        tag_key_elements=tag_key_elements,
    )


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
