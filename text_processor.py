# coding=utf-8
"""
Text Processing Module

Provides robust Unicode normalization and text processing utilities
for consistent handling of Spanish text in development plans.

Features:
- Unicode normalization (NFC, NFKC)
- Special character handling
- Whitespace normalization
- Accent and diacritic standardization
- Case normalization options
- Text segmentation utilities
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text processor with configurable normalization options.
    
    Provides methods for consistent text normalization, especially for
    Spanish language text with varying Unicode representations.
    """
    
    def __init__(
        self, 
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        normalize_case: bool = False,
        unicode_form: str = 'NFKC'
    ):
        """
        Initialize text processor with configuration options.
        
        Args:
            normalize_unicode: Whether to normalize Unicode characters
            normalize_whitespace: Whether to normalize whitespace
            normalize_case: Whether to normalize case (to lowercase)
            unicode_form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
        """
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.normalize_case = normalize_case
        
        if unicode_form not in ('NFC', 'NFKC', 'NFD', 'NFKD'):
            logger.warning("Invalid Unicode form '%s', using 'NFKC'", unicode_form)
            self.unicode_form = 'NFKC'
        else:
            self.unicode_form = unicode_form
        
        # Common patterns for normalization
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    def normalize_text(self, text: str) -> str:
        """
        Apply configured normalizations to text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove null bytes and control characters
        text = self.special_chars_pattern.sub(' ', text)
        
        # Apply Unicode normalization if enabled
        if self.normalize_unicode:
            text = unicodedata.normalize(self.unicode_form, text)
        
        # Normalize whitespace if enabled
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Convert to lowercase if enabled
        if self.normalize_case:
            text = text.lower()
        
        return text
    
    def segment_text(self, text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
        """
        Segment text into overlapping chunks of maximum length.
        
        Args:
            text: Input text to segment
            max_length: Maximum length of each segment
            overlap: Number of characters to overlap between segments
            
        Returns:
            List of text segments
        """
        if not text or max_length <= 0:
            return []
        
        # Normalize text first
        normalized_text = self.normalize_text(text)
        
        # If text is shorter than max_length, return as is
        if len(normalized_text) <= max_length:
            return [normalized_text]
        
        segments = []
        start = 0
        text_length = len(normalized_text)
        
        while start < text_length:
            # Calculate end position
            end = start + max_length
            
            # If we're at the end of the text, just add the final segment
            if end >= text_length:
                segments.append(normalized_text[start:])
                break
            
            # Find a good breaking point (period, question mark, exclamation, etc.)
            # Look for a sentence boundary within the last 20% of the segment
            search_start = end - int(max_length * 0.2)
            search_text = normalized_text[search_start:end]
            
            # Find the last sentence boundary
            boundary_match = re.search(r'[.!?]\s+', search_text)
            
            if boundary_match:
                # Break at the sentence boundary
                break_pos = search_start + boundary_match.end()
                segments.append(normalized_text[start:break_pos])
                start = break_pos - overlap
            else:
                # If no sentence boundary found, break at a space
                last_space = normalized_text.rfind(' ', search_start, end)
                
                if last_space != -1:
                    segments.append(normalized_text[start:last_space])
                    start = last_space - overlap
                else:
                    # If no space found, break at max_length
                    segments.append(normalized_text[start:end])
                    start = end - overlap
            
            # Ensure overlap doesn't go negative
            start = max(0, start)
        
        return segments
    
    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts for similarity after normalization.
        
        This is a simple character-based similarity metric.
        For more sophisticated semantic comparison, use embedding_model instead.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both texts
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Calculate similarity based on character differences
        if not norm1 and not norm2:
            return 1.0
        if not norm1 or not norm2:
            return 0.0
        
        # Use Levenshtein distance if available
        try:
            import Levenshtein
            distance = Levenshtein.distance(norm1, norm2)
            max_length = max(len(norm1), len(norm2))
            return 1.0 - (distance / max_length)
        except ImportError:
            # Fallback to simple ratio of matching characters
            s1 = set(norm1)
            s2 = set(norm2)
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            return intersection / union if union > 0 else 0.0


# ============================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility)
# ============================================================================

# Additional lightweight helpers expected by tests
# These functions intentionally keep logic minimal and Unicode-safe.

def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    """Normalize Unicode text using the specified form (default NFKC)."""
    if not text:
        return ""
    try:
        return unicodedata.normalize(form, text)
    except Exception:
        # Fallback to original text if something goes wrong
        return text


def find_quotes(text: str) -> List[str]:
    """Find quoted substrings using double quotes after normalization."""
    if not text:
        return []
    t = normalize_unicode(text)
    # Remove smart quotes by converting to straight quotes via NFKC, then match
    return re.findall(r'"(.*?)"', t)


def count_words(text: str) -> int:
    """Count words in Unicode text. Words are sequences of word chars or letters."""
    if not text:
        return 0
    t = normalize_unicode(text)
    # Split on whitespace for robustness with accents; filter empties
    return len([w for w in re.split(r'\s+', t.strip()) if w])


def extract_emails(text: str) -> List[str]:
    """Extract email addresses allowing Unicode letters in local/domain parts."""
    if not text:
        return []
    t = normalize_unicode(text)
    # Allow word chars (includes Unicode letters), dots, hyphens in local and domain
    pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}', re.UNICODE)
    return pattern.findall(t)


def replace_special_chars(text: str) -> str:
    """Replace common Unicode punctuation with ASCII equivalents or remove quotes."""
    if not text:
        return ""
    t = normalize_unicode(text)
    # Replace various dash types with ASCII hyphen
    t = re.sub(r'[\u2012\u2013\u2014\u2015\u2212]', '-', t)
    # Remove straight and smart double quotes entirely
    t = t.replace('"', '')
    t = t.replace('\u201C', '').replace('\u201D', '')
    return t


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using ., !, ? boundaries."""
    if not text:
        return []
    t = normalize_unicode(text)
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p for p in parts if p.strip()]


def search_pattern(text: str, pattern: str):
    """Search for a literal pattern in normalized text; returns a match or None."""
    if text is None or pattern is None:
        return None
    t = normalize_unicode(text)
    p = normalize_unicode(pattern)
    return re.search(re.escape(p), t)


def match_phone_numbers(text: str) -> List[str]:
    """Match common US phone number formats like 123-456-7890 or (123) 456-7890."""
    if not text:
        return []
    t = normalize_unicode(text)
    # Use finditer to capture entire matches
    regex = re.compile(r'(?:\(\d{3}\)\s*|\d{3}-)\d{3}-\d{4}')
    return [m.group(0) for m in regex.finditer(t)]


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords by wrapping them with ** in the normalized text."""
    if not text:
        return ""
    if not keywords:
        return normalize_unicode(text)
    t = normalize_unicode(text)
    for kw in keywords:
        if not kw:
            continue
        kwn = normalize_unicode(kw)
        # Replace exact occurrences; case-sensitive as tests expect exact matches
        t = t.replace(kwn, f'**{kwn}**')
    return t

def normalize_text(text: str) -> str:
    """
    Convenience function for standard text normalization.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    processor = TextProcessor(
        normalize_unicode=True,
        normalize_whitespace=True,
        normalize_case=False,
        unicode_form='NFKC'
    )
    return processor.normalize_text(text)


def clean_policy_text(text: str) -> str:
    """
    Clean policy document text with standard settings.

    Args:
        text: Policy document text

    Returns:
        Cleaned text
    """
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_unwanted_characters(text: str) -> str:
    """
    Remove unwanted characters from text.

    Args:
        text: Input text

    Returns:
        Text with unwanted characters removed
    """
    # Remove control characters and special symbols
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    return text


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    # Split by double newlines or multiple spaces
    paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
    # Clean and filter empty paragraphs
    return [p.strip() for p in paragraphs if p.strip()]


def standardize_accents(text: str) -> str:
    """
    Standardize accents in Spanish text.

    Args:
        text: Input text with potentially non-standard accents

    Returns:
        Text with standardized accents
    """
    # Normalize to NFC (canonical composition)
    return unicodedata.normalize('NFC', text)


def normalize_spanish_text(text: str) -> str:
    """
    Convenience function for Spanish text normalization.
    
    Args:
        text: Input Spanish text
        
    Returns:
        Normalized text
    """
    processor = TextProcessor(
        normalize_unicode=True,
        normalize_whitespace=True,
        normalize_case=False,
        unicode_form='NFKC'
    )
    return processor.normalize_text(text)


def create_text_processor(
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
    normalize_case: bool = False,
    unicode_form: str = 'NFKC'
) -> TextProcessor:
    """
    Factory function to create text processor with specified options.
    
    Args:
        normalize_unicode: Whether to normalize Unicode characters
        normalize_whitespace: Whether to normalize whitespace
        normalize_case: Whether to normalize case (to lowercase)
        unicode_form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
        
    Returns:
        Configured TextProcessor instance
    """
    return TextProcessor(
        normalize_unicode=normalize_unicode,
        normalize_whitespace=normalize_whitespace,
        normalize_case=normalize_case,
        unicode_form=unicode_form
    )


if __name__ == "__main__":
    # Example usage
    processor = create_text_processor()
    
    # Test with some Spanish text containing different Unicode representations
    test_text = "Este  es  un\ttexto   con\nespaçios  irregulares y carácteres especiales.\nÑandú.\r\n"
    normalized = processor.normalize_text(test_text)
    
    print(f"Original: {repr(test_text)}")
    print(f"Normalized: {repr(normalized)}")
    
    # Test segmentation
    long_text = "Primera oración. " * 50 + "Segunda oración muy larga. " * 50
    segments = processor.segment_text(long_text, max_length=200, overlap=20)
    
    print(f"\nSegmented into {len(segments)} chunks:")
    for i, segment in enumerate(segments[:3]):
        print(f"Segment {i+1}: {segment[:50]}...")
    if len(segments) > 3:
        print(f"...and {len(segments)-3} more segments")