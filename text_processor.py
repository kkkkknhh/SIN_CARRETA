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
            logger.warning(f"Invalid Unicode form '{unicode_form}', using 'NFKC'")
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