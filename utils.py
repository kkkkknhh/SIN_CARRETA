import re
import unicodedata
from typing import List, Dict, Optional


def normalize_text(text: str) -> str:
    """Normalize Unicode text using NFKC normalization for consistent regex matching."""
    return unicodedata.normalize("NFKC", text)


class TextAnalyzer:
    """Text analysis utility with Unicode normalization."""

    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}',
            'hashtag': r'#\w+',
            'mention': r'@\w+',
        }

    def find_pattern_matches(self, text: str, pattern_name: str) -> List[str]:
        """Find matches for a named pattern with Unicode normalization."""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        normalized_text = normalize_text(text)
        pattern = self.patterns[pattern_name]
        return re.findall(pattern, normalized_text)

    def count_pattern_matches(self, text: str, pattern_name: str) -> int:
        """Count matches for a named pattern with Unicode normalization."""
        matches = self.find_pattern_matches(text, pattern_name)
        return len(matches)

    def extract_all_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract all pattern matches with Unicode normalization."""
        normalized_text = normalize_text(text)
        results = {}

        for pattern_name, pattern in self.patterns.items():
            results[pattern_name] = re.findall(pattern, normalized_text)

        return results

    @staticmethod
    def clean_text(text: str, preserve_spaces: bool = True) -> str:
        """Clean text by removing special characters with Unicode normalization."""
        normalized_text = normalize_text(text)

        if preserve_spaces:
            pattern = r'[^\w\s]'
            return re.sub(pattern, '', normalized_text)
        else:
            pattern = r'[^\w]'
            return re.sub(pattern, '', normalized_text)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words with Unicode normalization."""
        normalized_text = normalize_text(text)
        pattern = r'\b\w+\b'
        return re.findall(pattern, normalized_text.lower())

    @staticmethod
    def find_quoted_text(text: str) -> List[str]:
        """Extract quoted text with Unicode normalization."""
        normalized_text = normalize_text(text)
        # Pattern to match various quote types
        pattern = r'[""\'\""]([^""\'\"]*)[""\'\""]'
        matches = re.findall(pattern, normalized_text)
        return matches

    @staticmethod
    def replace_unicode_punctuation(text: str, replacement: str = ' ') -> str:
        """Replace Unicode punctuation marks with specified replacement."""
        normalized_text = normalize_text(text)
        # Pattern for various Unicode punctuation
        pattern = r'[—–−‐‑‒―''""„‚‛‟«»‹›]'
        return re.sub(pattern, replacement, normalized_text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize all whitespace characters with Unicode normalization."""
        normalized_text = normalize_text(text)
        # Replace various Unicode whitespace with regular space
        pattern = r'\s+'
        return re.sub(pattern, ' ', normalized_text).strip()


def search_and_replace_normalized(text: str, search_pattern: str, replacement: str,
                                  flags: int = 0) -> str:
    """Search and replace with Unicode normalization."""
    normalized_text = normalize_text(text)
    normalized_pattern = normalize_text(search_pattern)
    normalized_replacement = normalize_text(replacement)

    return re.sub(normalized_pattern, normalized_replacement, normalized_text, flags=flags)


def split_normalized(text: str, pattern: str, maxsplit: int = 0) -> List[str]:
    """Split text with Unicode normalization."""
    normalized_text = normalize_text(text)
    normalized_pattern = normalize_text(pattern)

    return re.split(normalized_pattern, normalized_text, maxsplit=maxsplit)


def match_normalized(text: str, pattern: str, flags: int = 0) -> Optional[re.Match[str]]:
    """Match pattern with Unicode normalization."""
    normalized_text = normalize_text(text)
    normalized_pattern = normalize_text(pattern)

    return re.match(normalized_pattern, normalized_text, flags=flags)
