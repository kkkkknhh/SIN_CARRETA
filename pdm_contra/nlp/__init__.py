"""NLP utilities for ``pdm_contra``."""

from .nli import SpanishNLIDetector
from .patterns import PatternMatcher

__all__ = ["PatternMatcher", "SpanishNLIDetector"]
