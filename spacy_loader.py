# coding=utf-8
# coding=utf-8
"""
SpaCy Model Loader Module

Provides robust loading of spaCy models with automatic download capabilities
and graceful degradation when models are unavailable.

Features:
- Automatic model download with configurable retry logic
- Model caching to prevent redundant loading
- Graceful degradation to basic processing when models unavailable
- Thread-safe operations
- Comprehensive error logging
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread safety
_MODEL_CACHE = {}
_CACHE_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.RLock()

# Singleton
_SINGLETON_LOADER = None

# Constants
DEFAULT_SPANISH_MODEL = "es_core_news_sm"
DEFAULT_ENGLISH_MODEL = "en_core_web_sm"
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY = 2  # seconds


def download(model_name: str) -> None:
    """Wrapper for spaCy CLI download to allow easy patching in tests."""
    try:
        from spacy.cli import download as spacy_download
        spacy_download(model_name)
    except Exception as e:
        raise e


def get_spacy_model_loader() -> 'SpacyModelLoader':
    global _SINGLETON_LOADER
    if _SINGLETON_LOADER is None:
        _SINGLETON_LOADER = SpacyModelLoader()
    return _SINGLETON_LOADER


def _reset_spacy_singleton_for_testing():
    """Reset singleton and caches for isolated unit tests."""
    global _SINGLETON_LOADER
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
    _SINGLETON_LOADER = None


class SpacyModelLoader:
    """
    Robust spaCy model loader with automatic download and fallback mechanisms.
    
    Provides thread-safe loading of spaCy models with automatic download capabilities,
    retry logic, and graceful degradation when models are unavailable.
    """
    
    def __init__(
        self,
        default_model: str = DEFAULT_SPANISH_MODEL,
        auto_download: bool = True,
        enable_caching: bool = True,
        max_retries: int = DOWNLOAD_RETRY_ATTEMPTS,
    ):
        """
        Initialize the spaCy model loader.
        
        Args:
            default_model: Default model name to load if none specified
            auto_download: Whether to automatically download missing models
            enable_caching: Whether to cache loaded models
            max_retries: Maximum download retry attempts
        """
        self.default_model = default_model
        self.auto_download = auto_download
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        
        # Track download attempts to avoid repeated failures
        self._download_attempts = {}
    
    def load_model(self, model_name: Optional[str] = None, download_if_missing: bool = True) -> Any:
        """
        Load a spaCy model with automatic download if needed.
        
        Args:
            model_name: Name of the model to load (uses default if None)
            download_if_missing: Whether to attempt to download if missing
            
        Returns:
            Loaded spaCy model, or None if loading fails
        """
        try:
            import spacy
            
            # Use default model if none specified
            model_to_load = model_name or self.default_model
            
            # Check cache first if enabled
            if self.enable_caching:
                with _CACHE_LOCK:
                    if model_to_load in _MODEL_CACHE:
                        logger.debug("Using cached model: %s", model_to_load)
                        return _MODEL_CACHE[model_to_load]
            
            try:
                # Attempt to load model
                logger.info("Loading spaCy model: %s", model_to_load)
                nlp = spacy.load(model_to_load)
                
                # Cache model if enabled
                if self.enable_caching:
                    with _CACHE_LOCK:
                        _MODEL_CACHE[model_to_load] = nlp
                
                return nlp
            
            except OSError:
                # Model not found, attempt to download if enabled
                if self.auto_download and download_if_missing:
                    # Retry download up to max_retries times
                    for _ in range(self.max_retries):
                        try:
                            download(model_to_load)
                            # Try loading again after download
                            nlp = spacy.load(model_to_load)
                            if self.enable_caching:
                                with _CACHE_LOCK:
                                    _MODEL_CACHE[model_to_load] = nlp
                            return nlp
                        except Exception as e:
                            logger.warning("Download attempt failed for %s: %s", model_to_load, e)
                            continue
                else:
                    logger.warning("Model %s not found and download is disabled", model_to_load)
            except Exception as e:
                logger.error("Unexpected error loading model %s: %s", model_to_load, e)
        
        except ImportError:
            logger.error("spaCy not available. Install with: pip install spacy")
        
        return None
    
    def _download_and_load_model(self, model_name: str) -> Any:
        """
        Download and load a spaCy model with retry logic.
        (Deprecated internal helper)
        """
        # Backwards-compat shim uses new logic in load_model
        return self.load_model(model_name)
    
    @staticmethod
    def get_degraded_processor() -> 'SafeSpacyProcessor':
        """
        Get a degraded processor that can handle text without spaCy.
        
        Returns:
            SafeSpacyProcessor instance that works without spaCy
        """
        return SafeSpacyProcessor(None)


class SafeSpacyProcessor:
    """
    Safe wrapper around spaCy for text processing with graceful degradation.
    
    Provides basic text processing functionality that works even when
    spaCy models are unavailable, with graceful degradation.
    """
    
    def __init__(self, loader: Optional[SpacyModelLoader] = None):
        """
        Initialize the processor.
        
        Args:
            loader: SpacyModelLoader instance to obtain models; if None uses singleton
        """
        self.loader = loader or get_spacy_model_loader()
        self._nlp = None  # lazily loaded
        self._processing_mode = 'degraded'
    
    def _ensure_model(self) -> Optional[Any]:
        if self._nlp is not None:
            return self._nlp
        if self.loader:
            self._nlp = self.loader.load_model(self.loader.default_model)
        return self._nlp

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and return a dict with tokens, sentences, and mode."""
        if not text:
            return {"processing_mode": "degraded", "tokens": [], "sentences": []}
        nlp = self._ensure_model()
        if nlp is None:
            # Degraded mode – simple splits matching unit tests
            tokens = text.split()
            sentences = text.split('.')
            self._processing_mode = 'degraded'
            return {"processing_mode": "degraded", "tokens": tokens, "sentences": sentences}
        try:
            doc = nlp(text)
            tokens = [t.text for t in doc]
            sents = [s.text for s in getattr(doc, 'sents', [])]
            self._processing_mode = 'full'
            return {"processing_mode": "full", "tokens": tokens, "sentences": sents}
        except Exception:
            # Fallback to degraded
            tokens = text.split()
            sentences = text.split('.')
            self._processing_mode = 'degraded'
            return {"processing_mode": "degraded", "tokens": tokens, "sentences": sentences}

    def is_fully_functional(self) -> bool:
        return self._ensure_model() is not None

    # Backward-compatible helpers used elsewhere
    def process(self, text: str) -> Any:
        nlp = self._ensure_model()
        if nlp is None:
            return DegradedDoc(text, text.split())
        try:
            return nlp(text)
        except Exception:
            return DegradedDoc(text, text.split())

    def tokenize(self, text: str) -> List[str]:
        result = self.process_text(text)
        return result.get("tokens", [])

    def is_available(self) -> bool:
        return self.is_fully_functional()


class DegradedDoc:
    """Simple document class for degraded mode operation."""
    
    def __init__(self, text: str, tokens: List[str]):
        """
        Initialize degraded document.
        
        Args:
            text: Original text
            tokens: List of tokens
        """
        self.text = text
        self.tokens = tokens
        self.ents = []
    
    def __iter__(self):
        """Iterate over tokens."""
        for token in self.tokens:
            yield DegradedToken(token)
    
    def __len__(self):
        """Get number of tokens."""
        return len(self.tokens)


class DegradedToken:
    """Simple token class for degraded mode operation."""
    
    def __init__(self, text: str):
        """
        Initialize degraded token.
        
        Args:
            text: Token text
        """
        self.text = text
        self.lemma_ = text.lower()
        self.pos_ = "UNKNOWN"
        self.dep_ = "UNKNOWN"
        self.is_stop = False
    
    def __str__(self):
        """String representation of token."""
        return self.text


def load_spanish_model(auto_download: bool = True) -> Any:
    """
    Convenience function to load Spanish spaCy model.
    
    Args:
        auto_download: Whether to automatically download if missing
        
    Returns:
        Loaded Spanish spaCy model or None if unavailable
    """
    loader = SpacyModelLoader(DEFAULT_SPANISH_MODEL, auto_download)
    return loader.load_model()


def create_safe_processor(model_name: Optional[str] = None, auto_download: bool = True) -> SafeSpacyProcessor:
    """
    Create a safe spaCy processor that handles degraded operation.
    
    Args:
        model_name: Name of spaCy model to use
        auto_download: Whether to download model if missing
        
    Returns:
        SafeSpacyProcessor instance
    """
    loader = SpacyModelLoader(model_name or DEFAULT_SPANISH_MODEL, auto_download)
    nlp = loader.load_model()
    return SafeSpacyProcessor(nlp)


if __name__ == "__main__":
    # Example usage
    loader = SpacyModelLoader()
    nlp = loader.load_model()
    
    if nlp:
        processor = SafeSpacyProcessor(nlp)
        print("spaCy model loaded successfully")
    else:
        processor = loader.get_degraded_processor()
        print("Using degraded mode (spaCy model not available)")
    
    # Test text processing
    sample_text = "El plan de desarrollo municipal de Medellín 2024-2027 implementará programas educativos."
    doc = processor.process(sample_text)
    
    print("\nTokenization result:")
    tokens = processor.tokenize(sample_text)
    print(tokens)
    
    print("\nEntity extraction result:")
    entities = processor.extract_entities(sample_text)
    for entity in entities:
        print(f"- {entity['text']} ({entity['label']})")
    
    print(f"\nProcessor mode: {'Full spaCy' if processor.is_available() else 'Degraded'}")
