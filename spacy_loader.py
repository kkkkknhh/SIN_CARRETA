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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread safety
_MODEL_CACHE = {}
_CACHE_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.RLock()

# Constants
DEFAULT_SPANISH_MODEL = "es_core_news_sm"
DEFAULT_ENGLISH_MODEL = "en_core_web_sm"
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY = 2  # seconds


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
        enable_caching: bool = True
    ):
        """
        Initialize the spaCy model loader.
        
        Args:
            default_model: Default model name to load if none specified
            auto_download: Whether to automatically download missing models
            enable_caching: Whether to cache loaded models
        """
        self.default_model = default_model
        self.auto_download = auto_download
        self.enable_caching = enable_caching
        
        # Track download attempts to avoid repeated failures
        self._download_attempts = {}
    
    def load_model(self, model_name: Optional[str] = None) -> Any:
        """
        Load a spaCy model with automatic download if needed.
        
        Args:
            model_name: Name of the model to load (uses default if None)
            
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
                        logger.debug(f"Using cached model: {model_to_load}")
                        return _MODEL_CACHE[model_to_load]
            
            try:
                # Attempt to load model
                logger.info(f"Loading spaCy model: {model_to_load}")
                nlp = spacy.load(model_to_load)
                
                # Cache model if enabled
                if self.enable_caching:
                    with _CACHE_LOCK:
                        _MODEL_CACHE[model_to_load] = nlp
                
                return nlp
            
            except OSError:
                # Model not found, attempt to download if auto_download is enabled
                if self.auto_download:
                    return self._download_and_load_model(model_to_load)
                else:
                    logger.warning(f"Model {model_to_load} not found and auto_download is disabled")
        
        except ImportError:
            logger.error("spaCy not available. Install with: pip install spacy")
        
        return None
    
    def _download_and_load_model(self, model_name: str) -> Any:
        """
        Download and load a spaCy model with retry logic.
        
        Args:
            model_name: Name of the model to download and load
            
        Returns:
            Loaded spaCy model, or None if download/loading fails
        """
        # Check if we've already tried too many times
        if self._download_attempts.get(model_name, 0) >= DOWNLOAD_RETRY_ATTEMPTS:
            logger.warning(f"Maximum download attempts reached for {model_name}")
            return None
        
        # Increment attempt counter
        self._download_attempts[model_name] = self._download_attempts.get(model_name, 0) + 1
        
        # Use lock to prevent multiple threads from downloading simultaneously
        with _DOWNLOAD_LOCK:
            try:
                import spacy
                from spacy.cli.download import download as spacy_download
                
                logger.info(f"Downloading spaCy model: {model_name}")
                spacy_download(model_name)
                
                # Attempt to load the downloaded model
                logger.info(f"Loading downloaded model: {model_name}")
                nlp = spacy.load(model_name)
                
                # Cache model if enabled
                if self.enable_caching:
                    with _CACHE_LOCK:
                        _MODEL_CACHE[model_name] = nlp
                
                # Reset attempt counter on success
                self._download_attempts[model_name] = 0
                
                return nlp
            
            except Exception as e:
                logger.error(f"Failed to download/load model {model_name}: {e}")
                
                # Wait before retry
                if self._download_attempts.get(model_name, 0) < DOWNLOAD_RETRY_ATTEMPTS:
                    time.sleep(DOWNLOAD_RETRY_DELAY)
                    return self._download_and_load_model(model_name)
        
        return None
    
    def get_degraded_processor(self) -> 'SafeSpacyProcessor':
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
    
    def __init__(self, nlp: Any):
        """
        Initialize the processor.
        
        Args:
            nlp: Loaded spaCy model, or None for degraded mode
        """
        self.nlp = nlp
        self.is_degraded = nlp is None
        
        # Set up basic tokenization pattern for degraded mode
        self.token_pattern = r'(?u)\b\w\w+\b'
    
    def process(self, text: str) -> Any:
        """
        Process text with spaCy model or degraded functionality.
        
        Args:
            text: Text to process
            
        Returns:
            spaCy Doc object or DegradedDoc in degraded mode
        """
        if not text:
            return self._create_empty_doc()
        
        if self.is_degraded:
            return self._process_degraded(text)
        
        try:
            return self.nlp(text)
        except Exception as e:
            logger.warning(f"Error processing text with spaCy: {e}")
            self.is_degraded = True
            return self._process_degraded(text)
    
    def _process_degraded(self, text: str) -> 'DegradedDoc':
        """Process text in degraded mode."""
        import re
        tokens = re.findall(self.token_pattern, text)
        return DegradedDoc(text, tokens)
    
    def _create_empty_doc(self) -> Union[Any, 'DegradedDoc']:
        """Create an empty document object."""
        if self.is_degraded:
            return DegradedDoc("", [])
        try:
            return self.nlp("")
        except:
            self.is_degraded = True
            return DegradedDoc("", [])
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities with metadata
        """
        doc = self.process(text)
        
        if self.is_degraded:
            # Very basic entity extraction in degraded mode
            # This is just a placeholder - real NER requires the actual model
            return []
        
        try:
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
        except:
            return []
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token texts
        """
        doc = self.process(text)
        
        if self.is_degraded:
            return doc.tokens
        
        try:
            return [token.text for token in doc]
        except:
            import re
            return re.findall(self.token_pattern, text)
    
    def is_available(self) -> bool:
        """Check if full spaCy functionality is available."""
        return not self.is_degraded


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
