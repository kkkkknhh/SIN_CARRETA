"""
Text Truncation Logging Utilities
Provides text truncation mechanisms for logging systems that limit logged text content
to a maximum length and replace full text with hash references, page numbers, and relevance scores.
"""
import hashlib
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TextReference:
    """Represents a truncated text reference for logging."""
    hash_id: str
    page_number: Optional[int] = None
    relevance_score: Optional[float] = None
    length: int = 0
    
    def __str__(self) -> str:
        parts = [f"#{self.hash_id}"]
        if self.page_number is not None:
            parts.append(f"p{self.page_number}")
        if self.relevance_score is not None:
            parts.append(f"rel:{self.relevance_score:.2f}")
        if self.length > 0:
            parts.append(f"len:{self.length}")
        return f"[{' '.join(parts)}]"


class TextTruncationLogger:
    """
    Logger utility that truncates text content in logs while maintaining full content availability
    in output files (Markdown and JSON).
    """
    
    def __init__(self, max_log_length: int = 250, hash_length: int = 8):
        """
        Initialize text truncation logger.
        
        Args:
            max_log_length: Maximum characters to display in logs (default 250)
            hash_length: Length of hash identifiers (default 8)
        """
        self.max_log_length = max_log_length
        self.hash_length = hash_length
        self.text_registry = {}  # Store full text content mapped by hash
    
    def generate_text_hash(self, text: str) -> str:
        """
        Generate consistent hash identifier for text segments.
        
        Args:
            text: Input text to hash
            
        Returns:
            Truncated hash string for use as identifier
        """
        if not text or not text.strip():
            return "empty"
        
        # Use MD5 for fast, consistent hashing (not for security)
        hash_obj = hashlib.md5(text.encode('utf-8'))
        full_hash = hash_obj.hexdigest()
        return full_hash[:self.hash_length]
    
    def create_text_reference(
        self, 
        text: str, 
        page_number: Optional[int] = None, 
        relevance_score: Optional[float] = None
    ) -> TextReference:
        """
        Create a text reference object with hash and metadata.
        
        Args:
            text: Full text content
            page_number: Optional page number where text appears
            relevance_score: Optional relevance score for the text
            
        Returns:
            TextReference object for use in logging
        """
        hash_id = self.generate_text_hash(text)
        
        # Store full text in registry for later retrieval
        self.text_registry[hash_id] = text
        
        return TextReference(
            hash_id=hash_id,
            page_number=page_number,
            relevance_score=relevance_score,
            length=len(text)
        )
    
    def truncate_for_logging(
        self, 
        text: str, 
        page_number: Optional[int] = None, 
        relevance_score: Optional[float] = None
    ) -> str:
        """
        Truncate text for logging purposes, creating abbreviated reference if needed.
        
        Args:
            text: Full text content
            page_number: Optional page number where text appears
            relevance_score: Optional relevance score for the text
            
        Returns:
            Truncated text suitable for logging
        """
        if not text or len(text) <= self.max_log_length:
            return text
        
        # Create reference for long text
        ref = self.create_text_reference(text, page_number, relevance_score)
        
        # Return truncated text with reference
        truncated = text[:self.max_log_length - 20]  # Leave space for reference
        return f"{truncated}... {ref}"
    
    def get_full_text(self, hash_id: str) -> Optional[str]:
        """
        Retrieve full text content by hash identifier.
        
        Args:
            hash_id: Hash identifier for the text
            
        Returns:
            Full text content or None if not found
        """
        return self.text_registry.get(hash_id)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the text registry.
        
        Returns:
            Dictionary with registry statistics
        """
        if not self.text_registry:
            return {"total_texts": 0, "total_characters": 0}
        
        total_chars = sum(len(text) for text in self.text_registry.values())
        return {
            "total_texts": len(self.text_registry),
            "total_characters": total_chars,
            "average_length": total_chars / len(self.text_registry) if self.text_registry else 0,
            "hash_ids": list(self.text_registry.keys())
        }
    
    def clear_registry(self):
        """Clear the text registry to free memory."""
        self.text_registry.clear()


# Global instance for use throughout the application
_global_truncation_logger = TextTruncationLogger()


def get_truncation_logger() -> TextTruncationLogger:
    """Get the global text truncation logger instance."""
    return _global_truncation_logger


def truncate_text_for_log(
    text: str, 
    page_number: Optional[int] = None, 
    relevance_score: Optional[float] = None
) -> str:
    """
    Convenience function to truncate text for logging using the global logger.
    
    Args:
        text: Full text content
        page_number: Optional page number where text appears
        relevance_score: Optional relevance score for the text
        
    Returns:
        Truncated text suitable for logging
    """
    return _global_truncation_logger.truncate_for_logging(text, page_number, relevance_score)


def create_text_ref(
    text: str, 
    page_number: Optional[int] = None, 
    relevance_score: Optional[float] = None
) -> TextReference:
    """
    Convenience function to create text reference using the global logger.
    
    Args:
        text: Full text content
        page_number: Optional page number where text appears
        relevance_score: Optional relevance score for the text
        
    Returns:
        TextReference object for use in logging
    """
    return _global_truncation_logger.create_text_reference(text, page_number, relevance_score)


# Enhanced logging functions that automatically truncate text
def log_with_truncation(logger: logging.Logger, level: int, message: str, text_content: str = "", 
                       page_number: Optional[int] = None, relevance_score: Optional[float] = None):
    """
    Log message with automatic text truncation.
    
    Args:
        logger: Logger instance to use
        level: Logging level (logging.INFO, logging.WARNING, etc.)
        message: Base log message
        text_content: Text content to potentially truncate
        page_number: Optional page number for the text
        relevance_score: Optional relevance score for the text
    """
    if text_content:
        truncated = truncate_text_for_log(text_content, page_number, relevance_score)
        full_message = f"{message}: {truncated}"
    else:
        full_message = message
    
    logger.log(level, full_message)


def log_info_with_text(logger: logging.Logger, message: str, text_content: str = "",
                      page_number: Optional[int] = None, relevance_score: Optional[float] = None):
    """Log info message with text truncation."""
    log_with_truncation(logger, logging.INFO, message, text_content, page_number, relevance_score)


def log_warning_with_text(logger: logging.Logger, message: str, text_content: str = "",
                         page_number: Optional[int] = None, relevance_score: Optional[float] = None):
    """Log warning message with text truncation."""
    log_with_truncation(logger, logging.WARNING, message, text_content, page_number, relevance_score)


def log_debug_with_text(logger: logging.Logger, message: str, text_content: str = "",
                       page_number: Optional[int] = None, relevance_score: Optional[float] = None):
    """Log debug message with text truncation."""
    log_with_truncation(logger, logging.DEBUG, message, text_content, page_number, relevance_score)


def log_error_with_text(logger: logging.Logger, message: str, text_content: str = "",
                       page_number: Optional[int] = None, relevance_score: Optional[float] = None):
    """Log error message with text truncation."""
    log_with_truncation(logger, logging.ERROR, message, text_content, page_number, relevance_score)