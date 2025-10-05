"""
Tests for Text Truncation Logger Module
"""

import io
import logging
import sys

from text_truncation_logger import (
    TextReference,
    TextTruncationLogger,
    create_text_ref,
    get_truncation_logger,
    log_info_with_text,
    truncate_text_for_log,
)


class TestTextReference:
    """Test TextReference dataclass functionality."""

    @staticmethod
    def test_text_reference_creation():
        """Test basic TextReference creation."""
        ref = TextReference("abc123", page_number=5,
                            relevance_score=0.85, length=150)
        assert ref.hash_id == "abc123"
        assert ref.page_number == 5
        assert ref.relevance_score == 0.85
        assert ref.length == 150

    @staticmethod
    def test_text_reference_str_all_fields():
        """Test string representation with all fields."""
        ref = TextReference("abc123", page_number=5,
                            relevance_score=0.85, length=150)
        result = str(ref)
        assert "#abc123" in result
        assert "p5" in result
        assert "rel:0.85" in result
        assert "len:150" in result

    @staticmethod
    def test_text_reference_str_partial_fields():
        """Test string representation with partial fields."""
        ref = TextReference("abc123", page_number=5)
        result = str(ref)
        assert "#abc123" in result
        assert "p5" in result
        assert "rel:" not in result
        assert "len:" not in result

    @staticmethod
    def test_text_reference_str_hash_only():
        """Test string representation with hash only."""
        ref = TextReference("abc123")
        result = str(ref)
        assert "#abc123" in result
        assert "p" not in result
        assert "rel:" not in result
        assert "len:" not in result


class TestTextTruncationLogger:
    """Test TextTruncationLogger functionality."""

    @staticmethod
    def test_initialization():
        """Test logger initialization with defaults."""
        logger = TextTruncationLogger()
        assert logger.max_log_length == 250
        assert logger.hash_length == 8
        assert len(logger.text_registry) == 0

    @staticmethod
    def test_initialization_custom():
        """Test logger initialization with custom values."""
        logger = TextTruncationLogger(max_log_length=100, hash_length=12)
        assert logger.max_log_length == 100
        assert logger.hash_length == 12

    @staticmethod
    def test_generate_text_hash():
        """Test hash generation consistency."""
        logger = TextTruncationLogger()
        text = "This is a test text for hashing"

        hash1 = logger.generate_text_hash(text)
        hash2 = logger.generate_text_hash(text)

        assert hash1 == hash2  # Should be consistent
        assert len(hash1) == 8  # Default hash length
        assert isinstance(hash1, str)

    @staticmethod
    def test_generate_text_hash_empty():
        """Test hash generation for empty text."""
        logger = TextTruncationLogger()
        assert logger.generate_text_hash("") == "empty"
        assert logger.generate_text_hash("   ") == "empty"
        assert logger.generate_text_hash(None) == "empty"

    @staticmethod
    def test_generate_text_hash_different_texts():
        """Test that different texts generate different hashes."""
        logger = TextTruncationLogger()
        hash1 = logger.generate_text_hash("Text one")
        hash2 = logger.generate_text_hash("Text two")
        assert hash1 != hash2

    @staticmethod
    def test_create_text_reference():
        """Test text reference creation."""
        logger = TextTruncationLogger()
        text = "Sample text for reference creation"

        ref = logger.create_text_reference(
            text, page_number=10, relevance_score=0.75)

        assert isinstance(ref, TextReference)
        assert ref.page_number == 10
        assert ref.relevance_score == 0.75
        assert ref.length == len(text)
        assert ref.hash_id in logger.text_registry
        assert logger.text_registry[ref.hash_id] == text

    @staticmethod
    def test_truncate_for_logging_short_text():
        """Test truncation of short text (should not be truncated)."""
        logger = TextTruncationLogger(max_log_length=100)
        short_text = "This is a short text."

        result = logger.truncate_for_logging(short_text)
        assert result == short_text
        assert len(logger.text_registry) == 0  # Should not be registered

    @staticmethod
    def test_truncate_for_logging_long_text():
        """Test truncation of long text."""
        logger = TextTruncationLogger(max_log_length=50)
        long_text = "This is a very long text that should be truncated because it exceeds the maximum length limit set for logging purposes."

        result = logger.truncate_for_logging(
            long_text, page_number=15, relevance_score=0.9
        )

        # The result may be longer than max_log_length due to the reference appending
        # but should be significantly shorter than the original
        assert len(result) < len(long_text)
        assert "..." in result
        assert "#" in result  # Should contain hash reference
        assert "p15" in result
        assert "rel:0.90" in result
        assert len(logger.text_registry) == 1

    @staticmethod
    def test_get_full_text():
        """Test retrieval of full text by hash."""
        logger = TextTruncationLogger()
        text = "Original text content for testing retrieval"

        ref = logger.create_text_reference(text)
        retrieved = logger.get_full_text(ref.hash_id)

        assert retrieved == text

    @staticmethod
    def test_get_full_text_nonexistent():
        """Test retrieval of non-existent hash."""
        logger = TextTruncationLogger()
        result = logger.get_full_text("nonexistent")
        assert result is None

    @staticmethod
    def test_get_registry_summary_empty():
        """Test registry summary when empty."""
        logger = TextTruncationLogger()
        summary = logger.get_registry_summary()

        assert summary["total_texts"] == 0
        assert summary["total_characters"] == 0

    @staticmethod
    def test_get_registry_summary_with_data():
        """Test registry summary with data."""
        logger = TextTruncationLogger()
        text1 = "First text"  # 10 chars
        text2 = "Second text for testing"  # 23 chars

        logger.create_text_reference(text1)
        logger.create_text_reference(text2)

        summary = logger.get_registry_summary()

        assert summary["total_texts"] == 2
        assert summary["total_characters"] == 33
        assert summary["average_length"] == 16.5
        assert len(summary["hash_ids"]) == 2

    @staticmethod
    def test_clear_registry():
        """Test clearing the registry."""
        logger = TextTruncationLogger()
        logger.create_text_reference("Test text")

        assert len(logger.text_registry) == 1

        logger.clear_registry()

        assert len(logger.text_registry) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @staticmethod
    def test_truncate_text_for_log():
        """Test global convenience function."""
        result = truncate_text_for_log("Short text")
        assert result == "Short text"

        # Clear global registry for clean test
        get_truncation_logger().clear_registry()

        long_text = "A" * 300  # Long text
        result = truncate_text_for_log(long_text, page_number=5)

        assert len(result) < len(long_text)
        assert "p5" in result

    @staticmethod
    def test_create_text_ref():
        """Test global text reference creation."""
        text = "Test text for reference"
        ref = create_text_ref(text, page_number=8, relevance_score=0.6)

        assert isinstance(ref, TextReference)
        assert ref.page_number == 8
        assert ref.relevance_score == 0.6


class TestLoggingIntegration:
    """Test integration with Python logging."""

    @staticmethod
    def test_log_info_with_text():
        """Test logging integration with info level."""
        # Create a string buffer to capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)

        # Create test logger
        test_logger = logging.getLogger("test_truncation")
        test_logger.setLevel(logging.INFO)
        test_logger.addHandler(handler)

        # Test short text (should not be truncated)
        log_info_with_text(test_logger, "Testing short text", "Short content")

        # Test long text (should be truncated)
        long_text = "This is a very long text " * 20  # Very long text
        log_info_with_text(
            test_logger,
            "Testing long text",
            long_text,
            page_number=42,
            relevance_score=0.95,
        )

        # Get logged output
        log_output = log_stream.getvalue()

        assert "Testing short text: Short content" in log_output
        assert "Testing long text:" in log_output
        assert "p42" in log_output
        assert "rel:0.95" in log_output

        # Cleanup
        test_logger.removeHandler(handler)
        handler.close()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_very_small_max_length():
        """Test with very small maximum length."""
        logger = TextTruncationLogger(max_log_length=10)
        text = "This text is longer than 10 characters"

        result = logger.truncate_for_logging(text)

        # Should still work, though result might be mostly reference
        assert len(result) >= 10  # At least the minimum to show reference
        assert "#" in result

    @staticmethod
    def test_unicode_text():
        """Test with unicode characters."""
        logger = TextTruncationLogger()
        unicode_text = "Texto con acentos: café, niño, señor, corazón, música 🎵"

        hash_id = logger.generate_text_hash(unicode_text)
        ref = logger.create_text_reference(unicode_text)

        assert len(hash_id) == 8
        assert logger.get_full_text(ref.hash_id) == unicode_text

    @staticmethod
    def test_newlines_and_tabs():
        """Test with newlines and tabs in text."""
        logger = TextTruncationLogger()
        text_with_whitespace = "Line 1\nLine 2\tTabbed content\n\nBlank line above"

        ref = logger.create_text_reference(text_with_whitespace)
        retrieved = logger.get_full_text(ref.hash_id)

        assert retrieved == text_with_whitespace


def run_all_tests():
    """Run all test classes manually."""

    test_classes = [
        TestTextReference,
        TestTextTruncationLogger,
        TestConvenienceFunctions,
        TestLoggingIntegration,
        TestEdgeCases,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        instance = test_class()

        # Get all test methods
        test_methods = [
            method for method in dir(instance) if method.startswith("test_")
        ]

        for method_name in test_methods:
            try:
                print(f"  {method_name}...", end=" ")
                method = getattr(instance, method_name)
                method()
                print("PASSED")
                passed += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1
                # Uncomment for detailed error info:
                # traceback.print_exc()

    print("\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
