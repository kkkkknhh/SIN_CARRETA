import unittest
import unicodedata
from text_processor import (
    normalize_unicode, find_quotes, count_words, extract_emails,
    replace_special_chars, split_sentences, search_pattern,
    match_phone_numbers, highlight_keywords
)


class TestUnicodeNormalization(unittest.TestCase):
    """Test Unicode normalization in regex functions."""
    
    def setUp(self):
        # Test strings with different Unicode representations
        self.test_strings = {
            'quotes_mixed': '"Hello" "World" "Test" "Quote"',  # Various quote types
            'accented_text': 'café résumé naïve',  # Accented characters
            'composed_vs_decomposed': 'é vs é',  # Same visually, different encodings
            'smart_quotes': '"Smart quotes" vs "regular quotes"',
            'mixed_unicode': 'Test—with—em—dashes and "smart quotes"'
        }
    
    def test_normalization_consistency(self):
        """Test that normalization produces consistent results."""
        for key, text in self.test_strings.items():
            normalized = normalize_unicode(text)
            # Double normalization should be idempotent
            double_normalized = normalize_unicode(normalized)
            self.assertEqual(normalized, double_normalized, 
                           f"Double normalization failed for {key}")
    
    def test_quote_counting_normalization(self):
        """Test that quote counting is consistent after normalization."""
        # Different Unicode quote representations
        text_with_smart_quotes = '"Hello" "World"'
        text_with_regular_quotes = '"Hello" "World"'
        
        smart_quotes = find_quotes(text_with_smart_quotes)
        regular_quotes = find_quotes(text_with_regular_quotes)
        
        # Should find quotes regardless of Unicode representation
        self.assertTrue(len(smart_quotes) > 0)
        self.assertTrue(len(regular_quotes) > 0)
    
    def test_word_counting_with_accents(self):
        """Test word counting with accented characters."""
        text1 = 'café résumé naïve'
        text2 = 'cafe resume naive'  # Without accents
        
        count1 = count_words(text1)
        count2 = count_words(text2)
        
        # Both should count 3 words
        self.assertEqual(count1, 3)
        self.assertEqual(count2, 3)
    
    def test_email_extraction_unicode(self):
        """Test email extraction with Unicode characters."""
        text = 'Contact: josé@example.com or maría@test.org'
        emails = extract_emails(text)
        
        # Should extract both emails despite Unicode characters
        self.assertEqual(len(emails), 2)
    
    def test_pattern_search_unicode(self):
        """Test pattern searching with Unicode normalization."""
        text = 'This is a test—with em dash'
        pattern = 'test—with'
        
        match = search_pattern(text, pattern)
        self.assertIsNotNone(match)
    
    def test_sentence_splitting_unicode(self):
        """Test sentence splitting with Unicode punctuation."""
        text = 'First sentence. Second sentence! Third sentence?'
        sentences = split_sentences(text)
        
        # Should split into sentences, filtering empty strings
        non_empty_sentences = [s for s in sentences if s.strip()]
        self.assertEqual(len(non_empty_sentences), 3)
    
    def test_special_char_replacement(self):
        """Test special character replacement with Unicode."""
        text = 'Text—with—em—dashes and "smart quotes"'
        replaced = replace_special_chars(text)
        
        # Should replace Unicode punctuation
        self.assertNotIn('—', replaced)
        self.assertNotIn('"', replaced)
        self.assertNotIn('"', replaced)
    
    def test_phone_number_matching(self):
        """Test phone number matching with Unicode spaces."""
        text = 'Call 123-456-7890 or (555) 123-4567'
        phones = match_phone_numbers(text)
        
        # Should find both phone numbers
        self.assertEqual(len(phones), 2)
    
    def test_keyword_highlighting(self):
        """Test keyword highlighting with Unicode."""
        text = 'This café has great résumé services'
        keywords = ['café', 'résumé']
        
        highlighted = highlight_keywords(text, keywords)
        
        # Should highlight both Unicode keywords
        self.assertIn('**café**', highlighted)
        self.assertIn('**résumé**', highlighted)
    
    def test_normalization_prevents_overcounting(self):
        """Test that normalization prevents overcounting identical characters."""
        # Create text with same character in different Unicode forms  
        text1 = 'é'  # Composed form
        text2 = 'e\u0301'  # Decomposed form (base + combining accent)
        
        # Before normalization, they might be different
        self.assertNotEqual(text1, text2)
        
        # After normalization, they should be the same
        norm1 = normalize_unicode(text1)
        norm2 = normalize_unicode(text2)
        self.assertEqual(norm1, norm2)
        
        # Word counts should be identical
        count1 = count_words(text1 + ' word')
        count2 = count_words(text2 + ' word')
        self.assertEqual(count1, count2)


if __name__ == '__main__':
    # Run comparison test to show before/after normalization effects
    print("Unicode Normalization Test Results:")
    print("=" * 50)
    
    test_cases = [
        '"Smart quotes" vs "regular quotes"',
        'café vs cafe',
        'résumé—summary', 
        'Text with – various — dash types'
    ]
    
    for text in test_cases:
        print(f"\nOriginal text: {text}")
        print(f"Normalized:    {unicodedata.normalize('NFKC', text)}")
        
        # Show character differences
        original_chars = len(text)
        normalized_chars = len(unicodedata.normalize('NFKC', text))
        print(f"Character count - Original: {original_chars}, Normalized: {normalized_chars}")
    
    # Run unit tests
    unittest.main()