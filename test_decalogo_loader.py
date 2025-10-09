"""
Test suite for DECALOGO_INDUSTRIAL template loader.
"""
import unittest
from decalogo_loader import (
    load_decalogo_industrial,
    get_decalogo_industrial,
    load_dnp_standards,
    ensure_aligned_templates,
    get_question_by_id,
    get_dimension_weight,
    DECALOGO_INDUSTRIAL_TEMPLATE
)


class TestDecalogoLoader(unittest.TestCase):
    """Tests for the DECALOGO_INDUSTRIAL template loader."""
    
    def setUp(self):
        # Reset cache before each test
        import decalogo_loader
        decalogo_loader._DECALOGO_CACHE.clear()
    
    def test_load_industrial_template(self):
        """Test loading decalogo_industrial.json."""
        template = get_decalogo_industrial()
        self.assertIsInstance(template, dict)
        self.assertIn("version", template)
        self.assertIn("questions", template)
    
    def test_loading_dnp_standards(self):
        """Test loading dnp-standards.latest.clean.json."""
        from decalogo_loader import load_dnp_standards
        result = load_dnp_standards()
        self.assertIsInstance(result, dict)
        # The actual file has 'version' at root level, not nested in 'metadata'
        self.assertIn("version", result)

    def test_ensure_aligned_templates(self):
        """Test loading all templates together."""
        templates = ensure_aligned_templates()
        self.assertIn("decalogo_industrial", templates)
        self.assertIn("dnp_standards", templates)
        self.assertIn("alignment", templates)
        self.assertEqual(templates["alignment"]["questions_found"], 300)
    
    def test_get_question_by_id(self):
        """Test retrieving specific question."""
        question = get_question_by_id("D1-Q1")
        if question:
            self.assertIn("id", question)
            self.assertEqual(question["id"], "D1-Q1")
    
    def test_get_dimension_weight(self):
        """Test retrieving dimension weight."""
        weight = get_dimension_weight("P1", "D1")
        self.assertIsInstance(weight, float)
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
    
    def test_fallback_on_read_error(self):
        """Test fallback to template when file read fails."""
        content = load_decalogo_industrial("/nonexistent/path/file.json")
        self.assertEqual(content, DECALOGO_INDUSTRIAL_TEMPLATE)
    
    def test_caching(self):
        """Test that template content is cached."""
        content1 = get_decalogo_industrial()
        content2 = get_decalogo_industrial()
        self.assertIs(content1, content2)


if __name__ == "__main__":
    unittest.main()
