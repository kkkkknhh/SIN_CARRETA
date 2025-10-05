"""
Test suite for SpaCy model loader components.
"""
import logging
import threading
import unittest
from unittest.mock import MagicMock, patch

from spacy_loader import (
    SpacyModelLoader,
    SafeSpacyProcessor,
    _reset_spacy_singleton_for_testing,
    get_spacy_model_loader,
)

class TestSpacyModelLoader(unittest.TestCase):
    """Tests for the SpacyModelLoader class."""
    
    @patch('spacy_loader.spacy.load')
    def test_load_existing_model(self, mock_load):
        """Test loading an existing model."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        loader = SpacyModelLoader()
        model = loader.load_model("test_model")
        
        self.assertEqual(model, mock_model)
        mock_load.assert_called_once_with("test_model")
    
    @patch('spacy_loader.spacy.load')
    def test_load_cached_model(self, mock_load):
        """Test that models are cached."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        loader = SpacyModelLoader()
        # Load model first time
        model1 = loader.load_model("test_model")
        # Load same model second time
        model2 = loader.load_model("test_model")
        
        # Should be same instance and load called only once
        self.assertEqual(model1, model2)
        mock_load.assert_called_once_with("test_model")
    
    @patch('spacy_loader.spacy.load')
    @patch('spacy_loader.download')
    def test_download_missing_model(self, mock_download, mock_load):
        """Test downloading a missing model."""
        # First call raises OSError, second call succeeds
        mock_model = MagicMock()
        mock_load.side_effect = [OSError("Model not found"), mock_model]
        
        loader = SpacyModelLoader()
        model = loader.load_model("test_model")
        
        self.assertEqual(model, mock_model)
        mock_download.assert_called_once_with("test_model")
        self.assertEqual(mock_load.call_count, 2)
    
    @patch('spacy_loader.spacy.load')
    @patch('spacy_loader.download')
    def test_download_failure(self, mock_download, mock_load):
        """Test handling download failures."""
        mock_load.side_effect = OSError("Model not found")
        mock_download.side_effect = Exception("Download failed")
        
        loader = SpacyModelLoader(max_retries=2)
        model = loader.load_model("test_model")
        
        self.assertIsNone(model)
        self.assertEqual(mock_download.call_count, 2)  # Should retry twice
    
    @patch('spacy_loader.spacy.load')
    def test_unexpected_error(self, mock_load):
        """Test handling unexpected errors."""
        mock_load.side_effect = RuntimeError("Unexpected error")
        
        loader = SpacyModelLoader()
        model = loader.load_model("test_model")
        
        self.assertIsNone(model)
    
    @patch('spacy_loader.spacy.load')
    def test_no_download_if_disabled(self, mock_load):
        """Test that download is not attempted if disabled."""
        mock_load.side_effect = OSError("Model not found")
        
        loader = SpacyModelLoader()
        with patch('spacy_loader.download') as mock_download:
            model = loader.load_model("test_model", download_if_missing=False)
            
            self.assertIsNone(model)
            mock_download.assert_not_called()
    
    @patch('spacy_loader.spacy.load')
    def test_get_model_info(self, mock_load):
        """Test getting model information."""
        mock_model1 = MagicMock()
        mock_model1.pipe_names = ["tagger", "parser"]
        mock_model1.has_vector_data = True
        mock_model1.pipeline = [1, 2]  # Mock pipeline components
        
        mock_model2 = MagicMock()
        mock_model2.pipe_names = ["ner"]
        mock_model2.has_vector_data = False
        mock_model2.pipeline = [1]  # Mock pipeline component
        
        mock_load.side_effect = [mock_model1, mock_model2]
        
        loader = SpacyModelLoader()
        loader.load_model("model1")
        loader.load_model("model2")
        
        info = loader.get_model_info()
        self.assertEqual(len(info), 2)
        self.assertEqual(info["model1"]["pipeline"], ["tagger", "parser"])
        self.assertTrue(info["model1"]["vectors"])
        self.assertEqual(info["model2"]["components"], 1)


class TestSafeSpacyProcessor(unittest.TestCase):
    """Tests for the SafeSpacyProcessor class."""
    
    def setUp(self):
        _reset_spacy_singleton_for_testing()
        self.addCleanup(_reset_spacy_singleton_for_testing)

    def test_full_functionality_mode(self):
        """Test processor with full spaCy functionality"""
        with patch('spacy.load') as mock_load:
            # Create mock spaCy doc
            mock_doc = MagicMock()
            mock_token = MagicMock()
            mock_token.text = "test"
            mock_token.lemma_ = "test"
            mock_token.pos_ = "NOUN"
            mock_doc.__iter__ = MagicMock(return_value=iter([mock_token]))
            mock_doc.ents = []
            mock_doc.sents = [MagicMock(text="Test sentence.")]
            
            mock_model = MagicMock()
            mock_model.return_value = mock_doc
            mock_load.return_value = mock_model
            
            processor = SafeSpacyProcessor(loader=SpacyModelLoader())
            result = processor.process_text("Test sentence.")
            
            self.assertEqual(result['processing_mode'], 'full')
            self.assertTrue(processor.is_fully_functional())
    
    def test_degraded_mode(self):
        """Test processor in degraded mode"""
        with patch('spacy.load') as mock_load, \
             patch('spacy.cli.download') as mock_download:
            
            # Model loading and download both fail
            mock_load.side_effect = OSError("Model not found")
            mock_download.side_effect = Exception("Download failed")
            
            processor = SafeSpacyProcessor(loader=SpacyModelLoader())
            result = processor.process_text("Test sentence. Another sentence.")
            
            self.assertEqual(result['processing_mode'], 'degraded')
            self.assertEqual(result['tokens'], ['Test', 'sentence.', 'Another', 'sentence.'])
            self.assertEqual(result['sentences'], ['Test sentence', ' Another sentence', ''])
            self.assertFalse(processor.is_fully_functional())
    
    def test_no_system_exit_on_failure(self):
        """Test that SystemExit is never raised"""
        with patch('spacy.load') as mock_load, \
             patch('spacy.cli.download') as mock_download:
            
            # All operations fail
            mock_load.side_effect = Exception("Critical error")
            mock_download.side_effect = Exception("Download failed")
            
            # This should not raise SystemExit
            try:
                processor = SafeSpacyProcessor(loader=SpacyModelLoader())
                result = processor.process_text("Test")
                # Should still get a result in degraded mode
                self.assertIsNotNone(result)
                self.assertEqual(result['processing_mode'], 'degraded')
            except SystemExit:
                self.fail("SystemExit should not be raised")


class TestSpacySingleton(unittest.TestCase):

    def setUp(self):
        _reset_spacy_singleton_for_testing()
        self.addCleanup(_reset_spacy_singleton_for_testing)

    def test_singleton_returns_same_instance(self):
        """get_spacy_model_loader returns a stable per-process singleton."""

        loader_a = get_spacy_model_loader()
        loader_b = get_spacy_model_loader()

        self.assertIs(loader_a, loader_b)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main()