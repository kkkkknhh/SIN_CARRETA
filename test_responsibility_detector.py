"""
Test suite for responsibility detection.
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
from responsibility_detector import ResponsibilityDetector, ResponsibilityEntity, EntityType


class TestResponsibilityDetector(unittest.TestCase):
    """Tests for the ResponsibilityDetector class."""

    def setUp(self):
        """Set up mock model loader."""
        self.mock_model_loader = MagicMock()
        self.mock_nlp = MagicMock()
        self.mock_model_loader.load_model.return_value = self.mock_nlp

    def test_detect_empty_text(self):
        """Test detection with empty text."""
        detector = ResponsibilityDetector(self.mock_model_loader)
        result = detector.detect_entities("")
        self.assertEqual(result, [])

    def test_detect_government_pattern(self):
        """Test detection of government pattern."""
        detector = ResponsibilityDetector(self.mock_model_loader)
        text = "El Ministerio de Educación implementará la política."

        # Mock NER to return no entities
        mock_doc = MagicMock()
        mock_doc.ents = []
        self.mock_nlp.return_value = mock_doc

        result = detector.detect_entities(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Ministerio de Educación")
        self.assertEqual(result[0].entity_type, EntityType.GOVERNMENT)
        self.assertGreaterEqual(result[0].confidence, 0.8)

    def test_detect_position_pattern(self):
        """Test detection of position pattern."""
        detector = ResponsibilityDetector(self.mock_model_loader)
        text = "El alcalde anunció nuevas medidas para la ciudad."

        # Mock NER to return no entities
        mock_doc = MagicMock()
        mock_doc.ents = []
        self.mock_nlp.return_value = mock_doc

        result = detector.detect_entities(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "alcalde")
        self.assertEqual(result[0].entity_type, EntityType.POSITION)

    def test_detect_institution_pattern(self):
        """Test detection of institution pattern."""
        detector = ResponsibilityDetector(self.mock_model_loader)
        text = "La universidad implementará nuevos programas."

        # Mock NER to return no entities
        mock_doc = MagicMock()
        mock_doc.ents = []
        self.mock_nlp.return_value = mock_doc

        result = detector.detect_entities(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "universidad")
        self.assertEqual(result[0].entity_type, EntityType.INSTITUTION)

    def test_detect_ner_entities(self):
        """Test detection using spaCy NER."""
        detector = ResponsibilityDetector(self.mock_model_loader)
        text = "Juan Pérez presentó el informe."

        # Create mock entity
        mock_ent = MagicMock()
        mock_ent.text = "Juan Pérez"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 10

        # Create mock doc with entity
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        self.mock_nlp.return_value = mock_doc

        result = detector.detect_entities(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Juan Pérez")
        self.assertEqual(result[0].entity_type, EntityType.PERSON)

    def test_merge_overlapping_entities(self):
        """Test merging of overlapping entities."""
        detector = ResponsibilityDetector(self.mock_model_loader)

        entities = [
            ResponsibilityEntity(
                text="Ministerio",
                entity_type=EntityType.INSTITUTION,
                confidence=0.6,
                start_char=0,
                end_char=10,
            ),
            ResponsibilityEntity(
                text="Ministerio de Educación",
                entity_type=EntityType.GOVERNMENT,
                confidence=0.9,
                start_char=0,
                end_char=22,
            ),
        ]

        merged = detector._merge_overlapping(entities)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "Ministerio de Educación")
        self.assertEqual(merged[0].entity_type, EntityType.GOVERNMENT)
        self.assertEqual(merged[0].confidence, 0.9)

    def test_degraded_mode(self):
        """Test detection in degraded mode."""
        mock_loader = MagicMock()
        mock_loader.load_model.return_value = None

        detector = ResponsibilityDetector(mock_loader)
        self.assertTrue(detector.is_degraded)

        text = "El Ministerio de Educación implementará la política."
        result = detector.detect_entities(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Ministerio de Educación")
        self.assertEqual(result[0].entity_type, EntityType.GOVERNMENT)

    def test_get_status(self):
        """Test getting detector status."""
        detector = ResponsibilityDetector(self.mock_model_loader)
        status = detector.get_status()

        self.assertFalse(status["is_degraded"])
        self.assertEqual(status["capabilities"], "NER+pattern")
        self.assertGreater(status["patterns"]["government"], 0)
        self.assertGreater(status["patterns"]["position"], 0)
        self.assertGreater(status["patterns"]["institutional"], 0)


if __name__ == "__main__":
    unittest.main()
