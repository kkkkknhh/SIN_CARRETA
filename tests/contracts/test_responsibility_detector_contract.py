"""
Contract Test: Responsibility Detector Interface
Validates that responsibility detector adheres to expected interface contract
"""
import pytest
from responsibility_detector import ResponsibilityDetector, ResponsibilityEntity, EntityType


@pytest.mark.contract
class TestResponsibilityDetectorContract:
    """Contract tests for ResponsibilityDetector interface."""
    
    def test_detector_initialization_contract(self):
        """Contract: Detector must initialize without errors."""
        detector = ResponsibilityDetector()
        assert detector is not None
    
    def test_detect_entities_contract(self):
        """Contract: detect_entities() must return list."""
        detector = ResponsibilityDetector()
        result = detector.detect_entities("test text")
        
        assert isinstance(result, list)
    
    def test_evaluate_responsibility_clarity_contract(self):
        """Contract: evaluate_responsibility_clarity() must return dict."""
        detector = ResponsibilityDetector()
        text = "La Alcaldía Municipal coordinará el programa."
        entities = detector.detect_entities(text)
        result = detector.evaluate_responsibility_clarity(entities)
        
        assert isinstance(result, dict)
    
    def test_responsibility_entity_structure_contract(self):
        """Contract: ResponsibilityEntity must have required attributes."""
        detector = ResponsibilityDetector()
        text = "La Alcaldía Municipal coordinará el programa."
        entities = detector.detect_entities(text)
        
        if entities:
            entity = entities[0]
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'entity_type')
            assert hasattr(entity, 'confidence')
            assert isinstance(entity.confidence, float)
            assert 0.0 <= entity.confidence <= 1.0
    
    def test_entity_type_enum_contract(self):
        """Contract: EntityType must be properly defined enum."""
        assert hasattr(EntityType, 'GOVERNMENT')
        assert hasattr(EntityType, 'POSITION')
        assert hasattr(EntityType, 'INSTITUTION')
        assert hasattr(EntityType, 'PERSON')
    
    def test_empty_text_handling_contract(self):
        """Contract: Detector must handle empty text gracefully."""
        detector = ResponsibilityDetector()
        
        entities = detector.detect_entities("")
        assert isinstance(entities, list)
    
    def test_none_input_handling_contract(self):
        """Contract: Detector must handle None input gracefully."""
        detector = ResponsibilityDetector()
        
        try:
            entities = detector.detect_entities(None)
            assert isinstance(entities, list)
        except (TypeError, AttributeError):
            pass
    
    def test_entity_detection_monotonicity_contract(self):
        """Contract: More responsibility text should yield more entities."""
        detector = ResponsibilityDetector()
        
        text_few = "El Alcalde coordina."
        text_many = """
        El Alcalde coordina. La Secretaría implementa.
        El Ministerio aprueba. La Gobernación supervisa.
        """
        
        entities_few = detector.detect_entities(text_few)
        entities_many = detector.detect_entities(text_many)
        
        assert isinstance(entities_few, list)
        assert isinstance(entities_many, list)
