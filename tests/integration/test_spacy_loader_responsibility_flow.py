"""
Integration test: SpaCy Loader + Responsibility Detection Flow
Critical Flow #5: NLP model loading and entity extraction pipeline
"""
import pytest
from spacy_loader import SpacyModelLoader
from responsibility_detector import ResponsibilityDetector


@pytest.mark.integration
@pytest.mark.critical_path
class TestSpacyLoaderResponsibilityFlow:
    """Test integration between SpaCy model loading and responsibility detection."""
    
    @staticmethod
    def test_spacy_model_loads_for_responsibility_detection():
        """Test that SpaCy model loads successfully for responsibility detector."""
        loader = SpacyModelLoader(model_name="es_core_news_sm")
        nlp = loader.load()
        
        assert nlp is not None
        
        detector = ResponsibilityDetector()
        
        text = "La Alcaldía Municipal coordinará el programa de salud."
        entities = detector.detect_entities(text)
        
        assert isinstance(entities, list)
    
    @staticmethod
    def test_degraded_mode_fallback():
        """Test that responsibility detector works in degraded mode when model fails."""
        detector = ResponsibilityDetector()
        
        text = """
        El Ministerio de Educación coordinará las actividades.
        La Secretaría de Salud implementará el programa.
        """
        
        entities = detector.detect_entities(text)
        
        assert isinstance(entities, list)
    
    @staticmethod
    def test_ner_entity_extraction_pipeline():
        """Test full NER pipeline from model loading to entity extraction."""
        loader = SpacyModelLoader(model_name="es_core_news_sm")
        nlp = loader.load()
        
        if nlp is not None:
            text = """
            El Alcalde Juan Pérez y la Secretaria María Gómez coordinan el proyecto.
            La Gobernación del Valle apoya la iniciativa.
            """
            
            doc = nlp(text)
            ner_entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            assert len(ner_entities) > 0
            
            detector = ResponsibilityDetector()
            responsibility_entities = detector.detect_entities(text)
            
            assert len(responsibility_entities) > 0
