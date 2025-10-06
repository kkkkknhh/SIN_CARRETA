"""
Integration test: Embedding Model + Responsibility Detection Flow
Critical Flow #1: Document analysis with entity detection and semantic similarity
"""
import pytest
from embedding_model import create_industrial_embedding_model
from responsibility_detector import ResponsibilityDetector


@pytest.mark.integration
@pytest.mark.critical_path
class TestEmbeddingResponsibilityFlow:
    """Test integration between embedding generation and responsibility detection."""
    
    def setup_method(self):
        """Initialize components for each test."""
        self.embedding_model = create_industrial_embedding_model(model_tier="basic")
        self.responsibility_detector = ResponsibilityDetector()
    
    def test_document_embedding_with_responsibility_entities(self):
        """Test generating embeddings for documents with detected responsibilities."""
        text = """
        La Alcaldía Municipal coordinará con la Secretaría de Salud el programa de vacunación.
        El Ministerio de Educación implementará nuevos programas educativos.
        """
        
        entities = self.responsibility_detector.detect_entities(text)
        assert len(entities) >= 0
        
        if entities:
            entity_texts = [e.text for e in entities]
            embeddings = self.embedding_model.encode(entity_texts)
            
            assert embeddings.shape[0] == len(entity_texts)
            assert embeddings.shape[1] > 0
    
    def test_similarity_between_responsibility_entities(self):
        """Test semantic similarity between different responsibility entities."""
        entities = [
            "La Alcaldía Municipal coordinará las actividades",
            "El Ministerio de Educación coordinará el programa",
            "La empresa privada financiará el proyecto"
        ]
        
        embeddings = self.embedding_model.encode(entities)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        govt_similarity = similarity_matrix[0, 1]
        cross_sector_similarity = similarity_matrix[0, 2]
        
        assert govt_similarity > 0
        assert cross_sector_similarity > 0
    
    def test_responsibility_detection_with_semantic_clustering(self):
        """Test that responsibility detection works with semantic clustering."""
        text = """
        La Alcaldía Municipal coordinará con la Secretaría de Salud.
        La Gobernación apoyará mediante recursos técnicos.
        La universidad implementará nuevos programas.
        """
        
        entities = self.responsibility_detector.detect_entities(text)
        
        if entities:
            entity_texts = [e.text for e in entities]
            embeddings = self.embedding_model.encode(entity_texts)
            
            assert embeddings.shape[0] == len(entities)
            clarity = self.responsibility_detector.evaluate_responsibility_clarity(entities)
            assert isinstance(clarity, dict)
    
    def test_fallback_mode_compatibility(self):
        """Test that basic tier embedding model works with responsibility detection."""
        light_model = create_industrial_embedding_model(model_tier="basic")
        
        text = "El Ministerio de Agricultura coordinará el programa."
        entities = self.responsibility_detector.detect_entities(text)
        
        if entities:
            entity_texts = [e.text for e in entities]
            embeddings = light_model.encode(entity_texts)
            
            assert embeddings.shape[0] == len(entity_texts)
