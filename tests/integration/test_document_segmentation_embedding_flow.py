"""
Integration test: Document Segmentation + Embedding Flow
Critical Flow #4: Large document processing pipeline
"""
import pytest
from document_segmenter import DocumentSegmenter
from embedding_model import create_industrial_embedding_model


@pytest.mark.integration
@pytest.mark.critical_path
class TestDocumentSegmentationEmbeddingFlow:
    """Test integration between document segmentation and embedding generation."""
    
    def setup_method(self):
        """Initialize components."""
        self.segmenter = DocumentSegmenter(max_chunk_size=500)
        self.embedding_model = create_industrial_embedding_model(model_tier="basic")
    
    def test_segment_and_embed_large_document(self):
        """Test segmenting large document and generating embeddings for each segment."""
        large_doc = """
        Capítulo 1: Diagnóstico Municipal
        
        El municipio cuenta con una población de 50,000 habitantes distribuidos en 
        10 veredas y el casco urbano. Los principales desafíos identificados incluyen:
        
        1. Baja cobertura de servicios públicos en zona rural (45% de cobertura)
        2. Infraestructura vial deteriorada (80% de vías sin pavimentar)
        3. Limitado acceso a servicios de salud (1 centro de salud para toda la población)
        
        Capítulo 2: Objetivos Estratégicos
        
        Se plantean tres objetivos estratégicos para el período 2024-2027:
        
        Objetivo 1: Aumentar la cobertura de servicios públicos al 80% en zona rural.
        Objetivo 2: Mejorar el 50% de la infraestructura vial municipal.
        Objetivo 3: Fortalecer la red de atención en salud con 2 nuevos puestos de salud.
        
        Capítulo 3: Plan de Acción
        
        Para cada objetivo se definen actividades específicas, responsables y plazos.
        """
        
        segments = self.segmenter.segment_document(large_doc)
        
        assert len(segments) > 1, "Large document should be segmented"
        
        segment_embeddings = []
        for segment in segments:
            embedding = self.embedding_model.encode([segment])[0]
            segment_embeddings.append(embedding)
        
        assert len(segment_embeddings) == len(segments)
        
        embedding_dim = len(segment_embeddings[0])
        for emb in segment_embeddings:
            assert len(emb) == embedding_dim
    
    def test_semantic_search_across_segments(self):
        """Test semantic search functionality across document segments."""
        document = """
        Sección A: El municipio implementará programas de educación digital.
        
        Sección B: Se fortalecerá la infraestructura de conectividad rural.
        
        Sección C: La Secretaría de Salud expandirá los servicios de telemedicina.
        
        Sección D: Se construirán nuevas vías de acceso a veredas alejadas.
        """
        
        segments = self.segmenter.segment_document(document, delimiter="Sección")
        segment_embeddings = self.embedding_model.encode(segments)
        
        query = "tecnología y conectividad"
        query_embedding = self.embedding_model.encode([query])[0]
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], segment_embeddings)[0]
        
        most_relevant_idx = similarities.argmax()
        
        assert most_relevant_idx >= 0
        assert similarities[most_relevant_idx] > 0
    
    def test_batch_processing_efficiency(self):
        """Test efficient batch processing of multiple document segments."""
        document = "Sección {}: Contenido del municipio.\n\n" * 10
        
        segments = self.segmenter.segment_document(document.format(*range(10)))
        
        import time
        start = time.time()
        batch_embeddings = self.embedding_model.encode(segments)
        batch_time = time.time() - start
        
        assert batch_time < 30.0
        assert len(batch_embeddings) == len(segments)
