"""
Contract Test: Embedding Model Interface
Validates that embedding model adheres to expected interface contract
"""
import pytest
import numpy as np
from embedding_model import IndustrialEmbeddingModel, create_industrial_embedding_model


@pytest.mark.contract
class TestEmbeddingModelContract:
    """Contract tests for IndustrialEmbeddingModel interface."""
    
    def test_model_initialization_contract(self):
        """Contract: Model must initialize without errors."""
        model = create_industrial_embedding_model(model_tier="basic")
        assert model is not None
        assert isinstance(model, IndustrialEmbeddingModel)
    
    def test_encode_single_text_contract(self):
        """Contract: encode() must accept single text and return numpy array."""
        model = create_industrial_embedding_model(model_tier="basic")
        result = model.encode(["test text"])
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1
    
    def test_encode_batch_contract(self):
        """Contract: encode() must handle batch inputs."""
        model = create_industrial_embedding_model(model_tier="basic")
        texts = ["text 1", "text 2", "text 3"]
        result = model.encode(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(texts)
    
    def test_embedding_dimension_consistency_contract(self):
        """Contract: All embeddings must have same dimension."""
        model = create_industrial_embedding_model(model_tier="basic")
        texts = ["short", "medium length text", "a very long text " * 10]
        embeddings = model.encode(texts)
        
        for i, item in enumerate(embeddings):
            for j in range(i + 1, len(embeddings)):
                assert len(item) == len(embeddings[j])
    
    def test_deterministic_encoding_contract(self):
        """Contract: Same text must produce same embedding."""
        model = create_industrial_embedding_model(model_tier="basic")
        text = "deterministic test"
        
        embedding1 = model.encode([text])[0]
        embedding2 = model.encode([text])[0]
        
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
    
    def test_empty_input_handling_contract(self):
        """Contract: Model must handle empty input gracefully."""
        model = create_industrial_embedding_model(model_tier="basic")
        result = model.encode([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0
    
    def test_diagnostics_contract(self):
        """Contract: get_comprehensive_diagnostics() must return dict."""
        model = create_industrial_embedding_model(model_tier="basic")
        diagnostics = model.get_comprehensive_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert "model_info" in diagnostics
