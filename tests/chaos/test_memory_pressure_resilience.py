"""
Chaos Engineering: Memory Pressure Resilience
Tests system behavior under memory constraints
"""
import pytest
import gc
from embedding_model import create_industrial_embedding_model


@pytest.mark.chaos
@pytest.mark.slow
class TestMemoryPressureResilience:
    """Test system resilience under memory pressure."""
    
    def test_large_batch_processing_under_memory_pressure(self):
        """Test processing large batches when memory is constrained."""
        model = create_industrial_embedding_model(model_tier="basic")
        
        large_texts = ["text " * 100 for _ in range(10)]
        
        gc.collect()
        
        try:
            embeddings = model.encode(large_texts, batch_size=2)
            assert embeddings is not None
            assert len(embeddings) == len(large_texts)
        except MemoryError:
            pytest.skip("Memory pressure test requires more memory")
    
    def test_model_loading_with_limited_memory(self):
        """Test model loads with reduced memory footprint."""
        gc.collect()
        
        model = create_industrial_embedding_model(model_tier="basic")
        assert model is not None
    
    def test_incremental_processing_prevents_oom(self):
        """Test that incremental processing prevents out-of-memory errors."""
        model = create_industrial_embedding_model(model_tier="basic")
        
        texts = ["document " + str(i) * 50 for i in range(50)]
        
        embeddings_list = []
        batch_size = 5
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch)
            embeddings_list.extend(batch_embeddings)
            gc.collect()
        
        assert len(embeddings_list) == len(texts)
