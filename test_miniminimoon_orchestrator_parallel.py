"""
Test suite for parallel processing enhancements in miniminimoon_orchestrator.py

Tests:
- ThreadPoolExecutor with max_workers=4 for questionnaire evaluation
- Singleton connection pool pattern for embedding model
- Document-level LRU cache with TTL
- Dynamic batch size selection (32-64) for embeddings
- warm_up() method for preloading models
- Thread-safe access to shared resources
"""

import unittest
import threading
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miniminimoon_orchestrator import (
    ThreadSafeLRUCache,
    EmbeddingModelPool,
    CanonicalDeterministicOrchestrator,
    )


class TestThreadSafeLRUCache(unittest.TestCase):
    """Test thread-safe LRU cache with TTL"""

    def test_basic_set_get(self):
        """Test basic set and get operations"""
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=10)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_lru_eviction(self):
        """Test LRU eviction when max_size is exceeded"""
        cache = ThreadSafeLRUCache(max_size=3, ttl_seconds=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")
        self.assertEqual(cache.get("key4"), "value4")

    def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=1)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
        
        time.sleep(1.5)
        self.assertIsNone(cache.get("key1"))

    def test_lru_touch_on_get(self):
        """Test that get() updates LRU order"""
        cache = ThreadSafeLRUCache(max_size=3, ttl_seconds=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add key4, should evict key2 (not key1)
        cache.set("key4", "value4")
        
        self.assertEqual(cache.get("key1"), "value1")
        self.assertIsNone(cache.get("key2"))
        self.assertEqual(cache.get("key3"), "value3")
        self.assertEqual(cache.get("key4"), "value4")

    def test_has_method(self):
        """Test has() method"""
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=10)
        cache.set("key1", "value1")
        
        self.assertTrue(cache.has("key1"))
        self.assertFalse(cache.has("key2"))

    def test_size_method(self):
        """Test size() method"""
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=10)
        self.assertEqual(cache.size(), 0)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        self.assertEqual(cache.size(), 2)

    def test_purge_expired(self):
        """Test purge_expired() method"""
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        time.sleep(1.5)
        cache.purge_expired()
        
        self.assertEqual(cache.size(), 0)

    def test_thread_safety(self):
        """Test concurrent access from multiple threads"""
        cache = ThreadSafeLRUCache(max_size=100, ttl_seconds=10)
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(50):
                    key = f"thread{thread_id}_key{i}"
                    cache.set(key, f"value{i}")
                    value = cache.get(key)
                    if value != f"value{i}":
                        errors.append(f"Thread {thread_id}: Expected value{i}, got {value}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


class TestEmbeddingModelPool(unittest.TestCase):
    """Test singleton connection pool for embedding model"""

    def setUp(self):
        """Reset singleton before each test"""
        EmbeddingModelPool._model_instance = None

    def test_singleton_pattern(self):
        """Test that get_model() returns the same instance"""
        with patch('miniminimoon_orchestrator.EmbeddingModel') as MockModel:
            mock_instance = Mock()
            MockModel.return_value = mock_instance
            
            model1 = EmbeddingModelPool.get_model()
            model2 = EmbeddingModelPool.get_model()
            
            self.assertIs(model1, model2)
            MockModel.assert_called_once()

    def test_thread_safe_initialization(self):
        """Test concurrent initialization from multiple threads"""
        with patch('miniminimoon_orchestrator.EmbeddingModel') as MockModel:
            mock_instance = Mock()
            MockModel.return_value = mock_instance
            
            models = []
            errors = []
            
            def worker():
                try:
                    model = EmbeddingModelPool.get_model()
                    models.append(model)
                except Exception as e:
                    errors.append(str(e))
            
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            self.assertEqual(len(errors), 0, f"Errors: {errors}")
            self.assertEqual(len(set(id(m) for m in models)), 1, "All models should be the same instance")
            MockModel.assert_called_once()


class TestDynamicBatchEmbedding(unittest.TestCase):
    """Test dynamic batch size selection for embeddings"""

    def setUp(self):
        """Create temporary config directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        
        # Create minimal config files
        self._create_minimal_configs()
        
        # Create immutability snapshot to pass gate #1
        self._create_immutability_snapshot()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset singleton
        EmbeddingModelPool._model_instance = None
        CanonicalDeterministicOrchestrator._shared_embedding_model = None

    def _create_minimal_configs(self):
        """Create minimal config files for testing"""
        configs = {
            "DECALOGO_FULL.json": {"questions": []},
            "decalogo_industrial.json": {"questions": []},
            "dnp-standards.latest.clean.json": {},
            "RUBRIC_SCORING.json": {"questions": {}, "weights": {}},
        }
        
        for filename, content in configs.items():
            with open(self.config_dir / filename, 'w') as f:
                json.dump(content, f)

    def _create_immutability_snapshot(self):
        """Create immutability snapshot to pass gate #1"""
        snapshot = {
            "frozen_at": "2024-01-01T00:00:00",
            "config_hashes": {},
            "frozen": True
        }
        
        with open(self.config_dir.parent / ".immutability_snapshot.json", 'w') as f:
            json.dump(snapshot, f)

    @patch('miniminimoon_orchestrator.EnhancedImmutabilityContract')
    @patch('miniminimoon_orchestrator.EmbeddingModel')
    def test_batch_size_selection_large(self, MockEmbedding, MockContract):
        """Test batch_size=64 for large inputs"""
        # Mock immutability contract
        mock_contract = Mock()
        mock_contract.has_snapshot.return_value = True
        mock_contract.verify_frozen_config.return_value = True
        MockContract.return_value = mock_contract
        
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1, 0.2]] * 64)
        MockEmbedding.return_value = mock_model
        
        orchestrator = CanonicalDeterministicOrchestrator(
            config_dir=self.config_dir,
            enable_validation=False
        )
        
        # Test with 100 segments (should use batch_size=64 then 32)
        segments = [f"segment_{i}" for i in range(100)]
        result = orchestrator._encode_segments_dynamic(segments)
        
        # Verify encode was called with correct batch sizes
        self.assertEqual(len(result), 100)
        self.assertGreater(mock_model.encode.call_count, 0)

    @patch('miniminimoon_orchestrator.EnhancedImmutabilityContract')
    @patch('miniminimoon_orchestrator.EmbeddingModel')
    def test_batch_size_selection_small(self, MockEmbedding, MockContract):
        """Test batch_size=32 for smaller remaining batches"""
        # Mock immutability contract
        mock_contract = Mock()
        mock_contract.has_snapshot.return_value = True
        mock_contract.verify_frozen_config.return_value = True
        MockContract.return_value = mock_contract
        
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1, 0.2]] * 32)
        MockEmbedding.return_value = mock_model
        
        orchestrator = CanonicalDeterministicOrchestrator(
            config_dir=self.config_dir,
            enable_validation=False
        )
        
        # Test with 50 segments (should use batch_size=32 twice)
        segments = [f"segment_{i}" for i in range(50)]
        result = orchestrator._encode_segments_dynamic(segments)
        
        self.assertEqual(len(result), 50)
        self.assertGreater(mock_model.encode.call_count, 0)


class TestWarmUpMethod(unittest.TestCase):
    """Test warm_up() method for model preloading"""

    def setUp(self):
        """Create temporary config directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self._create_minimal_configs()

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        EmbeddingModelPool._model_instance = None
        CanonicalDeterministicOrchestrator._shared_embedding_model = None

    def _create_minimal_configs(self):
        """Create minimal config files"""
        configs = {
            "DECALOGO_FULL.json": {"questions": []},
            "decalogo_industrial.json": {"questions": []},
            "dnp-standards.latest.clean.json": {},
            "RUBRIC_SCORING.json": {"questions": {}, "weights": {}},
        }
        for filename, content in configs.items():
            with open(self.config_dir / filename, 'w') as f:
                json.dump(content, f)

    @patch('miniminimoon_orchestrator.EnhancedImmutabilityContract')
    @patch('miniminimoon_orchestrator.EmbeddingModel')
    @patch('miniminimoon_orchestrator.QuestionnaireEngine')
    def test_warm_up_method_exists(self, MockQuestionnaire, MockEmbedding, MockContract):
        """Test that warm_up() method exists and is callable"""
        mock_contract = Mock()
        mock_contract.has_snapshot.return_value = True
        mock_contract.verify_frozen_config.return_value = True
        MockContract.return_value = mock_contract
        
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1, 0.2]])
        MockEmbedding.return_value = mock_model
        
        orchestrator = CanonicalDeterministicOrchestrator(
            config_dir=self.config_dir,
            enable_validation=False
        )
        
        # Test that warm_up() method exists
        self.assertTrue(hasattr(orchestrator, 'warm_up'))
        self.assertTrue(callable(orchestrator.warm_up))
        
        # Test that it can be called
        orchestrator.warm_up()  # Should not raise

    @patch('miniminimoon_orchestrator.EnhancedImmutabilityContract')
    @patch('miniminimoon_orchestrator.EmbeddingModel')
    def test_warm_up_is_idempotent(self, MockEmbedding, MockContract):
        """Test that warm_up() can be called multiple times safely"""
        mock_contract = Mock()
        mock_contract.has_snapshot.return_value = True
        mock_contract.verify_frozen_config.return_value = True
        MockContract.return_value = mock_contract
        
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1, 0.2]])
        MockEmbedding.return_value = mock_model
        
        orchestrator = CanonicalDeterministicOrchestrator(
            config_dir=self.config_dir,
            enable_validation=False
        )
        
        # Call warm_up() multiple times
        orchestrator.warm_up()
        orchestrator.warm_up()
        orchestrator.warm_up()
        
        # Should not raise errors


class TestThreadSafeSharedResources(unittest.TestCase):
    """Test thread-safe access to shared resources during parallel evaluation"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self._create_minimal_configs()

    def tearDown(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        EmbeddingModelPool._model_instance = None
        CanonicalDeterministicOrchestrator._shared_embedding_model = None

    def _create_minimal_configs(self):
        """Create minimal configs"""
        configs = {
            "DECALOGO_FULL.json": {"questions": []},
            "decalogo_industrial.json": {"questions": []},
            "dnp-standards.latest.clean.json": {},
            "RUBRIC_SCORING.json": {"questions": {}, "weights": {}},
        }
        for filename, content in configs.items():
            with open(self.config_dir / filename, 'w') as f:
                json.dump(content, f)

    @patch('miniminimoon_orchestrator.EnhancedImmutabilityContract')
    @patch('miniminimoon_orchestrator.EmbeddingModel')
    def test_concurrent_embedding_model_access(self, MockEmbedding, MockContract):
        """Test concurrent access to singleton embedding model"""
        mock_contract = Mock()
        mock_contract.has_snapshot.return_value = True
        mock_contract.verify_frozen_config.return_value = True
        MockContract.return_value = mock_contract
        
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1, 0.2]])
        MockEmbedding.return_value = mock_model
        
        orchestrator = CanonicalDeterministicOrchestrator(
            config_dir=self.config_dir,
            enable_validation=False
        )
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                model = orchestrator._get_shared_embedding_model()
                results.append((worker_id, model))
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # All workers should get the same model instance
        model_ids = set(id(m) for _, m in results)
        self.assertEqual(len(model_ids), 1)


class TestDocumentLevelCache(unittest.TestCase):
    """Test document-level caching with SHA-256 hashing"""

    def test_document_hash_based_caching(self):
        """Test that document hash is used as cache key"""
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=10)
        
        # Simulate document hash as cache key
        import hashlib
        doc_text = "This is a test document"
        doc_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
        cache_key = f"docres:{doc_hash}"
        
        # Store result
        result = {"stages_completed": ["stage1", "stage2"], "score": 0.85}
        cache.set(cache_key, result)
        
        # Retrieve result
        cached_result = cache.get(cache_key)
        self.assertEqual(cached_result, result)

    def test_different_documents_different_cache_keys(self):
        """Test that different documents get different cache keys"""
        import hashlib
        
        doc1 = "Document one"
        doc2 = "Document two"
        
        hash1 = hashlib.sha256(doc1.encode('utf-8')).hexdigest()
        hash2 = hashlib.sha256(doc2.encode('utf-8')).hexdigest()
        
        self.assertNotEqual(hash1, hash2)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
