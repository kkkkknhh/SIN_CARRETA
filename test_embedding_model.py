"""
Test suite for the embedding model.
"""

from __future__ import annotations

import json
import pickle
import threading
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, List
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from embedding_model import (
    CalibrationCorpusStats,
    EmbeddingConfig,
    SotaEmbedding,
    _reset_embedding_singleton_for_testing,
    get_default_embedding,
)


class _FakeSentenceTransformer:
    """Deterministic stub mimicking SentenceTransformer for tests."""

    def __init__(self, dimension: int = 2) -> None:
        self._dimension = dimension
        self.encode_invocations = 0
        self.request_history: List[List[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)

        self.request_history.append(batch)
        self.encode_invocations += 1

        # Produce repeatable orthogonal embeddings with shape (n, dimension)
        base = np.eye(self._dimension, dtype=np.float32)
        repeats = int(np.ceil(len(batch) / self._dimension))
        tiled = np.tile(base, (repeats, 1))
        return tiled[: len(batch)]


class _FakeIsotonicRegression:
    """Simple stub returning a predictable calibration curve."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature parity
        self.fit_args = None

    def fit_transform(
        self, scores: Iterable[float], labels: Iterable[int]
    ) -> np.ndarray:
        scores = list(scores)
        self.fit_args = (scores, list(labels))
        # Return a smoothly increasing sequence with deterministic spread
        return np.linspace(0.2, 0.9, num=len(scores))


class TestSotaEmbedding(unittest.TestCase):
    """Exercise the modern embedding backend with deterministic stubs."""

    def setUp(self) -> None:  # noqa: D401 - unittest hook
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.calibration_path = Path(self.temp_dir.name) / "card.json"

        self.guard_patch = patch(
            "embedding_model.NoveltyGuard.check_dependencies", return_value=None
        )
        self.guard_patch.start()
        self.addCleanup(self.guard_patch.stop)

        def _build_fake_model(*args, **kwargs):
            model = _FakeSentenceTransformer()
            return model

        self.model_patch = patch(
            "embedding_model.SentenceTransformer", side_effect=_build_fake_model
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

        self.config = EmbeddingConfig(
            model="test/model",
            precision="fp32",
            batch_size=8,
            normalize_l2=False,
            calibration_card=str(self.calibration_path),
            domain_hint_default="PDM",
            device="cpu",
        )

        self.backend = SotaEmbedding(self.config)

    def test_initialization_creates_calibration_card(self) -> None:
        """The backend seeds a default calibration card on startup."""

        card = self.backend.calibration_card
        self.assertIsNotNone(card)
        self.assertEqual(card.model_name, self.config.model)
        self.assertTrue(Path(self.config.calibration_card).exists())

        with open(self.config.calibration_card, "r", encoding="utf-8") as handle:
            card_payload = json.load(handle)

        self.assertEqual(card_payload["model_name"], self.config.model)
        self.assertIn("conformal_thresholds", card_payload)

    def test_embed_texts_applies_domain_smoothing_and_cache(self) -> None:
        """Domain priors influence embeddings and results are cached."""

        self.backend.calibration_card.domain_priors["PDM"] = 0.5

        first = self.backend.embed_texts(["uno", "dos"], domain_hint="PDM")
        expected = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32)
        np.testing.assert_allclose(first, expected)

        cached = self.backend.embed_texts(["uno", "dos"], domain_hint="PDM")
        np.testing.assert_allclose(cached, first)
        self.assertEqual(self.backend.model.encode_invocations, 1)

    def test_embed_texts_respects_normalization_override(self) -> None:
        """Changing normalization bypasses cache and override caches independently."""

        baseline = self.backend.embed_texts(["uno"], domain_hint="PDM")
        override = self.backend.embed_texts(
            ["uno"], domain_hint="PDM", normalize_embeddings=False
        )

        self.assertEqual(self.backend.model.encode_invocations, 2)
        np.testing.assert_allclose(baseline, np.array([[1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(override, np.array([[1.0, 0.0]], dtype=np.float32))

        cached_override = self.backend.embed_texts(
            ["uno"], domain_hint="PDM", normalize_embeddings=False
        )
        np.testing.assert_allclose(cached_override, override)
        self.assertEqual(self.backend.model.encode_invocations, 2)

    def test_calibrate_updates_card_with_corpus_statistics(self) -> None:
        """Running calibration produces updated priors and thresholds."""

        corpus_stats = CalibrationCorpusStats(
            corpus_size=200,
            embedding_dim=2,
            similarity_mean=0.6,
            similarity_std=0.1,
            confidence_scores=list(np.linspace(0.1, 0.9, num=20)),
            gold_labels=[1] * 10 + [0] * 10,
            domain_distribution={"PDM": 80, "rural": 20},
        )

        with patch(
            "embedding_model.IsotonicRegression", return_value=_FakeIsotonicRegression()
        ):
            card = self.backend.calibrate(corpus_stats)

        self.assertEqual(card.embedding_dim, corpus_stats.embedding_dim)
        self.assertTrue(card.isotonic_calibrator["fitted"])
        self.assertAlmostEqual(card.normalization_params["mean"], 0.55, places=2)
        self.assertIn("alpha_0.1", card.conformal_thresholds)
        self.assertIn("rural", card.domain_priors)
        self.assertAlmostEqual(sum(card.domain_priors.values()), 1.0, places=2)

    def test_get_default_embedding_uses_configuration_factory(self) -> None:
        """The factory returns a SotaEmbedding wired with the supplied config."""

        _reset_embedding_singleton_for_testing()
        self.addCleanup(_reset_embedding_singleton_for_testing)

        config = self.config.model_copy(
            update={
                "calibration_card": str(Path(self.temp_dir.name) / "default_card.json")
            }
        )

        with patch("embedding_model.load_embedding_config", return_value=config):
            backend = get_default_embedding()

        self.assertIsInstance(backend, SotaEmbedding)
        self.assertEqual(backend.config.model, config.model)
        self.assertTrue(Path(config.calibration_card).exists())

    def test_backend_picklable_roundtrip(self) -> None:
        """SotaEmbedding instances can be pickled/unpickled safely."""

        self.backend.calibration_card.domain_priors["PDM"] = 0.4
        original = self.backend.embed_texts(["uno", "dos"], domain_hint="PDM")
        encode_calls = self.backend.model.encode_invocations

        payload = pickle.dumps(self.backend)
        restored: SotaEmbedding = pickle.loads(payload)

        # Cached result survives roundtrip and does not trigger extra encodes
        cached = restored.embed_texts(["uno", "dos"], domain_hint="PDM")

        np.testing.assert_allclose(cached, original)
        self.assertEqual(restored.model.encode_invocations, encode_calls)

    def test_encode_matches_sentence_transformer_contract(self) -> None:
        """The thin adapter mirrors the high-level SentenceTransformer API."""

        vector = self.backend.encode("single")
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (2,))

        batch = self.backend.encode(["uno", "dos"])
        self.assertEqual(batch.shape, (2, 2))

        tensor_batch = self.backend.encode(
            ["tres", "cuatro"], convert_to_numpy=False
        )
        self.assertIsInstance(tensor_batch, torch.Tensor)
        self.assertEqual(tuple(tensor_batch.shape), (2, 2))

        vector_tensor = self.backend.encode("cinco", convert_to_numpy=False)
        self.assertIsInstance(vector_tensor, torch.Tensor)
        self.assertEqual(tuple(vector_tensor.shape), (2,))

    def test_get_model_info_exposes_metadata(self) -> None:
        """The metadata helper surfaces essential fields for CLI/documentation."""

        info = self.backend.get_model_info()

        self.assertEqual(info["model_name"], self.config.model)
        self.assertEqual(info["embedding_dimension"], 2)
        self.assertIn("device", info)
        self.assertFalse(info["is_fallback"])
        self.assertTrue(info["calibrated"])

    def tearDown(self) -> None:  # noqa: D401 - unittest hook
        # Ensure patches from setUp are correctly removed even if assertions fail
        for patcher in [self.model_patch, self.guard_patch]:
            try:
                patcher.stop()
            except RuntimeError:
                # Already stopped by addCleanup
                pass


class TestEmbeddingModel(unittest.TestCase):
    """Tests for the EmbeddingModel class."""

    def test_embed_empty_list(self):
        """Test that embedding an empty list returns an empty array."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            model = EmbeddingModel()
            result = model.embed([])
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[0], 0)
            # Verify model was not called with empty input
            mock_model.encode.assert_not_called()

    def test_embed_single_text(self):
        """Test embedding a single text string."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_model
            model = EmbeddingModel()
            result = model.embed(["test text"])
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (1, 3))
            mock_model.encode.assert_called_once()

    def test_embed_multiple_texts(self):
        """Test embedding multiple text strings."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_st.return_value = mock_model
            model = EmbeddingModel()
            # Override batch size for testing
            model.batch_size = 10
            result = model.embed(["text1", "text2"])
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (2, 2))
            mock_model.encode.assert_called_once_with(["text1", "text2"], convert_to_numpy=True)

    def test_batch_processing(self):
        """Test that texts are processed in batches."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            # Return different arrays for different batches
            mock_model.encode.side_effect = [
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([[0.5, 0.6]])
            ]
            mock_st.return_value = mock_model
            model = EmbeddingModel()
            # Set small batch size to force multiple batches
            model.batch_size = 2
            result = model.embed(["text1", "text2", "text3"])
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (3, 2))
            # Check that encode was called twice with correct batches
            self.assertEqual(mock_model.encode.call_count, 2)

    def test_fallback_mechanism(self):
        """Test model fallback when primary model fails."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            # Make first call fail, second succeed
            mock_st.side_effect = [RuntimeError("Model not found"), MagicMock()]
            model = EmbeddingModel()
            # Should have tried to load fallback model
            self.assertEqual(mock_st.call_count, 2)
            self.assertEqual(model.model_name, EmbeddingModel.FALLBACK_MODEL)

    def test_complete_failure(self):
        """Test error handling when both models fail."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            # Make both primary and fallback model loading fail
            mock_st.side_effect = [
                RuntimeError("Primary model error"),
                RuntimeError("Fallback model error"),
            ]
            with self.assertRaises(RuntimeError):
                model = EmbeddingModel()

    def test_embedding_failure(self):
        """Test error handling when embedding fails."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("Encoding failed")
            mock_st.return_value = mock_model
            model = EmbeddingModel()
            with self.assertRaises(RuntimeError):
                model.embed(["test"])

    def test_get_model_info(self):
        """Test getting model information."""
        with patch("embedding_model.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model
            model = EmbeddingModel()
            info = model.get_model_info()
            self.assertEqual(info["model_name"], EmbeddingModel.PRIMARY_MODEL)
            self.assertEqual(info["embedding_dimension"], 768)
            self.assertEqual(info["batch_size"], EmbeddingModel.BATCH_SIZES[EmbeddingModel.PRIMARY_MODEL])
            self.assertFalse(info["is_fallback"])

    def test_factory_function(self):
        """Test the factory function for creating models."""
        with patch("embedding_model.EmbeddingModel") as mock_embedding_model:
            create_embedding_model()
            mock_embedding_model.assert_called_once()

    def test_factory_function_error(self):
        """Test error handling in factory function."""
        with patch("embedding_model.EmbeddingModel", side_effect=RuntimeError("Initialization failed")):
            with self.assertRaises(RuntimeError):
                create_embedding_model()


if __name__ == "__main__":
    unittest.main()
