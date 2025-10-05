"""Unit tests for SotaEmbedding dependency handling."""

import importlib.metadata
import os
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from embedding_model import EmbeddingConfig, SotaEmbedding


def test_sota_embedding_instantiates_with_declared_requirements():
    """Ensure SotaEmbedding works with the minimal declared requirements."""

    required_versions = {
        "sentence-transformers": "2.2.0",
        "torch": "1.9.0",
        "numpy": "1.21.0",
        "scikit-learn": "1.0.0",
    }

    def fake_version(package_name: str) -> str:
        if package_name in required_versions:
            return required_versions[package_name]
        raise importlib.metadata.PackageNotFoundError

    with patch("importlib.metadata.version", side_effect=fake_version):
        with patch("embedding_model.torch") as mock_torch:
            with patch(
                "embedding_model.SentenceTransformer"
            ) as mock_sentence_transformer:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.device.side_effect = lambda dev: SimpleNamespace(type=dev)
                mock_torch.inference_mode.return_value = nullcontext()
                mock_torch.autocast.return_value = nullcontext()
                mock_torch.quantization = SimpleNamespace(
                    quantize_dynamic=lambda model, modules, dtype=None: model
                )
                mock_torch.nn = SimpleNamespace(Linear=object)
                mock_torch.qint8 = "qint8"

                model_instance = MagicMock()
                model_instance.encode.return_value = np.ones((1, 3), dtype=np.float32)
                model_instance.get_sentence_embedding_dimension.return_value = 3
                mock_sentence_transformer.return_value = model_instance

                with tempfile.TemporaryDirectory() as tmp_dir:
                    card_path = os.path.join(tmp_dir, "card.json")
                    config = EmbeddingConfig(
                        calibration_card=card_path,
                        precision="fp32",
                    )
                    embedding = SotaEmbedding(config)

                assert isinstance(embedding, SotaEmbedding)
                assert embedding.model is model_instance
                assert embedding.calibration_card is not None
