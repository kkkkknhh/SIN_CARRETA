"""Regression tests for embedding_model import side effects."""

import importlib
import sys
import types


def test_import_does_not_trigger_post_install_setup(monkeypatch):
    """Importing embedding_model should not ejecutar setup ni cargar modelos."""

    # Asegurarse de trabajar con un import limpio
    original_module = sys.modules.pop("embedding_model", None)

    # Simular dependencia pesada que fallaría si se instanciara
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    instantiation_counter = {"count": 0}

    class _SentinelTransformer:
        def __init__(self, *args, **kwargs):
            instantiation_counter["count"] += 1
            raise RuntimeError(
                "SentenceTransformer should not be instantiated on import"
            )

    fake_sentence_transformers.SentenceTransformer = _SentinelTransformer
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", fake_sentence_transformers
    )

    try:
        module = importlib.import_module("embedding_model")
        assert instantiation_counter["count"] == 0
        assert not getattr(module, "_POST_INSTALL_SETUP_DONE", True)
    finally:
        sys.modules.pop("embedding_model", None)
        if original_module is not None:
            sys.modules["embedding_model"] = original_module
