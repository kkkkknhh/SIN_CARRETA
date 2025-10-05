# -*- coding: utf-8 -*-
"""Comprueba que las aplicaciones cargan el bundle por defecto."""

from __future__ import annotations

from pathlib import Path


def test_autoload_bundle() -> None:
    principal = Path("Decatalogo_principal.py").read_text(encoding="utf-8")
    evaluador = Path("Decatalogo_evaluador.py").read_text(encoding="utf-8")
    assert (
            "from pdm_contra.bridges.decatalogo_provider import provide_decalogos"
            in principal
    )
    assert "BUNDLE = provide_decalogos()" in principal
    assert (
            "from pdm_contra.bridges.decatalogo_provider import provide_decalogos"
            in evaluador
    )
    assert "BUNDLE = provide_decalogos()" in evaluador
