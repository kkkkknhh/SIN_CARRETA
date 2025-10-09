# -*- coding: utf-8 -*-
"""Pruebas de compatibilidad para el adapter del decálogo."""

from __future__ import annotations

from pdm_contra.bridges.decalogo_loader_adapter import load_decalogos


def test_loader_returns_bundle() -> None:
    bundle = load_decalogos(
        [
            "decalogo-full.latest.clean.json",
            "decalogo-industrial.latest.clean.json",
            "dnp-standards.latest.clean.json",
        ],
        crosswalk_path="crosswalk.latest.json",
    )
    assert bundle["version"] == "1.0.0"
    assert set(bundle["domains"]) == {"PDM", "Industrial", "DNP"}
    assert "clusters" in bundle and bundle["clusters"]
