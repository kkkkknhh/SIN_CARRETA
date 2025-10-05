# -*- coding: utf-8 -*-
"""Verifica la validez del competence_map del DNP."""

from __future__ import annotations

import json
from pathlib import Path


def test_competence_map_entries() -> None:
    payload = json.loads(
        Path("out/dnp-standards.v1.0.0.clean.json").read_text(encoding="utf-8")
    )
    competence = payload["competence_map"]
    assert set(competence.keys()) == {"municipal", "departamental", "nacional"}
    for value in competence.values():
        assert value in {"evidencia_insuficiente"}
