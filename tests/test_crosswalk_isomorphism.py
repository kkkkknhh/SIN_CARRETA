# -*- coding: utf-8 -*-
"""Verifica el isomorfismo débil del crosswalk."""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("out")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_crosswalk_integrity() -> None:
    crosswalk = _load(OUT_DIR / "crosswalk.v1.0.0.json")
    full = _load(OUT_DIR / "decalogo-full.v1.0.0.clean.json")
    industrial = _load(OUT_DIR / "decalogo-industrial.v1.0.0.clean.json")
    dnp = _load(OUT_DIR / "dnp-standards.v1.0.0.clean.json")

    def collect_codes(payload: dict, key: str) -> set:
        codes = set()
        for cluster in payload["clusters"]:
            if key == "clusters":
                codes.add(cluster["cluster_code"])
            elif key == "points":
                codes.add(cluster["points"][0]["point_code"])
            else:
                codes.add(cluster["points"][0]["questions"][0]["q_code"])
        return codes

    datasets = {
        "full": full,
        "industrial": industrial,
        "dnp": dnp,
    }

    for key in ("clusters", "points", "questions"):
        seen = set()
        for entry in crosswalk[key]:
            canonical = entry["canonical_code"]
            assert canonical not in seen
            seen.add(canonical)
            for dataset in ("full", "industrial", "dnp"):
                code = entry[dataset]
                assert code in collect_codes(datasets[dataset], key)
        # Debe existir al menos un elemento
        assert seen
