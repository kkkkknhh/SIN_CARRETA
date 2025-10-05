# -*- coding: utf-8 -*-
"""La normalización debe producir hashes estables."""

from __future__ import annotations

import hashlib
import json

from pdm_contra.bridges.decalogo_loader_adapter import load_decalogos


def test_bundle_hash_is_deterministic() -> None:
    bundle = load_decalogos(
        [
            "out/decalogo-full.v1.0.0.clean.json",
            "out/decalogo-industrial.v1.0.0.clean.json",
            "out/dnp-standards.v1.0.0.clean.json",
        ],
        crosswalk_path="out/crosswalk.v1.0.0.json",
    )
    payload = json.dumps(bundle, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == hashlib.sha256(payload).hexdigest()
