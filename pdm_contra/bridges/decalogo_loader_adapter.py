# -*- coding: utf-8 -*-
"""Adaptador para cargar los decálogos limpios y validar sus contratos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from jsonschema import validator_for, RefResolver

SCHEMA_DIR = Path("schemas")


@dataclass
class CanonicalDecalogoBundle:
    version: str
    domains: List[str]
    clusters: List[Dict[str, object]]
    crosswalk: Dict[str, List[Dict[str, str]]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "domains": self.domains,
            "clusters": self.clusters,
            "crosswalk": self.crosswalk,
        }


def _load_schema(name: str) -> Dict[str, object]:
    schema_path = SCHEMA_DIR / name
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_payload(schema_name: str, payload: Dict[str, object]) -> None:
    schema = _load_schema(schema_name)
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    resolver = RefResolver(
        base_uri=f"file://{SCHEMA_DIR.resolve()}/", referrer=schema
    )
    validator = validator_cls(schema, resolver=resolver)
    validator.validate(payload)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_decalogos(
        paths: List[str], crosswalk_path: Optional[str] = None
) -> Dict[str, object]:
    if len(paths) != 3:
        raise ValueError("Se requieren tres rutas: full, industrial y dnp")
    payloads = []
    domains = []
    version = None
    for path_str, schema_name in zip(
            paths,
            [
                "decalogo-full.schema.json",
                "decalogo-industrial.schema.json",
                "dnp-standards.schema.json",
            ],
    ):
        path = Path(path_str)
        payload = _load_json(path)
        _validate_payload(schema_name, payload)
        payloads.append(payload)
        domains.append(payload.get("domain", ""))
        if version is None:
            version = payload.get("version", "0.0.0")
        elif version != payload.get("version"):
            raise ValueError("Las versiones de los decálogos no coinciden")
    if crosswalk_path:
        crosswalk = _load_json(Path(crosswalk_path))
    else:
        crosswalk = payloads[0].get("crosswalk", {})
    return CanonicalDecalogoBundle(
        version=version or "0.0.0",
        domains=domains,
        clusters=payloads[0]["clusters"],
        crosswalk=crosswalk,
    ).to_dict()
