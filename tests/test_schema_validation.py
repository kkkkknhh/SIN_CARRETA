# -*- coding: utf-8 -*-
"""Validaciones de esquema para los decálogos limpios."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from jsonschema.validators import validator_for

SCHEMA_DIR = Path("schemas")
OUT_DIR = Path("out")


def _load_schema(name: str) -> dict:
    with (SCHEMA_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate(payload: dict, schema_name: str) -> None:
    schema = _load_schema(schema_name)
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    resolver = jsonschema.RefResolver(
        base_uri=f"file://{SCHEMA_DIR.resolve()}/", referrer=schema
    )
    validator = validator_cls(schema, resolver=resolver)
    validator.validate(payload)


def test_full_schema() -> None:
    payload = json.loads(
        (OUT_DIR / "decalogo-full.v1.0.0.clean.json").read_text(encoding="utf-8")
    )
    _validate(payload, "decalogo-full.schema.json")


def test_industrial_schema() -> None:
    payload = json.loads(
        (OUT_DIR / "decalogo-industrial.v1.0.0.clean.json").read_text(encoding="utf-8")
    )
    _validate(payload, "decalogo-industrial.schema.json")


def test_dnp_schema() -> None:
    payload = json.loads(
        (OUT_DIR / "dnp-standards.v1.0.0.clean.json").read_text(encoding="utf-8")
    )
    _validate(payload, "dnp-standards.schema.json")
