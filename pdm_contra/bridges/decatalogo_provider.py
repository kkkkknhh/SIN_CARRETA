# -*- coding: utf-8 -*-
"""Proveedor centralizado de decálogos canónicos."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from .decalogo_loader_adapter import load_decalogos

_BRIDGES_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _BRIDGES_DIR.parent / "config" / "decalogo.yaml"


class DecalogoProviderError(RuntimeError):
    """Error de configuración del proveedor de decálogos."""


def _read_config(path: Path | None = None) -> Dict[str, object]:
    config_path = Path(path) if path is not None else CONFIG_PATH
    if not config_path.exists():
        raise DecalogoProviderError(
            f"No existe configuración de decálogo en {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def provide_decalogos(config_path: Path | None = None) -> Dict[str, object]:
    """Carga el bundle canónico de decálogos usando la configuración empaquetada."""

    config = _read_config(config_path)
    config_file = CONFIG_PATH if config_path is None else Path(
        config_path).resolve()
    config_dir = config_file.parent

    if not config.get("autoload", False):
        raise DecalogoProviderError(
            "El autoload está deshabilitado en la configuración"
        )

    paths_config = config.get("paths", {})
    try:
        full_path = _resolve_path(config_dir, paths_config["full"])
        industrial_path = _resolve_path(config_dir, paths_config["industrial"])
        dnp_path = _resolve_path(config_dir, paths_config["dnp"])
    except KeyError as exc:
        raise DecalogoProviderError(
            f"Falta la ruta obligatoria en configuración: {exc}"
        ) from exc

    crosswalk_raw = config.get("crosswalk")
    if crosswalk_raw is None:
        crosswalk_raw = str(full_path.parent / "crosswalk.latest.json")
    crosswalk_path = _resolve_path(config_dir, crosswalk_raw)

    for path in (full_path, industrial_path, dnp_path, crosswalk_path):
        if not path.exists():
            raise DecalogoProviderError(
                f"No se encuentra el archivo requerido: {path}")

    bundle = load_decalogos(
        [str(full_path), str(industrial_path), str(dnp_path)],
        crosswalk_path=str(crosswalk_path),
    )
    return bundle


__all__ = ["provide_decalogos", "DecalogoProviderError"]
