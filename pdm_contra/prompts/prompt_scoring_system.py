"""Carga la especificación técnica del sistema de scoring para PDM."""

from __future__ import annotations

from importlib import resources

from .prompt_maestro import PromptLoadError

PROMPT_FILENAME = "prompt_scoring_system.md"


def load_scoring_system_prompt() -> str:
    """Devuelve la especificación técnica del sistema de scoring 0-3 puntos.

    Returns:
        str: Contenido en formato Markdown con la taxonomía de modalidades,
            fórmulas y casos especiales para calcular los 300 puntajes.

    Raises:
        PromptLoadError: Si el recurso empaquetado no está disponible o no se
            puede leer.
    """

    try:
        return resources.read_text(__package__, PROMPT_FILENAME, encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - improbable en producción
        raise PromptLoadError(
            f"No se encontró el recurso {PROMPT_FILENAME} en el paquete"
        ) from exc
    except OSError as exc:  # pragma: no cover - errores de lectura
        raise PromptLoadError(
            f"No se pudo leer el recurso {PROMPT_FILENAME}: {exc}"
        ) from exc


__all__ = ["load_scoring_system_prompt", "PromptLoadError"]
