"""Carga el Prompt Maestro para evaluación causal de PDM."""

from __future__ import annotations

from importlib import resources

PROMPT_FILENAME = "prompt_maestro_pdm.md"


class PromptLoadError(RuntimeError):
    """Error al cargar un prompt desde los recursos del paquete."""


def load_prompt_maestro() -> str:
    """Devuelve el contenido íntegro del Prompt Maestro para PDM.

    Returns:
        str: Texto en formato Markdown con instrucciones completas para el
            pipeline automatizado.

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


__all__ = ["load_prompt_maestro", "PromptLoadError"]
