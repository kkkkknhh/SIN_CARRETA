"""Prompts disponibles para el motor PDM."""

from .prompt_maestro import PromptLoadError, load_prompt_maestro
from .prompt_scoring_system import load_scoring_system_prompt

__all__ = [
    "load_prompt_maestro",
    "load_scoring_system_prompt",
    "PromptLoadError",
]
