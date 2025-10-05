"""Tests for the Prompt Maestro and scoring prompts."""

from pdm_contra.prompts import (
    load_prompt_maestro,
    load_scoring_system_prompt,
)


def test_prompt_maestro_contains_expected_sections() -> None:
    content = load_prompt_maestro()
    assert "# PROMPT MAESTRO PARA INTEGRACIÓN EN PIPELINE AUTOMATIZADO" in content
    assert "## II. CUESTIONARIO BASE" in content
    assert "CONFIG = {" in content
    assert "## ESTE PROMPT ESTÁ LISTO PARA INTEGRACIÓN" in content


def test_prompt_maestro_is_not_empty() -> None:
    content = load_prompt_maestro().strip()
    assert len(content) > 1000


def test_scoring_prompt_contains_expected_sections() -> None:
    content = load_scoring_system_prompt()
    assert "# PROMPT SISTEMA DE SCORING - ESPECIFICACIÓN TÉCNICA COMPLETA" in content
    assert "## I. TAXONOMÍA DE MODALIDADES DE SCORING" in content
    assert "## III. AGREGACIÓN DE SCORES" in content
    assert "## IX. CHECKLIST DE IMPLEMENTACIÓN" in content


def test_scoring_prompt_is_not_empty() -> None:
    content = load_scoring_system_prompt().strip()
    assert len(content) > 1000
