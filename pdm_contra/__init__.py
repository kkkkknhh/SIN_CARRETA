"""Public interface for the ``pdm_contra`` package."""

from pdm_contra.core import ContradictionDetector
from pdm_contra.explain.tracer import ExplanationTracer
from pdm_contra.models import (
    AgendaGap,
    CompetenceValidation,
    ContradictionAnalysis,
    ContradictionMatch,
    PDMDocument,
    RiskLevel,
)
from pdm_contra.utils.guard_novelty import check_dependencies

__all__ = [
    "AgendaGap",
    "CompetenceValidation",
    "ContradictionAnalysis",
    "ContradictionDetector",
    "ContradictionMatch",
    "ExplanationTracer",
    "PDMDocument",
    "RiskLevel",
    "check_dependencies",
]
