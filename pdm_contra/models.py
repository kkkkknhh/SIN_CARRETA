"""Lightweight data models used by the ``pdm_contra`` package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskLevel(str, Enum):
    """Enumerates the risk levels used across the contradiction pipeline."""

    LOW = "low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium-high"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PDMDocument:
    """Represents a parsed PDM document ready for analysis."""

    id: str
    municipality: str
    department: str
    year_range: str
    text: str
    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContradictionMatch:
    """Container for a contradiction detected in the text."""

    id: str
    type: str
    premise: str
    hypothesis: str
    context: str
    sector: Optional[str] = None
    premise_location: Dict[str, int] = field(default_factory=dict)
    hypothesis_location: Dict[str, int] = field(default_factory=dict)
    adversatives: List[str] = field(default_factory=list)
    quantifiers: List[str] = field(default_factory=list)
    quantitative_targets: List[str] = field(default_factory=list)
    action_verbs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    nli_score: Optional[float] = None
    nli_label: Optional[str] = None
    explanation: str = ""
    suggested_fix: Optional[str] = None
    rules_fired: List[str] = field(default_factory=list)
    pair_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetenceValidation:
    """Outcome of validating a policy action against competence rules."""

    id: str
    sector: str
    level: str
    action_text: str
    is_valid: bool
    competence_type: str
    required_level: str
    action_verb: Optional[str] = None
    legal_basis: List[str] = field(default_factory=list)
    explanation: str = ""
    suggested_fix: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgendaGap:
    """Represents a break in the agenda-setting alignment chain."""

    id: str
    type: str
    from_element: str
    to_element: str
    missing_link: str
    found_elements: List[str] = field(default_factory=list)
    expected_elements: List[str] = field(default_factory=list)
    severity: str = "medium"
    explanation: str = ""
    impact: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContradictionAnalysis:
    """Aggregated output of the contradiction detection pipeline."""

    contradictions: List[ContradictionMatch] = field(default_factory=list)
    competence_mismatches: List[CompetenceValidation] = field(default_factory=list)
    agenda_gaps: List[AgendaGap] = field(default_factory=list)
    total_contradictions: int = 0
    total_competence_issues: int = 0
    total_agenda_gaps: int = 0
    risk_score: float = 0.0
    risk_level: str = RiskLevel.LOW.value
    confidence_intervals: Dict[str, List[float]] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    calibration_info: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_seconds: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)


__all__ = [
    "AgendaGap",
    "CompetenceValidation",
    "ContradictionAnalysis",
    "ContradictionMatch",
    "PDMDocument",
    "RiskLevel",
]
