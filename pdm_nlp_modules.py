"""
Data models for PDM contradiction detection.
"""

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(Enum):
    """Risk level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium-high"
    HIGH = "high"
    CRITICAL = "critical"


class ContradictionType(Enum):
    """Types of contradictions."""

    ADVERSATIVE = "adversative"
    SEMANTIC = "semantic"
    QUANTITATIVE = "quantitative"
    COMPETENCE = "competence"
    AGENDA = "agenda"
    TEMPORAL = "temporal"
    BUDGETARY = "budgetary"


class PDMDocument(BaseModel):
    """PDM document model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Document ID")
    municipality: str = Field(description="Municipality name")
    department: str = Field(description="Department name")
    year_range: str = Field(description="PDM validity period")
    text: str = Field(description="Full document text")
    sections: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ContradictionMatch(BaseModel):
    """Model for a detected contradiction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Unique match ID")
    type: ContradictionType = Field(description="Type of contradiction")
    sector: Optional[str] = Field(None, description="Related sector")

    # Text spans
    premise: str = Field(description="First statement")
    hypothesis: str = Field(description="Contradicting statement")
    context: str = Field(description="Surrounding context")

    # Location info
    premise_location: Dict[str, int] = Field(description="Start/end positions")
    hypothesis_location: Dict[str, int] = Field(
        description="Start/end positions")

    # Detection evidence
    adversatives: List[str] = Field(default_factory=list)
    quantifiers: List[str] = Field(default_factory=list)
    action_verbs: List[str] = Field(default_factory=list)

    # Scores
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    nli_score: Optional[float] = Field(None, ge=0, le=1)
    nli_label: Optional[str] = Field(None)

    # Risk assessment
    risk_level: RiskLevel = Field(description="Risk level")
    risk_score: float = Field(ge=0, le=1)

    # Explanation
    explanation: str = Field(description="Human-readable explanation")
    suggested_fix: Optional[str] = Field(None)
    rules_fired: List[str] = Field(default_factory=list)


class CompetenceValidation(BaseModel):
    """Competence validation result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Validation ID")
    sector: str = Field(description="Government sector")
    level: str = Field(
        description="Government level (municipal/departmental/national)")

    # Action analyzed
    action_text: str = Field(description="Action text analyzed")
    action_verb: str = Field(description="Main action verb")

    # Validation result
    is_valid: bool = Field(description="Whether competence is valid")
    competence_type: str = Field(description="Type of competence issue")
    required_level: str = Field(description="Required government level")

    # Evidence
    legal_basis: List[str] = Field(
        default_factory=list, description="Legal references")
    explanation: str = Field(description="Explanation of issue")
    suggested_fix: str = Field(description="Suggested correction")

    # Risk
    risk_level: RiskLevel = Field(description="Risk level")
    confidence: float = Field(ge=0, le=1)


class AgendaGap(BaseModel):
    """Agenda-setting gap or misalignment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Gap ID")
    type: str = Field(description="Type of gap")

    # Chain elements
    from_element: str = Field(description="Source element in chain")
    to_element: str = Field(description="Target element in chain")

    # Evidence
    missing_link: str = Field(description="What is missing")
    found_elements: List[str] = Field(default_factory=list)
    expected_elements: List[str] = Field(default_factory=list)

    # Assessment
    severity: str = Field(description="Gap severity")
    explanation: str = Field(description="Explanation")
    impact: str = Field(description="Potential impact")

    # Risk
    risk_level: RiskLevel = Field(description="Risk level")
    confidence: float = Field(ge=0, le=1)


class ContradictionAnalysis(BaseModel):
    """Complete contradiction analysis results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Findings
    contradictions: List[ContradictionMatch] = Field(default_factory=list)
    competence_mismatches: List[CompetenceValidation] = Field(
        default_factory=list)
    agenda_gaps: List[AgendaGap] = Field(default_factory=list)

    # Counts
    total_contradictions: int = Field(ge=0)
    total_competence_issues: int = Field(ge=0)
    total_agenda_gaps: int = Field(ge=0)

    # Risk assessment
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel = Field()
    confidence_intervals: Dict[str, List[float]] = Field(default_factory=dict)

    # Explanations
    explanations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # Calibration
    calibration_info: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: float = Field(ge=0)
    model_versions: Dict[str, str] = Field(default_factory=dict)


# pdm_contra/nlp/patterns.py
"""
Pattern matching for Spanish adversatives and linguistic markers.
"""

logger = logging.getLogger(__name__)


class PatternMatcher:
    """
    Advanced pattern matcher for Spanish PDM texts.
    """

    def __init__(self, language: str = "es"):
        """Initialize pattern matcher with language-specific patterns."""
        self.language = language
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        flags = re.IGNORECASE | re.UNICODE

        # Adversative connectors (expanded set)
        self.adversatives = self._compile_list(
            [
                r"\bsin\s+embargo\b",
                r"\bpero\b",
                r"\baunque\b",
                r"\bno\s+obstante\b",
                r"\ba\s+pesar\s+de(\s+que)?\b",
                r"\bempero\b",
                r"\bmientras\s+que\b",
                r"\bpor\s+el\s+contrario\b",
                r"\ben\s+cambio\b",
                r"\bcon\s+todo\b",
                r"\bahora\s+bien\b",
                r"\bsin\s+que\b",
                r"\bpese\s+a(\s+que)?\b",
                r"\baun\s+cuando\b",
                r"\bsi\s+bien\b",
            ],
            flags,
        )

        # Goal and objective indicators
        self.goals = self._compile_list(
            [
                r"\bmetas?\b",
                r"\bobjetivos?\b",
                r"\bprop[óo]sitos?\b",
                r"\bfinalidad(es)?\b",
                r"\bresultados?\s+esperados?\b",
                r"\blogros?\s+esperados?\b",
                r"\balcanzar\b",
                r"\bconseguir\b",
                r"\bobtener\b",
                r"\blograr\b",
                r"\baspiraci[óo]n(es)?\b",
                r"\bpretende(r|mos)?\b",
                r"\bbusca(r|mos)?\b",
                r"\bse\s+espera\b",
                r"\bse\s+proyecta\b",
                r"\bse\s+planea\b",
                r"\blineamiento\s+estrat[ée]gico\b",
            ],
            flags,
        )

        # Action verbs (governance-specific)
        self.action_verbs = self._compile_list(
            [
                # Implementation verbs
                r"\bimplementar\b",
                r"\bejecutar\b",
                r"\bdesarrollar\b",
                r"\brealizar\b",
                r"\bllevar\s+a\s+cabo\b",
                r"\bponer\s+en\s+marcha\b",
                # Creation verbs
                r"\bestablece(r|mos)?\b",
                r"\bcrea(r|mos)?\b",
                r"\bconstrui(r|mos)?\b",
                r"\bforma(r|mos)?\b",
                r"\bfunda(r|mos)?\b",
                r"\binstaura(r|mos)?\b",
                # Promotion verbs
                r"\bpromove(r|mos)?\b",
                r"\bfomenta(r|mos)?\b",
                r"\bimpulsa(r|mos)?\b",
                r"\bfortalece(r|mos)?\b",
                r"\bestimula(r|mos)?\b",
                r"\bincentivar?\b",
                # Improvement verbs
                r"\bmejora(r|mos)?\b",
                r"\boptimiza(r|mos)?\b",
                r"\bincrementa(r|mos)?\b",
                r"\baumenta(r|mos)?\b",
                r"\bamplia(r|mos)?\b",
                r"\bexpandi(r|mos)?\b",
                # Reduction verbs
                r"\breduci(r|mos)?\b",
                r"\bdisminui(r|mos)?\b",
                r"\bminimiza(r|mos)?\b",
                r"\bcontrola(r|mos)?\b",
                r"\blimita(r|mos)?\b",
                r"\brestringe?\b",
                # Guarantee verbs
                r"\bgarantiza(r|mos)?\b",
                r"\basegura(r|mos)?\b",
                r"\bprotege(r|mos)?\b",
                r"\bdefiende?\b",
                r"\bsalvaguarda(r|mos)?\b",
                r"\brespeta(r|mos)?\b",
                # Coordination verbs
                r"\bcoordina(r|mos)?\b",
                r"\barticula(r|mos)?\b",
                r"\bgestiona(r|mos)?\b",
                r"\blidera(r|mos)?\b",
                r"\bdirige?\b",
                r"\badministra(r|mos)?\b",
                # Monitoring verbs
                r"\bmonitorea(r|mos)?\b",
                r"\bsupervisa(r|mos)?\b",
                r"\bevalúa(r|mos)?\b",
                r"\beverifica(r|mos)?\b",
                r"\binspecciona(r|mos)?\b",
                r"\bfiscaliza(r|mos)?\b",
                # Capacity building
                r"\bcapacita(r|mos)?\b",
                r"\bforma(r|mos)?\b",
                r"\beduca(r|mos)?\b",
                r"\bsensibiliza(r|mos)?\b",
                r"\bentrena(r|mos)?\b",
                r"\bprepara(r|mos)?\b",
                # Problem solving
                r"\bpreviene?\b",
                r"\batiende?\b",
                r"\bresuelve?\b",
                r"\bsoluciona(r|mos)?\b",
                r"\benfrenta(r|mos)?\b",
                r"\baborda(r|mos)?\b",
            ],
            flags,
        )

        # Quantitative indicators
        self.quantitative = self._compile_list(
            [
                # Percentages
                r"\d+(?:[.,]\d+)?\s*(?:%|por\s*ciento|porciento)\b",
                # Large numbers
                r"\d+(?:[.,]\d+)?\s*(?:millones?|mil(?:es)?|MM|M)\b",
                # Increases/decreases with numbers
                r"\b(?:incrementa|aumenta|sube|eleva)(?:r|rá|remos)?\s+(?:en\s+|hasta\s+|un\s+)?\d+(?:[.,]\d+)?",
                r"\b(?:reduce|disminuye|baja|decrece)(?:r|rá|remos)?\s+(?:en\s+|hasta\s+|un\s+)?\d+(?:[.,]\d+)?",
                # Currency
                r"(?:COP\s*)?\$?\s*\d+(?:[.,]\d+)?\s*(?:COP|mil(?:lones)?|MM|pesos|USD|dólares?)\b",
                # Coverage targets
                r"\bcobertura\s+del?\s+\d+(?:[.,]\d+)?\s*%",
                r"\b(?:beneficia|atiende|cubre)\s+a?\s*\d+(?:[.,]\d+)?\s*(?:personas?|familias?|hogares?|habitantes?)",
                # Area measurements
                r"\d+(?:[.,]\d+)?\s*(?:hectáreas?|km²|m²|metros?\s+cuadrados?)",
                # Time frames
                r"\b(?:para|en|antes\s+del?|hasta)\s+(?:el\s+)?(?:año\s+)?\d{4}\b",
                r"\b(?:durante|en)\s+\d+\s+(?:años?|meses?|días?|semanas?)",
                # Indicators
                r"\b(?:índice|tasa|porcentaje|proporción)\s+de\s+\d+(?:[.,]\d+)?",
                r"\b\d+(?:[.,]\d+)?\s*(?:puntos?\s+porcentuales?|pp|p\.p\.)",
                # Absolute goals
                r"\bmeta\s+de\s+\d+(?:[.,]\d+)?",
                r"\b(?:alcanzar|lograr|conseguir)\s+\d+(?:[.,]\d+)?",
            ],
            flags,
        )

        # Modal verbs and uncertainty markers
        self.modals = self._compile_list(
            [
                r"\b(?:podría|debería|tendría|habría)(?:n|mos)?\b",
                r"\b(?:puede|debe|tiene)(?:n)?\s+que\b",
                r"\bes\s+posible\s+que\b",
                r"\bprobablemente\b",
                r"\bposiblemente\b",
                r"\beventualmente\b",
                r"\btal\s+vez\b",
                r"\bquizás?\b",
                r"\bacaso\b",
            ],
            flags,
        )

        # Negation markers
        self.negations = self._compile_list(
            [
                r"\bno\b",
                r"\bnunca\b",
                r"\bjamás\b",
                r"\btampoco\b",
                r"\bni\b",
                r"\bningún\w*\b",
                r"\bnada\b",
                r"\bnadie\b",
            ],
            flags,
        )

        # Exception and condition markers
        self.exceptions = self._compile_list(
            [
                r"\bexcepto\b",
                r"\bsalvo\b",
                r"\bmenos\b",
                r"\bexcluyendo\b",
                r"\bcon\s+excepción\s+de\b",
                r"\ba\s+menos\s+que\b",
                r"\bsiempre\s+que\b",
                r"\bsiempre\s+y\s+cuando\b",
                r"\ben\s+caso\s+de\s+que?\b",
                r"\bsi\s+y\s+s[óo]lo\s+si\b",
            ],
            flags,
        )

    @staticmethod
    def _compile_list(patterns: List[str], flags: int) -> List[Pattern]:
        """Compile a list of regex patterns."""
        return [re.compile(p, flags) for p in patterns]

    def find_adversatives(self, text: str, context_window: int = 200) -> List[Dict]:
        """
        Find adversative patterns with context.

        Args:
            text: Input text
            context_window: Characters of context to extract

        Returns:
            List of match dictionaries
        """
        matches = []

        for pattern in self.adversatives:
            for match in pattern.finditer(text):
                context_start = max(0, match.start() - context_window // 2)
                context_end = min(len(text), match.end() + context_window // 2)
                context = text[context_start:context_end]

                # Find associated patterns in context
                goals_found = self._find_in_context(context, self.goals)
                verbs_found = self._find_in_context(context, self.action_verbs)
                quants_found = self._find_in_context(
                    context, self.quantitative)
                modals_found = self._find_in_context(context, self.modals)
                negations_found = self._find_in_context(
                    context, self.negations)

                # Calculate complexity score
                complexity = len(goals_found) + \
                    len(verbs_found) + len(quants_found)
                has_modal = len(modals_found) > 0
                has_negation = len(negations_found) > 0

                match_dict = {
                    "adversative": match.group(),
                    "position": {"start": match.start(), "end": match.end()},
                    "context": context,
                    "goals": goals_found,
                    "action_verbs": verbs_found,
                    "quantitative": quants_found,
                    "modals": modals_found,
                    "negations": negations_found,
                    "complexity": complexity,
                    "has_uncertainty": has_modal,
                    "has_negation": has_negation,
                }

                matches.append(match_dict)

        return matches

    @staticmethod
    def _find_in_context(context: str, patterns: List[Pattern]) -> List[str]:
        """Find all pattern matches in context."""
        found = []
        for pattern in patterns:
            for match in pattern.finditer(context):
                found.append(match.group().strip())
        return list(set(found))  # Remove duplicates

    @staticmethod
    def extract_competence_verbs(text: str) -> List[Tuple[str, int, int]]:
        """
        Extract action verbs that might indicate competence issues.

        Returns:
            List of (verb, start_pos, end_pos) tuples
        """
        verbs = []

        # Special competence-sensitive verbs
        sensitive_patterns = [
            # Health sector (departmental/national)
            r"\b(?:construir|administrar|dirigir)\s+(?:hospital|clínica|ESE|IPS)\b",
            r"\basignar\s+(?:médicos?|especialistas?|personal\s+de\s+salud)\b",
            r"\bcontratar\s+(?:médicos?|especialistas?)\b",
            # Education sector (departmental/national)
            r"\b(?:nombrar|asignar|contratar)\s+(?:docentes?|profesores?|maestros?)\b",
            r"\badministrar\s+(?:instituciones?\s+educativas?|colegios?|escuelas?)\b",
            r"\bdefinir\s+(?:currículo|plan\s+de\s+estudios)\b",
            # Security (national)
            r"\b(?:comandar|dirigir)\s+(?:policía|fuerzas?\s+armadas?|ejército)\b",
            r"\bcombatir\s+(?:narcotráfico|terrorismo|crimen\s+organizado)\b",
            # Infrastructure (depends on scale)
            r"\bconstruir\s+(?:autopistas?|aeropuertos?|puertos?)\b",
            r"\badministrar\s+(?:peajes?|concesiones?\s+viales?)\b",
            # Justice (national)
            r"\b(?:nombrar|designar)\s+(?:jueces?|magistrados?|fiscales?)\b",
            r"\badministrar\s+justicia\b",
        ]

        flags = re.IGNORECASE | re.UNICODE
        for pattern_str in sensitive_patterns:
            pattern = re.compile(pattern_str, flags)
            for match in pattern.finditer(text):
                verbs.append((match.group(), match.start(), match.end()))

        return verbs
