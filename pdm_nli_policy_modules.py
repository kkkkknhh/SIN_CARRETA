"""
Natural Language Inference for Spanish contradiction detection.
Uses state-of-the-art multilingual models.
"""

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from mapie.regression import MapieRegressor
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class SpanishNLIDetector:
    """
    NLI-based contradiction detector for Spanish text.
    Uses multilingual models fine-tuned for Spanish.
    """

    def __init__(self, light_mode: bool = False):
        """
        Initialize NLI detector.

        Args:
            light_mode: Use lightweight model if True
        """
        self.light_mode = light_mode

        # Select model based on mode
        if light_mode:
            # Lightweight multilingual model
            model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        else:
            # Full-size Spanish-optimized model
            model_name = "Recognai/bert-base-spanish-wwm-cased-xnli"

        logger.info(f"Loading NLI model: {model_name}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)

        # Move to GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        logger.info(f"NLI model loaded on {self.device}")

    def check_contradiction(
        self, premise: str, hypothesis: str, return_all_scores: bool = False
    ) -> Dict[str, any]:
        """
        Check if premise and hypothesis contradict each other.

        Args:
            premise: First statement
            hypothesis: Second statement
            return_all_scores: Return scores for all labels

        Returns:
            Dictionary with label, score, and optionally all scores
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Get predicted label
        predicted_label_idx = np.argmax(probs)
        predicted_label = self.label_map[predicted_label_idx]
        confidence = float(probs[predicted_label_idx])

        result = {
            "label": predicted_label,
            "score": confidence,
            "is_contradiction": predicted_label == "contradiction",
        }

        if return_all_scores:
            result["all_scores"] = {
                label: float(probs[idx]) for idx, label in self.label_map.items()
            }

        return result

    def batch_check(
        self, pairs: List[Tuple[str, str]], batch_size: int = 8
    ) -> List[Dict[str, any]]:
        """
        Check multiple premise-hypothesis pairs in batches.

        Args:
            pairs: List of (premise, hypothesis) tuples
            batch_size: Batch size for processing

        Returns:
            List of results for each pair
        """
        results = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i: i + batch_size]
            premises, hypotheses = zip(*batch_pairs)

            # Tokenize batch
            inputs = self.tokenizer(
                list(premises),
                list(hypotheses),
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Get batch predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Process each result
            for j, prob_dist in enumerate(probs):
                predicted_idx = np.argmax(prob_dist)
                predicted_label = self.label_map[predicted_idx]

                results.append(
                    {
                        "premise": premises[j][:100],
                        "hypothesis": hypotheses[j][:100],
                        "label": predicted_label,
                        "score": float(prob_dist[predicted_idx]),
                        "is_contradiction": predicted_label == "contradiction",
                        "contradiction_score": float(
                            prob_dist[2]
                        ),  # Index 2 is contradiction
                    }
                )

        return results

    def find_contradictory_pairs(
        self, statements: List[str], threshold: float = 0.7
    ) -> List[Dict[str, any]]:
        """
        Find all contradictory pairs in a list of statements.

        Args:
            statements: List of statements to compare
            threshold: Minimum contradiction score

        Returns:
            List of contradictory pairs
        """
        contradictions = []

        # Generate all pairs
        pairs = []
        indices = []
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                pairs.append((statements[i], statements[j]))
                indices.append((i, j))

        if not pairs:
            return contradictions

        # Check all pairs
        results = self.batch_check(pairs)

        # Filter contradictions
        for (i, j), result in zip(indices, results):
            if result["contradiction_score"] >= threshold:
                contradictions.append(
                    {
                        "statement1": statements[i],
                        "statement2": statements[j],
                        "indices": (i, j),
                        "contradiction_score": result["contradiction_score"],
                        "confidence": result["score"],
                    }
                )

        return contradictions


# pdm_contra/policy/competence.py
"""
Competence validation for municipal PDM actions.
Checks if proposed actions fall within municipal jurisdiction.
"""

logger = logging.getLogger(__name__)


class CompetenceMatrix(BaseModel):
    """Model for competence matrix data."""

    municipal: Dict[str, List[str]] = Field(default_factory=dict)
    departmental: Dict[str, List[str]] = Field(default_factory=dict)
    national: Dict[str, List[str]] = Field(default_factory=dict)
    shared: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    legal_basis: Dict[str, List[str]] = Field(default_factory=dict)


class CompetenceValidator:
    """
    Validates if actions fall within proper governmental competence levels.
    Based on Colombian constitutional and legal framework.
    """

    def __init__(self, matrix_path: Optional[Path] = None):
        """
        Initialize competence validator.

        Args:
            matrix_path: Path to competence matrix JSON file
        """
        self.matrix = self._load_matrix(matrix_path)
        self._compile_patterns()

    def _load_matrix(self, path: Optional[Path]) -> CompetenceMatrix:
        """Load competence matrix from file or use defaults."""
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CompetenceMatrix(**data)
        else:
            # Use default Colombian competence matrix
            return self._get_default_matrix()

    @staticmethod
    def _get_default_matrix() -> CompetenceMatrix:
        """Get default competence matrix based on Colombian law."""
        return CompetenceMatrix(
            municipal={
                "salud": [
                    "gestionar",
                    "coordinar",
                    "articular",
                    "promover",
                    "vigilar",
                    "supervisar",
                    "cofinanciar",
                    "apoyar",
                ],
                "educacion": [
                    "mantener infraestructura",
                    "dotar",
                    "gestionar",
                    "promover acceso",
                    "coordinar",
                    "apoyar",
                ],
                "agua": [
                    "prestar servicio",
                    "garantizar acceso",
                    "mantener",
                    "construir acueducto local",
                    "supervisar",
                ],
                "ambiente": [
                    "proteger recursos locales",
                    "gestionar residuos",
                    "controlar emisiones locales",
                    "educar",
                ],
                "movilidad": [
                    "mantener vías terciarias",
                    "señalizar",
                    "organizar",
                    "regular transporte local",
                    "construir andenes",
                ],
                "seguridad": [
                    "apoyar",
                    "coordinar",
                    "prevenir",
                    "promover convivencia",
                    "gestionar fondos seguridad",
                ],
                "cultura": [
                    "promover",
                    "apoyar",
                    "gestionar espacios",
                    "fomentar",
                    "preservar patrimonio local",
                ],
                "vivienda": [
                    "promover",
                    "gestionar subsidios",
                    "titular predios",
                    "mejorar entorno",
                    "legalizar asentamientos",
                ],
            },
            departmental={
                "salud": [
                    "administrar red hospitalaria",
                    "contratar especialistas",
                    "dirigir ESE nivel 2-3",
                    "coordinar referencia",
                ],
                "educacion": [
                    "nombrar docentes",
                    "administrar nómina",
                    "evaluar",
                    "certificar instituciones",
                    "definir plantas",
                ],
                "agua": [
                    "planificar regional",
                    "asistir técnicamente",
                    "coordinar intermunicipal",
                    "grandes obras",
                ],
                "ambiente": [
                    "otorgar licencias regionales",
                    "administrar parques",
                    "controlar cuencas",
                    "regular regional",
                ],
                "movilidad": [
                    "construir vías secundarias",
                    "mantener red departamental",
                    "regular intermunicipal",
                    "grandes terminales",
                ],
                "seguridad": [
                    "coordinar regional",
                    "apoyar policía departamental",
                    "gestionar cárceles departamentales",
                ],
            },
            national={
                "salud": [
                    "definir política",
                    "regular sistema",
                    "administrar FOSYGA",
                    "habilitar IPS",
                    "formar especialistas",
                ],
                "educacion": [
                    "definir currículo nacional",
                    "expedir títulos superiores",
                    "regular profesiones",
                    "administrar ICFES",
                ],
                "defensa": [
                    "comandar fuerzas",
                    "combatir narcotráfico",
                    "defender soberanía",
                    "inteligencia nacional",
                ],
                "justicia": [
                    "nombrar jueces",
                    "administrar justicia",
                    "legislar",
                    "indultar",
                ],
                "relaciones_exteriores": [
                    "representar país",
                    "negociar tratados",
                    "política exterior",
                    "servicio diplomático",
                ],
            },
            shared={
                "salud": {
                    "municipal": ["promoción", "prevención", "atención básica"],
                    "departamental": ["red hospitalaria", "referencia"],
                    "nacional": ["política", "regulación", "inspección"],
                },
                "educacion": {
                    "municipal": ["infraestructura", "dotación", "alimentación"],
                    "departamental": ["docentes", "calidad", "cobertura"],
                    "nacional": ["lineamientos", "evaluación", "educación superior"],
                },
            },
            legal_basis={
                "constitucion": [
                    "Art. 311",
                    "Art. 313",
                    "Art. 287",
                    "Art. 356",
                    "Art. 357",
                ],
                "leyes": [
                    "Ley 136/1994",
                    "Ley 715/2001",
                    "Ley 1551/2012",
                    "Ley 152/1994",
                ],
                "decretos": ["Decreto 111/1996", "Decreto 28/2008"],
            },
        )

    def _compile_patterns(self):
        """Compile regex patterns for competence detection."""
        flags = re.IGNORECASE | re.UNICODE

        # Patterns that indicate overreach
        self.overreach_patterns = {
            "salud": re.compile(
                r"\b(construir|administrar|dirigir)\s+(hospital|clínica|ESE\s+nivel\s+[23]|"
                r"red\s+hospitalaria|centro\s+especializado)",
                flags,
            ),
            "educacion": re.compile(
                r"\b(nombrar|contratar|asignar|evaluar|sancionar)\s+(docentes?|profesores?|"
                r"rectores?|coordinadores?|personal\s+educativo)",
                flags,
            ),
            "seguridad": re.compile(
                r"\b(comandar|dirigir|ordenar|instruir)\s+(polic[íi]a|ej[ée]rcito|"
                r"fuerzas?\s+armadas?|militares?)",
                flags,
            ),
            "justicia": re.compile(
                r"\b(nombrar|designar|remover)\s+(jueces?|magistrados?|fiscales?)|"
                r"\badministrar\s+justicia",
                flags,
            ),
        }

        # Patterns for valid municipal actions
        self.valid_patterns = re.compile(
            r"\b(gestionar|coordinar|articular|apoyar|promover|facilitar|"
            r"cofinanciar|vigilar|supervisar|colaborar|coadyuvar)\b",
            flags,
        )

    def validate_segment(
        self, text: str, sectors: List[str], level: str = "municipal"
    ) -> List[Dict[str, any]]:
        """
        Validate competences in a text segment.

        Args:
            text: Text to validate
            sectors: Relevant sectors
            level: Government level (municipal/departmental/national)

        Returns:
            List of competence validation results
        """
        issues = []

        # Check for overreach patterns
        for sector in sectors:
            if sector in self.overreach_patterns:
                pattern = self.overreach_patterns[sector]

                for match in pattern.finditer(text):
                    # Extract context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    # Check if it's mitigated by valid coordination verb
                    if self.valid_patterns.search(context):
                        continue

                    issue = {
                        "type": "competence_overreach",
                        "sector": sector,
                        "level": level,
                        "text": match.group(),
                        "position": {"start": match.start(), "end": match.end()},
                        "context": context,
                        "required_level": self._get_required_level(
                            match.group(), sector
                        ),
                        "legal_basis": self._get_legal_basis(sector),
                        "explanation": self._explain_overreach(
                            match.group(), sector, level
                        ),
                        "suggested_fix": self._suggest_fix(match.group(), sector),
                    }

                    issues.append(issue)

        # Check for missing competences
        issues.extend(self._check_missing_competences(text, sectors, level))

        return issues

    def _get_required_level(self, action: str, sector: str) -> str:
        """Determine required government level for an action."""
        action_lower = action.lower()

        # Check each level's competences
        for level, competences in [
            ("municipal", self.matrix.municipal),
            ("departmental", self.matrix.departmental),
            ("national", self.matrix.national),
        ]:
            if sector in competences:
                for verb in competences[sector]:
                    if verb in action_lower:
                        return level

        # Check for specific keywords
        if any(
            word in action_lower
            for word in ["hospital", "ESE nivel 2", "ESE nivel 3", "red hospitalaria"]
        ):
            return "departmental"

        if any(
            word in action_lower
            for word in [
                "docentes",
                "profesores",
                "nómina educativa",
                "plantas docentes",
            ]
        ):
            return "departmental"

        if any(
            word in action_lower
            for word in [
                "ejército",
                "fuerzas armadas",
                "policía nacional",
                "narcotráfico",
            ]
        ):
            return "national"

        return "municipal"  # Default

    @staticmethod
    def _get_legal_basis(sector: str) -> List[str]:
        """Get legal basis for sector competences."""
        basis = []

        # General municipal competence laws
        basis.extend(
            [
                "Constitución Política Art. 311, 313",
                "Ley 136 de 1994",
                "Ley 1551 de 2012",
            ]
        )

        # Sector-specific laws
        sector_laws = {
            "salud": ["Ley 715 de 2001 Art. 44-45", "Ley 1438 de 2011"],
            "educacion": ["Ley 715 de 2001 Art. 7-8", "Ley 115 de 1994"],
            "agua": ["Ley 142 de 1994", "Decreto 1077 de 2015"],
            "ambiente": ["Ley 99 de 1993", "Decreto 1076 de 2015"],
            "movilidad": ["Ley 105 de 1993", "Ley 336 de 1996"],
            "seguridad": ["Ley 62 de 1993", "Ley 1801 de 2016"],
            "vivienda": ["Ley 388 de 1997", "Ley 1537 de 2012"],
        }

        if sector in sector_laws:
            basis.extend(sector_laws[sector])

        return basis

    @staticmethod
    def _explain_overreach(action: str, sector: str, level: str) -> str:
        """Generate explanation for competence overreach."""
        explanations = {
            "salud": f"La acción '{action}' excede las competencias municipales en salud. "
            f"Los municipios pueden gestionar y cofinanciar, pero no administrar "
            f"directamente hospitales o ESE de nivel 2-3.",
            "educacion": f"La acción '{action}' corresponde al nivel departamental. "
            f"Los municipios no tienen competencia para nombrar o contratar "
            f"docentes directamente.",
            "seguridad": f"La acción '{action}' es competencia exclusiva del nivel nacional. "
            f"Los municipios solo pueden apoyar y coordinar con las autoridades.",
            "justicia": f"La acción '{action}' es función exclusiva de la rama judicial. "
            f"Los municipios no tienen competencia en administración de justicia.",
        }

        return explanations.get(
            sector,
            f"La acción '{action}' podría exceder las competencias del nivel {level}.",
        )

    @staticmethod
    def _suggest_fix(action: str, sector: str) -> str:
        """Suggest alternative wording that respects competences."""
        suggestions = {
            "salud": "Considere usar: 'gestionar convenios para', 'cofinanciar', "
            "'apoyar la operación de', 'coordinar con el departamento'",
            "educacion": "Considere usar: 'gestionar ante el departamento', "
            "'solicitar asignación de', 'apoyar con recursos para'",
            "seguridad": "Considere usar: 'coordinar con', 'apoyar logísticamente', "
            "'gestionar recursos para', 'facilitar la acción de'",
            "justicia": "Considere usar: 'facilitar el acceso a', 'promover', "
            "'gestionar casas de justicia', 'apoyar programas de'",
        }

        return suggestions.get(
            sector, "Considere reformular usando verbos de coordinación y apoyo."
        )

    @staticmethod
    def _check_missing_competences(
        text: str, sectors: List[str], level: str
    ) -> List[Dict[str, any]]:
        """Check for missing essential competences in text."""
        issues = []

        # Essential municipal competences that should be present
        essential = {
            "salud": ["promoción", "prevención", "atención básica"],
            "educacion": ["infraestructura educativa", "dotación escolar"],
            "agua": ["acceso agua potable", "saneamiento básico"],
            "ambiente": ["gestión residuos", "protección recursos"],
        }

        for sector in sectors:
            if sector in essential and level == "municipal":
                for competence in essential[sector]:
                    if competence not in text.lower():
                        issues.append(
                            {
                                "type": "missing_essential_competence",
                                "sector": sector,
                                "competence": competence,
                                "explanation": f"No se encontró mención de '{competence}', "
                                f"competencia esencial municipal en {sector}",
                            }
                        )

        return issues


# pdm_contra/scoring/risk.py
"""
Risk scoring with conformal prediction for uncertainty quantification.
"""

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Risk scoring with conformal prediction for PDM analysis.
    Provides calibrated uncertainty estimates.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize risk scorer.

        Args:
            alpha: Significance level for conformal prediction (default 0.1 = 90% coverage)
        """
        self.alpha = alpha
        self.calibration_data = []
        self.risk_model = None
        self.conformal_predictor = None

    def calculate_risk(
        self,
        contradictions: List[Any],
        competence_issues: List[Any],
        agenda_gaps: List[Any],
    ) -> Dict[str, Any]:
        """
        Calculate overall risk with uncertainty quantification.

        Args:
            contradictions: List of detected contradictions
            competence_issues: List of competence validation issues
            agenda_gaps: List of agenda alignment gaps

        Returns:
            Risk analysis with scores and confidence intervals
        """
        # Calculate component scores
        contradiction_score = self._score_contradictions(contradictions)
        competence_score = self._score_competences(competence_issues)
        agenda_score = self._score_agenda(agenda_gaps)

        # Combine scores
        weights = {"contradiction": 0.4, "competence": 0.35, "agenda": 0.25}

        overall_risk = (
            weights["contradiction"] * contradiction_score
            + weights["competence"] * competence_score
            + weights["agenda"] * agenda_score
        )

        # Get confidence intervals using conformal prediction
        confidence_intervals = self._get_confidence_intervals(
            contradiction_score, competence_score, agenda_score
        )

        # Determine risk level
        risk_level = self._score_to_level(overall_risk)

        # Calculate empirical coverage if we have calibration data
        empirical_coverage = (
            self._calculate_coverage() if self.calibration_data else 0.9
        )

        return {
            "overall_risk": float(overall_risk),
            "risk_level": (
                risk_level.value if hasattr(
                    risk_level, "value") else risk_level
            ),
            "component_scores": {
                "contradiction": float(contradiction_score),
                "competence": float(competence_score),
                "agenda": float(agenda_score),
            },
            "confidence_intervals": confidence_intervals,
            "empirical_coverage": empirical_coverage,
            "calibration_samples": len(self.calibration_data),
        }

    @staticmethod
    def _score_contradictions(contradictions: List[Any]) -> float:
        """Score contradiction severity."""
        if not contradictions:
            return 0.0

        scores = []
        for c in contradictions:
            # Base score from confidence
            base_score = getattr(c, "confidence", 0.5)

            # Adjust for type
            type_multiplier = 1.0
            if hasattr(c, "type"):
                type_multipliers = {
                    "semantic": 1.2,
                    "quantitative": 1.1,
                    "budgetary": 1.3,
                    "temporal": 0.9,
                }
                type_str = str(c.type).lower()
                type_multiplier = type_multipliers.get(type_str, 1.0)

            # Adjust for NLI score if available
            nli_boost = 0.0
            if hasattr(c, "nli_score") and c.nli_score is not None:
                nli_boost = c.nli_score * 0.2

            score = min(1.0, base_score * type_multiplier + nli_boost)
            scores.append(score)

        # Aggregate with emphasis on high scores
        if scores:
            # Use a combination of mean and max
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            # Normalize by expected count
            count_factor = min(1.0, len(scores) / 10)

            return float(0.6 * mean_score + 0.3 * max_score + 0.1 * count_factor)

        return 0.0

    @staticmethod
    def _score_competences(issues: List[Any]) -> float:
        """Score competence issues severity."""
        if not issues:
            return 0.0

        scores = []
        for issue in issues:
            # Base severity
            if isinstance(issue, dict):
                issue_type = issue.get("type", "")
                if "overreach" in issue_type:
                    base_score = 0.8
                elif "missing" in issue_type:
                    base_score = 0.6
                else:
                    base_score = 0.5
            else:
                base_score = getattr(issue, "confidence", 0.5)

            scores.append(base_score)

        if scores:
            # Weight by count and max severity
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            count_penalty = min(0.3, len(scores) * 0.05)

            return float(min(1.0, 0.5 * mean_score + 0.2 * max_score + count_penalty))

        return 0.0

    @staticmethod
    def _score_agenda(gaps: List[Any]) -> float:
        """Score agenda alignment gaps."""
        if not gaps:
            return 0.0

        scores = []
        for gap in gaps:
            if isinstance(gap, dict):
                gap_type = gap.get("type", "")
                severity = gap.get("severity", "medium")

                # Type-based scoring
                type_scores = {
                    "missing_backward_alignment": 0.6,
                    "missing_forward_element": 0.8,
                    "broken_chain": 0.9,
                }
                base_score = type_scores.get(gap_type, 0.5)

                # Severity adjustment
                severity_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}
                multiplier = severity_multipliers.get(severity, 1.0)

                scores.append(base_score * multiplier)
            else:
                scores.append(0.5)

        if scores:
            return float(min(1.0, np.mean(scores) + len(scores) * 0.02))

        return 0.0

    def _get_confidence_intervals(
        self, contradiction_score: float, competence_score: float, agenda_score: float
    ) -> Dict[str, List[float]]:
        """
        Calculate confidence intervals using conformal prediction.

        Returns:
            Dictionary with confidence intervals for each component
        """
        # If we have enough calibration data, use conformal prediction
        if len(self.calibration_data) >= 30:
            intervals = self._conformal_intervals(
                contradiction_score, competence_score, agenda_score
            )
        else:
            # Use bootstrap-based intervals as fallback
            intervals = self._bootstrap_intervals(
                contradiction_score, competence_score, agenda_score
            )

        return intervals

    def _conformal_intervals(
        self, contra_score: float, comp_score: float, agenda_score: float
    ) -> Dict[str, List[float]]:
        """Calculate conformal prediction intervals."""
        # Prepare features
        X = np.array([[contra_score, comp_score, agenda_score]])

        if self.conformal_predictor is None:
            # Initialize and train conformal predictor
            self._train_conformal_predictor()

        if self.conformal_predictor is not None:
            # Get prediction intervals
            y_pred, y_pis = self.conformal_predictor.predict(
                X, ensemble=True, alpha=self.alpha
            )

            # Extract intervals
            lower = float(y_pis[0, 0, 0])
            upper = float(y_pis[0, 1, 0])

            # Component-wise intervals (simplified)
            return {
                "overall": [lower, upper],
                "contradiction": [
                    max(0, contra_score - 0.1),
                    min(1, contra_score + 0.1),
                ],
                "competence": [max(0, comp_score - 0.1), min(1, comp_score + 0.1)],
                "agenda": [max(0, agenda_score - 0.1), min(1, agenda_score + 0.1)],
            }
        else:
            # Fallback to bootstrap
            return self._bootstrap_intervals(contra_score, comp_score, agenda_score)

    def _bootstrap_intervals(
        self, contra_score: float, comp_score: float, agenda_score: float
    ) -> Dict[str, List[float]]:
        """Calculate bootstrap confidence intervals."""
        # Simple bootstrap simulation
        n_bootstrap = 1000
        samples = []

        for _ in range(n_bootstrap):
            # Add noise to simulate uncertainty
            noise = np.random.normal(0, 0.05, 3)
            sample_contra = np.clip(contra_score + noise[0], 0, 1)
            sample_comp = np.clip(comp_score + noise[1], 0, 1)
            sample_agenda = np.clip(agenda_score + noise[2], 0, 1)

            # Calculate overall with weights
            sample_overall = (
                0.4 * sample_contra + 0.35 * sample_comp + 0.25 * sample_agenda
            )
            samples.append(sample_overall)

        # Calculate percentiles
        lower = np.percentile(samples, (self.alpha / 2) * 100)
        upper = np.percentile(samples, (1 - self.alpha / 2) * 100)

        return {
            "overall": [float(lower), float(upper)],
            "contradiction": [max(0, contra_score - 0.1), min(1, contra_score + 0.1)],
            "competence": [max(0, comp_score - 0.1), min(1, comp_score + 0.1)],
            "agenda": [max(0, agenda_score - 0.1), min(1, agenda_score + 0.1)],
        }

    def _train_conformal_predictor(self):
        """Train conformal predictor on calibration data."""
        if len(self.calibration_data) < 30:
            logger.warning(
                "Insufficient calibration data for conformal prediction")
            return

        try:
            # Prepare training data
            X = np.array(
                [[d["contra"], d["comp"], d["agenda"]]
                    for d in self.calibration_data]
            )
            y = np.array([d["overall"] for d in self.calibration_data])

            # Split data
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Base regressor
            base_regressor = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42
            )

            # Conformal predictor
            self.conformal_predictor = MapieRegressor(
                estimator=base_regressor, method="plus", cv="prefit"
            )

            # Fit on training data
            base_regressor.fit(X_train, y_train)

            # Calibrate on calibration set
            self.conformal_predictor.fit(X_cal, y_cal)

            logger.info("Conformal predictor trained successfully")

        except Exception as e:
            logger.error(f"Failed to train conformal predictor: {e}")
            self.conformal_predictor = None

    def _calculate_coverage(self) -> float:
        """Calculate empirical coverage from calibration data."""
        if not self.calibration_data or len(self.calibration_data) < 10:
            return 0.9  # Default assumption

        # Simple coverage calculation
        in_interval_count = 0
        for data in self.calibration_data[-100:]:  # Last 100 samples
            if "interval" in data and "true_value" in data:
                lower, upper = data["interval"]
                if lower <= data["true_value"] <= upper:
                    in_interval_count += 1

        if len(self.calibration_data) > 0:
            return in_interval_count / min(100, len(self.calibration_data))
        return 0.9

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Convert numerical score to risk level."""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM_HIGH"
        elif score >= 0.2:
            return "MEDIUM"
        else:
            return "LOW"

    def add_calibration_sample(
        self,
        contradiction_score: float,
        competence_score: float,
        agenda_score: float,
        true_overall: float,
        interval: Optional[Tuple[float, float]] = None,
    ):
        """
        Add a calibration sample for improving predictions.

        Args:
            contradiction_score: Contradiction component score
            competence_score: Competence component score
            agenda_score: Agenda component score
            true_overall: True overall risk score
            interval: Optional interval prediction
        """
        sample = {
            "contra": contradiction_score,
            "comp": competence_score,
            "agenda": agenda_score,
            "overall": true_overall,
        }

        if interval:
            sample["interval"] = interval
            sample["true_value"] = true_overall

        self.calibration_data.append(sample)

        # Retrain if we have enough new data
        if len(self.calibration_data) % 50 == 0:
            self._train_conformal_predictor()
