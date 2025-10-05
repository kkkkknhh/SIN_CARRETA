"""Core contradiction detection engine with hybrid heuristics."""

from __future__ import annotations

import logging
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - defensive
    torch = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover - defensive
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

from pdm_contra.explain.tracer import ExplanationTracer
from pdm_contra.models import (
    AgendaGap,
    CompetenceValidation,
    ContradictionAnalysis,
    ContradictionMatch,
    RiskLevel,
)
from pdm_contra.nlp.nli import SpanishNLIDetector
from pdm_contra.nlp.patterns import PatternMatcher
from pdm_contra.policy.competence import CompetenceValidator
from pdm_contra.scoring.risk import RiskScorer

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """Hybrid contradiction detector for PDM analysis."""

    def __init__(
        self,
        competence_matrix_path: Optional[Path] = None,
        language: str = "es",
        mode_light: bool = False,
    ) -> None:
        self.language = language
        self.mode_light = mode_light

        # Initialize components
        self.pattern_matcher = PatternMatcher(language=language)
        self.nli_detector = SpanishNLIDetector(light_mode=mode_light)
        self.competence_validator = CompetenceValidator(
            matrix_path=competence_matrix_path
        )
        self.risk_scorer = RiskScorer()
        self.explainer = ExplanationTracer()

        self.encoder = None
        self._encoder_available = False
        if SentenceTransformer is not None:
            model_name = (
                "sentence-transformers/all-MiniLM-L6-v2"
                if mode_light
                else "sentence-transformers/multilingual-e5-base"
            )
            try:
                self.encoder = SentenceTransformer(model_name)
                self._encoder_available = True
            except Exception as exc:  # pragma: no cover - network/dependency issues
                logger.warning(
                    "Falling back to lexical alignment because encoder could not be loaded: %s",
                    exc,
                )

        logger.info(
            "Initialized ContradictionDetector in %s mode",
            "light" if mode_light else "full",
        )

    def detect_contradictions(
        self,
        text: str,
        sectors: Optional[List[str]] = None,
        pdm_structure: Optional[Dict[str, Any]] = None,
    ) -> ContradictionAnalysis:
        """Detect contradictions and related governance risks."""

        start_time = time.perf_counter()
        normalized_text = self._normalize_text(text)
        segments = self._segment_document(normalized_text, pdm_structure)

        contradictions: List[ContradictionMatch] = []
        competence_issues: List[CompetenceValidation] = []
        agenda_gaps: List[AgendaGap] = []

        for seg_type, seg_text, seg_meta in segments:
            pattern_matches = self.pattern_matcher.find_adversatives(seg_text)
            for match_idx, match in enumerate(pattern_matches):
                contradictions.append(
                    self._build_pattern_match(seg_type, match, match_idx)
                )

            if pdm_structure and seg_type in {"objetivos", "metas", "indicadores"}:
                contradictions.extend(
                    self._check_nli_contradictions(seg_text, pdm_structure, seg_type)
                )

            if sectors and seg_type in {"programas", "acciones"}:
                comp_results = self.competence_validator.validate_segment(
                    seg_text, sectors, seg_meta.get("nivel", "municipal")
                )
                for issue in comp_results:
                    competence_issues.append(
                        self._convert_competence_issue(seg_type, issue)
                    )

            if pdm_structure:
                agenda_gaps.extend(
                    self._check_agenda_alignment(seg_text, seg_type, pdm_structure)
                )

        risk_analysis = self.risk_scorer.calculate_risk(
            contradictions, competence_issues, agenda_gaps
        )

        explanations = self.explainer.generate_explanations(
            contradictions, competence_issues, agenda_gaps
        )

        processing_time = time.perf_counter() - start_time

        return ContradictionAnalysis(
            contradictions=contradictions,
            competence_mismatches=competence_issues,
            agenda_gaps=agenda_gaps,
            total_contradictions=len(contradictions),
            total_competence_issues=len(competence_issues),
            total_agenda_gaps=len(agenda_gaps),
            risk_score=risk_analysis.get("overall_risk", 0.0),
            risk_level=risk_analysis.get("risk_level", RiskLevel.LOW.value),
            confidence_intervals=risk_analysis.get("confidence_intervals", {}),
            explanations=explanations,
            calibration_info={
                "method": "conformal_prediction",
                "alpha": 0.1,
                "coverage": risk_analysis.get("empirical_coverage", 0.9),
            },
            processing_time_seconds=processing_time,
            model_versions={
                "nli": "heuristic-spanish" if self.mode_light else "heuristic-spanish",
                "pattern": "heuristic-adversatives",
            },
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _segment_document(
        text: str, structure: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        if not text:
            return [("general", "", {"start": 0, "end": 0})]

        section_patterns = {
            "diagnostico": r"(?i)(diagnóstico|análisis\s+situacional|contexto)",
            "objetivos": r"(?i)(objetivos?\s+estratégicos?|objetivos?\s+generales?)",
            "metas": r"(?i)(metas?\s+de?\s+resultado|metas?\s+de?\s+producto)",
            "indicadores": r"(?i)(indicadores?\s+de?\s+gestión|indicadores?\s+de?\s+impacto)",
            "programas": r"(?i)(programas?\s+y?\s+proyectos?|líneas?\s+estratégicas?)",
            "presupuesto": r"(?i)(presupuesto|plan\s+plurianual|recursos)",
        }

        segments: List[Tuple[str, str, Dict[str, Any]]] = []
        for sec_type, pattern in section_patterns.items():
            for match in re.finditer(pattern, text):
                start = match.end()
                end = min(start + 2000, len(text))

                next_sections: List[int] = []
                for other_pattern in section_patterns.values():
                    next_match = re.search(other_pattern, text[start:end])
                    if next_match:
                        next_sections.append(start + next_match.start())

                if next_sections:
                    end = min(next_sections)

                segment_text = text[start:end].strip()
                if segment_text:
                    segments.append(
                        (
                            sec_type,
                            segment_text,
                            {"start": start, "end": end, "header": match.group()},
                        )
                    )

        if not segments:
            segments.append(("general", text, {"start": 0, "end": len(text)}))

        return segments

    def _build_pattern_match(
        self, segment_type: str, match: Dict[str, Any], match_idx: int
    ) -> ContradictionMatch:
        context = match.get("context", "").strip()
        adversative = match.get("adversative", "").strip()
        lower_context = context.lower()
        lower_adv = adversative.lower()
        split_pos = lower_context.find(lower_adv) if adversative else -1

        if split_pos >= 0:
            premise = context[:split_pos].strip()
            hypothesis = context[split_pos + len(adversative) :].strip()
        else:
            half = len(context) // 2
            premise = context[:half].strip()
            hypothesis = context[half:].strip()

        if not premise:
            premise = context
        if not hypothesis:
            hypothesis = context

        confidence = 0.35
        if match.get("goals"):
            confidence += 0.2
        if match.get("action_verbs"):
            confidence += 0.2
        if match.get("quantitative"):
            confidence += 0.2
        if match.get("has_negation"):
            confidence += 0.05
        if match.get("has_uncertainty"):
            confidence -= 0.05

        confidence = max(0.0, min(1.0, confidence))
        risk_level = self._score_to_risk(confidence)

        return ContradictionMatch(
            id=f"pattern-{segment_type}-{match_idx}-{uuid.uuid4().hex[:8]}",
            type="adversative",
            premise=premise,
            hypothesis=hypothesis,
            context=context,
            adversatives=[adversative] if adversative else [],
            quantifiers=list(match.get("modals", [])),
            quantitative_targets=list(match.get("quantitative", [])),
            action_verbs=list(match.get("action_verbs", [])),
            confidence=confidence,
            risk_level=risk_level,
            risk_score=confidence,
            explanation=(
                "Se detectó un conector adversativo en el segmento "
                f"'{segment_type}' que sugiere posible contradicción."
            ),
            rules_fired=["pattern_adversative"],
            metadata={
                "segment": segment_type,
                "position": match.get("position", {}),
                "complexity": match.get("complexity"),
            },
        )

    def _check_nli_contradictions(
        self, text: str, structure: Dict[str, Any], segment_type: str
    ) -> List[ContradictionMatch]:
        pairs: List[Tuple[str, str, str]] = []
        if segment_type == "objetivos" and structure.get("metas"):
            for objetivo in self._extract_items(text, "objetivo"):
                for meta in structure.get("metas", []):
                    pairs.append((objetivo, str(meta), "objetivo-meta"))
        elif segment_type == "metas" and structure.get("indicadores"):
            for meta in self._extract_items(text, "meta"):
                for indicador in structure.get("indicadores", []):
                    pairs.append((meta, str(indicador), "meta-indicador"))

        matches: List[ContradictionMatch] = []
        for idx, (premise, hypothesis, pair_type) in enumerate(pairs):
            result = self.nli_detector.check_contradiction(premise, hypothesis)
            if result["is_contradiction"] and result.get("score", 0.0) > 0.6:
                score = float(result.get("score", 0.0))
                matches.append(
                    ContradictionMatch(
                        id=f"nli-{pair_type}-{idx}-{uuid.uuid4().hex[:8]}",
                        type="semantic",
                        premise=premise[:200],
                        hypothesis=hypothesis[:200],
                        context=f"{premise[:100]} ... {hypothesis[:100]}",
                        confidence=score,
                        risk_level=self._score_to_risk(score),
                        risk_score=score,
                        nli_score=score,
                        nli_label=result.get("label", "contradiction"),
                        explanation=(
                            "La comparación NLI detectó contradicción entre "
                            f"{pair_type.replace('-', ' ')}."
                        ),
                        rules_fired=["semantic_nli"],
                        pair_type=pair_type,
                    )
                )
        return matches

    def _convert_competence_issue(
        self, segment_type: str, issue: Dict[str, Any]
    ) -> CompetenceValidation:
        risk_level = (
            RiskLevel.HIGH
            if issue.get("type") == "competence_overreach"
            else RiskLevel.MEDIUM
        )
        return CompetenceValidation(
            id=f"competence-{segment_type}-{uuid.uuid4().hex[:8]}",
            sector=str(issue.get("sector", "general")),
            level=str(issue.get("level", "municipal")),
            action_text=str(issue.get("text", "")),
            is_valid=issue.get("type") != "competence_overreach",
            competence_type=str(issue.get("type", "")),
            required_level=str(issue.get("required_level", "municipal")),
            action_verb=issue.get("action_verb"),
            legal_basis=list(issue.get("legal_basis", [])),
            explanation=str(issue.get("explanation", "")),
            suggested_fix=issue.get("suggested_fix"),
            risk_level=risk_level,
            confidence=0.7 if risk_level is RiskLevel.HIGH else 0.5,
            metadata={
                "segment": segment_type,
                "position": issue.get("position", {}),
            },
        )

    def _check_agenda_alignment(
        self, text: str, segment_type: str, structure: Dict[str, Any]
    ) -> List[AgendaGap]:
        chain = [
            "diagnostico",
            "objetivos",
            "estrategias",
            "metas",
            "indicadores",
            "presupuesto",
        ]
        issues: List[AgendaGap] = []

        if segment_type in chain:
            idx = chain.index(segment_type)
            if idx > 0:
                prev_type = chain[idx - 1]
                prev_items = structure.get(prev_type) or []
                if prev_items and not self._has_alignment(text, prev_items):
                    issues.append(
                        AgendaGap(
                            id=f"agenda-back-{uuid.uuid4().hex[:8]}",
                            type="missing_backward_alignment",
                            from_element=segment_type,
                            to_element=prev_type,
                            missing_link=prev_type,
                            severity="medium",
                            explanation=(
                                f"No se evidencia alineación entre {segment_type} y {prev_type}."
                            ),
                            impact="Posible ruptura de cadena lógica",
                            risk_level=RiskLevel.MEDIUM,
                            confidence=0.5,
                            metadata={"segment": segment_type},
                        )
                    )
            if idx < len(chain) - 1:
                next_type = chain[idx + 1]
                if not structure.get(next_type):
                    issues.append(
                        AgendaGap(
                            id=f"agenda-forward-{uuid.uuid4().hex[:8]}",
                            type="missing_forward_element",
                            from_element=segment_type,
                            to_element=next_type,
                            missing_link=next_type,
                            severity="high",
                            explanation=(
                                f"El documento no incluye {next_type} posterior a {segment_type}."
                            ),
                            impact="Cadena de planeación incompleta",
                            risk_level=RiskLevel.HIGH,
                            confidence=0.6,
                            metadata={"segment": segment_type},
                        )
                    )
        return issues

    @staticmethod
    def _extract_items(text: str, item_type: str) -> List[str]:
        patterns = {
            "objetivo": r"objetivo\s*\d*[:.]?\s*([^.]+\.)",
            "meta": r"meta\s*\d*[:.]?\s*([^.]+\.)",
            "indicador": r"indicador\s*[:.]?\s*([^.]+\.)",
        }
        if item_type not in patterns:
            return []
        results = []
        for match in re.finditer(patterns[item_type], text, re.IGNORECASE):
            results.append(match.group(1).strip())
        return results

    def _has_alignment(self, text1: str, items: Iterable[str]) -> bool:
        candidates = list(items)
        if not candidates:
            return False

        if self._encoder_available and self.encoder is not None and util is not None and torch is not None:
            try:
                emb1 = self.encoder.encode(text1[:500], convert_to_tensor=True)
                emb2 = self.encoder.encode(candidates[:5], convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(emb1, emb2)
                max_sim = float(torch.max(similarities).item())
                return max_sim > 0.6
            except Exception as exc:  # pragma: no cover - runtime fallback
                logger.debug("Encoder alignment failed, using lexical fallback: %s", exc)

        # Lexical fallback: Jaccard similarity on meaningful tokens
        tokens1 = set(self._tokenize(text1))
        best_score = 0.0
        for candidate in candidates[:5]:
            tokens2 = set(self._tokenize(candidate))
            if not tokens1 or not tokens2:
                continue
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            score = intersection / union
            best_score = max(best_score, score)
        return best_score > 0.3

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in re.findall(r"\b\w+\b", text.lower()) if len(token) > 3]

    @staticmethod
    def _score_to_risk(score: float) -> RiskLevel:
        if score >= 0.8:
            return RiskLevel.HIGH
        if score >= 0.6:
            return RiskLevel.MEDIUM_HIGH
        if score >= 0.4:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
