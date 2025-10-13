"""
Answer Assembler for DECALOGO evaluation (P–D–Q canonical).
Compatible con el orquestador y la rúbrica basada en D#-Q#.
No modifica ningún otro archivo.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------
# Regex canónicos P–D–Q
# ---------------------------
_RX_UID = re.compile(r"^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$")
_RX_POLICY = re.compile(r"^P(10|[1-9])$")
_RX_DIM = re.compile(r"^D[1-6]$")
_RX_Q = re.compile(r"^Q[1-9][0-9]*$")
_RX_RUBRIC = re.compile(r"^D[1-6]-Q[1-9][0-9]*$")


def _std_qid(raw_id: str) -> str:
    """Normaliza IDs legados a formato P#-D#-Q# o lanza ValueError."""
    if not raw_id:
        raise ValueError("ERROR_QID_FORMAT: empty id")
    if _RX_UID.match(raw_id):
        return raw_id
    # Intentos de normalización: buscar P, D, Q en cualquier orden
    # Use word boundary to avoid matching P1 in P11
    mP = re.search(r"\bP(10|[1-9])\b", raw_id, flags=re.IGNORECASE)
    mD = re.search(r"\bD([1-6])\b", raw_id, flags=re.IGNORECASE)
    mQ = re.search(r"\bQ([1-9][0-9]*)\b", raw_id, flags=re.IGNORECASE)
    if mP and mD and mQ:
        uid = f"P{mP.group(1)}-D{mD.group(1)}-Q{mQ.group(1)}"
        if _RX_UID.match(uid):
            return uid
    raise ValueError(f"ERROR_QID_FORMAT: invalid id '{raw_id}' — expected 'P#-D#-Q#'")


def _rubric_key_from_uid(uid: str) -> str:
    """Extrae D#-Q# desde P#-D#-Q#."""
    if not _RX_UID.match(uid):
        raise ValueError(f"ERROR_QID_FORMAT: '{uid}'")
    _, d, q = uid.split("-")
    rk = f"{d}-{q}"
    if not _RX_RUBRIC.match(rk):
        raise ValueError(f"ERROR_RUBRIC_KEY_DERIVATION: '{rk}' from '{uid}'")
    return rk


@dataclass
class _Quote:
    text: str
    confidence: float = 0.5


class AnswerAssembler:
    """Assembles answers from evidence registry and rubric."""

    def __init__(
        self,
        rubric_path: Optional[Path] = None,
        evidence_registry: Optional[Any] = None,
    ):
        """
        Args:
            rubric_path: Ruta al JSON de rúbrica (con 'weights' por D#-Q#).
            evidence_registry: Registro de evidencias; se espera método
                               `get_evidence_for_question(question_unique_id)` o adaptador equivalente.
        """
        self.rubric_path = Path(rubric_path) if rubric_path else None
        self.evidence_registry = evidence_registry
        self.rubric: Dict[str, Any] = {}
        if self.rubric_path and self.rubric_path.exists():
            self._load_rubric()

    # ---------------------------
    # Rúbrica
    # ---------------------------
    def _load_rubric(self) -> None:
        try:
            with open(self.rubric_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"ERROR_RUBRIC_LOAD: {e}") from e

        if (
            not isinstance(data, dict)
            or "weights" not in data
            or not isinstance(data["weights"], dict)
        ):
            raise ValueError("ERROR_RUBRIC_SCHEMA: missing 'weights' dict")

        # Validar claves D#-Q#
        for k in data["weights"].keys():
            if not _RX_RUBRIC.match(k):
                raise ValueError(
                    f"ERROR_RUBRIC_KEY: invalid rubric key '{k}' (expected D#-Q#)"
                )
        self.rubric = data

    # ---------------------------
    # Evidencia
    # ---------------------------
    def _get_evidence_for_question(self, uid: str) -> List[_Quote]:
        """Recupera evidencias desde el registry con tolerancia al adaptador."""
        evs: List[_Quote] = []
        if not self.evidence_registry:
            return evs

        items = []

        # Ruta 1: método estándar get_evidence_for_question (adaptador custom)
        if hasattr(self.evidence_registry, "get_evidence_for_question"):
            try:
                items = self.evidence_registry.get_evidence_for_question(uid) or []
            except Exception:
                items = []
        # Ruta 2: método for_question (EvidenceRegistry nativo) - usa D#-Q# format
        elif hasattr(self.evidence_registry, "for_question"):
            try:
                # Extract rubric key (D#-Q#) from uid (P#-D#-Q#)
                rk = _rubric_key_from_uid(uid)
                items = self.evidence_registry.for_question(rk) or []
            except Exception:
                items = []
        # Ruta 3: acceso directo al diccionario interno (mejor evitar, pero tolerado)
        else:
            reg = getattr(self.evidence_registry, "registry", None)
            if reg is not None and hasattr(reg, "_evidence"):
                try:
                    for entry in reg._evidence.values():  # type: ignore[attr-defined]
                        stored = getattr(entry, "metadata", {}).get(
                            "question_unique_id", ""
                        )
                        try:
                            if _std_qid(stored) == uid:
                                txt = None
                                conf = getattr(entry, "confidence", 0.5)
                                meta = getattr(entry, "metadata", {})
                                if isinstance(meta, dict):
                                    txt = meta.get("text") or meta.get("excerpt") or ""
                                if not txt:
                                    cnt = getattr(entry, "content", None)
                                    if isinstance(cnt, dict):
                                        txt = cnt.get("text") or ""
                                evs.append(
                                    _Quote(
                                        text=(txt[:147] + "...")
                                        if txt and len(txt) > 150
                                        else (txt or ""),
                                        confidence=conf,
                                    )
                                )
                        except Exception:
                            continue
                except Exception:
                    items = []

        # Normalizar a _Quote
        norm: List[_Quote] = []
        for it in items:
            if isinstance(it, _Quote):
                norm.append(it)
            else:
                # intento de adaptación mínima
                text = ""
                conf = 0.5
                if hasattr(it, "metadata") and isinstance(it.metadata, dict):
                    text = it.metadata.get("text") or it.metadata.get("excerpt") or text
                if hasattr(it, "confidence"):
                    try:
                        conf = float(it.confidence)
                    except Exception:
                        pass
                if not text and hasattr(it, "content"):
                    if isinstance(it.content, dict):
                        text = it.content.get("text") or text
                norm.append(
                    _Quote(
                        text=(text[:147] + "...") if text and len(text) > 150 else text,
                        confidence=conf,
                    )
                )
        return norm

    # ---------------------------
    # Ensamblado
    # ---------------------------
    def assemble(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensambla el reporte final en dos niveles:
          - question_answers: lista de respuestas por P–D–Q con argumentación doctoral
          - global_summary: cobertura, pesos y score ponderado

        evaluation_inputs espera `{"questionnaire_eval": {"question_results": [...]}}`

        ENHANCED: Integrates doctoral_argumentation_engine for rigorous justifications
        with complete Toulmin structure validation and quality thresholds.
        """
        # Import doctoral argumentation engine
        try:
            from doctoral_argumentation_engine import (
                DoctoralArgumentationEngine,
                StructuredEvidence,
            )

            doctoral_engine_available = True
        except ImportError:
            doctoral_engine_available = False
            logger.warning(
                "doctoral_argumentation_engine not available, using basic rationale"
            )

        # Initialize doctoral engine if available
        if doctoral_engine_available and self.evidence_registry:
            try:
                doctoral_engine = DoctoralArgumentationEngine(
                    evidence_registry=self.evidence_registry
                )
            except Exception as e:
                logger.warning(f"Could not initialize doctoral engine: {e}")
                doctoral_engine = None
        else:
            doctoral_engine = None

        questionnaire_eval = (
            evaluation_inputs.get("questionnaire_eval", {})
            if isinstance(evaluation_inputs, dict)
            else {}
        )
        q_results: Sequence[Dict[str, Any]] = (
            questionnaire_eval.get("question_results", []) or []
        )

        if not isinstance(q_results, Sequence):
            raise ValueError(
                "ERROR_INPUT_FORMAT: 'questionnaire_eval.question_results' must be a list"
            )

        weights = (self.rubric or {}).get("weights", {})
        question_answers: List[Dict[str, Any]] = []
        argumentation_failures: List[Dict[str, Any]] = []

        for res in q_results:
            raw_id = str(res.get("question_id", "")).strip()
            score = float(res.get("score", 0.0) or 0.0)

            uid = _std_qid(raw_id)  # normaliza o falla
            rk = _rubric_key_from_uid(uid)
            weight = float(weights.get(rk, 0.0))

            # Evidencias y metadatos
            evidence_list = self._get_evidence_for_question(uid)
            confidence = self._calc_confidence(evidence_list, score)
            caveats = self._caveats(evidence_list, score)
            rationale = self._rationale(uid, evidence_list, score)

            # ========================================
            # DOCTORAL ARGUMENTATION INTEGRATION
            # ========================================
            doctoral_argument = None
            toulmin_structure = None
            argument_quality = {}
            argumentation_status = "not_attempted"

            # VALIDATION GATE 1: Check minimum evidence requirement (≥3 sources)
            if len(evidence_list) < 3:
                caveats.append(
                    f"Insufficient evidence for doctoral argumentation: {len(evidence_list)}/3 required sources"
                )
                argumentation_status = "insufficient_evidence"
                argumentation_failures.append(
                    {
                        "question_id": uid,
                        "reason": "insufficient_evidence",
                        "evidence_count": len(evidence_list),
                        "required_count": 3,
                    }
                )

            if (
                doctoral_engine
                and len(evidence_list) >= 3
                and argumentation_status == "not_attempted"
            ):
                try:
                    # Convert evidence to StructuredEvidence format
                    structured_evidence = []
                    for idx, ev in enumerate(evidence_list):
                        structured_evidence.append(
                            StructuredEvidence(
                                source_module=f"detector_{idx}",
                                evidence_type="textual",
                                content=ev.text,
                                confidence=ev.confidence,
                                applicable_questions=[uid],
                                metadata={},
                            )
                        )

                    # Create Bayesian posterior with proper format
                    bayesian_posterior = {
                        "posterior_mean": confidence,
                        "credible_interval_95": (
                            max(0.0, confidence - 0.1),
                            min(1.0, confidence + 0.1),
                        ),
                    }

                    # Generate doctoral argument with quality validation
                    argument_result = doctoral_engine.generate_argument(
                        question_id=uid,
                        score=score,
                        evidence_list=structured_evidence,
                        bayesian_posterior=bayesian_posterior,
                    )

                    # Extract Toulmin structure components
                    toulmin_structure = argument_result.get("toulmin_structure", {})

                    # VALIDATION GATE 2: Verify complete Toulmin structure
                    required_toulmin_fields = [
                        "claim",
                        "ground",
                        "warrant",
                        "backing",
                        "qualifier",
                        "rebuttal",
                    ]
                    missing_fields = [
                        field
                        for field in required_toulmin_fields
                        if not toulmin_structure.get(field)
                    ]

                    if missing_fields:
                        raise ValueError(
                            f"Incomplete Toulmin structure: missing {', '.join(missing_fields)}"
                        )

                    # VALIDATION GATE 3: Check coherence_score threshold (≥0.85)
                    coherence_score = argument_result.get(
                        "logical_coherence_score", 0.0
                    )
                    if coherence_score < 0.85:
                        raise ValueError(
                            f"Coherence score {coherence_score:.3f} below threshold 0.85"
                        )

                    # VALIDATION GATE 4: Check quality_score threshold (≥0.80)
                    academic_quality = argument_result.get(
                        "academic_quality_scores", {}
                    )
                    quality_score = academic_quality.get("overall_score", 0.0)
                    if quality_score < 0.80:
                        raise ValueError(
                            f"Quality score {quality_score:.3f} below threshold 0.80"
                        )

                    # All validations passed - assemble doctoral argument
                    doctoral_argument = {
                        "paragraphs": argument_result.get("argument_paragraphs", []),
                        "claim": toulmin_structure.get("claim", ""),
                        "ground": toulmin_structure.get("ground", ""),
                        "warrant": toulmin_structure.get("warrant", ""),
                        "backing": toulmin_structure.get("backing", []),
                        "qualifier": toulmin_structure.get("qualifier", ""),
                        "rebuttal": toulmin_structure.get("rebuttal", ""),
                    }

                    argument_quality = {
                        "coherence_score": coherence_score,
                        "quality_score": quality_score,
                        "academic_quality_breakdown": academic_quality,
                        "evidence_sources": toulmin_structure.get(
                            "evidence_sources", []
                        ),
                        "evidence_synthesis_map": argument_result.get(
                            "evidence_synthesis_map", {}
                        ),
                        "meets_doctoral_standards": True,
                        "confidence_alignment_error": argument_result.get(
                            "confidence_alignment_error", 0.0
                        ),
                        "validation_timestamp": argument_result.get(
                            "validation_timestamp", ""
                        ),
                    }

                    argumentation_status = "success"

                except ValueError as e:
                    # Quality threshold failures
                    error_msg = str(e)
                    caveats.append(f"Doctoral argument validation failed: {error_msg}")
                    argumentation_status = "validation_failed"
                    argumentation_failures.append(
                        {
                            "question_id": uid,
                            "reason": "validation_failed",
                            "error": error_msg,
                            "evidence_count": len(evidence_list),
                        }
                    )
                    logger.warning(f"Argumentation validation failed for {uid}: {e}")

                except Exception as e:
                    # Other errors (generation failures)
                    caveats.append(
                        f"Doctoral argument generation error: {type(e).__name__}"
                    )
                    argumentation_status = "generation_failed"
                    argumentation_failures.append(
                        {
                            "question_id": uid,
                            "reason": "generation_failed",
                            "error": str(e),
                            "evidence_count": len(evidence_list),
                        }
                    )
                    logger.error(f"Argumentation generation failed for {uid}: {e}")

            # Build answer entry
            answer_entry = {
                "question_id": uid,
                "dimension": uid.split("-")[1],  # D#
                "raw_score": score,
                "rubric_key": rk,
                "rubric_weight": weight,
                "confidence": confidence,
                "evidence_ids": [],
                "evidence_count": len(evidence_list),
                "supporting_quotes": [q.text for q in evidence_list[:3] if q.text],
                "caveats": caveats,
                "scoring_modality": "CANONICAL_PDQ",
                "rationale": rationale,
                "argumentation_status": argumentation_status,
            }

            # Add doctoral argumentation if successfully generated and validated
            if doctoral_argument and toulmin_structure:
                answer_entry["doctoral_justification"] = doctoral_argument
                answer_entry["toulmin_structure"] = toulmin_structure
                answer_entry["argument_quality"] = argument_quality

            question_answers.append(answer_entry)

        total_weight = sum(qa["rubric_weight"] for qa in question_answers)
        weighted_score = sum(
            qa["raw_score"] * qa["rubric_weight"] for qa in question_answers
        )

        # Calculate doctoral quality statistics
        doctoral_coverage = sum(
            1 for qa in question_answers if "doctoral_justification" in qa
        )
        doctoral_high_quality = sum(
            1
            for qa in question_answers
            if qa.get("argument_quality", {}).get("meets_doctoral_standards", False)
        )

        # Categorize argumentation outcomes
        argumentation_by_status = {
            "success": sum(
                1 for qa in question_answers if qa.get("argumentation_status") == "success"
            ),
            "insufficient_evidence": sum(
                1
                for qa in question_answers
                if qa.get("argumentation_status") == "insufficient_evidence"
            ),
            "validation_failed": sum(
                1
                for qa in question_answers
                if qa.get("argumentation_status") == "validation_failed"
            ),
            "generation_failed": sum(
                1
                for qa in question_answers
                if qa.get("argumentation_status") == "generation_failed"
            ),
            "not_attempted": sum(
                1
                for qa in question_answers
                if qa.get("argumentation_status") == "not_attempted"
            ),
        }

        # Calculate quality metrics for successful arguments
        successful_arguments = [
            qa
            for qa in question_answers
            if qa.get("argumentation_status") == "success"
            and "argument_quality" in qa
        ]

        avg_coherence = (
            sum(
                qa["argument_quality"].get("coherence_score", 0.0)
                for qa in successful_arguments
            )
            / len(successful_arguments)
            if successful_arguments
            else 0.0
        )

        avg_quality = (
            sum(
                qa["argument_quality"].get("quality_score", 0.0)
                for qa in successful_arguments
            )
            / len(successful_arguments)
            if successful_arguments
            else 0.0
        )

        global_summary = {
            "answered_questions": len(question_answers),
            "total_questions": 300,  # contrato canónico por defecto (10×6×5)
            "total_weight": total_weight,
            "weighted_score": weighted_score,
            "average_confidence": (
                sum(qa["confidence"] for qa in question_answers) / len(question_answers)
            )
            if question_answers
            else 0.0,
            "doctoral_argumentation": {
                "coverage": doctoral_coverage,
                "coverage_percentage": (doctoral_coverage / len(question_answers) * 100)
                if question_answers
                else 0.0,
                "high_quality_count": doctoral_high_quality,
                "high_quality_percentage": (
                    doctoral_high_quality / doctoral_coverage * 100
                )
                if doctoral_coverage > 0
                else 0.0,
                "engine_available": doctoral_engine_available,
                "status_breakdown": argumentation_by_status,
                "average_coherence_score": round(avg_coherence, 3),
                "average_quality_score": round(avg_quality, 3),
                "validation_thresholds": {
                    "min_evidence_sources": 3,
                    "min_coherence_score": 0.85,
                    "min_quality_score": 0.80,
                },
                "failures": argumentation_failures if argumentation_failures else [],
            },
        }

        return {
            "question_answers": question_answers,
            "global_summary": global_summary,
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "rubric_loaded": bool(weights),
                "rubric_path": str(self.rubric_path) if self.rubric_path else None,
                "doctoral_engine_enabled": doctoral_engine is not None,
                "toulmin_components_validated": [
                    "claim",
                    "ground",
                    "warrant",
                    "backing",
                    "qualifier",
                    "rebuttal",
                ],
            },
        }

    # ---------------------------
    # Heurísticas de confianza/narrativa
    # ---------------------------
    @staticmethod
    def _calc_confidence(evidence: List[_Quote], score: float) -> float:
        if not evidence:
            base = 0.3
        else:
            avg = sum(max(0.0, min(1.0, q.confidence)) for q in evidence) / len(
                evidence
            )
            factor = min(len(evidence) / 3.0, 1.0)
            base = avg * factor
        extremity = abs(score - 0.5) * 2
        penalty = 0.85 if (extremity > 0.7 and len(evidence) < 2) else 1.0
        conf = base * penalty
        return round(max(0.0, min(1.0, conf)), 2)

    @staticmethod
    def _caveats(evidence: List[_Quote], score: float) -> List[str]:
        caveats: List[str] = []
        if len(evidence) == 0:
            caveats.append("No supporting evidence found")
        elif len(evidence) == 1:
            caveats.append("Based on single evidence source")
        if score > 0.8 and len(evidence) < 2:
            caveats.append("High score with limited evidence—verify manually")
        return caveats

    @staticmethod
    def _rationale(uid: str, evidence: List[_Quote], score: float) -> str:
        dimension = uid.split("-")[1]
        if not evidence:
            return f"No evidence found for {dimension}. Score reflects absence of required information."
        if score > 0.7:
            return f"Strong evidence supports high compliance in {dimension}."
        if score > 0.4:
            return f"Partial evidence indicates moderate compliance in {dimension}."
        return f"Limited evidence suggests low compliance in {dimension}."


__all__ = ["AnswerAssembler"]
