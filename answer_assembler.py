"""
Answer Assembler for DECALOGO evaluation (P–D–Q canonical).
Compatible con el orquestador y la rúbrica basada en D#-Q#.
No modifica ningún otro archivo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
            print(
                "Warning: doctoral_argumentation_engine not available, using basic rationale"
            )

        # Initialize doctoral engine if available
        if doctoral_engine_available and self.evidence_registry:
            try:
                doctoral_engine = DoctoralArgumentationEngine(
                    evidence_registry=self.evidence_registry
                )
            except Exception as e:
                print(f"Warning: Could not initialize doctoral engine: {e}")
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

            if doctoral_engine and len(evidence_list) >= 1:
                try:
                    # Convert evidence to StructuredEvidence format
                    structured_evidence = []
                    for idx, ev in enumerate(evidence_list[:5]):  # Top 5 evidence items
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

                    # Create Bayesian posterior (simplified)
                    bayesian_posterior = {
                        "mean": confidence,
                        "lower_bound": max(0.0, confidence - 0.1),
                        "upper_bound": min(1.0, confidence + 0.1),
                    }

                    # Generate doctoral argument
                    if len(structured_evidence) >= 1:
                        argument_result = doctoral_engine.generate_argument(
                            question_id=uid,
                            score=score,
                            evidence_list=structured_evidence,
                            bayesian_posterior=bayesian_posterior,
                        )

                        # Extract components
                        doctoral_argument = {
                            "paragraphs": argument_result.get(
                                "argument_paragraphs", []
                            ),
                            "claim": argument_result.get("toulmin_structure", {}).get(
                                "claim", ""
                            ),
                            "ground": argument_result.get("toulmin_structure", {}).get(
                                "ground", ""
                            ),
                            "warrant": argument_result.get("toulmin_structure", {}).get(
                                "warrant", ""
                            ),
                        }

                        toulmin_structure = argument_result.get("toulmin_structure", {})

                        argument_quality = {
                            "logical_coherence": argument_result.get(
                                "logical_coherence_score", 0.0
                            ),
                            "academic_quality": argument_result.get(
                                "academic_quality_scores", {}
                            ),
                            "evidence_sources": argument_result.get(
                                "toulmin_structure", {}
                            ).get("evidence_sources", []),
                            "meets_doctoral_standards": (
                                argument_result.get("logical_coherence_score", 0.0)
                                >= 0.85
                                and argument_result.get(
                                    "academic_quality_scores", {}
                                ).get("overall", 0.0)
                                >= 0.80
                            ),
                        }

                except Exception as e:
                    print(
                        f"Warning: Could not generate doctoral argument for {uid}: {e}"
                    )
                    doctoral_argument = None

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
            }

            # Add doctoral argumentation if available
            if doctoral_argument:
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
