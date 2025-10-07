#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Answer Assembler - Sistema Industrial de Ensamblaje de Respuestas
Versión: 4.0 (Single Source of Truth: RUBRIC_SCORING.json)

Genera reportes granulares con trazabilidad completa de evidencia, cargando toda
la configuración de scoring desde RUBRIC_SCORING.json:
- 'weights' section: per-question weight mappings
- 'questions' section: scoring modalities and expected elements
- 'dimensions' section: aggregation metadata
- 'score_bands' section: thresholds and recommendations

RUBRIC_SCORING.json es la única fuente de verdad para configuración de rúbrica.
"""

from __future__ import annotations

import json
import logging
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# El import de pandas es opcional y solo necesario para exportar a Excel/CSV.
# Si no está disponible, se puede comentar junto con las funciones de exportación.
try:
    import pandas as pd
except ImportError:
    pd = None

# ==================== CONFIGURACIÓN DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout
)
LOGGER = logging.getLogger("AnswerAssembler")


# ==================== CLASE MOCK PARA PRUEBAS ====================
# Esta es una simulación de la clase EvidenceRegistry para que el script sea ejecutable.
# En el pipeline real, esta clase sería importada desde su propio módulo.
@dataclass
class MockEvidence:
    confidence: float
    metadata: Dict[str, Any]


class EvidenceRegistry:
    def __init__(self, evidences: List[MockEvidence] = None):
        self._evidence_map = defaultdict(list)
        if evidences:
            for ev in evidences:
                # Se asume que la evidencia tiene un question_unique_id
                q_id = ev.metadata.get("question_unique_id")
                if q_id:
                    self._evidence_map[q_id].append(ev)

    def get_evidence_for_question(self, question_unique_id: str) -> List[MockEvidence]:
        return self._evidence_map.get(question_unique_id, [])


# ==================== ENUMS Y CONSTANTES ====================
from enum import Enum


class EvidenceQuality(Enum):
    EXCELENTE = "excelente"
    BUENA = "buena"
    ACEPTABLE = "aceptable"
    DEBIL = "débil"
    INSUFICIENTE = "insuficiente"


# ==================== DATACLASSES (Estructuras de Datos del Reporte) ====================
@dataclass
class EvidenceMetrics:
    total_evidences: int
    avg_confidence: float
    max_confidence: float
    min_confidence: float
    quality_distribution: Dict[str, int]
    sources_diversity: int


@dataclass
class QuestionAnswer:
    question_id: str
    dimension: str
    point_code: str
    question_number: int
    raw_score: float
    max_score: float
    score_percentage: float
    scoring_modality: str
    rubric_weight: float
    evidence_ids: List[str]
    evidence_count: int
    evidence_metrics: EvidenceMetrics
    confidence: float
    quality_assessment: str
    rationale: str
    warnings: List[str] = field(default_factory=list)
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionSummary:
    dimension_id: str
    dimension_name: str
    point_code: str
    question_scores: List[float]
    total_score: float
    max_score: float
    percentage: float
    total_evidences: int
    avg_confidence: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class PointSummary:
    point_code: str
    point_title: str
    dimension_summaries: List[DimensionSummary]
    average_percentage: float
    total_score: float
    max_score: float
    is_applicable: bool = True
    na_reason: Optional[str] = None
    critical_gaps: List[str] = field(default_factory=list)


@dataclass
class GlobalSummary:
    global_percentage: float
    score_band: str
    score_band_emoji: str
    total_questions: int
    answered_questions: int
    questions_with_evidence: int
    applicable_points: int
    total_points: int
    total_evidences: int
    avg_confidence: float
    evidence_quality_distribution: Dict[str, int]
    score_band_description: str
    main_recommendation: str
    critical_findings: List[str] = field(default_factory=list)


# ==================== ANSWER ASSEMBLER (VERSIÓN CORREGIDA) ====================

class AnswerAssembler:
    """
    Ensamblador industrial de respuestas del sistema 300Q.
    Integra evidencia y scoring, adhiriéndose estrictamente a la estructura de
    rubric_scoring.json y DECALOGO_FULL.json.
    """

    def __init__(
            self,
            rubric_path: str = "rubric_scoring.json",
            decalogo_path: str = "DECALOGO_FULL.json",
            weights_path: str = "dnp-standards.latest.clean.json"
    ) -> None:
        self.rubric_path = pathlib.Path(rubric_path)
        self.decalogo_path = pathlib.Path(decalogo_path)
        self.weights_path = pathlib.Path(weights_path)

        self.rubric_config = self._load_json_config(self.rubric_path, "Rúbrica de Scoring")
        self.decalogo_config = self._load_json_config(self.decalogo_path, "Decálogo de Preguntas")
        self.weights_config = self._load_json_config(self.weights_path, "DNP Standards Weights")

        # Load core configuration from RUBRIC_SCORING.json (single source of truth)
        self.score_bands = self.rubric_config.get("score_bands", {})
        self.dimensions_meta = self.rubric_config.get("dimensions", {})
        
        # Parse 'questions' section for scoring modalities and expected elements
        self.question_templates = self._parse_question_templates()
        
        # Parse 'weights' section for per-question weight mappings
        self.weights = self._parse_weights()
        
        # Organize DECALOGO questions by unique ID
        self.questions_by_unique_id = self._organize_decalogo_questions()
        
        # Load weights from dnp-standards for dimension/point calculations
        self.weights_lookup = self._load_and_validate_weights()
        
        # Validate strict 1:1 alignment between questions and weights (GATE #5)
        self._validate_rubric_coverage()

        LOGGER.info(
            f"✅ AnswerAssembler inicializado con {len(self.question_templates)} plantillas, {len(self.questions_by_unique_id)} preguntas únicas, {len(self.weights)} rubric weights y {len(self.weights_lookup)} pesos cargados.")

    def _load_json_config(self, path: pathlib.Path, config_name: str) -> Dict[str, Any]:
        if not path.exists():
            LOGGER.critical(f"❌ CONFIGURACIÓN CRÍTICA FALTANTE: No se encontró '{config_name}' en {path}.")
            raise FileNotFoundError(
                f"El archivo de configuración '{config_name}' es requerido y no se encontró en {path}.")

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            LOGGER.critical(f"❌ Error fatal de formato en '{config_name}': {e}")
            raise

    def _organize_decalogo_questions(self) -> Dict[str, Dict[str, Any]]:
        questions_map = {}
        for q in self.decalogo_config.get("questions", []):
            unique_id = f"{q['id']}-{q['point_code']}"
            questions_map[unique_id] = q
        return questions_map

    def _parse_question_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse 'questions' section from RUBRIC_SCORING.json for scoring modalities and expected elements.
        
        Returns a dictionary mapping question template IDs to their configuration.
        
        Raises:
            ValueError: If 'questions' key is missing from rubric (GATE #5).
        """
        if "questions" not in self.rubric_config:
            raise ValueError(
                f"GATE #5 FAILED: 'questions' key missing from rubric at {self.rubric_path}. "
                "Rubric must contain questions section with scoring modalities and expected elements."
            )
        
        questions = self.rubric_config.get("questions", [])
        templates = {}
        for q in questions:
            q_id = q.get('id')
            if q_id:
                templates[q_id] = {
                    'id': q_id,
                    'scoring_modality': q.get('scoring_modality', 'UNKNOWN'),
                    'expected_elements': q.get('expected_elements', []),
                    'max_score': q.get('max_score', 3.0),
                    'dimension': q.get('dimension', 'UNKNOWN'),
                    'question_no': q.get('question_no', 0)
                }
        
        LOGGER.info(f"✓ Parsed {len(templates)} question templates from rubric")
        return templates

    def _load_rubric_config(self) -> Dict[str, float]:
        """
        Load all scoring configuration from RUBRIC_SCORING.json 'weights' section.
        This is the single source of truth for per-question weight mappings.
        
        Returns the weights dictionary mapping question unique IDs to their weights.
        
        Raises:
            ValueError: If 'weights' key is missing from rubric (GATE #5).
        """
        if "weights" not in self.rubric_config:
            raise ValueError(
                f"GATE #5 FAILED: 'weights' key missing from rubric at {self.rubric_path}. "
                "Rubric must contain weights section for all 300 questions."
            )
        
        weights = self.rubric_config.get("weights", {})
        LOGGER.info(f"✓ Loaded rubric weights config from RUBRIC_SCORING.json: {len(weights)} weight entries")
        return weights
    
    def _parse_weights(self) -> Dict[str, float]:
        """
        Parse 'weights' section from RUBRIC_SCORING.json for per-question weight mappings.
        Delegates to _load_rubric_config() for actual loading.
        
        Returns the weights dictionary mapping question unique IDs to their weights.
        """
        return self._load_rubric_config()

    def _validate_rubric_coverage(self) -> None:
        """
        Validate strict 1:1 alignment between questions and weights.
        
        Ensures every question in DECALOGO_FULL.json has a corresponding weight
        in RUBRIC_SCORING.json, and vice versa (no extra weights).
        
        Raises:
            ValueError: If there's a mismatch between questions and weights.
        """
        question_ids = set(self.questions_by_unique_id.keys())
        weight_ids = set(self.weights.keys())
        
        missing_weights = question_ids - weight_ids
        extra_weights = weight_ids - question_ids
        
        if missing_weights or extra_weights:
            error_parts = []
            if missing_weights:
                sample = sorted(missing_weights)[:10]
                error_parts.append(
                    f"Missing weights for {len(missing_weights)} questions: {sample}"
                    + (" ..." if len(missing_weights) > 10 else "")
                )
            if extra_weights:
                sample = sorted(extra_weights)[:10]
                error_parts.append(
                    f"Extra weights for {len(extra_weights)} non-existent questions: {sample}"
                    + (" ..." if len(extra_weights) > 10 else "")
                )
            
            raise ValueError(
                f"GATE #5 FAILED: Rubric weight coverage mismatch. " + "; ".join(error_parts)
            )
        
        LOGGER.info(
            f"✓ Rubric validated (gate #5): {len(question_ids)}/{len(question_ids)} questions with weights"
        )

    def _load_and_validate_weights(self) -> Dict[str, float]:
        dimension_mapping = self.weights_config.get("decalogo_dimension_mapping", {})
        
        weights_lookup = {}
        for point_code, point_weights in dimension_mapping.items():
            for dim_id in ["D1", "D2", "D3", "D4", "D5", "D6"]:
                weight_key = f"{dim_id}_weight"
                if weight_key not in point_weights:
                    raise ValueError(
                        f"❌ Missing weight for {dim_id} in point {point_code}. "
                        f"Each point must have weights for all 6 dimensions."
                    )
                weight_value = point_weights[weight_key]
                weights_lookup[f"{point_code}_{dim_id}"] = weight_value
        
        for question_unique_id in self.questions_by_unique_id.keys():
            parts = question_unique_id.split("-")
            if len(parts) >= 3:
                dim_id = parts[0]
                point_code = parts[2]
                weight_key = f"{point_code}_{dim_id}"
                if weight_key not in weights_lookup:
                    raise ValueError(
                        f"❌ Weight constraint violation: Question '{question_unique_id}' "
                        f"requires weight '{weight_key}' which is not defined in weights config. "
                        f"Every question must have exactly one corresponding weight entry."
                    )
        
        for weight_key in weights_lookup.keys():
            point_code, dim_id = weight_key.split("_")
            matching_questions = [
                qid for qid in self.questions_by_unique_id.keys()
                if qid.startswith(f"{dim_id}-") and qid.endswith(f"-{point_code}")
            ]
            if not matching_questions:
                LOGGER.warning(
                    f"⚠️ Weight entry '{weight_key}' has no corresponding questions in DECALOGO. "
                    f"This weight will not be used."
                )
        
        LOGGER.info(f"✅ Weight validation passed: {len(weights_lookup)} weights loaded and validated.")
        return weights_lookup

    def assemble(
            self,
            evidence_registry: EvidenceRegistry,
            evaluation_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        LOGGER.info("🔧 Ensamblando reporte completo, alineado con la rúbrica oficial...")

        all_question_answers = []
        for result in evaluation_results.get("question_scores", []):
            q_unique_id = result.get("question_unique_id")
            score = result.get("score")

            if q_unique_id and score is not None:
                answer = self._assemble_single_answer(q_unique_id, evidence_registry, score)
                all_question_answers.append(answer)

        LOGGER.info(f"✅ Procesadas {len(all_question_answers)} respuestas individuales.")

        # --- Agregación Jerárquica ---
        questions_by_point_dim = defaultdict(list)
        for qa in all_question_answers:
            key = (qa.point_code, qa.dimension)
            questions_by_point_dim[key].append(qa)

        dimension_summaries_by_point = defaultdict(list)
        for (point_code, dim_id), questions in questions_by_point_dim.items():
            if len(questions) == 5:
                dim_summary = self._aggregate_to_dimension(questions)
                dimension_summaries_by_point[point_code].append(dim_summary)

        point_summaries = []
        for point_code, dim_summaries in dimension_summaries_by_point.items():
            if len(dim_summaries) == 6:
                point_summary = self._aggregate_to_point(dim_summaries)
                point_summaries.append(point_summary)

        global_summary = self._aggregate_to_global(point_summaries, all_question_answers)

        LOGGER.info(f"✅ Ensamblaje completado: {global_summary.global_percentage:.1f}% - {global_summary.score_band}")

        return {
            "metadata": {
                "system_version": "3.0",
                "evaluation_timestamp": datetime.now().isoformat(),
                "rubric_file": str(self.rubric_path),
                "decalogo_file": str(self.decalogo_path)
            },
            "global_summary": asdict(global_summary),
            "point_summaries": [asdict(ps) for ps in sorted(point_summaries, key=lambda x: x.point_code)],
            "question_answers": [qa.to_dict() for qa in sorted(all_question_answers, key=lambda x: x.question_id)]
        }

    def _assemble_single_answer(
            self,
            question_unique_id: str,
            evidence_registry: EvidenceRegistry,
            raw_score: float
    ) -> QuestionAnswer:
        # Get question metadata from DECALOGO
        q_meta = self.questions_by_unique_id.get(question_unique_id, {})
        template_id = q_meta.get('id', 'UNKNOWN')
        
        # Get scoring configuration from RUBRIC_SCORING.json (single source of truth)
        q_template = self.question_templates.get(template_id, {})
        scoring_modality = q_template.get("scoring_modality", "UNKNOWN")
        max_score = q_template.get("max_score", 3.0)
        
        # Look up rubric weight from weights dictionary (RUBRIC_SCORING.json)
        rubric_weight = self.weights.get(question_unique_id)
        if rubric_weight is None:
            raise ValueError(
                f"VALIDATION ERROR: Question '{question_unique_id}' lacks weight entry in RUBRIC_SCORING.json. "
                f"All questions must have corresponding weight mappings."
            )

        evidences = evidence_registry.get_evidence_for_question(question_unique_id)
        ev_metrics, quality = self._analyze_evidence_quality(evidences)
        confidence = self._calculate_confidence_bayesian(evidences)

        rationale = self._generate_rationale(raw_score, evidences, quality, scoring_modality)
        warnings = self._identify_warnings(raw_score, evidences, quality)

        return QuestionAnswer(
            question_id=question_unique_id,
            dimension=q_meta.get("dimension", "N/A"),
            point_code=q_meta.get("point_code", "N/A"),
            question_number=q_meta.get("question_no", 0),
            raw_score=round(raw_score, 2),
            max_score=max_score,
            score_percentage=round((raw_score / max_score) * 100, 1),
            scoring_modality=scoring_modality,
            rubric_weight=rubric_weight,
            evidence_ids=[ev.metadata.get("evidence_id", "N/A") for ev in evidences],
            evidence_count=len(evidences),
            evidence_metrics=ev_metrics,
            confidence=round(confidence, 4),
            quality_assessment=quality.value,
            rationale=rationale,
            warnings=warnings
        )

    def _analyze_evidence_quality(self, evidences: List[MockEvidence]) -> Tuple[EvidenceMetrics, EvidenceQuality]:
        if not evidences:
            return EvidenceMetrics(0, 0.0, 0.0, 0.0, {}, 0), EvidenceQuality.INSUFICIENTE

        confidences = [ev.confidence for ev in evidences]
        quality_dist = defaultdict(int)
        for conf in confidences:
            if conf >= 0.9:
                quality_dist["excelente"] += 1
            elif conf >= 0.75:
                quality_dist["buena"] += 1
            else:
                quality_dist["aceptable"] += 1

        sources = {ev.metadata.get("source_page") for ev in evidences}

        metrics = EvidenceMetrics(
            total_evidences=len(evidences),
            avg_confidence=float(np.mean(confidences)),
            max_confidence=float(np.max(confidences)),
            min_confidence=float(np.min(confidences)),
            quality_distribution=dict(quality_dist),
            sources_diversity=len(sources)
        )

        avg_conf = metrics.avg_confidence
        if avg_conf >= 0.85 and len(evidences) >= 3:
            quality = EvidenceQuality.EXCELENTE
        elif avg_conf >= 0.70 and len(evidences) >= 2:
            quality = EvidenceQuality.BUENA
        elif avg_conf >= 0.55:
            quality = EvidenceQuality.ACEPTABLE
        else:
            quality = EvidenceQuality.DEBIL

        return metrics, quality

    def _calculate_confidence_bayesian(self, evidences: List[MockEvidence], prior: float = 0.5) -> float:
        if not evidences: return 0.0
        posterior = prior
        for ev in evidences:
            likelihood = ev.confidence
            numerator = likelihood * posterior
            denominator = numerator + (1 - likelihood) * (1 - posterior)
            posterior = numerator / denominator if denominator > 0 else posterior
        return posterior

    def _generate_rationale(self, score: float, evidences: List[MockEvidence], quality: EvidenceQuality,
                            modality: str) -> str:
        if not evidences:
            return f"Sin evidencia trazable. Puntaje asignado: {score:.1f}/3.0. Modalidad: {modality}."

        level = "sólida" if score >= 2.0 else "aceptable" if score >= 1.0 else "débil"
        return (f"Puntaje {score:.1f}/3.0 basado en {len(evidences)} pieza(s) de evidencia de calidad {quality.value}. "
                f"La evidencia se considera {level}. Modalidad de scoring: {modality}.")

    def _identify_warnings(self, score: float, evidences: List[MockEvidence], quality: EvidenceQuality) -> List[str]:
        warnings = []
        if not evidences and score > 0:
            warnings.append("INCONSISTENCIA: Puntaje asignado sin evidencia registrada.")
        if quality in [EvidenceQuality.DEBIL, EvidenceQuality.INSUFICIENTE] and score >= 1.5:
            warnings.append(
                f"ADVERTENCIA: Puntaje relativamente alto ({score:.1f}) con evidencia de calidad baja ({quality.value}).")
        return warnings

    def _aggregate_to_dimension(self, question_answers: List[QuestionAnswer]) -> DimensionSummary:
        dim_meta = self.dimensions_meta.get(question_answers[0].dimension, {})
        scores = [qa.raw_score for qa in question_answers]
        total_score = sum(scores)
        
        # Calculate max_score from rubric configuration
        max_score_per_question = dim_meta.get("max_score", 15) / len(question_answers) if question_answers else 3.0
        max_score = max_score_per_question * len(question_answers)

        return DimensionSummary(
            dimension_id=question_answers[0].dimension,
            dimension_name=dim_meta.get("name", "Dimensión Desconocida"),
            point_code=question_answers[0].point_code,
            question_scores=scores,
            total_score=round(total_score, 2),
            max_score=max_score,
            percentage=round((total_score / max_score) * 100, 1) if max_score > 0 else 0.0,
            total_evidences=sum(qa.evidence_count for qa in question_answers),
            avg_confidence=round(np.mean([qa.confidence for qa in question_answers if qa.evidence_count > 0] or [0]),
                                 4),
            strengths=[f"Pregunta {qa.question_number} con puntaje alto ({qa.raw_score})" for qa in question_answers if
                       qa.raw_score >= 2.5],
            weaknesses=[f"Pregunta {qa.question_number} con puntaje bajo ({qa.raw_score})" for qa in question_answers if
                        qa.raw_score < 1.0]
        )

    def _aggregate_to_point(self, dimension_summaries: List[DimensionSummary]) -> PointSummary:
        point_code = dimension_summaries[0].point_code
        point_title = next(
            (q['point_title'] for q in self.decalogo_config['questions'] if q['point_code'] == point_code),
            "Título Desconocido")

        # Use weighted aggregation from dnp-standards configuration
        point_weights = self.weights_config.get("decalogo_dimension_mapping", {}).get(point_code, {})
        
        weighted_percentages = []
        for ds in dimension_summaries:
            weight_key = f"{ds.dimension_id}_weight"
            weight = point_weights.get(weight_key, 1.0 / len(dimension_summaries))
            weighted_percentages.append(ds.percentage * weight)
        
        weighted_avg_percentage = sum(weighted_percentages)
        
        # Calculate max_score from aggregation (sum of dimension max_scores)
        max_score = sum(ds.max_score for ds in dimension_summaries)

        return PointSummary(
            point_code=point_code,
            point_title=point_title,
            dimension_summaries=dimension_summaries,
            average_percentage=round(weighted_avg_percentage, 1),
            total_score=round(sum(ds.total_score for ds in dimension_summaries), 2),
            max_score=max_score,
            critical_gaps=[f"Dimensión {ds.dimension_id} en estado crítico ({ds.percentage:.1f}%)" for ds in
                           dimension_summaries if ds.percentage < 55.0]
        )

    def _aggregate_to_global(self, point_summaries: List[PointSummary], all_qas: List[QuestionAnswer]) -> GlobalSummary:
        applicable_points = [ps for ps in point_summaries if ps.is_applicable]
        global_pct = np.mean([ps.average_percentage for ps in applicable_points] or [0])

        score_band_name, emoji = self._get_score_band(global_pct)
        band_info = self.score_bands.get(score_band_name, {})

        quality_dist = defaultdict(int)
        for qa in all_qas:
            quality_dist[qa.quality_assessment] += 1

        return GlobalSummary(
            global_percentage=round(global_pct, 1),
            score_band=score_band_name,
            score_band_emoji=emoji,
            total_questions=300,
            answered_questions=len(all_qas),
            questions_with_evidence=sum(1 for qa in all_qas if qa.evidence_count > 0),
            applicable_points=len(applicable_points),
            total_points=10,
            total_evidences=sum(qa.evidence_count for qa in all_qas),
            avg_confidence=round(np.mean([qa.confidence for qa in all_qas if qa.evidence_count > 0] or [0]), 4),
            evidence_quality_distribution=dict(quality_dist),
            score_band_description=band_info.get("description", ""),
            main_recommendation=band_info.get("recommendation", ""),
            critical_findings=[
                f"Punto '{ps.point_title}' ({ps.point_code}) con bajo desempeño ({ps.average_percentage:.1f}%)" for ps
                in point_summaries if ps.average_percentage < 55.0]
        )

    def _get_score_band(self, percentage: float) -> Tuple[str, str]:
        for band_name, band_data in self.score_bands.items():
            if band_data.get("min", 0) <= percentage <= band_data.get("max", 100):
                return band_name, band_data.get("emoji", "⚪")
        return "DEFICIENTE", "❌"

    def save_report_json(self, report: Dict[str, Any], output_dir: str = "artifacts") -> pathlib.Path:
        """Guarda el reporte completo en formato JSON."""
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "answers_report.json"

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"📄 Reporte JSON guardado en: {file_path}")
        return file_path


# ==================== BLOQUE DE EJECUCIÓN PARA PRUEBAS ====================

if __name__ == "__main__":
    LOGGER.info("🧪 Ejecutando auto-test de AnswerAssembler...")

    # --- Verificación de archivos de configuración ---
    try:
        assembler = AnswerAssembler(
            rubric_path="rubric_scoring.json",
            decalogo_path="DECALOGO_FULL.json",
            weights_path="dnp-standards.latest.clean.json"
        )
    except FileNotFoundError as e:
        LOGGER.error(f"Error Crítico: No se pudo inicializar el ensamblador. {e}")
        sys.exit(1)

    # --- Simulación de Entradas del Pipeline ---
    # 1. Simular un registro de evidencia
    mock_evidences = [
        MockEvidence(confidence=0.95,
                     metadata={"evidence_id": "ev-001", "question_unique_id": "D1-Q1-P1", "source_page": 10}),
        MockEvidence(confidence=0.88,
                     metadata={"evidence_id": "ev-002", "question_unique_id": "D1-Q1-P1", "source_page": 12}),
        MockEvidence(confidence=0.65,
                     metadata={"evidence_id": "ev-003", "question_unique_id": "D1-Q2-P1", "source_page": 15}),
        # Pregunta sin evidencia
        # D1-Q3-P1
        MockEvidence(confidence=0.40,
                     metadata={"evidence_id": "ev-004", "question_unique_id": "D1-Q4-P1", "source_page": 20}),
    ]
    mock_evidence_registry = EvidenceRegistry(mock_evidences)

    # 2. Simular resultados de la evaluación (una lista plana de scores por pregunta única)
    mock_evaluation_results = {
        "question_scores": [
            {"question_unique_id": "D1-Q1-P1", "score": 3.0},
            {"question_unique_id": "D1-Q2-P1", "score": 1.5},
            {"question_unique_id": "D1-Q3-P1", "score": 0.0},  # Score 0 por falta de evidencia
            {"question_unique_id": "D1-Q4-P1", "score": 0.75},
            {"question_unique_id": "D1-Q5-P1", "score": 2.25},
            # ... se necesitarían las 300 para un reporte completo
            # Para este test, solo una dimensión de un punto está parcialmente completa.
        ]
    }

    LOGGER.info("--- Entradas Simuladas Creadas ---")

    # --- Ejecución del Ensamblador ---
    try:
        final_report = assembler.assemble(
            evidence_registry=mock_evidence_registry,
            evaluation_results=mock_evaluation_results
        )

        # Guardar el reporte de prueba
        report_path = assembler.save_report_json(final_report, output_dir="artifacts_test")

        # Imprimir un resumen para verificación
        print("\n--- RESUMEN DEL REPORTE GENERADO ---")
        global_summary = final_report.get("global_summary", {})
        print(f"Puntaje Global: {global_summary.get('global_percentage')}%")
        print(f"Banda de Calificación: {global_summary.get('score_band')} {global_summary.get('score_band_emoji')}")
        print(f"Total de Respuestas Ensambladas: {len(final_report.get('question_answers', []))}")
        print(f"Reporte de prueba guardado en: {report_path}")

    except Exception as e:
        LOGGER.error(f"❌ Falló la ejecución del ensamblaje durante el test: {e}", exc_info=True)

    LOGGER.info("✅ Auto-test de AnswerAssembler completado.")