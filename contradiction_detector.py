"""
Contradiction Detection Module

Detecta contradicciones en texto en español buscando conectores adversativos
cercanos a indicadores de metas, verbos de acción y objetivos cuantitativos.

Esta versión refactorizada mejora:
- Limpieza y corrección de encabezados inválidos
- Tipado estático y docstrings completos
- Compilación y organización de patrones regex
- Cálculo consistente de posiciones y contexto
- Registro (logging) configurado y uso seguro de valores opcionales
- Interfaz simple para integrarse en pipelines o usarse desde CLI

"""

from __future__ import annotations

import inspect
import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium-high"
    HIGH = "high"


@dataclass(frozen=True)
class ContradictionMatch:
    adversative_connector: str
    goal_keywords: List[str]
    action_verbs: List[str]
    quantitative_targets: List[str]
    full_text: str
    start_pos: int  # absolute position in the original text
    end_pos: int
    risk_level: RiskLevel
    confidence: float
    context_window: str


@dataclass
class ContradictionAnalysis:
    contradictions: List[ContradictionMatch]
    total_contradictions: int
    risk_score: float
    risk_level: RiskLevel
    highest_confidence_contradiction: Optional[ContradictionMatch]
    summary: Dict[str, int]


class ContradictionDetector:
    """
    Industrial-grade contradiction detector with provenance tracking.

    **NORMALIZED OUTPUTS**: Strict schema with confidence scores.
    **EVIDENCE REGISTRY**: Auto-registration with question mapping.
    """

    DEFAULT_CONTEXT_WINDOW = 150

    def __init__(self, *args, **kwargs) -> None:
        if args:
            raise TypeError(
                "ContradictionDetector does not accept positional arguments"
            )

        self.context_window = int(
            kwargs.pop("context_window", self.DEFAULT_CONTEXT_WINDOW)
        )
        self.evidence_registry = kwargs.pop("evidence_registry", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        # Definir patrones relevantes (español).
        # Se evitan patrones demasiado genéricos y se normaliza el texto antes de buscar.
        adversative = [
            r"sin\s+embargo,?",  # sin embargo / sin embargo,
            r"aunque",
            r"pero",
            r"no\s+obstante,?",
            r"a\s+pesar\s+de",
            r"empero",
            r"mientras\s+que",
        ]

        goals = [
            r"\bmeta(s)?\b",
            r"\bobjetivo(s)?\b",
            r"\bpropósitos?\b",
            r"\bfinalidad(es)?\b",
            r"\bresultado\s+esperado\b",
            r"\b(lograr|alcanzar|conseguir|obtener)\b",
            r"\b(pretende|pretender|busca|buscar)\b",
            r"\b(se\s+espera|se\s+proyecta)\b",
            r"\baspiraci[oó]n(es)?\b",
            r"\b(targets?|goals?)\b",
        ]

        actions = [
            r"\b(implementar|ejecutar|desarrollar|realizar)\b",
            r"\b(establecer|crear|construir|formar)\b",
            r"\b(promover|fomentar|impulsar|fortalecer)\b",
            r"\b(mejorar|optimizar|incrementar|aumentar)\b",
            r"\b(reducir|disminuir|minimizar|controlar)\b",
            r"\b(garantizar|asegurar|proteger|defender)\b",
            r"\b(coordinar|articular|gestionar|liderar)\b",
            r"\b(monitorear|supervisar|evaluar|verificar)\b",
            r"\b(capacitar|formar|educar|sensibilizar)\b",
            r"\b(prevenir|atender|resolver|solucionar)\b",
        ]

        quantitative = [
            # porcentajes: 95%, 95 % o '95 por ciento'
            r"\d+(?:[.,]\d+)?\s*(?:%|por\s+ciento|porciento)\b",
            # cantidades con multiplicadores comunes
            r"\d+(?:[.,]\d+)?\s*(?:millones?|mil|thousands?|millions?)\b",
            # verbos con número: 'aumentar en 10', 'reducir 5'
            r"\b(?:incrementar|aumentar|reducir|disminuir)\s+(?:en\s+|hasta\s+)?\d+(?:[.,]\d+)?\b",
            # monedas
            r"\d+(?:[.,]\d+)?\s*(?:COP|\$|USD|pesos|d[oó]lares?)\b",
            r"\bmeta\s+de\s+\d+(?:[.,]\d+)?\b",
            r"\bhasta\s+el\s+\d+(?:[.,]\d+)?\b",
            r"\d+\s*(?:personas|beneficiarios|familias|hogares)\b",
            r"\d+\s*(?:hect[aá]reas|km\u00b2|metros)\b",
            r"\d+(?:[.,]\d+)?\s*(?:puntos|bases)\b",
        ]

        # Compilar patrones para rendimiento y uso consistente
        flags = re.IGNORECASE | re.UNICODE
        self.compiled_adversative = [re.compile(p, flags) for p in adversative]
        self.compiled_goals = [re.compile(p, flags) for p in goals]
        self.compiled_actions = [re.compile(p, flags) for p in actions]
        self.compiled_quantitative = [re.compile(p, flags) for p in quantitative]

    def attach_evidence_registry(self, evidence_registry) -> None:
        """Attach an evidence registry instance after construction."""
        self.evidence_registry = evidence_registry

    def set_context_window(self, context_window: int) -> None:
        """Update the context window length in characters."""
        self.context_window = int(context_window)

    def configure(
        self, *, context_window: Optional[int] = None, evidence_registry=None
    ) -> None:
        """Convenience helper to update runtime configuration."""
        if context_window is not None:
            self.set_context_window(context_window)
        if evidence_registry is not None:
            self.attach_evidence_registry(evidence_registry)

    __signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normaliza texto usando NFKC y eliminando espacios duplicados."""
        if text is None:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        # colapsar espacios múltiples
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _find_pattern_matches(
        text: str, patterns: List[Pattern], _tag: Optional[str] = None
    ) -> List[Tuple[str, int, int]]:
        """Devuelve lista de (matched_text, start, end) para los patrones dados."""
        matches = []
        for pat in patterns:
            for m in pat.finditer(text):
                matches.append((m.group(0), m.start(), m.end()))
        return matches

    # Compatibilidad con versiones previas de la API de tests
    def _extract_context_window(self, text: str, center_abs_pos: int) -> str:
        """Wrapper antiguo que devolvía el texto de la ventana de contexto; preserva compatibilidad.

        Retorna únicamente la cadena de contexto (posibles usos antiguos en tests).
        """
        context_text, _ = self._extract_context(text, center_abs_pos)
        return context_text

    def _extract_context(self, text: str, center_abs_pos: int) -> Tuple[str, int]:
        """Extrae ventana de contexto alrededor de una posición absoluta.

        Retorna (context_text, context_start_absolute_pos).
        """
        half = self.context_window // 2
        start = max(0, center_abs_pos - half)
        end = min(len(text), center_abs_pos + half)
        return text[start:end].strip(), start

    @staticmethod
    def _calculate_contradiction_confidence(
        adversative_pos_in_context: int,
        goal_matches: List[Tuple[str, int, int]],
        action_matches: List[Tuple[str, int, int]],
        quantitative_matches: List[Tuple[str, int, int]],
    ) -> float:
        """Calcula una puntuación de confianza entre 0 y 1 basada en proximidad y presencia.

        adversative_pos_in_context: posición (índice) dentro de la ventana de contexto.
        """
        confidence = 0.0
        confidence += 0.3  # base por tener conector adversativo

        # Evitar división por cero y manejar distancias
        for matches, weight in (
            (goal_matches, 0.4),
            (action_matches, 0.2),
            (quantitative_matches, 0.3),
        ):
            if matches:
                # distancia mínima desde el inicio de cada match a la posición adversativa
                min_distance = min(
                    abs(m[1] - adversative_pos_in_context) for m in matches
                )
                # proximidad con decaimiento exponencial (50 chars ~ 0.5)
                proximity_score = weight * (0.5 ** (min_distance / 50))
                confidence += proximity_score

        # En lugar de contar ocurrencias, contamos tipos presentes
        types_present = (
            int(bool(goal_matches))
            + int(bool(action_matches))
            + int(bool(quantitative_matches))
        )
        confidence += types_present * 0.1

        return min(1.0, confidence)

    @staticmethod
    def _determine_risk_level(confidence: float, context_complexity: int) -> RiskLevel:
        """Devuelve un RiskLevel ajustado por complejidad contextual (número de matches)."""
        adjusted_confidence = confidence * (1 + context_complexity * 0.05)
        if adjusted_confidence >= 0.8:
            return RiskLevel.HIGH
        if adjusted_confidence >= 0.6:
            return RiskLevel.MEDIUM_HIGH
        if adjusted_confidence >= 0.4:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def detect_contradictions(self, text: str) -> ContradictionAnalysis:
        """Analiza `text` y devuelve un objeto `ContradictionAnalysis` con los hallazgos."""
        normalized = self._normalize_text(text)
        if not normalized:
            return ContradictionAnalysis(
                [],
                0,
                0.0,
                RiskLevel.LOW,
                None,
                {"low": 0, "medium": 0, "medium-high": 0, "high": 0},
            )

        # Buscar conectores adversativos en el texto completo (posiciones absolutas)
        adversative_matches = self._find_pattern_matches(
            normalized, self.compiled_adversative
        )
        if not adversative_matches:
            return ContradictionAnalysis(
                [],
                0,
                0.0,
                RiskLevel.LOW,
                None,
                {"low": 0, "medium": 0, "medium-high": 0, "high": 0},
            )

        contradictions = []

        for adv_text, adv_start, adv_end in adversative_matches:
            # Extraer ventana de contexto alrededor del inicio del conector
            context_text, context_start = self._extract_context(normalized, adv_start)
            # Buscar patrones dentro de la ventana de contexto; las posiciones devueltas
            # son relativas a la ventana (0..len(context_text)) si queremos compararlas
            # con la posición del adversativo en esa ventana, debemos ajustar.
            goal_matches = self._find_pattern_matches(context_text, self.compiled_goals)
            action_matches = self._find_pattern_matches(
                context_text, self.compiled_actions
            )
            quantitative_matches = self._find_pattern_matches(
                context_text, self.compiled_quantitative
            )

            if not (goal_matches or action_matches or quantitative_matches):
                # No hay evidencia de contradicción en la ventana
                continue

            # Posición del adversativo dentro de la ventana de contexto
            adv_pos_in_context = adv_start - context_start

            # Calcular confianza y nivel de riesgo
            confidence = self._calculate_contradiction_confidence(
                adv_pos_in_context, goal_matches, action_matches, quantitative_matches
            )
            context_complexity = (
                len(goal_matches) + len(action_matches) + len(quantitative_matches)
            )
            risk_level = self._determine_risk_level(confidence, context_complexity)

            contradiction = ContradictionMatch(
                adversative_connector=adv_text,
                goal_keywords=[m[0] for m in goal_matches],
                action_verbs=[m[0] for m in action_matches],
                quantitative_targets=[m[0] for m in quantitative_matches],
                full_text=context_text,
                start_pos=adv_start,
                end_pos=adv_end,
                risk_level=risk_level,
                confidence=confidence,
                context_window=context_text,
            )

            contradictions.append(contradiction)

        # Agregar cálculos agregados
        if contradictions:
            risk_score = sum(c.confidence for c in contradictions) / len(contradictions)
            highest_conf = max(contradictions, key=lambda c: c.confidence)
            overall_risk = self._determine_risk_level(risk_score, len(contradictions))
        else:
            risk_score = 0.0
            highest_conf = None
            overall_risk = RiskLevel.LOW

        summary = {"low": 0, "medium": 0, "medium-high": 0, "high": 0}
        for c in contradictions:
            if c.risk_level == RiskLevel.LOW:
                summary["low"] += 1
            elif c.risk_level == RiskLevel.MEDIUM:
                summary["medium"] += 1
            elif c.risk_level == RiskLevel.MEDIUM_HIGH:
                summary["medium-high"] += 1
            elif c.risk_level == RiskLevel.HIGH:
                summary["high"] += 1

        return ContradictionAnalysis(
            contradictions=contradictions,
            total_contradictions=len(contradictions),
            risk_score=risk_score,
            risk_level=overall_risk,
            highest_confidence_contradiction=highest_conf,
            summary=summary,
        )

    def integrate_with_risk_assessment(
        self, text: str, existing_score: float = 0.0
    ) -> Dict[str, float]:
        """Integra el análisis de contradicciones con una puntuación de riesgo existente.

        Devuelve un diccionario con la puntuación integrada y metadatos.
        """
        analysis = self.detect_contradictions(text)

        contradiction_risk = 0.0
        if analysis.total_contradictions > 0:
            base_risk = min(0.3, analysis.total_contradictions * 0.1)
            confidence_risk = analysis.risk_score * 0.4
            high_count = analysis.summary.get("high", 0)
            medium_high_count = analysis.summary.get("medium-high", 0)
            severity_risk = (high_count * 0.2) + (medium_high_count * 0.15)
            contradiction_risk = min(1.0, base_risk + confidence_risk + severity_risk)

        integrated_score = min(1.0, float(existing_score) + contradiction_risk * 0.3)

        highest_confidence = (
            analysis.highest_confidence_contradiction.confidence
            if analysis.highest_confidence_contradiction
            else 0.0
        )

        return {
            "base_score": float(existing_score),
            "contradiction_risk": contradiction_risk,
            "integrated_score": integrated_score,
            "contradiction_count": analysis.total_contradictions,
            "highest_contradiction_confidence": float(highest_confidence),
        }

    def detect(self, text: str, plan_name: str = "unknown") -> Dict[str, Any]:
        """
        Detect contradictions with NORMALIZED OUTPUT SCHEMA.

        Returns:
            {
                "matches": List[{
                    "statement_1": str,
                    "statement_2": str,
                    "contradiction_type": str,  # RiskLevel.value
                    "confidence": float,  # 0.0-1.0
                    "severity": float,  # 0.0-1.0
                    "context_1": str,
                    "context_2": str,
                    "position_1": int,
                    "position_2": int,
                    "applicable_questions": List[str],
                    "provenance": Dict[str, str]
                }],
                "total_contradictions": int,
                "high_severity_count": int,
                "coherence_score": float  # 0.0-1.0 (1.0 = sin contradicciones)
            }
        """
        matches = []

        # Detectar contradicciones numéricas
        numeric_contradictions = self._detect_numeric_contradictions(text)

        # Detectar contradicciones semánticas
        semantic_contradictions = self._detect_semantic_contradictions(text)

        # Detectar contradicciones de plazos
        temporal_contradictions = self._detect_temporal_contradictions(text)

        all_contradictions = (
            numeric_contradictions + semantic_contradictions + temporal_contradictions
        )

        # Normalizar outputs
        for contradiction in all_contradictions:
            # Calcular confidence DETERMINISTA
            confidence = self._calculate_confidence(contradiction)

            # Calcular severity
            severity = self._calculate_severity(contradiction)

            # Mapear a preguntas aplicables
            applicable_qs = self._map_to_questions(contradiction["type"])

            normalized = {
                "statement_1": contradiction["statement_1"],
                "statement_2": contradiction["statement_2"],
                "contradiction_type": contradiction["type"],
                "confidence": confidence,
                "severity": severity,
                "context_1": contradiction.get("context_1", ""),
                "context_2": contradiction.get("context_2", ""),
                "position_1": contradiction.get("pos_1", 0),
                "position_2": contradiction.get("pos_2", 0),
                "applicable_questions": applicable_qs,
                "provenance": {
                    "plan_name": plan_name,
                    "detector": "contradiction_detector",
                    "detection_method": contradiction.get("method", "pattern"),
                },
            }

            matches.append(normalized)

            # Registrar automáticamente si hay registry
            if self.evidence_registry:
                self.evidence_registry.register(
                    source_component="contradiction_detector",
                    evidence_type="contradiction",
                    content=normalized,
                    confidence=confidence,
                    applicable_questions=applicable_qs,
                )

        # Calcular coherence score
        coherence_score = self._calculate_coherence_score(len(matches), len(text))

        return {
            "matches": matches,
            "total_contradictions": len(matches),
            "high_severity_count": sum(1 for m in matches if m["severity"] > 0.7),
            "coherence_score": coherence_score,
        }

    @staticmethod
    def _calculate_confidence(contradiction: Dict) -> float:
        """Calcular confidence DETERMINISTA"""
        confidence = 0.6  # Base

        # Boost por tipo de contradicción
        if contradiction["type"] == "numeric":
            confidence += 0.2  # Contradicciones numéricas son más confiables
        elif contradiction["type"] == "temporal":
            confidence += 0.15

        # Boost por palabras clave explícitas
        keywords = ["sin embargo", "pero", "aunque", "contradice"]
        if any(k in contradiction.get("context_1", "").lower() for k in keywords):
            confidence += 0.1

        return min(1.0, confidence)

    @staticmethod
    def _calculate_severity(contradiction: Dict) -> float:
        """Calcular severidad de la contradicción"""
        severity = 0.5  # Base

        # Mayor severidad para contradicciones numéricas grandes
        if contradiction["type"] == "numeric":
            # Analizar diferencia numérica si está disponible
            severity += 0.3

        # Mayor severidad si afecta objetivos principales
        if "objetivo" in contradiction.get("context_1", "").lower():
            severity += 0.2

        return min(1.0, severity)

    @staticmethod
    def _map_to_questions(contradiction_type: str) -> List[str]:
        """Mapear contradicción a preguntas del cuestionario"""
        questions = []

        # Contradicciones afectan coherencia (D1)
        questions.extend([f"D1-Q{i}" for i in [1, 3, 5, 8]])

        # También afectan calidad del diagnóstico (D5)
        questions.extend([f"D5-Q{i}" for i in [2, 5, 10]])

        # Si es temporal, afecta planificación (D2)
        if contradiction_type == "temporal":
            questions.extend([f"D2-Q{i}" for i in [5, 10]])

        return questions

    @staticmethod
    def _calculate_coherence_score(num_contradictions: int, text_length: int) -> float:
        """Calcular score de coherencia global (1.0 = sin contradicciones)"""
        if text_length == 0:
            return 0.5

        # Normalizar por longitud del texto
        contradiction_density = num_contradictions / (
            text_length / 1000
        )  # Por cada 1000 chars

        # Score inverso (más contradicciones = menor score)
        coherence = 1.0 - min(1.0, contradiction_density * 0.2)

        return max(0.0, coherence)

    # ...existing code...


if __name__ == "__main__":
    # Ejecución de ejemplo rápida para auto-comprobación.
    logging.basicConfig(level=logging.INFO)
    detector = ContradictionDetector()
    sample = (
        "El objetivo es aumentar la cobertura educativa al 95% para 2027, "
        "sin embargo, los recursos presupuestales han sido reducidos en un 30% este año."
    )
    analysis = detector.detect_contradictions(sample)
    logger.info("Total contradicciones: %s", analysis.total_contradictions)
    for c in analysis.contradictions:
        logger.info(
            f"Connector: {c.adversative_connector} | confidence: {c.confidence:.3f} | risk: {c.risk_level.value}"
        )
