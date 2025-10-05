# coding=utf-8
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

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Pattern, Tuple

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
    """Detector de contradicciones por patrones adversativos y palabras clave.

    Attributes:
        context_window: número de caracteres alrededor del conector adversativo a analizar.
    """

    DEFAULT_CONTEXT_WINDOW = 150

    def __init__(self, context_window: int = DEFAULT_CONTEXT_WINDOW) -> None:
        self.context_window = int(context_window)

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
        self.compiled_quantitative = [
            re.compile(p, flags) for p in quantitative]

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
            context_text, context_start = self._extract_context(
                normalized, adv_start)
            # Buscar patrones dentro de la ventana de contexto; las posiciones devueltas
            # son relativas a la ventana (0..len(context_text)) si queremos compararlas
            # con la posición del adversativo en esa ventana, debemos ajustar.
            goal_matches = self._find_pattern_matches(
                context_text, self.compiled_goals)
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
                len(goal_matches) + len(action_matches) +
                len(quantitative_matches)
            )
            risk_level = self._determine_risk_level(
                confidence, context_complexity)

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
            risk_score = sum(
                c.confidence for c in contradictions) / len(contradictions)
            highest_conf = max(contradictions, key=lambda c: c.confidence)
            overall_risk = self._determine_risk_level(
                risk_score, len(contradictions))
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
            contradiction_risk = min(
                1.0, base_risk + confidence_risk + severity_risk)

        integrated_score = min(1.0, float(
            existing_score) + contradiction_risk * 0.3)

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


if __name__ == "__main__":
    # Ejecución de ejemplo rápida para auto-comprobación.
    logging.basicConfig(level=logging.INFO)
    detector = ContradictionDetector()
    sample = (
        "El objetivo es aumentar la cobertura educativa al 95% para 2027, "
        "sin embargo, los recursos presupuestales han sido reducidos en un 30% este año."
    )
    analysis = detector.detect_contradictions(sample)
    logger.info("Total contradicciones: %d", analysis.total_contradictions)
    for c in analysis.contradictions:
        logger.info(
            "Connector: %s | confidence: %.3f | risk: %s",
            c.adversative_connector,
            c.confidence,
            c.risk_level.value,
        )
