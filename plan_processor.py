# coding=utf-8
# coding=utf-8
"""
Causal Framework Plan Processor Module (Industrial Grade)

Processes planning documents to extract structured evidence aligned with the
DECALOGO Causal Framework Questionnaire. This version is specifically designed
to find evidence related to the causal chain (D1-D6) for each of the 10
defined policy points (P1-P10).

Version: 2.0.0
Date: 2025-10-09
Author: System Architect
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Union

# Use the orchestrator's logger
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalizes a string by converting to lowercase and simplifying whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).lower().strip()


class PlanProcessor:
    """
    Extracts structured information from plan documents, specifically aligned
    with the DECALOGO Causal Framework Questionnaire. This component is the
    primary evidence provider for the QuestionnaireEngine.
    """
    QUESTIONNAIRE_FILENAME = "decalogo-industrial.latest.clean.json"

    def __init__(self, config_dir: Union[str, Path]):
        """
        Initializes the processor by building regex patterns from the questionnaire.

        Args:
            config_dir: The path to the configuration directory containing
                        the questionnaire JSON file.
        """
        self.config_dir = Path(config_dir)
        self.questionnaire = self._load_questionnaire()

        self.point_codes: List[str] = []
        self.point_patterns: Dict[str, re.Pattern] = {}

        # Evidence patterns derived from the questionnaire's dimensions (D1-D6)
        # Using word boundaries (\b) for better precision
        self.evidence_patterns = {
            # D1: Diagnóstico y Recursos
            "diagnostico_lineas_base": r"(?i)\b(l[íi]neas?\s+base|diagn[óo]stico\s+cuantitativo|fuentes?\s+de\s+datos|series?\s+temporales|m[ée]todos?\s+de\s+medici[óo]n)\b",
            "recursos_presupuestales": r"(?i)\b(recursos?\s+del\s+ppi|plan\s+indicativo|trazabilidad\s+program[áa]tica|suficiencia\s+presupuestal|asignaci[óo]n\s+de\s+recursos?)\b",
            "capacidades_institucionales": r"(?i)\b(capacidades?\s+institucionales?|talento\s+humano|gobernanza\s+de\s+datos|cuellos?\s+de\s+botella)\b",
            # D2: Actividades
            "actividades_formalizadas": r"(?i)\b(plan\s+de\s+acci[óo]n|tablas?\s+de\s+actividades?|responsables?|insumos?|outputs?|calendarios?|costos?\s+unitarios?)\b",
            "mecanismo_causal_actividad": r"(?i)\b(mecanismos?\s+causales?|poblaci[óo]n\s+diana|causas?\s+ra[íi]z|eslab[óo]n\s+causal)\b",
            # D3: Productos
            "productos_verificables": r"(?i)\b(indicador(?:es)?\s+de\s+producto|metas?\s+de\s+producto|f[óo]rmulas?\s+del\s+indicador|validaci[óo]n\s+del\s+mecanismo)\b",
            # D4: Resultados
            "resultados_definidos": r"(?i)\b(m[ée]tricas?\s+de\s+outcome|indicador(?:es)?\s+de\s+resultado|ventanas?\s+de\s+maduraci[óo]n|criterios?\s+de\s+[éex]ito)\b",
            # D5: Impacto
            "impacto_largo_plazo": r"(?i)\b(impactos?\s+de\s+largo\s+plazo|rutas?\s+de\s+transmisi[óo]n|indicadores?\s+compuestos?|proxies\s+mensurables?)\b",
            # D6: Lógica Causal Integrada
            "teoria_de_cambio": r"(?i)\b(teor[íi]as?\s+de\s+cambio|diagramas?\s+causales?|supuestos?\s+verificables?|enlaces?\s+causales?|pilotos?|pruebas?\s+de\s+mecanismo)\b",
        }

        self._build_patterns_from_questionnaire()

    def _load_questionnaire(self) -> Dict[str, Any]:
        """Loads the questionnaire from the specified config directory."""
        questionnaire_path = Path(self.QUESTIONNAIRE_FILENAME)
        try:
            with open(questionnaire_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("FATAL: Could not load or parse questionnaire at %s: %s", questionnaire_path, e)
            raise IOError(f"Questionnaire file not found or invalid: {questionnaire_path}") from e

    def _build_patterns_from_questionnaire(self):
        """Dynamically builds regex patterns for each policy point using its title and hints."""
        point_keywords: Dict[str, Set[str]] = defaultdict(set)
        for q in self.questionnaire.get("questions", []):
            point_code = q.get("point_code")
            if not point_code:
                continue

            point_keywords[point_code].add(q.get("point_title", "").lower())
            for hint in q.get("hints", []):
                cleaned_hint = re.sub(r'\(.*?\)', '', hint).strip()
                if cleaned_hint:
                    point_keywords[point_code].add(cleaned_hint.lower())

        self.point_codes = sorted(point_keywords.keys())
        for code, keywords in point_keywords.items():
            # Create a large OR pattern, prioritizing longer phrases first
            # Use word boundaries for precise matching
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            pattern_str = "|".join(r'\b' + re.escape(kw) + r'\b' for kw in sorted_keywords if kw)
            self.point_patterns[code] = re.compile(f"(?i)({pattern_str})")

        logger.info("Built keyword patterns for %s policy points.", len(self.point_patterns))

    def process(self, text: str) -> Dict[str, Any]:
        """
        Processes a plan document to extract evidence structured by policy point.

        Args:
            text: The raw, sanitized text content of the plan document.

        Returns:
            A dictionary ('doc_struct') containing metadata and evidence organized by
            policy point, ready for the QuestionnaireEngine.
        """
        if not text:
            logger.warning("PlanProcessor received empty text.")
            return {"metadata": {}, "point_evidence": {}, "full_text": "", "processing_status": "failed"}

        normalized_text = normalize_text(text)
        point_evidence = {}

        for point_code in self.point_codes:
            evidence = self._extract_point_evidence(normalized_text, point_code)
            if evidence:
                point_evidence[point_code] = evidence

        return {
            "metadata": self._extract_metadata(normalized_text),
            "point_evidence": point_evidence,
            "text_length": len(normalized_text),
            "full_text": normalized_text,
            "processing_status": "complete"
        }

    def _extract_point_evidence(self, text: str, point_code: str) -> Dict[str, List[str]]:
        """
        Extracts all relevant evidence for a single policy point by first finding
        relevant sentences and then searching for evidence patterns within them.
        """
        point_pattern = self.point_patterns.get(point_code)
        if not point_pattern:
            return {}

        # 1. Efficiently find all sentences relevant to this policy point.
        relevant_sentences = []
        # Split text into sentences for more granular context matching.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if point_pattern.search(sentence):
                relevant_sentences.append(sentence)

        contextual_text = " ".join(relevant_sentences)
        if not contextual_text:
            return {}

        # 2. Search for specific evidence types within this highly relevant context.
        point_evidence = defaultdict(set)
        for ev_type, ev_pattern_str in self.evidence_patterns.items():
            ev_pattern = re.compile(ev_pattern_str)
            for sentence in relevant_sentences:
                if ev_pattern.search(sentence):
                    # Add the full sentence as evidence for better context.
                    point_evidence[ev_type].add(sentence.strip())

        return {k: list(v) for k, v in point_evidence.items()}

    @staticmethod
    def _extract_metadata(text: str) -> Dict[str, Any]:
        """Extracts key metadata from the first part of the document."""
        title_match = re.search(r"(?i)^(?:plan\s+de\s+desarrollo\s+)?(.*?)(?:\n|\.|\d{4})", text[:1000])
        title = title_match.group(1).strip() if title_match and len(
            title_match.group(1).strip()) > 5 else "Untitled Plan"

        entity_match = re.search(r"(?i)(?:municipio|alcald[íi]a|gobernaci[óo]n)\s+de\s+([\w\s]+?)(?:\n|\.)",
                                 text[:2000])
        entity = entity_match.group(1).strip() if entity_match else "Unknown Entity"

        date_match = re.search(r"(20\d{2})\s*(?:-|a|al)\s*(20\d{2})", text[:2000])
        date_range = {"start_year": date_match.group(1), "end_year": date_match.group(2)} if date_match else {}

        return {"title": title, "entity": entity, "date_range": date_range}