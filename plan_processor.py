"""
Plan Processor Module

Processes plan documents to extract structured information aligned with
the DECALOGO evaluation framework, ensuring all necessary evidence for
answering evaluation questions is identified.

This module supports the canonical flow by extracting key information from
plan documents that will be analyzed by specialized detectors.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from text_processor import normalize_text, remove_unwanted_characters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load DECALOGO structure
def load_decalogo_structure():
    """Load the DECALOGO_FULL structure to align processing with evaluation questions."""
    try:
        decalogo_path = os.path.join(os.path.dirname(__file__), "DECALOGO_FULL.json")
        with open(decalogo_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load DECALOGO_FULL.json: {e}")
        return {
            "dimensiones": [],
            "mapeo_evidencia_componentes": {},
            "clusters_politica": []  # Add default empty clusters list
        }

DECALOGO = load_decalogo_structure()

class PlanProcessor:
    """
    Processes plan documents to extract structured information aligned
    with DECALOGO evaluation framework.
    
    Attributes:
        section_patterns: Regex patterns for identifying document sections
        evidence_patterns: Patterns for extracting evidence required by DECALOGO questions
        decalogo_structure: Reference to DECALOGO evaluation structure
    """
    
    def __init__(self):
        """Initialize the plan processor with pattern definitions."""
        # Define section patterns
        self.section_patterns = {
            "diagnostico": r"(?i)(diagn[oó]stico|antecedentes|contexto|situaci[oó]n actual)",
            "vision": r"(?i)(visi[oó]n|futuro deseado|prospectiva|escenario futuro)",
            "objetivos": r"(?i)(objetivos?|metas? estrat[eé]gicas?|prop[oó]sitos?)",
            "estrategias": r"(?i)(estrategias?|l[ií]neas? de acci[oó]n|programas?)",
            "indicadores": r"(?i)(indicadores?|mediciones|m[eé]tricas)",
            "presupuesto": r"(?i)(presupuesto|recursos financieros|financiaci[oó]n|inversi[oó]n)",
            "responsables": r"(?i)(responsables?|actores involucrados|entidades ejecutoras)",
            "seguimiento": r"(?i)(seguimiento|monitoreo|evaluaci[oó]n|control)",
            "cronograma": r"(?i)(cronograma|calendario|plazos|tiempos)",
        }
        
        # Define evidence patterns aligned with DECALOGO questions
        self.evidence_patterns = {
            # DE-1: Lógica de Intervención
            "indicadores": r"(?i)(indicador(?:es)?:?|KPI:?|m[eé]trica(?:s)?:?).*?(?:\d+(?:[.,]\d+)?%?|[\w\s]+)",
            "metas": r"(?i)(?:meta(?:s)?:?|objetivo(?:s)? cuantitativo(?:s)?:?).*?(?:\d+(?:[.,]\d+)?%?|[\w\s]+)",
            "lineas_base": r"(?i)(?:l[ií]nea(?:s)? base:?|situaci[oó]n inicial:?).*?(?:\d+(?:[.,]\d+)?%?|[\w\s]+)",
            "responsables": r"(?i)(?:responsable(?:s)?:?|a cargo:?|entidad(?:es)? ejecutora(?:s)?:?).*?([\w\s]+)",
            
            # DE-2: Inclusión Temática
            "objetivos_pnd": r"(?i)(?:alineaci[oó]n(?: con)? (?:el)? PND|relaci[oó]n(?: con)? (?:el)? Plan Nacional).*?([\w\s]+)",
            "ods": r"(?i)(?:ODS|Objetivos? de Desarrollo Sostenible).*?(?:\d{1,2}|[\w\s]+)",
            "grupos_vulnerables": r"(?i)(?:grupos? vulnerables?|poblaci[oó]n(?:es)? vulnerable(?:s)?|enfoque diferencial).*?([\w\s]+)",
            
            # DE-3: Participación y Gobernanza
            "mesas_tecnicas": r"(?i)(?:mesa(?:s)? t[eé]cnica(?:s)?|comit[eé](?:s)? t[eé]cnico(?:s)?|grupo(?:s)? de trabajo).*?([\w\s]+)",
            "dialogos_ciudadanos": r"(?i)(?:di[aá]logo(?:s)? ciudadano(?:s)?|participaci[oó]n ciudadana|consulta(?:s)? p[uú]blica(?:s)?).*?([\w\s]+)",
            "rendicion_cuentas": r"(?i)(?:rendici[oó]n(?:es)? de cuentas?|mecanismo(?:s)? de transparencia|audiencia(?:s)? p[uú]blica(?:s)?).*?([\w\s]+)",
            
            # DE-4: Orientación a Resultados
            "sistema_monitoreo": r"(?i)(?:sistema(?:s)? de (?:monitoreo|seguimiento)|mecanismo(?:s)? de evaluaci[oó]n).*?([\w\s]+)",
            "hitos": r"(?i)(?:hito(?:s)?|punto(?:s)? de control|milestone(?:s)?).*?([\w\s]+)",
            "tablero_control": r"(?i)(?:tablero(?:s)? de (?:control|mando)|dashboard(?:s)?|panel(?:es)? de indicadores).*?([\w\s]+)",
        }
        
        # Add cluster-specific evidence patterns
        self._add_cluster_evidence_patterns()
        
        # Load DECALOGO structure for alignment
        self.decalogo_structure = DECALOGO

    def _add_cluster_evidence_patterns(self):
        """Add evidence patterns specific to clusters in DECALOGO_FULL."""
        # Get clusters from DECALOGO structure
        clusters = self.decalogo_structure.get("clusters_politica", [])
        
        for cluster in clusters:
            cluster_id = cluster.get("id")
            if not cluster_id:
                continue
                
            # Add patterns for cluster-specific areas
            for area in cluster.get("areas_tematicas", []):
                pattern_name = f"area_{cluster_id}_{area}"
                pattern = r"(?i)" + re.escape(area) + r".*?(?:\.|,|\n)"
                self.evidence_patterns[pattern_name] = pattern
            
            # Add patterns for cluster-specific indicators
            for indicator in cluster.get("indicadores_clave", []):
                pattern_name = f"indicador_{cluster_id}_{indicator}"
                pattern = r"(?i)" + re.escape(indicator) + r".*?(?:\d+(?:[.,]\d+)?%?|[\w\s]+)"
                self.evidence_patterns[pattern_name] = pattern
            
            # Add patterns for cluster name mentions
            cluster_name = cluster.get("nombre", "")
            if cluster_name:
                pattern_name = f"nombre_{cluster_id}"
                pattern = r"(?i)" + re.escape(cluster_name) + r".*?(?:\.|,|\n)"
                self.evidence_patterns[pattern_name] = pattern

    def process(self, text: str) -> Dict[str, Any]:
        """
        Process a plan document text and extract structured information
        aligned with the DECALOGO evaluation framework.
        
        Args:
            text: Raw text of the plan document
            
        Returns:
            Dictionary with structured plan information and extracted evidence
        """
        if not text:
            return {"error": "Empty text provided"}
            
        # Normalize text
        normalized_text = normalize_text(text)
        
        # Extract metadata
        metadata = self._extract_metadata(normalized_text)
        
        # Identify document sections
        sections = self._identify_sections(normalized_text)
        
        # Extract evidence for each DECALOGO dimension
        evidence = {}
        
        # Process each dimension's evidence requirements
        for dimension in self.decalogo_structure.get("dimensiones", []):
            dim_id = dimension.get("id")
            for question in dimension.get("preguntas", []):
                required_evidence = question.get("evidencia_requerida", [])
                for evidence_type in required_evidence:
                    matches = self._extract_evidence(normalized_text, evidence_type)
                    if matches:
                        if evidence_type not in evidence:
                            evidence[evidence_type] = matches
                        else:
                            evidence[evidence_type].extend(matches)
        
        # Extract cluster-specific evidence
        cluster_evidence = self._extract_cluster_evidence(normalized_text)
        
        # Extract additional structured elements
        structured_elements = {
            "indicadores_con_metas": self._extract_indicators_with_targets(normalized_text),
            "relaciones_causales": self._extract_causal_relationships(normalized_text),
            "referencias_territoriales": self._extract_territorial_references(normalized_text)
        }
        
        # Create cluster-dimension mapping
        cluster_dimension_mapping = self._create_cluster_dimension_mapping()
        
        return {
            "metadata": metadata,
            "sections": sections,
            "evidence": evidence,
            "cluster_evidence": cluster_evidence,
            "structured_elements": structured_elements,
            "cluster_dimension_mapping": cluster_dimension_mapping,
            "text_length": len(normalized_text),
            "processing_status": "complete"
        }

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract basic metadata from the plan document."""
        metadata = {
            "title": self._extract_title(text),
            "date_range": self._extract_date_range(text),
            "entity": self._extract_entity(text),
            "plan_type": self._identify_plan_type(text),
        }
        return metadata
        
    def _extract_title(self, text: str) -> str:
        """Extract the document title."""
        # Look for common title patterns at the start of the document
        title_patterns = [
            r"(?i)^(?:plan\s+(?:de\s+)?desarrollo\s+)?(.*?)(?:\n|\.|\(|\d{4})",
            r"(?i)(?:^|\n)(?:PLAN\s+DE\s+DESARROLLO\s+)(.*?)(?:\n|\.|$)",
            r"(?:^|\n)[\"\']([^\"\']+?)[\"\'](?:\n|\.|$)",
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text[:1000])
            if match and match.group(1) and len(match.group(1).strip()) > 10:
                return match.group(1).strip()
        
        # Fallback: use first line if it looks like a title
        first_line = text.split("\n")[0].strip()
        if 10 < len(first_line) < 200 and not first_line.endswith("."):
            return first_line
            
        return "Untitled Plan"

    def _extract_date_range(self, text: str) -> Dict[str, str]:
        """Extract plan date range."""
        # Look for year ranges like 2020-2023, 2020 a 2023, etc.
        year_patterns = [
            r"(?i)(?:20\d{2})\s*(?:-|a|hasta|al)\s*(20\d{2})",
            r"(?i)(?:período|periodo|vigencia)(?:\s+\w+){0,5}\s+(20\d{2})\s*(?:-|a|hasta|al)\s*(20\d{2})",
            r"(?i)(?:plan)\s+(?:de\s+)?(?:desarrollo)\s+(?:\w+\s+){0,10}?(20\d{2})\s*(?:-|a|hasta|al)\s*(20\d{2})",
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text[:5000])
            if match:
                if len(match.groups()) == 1:
                    # Only end year captured
                    start_year = str(int(match.group(1)) - 4)  # Assume 4 year period
                    end_year = match.group(1)
                else:
                    # Both years captured
                    start_year = match.group(1)
                    end_year = match.group(2)
                return {"start_year": start_year, "end_year": end_year}
        
        return {"start_year": "Unknown", "end_year": "Unknown"}

    def _extract_entity(self, text: str) -> str:
        """Extract the responsible government entity."""
        entity_patterns = [
            r"(?i)(?:municipio|alcaldía|gobernación|departamento|distrito)\s+de\s+([\w\s]+?)(?:\n|\.|,)",
            r"(?i)(?:alcaldía|gobernación|administración)\s+([\w\s]+?)(?:\n|\.|,)",
        ]
        
        for pattern in entity_patterns:
            match = re.search(pattern, text[:5000])
            if match and match.group(1):
                return match.group(1).strip()
        
        return "Unknown Entity"

    def _identify_plan_type(self, text: str) -> str:
        """Identify the type of plan."""
        if re.search(r"(?i)plan\s+de\s+desarrollo\s+municipal", text[:5000]):
            return "Municipal"
        elif re.search(r"(?i)plan\s+de\s+desarrollo\s+departamental", text[:5000]):
            return "Departmental"
        elif re.search(r"(?i)plan\s+de\s+desarrollo\s+distrital", text[:5000]):
            return "District"
        else:
            return "General"

    def _identify_sections(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Identify document sections based on patterns."""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            # Find all matches for this section pattern
            matches = list(re.finditer(pattern, text))
            
            if not matches:
                continue
                
            # For each match, find the section content
            for i, match in enumerate(matches):
                start_pos = match.start()
                
                # Find the end of this section
                if i < len(matches) - 1:
                    end_pos = matches[i + 1].start()
                else:
                    # Last section extends to end of text or max 10000 chars
                    end_pos = min(start_pos + 10000, len(text))
                
                # Extract section content
                section_text = text[start_pos:end_pos]
                
                # Store with unique key if multiple sections of same type
                key = section_name if i == 0 else f"{section_name}_{i+1}"
                sections[key] = {
                    "title": text[start_pos:start_pos + match.end() - match.start()].strip(),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "text_length": len(section_text),
                }
        
        return sections

    def _extract_evidence(self, text: str, evidence_type: str) -> List[str]:
        """
        Extract evidence of a specific type from text.
        
        Args:
            text: Text to search in
            evidence_type: Type of evidence to extract
            
        Returns:
            List of extracted evidence items
        """
        if evidence_type not in self.evidence_patterns:
            return []
            
        pattern = self.evidence_patterns[evidence_type]
        matches = re.finditer(pattern, text)
        
        evidence_items = []
        for match in matches:
            # If the pattern has a capture group, use that, otherwise use the whole match
            if match.lastindex and match.lastindex >= 1:
                evidence = match.group(1).strip()
            else:
                evidence = match.group(0).strip()
                
            # Clean up the evidence
            evidence = remove_unwanted_characters(evidence)
            evidence = re.sub(r'\s+', ' ', evidence).strip()
            
            # Only add non-empty, meaningful evidence
            if evidence and len(evidence) > 3:
                evidence_items.append(evidence)
        
        return evidence_items

    def _extract_cluster_evidence(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract evidence specific to policy clusters.
        
        Args:
            text: Text to search in
            
        Returns:
            Dictionary mapping cluster IDs to evidence types and their matches
        """
        cluster_evidence = {}
        
        # Get clusters from DECALOGO structure
        clusters = self.decalogo_structure.get("clusters_politica", [])
        
        for cluster in clusters:
            cluster_id = cluster.get("id")
            if not cluster_id:
                continue
                
            # Collect evidence for this cluster
            evidence_for_cluster = {}
            
            # Area evidence
            for area in cluster.get("areas_tematicas", []):
                pattern_name = f"area_{cluster_id}_{area}"
                if pattern_name in self.evidence_patterns:
                    matches = self._extract_evidence(text, pattern_name)
                    if matches:
                        if "areas_tematicas" not in evidence_for_cluster:
                            evidence_for_cluster["areas_tematicas"] = []
                        evidence_for_cluster["areas_tematicas"].extend(matches)
            
            # Indicator evidence
            for indicator in cluster.get("indicadores_clave", []):
                pattern_name = f"indicador_{cluster_id}_{indicator}"
                if pattern_name in self.evidence_patterns:
                    matches = self._extract_evidence(text, pattern_name)
                    if matches:
                        if "indicadores_clave" not in evidence_for_cluster:
                            evidence_for_cluster["indicadores_clave"] = []
                        evidence_for_cluster["indicadores_clave"].extend(matches)
            
            # Add direct mentions of cluster name or description
            name_pattern_name = f"nombre_{cluster_id}"
            if name_pattern_name in self.evidence_patterns:
                name_matches = self._extract_evidence(text, name_pattern_name)
                if name_matches:
                    evidence_for_cluster["menciones_directas"] = name_matches
            
            # Check related policies based on cluster points
            related_policies = []
            for punto_id in cluster.get("puntos_decalogo", []):
                punto_mentions = self._extract_punto_mentions(text, punto_id)
                if punto_mentions:
                    related_policies.extend(punto_mentions)
            
            if related_policies:
                evidence_for_cluster["politicas_relacionadas"] = related_policies
            
            # Only add cluster if it has evidence
            if evidence_for_cluster:
                cluster_evidence[cluster_id] = evidence_for_cluster
        
        return cluster_evidence

    def _extract_punto_mentions(self, text: str, punto_id: int) -> List[str]:
        """Extract mentions related to a specific decalogo point."""
        # Get decalogo point standards
        punto_name = self._get_punto_name(punto_id)
        if not punto_name:
            return []
        
        # Create pattern for this point
        pattern = r"(?i)" + re.escape(punto_name) + r".*?(?:\.|,|\n)"
        matches = list(re.finditer(pattern, text))
        
        results = []
        for match in matches:
            mention = match.group(0).strip()
            mention = remove_unwanted_characters(mention)
            mention = re.sub(r'\s+', ' ', mention).strip()
            if mention and len(mention) > 3:
                results.append(mention)
        
        return results
    
    def _get_punto_name(self, punto_id: int) -> str:
        """Get the name of a decalogo point by ID."""
        # Try to get from DNP standards
        try:
            dnp_standards_path = os.path.join(os.path.dirname(__file__), "DNP_STANDARDS.json")
            if os.path.exists(dnp_standards_path):
                with open(dnp_standards_path, 'r', encoding='utf-8') as f:
                    standards = json.load(f)
                
                punto_standards = standards.get("decalogo_point_standards", {}).get(str(punto_id))
                if punto_standards and "name" in punto_standards:
                    return punto_standards["name"]
        except Exception as e:
            logger.debug(f"Could not get punto name from standards: {e}")
        
        # Fallback to common punto names
        punto_names = {
            1: "Derecho a la Vida, Libertad e Integridad",
            2: "Derecho a la Participación y Ejercicio de Ciudadanía",
            3: "Derecho al Ambiente Sano y al Agua",
            4: "Derecho a la Igualdad",
            5: "Derecho de las Víctimas del Conflicto Armado",
            6: "Derecho a la Educación",
            7: "Derecho a la Salud",
            8: "Derecho a la Protección de Líderes y Defensores",
            9: "Derecho al Trabajo Digno",
            10: "Derecho a la Vivienda y Hábitat Dignos"
        }
        return punto_names.get(punto_id, "")

    def _create_cluster_dimension_mapping(self) -> Dict[str, List[str]]:
        """Create a mapping from clusters to relevant dimensions."""
        mapping = {}
        
        # Get clusters from DECALOGO structure
        clusters = self.decalogo_structure.get("clusters_politica", [])
        
        for cluster in clusters:
            cluster_id = cluster.get("id")
            if not cluster_id:
                continue
            
            # Default: all dimensions are relevant
            dimensions = ["DE-1", "DE-2", "DE-3", "DE-4"]
            
            # Check if the structure specifies particular dimensions for this cluster
            analisis_dimensional = self.decalogo_structure.get("analisis_dimensional", {})
            descripciones = analisis_dimensional.get("descripciones", {})
            
            # Find dimensions specifically mentioned for this cluster
            specified_dimensions = []
            for key in descripciones:
                parts = key.split("_")
                if len(parts) == 2 and parts[1] == cluster_id:
                    specified_dimensions.append(parts[0])
            
            # If specific dimensions are mentioned, use those
            if specified_dimensions:
                dimensions = specified_dimensions
            
            mapping[cluster_id] = dimensions
        
        return mapping

    def _extract_indicators_with_targets(self, text: str) -> List[Dict[str, str]]:
        """Extract indicators paired with targets and baseline."""
        indicator_sections = re.finditer(
            r"(?i)(?:indicador(?:es)?|meta(?:s)?|l[ií]nea(?:s)?\s+base)(?:.*?)(?:\n|$)", 
            text
        )
        
        results = []
        for section in indicator_sections:
            section_text = section.group(0)
            
            # Find indicator name
            indicator_match = re.search(r"(?i)(?:indicador(?:es)?:?\s*)(.*?)(?:\n|$)", section_text)
            indicator = indicator_match.group(1).strip() if indicator_match else None
            
            # Find target
            target_match = re.search(r"(?i)(?:meta(?:s)?:?\s*)(.*?)(?:\n|$)", section_text)
            target = target_match.group(1).strip() if target_match else None
            
            # Find baseline
            baseline_match = re.search(r"(?i)(?:l[ií]nea(?:s)?\s+base:?\s*)(.*?)(?:\n|$)", section_text)
            baseline = baseline_match.group(1).strip() if baseline_match else None
            
            if indicator or target or baseline:
                results.append({
                    "indicator": indicator or "",
                    "target": target or "",
                    "baseline": baseline or ""
                })
        
        return results

    def _extract_causal_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract causal relationships for theory of change."""
        # Look for causal relationship patterns
        causal_patterns = [
            # "A causes B" patterns
            r"(?i)([\w\s]+?)\s+(?:causa|produce|genera|conduce a|lleva a|resulta en)\s+([\w\s]+?)(?:\.|,|\n)",
            # "Due to A, B" patterns
            r"(?i)(?:debido a|por causa de|gracias a|como resultado de)\s+([\w\s]+?),\s+([\w\s]+?)(?:\.|,|\n)",
            # "To achieve A, need B" patterns
            r"(?i)(?:para|con el fin de|a fin de)\s+([\w\s]+?)\s+(?:se requiere|es necesario|se necesita)\s+([\w\s]+?)(?:\.|,|\n)",
        ]
        
        relationships = []
        for pattern in causal_patterns:
            for match in re.finditer(pattern, text):
                # Extract cause and effect based on pattern type
                if "causa|produce|genera" in pattern:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                elif "debido a|por causa de" in pattern:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                elif "para|con el fin de" in pattern:
                    effect = match.group(1).strip()
                    cause = match.group(2).strip()
                else:
                    continue
                    
                # Clean up and add to results
                if cause and effect and len(cause) > 3 and len(effect) > 3:
                    relationships.append({
                        "cause": cause,
                        "effect": effect,
                        "text": match.group(0).strip()
                    })
        
        return relationships

    def _extract_territorial_references(self, text: str) -> List[Dict[str, str]]:
        """Extract territorial references."""
        # Look for mentions of specific territories
        territory_patterns = [
            r"(?i)(?:municipio|vereda|corregimiento|barrio|comuna|localidad)\s+de\s+([\w\s]+?)(?:\.|,|\n)",
            r"(?i)(?:zona|sector|región|área)\s+([\w\s]+?)(?:\.|,|\n)",
        ]
        
        territories = []
        for pattern in territory_patterns:
            for match in re.finditer(pattern, text):
                territory_type = re.search(r"(?i)(municipio|vereda|corregimiento|barrio|comuna|localidad|zona|sector|región|área)", match.group(0))
                territory_name = match.group(1).strip()
                
                if territory_name and len(territory_name) > 2:
                    territories.append({
                        "type": territory_type.group(1).lower() if territory_type else "unknown",
                        "name": territory_name,
                        "mention": match.group(0).strip()
                    })
        
        return territories

    def get_evidence_requirements(self) -> Dict[str, List[str]]:
        """
        Get evidence requirements mapped by DECALOGO dimension.
        
        Returns:
            Dictionary mapping dimension IDs to required evidence types
        """
        requirements = {}
        
        for dimension in self.decalogo_structure.get("dimensiones", []):
            dim_id = dimension.get("id")
            requirements[dim_id] = []
            
            for question in dimension.get("preguntas", []):
                requirements[dim_id].extend(question.get("evidencia_requerida", []))
            
            # Remove duplicates
            requirements[dim_id] = list(set(requirements[dim_id]))
        
        return requirements

    def extract_dimension_evidence(self, text: str, dimension_id: str) -> Dict[str, List[str]]:
        """
        Extract all evidence required for a specific dimension.
        
        Args:
            text: Document text
            dimension_id: Dimension ID (e.g., 'DE-1')
            
        Returns:
            Dictionary mapping evidence types to extracted evidence
        """
        evidence = {}
        
        # Find the dimension
        for dimension in self.decalogo_structure.get("dimensiones", []):
            if dimension.get("id") == dimension_id:
                # Extract evidence for each question in the dimension
                for question in dimension.get("preguntas", []):
                    for evidence_type in question.get("evidencia_requerida", []):
                        matches = self._extract_evidence(text, evidence_type)
                        if matches:
                            evidence[evidence_type] = matches
        
        return evidence

    def extract_cluster_dimension_evidence(
        self, 
        text: str, 
        cluster_id: str, 
        dimension_id: str
    ) -> Dict[str, List[str]]:
        """
        Extract evidence specific to a cluster-dimension combination.
        
        Args:
            text: Document text
            cluster_id: Cluster ID (e.g., 'C1')
            dimension_id: Dimension ID (e.g., 'DE-1')
            
        Returns:
            Dictionary mapping evidence types to extracted evidence
        """
        evidence = {}
        
        # Get the cluster details
        clusters = self.decalogo_structure.get("clusters_politica", [])
        cluster = None
        for c in clusters:
            if c.get("id") == cluster_id:
                cluster = c
                break
        
        if not cluster:
            return {}
        
        # Get the dimension details
        dimensions = self.decalogo_structure.get("dimensiones", [])
        dimension = None
        for d in dimensions:
            if d.get("id") == dimension_id:
                dimension = d
                break
        
        if not dimension:
            return {}
        
        # Find the specific evidence requirements for this dimension in this cluster
        analisis_dimensional = self.decalogo_structure.get("analisis_dimensional", {})
        descripciones = analisis_dimensional.get("descripciones", {})
        umbrales = analisis_dimensional.get("umbrales_criticos", {})
        
        # Check if there's a specific description for this combination
        combo_key = f"{dimension_id}_{cluster_id}"
        if combo_key in descripciones:
            # This cluster-dimension combo has special handling
            evidence["descripcion_especial"] = [descripciones[combo_key]]
            
        if combo_key in umbrales:
            evidence["umbral_critico"] = [str(umbrales[combo_key])]
        
        # Extract general dimension evidence
        dimension_evidence = self.extract_dimension_evidence(text, dimension_id)
        
        # Extract cluster evidence
        cluster_evidence = {}
        for area in cluster.get("areas_tematicas", []):
            pattern_name = f"area_{cluster_id}_{area}"
            if pattern_name in self.evidence_patterns:
                matches = self._extract_evidence(text, pattern_name)
                if matches:
                    if "areas_tematicas" not in cluster_evidence:
                        cluster_evidence["areas_tematicas"] = []
                    cluster_evidence["areas_tematicas"].extend(matches)
        
        # For each question in this dimension, extract evidence related to this cluster's themes
        for question in dimension.get("preguntas", []):
            question_id = question.get("id")
            for evidence_type in question.get("evidencia_requerida", []):
                # Get general evidence for this type
                if evidence_type in dimension_evidence:
                    # Filter by cluster themes
                    filtered_evidence = []
                    for item in dimension_evidence[evidence_type]:
                        # Check if evidence item mentions any cluster theme
                        for area in cluster.get("areas_tematicas", []):
                            if re.search(rf"\b{re.escape(area)}\b", item, re.IGNORECASE):
                                filtered_evidence.append(item)
                                break
                    
                    if filtered_evidence:
                        key = f"{question_id}_{evidence_type}"
                        evidence[key] = filtered_evidence
        
        return evidence

    def get_cluster_alignment_score(self, text: str, cluster_id: str) -> Dict[str, Any]:
        """
        Calculate alignment score between plan text and a specific cluster.
        
        Args:
            text: Document text
            cluster_id: Cluster ID to evaluate alignment with
            
        Returns:
            Dictionary with alignment metrics
        """
        # Get the cluster details
        clusters = self.decalogo_structure.get("clusters_politica", [])
        cluster = None
        for c in clusters:
            if c.get("id") == cluster_id:
                cluster = c
                break
        
        if not cluster:
            return {"alignment_score": 0.0, "coverage": {}, "metrics": {}}
        
        # Process text and extract cluster evidence
        normalized_text = normalize_text(text)
        cluster_evidence = self._extract_cluster_evidence(normalized_text).get(cluster_id, {})
        
        # Calculate coverage metrics
        total_areas = len(cluster.get("areas_tematicas", []))
        covered_areas = len(cluster_evidence.get("areas_tematicas", []))
        area_coverage = covered_areas / total_areas if total_areas > 0 else 0
        
        total_indicators = len(cluster.get("indicadores_clave", []))
        covered_indicators = len(cluster_evidence.get("indicadores_clave", []))
        indicator_coverage = covered_indicators / total_indicators if total_indicators > 0 else 0
        
        # Calculate direct mentions
        direct_mentions = len(cluster_evidence.get("menciones_directas", []))
        
        # Calculate related policies
        related_policies = len(cluster_evidence.get("politicas_relacionadas", []))
        
        # Calculate overall alignment score
        alignment_score = (
            0.4 * area_coverage +
            0.3 * indicator_coverage +
            0.2 * min(1.0, direct_mentions / 3) +  # Cap at 1.0
            0.1 * min(1.0, related_policies / 2)   # Cap at 1.0
        )
        
        return {
            "alignment_score": alignment_score,
            "coverage": {
                "areas_tematicas": area_coverage,
                "indicadores_clave": indicator_coverage,
                "menciones_directas": direct_mentions,
                "politicas_relacionadas": related_policies
            },
            "metrics": {
                "covered_areas": covered_areas,
                "total_areas": total_areas,
                "covered_indicators": covered_indicators,
                "total_indicators": total_indicators
            }
        }