"""
Feasibility Scoring Module

Evaluates the presence and quality of baselines, targets, and timeframes in plans,
which are essential for answering key DECALOGO questions:
- DE-1 Q3: "Do outcomes have baselines and targets?"
- DE-4 Q1: "Do products have measurable KPIs?"
- DE-4 Q2: "Do results have baselines?"

Features:
- Detection of quantitative targets
- Baseline identification
- Timeframe recognition
- Indicator quality assessment
- SMART criteria evaluation
"""

import logging
import re
import argparse
import datetime
import gzip
import hashlib
import io
import json
import uuid
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Iterator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if pandas is available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ComponentType(Enum):
    """Component types that can be detected in plans."""
    BASELINE = "baseline"           # Starting point measurement
    TARGET = "target"               # Target to achieve
    TIME_HORIZON = "time_horizon"   # Time horizon for achievement
    DATE = "date"                   # Specific date reference
    NUMERICAL = "numerical"         # General numerical reference
    INDICATOR = "indicator"         # Performance indicator
    PERCENTAGE = "percentage"       # Percentage value
    RESPONSIBLE = "responsible"     # Entity responsible for achievement


@dataclass
class DetectionResult:
    """
    Result of a detection in text with metadata.
    
    Attributes:
        text: Detected text
        component_type: Type of detected component
        confidence: Confidence score (0-1)
        start_pos: Starting position in text
        end_pos: Ending position in text
        numeric_value: Extracted numerical value if applicable
        unit: Unit of measurement if applicable
        metadata: Additional metadata
    """
    text: str
    component_type: ComponentType
    confidence: float
    start_pos: int
    end_pos: int
    numeric_value: Optional[float] = None
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "component_type": self.component_type.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            **self.metadata
        }


@dataclass
class IndicatorScore:
    """
    Comprehensive evaluation of an indicator's quality and feasibility.
    
    Attributes:
        text: The indicator text
        has_baseline: Whether the indicator has a baseline
        has_target: Whether the indicator has a target
        has_timeframe: Whether the indicator has a timeframe
        has_quantitative_target: Whether the target is quantitative
        has_unit: Whether a unit of measurement is specified
        has_responsible: Whether a responsible entity is specified
        smart_score: Score for SMART criteria (0-1)
        feasibility_score: Overall feasibility score (0-1)
        components: Detected components related to this indicator
    """
    text: str
    has_baseline: bool = False
    has_target: bool = False
    has_timeframe: bool = False
    has_quantitative_target: bool = False
    has_unit: bool = False
    has_responsible: bool = False
    smart_score: float = 0.0
    feasibility_score: float = 0.0
    components: List[DetectionResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "has_baseline": self.has_baseline,
            "has_target": self.has_target,
            "has_timeframe": self.has_timeframe,
            "has_quantitative_target": self.has_quantitative_target,
            "has_unit": self.has_unit,
            "has_responsible": self.has_responsible,
            "smart_score": self.smart_score,
            "feasibility_score": self.feasibility_score,
            "components": [c.to_dict() for c in self.components]
        }


# For write safety utilities
@dataclass
class SafeWriteResult:
    """Result of a safe write operation."""
    status: str
    path: Optional[Path] = None
    key: Optional[str] = None
    error: Optional[Exception] = None


def safe_write_text(path: Path, content: str, label: str = "text", encoding: str = "utf-8") -> SafeWriteResult:
    """Write text content safely to file, with fallback options."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / f".{path.name}.tmp{uuid.uuid4().hex[:8]}"
        
        with open(temp_path, "w", encoding=encoding) as f:
            f.write(content)
        temp_path.rename(path)
        return SafeWriteResult(status="primary", path=path)
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to write {label} to {path}: {e}")
        return SafeWriteResult(status="error", error=e)


def safe_write_bytes(path: Path, content: bytes, label: str = "binary") -> SafeWriteResult:
    """Write binary content safely to file, with fallback options."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / f".{path.name}.tmp{uuid.uuid4().hex[:8]}"
        
        with open(temp_path, "wb") as f:
            f.write(content)
        temp_path.rename(path)
        return SafeWriteResult(status="primary", path=path)
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to write {label} to {path}: {e}")
        return SafeWriteResult(status="error", error=e)


def safe_write_json(path: Path, data: Any, label: str = "json") -> SafeWriteResult:
    """Write JSON data safely to file."""
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return safe_write_text(path, content, label=label)
    except Exception as e:
        logger.warning(f"Failed to serialize {label} JSON: {e}")
        return SafeWriteResult(status="error", error=e)


class FeasibilityScorer:
    """
    Evaluates the feasibility of plan elements by detecting baselines, targets, and timeframes.
    
    This scorer is essential for answering:
    - DE-1 Q3: "Do outcomes have baselines and targets?"
    - DE-4 Q1: "Do products have measurable KPIs?"
    - DE-4 Q2: "Do results have baselines?"
    
    It detects and evaluates indicators according to SMART criteria and provides
    comprehensive scoring of their feasibility.
    """
    
    def __init__(self, enable_parallel: bool = False):
        """
        Initialize the feasibility scorer.
        
        Args:
            enable_parallel: Whether to enable parallel processing for large documents
        """
        self.enable_parallel = enable_parallel
        self.logger = logger
        
        # Define detection patterns
        
        # Baseline patterns
        self.baseline_patterns = [
            r'(?i)l[ií]nea\s+(?:de\s+)?base\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)(?:valor|medici[óo]n)\s+(?:inicial|actual|presente)\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)(?:actualmente|al presente|al inicio|hoy)[^.]*?(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:situaci[óo]n actual|escenario base)[^.]*?(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:indicador|valor)[^.]*?(\d{4})[^.]*?(?:fue|era|correspondi[óo] a)\s*(\d+(?:[.,]\d+)?(?:\s*%))',
        ]
        
        # Target patterns
        self.target_patterns = [
            r'(?i)meta\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%)?)',
            r'(?i)(?:alcanzar|lograr|conseguir|obtener|llegar a)\s*(?:un|una|el|la)?[^.]*?(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:valor|nivel)\s+(?:esperado|objetivo|deseado)\s*[:=]?\s*(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:aumentar|incrementar|crecer|elevar)[^.]*?(?:hasta|a)\s*(\d+(?:[.,]\d+)?(?:\s*%))',
            r'(?i)(?:reducir|disminuir|bajar|decrecer)[^.]*?(?:hasta|a)\s*(\d+(?:[.,]\d+)?(?:\s*%))',
        ]
        
        # Time horizon patterns
        self.time_horizon_patterns = [
            r'(?i)(?:para|en)(?:\s+el)?\s+(?:año)?\s*(20\d{2})',
            r'(?i)(?:durante el|en el)?\s+(?:periodo|período|cuatrienio|horizonte)\s+(?:20\d{2})[^.]*?(?:20\d{2})',
            r'(?i)a[lñ]\s+(?:finalizar|terminar|concluir)[^.]*?(?:20\d{2})',
            r'(?i)(?:plazo|horizonte)(?:\s+de\s+tiempo)?\s*[:=]?\s*(?:\d+)\s*(?:años?|meses|semanas)',
            r'(?i)(?:mensual|anual|semestral|trimestral)(?:mente)?',
        ]
        
        # Indicator patterns
        self.indicator_patterns = [
            r'(?i)(?:indicador|KPI|métricas)[^.]*?:[^.]*?(\w[^.]+)',
            r'(?i)(?:tasa|porcentaje|índice|ratio)\s+de\s+(\w[^.]+)',
            r'(?i)(?:porcentaje|proporción|número|cantidad|total)\s+de\s+(\w[^.]+)',
            r'(?i)(?:cobertura|acceso)[^.]*?(?:servicios?|atención|asistencia)',
        ]
        
        # Unit patterns
        self.unit_patterns = [
            r'(?i)(?:porcentaje|porciento|%)',
            r'(?i)(?:kilómetros?|km|metros?|m)',
            r'(?i)(?:toneladas?|kg|kilogramos?|gramos?)',
            r'(?i)(?:litros?|l|metros? cúbicos?|m3)',
            r'(?i)(?:hectáreas?|ha|metros? cuadrados?|m2)',
            r'(?i)(?:personas?|habitantes?|individuos?|beneficiarios?)',
            r'(?i)(?:viviendas?|hogares?|casas?|unidades habitacionales?)',
            r'(?i)(?:días?|semanas?|meses?|años?|bimestres?|trimestres?|semestres?)',
            r'(?i)(?:pesos|COP|\$|USD|EUR|dólares?|euros?)',
        ]
        
        # Responsible entity patterns
        self.responsible_patterns = [
            r'(?i)(?:responsable)[^.]*?:\s*([^.]+)',
            r'(?i)(?:a cargo de|ejecutado por|implementado por)[^.]*?([^.]+)',
        ]
        
        # Compile patterns for efficiency
        self.compiled_baseline_patterns = [re.compile(p) for p in self.baseline_patterns]
        self.compiled_target_patterns = [re.compile(p) for p in self.target_patterns]
        self.compiled_time_horizon_patterns = [re.compile(p) for p in self.time_horizon_patterns]
        self.compiled_indicator_patterns = [re.compile(p) for p in self.indicator_patterns]
        self.compiled_unit_patterns = [re.compile(p) for p in self.unit_patterns]
        self.compiled_responsible_patterns = [re.compile(p) for p in self.responsible_patterns]
    
    def detect_components(self, text: str) -> Dict[ComponentType, List[DetectionResult]]:
        """
        Detect all components (baselines, targets, timeframes, etc.) in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping component types to lists of detection results
        """
        if not text:
            return {ct: [] for ct in ComponentType}
        
        results = {
            ComponentType.BASELINE: self._detect_baselines(text),
            ComponentType.TARGET: self._detect_targets(text),
            ComponentType.TIME_HORIZON: self._detect_time_horizons(text),
            ComponentType.INDICATOR: self._detect_indicators(text),
            ComponentType.NUMERICAL: self._detect_numerical_values(text),
            ComponentType.PERCENTAGE: self._detect_percentages(text),
            ComponentType.RESPONSIBLE: self._detect_responsible_entities(text),
            ComponentType.DATE: self._detect_dates(text),
        }
        
        return results
    
    def evaluate_indicator(self, text: str) -> IndicatorScore:
        """
        Evaluate an indicator text for feasibility.
        
        Args:
            text: Indicator text to evaluate
            
        Returns:
            IndicatorScore with comprehensive evaluation
        """
        # Detect all components
        components_dict = self.detect_components(text)
        
        # Flatten components list
        all_components = []
        for component_list in components_dict.values():
            all_components.extend(component_list)
        
        # Check for presence of key components
        has_baseline = len(components_dict[ComponentType.BASELINE]) > 0
        has_target = len(components_dict[ComponentType.TARGET]) > 0
        has_timeframe = len(components_dict[ComponentType.TIME_HORIZON]) > 0 or len(components_dict[ComponentType.DATE]) > 0
        
        # Check for quantitative aspects
        has_quantitative_target = False
        for target in components_dict[ComponentType.TARGET]:
            if target.numeric_value is not None:
                has_quantitative_target = True
                break
        
        # Check for units
        has_unit = False
        for component in all_components:
            if component.unit:
                has_unit = True
                break
        
        # Check for responsible entities
        has_responsible = len(components_dict[ComponentType.RESPONSIBLE]) > 0
        
        # Calculate SMART score
        smart_score = self._calculate_smart_score(
            has_baseline, has_target, has_timeframe, has_quantitative_target, has_unit, has_responsible
        )
        
        # Calculate overall feasibility score
        feasibility_score = self._calculate_feasibility_score(
            has_baseline, has_target, has_timeframe, has_quantitative_target, has_unit, has_responsible, smart_score
        )
        
        return IndicatorScore(
            text=text,
            has_baseline=has_baseline,
            has_target=has_target,
            has_timeframe=has_timeframe,
            has_quantitative_target=has_quantitative_target,
            has_unit=has_unit,
            has_responsible=has_responsible,
            smart_score=smart_score,
            feasibility_score=feasibility_score,
            components=all_components
        )
    
    def evaluate_indicators(self, texts: List[str]) -> List[IndicatorScore]:
        """
        Evaluate multiple indicator texts.
        
        Args:
            texts: List of indicator texts to evaluate
            
        Returns:
            List of IndicatorScore objects
        """
        if not texts:
            return []
        
        # If parallel processing is enabled and there are many texts, use parallel evaluation
        if self.enable_parallel and len(texts) > 10:
            try:
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(texts))) as executor:
                    return list(executor.map(self.evaluate_indicator, texts))
            except ImportError:
                logger.warning("concurrent.futures not available. Using sequential processing.")
        
        # Sequential processing
        return [self.evaluate_indicator(text) for text in texts]
    
    def evaluate_plan_feasibility(self, text: str, extract_indicators: bool = True) -> Dict[str, Any]:
        """
        Evaluate overall plan feasibility based on indicators and components.
        
        Args:
            text: Plan text to analyze
            extract_indicators: Whether to attempt indicator extraction from text
            
        Returns:
            Dictionary with comprehensive feasibility evaluation
        """
        # Detect all components
        components_dict = self.detect_components(text)
        
        # Extract and evaluate indicators if requested
        indicator_scores = []
        if extract_indicators:
            # Extract indicator-like segments
            potential_indicators = self._extract_indicator_segments(text)
            if potential_indicators:
                indicator_scores = self.evaluate_indicators(potential_indicators)
        
        # Calculate component statistics
        component_counts = {ct.value: len(results) for ct, results in components_dict.items()}
        
        # Calculate baseline-target alignment
        baseline_target_alignment = self._calculate_baseline_target_alignment(
            components_dict[ComponentType.BASELINE],
            components_dict[ComponentType.TARGET]
        )
        
        # Calculate quantitative orientation
        quantitative_orientation = (
            component_counts['numerical'] + 
            component_counts['percentage'] + 
            component_counts['target']
        ) / max(1, len(text) / 500)  # Normalize by text length
        
        # Calculate time specification
        time_specification = (
            component_counts['time_horizon'] + 
            component_counts['date']
        ) / max(1, len(text) / 1000)  # Normalize by text length
        
        # Calculate accountability presence
        accountability = component_counts['responsible'] / max(1, len(text) / 1000)
        
        # Calculate indicator quality
        avg_indicator_score = sum(score.feasibility_score for score in indicator_scores) / len(indicator_scores) if indicator_scores else 0
        
        # Calculate overall feasibility score
        overall_feasibility = (
            baseline_target_alignment * 0.3 +
            min(1.0, quantitative_orientation) * 0.3 +
            min(1.0, time_specification) * 0.2 +
            min(1.0, accountability) * 0.1 +
            avg_indicator_score * 0.1
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            baseline_target_alignment,
            quantitative_orientation,
            time_specification,
            accountability,
            avg_indicator_score,
            component_counts
        )
        
        # Generate results for DECALOGO questions
        de1_q3_answer = self._evaluate_de1_q3(components_dict, indicator_scores)
        de4_q1_answer = self._evaluate_de4_q1(components_dict, indicator_scores)
        de4_q2_answer = self._evaluate_de4_q2(components_dict, indicator_scores)
        
        return {
            "overall_feasibility": overall_feasibility,
            "component_counts": component_counts,
            "baseline_target_alignment": baseline_target_alignment,
            "quantitative_orientation": min(1.0, quantitative_orientation),
            "time_specification": min(1.0, time_specification),
            "accountability": min(1.0, accountability),
            "avg_indicator_score": avg_indicator_score,
            "recommendations": recommendations,
            "indicators": [score.to_dict() for score in indicator_scores],
            "decalogo_answers": {
                "DE1_Q3": de1_q3_answer,
                "DE4_Q1": de4_q1_answer,
                "DE4_Q2": de4_q2_answer
            }
        }
    
    def _detect_baselines(self, text: str) -> List[DetectionResult]:
        """Detect baseline references in text."""
        results = []
        
        for pattern in self.compiled_baseline_patterns:
            for match in pattern.finditer(text):
                # Extract the numeric value if available
                numeric_value = None
                unit = ""
                
                # Check if we have a capture group with numeric value
                if match.lastindex and match.group(1):
                    # Extract and clean numeric value
                    value_text = match.group(1)
                    
                    # Check for percentage
                    if "%" in value_text:
                        unit = "%"
                        value_text = value_text.replace("%", "").strip()
                    
                    # Convert to float
                    try:
                        numeric_value = float(value_text.replace(",", ".").strip())
                    except ValueError:
                        pass
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.BASELINE,
                    confidence=0.8,  # High confidence for explicit baseline patterns
                    start_pos=match.start(),
                    end_pos=match.end(),
                    numeric_value=numeric_value,
                    unit=unit
                )
                
                results.append(result)
        
        return results
    
    def _detect_targets(self, text: str) -> List[DetectionResult]:
        """Detect target references in text."""
        results = []
        
        for pattern in self.compiled_target_patterns:
            for match in pattern.finditer(text):
                # Extract the numeric value if available
                numeric_value = None
                unit = ""
                
                # Check if we have a capture group with numeric value
                if match.lastindex and match.group(1):
                    # Extract and clean numeric value
                    value_text = match.group(1)
                    
                    # Check for percentage
                    if "%" in value_text:
                        unit = "%"
                        value_text = value_text.replace("%", "").strip()
                    
                    # Convert to float
                    try:
                        numeric_value = float(value_text.replace(",", ".").strip())
                    except ValueError:
                        pass
                
                # Confidence level depends on how explicit the target is
                confidence = 0.8  # Base confidence
                if "meta" in match.group(0).lower():
                    confidence = 0.9  # Explicit mention of "meta" (goal)
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.TARGET,
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    numeric_value=numeric_value,
                    unit=unit
                )
                
                results.append(result)
        
        return results
    
    def _detect_time_horizons(self, text: str) -> List[DetectionResult]:
        """Detect time horizon references in text."""
        results = []
        
        for pattern in self.compiled_time_horizon_patterns:
            for match in pattern.finditer(text):
                # Extract the year if available
                year = None
                
                # Check if we have a capture group with a year
                if match.lastindex and match.group(1) and match.group(1).isdigit():
                    year = int(match.group(1))
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.TIME_HORIZON,
                    confidence=0.8,  # Base confidence for time horizons
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={"year": year} if year else {}
                )
                
                results.append(result)
        
        return results
    
    def _detect_indicators(self, text: str) -> List[DetectionResult]:
        """Detect indicator references in text."""
        results = []
        
        for pattern in self.compiled_indicator_patterns:
            for match in pattern.finditer(text):
                # Extract indicator name if available
                indicator_name = match.group(1) if match.lastindex else match.group(0)
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.INDICATOR,
                    confidence=0.7,  # Base confidence for indicators
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={"indicator_name": indicator_name}
                )
                
                results.append(result)
        
        return results
    
    def _detect_numerical_values(self, text: str) -> List[DetectionResult]:
        """Detect general numerical values in text."""
        results = []
        
        # Pattern for general numerical values
        numerical_pattern = r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b'
        for match in re.finditer(numerical_pattern, text):
            value_text = match.group(0)
            
            # Try to convert to float
            try:
                numeric_value = float(value_text.replace(",", ".").strip())
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.NUMERICAL,
                    confidence=0.6,  # Lower confidence for general numbers
                    start_pos=match.start(),
                    end_pos=match.end(),
                    numeric_value=numeric_value
                )
                
                results.append(result)
            except ValueError:
                pass
        
        return results
    
    def _detect_percentages(self, text: str) -> List[DetectionResult]:
        """Detect percentage values in text."""
        results = []
        
        # Pattern for percentages
        percentage_pattern = r'\b\d{1,3}(?:[.,]\d+)?%'
        for match in re.finditer(percentage_pattern, text):
            value_text = match.group(0).replace("%", "").strip()
            
            # Convert to float
            try:
                numeric_value = float(value_text.replace(",", ".").strip())
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.PERCENTAGE,
                    confidence=0.8,  # High confidence for explicit percentages
                    start_pos=match.start(),
                    end_pos=match.end(),
                    numeric_value=numeric_value,
                    unit="%"
                )
                
                results.append(result)
            except ValueError:
                pass
        
        return results
    
    def _detect_responsible_entities(self, text: str) -> List[DetectionResult]:
        """Detect responsible entities in text."""
        results = []
        
        for pattern in self.compiled_responsible_patterns:
            for match in pattern.finditer(text):
                # Extract entity name if available
                entity_name = match.group(1) if match.lastindex else ""
                
                # Create detection result
                result = DetectionResult(
                    text=match.group(0),
                    component_type=ComponentType.RESPONSIBLE,
                    confidence=0.7,  # Base confidence for responsibility mentions
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={"entity_name": entity_name.strip() if entity_name else ""}
                )
                
                results.append(result)
        
        return results
    
    def _detect_dates(self, text: str) -> List[DetectionResult]:
        """Detect date references in text."""
        results = []
        
        # Pattern for dates (basic)
        date_pattern = r'\b(?:(?:0?[1-9]|[12][0-9]|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}|(?:19|20)\d{2}[/.-](?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12][0-9]|3[01]))\b'
        for match in re.finditer(date_pattern, text):
            # Create detection result
            result = DetectionResult(
                text=match.group(0),
                component_type=ComponentType.DATE,
                confidence=0.8,  # High confidence for explicit dates
                start_pos=match.start(),
                end_pos=match.end(),
                metadata={"date_string": match.group(0)}
            )
            
            results.append(result)
            
        # Also detect year references (e.g., "2025")
        year_pattern = r'\b(19|20)\d{2}\b'
        for match in re.finditer(year_pattern, text):
            # Create detection result
            result = DetectionResult(
                text=match.group(0),
                component_type=ComponentType.DATE,
                confidence=0.7,  # Slightly lower confidence for just year
                start_pos=match.start(),
                end_pos=match.end(),
                metadata={"year": int(match.group(0))}
            )
            
            results.append(result)
        
        return results
    
    def _extract_indicator_segments(self, text: str) -> List[str]:
        """
        Extract indicator-like segments from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of text segments likely containing indicators
        """
        indicator_segments = []
        
        # Look for explicit indicator sections
        indicator_section_patterns = [
            r'(?i)(?:indicadores|kpis|métricas)[^.]*?:(.+?)(?:\n\n|\n[A-Z]|\Z)',
            r'(?i)(?:tabla|cuadro) de indicadores(.+?)(?:\n\n|\n[A-Z]|\Z)',
        ]
        
        for pattern in indicator_section_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                if match.group(1):
                    indicator_segments.append(match.group(1).strip())
        
        # Look for indicator-like sentences
        indicator_sentence_patterns = [
            r'(?i)(?:indicador|meta|objetivo)(?:[^.]+?)\d+(?:[.,]\d+)?%?[^.]*?\.', 
            r'(?i)(?:línea base|valor actual)(?:[^.]+?)\d+(?:[.,]\d+)?%?[^.]*?\.',
            r'(?i)(?:aumentar|incrementar|reducir|disminuir)(?:[^.]+?)\d+(?:[.,]\d+)?%?[^.]*?\.'
        ]
        
        for pattern in indicator_sentence_patterns:
            for match in re.finditer(pattern, text):
                indicator_segments.append(match.group(0).strip())
        
        # Deduplicate
        unique_segments = []
        seen = set()
        for segment in indicator_segments:
            normalized = ' '.join(segment.lower().split())
            if normalized not in seen and len(segment) > 10:
                seen.add(normalized)
                unique_segments.append(segment)
        
        return unique_segments
    
    def _calculate_smart_score(
        self, has_baseline: bool, has_target: bool, has_timeframe: bool, 
        has_quantitative_target: bool, has_unit: bool, has_responsible: bool
    ) -> float:
        """
        Calculate SMART criteria score for an indicator.
        
        Args:
            has_baseline: Whether baseline is present
            has_target: Whether target is present
            has_timeframe: Whether timeframe is present
            has_quantitative_target: Whether target is quantitative
            has_unit: Whether unit is specified
            has_responsible: Whether responsible entity is specified
            
        Returns:
            Score between 0 and 1 representing SMART criteria compliance
        """
        # Specific - quantitative with unit
        specific_score = (0.7 if has_quantitative_target else 0.0) + (0.3 if has_unit else 0.0)
        
        # Measurable - baseline and target
        measurable_score = (0.5 if has_baseline else 0.0) + (0.5 if has_target else 0.0)
        
        # Assignable - responsible entity
        assignable_score = 1.0 if has_responsible else 0.0
        
        # Realistic - baseline and not extreme targets (approximated by presence of baseline)
        realistic_score = 0.8 if has_baseline else 0.3  # Assume moderate realism even without baseline
        
        # Time-bound - timeframe specified
        timebound_score = 1.0 if has_timeframe else 0.0
        
        # Weighted average of SMART dimensions
        weights = {"S": 0.25, "M": 0.3, "A": 0.1, "R": 0.15, "T": 0.2}
        smart_score = (
            weights["S"] * specific_score + 
            weights["M"] * measurable_score + 
            weights["A"] * assignable_score + 
            weights["R"] * realistic_score + 
            weights["T"] * timebound_score
        )
        
        return smart_score
    
    def _calculate_feasibility_score(
        self, has_baseline: bool, has_target: bool, has_timeframe: bool, 
        has_quantitative_target: bool, has_unit: bool, has_responsible: bool, 
        smart_score: float
    ) -> float:
        """
        Calculate overall feasibility score for an indicator.
        
        Args:
            has_baseline: Whether baseline is present
            has_target: Whether target is present
            has_timeframe: Whether timeframe is present
            has_quantitative_target: Whether target is quantitative
            has_unit: Whether unit is specified
            has_responsible: Whether responsible entity is specified
            smart_score: SMART criteria score
            
        Returns:
            Score between 0 and 1 representing overall feasibility
        """
        # Essential components check (baseline, target, timeframe)
        essential_components = (
            (1.0 if has_baseline else 0.0) * 0.4 + 
            (1.0 if has_target else 0.0) * 0.4 + 
            (1.0 if has_timeframe else 0.0) * 0.2
        )
        
        # Quantitative components boost
        quantitative_boost = 0.0
        if has_quantitative_target and has_unit:
            quantitative_boost = 0.2
        elif has_quantitative_target:
            quantitative_boost = 0.1
        
        # Responsible entity boost
        responsible_boost = 0.1 if has_responsible else 0.0
        
        # Calculate feasibility score with SMART criteria
        feasibility_score = (
            essential_components * 0.6 + 
            smart_score * 0.2 + 
            quantitative_boost + 
            responsible_boost
        )
        
        return min(1.0, feasibility_score)  # Cap at 1.0
    
    def _calculate_baseline_target_alignment(
        self, baselines: List[DetectionResult], targets: List[DetectionResult]
    ) -> float:
        """
        Calculate alignment score between baselines and targets.
        
        Args:
            baselines: List of baseline detection results
            targets: List of target detection results
            
        Returns:
            Alignment score between 0 and 1
        """
        if not baselines or not targets:
            return 0.0
        
        # Count baselines and targets with numeric values
        numeric_baselines = sum(1 for b in baselines if b.numeric_value is not None)
        numeric_targets = sum(1 for t in targets if t.numeric_value is not None)
        
        # Count matches in units
        unit_match_count = 0
        baseline_units = set(b.unit for b in baselines if b.unit)
        target_units = set(t.unit for t in targets if t.unit)
        
        # If both have units, calculate matches
        if baseline_units and target_units:
            unit_match_count = len(baseline_units.intersection(target_units))
        
        # Calculate alignment score
        if numeric_baselines == 0 or numeric_targets == 0:
            return 0.2  # Low score if either has no numeric values
        
        # Base alignment based on presence
        presence_score = min(1.0, (len(baselines) / len(targets)) if targets else 0)
        
        # Numeric alignment
        numeric_score = min(1.0, (numeric_baselines / numeric_targets) if numeric_targets else 0)
        
        # Unit match score
        unit_score = 0.0
        if baseline_units and target_units:
            unit_score = unit_match_count / max(len(baseline_units), len(target_units))
        
        # Weighted alignment score
        alignment_score = (
            presence_score * 0.3 + 
            numeric_score * 0.5 + 
            unit_score * 0.2
        )
        
        return alignment_score
    
    def _generate_recommendations(
        self, 
        baseline_target_alignment: float,
        quantitative_orientation: float,
        time_specification: float,
        accountability: float,
        avg_indicator_score: float,
        component_counts: Dict[str, int]
    ) -> List[str]:
        """
        Generate recommendations based on feasibility analysis.
        
        Args:
            baseline_target_alignment: Alignment score between baselines and targets
            quantitative_orientation: Score for quantitative orientation
            time_specification: Score for time specification
            accountability: Score for accountability presence
            avg_indicator_score: Average indicator quality score
            component_counts: Counts of different component types
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Baseline-target recommendations
        if baseline_target_alignment < 0.3:
            if component_counts["baseline"] == 0:
                recommendations.append(
                    "Incluir líneas base para todos los indicadores para permitir una medición adecuada del progreso."
                )
            if component_counts["target"] == 0:
                recommendations.append(
                    "Definir metas claras y cuantificables para todos los indicadores."
                )
            if component_counts["baseline"] > 0 and component_counts["target"] > 0:
                recommendations.append(
                    "Mejorar la alineación entre líneas base y metas para asegurar consistencia en la medición."
                )
        
        # Quantitative orientation recommendations
        if quantitative_orientation < 0.5:
            recommendations.append(
                "Aumentar el uso de medidas cuantitativas en objetivos y metas para facilitar la evaluación del progreso."
            )
            
            if component_counts["numerical"] == 0 and component_counts["percentage"] == 0:
                recommendations.append(
                    "Incluir valores numéricos específicos (cantidades, porcentajes) en la definición de metas y resultados."
                )
        
        # Time specification recommendations
        if time_specification < 0.4:
            recommendations.append(
                "Especificar horizontes temporales claros para el logro de objetivos y metas."
            )
            
            if component_counts["time_horizon"] == 0 and component_counts["date"] == 0:
                recommendations.append(
                    "Definir fechas o períodos concretos para cada meta y resultado esperado."
                )
        
        # Accountability recommendations
        if accountability < 0.3:
            recommendations.append(
                "Asignar claramente responsabilidades para cada objetivo, meta y acción."
            )
            
            if component_counts["responsible"] == 0:
                recommendations.append(
                    "Especificar entidades y/o cargos responsables para la implementación y seguimiento."
                )
        
        # SMART criteria recommendations
        if avg_indicator_score < 0.5:
            recommendations.append(
                "Formular indicadores siguiendo criterios SMART (Específicos, Medibles, Asignables, Realistas, y con Tiempo definido)."
            )
        
        # Overall recommendations
        if len(recommendations) >= 3:
            recommendations.insert(0, 
                "Revisar y fortalecer el marco de seguimiento y evaluación del plan con enfoque en resultados."
            )
        
        return recommendations
    
    def _evaluate_de1_q3(
        self, 
        components_dict: Dict[ComponentType, List[DetectionResult]], 
        indicator_scores: List[IndicatorScore]
    ) -> Dict[str, Any]:
        """
        Evaluate DE-1 Q3: "Do outcomes have baselines and targets?"
        
        Args:
            components_dict: Dictionary of detected components
            indicator_scores: List of evaluated indicators
            
        Returns:
            Dictionary with evaluation results
        """
        baselines = components_dict[ComponentType.BASELINE]
        targets = components_dict[ComponentType.TARGET]
        
        has_baselines = len(baselines) > 0
        has_targets = len(targets) > 0
        
        # Check for outcomes with both baseline and target
        outcomes_with_both = 0
        total_outcomes = 0
        
        for indicator in indicator_scores:
            if "resultado" in indicator.text.lower() or "outcome" in indicator.text.lower():
                total_outcomes += 1
                if indicator.has_baseline and indicator.has_target:
                    outcomes_with_both += 1
        
        # Calculate coverage ratio
        coverage_ratio = outcomes_with_both / total_outcomes if total_outcomes > 0 else 0.0
        
        # Determine answer
        if coverage_ratio >= 0.7:
            answer = "Sí"
            confidence = 0.9
        elif coverage_ratio >= 0.3:
            answer = "Parcial"
            confidence = 0.7
        else:
            answer = "No"
            confidence = 0.8
        
        return {
            "answer": answer,
            "confidence": confidence,
            "outcomes_with_baselines_targets": outcomes_with_both,
            "total_outcomes": total_outcomes,
            "coverage_ratio": coverage_ratio,
            "has_any_baselines": has_baselines,
            "has_any_targets": has_targets
        }
    
    def _evaluate_de4_q1(
        self, 
        components_dict: Dict[ComponentType, List[DetectionResult]], 
        indicator_scores: List[IndicatorScore]
    ) -> Dict[str, Any]:
        """
        Evaluate DE-4 Q1: "Do products have measurable KPIs?"
        
        Args:
            components_dict: Dictionary of detected components
            indicator_scores: List of evaluated indicators
            
        Returns:
            Dictionary with evaluation results
        """
        indicators = components_dict[ComponentType.INDICATOR]
        
        has_indicators = len(indicators) > 0
        
        # Check for product indicators with quantitative metrics
        product_indicators = 0
        measurable_product_indicators = 0
        
        for indicator in indicator_scores:
            if "producto" in indicator.text.lower() or "product" in indicator.text.lower():
                product_indicators += 1
                if indicator.has_quantitative_target:
                    measurable_product_indicators += 1
        
        # Calculate coverage ratio
        coverage_ratio = measurable_product_indicators / product_indicators if product_indicators > 0 else 0.0
        
        # Determine answer
        if coverage_ratio >= 0.7:
            answer = "Sí"
            confidence = 0.85
        elif coverage_ratio >= 0.3:
            answer = "Parcial"
            confidence = 0.7
        else:
            answer = "No"
            confidence = 0.8
        
        return {
            "answer": answer,
            "confidence": confidence,
            "product_indicators": product_indicators,
            "measurable_product_indicators": measurable_product_indicators,
            "coverage_ratio": coverage_ratio,
            "has_any_indicators": has_indicators
        }
    
    def _evaluate_de4_q2(
        self, 
        components_dict: Dict[ComponentType, List[DetectionResult]], 
        indicator_scores: List[IndicatorScore]
    ) -> Dict[str, Any]:
        """
        Evaluate DE-4 Q2: "Do results have baselines?"
        
        Args:
            components_dict: Dictionary of detected components
            indicator_scores: List of evaluated indicators
            
        Returns:
            Dictionary with evaluation results
        """
        baselines = components_dict[ComponentType.BASELINE]
        
        has_baselines = len(baselines) > 0
        
        # Check for result indicators with baselines
        result_indicators = 0
        result_indicators_with_baselines = 0
        
        for indicator in indicator_scores:
            if "resultado" in indicator.text.lower() or "result" in indicator.text.lower():
                result_indicators += 1
                if indicator.has_baseline:
                    result_indicators_with_baselines += 1
        
        # Calculate coverage ratio
        coverage_ratio = result_indicators_with_baselines / result_indicators if result_indicators > 0 else 0.0
        
        # Determine answer
        if coverage_ratio >= 0.7:
            answer = "Sí"
            confidence = 0.85
        elif coverage_ratio >= 0.3:
            answer = "Parcial"
            confidence = 0.7
        else:
            answer = "No"
            confidence = 0.8
        
        return {
            "answer": answer,
            "confidence": confidence,
            "result_indicators": result_indicators,
            "result_indicators_with_baselines": result_indicators_with_baselines,
            "coverage_ratio": coverage_ratio,
            "has_any_baselines": has_baselines
        }

    def _calculate_structure_penalty(self, text: str) -> float:
        """Calculate penalty for indicators in titles/bullets without values."""
        stripped = text.strip()
        if not stripped:
            return 0.0

        is_bullet = bool(re.match(r"^[-•*]\s+", stripped))
        is_header = bool(re.match(r"^#{1,6}\s+", stripped))
        is_all_caps = stripped.isupper()
        is_caps_title = stripped.endswith(":") and stripped[:-1].strip().isupper()

        if not (is_bullet or is_header or is_all_caps or is_caps_title):
            return 0.0

        # Check if title has associated quantitative values
        value_patterns = [
            r"\d+(?:[.,]\d+)?(?:\s*%|\s*millones?|\s*mil)",
            r"[\$][\d,.\s]+",
            r"cop\s*[\d,.\s]+",
            r"\d{4}",  # Years
            r"q[1-4]",  # Quarters
        ]

        has_values = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in value_patterns
        )

        # Apply penalty if title-like without values
        return 1.0 if not has_values else 0.0

    def generate_report(self, indicators: List[str], output_path: str) -> None:
        """
        Generate a comprehensive feasibility report and save it to file using atomic operations.

        Uses atomic file operations to prevent corrupted output files if the process is
        interrupted during report generation. This is achieved by:
        1. Writing the complete report content to a temporary file in the same directory
        2. Using Path.rename() to atomically move the temporary file to the final destination

        Note: Atomicity may not be guaranteed on some remote filesystems (NFS, SMB) due to
        their implementation of rename operations. For local filesystems (ext4, NTFS, APFS),
        the rename operation is atomic.

        Args:
            indicators: List of indicator texts to analyze
            output_path: Path where the report should be saved

        Raises:
            IOError: If file operations fail
            ValueError: If indicators list is empty
        """
        if not indicators:
            raise ValueError("Indicators list cannot be empty")

        output_file = Path(output_path)
        report_content = self._generate_report_content(indicators)

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as exc:
            if self._is_recoverable_io_error(exc):
                result = safe_write_text(
                    output_file, report_content, label="feasibility_report"
                )
                self._log_safe_write_result(
                    "Reporte de factibilidad", output_file, result
                )
                return
            raise IOError(f"Failed to prepare report directory: {exc}") from exc

        temp_file = (
            output_file.parent
            / f"{output_file.name}.tmp.{uuid.uuid4().hex[:8]}"
        )

        try:
            with temp_file.open("w", encoding="utf-8") as f:
                f.write(report_content)
                f.flush()
            temp_file.rename(output_file)
            self.logger.info("Reporte de factibilidad guardado en %s", output_file)
        except (PermissionError, OSError) as exc:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            if self._is_recoverable_io_error(exc):
                result = safe_write_text(
                    output_file, report_content, label="feasibility_report"
                )
                self._log_safe_write_result(
                    "Reporte de factibilidad", output_file, result
                )
                return
            raise IOError(f"Failed to generate report: {exc}") from exc
        except Exception as exc:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed to generate report: {exc}") from exc

    def _generate_report_content(self, indicators: List[str]) -> str:
        """Generate the complete report content for the given indicators."""
        results = [self.evaluate_indicator(indicator) for indicator in indicators]

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build report content
        content_parts = []
        content_parts.append("# Feasibility Assessment Report")
        content_parts.append(f"Generated on: {timestamp}")
        content_parts.append(f"Total indicators analyzed: {len(indicators)}")
        content_parts.append("")

        # Summary statistics
        scores = [result.feasibility_score for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Calculate quality tiers
        quality_tiers = {}
        for result in results:
            feasibility_score = result.feasibility_score
            if feasibility_score >= 0.8:
                quality_tier = "high"
            elif feasibility_score >= 0.6:
                quality_tier = "medium"
            elif feasibility_score >= 0.4:
                quality_tier = "low"
            elif feasibility_score >= 0.2:
                quality_tier = "poor"
            else:
                quality_tier = "insufficient"
            
            quality_tiers[quality_tier] = quality_tiers.get(quality_tier, 0) + 1

        content_parts.append("## Summary")
        content_parts.append(f"Average feasibility score: {avg_score:.3f}")
        content_parts.append("Quality tier distribution:")
        for tier, count in sorted(quality_tiers.items()):
            percentage = (count / len(results)) * 100
            content_parts.append(f"  - {tier}: {count} ({percentage:.1f}%)")
        content_parts.append("")

        # Detailed results
        content_parts.append("## Detailed Analysis")
        content_parts.append("")

        # Sort results by score (highest first)
        sorted_results = list(zip(indicators, results))
        sorted_results.sort(key=lambda x: x[1].feasibility_score, reverse=True)

        for i, (indicator, result) in enumerate(sorted_results, 1):
            content_parts.append(f"### {i}. Indicator Analysis")
            content_parts.append(f"**Text:** {indicator}")
            content_parts.append(f"**Score:** {result.feasibility_score:.3f}")
            
            # Calculate quality tier for this result
            if result.feasibility_score >= 0.8:
                quality_tier = "high"
            elif result.feasibility_score >= 0.6:
                quality_tier = "medium"
            elif result.feasibility_score >= 0.4:
                quality_tier = "low"
            elif result.feasibility_score >= 0.2:
                quality_tier = "poor"
            else:
                quality_tier = "insufficient"
                
            content_parts.append(f"**Quality Tier:** {quality_tier}")
            content_parts.append(
                f"**Quantitative Baseline:** {'Yes' if result.has_baseline else 'No'}"
            )
            content_parts.append(
                f"**Quantitative Target:** {'Yes' if result.has_quantitative_target else 'No'}"
            )

            components_detected = set(c.component_type for c in result.components)
            if components_detected:
                content_parts.append(
                    f"**Components Detected:** {', '.join(c.value for c in components_detected)}"
                )

            # Add components as detailed_matches for compatibility
            result.detailed_matches = result.components
            
            if result.detailed_matches:
                content_parts.append("**Pattern Matches:**")
                for match in result.detailed_matches:
                    content_parts.append(
                        f"  - {match.component_type.value}: '{match.text}' (confidence: {match.confidence:.2f})"
                    )

            content_parts.append("")

        # Recommendations
        content_parts.append("## Recommendations")

        low_quality_count = sum(
            1 for result in results if result.feasibility_score < 0.5
        )
        if low_quality_count > 0:
            content_parts.append(
                f"- {low_quality_count} indicators have scores below 0.5 and need improvement"
            )
            content_parts.append(
                "- Focus on adding quantitative baselines and targets")
            content_parts.append(
                "- Include specific time horizons where missing")

        insufficient_count = sum(
            1 for result in results if result.feasibility_score < 0.2
        )
        if insufficient_count > 0:
            content_parts.append(
                f"- {insufficient_count} indicators are missing core components (baseline or target)"
            )
            content_parts.append(
                "- These require fundamental restructuring to be measurable"
            )

        content_parts.append("")
        content_parts.append("---")
        content_parts.append("*Report generated by Feasibility Scorer v1.0*")

        return "\n".join(content_parts)

    def _is_recoverable_io_error(self, exc: Exception) -> bool:
        """Check if an I/O error is recoverable with a fallback strategy."""
        if isinstance(exc, (PermissionError, OSError)):
            return True
        return False

    def _log_safe_write_result(
        self, description: str, intended_path: Path, result: SafeWriteResult
    ) -> None:
        """Log the result of a safe write operation."""
        if result.status == "primary":
            self.logger.info("%s guardado en %s", description, result.path)
        elif result.status == "fallback":
            self.logger.info(
                "%s guardado en ubicación alternativa %s (original: %s)", 
                description, result.path, intended_path
            )
        else:
            self.logger.error(
                "%s no pudo ser guardado: %s", description, result.error
            )

    def generate_traceability_matrix_csv(
        self, results: Dict[str, IndicatorScore], output_dir: str = "."
    ) -> str:
        """
        Generate consolidated traceability matrix CSV with all evaluation dimensions.

        Args:
            results: Dictionary mapping plan filenames to IndicatorScore objects
            output_dir: Directory to save the CSV file

        Returns:
            Path to the generated CSV file

        Raises:
            ImportError: If pandas is not available
        """
        if not PANDAS_AVAILABLE:
            # Fallback: generate CSV manually without pandas
            return self._generate_csv_fallback(results, output_dir)

        # Spanish column headers for the traceability matrix
        columns = {
            "archivo_plan": "Archivo del Plan",
            "puntuacion_factibilidad": "Puntuación de Factibilidad",
            "nivel_calidad": "Nivel de Calidad",
            "linea_base_cuantitativa": "Línea Base Cuantitativa",
            "meta_cuantitativa": "Meta Cuantitativa",
            "componentes_detectados": "Componentes Detectados",
            "tiene_linea_base": "Tiene Línea Base",
            "tiene_meta": "Tiene Meta",
            "tiene_horizonte_temporal": "Tiene Horizonte Temporal",
            "tiene_valores_numericos": "Tiene Valores Numéricos",
            "tiene_fechas": "Tiene Fechas",
            "coincidencias_detalladas": "Coincidencias Detalladas",
            "recomendacion_general": "Recomendación General",
        }

        # Build rows for the DataFrame
        rows = []
        for plan_filename, score in results.items():
            # Generate overall recommendation based on quality tier
            recommendation = self._get_recommendation_spanish(
                score.quality_tier
            )

            # Count component types
            components = score.components_detected
            has_baseline = ComponentType.BASELINE in components
            has_target = ComponentType.TARGET in components
            has_time_horizon = ComponentType.TIME_HORIZON in components
            has_numerical = ComponentType.NUMERICAL in components
            has_dates = ComponentType.DATE in components

            # Format detected components in Spanish
            component_names_spanish = {
                ComponentType.BASELINE: "línea base",
                ComponentType.TARGET: "meta",
                ComponentType.TIME_HORIZON: "horizonte temporal",
                ComponentType.NUMERICAL: "valores numéricos",
                ComponentType.DATE: "fechas",
            }

            components_list = [
                component_names_spanish.get(comp, comp.value) for comp in components
            ]
            components_str = (
                ", ".join(components_list) if components_list else "ninguno"
            )

            # Format detailed matches
            matches_details = []
            for match in score.detailed_matches:
                match_info = f"{component_names_spanish.get(match.component_type, match.component_type.value)}: '{match.matched_text}' (confianza: {match.confidence:.2f})"
                matches_details.append(match_info)
            matches_str = (
                "; ".join(matches_details)
                if matches_details
                else "ninguna coincidencia"
            )

            row = {
                "archivo_plan": plan_filename,
                "puntuacion_factibilidad": round(score.feasibility_score, 3),
                "nivel_calidad": self._translate_quality_tier_spanish(
                    score.quality_tier
                ),
                "linea_base_cuantitativa": (
                    "Sí" if score.has_quantitative_baseline else "No"
                ),
                "meta_cuantitativa": "Sí" if score.has_quantitative_target else "No",
                "componentes_detectados": components_str,
                "tiene_linea_base": "Sí" if has_baseline else "No",
                "tiene_meta": "Sí" if has_target else "No",
                "tiene_horizonte_temporal": "Sí" if has_time_horizon else "No",
                "tiene_valores_numericos": "Sí" if has_numerical else "No",
                "tiene_fechas": "Sí" if has_dates else "No",
                "coincidencias_detalladas": matches_str,
                "recomendacion_general": recommendation,
            }
            rows.append(row)

        # Create DataFrame with Spanish column names
        df = pd.DataFrame(rows)
        df = df.rename(columns=columns)

        # Sort by feasibility score (descending)
        df = df.sort_values("Puntuación de Factibilidad", ascending=False)

        csv_content = df.to_csv(index=False, encoding="utf-8-sig")
        csv_bytes = csv_content.encode("utf-8-sig")
        file_size_mb = len(csv_bytes) / (1024 * 1024)

        intended_path = Path(output_dir) / "matriz_trazabilidad_factibilidad.csv"
        if file_size_mb > 5.0:
            intended_path = intended_path.with_suffix(".csv.gz")
            result = safe_write_bytes(
                intended_path, gzip.compress(csv_bytes), label="feasibility_traceability_csv"
            )
            summary = "CSV exportado con compresión gzip"
        else:
            result = safe_write_text(
                intended_path,
                csv_content,
                label="feasibility_traceability_csv",
                encoding="utf-8-sig",
            )
            summary = "CSV exportado"

        self._log_safe_write_result(
            "Matriz de trazabilidad CSV", intended_path, result
        )

        if result.status == "primary":
            print(f"{summary}: {result.path} (tamaño: {file_size_mb:.1f}MB)")
        elif result.status == "fallback":
            print(
                f"{summary} (fallback en {result.path}, original: {intended_path})"
                f" [tamaño estimado: {file_size_mb:.1f}MB]"
            )
        else:
            print(
                f"{summary} retenido en memoria (clave {result.key})"
                f" [tamaño estimado: {file_size_mb:.1f}MB]"
            )

        return str(result.path or result.key or intended_path)

    @staticmethod
    def translate_quality_tier_spanish(tier: str) -> str:
        """Public wrapper to translate quality tier to Spanish."""
        return FeasibilityScorer._translate_quality_tier_spanish(tier)

    @staticmethod
    def get_recommendation_spanish(quality_tier: str) -> str:
        """Public wrapper to obtain recommendation in Spanish for a quality tier."""
        return FeasibilityScorer._get_recommendation_spanish(quality_tier)

    @staticmethod
    def _translate_quality_tier_spanish(tier: str) -> str:
        """Translate quality tier to Spanish."""
        translations = {
            "high": "Alto",
            "medium": "Medio",
            "low": "Bajo",
            "poor": "Deficiente",
            "insufficient": "Insuficiente",
        }
        return translations.get(tier, tier)

    @staticmethod
    def _get_recommendation_spanish(quality_tier: str) -> str:
        """Generate recommendation in Spanish based on quality tier."""
        recommendations = {
            "high": "Indicador de alta calidad. Mantener el nivel de especificidad.",
            "medium": "Indicador aceptable. Considerar agregar elementos cuantitativos adicionales.",
            "low": "Indicador básico. Requiere mejoras en elementos cuantitativos y horizonte temporal.",
            "poor": "Indicador deficiente. Necesita revisión integral para incluir línea base y metas claras.",
            "insufficient": "Indicador insuficiente. Debe incluir tanto línea base como metas para ser viable.",
        }
        return recommendations.get(quality_tier, "Evaluación pendiente.")

    def _generate_csv_fallback(
        self, results: Dict[str, IndicatorScore], output_dir: str = "."
    ) -> str:
        """
        Fallback CSV generation without pandas dependency.

        Args:
            results: Dictionary mapping plan filenames to IndicatorScore objects
            output_dir: Directory to save the CSV file

        Returns:
            Path to the generated CSV file
        """
        import csv

        # CSV headers in Spanish
        headers = [
            "Archivo del Plan",
            "Puntuación de Factibilidad",
            "Nivel de Calidad",
            "Línea Base Cuantitativa",
            "Meta Cuantitativa",
            "Componentes Detectados",
            "Tiene Línea Base",
            "Tiene Meta",
            "Tiene Horizonte Temporal",
            "Tiene Valores Numéricos",
            "Tiene Fechas",
            "Coincidencias Detalladas",
            "Recomendación General",
        ]

        # Generate rows
        rows = []
        for plan_filename, score in results.items():
            # Generate overall recommendation based on quality tier
            recommendation = self._get_recommendation_spanish(
                score.quality_tier
            )

            # Count component types
            components = score.components_detected
            has_baseline = ComponentType.BASELINE in components
            has_target = ComponentType.TARGET in components
            has_time_horizon = ComponentType.TIME_HORIZON in components
            has_numerical = ComponentType.NUMERICAL in components
            has_dates = ComponentType.DATE in components

            # Format detected components in Spanish
            component_names_spanish = {
                ComponentType.BASELINE: "línea base",
                ComponentType.TARGET: "meta",
                ComponentType.TIME_HORIZON: "horizonte temporal",
                ComponentType.NUMERICAL: "valores numéricos",
                ComponentType.DATE: "fechas",
            }

            components_list = [
                component_names_spanish.get(comp, comp.value) for comp in components
            ]
            components_str = (
                ", ".join(components_list) if components_list else "ninguno"
            )

            # Format detailed matches
            matches_details = []
            for match in score.detailed_matches:
                match_info = f"{component_names_spanish.get(match.component_type, match.component_type.value)}: '{match.matched_text}' (confianza: {match.confidence:.2f})"
                matches_details.append(match_info)
            matches_str = (
                "; ".join(matches_details)
                if matches_details
                else "ninguna coincidencia"
            )

            row = [
                plan_filename,
                f"{score.feasibility_score:.3f}",
                self._translate_quality_tier_spanish(
                    score.quality_tier),
                "Sí" if score.has_quantitative_baseline else "No",
                "Sí" if score.has_quantitative_target else "No",
                components_str,
                "Sí" if has_baseline else "No",
                "Sí" if has_target else "No",
                "Sí" if has_time_horizon else "No",
                "Sí" if has_numerical else "No",
                "Sí" if has_dates else "No",
                matches_str,
                recommendation,
            ]
            # Include score for sorting
            rows.append((score.feasibility_score, row))

        # Sort by feasibility score (descending)
        rows.sort(key=lambda x: x[0], reverse=True)
        sorted_rows = [row[1] for row in rows]

        intended_path = Path(output_dir) / "matriz_trazabilidad_factibilidad.csv"
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(headers)
        writer.writerows(sorted_rows)
        csv_content = buffer.getvalue()
        csv_bytes = csv_content.encode("utf-8-sig")
        file_size_mb = len(csv_bytes) / (1024 * 1024)

        if file_size_mb > 5.0:
            intended_path = intended_path.with_suffix(".csv.gz")
            result = safe_write_bytes(
                intended_path,
                gzip.compress(csv_bytes),
                label="feasibility_traceability_csv_fallback",
            )
            summary = "CSV exportado con compresión gzip"
        else:
            result = safe_write_text(
                intended_path,
                csv_content,
                label="feasibility_traceability_csv_fallback",
                encoding="utf-8-sig",
            )
            summary = "CSV exportado"

        self._log_safe_write_result(
            "Matriz de trazabilidad CSV (fallback)", intended_path, result
        )

        if result.status == "primary":
            print(f"{summary}: {result.path} (tamaño: {file_size_mb:.1f}MB)")
        elif result.status == "fallback":
            print(
                f"{summary} (fallback en {result.path}, original: {intended_path})"
                f" [tamaño estimado: {file_size_mb:.1f}MB]"
            )
        else:
            print(
                f"{summary} retenido en memoria (clave {result.key})"
                f" [tamaño estimado: {file_size_mb:.1f}MB]"
            )

        return str(result.path or result.key or intended_path)

    def batch_score(self, indicators: List[str]) -> List[IndicatorScore]:
        """Process a batch of indicators with optimized execution."""
        # This method would be implemented in the full version
        return [self.evaluate_indicator(indicator) for indicator in indicators]


# Example usage when run as a script
if __name__ == "__main__":
    import time
    
    # Create feasibility scorer
    scorer = FeasibilityScorer()
    
    # Example indicator text
    example_indicator = """
    Indicador: Porcentaje de cobertura en salud
    Línea base: 65% (2023)
    Meta: 85% (2027)
    Responsable: Secretaría de Salud Municipal
    """
    
    print("Evaluating indicator feasibility...")
    start_time = time.time()
    
    # Evaluate indicator
    result = scorer.evaluate_indicator(example_indicator)
    
    print(f"Evaluation completed in {time.time() - start_time:.3f} seconds")
    print(f"SMART Score: {result.smart_score:.2f}")
    print(f"Feasibility Score: {result.feasibility_score:.2f}")
    print(f"Has baseline: {result.has_baseline}")
    print(f"Has target: {result.has_target}")
    print(f"Has timeframe: {result.has_timeframe}")
    print(f"Has quantitative target: {result.has_quantitative_target}")
    print(f"Has unit: {result.has_unit}")
    print(f"Has responsible: {result.has_responsible}")
    
    # Example plan text
    example_plan = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    
    DIAGNÓSTICO
    Actualmente la cobertura en educación es del 75% y en salud del 65%.
    La tasa de desempleo ha aumentado al 12% en los últimos años.
    
    OBJETIVOS Y METAS
    1. Aumentar la cobertura educativa al 95% para el año 2027.
       Responsable: Secretaría de Educación Municipal
    
    2. Reducir la tasa de desempleo al 8% durante el cuatrienio.
       Línea base: 12% (2023)
       Meta: 8% (2027)
    
    3. Construir 500 viviendas de interés social.
       Plazo: 4 años
    
    SEGUIMIENTO
    Se realizará seguimiento trimestral a través de un tablero de control.
    """
    
    print("\nEvaluating plan feasibility...")
    start_time = time.time()
    
    # Evaluate plan
    plan_result = scorer.evaluate_plan_feasibility(example_plan)
    
    print(f"Evaluation completed in {time.time() - start_time:.3f} seconds")
    print(f"Overall feasibility score: {plan_result['overall_feasibility']:.2f}")
    print(f"Baseline-target alignment: {plan_result['baseline_target_alignment']:.2f}")
    print(f"Component counts: {plan_result['component_counts']}")
    print("\nRecommendations:")
    for rec in plan_result["recommendations"]:
        print(f"- {rec}")
    
    print("\nDECALOGO answers:")
    print(f"DE1_Q3: {plan_result['decalogo_answers']['DE1_Q3']['answer']} (confidence: {plan_result['decalogo_answers']['DE1_Q3']['confidence']:.2f})")
    print(f"DE4_Q1: {plan_result['decalogo_answers']['DE4_Q1']['answer']} (confidence: {plan_result['decalogo_answers']['DE4_Q1']['confidence']:.2f})")
    print(f"DE4_Q2: {plan_result['decalogo_answers']['DE4_Q2']['answer']} (confidence: {plan_result['decalogo_answers']['DE4_Q2']['confidence']:.2f})")


def main():
    """Command-line interface for the Feasibility Scorer."""
    parser = argparse.ArgumentParser(
        description="Feasibility Scorer - Evaluate indicator quality based on baseline, targets, and time horizons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single indicator
  python feasibility_scorer.py --text "Incrementar la línea base de 65% a una meta de 85% para 2025"
  
  # Export CSV traceability matrix 
  python feasibility_scorer.py --export-csv --output-dir ./reports/
  
  # Process batch from file
  python feasibility_scorer.py --batch-file indicators.txt --export-csv
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str, help="Single indicator text to evaluate"
    )
    input_group.add_argument(
        "--batch-file",
        type=str,
        help="Path to file containing multiple indicators (one per line)",
    )
    input_group.add_argument(
        "--demo", action="store_true", help="Run demonstration with built-in examples"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for reports (default: current directory)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Generate consolidated traceability matrix CSV file",
    )
    if args.export_json:
        json_path = output_dir / "resultados_factibilidad.json"
        json_data = {}
        for filename, score in results.items():
            json_data[filename] = {
                "feasibility_score": score.feasibility_score,
                "quality_tier": score.quality_tier,
                "components_detected": [c.value for c in score.components_detected],
                "has_quantitative_baseline": score.has_quantitative_baseline,
                "has_quantitative_target": score.has_quantitative_target,
                "detailed_matches": [
                    {
                        "component_type": match.component_type.value,
                        "matched_text": match.matched_text,
                        "confidence": match.confidence,
                        "position": match.position,
                    }
                    for match in score.detailed_matches
                ],
            }

        result = safe_write_json(json_path, json_data, label="feasibility_results_json")
        scorer._log_safe_write_result("Resultados JSON", json_path, result)
        if result.status == "primary":
            print(f"✓ Resultados JSON exportados: {result.path}")
        elif result.status == "fallback":
            print(
                f"✓ Resultados JSON exportados en fallback: {result.path} (original: {json_path})"
            )
        else:
            print(
                f"✓ Resultados JSON almacenados en memoria (clave {result.key})"
            )

    if args.export_markdown:
        md_path = output_dir / "reporte_factibilidad.md"
        md_lines: List[str] = []
        md_lines.append("# Reporte de Evaluación de Factibilidad de Indicadores\n\n")
        md_lines.append(
            f"**Fecha de generación:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        md_lines.append(
            f"**Número total de indicadores evaluados:** {len(results)}\n\n"
        )

        scores = [score.feasibility_score for score in results.values()]
        tiers = [score.quality_tier for score in results.values()]

        md_lines.append("## Resumen Estadístico\n\n")
        if scores:
            md_lines.append(
                f"- **Puntuación promedio:** {sum(scores) / len(scores):.3f}\n"
            )
            md_lines.append(f"- **Puntuación máxima:** {max(scores):.3f}\n")
            md_lines.append(f"- **Puntuación mínima:** {min(scores):.3f}\n")

            tier_counts = {tier: tiers.count(tier) for tier in set(tiers)}
            md_lines.append("\n### Distribución por Nivel de Calidad\n\n")
            for tier, count in tier_counts.items():
                tier_spanish = scorer.translate_quality_tier_spanish(tier)
                percentage = (count / len(tiers)) * 100 if tiers else 0
                md_lines.append(
                    f"- {tier_spanish}: {count} indicadores ({percentage:.1f}%)\n"
                )

        md_lines.append("\n## Resultados Detallados\n\n")
        if not results:
            md_lines.append("No se encontraron indicadores para evaluar.\n")
        else:
            sorted_results = sorted(
                results.items(), key=lambda x: x[1].feasibility_score, reverse=True
            )
            for filename, score in sorted_results:
                tier_spanish = scorer.translate_quality_tier_spanish(score.quality_tier)
                recommendation = scorer.get_recommendation_spanish(score.quality_tier)
                md_lines.append(f"### {filename}\n")
                md_lines.append(
                    f"- **Puntuación de factibilidad:** {score.feasibility_score:.3f}\n"
                )
                md_lines.append(f"- **Nivel de calidad:** {tier_spanish}\n")
                md_lines.append(f"- **Recomendación:** {recommendation}\n")

                components = score.components_detected
                has_baseline = ComponentType.BASELINE in components
                has_target = ComponentType.TARGET in components
                has_time_horizon = ComponentType.TIME_HORIZON in components
                has_numerical = ComponentType.NUMERICAL in components
                has_dates = ComponentType.DATE in components

                md_lines.append("- **Componentes detectados:**\n")
                md_lines.append(f"  - Línea base: {'Sí' if has_baseline else 'No'}\n")
                md_lines.append(f"  - Meta: {'Sí' if has_target else 'No'}\n")
                md_lines.append(
                    f"  - Horizonte temporal: {'Sí' if has_time_horizon else 'No'}\n"
                )
                md_lines.append(
                    f"  - Valores numéricos: {'Sí' if has_numerical else 'No'}\n"
                )
                md_lines.append(f"  - Fechas: {'Sí' if has_dates else 'No'}\n")

                if score.detailed_matches:
                    md_lines.append("- **Coincidencias detalladas:**\n")
                    for match in score.detailed_matches:
                        md_lines.append(
                            f"  - {match.component_type.value}: '{match.matched_text}' (confianza: {match.confidence:.2f})\n"
                        )
                else:
                    md_lines.append("- **Coincidencias detalladas:** Ninguna\n")

                md_lines.append("\n")

        md_content = "".join(md_lines)
        result = safe_write_text(md_path, md_content, label="feasibility_markdown_report")
        scorer._log_safe_write_result("Reporte Markdown", md_path, result)
        if result.status == "primary":
            print(f"✓ Reporte Markdown exportado: {result.path}")
        elif result.status == "fallback":
            print(
                f"✓ Reporte Markdown exportado en fallback: {result.path} (original: {md_path})"
            )
        else:
            print(f"✓ Reporte Markdown en memoria (clave {result.key})")

    return 0


def audit_performance_hotspots() -> Dict[str, List[str]]:
    """Resumen estático de cuellos de botella, efectos laterales y vectorización posible."""

    return {
        "bottlenecks": [
            "FeasibilityScorer.batch_score: procesa indicadores en serie cuando JOBLIB no está disponible.",
            "FeasibilityScorer.generate_traceability_matrix_csv: construye DataFrame completo en memoria antes de exportar.",
        ],
        "side_effects": [
            "FeasibilityScorer.generate_report: crea y renombra archivos temporales en disco.",
            "main: garantiza directorios de salida y puede generar múltiples archivos CSV/JSON/Markdown.",
        ],
        "vectorization_opportunities": [
            "FeasibilityScorer.detect_components: evalúa expresiones regulares secuencialmente; se puede vectorizar sobre lotes de texto.",
            "FeasibilityScorer.calculate_feasibility_score: combina operaciones de conteo que admiten paralelización segura por chunks.",
        ],
    }


if __name__ == "__main__":
    exit(main())
