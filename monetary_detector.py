"""
Monetary Detection Module

Extracts financial commitments and monetary values from plan documents,
essential for budget planning evaluation (DE-2) and resource adequacy
assessment (DE-4).

Features:
- Currency symbol detection
- Numeric value extraction with contextual analysis
- Budget allocation identification
- Financial period recognition
- Resource adequacy scoring
- Source attribution for financial commitments
- Confidence scoring for extracted values
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonetaryCategory(Enum):
    """Categories of monetary values found in plans."""
    BUDGET = "budget"              # General budget allocations
    INVESTMENT = "investment"      # Specific investments
    OPERATING = "operating"        # Operating costs
    TRANSFER = "transfer"          # Transfers or subsidies
    REVENUE = "revenue"            # Expected revenues
    GRANT = "grant"                # External grants
    LOAN = "loan"                  # Loans or financing
    UNDEFINED = "undefined"        # Category not clearly defined


class FinancialTimeframe(Enum):
    """Timeframes for financial commitments."""
    ANNUAL = "annual"              # Yearly allocation
    MULTIANNUAL = "multiannual"    # Multi-year allocation
    TOTAL = "total"                # Total amount for entire plan
    QUARTERLY = "quarterly"        # Quarterly allocation
    UNDEFINED = "undefined"        # Timeframe not specified


@dataclass
class MonetaryMatch:
    """
    A monetary value detected in text with contextual information.
    
    Attributes:
        raw_text: Original text containing the monetary value
        amount: Normalized numeric amount
        currency: Currency identifier (e.g., COP, USD, $)
        category: Type of monetary allocation (budget, investment, etc.)
        timeframe: Temporal scope of the allocation (annual, total, etc.)
        start_pos: Starting position in text
        end_pos: Ending position in text
        context: Surrounding text for context
        confidence: Confidence score for the extraction (0-1)
        source: Attributed source of funding if specified
    """
    raw_text: str
    amount: float
    currency: str
    category: MonetaryCategory = MonetaryCategory.UNDEFINED
    timeframe: FinancialTimeframe = FinancialTimeframe.UNDEFINED
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""
    confidence: float = 1.0
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "raw_text": self.raw_text,
            "amount": self.amount,
            "currency": self.currency,
            "category": self.category.value,
            "timeframe": self.timeframe.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "context_excerpt": self.context[:100] + "..." if len(self.context) > 100 else self.context,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class MonetaryAnalysis:
    """
    Analysis of monetary values detected in a document.
    
    Attributes:
        matches: List of monetary matches found
        total_allocations: Total monetary allocations detected
        total_amount: Sum of all monetary values
        currency_breakdown: Count of values by currency
        category_breakdown: Count of values by category
        timeframe_breakdown: Count of values by timeframe
        resource_adequacy_score: Score indicating resource adequacy (0-1)
        budget_coverage: Percentage of plan covered by explicit budget
    """
    matches: List[MonetaryMatch]
    total_allocations: int
    total_amount: Dict[str, float]
    currency_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
    timeframe_breakdown: Dict[str, int]
    resource_adequacy_score: float
    budget_coverage: float
    
    def get_matches_by_category(self, category: MonetaryCategory) -> List[MonetaryMatch]:
        """Get monetary matches for a specific category."""
        return [m for m in self.matches if m.category == category]
    
    def get_matches_by_timeframe(self, timeframe: FinancialTimeframe) -> List[MonetaryMatch]:
        """Get monetary matches for a specific timeframe."""
        return [m for m in self.matches if m.timeframe == timeframe]
    
    def has_adequate_resources(self) -> bool:
        """Check if the plan has adequate resources based on the score."""
        return self.resource_adequacy_score >= 0.7
    
    def get_primary_currency(self) -> Optional[str]:
        """Get the most frequent currency in the document."""
        if not self.currency_breakdown:
            return None
        return max(self.currency_breakdown.items(), key=lambda x: x[1])[0]


class MonetaryDetector:
    """Industrial-grade monetary value detector with auditable registry integration."""

    def __init__(self) -> None:
        """Initialize detector with default configuration."""

        self.logger = logging.getLogger(__name__)
        self.evidence_registry: Optional[Any] = None

        # Compile all regex patterns
        self.patterns = self._compile_patterns()
        self._trace_state("Initialized")

    def attach_evidence_registry(self, evidence_registry: Any) -> None:
        """Attach an evidence registry for optional auto-registration support."""

        if not hasattr(evidence_registry, "register"):
            raise TypeError("Evidence registry must expose a 'register' method")
        self.evidence_registry = evidence_registry
        self._trace_state("Registry attached")

    def detach_evidence_registry(self) -> None:
        """Detach the currently configured evidence registry."""

        self.evidence_registry = None
        self._trace_state("Registry detached")

    def _trace_state(self, event: str) -> None:
        registry_type = type(self.evidence_registry).__name__ if self.evidence_registry else "None"
        self.logger.info(
            "[MonetaryDetector] %s | registry=%s",
            event,
            registry_type,
        )

    @staticmethod
    def _compile_patterns() -> Dict[str, re.Pattern]:
        """Compile all regex patterns for monetary detection."""
        # Define regex patterns for different monetary formats
        
        # Basic monetary patterns with currency symbols
        basic_patterns = [
            # Dollar with symbol before amount: $1,000 or $1.000
            r'[$USD]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
            
            # Euro with symbol before amount: €1,000 or €1.000
            r'[€EUR]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
            
            # Colombian Peso with symbol before amount: $1,000 or $1.000 followed by COP
            r'[$]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*(?:COP|pesos?)',
            
            # Amount followed by currency: 1,000 USD or 1.000 EUR
            r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*(?:USD|EUR|COP|pesos?|dólares?|euros?)',
            
            # Amount with million/billion: 1 million USD or 2.5 millones de pesos
            r'\d+(?:[.,]\d+)?\s*(?:mill[oó]n(?:es)?|bill[oó]n(?:es)?)\s*(?:de)?\s*(?:USD|EUR|COP|pesos?|dólares?|euros?)',
        ]
        
        # Budget-specific patterns (for DE-2)
        budget_patterns = [
            r'(?i)(?:presupuesto|recursos?)\s+(?:de|por|:)?\s+[$€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
            r'(?i)(?:asignaci[oó]n presupuestal|recursos asignados?)\s+(?:de|por|:)?\s+[$€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
            r'(?i)(?:inversi[oó]n|fondos?)\s+(?:de|por|:)?\s+[$€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
        ]
        
        # Timeframe patterns
        timeframe_patterns = [
            (r'(?i)(?:anual(?:es)?|por año|al año|cada año)', FinancialTimeframe.ANNUAL),
            (r'(?i)(?:cuatrienio|multianual|plurianual|plan plurianual)', FinancialTimeframe.MULTIANNUAL),
            (r'(?i)(?:total|global|completo)', FinancialTimeframe.TOTAL),
            (r'(?i)(?:trimestral(?:es)?|por trimestre)', FinancialTimeframe.QUARTERLY),
        ]
        
        # Category patterns
        category_patterns = [
            (r'(?i)(?:presupuesto(?:s)?|recursos financieros)', MonetaryCategory.BUDGET),
            (r'(?i)(?:inversi[oó]n(?:es)?|capital)', MonetaryCategory.INVESTMENT),
            (r'(?i)(?:operaci[oó]n|funcionamiento)', MonetaryCategory.OPERATING),
            (r'(?i)(?:transferencia(?:s)?|subsidio(?:s)?)', MonetaryCategory.TRANSFER),
            (r'(?i)(?:ingreso(?:s)?|recaudaci[oó]n)', MonetaryCategory.REVENUE),
            (r'(?i)(?:donaci[oó]n(?:es)?|aporte(?:s)?|cooperaci[oó]n internacional)', MonetaryCategory.GRANT),
            (r'(?i)(?:pr[eé]stamo(?:s)?|cr[eé]dito(?:s)?|financiaci[oó]n)', MonetaryCategory.LOAN),
        ]
        
        # Source attribution patterns
        source_patterns = [
            r'(?i)(?:financiado|aportado|suministrado)(?:\s+por)?\s+([\w\s]+?)(?:\.|\,|\;|$)',
            r'(?i)recursos\s+(?:de|del|provenientes de)\s+([\w\s]+?)(?:\.|\,|\;|$)',
        ]
        
        # Compile patterns for efficiency
        compiled_basic_patterns = [re.compile(p) for p in basic_patterns]
        compiled_budget_patterns = [re.compile(p) for p in budget_patterns]
        compiled_timeframe_patterns = [(re.compile(p), t) for p, t in timeframe_patterns]
        compiled_category_patterns = [(re.compile(p), c) for p, c in category_patterns]
        compiled_source_patterns = [re.compile(p) for p in source_patterns]

        return {
            "basic": compiled_basic_patterns,
            "budget": compiled_budget_patterns,
            "timeframe": compiled_timeframe_patterns,
            "category": compiled_category_patterns,
            "source": compiled_source_patterns
        }

    def detect(self, text: str, plan_name: str = "unknown") -> List[Dict[str, Any]]:
        """
        Detect monetary values with NORMALIZED OUTPUT SCHEMA.

        Returns:
            List of dicts with strict schema:
            {
                "amount": float,
                "currency": str,
                "original_text": str,
                "category": str,  # MonetaryCategory.value
                "timeframe": str,  # FinancialTimeframe.value
                "confidence": float,  # 0.0-1.0
                "impact_weight": float,  # 0.0-1.0 (importancia relativa)
                "context": str,
                "source_position": int,
                "applicable_questions": List[str],
                "provenance": Dict[str, str]
            }
        """
        matches = []
        
        # Detectar todos los patrones
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                # Extraer monto numérico
                amount = self._parse_amount(match.group())

                # Clasificar categoría
                category = self._classify_category(match.group(), text[max(0, match.start()-100):match.end()+100])

                # Determinar timeframe
                timeframe = self._detect_timeframe(text[max(0, match.start()-50):match.end()+50])

                # Calcular confidence DETERMINISTA
                confidence = self._calculate_confidence(match.group(), amount, category)

                # Calcular peso de impacto
                impact_weight = self._calculate_impact_weight(amount, category)

                # Mapear a preguntas aplicables
                applicable_qs = self._map_to_questions(category, timeframe, amount)

                monetary_match = {
                    "amount": amount,
                    "currency": "COP",  # Asumir pesos colombianos
                    "original_text": match.group(),
                    "category": category.value if isinstance(category, MonetaryCategory) else category,
                    "timeframe": timeframe.value if isinstance(timeframe, FinancialTimeframe) else timeframe,
                    "confidence": confidence,
                    "impact_weight": impact_weight,
                    "context": text[max(0, match.start()-100):match.end()+100],
                    "source_position": match.start(),
                    "applicable_questions": applicable_qs,
                    "provenance": {
                        "plan_name": plan_name,
                        "detector": "monetary_detector",
                        "pattern_type": pattern_name
                    }
                }

                matches.append(monetary_match)

                # Registrar automáticamente si hay registry
                if self.evidence_registry:
                    self.evidence_registry.register(
                        source_component="monetary_detector",
                        evidence_type="monetary_value",
                        content=monetary_match,
                        confidence=confidence,
                        applicable_questions=applicable_qs
                    )

        # Ordenar por posición
        matches.sort(key=lambda x: x["source_position"])

        return matches

    @staticmethod
    def _parse_amount(text: str) -> float:
        """
        Parse amount from text, handling different numeric formats.

        Args:
            text: Text containing the amount

        Returns:
            Normalized float amount
        """
        # Check for standard formats
        
        # Format: $1,000 or $1.000
        dollar_match = re.search(r'[$]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)', text)
        if dollar_match:
            # Process amount by removing commas and converting periods to decimal points
            amount_str = re.sub(r'\.(?=\d{3})', '', dollar_match.group(1))  # Remove dots in thousands
            amount_str = re.sub(r',(?=\d{3})', '', amount_str)  # Remove commas in thousands
            amount_str = amount_str.replace(',', '.')  # Convert remaining commas to decimal points
            try:
                return float(amount_str)
            except ValueError:
                pass
        
        # Format: €1,000 or €1.000
        euro_match = re.search(r'[€]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)', text)
        if euro_match:
            # Process amount
            amount_str = re.sub(r'\.(?=\d{3})', '', euro_match.group(1))
            amount_str = re.sub(r',(?=\d{3})', '', amount_str)
            amount_str = amount_str.replace(',', '.')
            try:
                return float(amount_str)
            except ValueError:
                pass
        
        # Format: 1,000 USD or 1.000 EUR
        numeric_match = re.search(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(USD|EUR|COP|pesos?|dólares?|euros?)', text, re.IGNORECASE)
        if numeric_match:
            # Process amount
            amount_str = re.sub(r'\.(?=\d{3})', '', numeric_match.group(1))
            amount_str = re.sub(r',(?=\d{3})', '', amount_str)
            amount_str = amount_str.replace(',', '.')
            currency = numeric_match.group(2).upper()
            
            # Normalize currency
            if re.match(r'PESOS?', currency, re.IGNORECASE):
                currency = "COP"
            elif re.match(r'D[OÓ]LARES?', currency, re.IGNORECASE):
                currency = "USD"
            elif re.match(r'EUROS?', currency, re.IGNORECASE):
                currency = "EUR"
            
            try:
                return float(amount_str), currency
            except ValueError:
                pass
        
        # Format: 1 million USD or 2.5 millones de pesos
        million_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:mill[oó]n(?:es)?|bill[oó]n(?:es)?)\s*(?:de)?\s*(USD|EUR|COP|pesos?|dólares?|euros?)', text, re.IGNORECASE)
        if million_match:
            # Process amount
            amount_str = million_match.group(1).replace(',', '.')
            multiplier = 1000000  # Default to millions
            
            # Check for billions
            if re.search(r'bill[oó]n(?:es)?', million_match.group(0), re.IGNORECASE):
                multiplier = 1000000000
            
            currency = million_match.group(2).upper()
            
            # Normalize currency
            if re.match(r'PESOS?', currency, re.IGNORECASE):
                currency = "COP"
            elif re.match(r'D[OÓ]LARES?', currency, re.IGNORECASE):
                currency = "USD"
            elif re.match(r'EUROS?', currency, re.IGNORECASE):
                currency = "EUR"
            
            try:
                amount = float(amount_str) * multiplier
                return amount, currency
            except ValueError:
                pass
        
        return None, ""
    
    def _detect_category(self, context: str, position: int) -> MonetaryCategory:
        """
        Detect monetary category from context.
        
        Args:
            context: Context text
            position: Position of monetary value in context
            
        Returns:
            MonetaryCategory enum
        """
        # Check for category patterns
        for pattern, category in self.patterns["category"]:
            # Check in a window around the position
            start_window = max(0, position - 50)
            end_window = min(len(context), position + 50)
            window = context[start_window:end_window]
            
            if pattern.search(window):
                return category
        
        return MonetaryCategory.UNDEFINED
    
    def _detect_timeframe(self, context: str, position: int) -> FinancialTimeframe:
        """
        Detect financial timeframe from context.
        
        Args:
            context: Context text
            position: Position of monetary value in context
            
        Returns:
            FinancialTimeframe enum
        """
        # Check for timeframe patterns
        for pattern, timeframe in self.patterns["timeframe"]:
            # Check in a window around the position
            start_window = max(0, position - 50)
            end_window = min(len(context), position + 150)
            window = context[start_window:end_window]
            
            if pattern.search(window):
                return timeframe
        
        return FinancialTimeframe.UNDEFINED
    
    def _extract_source(self, context: str, position: int) -> str:
        """
        Extract funding source from context.
        
        Args:
            context: Context text
            position: Position of monetary value in context
            
        Returns:
            Source string if found, empty string otherwise
        """
        # Check for source attribution patterns
        for pattern in self.patterns["source"]:
            # Check primarily after the monetary value
            end_window = min(len(context), position + 200)
            window = context[position:end_window]
            
            match = pattern.search(window)
            if match and match.group(1):
                return match.group(1).strip()
            
            # If not found, check before the monetary value
            start_window = max(0, position - 150)
            window = context[start_window:position]
            
            match = pattern.search(window)
            if match and match.group(1):
                return match.group(1).strip()
        
        return ""
    
    @staticmethod
    def _calculate_confidence(text: str, amount: float, category) -> float:
        """Calcular confidence DETERMINISTA"""
        confidence = 0.7  # Base

        # Boost por contexto explícito
        if "presupuesto" in text.lower() or "inversión" in text.lower():
            confidence += 0.15

        # Boost por formato numérico claro
        if "$" in text:
            confidence += 0.1
        
        # Boost por montos redondos (más probables de ser presupuestos)
        if amount % 1000000 == 0:  # Múltiplo de millón
            confidence += 0.05
        
        return min(1.0, confidence)

    @staticmethod
    def _calculate_impact_weight(amount: float, category) -> float:
        """Calcular peso de impacto relativo basado en monto y categoría"""
        # Normalizar por magnitud (log scale)
        if amount == 0:
            return 0.0
        
        import math
        magnitude = min(1.0, math.log10(amount) / 12)  # Normalizar hasta billones

        # Ajustar por categoría
        category_multipliers = {
            MonetaryCategory.INVESTMENT: 1.0,
            MonetaryCategory.BUDGET: 0.9,
            MonetaryCategory.OPERATING: 0.7,
            MonetaryCategory.REVENUE: 0.8,
            MonetaryCategory.TRANSFER: 0.85,
            MonetaryCategory.UNDEFINED: 0.5
        }

        multiplier = category_multipliers.get(category, 0.5)

        return magnitude * multiplier

    @staticmethod
    def _map_to_questions(category, timeframe, amount: float) -> List[str]:
        """Mapear detección a preguntas específicas del cuestionario"""
        questions = []

        # Preguntas sobre presupuesto y recursos (D3)
        questions.extend([f"D3-Q{i}" for i in [1, 5, 10, 15, 20, 25]])

        # Si es inversión grande, añadir D1 (estrategia)
        if amount > 1000000000:  # > mil millones
            questions.extend([f"D1-Q{i}" for i in [5, 10]])

        # Si tiene timeframe específico, añadir D2 (planificación)
        if timeframe != FinancialTimeframe.UNDEFINED:
            questions.extend([f"D2-Q{i}" for i in [3, 8]])

        return questions


def create_monetary_detector() -> MonetaryDetector:
    """
    Factory function to create a monetary detector.
    
    Returns:
        Initialized MonetaryDetector instance
    """
    return MonetaryDetector()


# Example usage
if __name__ == "__main__":
    detector = create_monetary_detector()
    
    # Example text with monetary values
    sample_text = """
    El presupuesto total para el plan de desarrollo es de $500.000 millones de pesos para el cuatrienio.
    
    Para el sector educativo se asignan $100 millones de pesos anuales, que serán
    ejecutados por la Secretaría de Educación Municipal. La inversión en infraestructura
    será de 50 millones de dólares, financiados mediante un préstamo del Banco Mundial.
    
    El programa de vivienda social tendrá un presupuesto de 20.000 millones COP,
    para la construcción de 1000 unidades habitacionales.
    """
    
    # Detectar valores monetarios
    detections = detector.detect(sample_text, plan_name="Plan de Desarrollo 2023")

    print(f"Se encontraron {len(detections)} detecciones monetarias:")
    for detection in detections:
        print(f"- {detection['original_text']}: {detection['amount']} {detection['currency']} "
              f"({detection['category']}, {detection['timeframe']}) - Confianza: {detection['confidence']:.2f}, "
              f"Peso de impacto: {detection['impact_weight']:.2f}")

    # Nota: El registro automático en el evidence_registry se realiza en el método detect()
