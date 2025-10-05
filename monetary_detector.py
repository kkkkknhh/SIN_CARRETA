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
from dataclasses import dataclass, field
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
    """
    Detector for monetary values in plan documents.
    
    This detector identifies financial commitments, budget allocations,
    and other monetary values that are essential for evaluating:
    - Budget planning in DE-2 Thematic Inclusion
    - Resource adequacy for results in DE-4 Results Orientation
    
    Methods:
        detect_monetary_values: Extract monetary values from text
        analyze_monetary_coverage: Analyze monetary coverage of a document
        evaluate_resource_adequacy: Evaluate if resources are adequate for goals
    """
    
    def __init__(self, context_window: int = 100):
        """
        Initialize the monetary detector.
        
        Args:
            context_window: Size of context window to include around monetary values
        """
        self.context_window = context_window
        
        # Define regex patterns for different monetary formats
        
        # Basic monetary patterns with currency symbols
        self.basic_patterns = [
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
        self.budget_patterns = [
            r'(?i)(?:presupuesto|recursos?)\s+(?:de|por|:)?\s+[$€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
            r'(?i)(?:asignaci[oó]n presupuestal|recursos asignados?)\s+(?:de|por|:)?\s+[$€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
            r'(?i)(?:inversi[oó]n|fondos?)\s+(?:de|por|:)?\s+[$€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?',
        ]
        
        # Timeframe patterns
        self.timeframe_patterns = [
            (r'(?i)(?:anual(?:es)?|por año|al año|cada año)', FinancialTimeframe.ANNUAL),
            (r'(?i)(?:cuatrienio|multianual|plurianual|plan plurianual)', FinancialTimeframe.MULTIANNUAL),
            (r'(?i)(?:total|global|completo)', FinancialTimeframe.TOTAL),
            (r'(?i)(?:trimestral(?:es)?|por trimestre)', FinancialTimeframe.QUARTERLY),
        ]
        
        # Category patterns
        self.category_patterns = [
            (r'(?i)(?:presupuesto(?:s)?|recursos financieros)', MonetaryCategory.BUDGET),
            (r'(?i)(?:inversi[oó]n(?:es)?|capital)', MonetaryCategory.INVESTMENT),
            (r'(?i)(?:operaci[oó]n|funcionamiento)', MonetaryCategory.OPERATING),
            (r'(?i)(?:transferencia(?:s)?|subsidio(?:s)?)', MonetaryCategory.TRANSFER),
            (r'(?i)(?:ingreso(?:s)?|recaudaci[oó]n)', MonetaryCategory.REVENUE),
            (r'(?i)(?:donaci[oó]n(?:es)?|aporte(?:s)?|cooperaci[oó]n internacional)', MonetaryCategory.GRANT),
            (r'(?i)(?:pr[eé]stamo(?:s)?|cr[eé]dito(?:s)?|financiaci[oó]n)', MonetaryCategory.LOAN),
        ]
        
        # Source attribution patterns
        self.source_patterns = [
            r'(?i)(?:financiado|aportado|suministrado)(?:\s+por)?\s+([\w\s]+?)(?:\.|\,|\;|$)',
            r'(?i)recursos\s+(?:de|del|provenientes de)\s+([\w\s]+?)(?:\.|\,|\;|$)',
        ]
        
        # Compile patterns for efficiency
        self.compiled_basic_patterns = [re.compile(p) for p in self.basic_patterns]
        self.compiled_budget_patterns = [re.compile(p) for p in self.budget_patterns]
        self.compiled_timeframe_patterns = [(re.compile(p), t) for p, t in self.timeframe_patterns]
        self.compiled_category_patterns = [(re.compile(p), c) for p, c in self.category_patterns]
        self.compiled_source_patterns = [re.compile(p) for p in self.source_patterns]
    
    def detect_monetary_values(self, text: str) -> List[MonetaryMatch]:
        """
        Detect monetary values in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected MonetaryMatch objects
        """
        if not text:
            return []
        
        matches = []
        
        # Detect basic monetary patterns
        for pattern in self.compiled_basic_patterns:
            for match in pattern.finditer(text):
                raw_text = match.group(0).strip()
                
                # Extract amount and currency
                amount, currency = self._extract_amount_and_currency(raw_text)
                
                if amount is not None and currency:
                    # Get start and end positions
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context around match
                    context_start = max(0, start_pos - self.context_window)
                    context_end = min(len(text), end_pos + self.context_window)
                    context = text[context_start:context_end]
                    
                    # Determine category and timeframe
                    category = self._detect_category(context, start_pos - context_start)
                    timeframe = self._detect_timeframe(context, start_pos - context_start)
                    
                    # Extract source if available
                    source = self._extract_source(context, start_pos - context_start)
                    
                    # Calculate confidence score
                    confidence = self._calculate_confidence(raw_text, amount, currency, category, timeframe)
                    
                    # Create match object
                    monetary_match = MonetaryMatch(
                        raw_text=raw_text,
                        amount=amount,
                        currency=currency,
                        category=category,
                        timeframe=timeframe,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        context=context,
                        confidence=confidence,
                        source=source
                    )
                    
                    matches.append(monetary_match)
        
        # Detect budget-specific patterns
        for pattern in self.compiled_budget_patterns:
            for match in pattern.finditer(text):
                raw_text = match.group(0).strip()
                
                # Extract amount and currency
                amount, currency = self._extract_amount_and_currency(raw_text)
                
                if amount is not None and currency:
                    # This is definitely a budget item
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context around match
                    context_start = max(0, start_pos - self.context_window)
                    context_end = min(len(text), end_pos + self.context_window)
                    context = text[context_start:context_end]
                    
                    # Budget-specific patterns are already categorized
                    category = MonetaryCategory.BUDGET
                    timeframe = self._detect_timeframe(context, start_pos - context_start)
                    
                    # Extract source if available
                    source = self._extract_source(context, start_pos - context_start)
                    
                    # Calculate confidence score - higher for budget-specific patterns
                    confidence = self._calculate_confidence(raw_text, amount, currency, category, timeframe, is_budget_pattern=True)
                    
                    # Create match object
                    monetary_match = MonetaryMatch(
                        raw_text=raw_text,
                        amount=amount,
                        currency=currency,
                        category=category,
                        timeframe=timeframe,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        context=context,
                        confidence=confidence,
                        source=source
                    )
                    
                    matches.append(monetary_match)
        
        # Remove duplicates by comparing start positions
        unique_matches = []
        start_positions = set()
        
        for match in sorted(matches, key=lambda m: m.confidence, reverse=True):
            if match.start_pos not in start_positions:
                unique_matches.append(match)
                start_positions.add(match.start_pos)
        
        return unique_matches
    
    def analyze_monetary_coverage(self, text: str) -> MonetaryAnalysis:
        """
        Analyze monetary coverage of a document.
        
        Args:
            text: Text to analyze
            
        Returns:
            MonetaryAnalysis with detailed monetary coverage information
        """
        # Detect monetary values
        matches = self.detect_monetary_values(text)
        
        if not matches:
            # Return empty analysis
            return MonetaryAnalysis(
                matches=[],
                total_allocations=0,
                total_amount={},
                currency_breakdown={},
                category_breakdown={},
                timeframe_breakdown={},
                resource_adequacy_score=0.0,
                budget_coverage=0.0
            )
        
        # Calculate total amount by currency
        total_amount = {}
        for match in matches:
            if match.currency in total_amount:
                total_amount[match.currency] += match.amount
            else:
                total_amount[match.currency] = match.amount
        
        # Count currencies
        currency_breakdown = {}
        for match in matches:
            if match.currency in currency_breakdown:
                currency_breakdown[match.currency] += 1
            else:
                currency_breakdown[match.currency] = 1
        
        # Count categories
        category_breakdown = {}
        for match in matches:
            category = match.category.value
            if category in category_breakdown:
                category_breakdown[category] += 1
            else:
                category_breakdown[category] = 1
        
        # Count timeframes
        timeframe_breakdown = {}
        for match in matches:
            timeframe = match.timeframe.value
            if timeframe in timeframe_breakdown:
                timeframe_breakdown[timeframe] += 1
            else:
                timeframe_breakdown[timeframe] = 1
        
        # Calculate resource adequacy score
        resource_adequacy_score = self._calculate_resource_adequacy_score(matches, text)
        
        # Calculate budget coverage
        budget_coverage = self._calculate_budget_coverage(matches, text)
        
        return MonetaryAnalysis(
            matches=matches,
            total_allocations=len(matches),
            total_amount=total_amount,
            currency_breakdown=currency_breakdown,
            category_breakdown=category_breakdown,
            timeframe_breakdown=timeframe_breakdown,
            resource_adequacy_score=resource_adequacy_score,
            budget_coverage=budget_coverage
        )
    
    def evaluate_resource_adequacy(self, text: str, goal_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate if resources are adequate for stated goals (for DE-4).
        
        Args:
            text: Text to analyze
            goal_keywords: Optional list of goal-related keywords to check against
            
        Returns:
            Dictionary with resource adequacy evaluation
        """
        # Default goal keywords if none provided
        if goal_keywords is None:
            goal_keywords = [
                "meta", "objetivo", "resultado", "impacto", "logro", 
                "alcanzar", "cumplir", "conseguir", "obtener"
            ]
        
        # Analyze monetary coverage
        analysis = self.analyze_monetary_coverage(text)
        
        # Count resources associated with goals
        resource_goal_matches = 0
        goal_related_amount = {}
        
        for match in analysis.matches:
            context = match.context.lower()
            if any(keyword.lower() in context for keyword in goal_keywords):
                resource_goal_matches += 1
                if match.currency in goal_related_amount:
                    goal_related_amount[match.currency] += match.amount
                else:
                    goal_related_amount[match.currency] = match.amount
        
        # Percentage of resources explicitly linked to goals
        goal_linkage_ratio = resource_goal_matches / len(analysis.matches) if analysis.matches else 0
        
        # Different resource timeframes
        resource_timeframes = len(analysis.timeframe_breakdown)
        
        # Calculate implementation feasibility based on resources
        implementation_feasibility = min(1.0, (
            analysis.resource_adequacy_score * 0.5 +
            goal_linkage_ratio * 0.3 +
            min(1.0, resource_timeframes / 3) * 0.2
        ))
        
        # Recommendation based on analysis
        if implementation_feasibility >= 0.7:
            recommendation = "Los recursos financieros parecen adecuados para los objetivos planteados."
        elif implementation_feasibility >= 0.4:
            recommendation = "Los recursos financieros son parcialmente adecuados. Se recomienda aclarar o aumentar las asignaciones presupuestales."
        else:
            recommendation = "Los recursos financieros parecen insuficientes para los objetivos planteados. Se requiere una mayor asignación o clarificación presupuestal."
        
        return {
            "resource_adequacy_score": analysis.resource_adequacy_score,
            "budget_coverage": analysis.budget_coverage,
            "total_monetary_allocations": len(analysis.matches),
            "goal_related_allocations": resource_goal_matches,
            "goal_linkage_ratio": goal_linkage_ratio,
            "implementation_feasibility": implementation_feasibility,
            "recommendation": recommendation,
            "de2_budget_evaluation": analysis.budget_coverage >= 0.5,
            "de4_resource_adequacy": analysis.resource_adequacy_score >= 0.6,
            "confidence": sum(m.confidence for m in analysis.matches) / len(analysis.matches) if analysis.matches else 0
        }
    
    def _extract_amount_and_currency(self, text: str) -> Tuple[Optional[float], str]:
        """
        Extract amount and currency from monetary text.
        
        Args:
            text: Text containing monetary value
            
        Returns:
            Tuple of (amount, currency)
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
                amount = float(amount_str)
                
                # Check if followed by COP or pesos
                if re.search(r'COP|pesos?', text, re.IGNORECASE):
                    return amount, "COP"
                else:
                    return amount, "USD"
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
                return float(amount_str), "EUR"
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
        for pattern, category in self.compiled_category_patterns:
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
        for pattern, timeframe in self.compiled_timeframe_patterns:
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
        for pattern in self.compiled_source_patterns:
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
    
    def _calculate_confidence(
        self, 
        raw_text: str, 
        amount: float, 
        currency: str, 
        category: MonetaryCategory, 
        timeframe: FinancialTimeframe,
        is_budget_pattern: bool = False
    ) -> float:
        """
        Calculate confidence score for a monetary match.
        
        Args:
            raw_text: Original matched text
            amount: Extracted amount
            currency: Extracted currency
            category: Detected category
            timeframe: Detected timeframe
            is_budget_pattern: Whether match was from a budget-specific pattern
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.6  # Base confidence
        
        # Adjust for specific patterns
        if is_budget_pattern:
            confidence += 0.2  # Budget patterns are more reliable
        
        # Adjust for amount and currency clarity
        if amount > 0 and currency:
            confidence += 0.1
        
        # Adjust for category and timeframe
        if category != MonetaryCategory.UNDEFINED:
            confidence += 0.05
        
        if timeframe != FinancialTimeframe.UNDEFINED:
            confidence += 0.05
        
        # Check for possible false positives
        if len(raw_text) < 5:  # Very short matches are suspicious
            confidence -= 0.1
        
        # Cap at 1.0
        return min(1.0, confidence)
    
    def _calculate_resource_adequacy_score(self, matches: List[MonetaryMatch], text: str) -> float:
        """
        Calculate a score indicating resource adequacy for DE-4 evaluation.
        
        Args:
            matches: List of monetary matches
            text: Full document text
            
        Returns:
            Resource adequacy score between 0 and 1
        """
        if not matches:
            return 0.0
        
        # Factors that contribute to resource adequacy
        factors = []
        
        # Factor 1: Number of monetary allocations (more is better, up to a point)
        allocation_count_factor = min(1.0, len(matches) / 10)  # Cap at 10 allocations
        factors.append(allocation_count_factor * 0.2)  # 20% weight
        
        # Factor 2: Variety of categories (more diverse allocations are better)
        categories = set(m.category for m in matches)
        category_factor = min(1.0, len(categories) / 5)  # Cap at 5 categories
        factors.append(category_factor * 0.2)  # 20% weight
        
        # Factor 3: Timeframe specification (specified timeframes are better)
        defined_timeframes = sum(1 for m in matches if m.timeframe != FinancialTimeframe.UNDEFINED)
        timeframe_factor = defined_timeframes / len(matches) if matches else 0
        factors.append(timeframe_factor * 0.15)  # 15% weight
        
        # Factor 4: Budget category presence (specific budget allocations are important)
        budget_matches = sum(1 for m in matches if m.category == MonetaryCategory.BUDGET)
        budget_factor = min(1.0, budget_matches / 3)  # Cap at 3 budget allocations
        factors.append(budget_factor * 0.25)  # 25% weight
        
        # Factor 5: Source specification (attributing funding sources is good)
        sourced_matches = sum(1 for m in matches if m.source)
        source_factor = sourced_matches / len(matches) if matches else 0
        factors.append(source_factor * 0.1)  # 10% weight
        
        # Factor 6: Confidence in extractions
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        factors.append(avg_confidence * 0.1)  # 10% weight
        
        # Combine factors
        return sum(factors)
    
    def _calculate_budget_coverage(self, matches: List[MonetaryMatch], text: str) -> float:
        """
        Calculate budget coverage for DE-2 evaluation.
        
        Args:
            matches: List of monetary matches
            text: Full document text
            
        Returns:
            Budget coverage score between 0 and 1
        """
        if not matches:
            return 0.0
        
        # Count budget-related matches
        budget_matches = sum(1 for m in matches if m.category == MonetaryCategory.BUDGET)
        
        # Calculate coverage score
        coverage_score = min(1.0, budget_matches / 5)  # Cap at 5 budget allocations
        
        # Check if budget allocations have timeframes
        budget_with_timeframe = sum(1 for m in matches 
                                  if m.category == MonetaryCategory.BUDGET and 
                                  m.timeframe != FinancialTimeframe.UNDEFINED)
        
        timeframe_ratio = budget_with_timeframe / budget_matches if budget_matches else 0
        
        # Final coverage score adjusts for timeframe specification
        return coverage_score * (0.7 + 0.3 * timeframe_ratio)


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
    
    # Analyze monetary coverage
    analysis = detector.analyze_monetary_coverage(sample_text)
    
    print(f"Found {analysis.total_allocations} monetary allocations")
    print(f"Resource adequacy score: {analysis.resource_adequacy_score:.2f}")
    print(f"Budget coverage: {analysis.budget_coverage:.2f}")
    print("\nMonetary values:")
    for match in analysis.matches:
        print(f"- {match.raw_text}: {match.amount} {match.currency} ({match.category.value})")
    
    # Evaluate resource adequacy
    evaluation = detector.evaluate_resource_adequacy(sample_text)
    print(f"\nImplementation feasibility: {evaluation['implementation_feasibility']:.2f}")
    print(f"Recommendation: {evaluation['recommendation']}")
