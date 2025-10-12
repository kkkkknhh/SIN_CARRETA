"""
Responsibility Detection Module

Detects and classifies entities responsible for plan implementation,
with specific focus on answering DE-1 Q2: "Are institutional responsibilities clearly defined?"

Features:
- Entity detection with NER (PERSON, ORG)
- Government institution pattern matching
- Official position detection
- Entity type classification
- Confidence scoring
- Hierarchical entity resolution
- Role classification

Output includes entity type, confidence, and role classification for comprehensive
evaluation of institutional responsibility definition in plans.
"""

import inspect
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import spaCy for NER
try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Using pattern-based fallback for entity detection.")


class EntityType(Enum):
    """Types of responsibility entities."""
    GOVERNMENT = "government"  # Government institutions (ministries, secretariats)
    POSITION = "position"      # Official positions (mayor, director)
    INSTITUTION = "institution"  # Generic institutions (schools, hospitals)
    PERSON = "person"          # Named individuals


@dataclass
class ResponsibilityEntity:
    """
    An entity identified as responsible for implementation.
    
    Attributes:
        text: The text representation of the entity
        entity_type: Type of entity (government, position, institution, person)
        confidence: Confidence score (0-1)
        start_pos: Starting position in text
        end_pos: Ending position in text
        context: Surrounding context text
        parent_entity: Parent entity (if any)
        has_explicit_role: Whether an explicit role was detected
        role_description: Description of the entity's role
    """
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    parent_entity: Optional[str] = None
    has_explicit_role: bool = False
    role_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "context": self.context,
            "parent_entity": self.parent_entity,
            "has_explicit_role": self.has_explicit_role,
            "role_description": self.role_description,
        }


class ResponsibilityDetector:
    """
    Detects institutional responsibilities and accountability structures in municipal plans.

    **NORMALIZED OUTPUTS**: All outputs follow strict typing with confidence scores.
    **EVIDENCE REGISTRY**: Automatically registers findings with applicable questions.
    """
    
    DEFAULT_MODEL_NAME = "es_core_news_sm"

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the detector with default configuration."""
        self.logger = logging.getLogger(__name__)
        self.model_name = kwargs.pop("model_name", self.DEFAULT_MODEL_NAME)
        self.evidence_registry = kwargs.pop("evidence_registry", None)

        model_loader = kwargs.pop("model_loader", None)
        if args:
            if len(args) > 1:
                raise TypeError("ResponsibilityDetector accepts at most one positional argument")
            model_loader = args[0]

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if model_loader is not None:
            self.nlp = self._load_from_model_loader(model_loader)
        else:
            self.nlp = self._load_spacy_model(self.model_name)

        # Define patterns for different entity types
        
        # High-priority government institution patterns
        self.government_patterns = [
            r"(?i)(?:ministerio|secretar[ií]a|direcci[oó]n) (?:de|del|de la) (?:\w+ ?){1,5}",
            r"(?i)(?:alcald[ií]a|gobernaci[oó]n) (?:de|del|de la)? (?:\w+ ?){1,5}",
            r"(?i)(?:instituto|agencia|autoridad|consejo|fondo) (?:\w+ ?){1,5}",
            r"(?i)(?:superintendencia|comisi[oó]n|unidad) (?:de|del|de la)? (?:\w+ ?){1,5}",
            r"(?i)(?:departamento|municipio|distrito) (?:de|del|de la)? (?:\w+ ?){1,5}",
        ]
        
        # Official position patterns
        self.position_patterns = [
            r"(?i)(?:alcalde|gobernador|presidente|ministro|secretario|director) (?:de|del|de la)? (?:\w+ ?){0,5}",
            r"(?i)(?:coordinador|jefe|gerente|supervisor) (?:de|del|de la)? (?:\w+ ?){1,5}",
            r"(?i)(?:profesional|especialista|técnico) (?:de|del|de la)? (?:\w+ ?){1,5}",
            r"(?i)(?:líder|responsable|encargado) (?:de|del|de la)? (?:\w+ ?){1,5}",
        ]
        
        # Generic institution patterns (lower priority)
        self.institution_patterns = [
            r"(?i)(?:entidad|organización|institución) (?:de|del|de la)? (?:\w+ ?){1,5}",
            r"(?i)(?:universidad|colegio|escuela|hospital|centro) (?:\w+ ?){1,5}",
            r"(?i)(?:fundación|corporación|asociación|cooperativa) (?:\w+ ?){1,5}",
            r"(?i)(?:empresa|compañía|sociedad) (?:\w+ ?){1,5}",
        ]
        
        # Role detection patterns
        self.role_patterns = [
            r"(?i)(?:responsable|encargado) (?:de|del|de la) (?:\w+ ?){1,10}",
            r"(?i)a cargo de(?: la)? (?:\w+ ?){1,10}",
            r"(?i)(?:lidera|coordina|supervisa|ejecuta) (?:\w+ ?){1,10}",
            r"(?i)(?:función|rol|papel) de (?:\w+ ?){1,10}",
        ]
        
        # Compile patterns for efficiency
        self.compiled_government_patterns = [re.compile(p) for p in self.government_patterns]
        self.compiled_position_patterns = [re.compile(p) for p in self.position_patterns]
        self.compiled_institution_patterns = [re.compile(p) for p in self.institution_patterns]
        self.compiled_role_patterns = [re.compile(p) for p in self.role_patterns]

    def attach_evidence_registry(self, evidence_registry) -> None:
        """Attach an evidence registry after initialization."""
        self.evidence_registry = evidence_registry

    def use_spacy_model(self, model_name: str) -> None:
        """Load a different spaCy model after instantiation."""
        self.model_name = model_name
        self.nlp = self._load_spacy_model(model_name)

    def _load_spacy_model(self, model_name: str):
        """Load a spaCy model, returning ``None`` on failure or absence."""
        if not SPACY_AVAILABLE:
            return None
        try:
            model = spacy.load(model_name)
            self.logger.info("Loaded spaCy model: %s", model_name)
            return model
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Error loading spaCy model %s: %s", model_name, exc)
            return None

    def _load_from_model_loader(self, model_loader):
        """Load the NLP pipeline using an injected loader for testing compatibility."""
        load_callable = getattr(model_loader, "load_model", None)
        if callable(load_callable):
            return load_callable(self.model_name)
        if callable(model_loader):  # pragma: no cover - graceful fallback
            return model_loader(self.model_name)
        raise TypeError("model_loader must be callable or expose a load_model method")

    __signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    
    def detect_entities(self, text: str, context_window: int = 100) -> List[ResponsibilityEntity]:
        """
        Detect responsibility entities in text.
        
        Args:
            text: Text to analyze
            context_window: Size of context window to include around entities
            
        Returns:
            List of detected ResponsibilityEntity objects
        """
        if not text:
            return []
        
        entities = []
        
        # First, try spaCy-based NER
        spacy_entities = []
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                spacy_entities = self._extract_spacy_entities(doc, context_window)
                entities.extend(spacy_entities)
            except Exception as e:
                logger.warning("Error in spaCy entity extraction: %s", e)
        
        # Then, use pattern matching for specific entity types
        pattern_entities = self._extract_pattern_entities(text, context_window)
        
        # Merge entities from all sources, resolving overlaps
        all_entities = self._merge_entities(entities, pattern_entities)
        
        # Detect entity roles
        all_entities = self._detect_entity_roles(all_entities, text)
        
        return all_entities
    
    @staticmethod
    def _extract_spacy_entities(doc: "Doc", context_window: int) -> List[ResponsibilityEntity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        for ent in doc.ents:
            # Focus on ORG and PERSON entities
            if ent.label_ in ["ORG", "PERSON"]:
                entity_type = EntityType.INSTITUTION if ent.label_ == "ORG" else EntityType.PERSON
                confidence = 0.6  # Base confidence for spaCy entities
                
                # Increase confidence for longer entities
                if len(ent.text) > 3:
                    confidence += 0.1
                if len(ent.text.split()) > 1:
                    confidence += 0.1
                
                # Extract context around entity
                context_start = max(0, ent.start_char - context_window)
                context_end = min(len(doc.text), ent.end_char + context_window)
                context_text = doc.text[context_start:context_end]
                
                # Create entity
                entity = ResponsibilityEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    confidence=min(confidence, 1.0),  # Cap at 1.0
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    context=context_text,
                )
                
                entities.append(entity)
        
        return entities
    
    def _extract_pattern_entities(self, text: str, context_window: int) -> List[ResponsibilityEntity]:
        """Extract entities using pattern matching."""
        entities = []
        
        # Find government institutions (highest priority)
        for pattern in self.compiled_government_patterns:
            for match in pattern.finditer(text):
                entity = self._create_entity_from_match(
                    match, text, context_window,
                    entity_type=EntityType.GOVERNMENT,
                    base_confidence=0.8,
                    boost_keywords=["ministerio", "secretaría", "dirección"]
                )
                entities.append(entity)
        
        # Find positions
        for pattern in self.compiled_position_patterns:
            for match in pattern.finditer(text):
                entity = self._create_entity_from_match(
                    match, text, context_window,
                    entity_type=EntityType.POSITION,
                    base_confidence=0.75,
                    boost_keywords=["alcalde", "gobernador", "director"]
                )
                entities.append(entity)
        
        # Find institutions (lower priority)
        for pattern in self.compiled_institution_patterns:
            for match in pattern.finditer(text):
                entity = self._create_entity_from_match(
                    match, text, context_window,
                    entity_type=EntityType.INSTITUTION,
                    base_confidence=0.7,
                    boost_keywords=None
                )
                entities.append(entity)
        
        return entities
    
    @staticmethod
    def _create_entity_from_match(match, text: str, context_window: int,
                                   entity_type: 'EntityType', base_confidence: float,
                                   boost_keywords: Optional[List[str]] = None) -> 'ResponsibilityEntity':
        """
        Create a ResponsibilityEntity from a regex match.
        
        Args:
            match: Regex match object
            text: Full text being processed
            context_window: Number of characters to include around match for context
            entity_type: Type of entity (GOVERNMENT, POSITION, or INSTITUTION)
            base_confidence: Base confidence score for this entity type
            boost_keywords: Keywords that increase confidence if found in entity text
            
        Returns:
            ResponsibilityEntity object
        """
        entity_text = match.group(0).strip()
        
        # Calculate confidence based on length and specificity
        confidence = base_confidence
        if len(entity_text.split()) > 2:
            confidence += 0.1
        if boost_keywords and any(keyword in entity_text.lower() for keyword in boost_keywords):
            confidence += 0.1
        
        # Extract context
        context_start = max(0, match.start() - context_window)
        context_end = min(len(text), match.end() + context_window)
        context_text = text[context_start:context_end]
        
        return ResponsibilityEntity(
            text=entity_text,
            entity_type=entity_type,
            confidence=min(confidence, 1.0),
            start_pos=match.start(),
            end_pos=match.end(),
            context=context_text,
        )
    
    @staticmethod
    def _merge_entities(entities1: List[ResponsibilityEntity], 
                        entities2: List[ResponsibilityEntity]) -> List[ResponsibilityEntity]:
        """
        Merge entities from different sources, resolving overlaps by keeping the higher confidence one.
        
        Args:
            entities1: First list of entities
            entities2: Second list of entities
            
        Returns:
            Merged list of entities with overlaps resolved
        """
        all_entities = entities1 + entities2
        
        # Sort by start position for overlap checking
        all_entities.sort(key=lambda e: e.start_pos)
        
        if not all_entities:
            return []
        
        # Resolve overlaps
        merged = [all_entities[0]]
        for current in all_entities[1:]:
            previous = merged[-1]
            
            # Check for overlap
            if current.start_pos < previous.end_pos:
                # Overlapping entities - keep the one with higher confidence
                if current.confidence > previous.confidence:
                    merged[-1] = current
                elif current.confidence == previous.confidence and current.entity_type == EntityType.GOVERNMENT:
                    # Prefer government entities when confidence is equal
                    merged[-1] = current
            else:
                # No overlap - add current entity
                merged.append(current)
        
        return merged
    
    def _detect_entity_roles(self, entities: List[ResponsibilityEntity], text: str) -> List[ResponsibilityEntity]:
        """
        Detect roles for each entity by analyzing surrounding context.
        
        Args:
            entities: List of detected entities
            text: Full text for context analysis
            
        Returns:
            Entities with role information added
        """
        for entity in entities:
            # Try to detect explicit role in context
            role_detected = False
            
            for pattern in self.compiled_role_patterns:
                # Look for role patterns in entity context
                matches = list(pattern.finditer(entity.context))
                
                for match in matches:
                    role_text = match.group(0)
                    # Verify this role description is associated with our entity
                    entity_pos_in_context = entity.start_pos - (max(0, entity.start_pos - 100))
                    match_pos = match.start()
                    
                    # Role should be close to entity (within 50 chars)
                    if abs(entity_pos_in_context - match_pos) < 50:
                        entity.has_explicit_role = True
                        entity.role_description = role_text
                        role_detected = True
                        break
                
                if role_detected:
                    break
            
            # Infer parent-child relationships between entities
            if entity.entity_type == EntityType.POSITION:
                # Try to find parent organization for position
                for other in entities:
                    if (other.entity_type in [EntityType.GOVERNMENT, EntityType.INSTITUTION] and 
                        entity.text.lower() in other.context.lower()):
                        entity.parent_entity = other.text
                        break
        
        return entities
    
    @staticmethod
    def evaluate_responsibility_clarity(entities: List[ResponsibilityEntity]) -> Dict[str, Any]:
        """
        Evaluate overall clarity of responsibility definition based on detected entities.
        
        Args:
            entities: List of detected responsibility entities
            
        Returns:
            Evaluation metrics including score, confidence, and categorization
        """
        if not entities:
            return {
                "clarity_score": 0.0,
                "confidence": 0.0,
                "has_government_entities": False,
                "has_positions": False,
                "has_explicit_roles": False,
                "entity_count": 0,
                "categorization": "undefined",
                "recommendation": "No se identificaron entidades responsables en el documento."
            }
        
        # Count entity types
        gov_count = sum(1 for e in entities if e.entity_type == EntityType.GOVERNMENT)
        position_count = sum(1 for e in entities if e.entity_type == EntityType.POSITION)
        explicit_roles = sum(1 for e in entities if e.has_explicit_role)
        
        # Calculate base score
        base_score = min(1.0, len(entities) / 5)  # Cap at 1.0 for having 5+ entities
        
        # Add bonus for variety of entity types
        type_bonus = 0.0
        if gov_count > 0:
            type_bonus += 0.3
        if position_count > 0:
            type_bonus += 0.2
        
        # Add bonus for explicit roles
        role_bonus = min(0.5, explicit_roles / len(entities) * 0.5)
        
        # Calculate overall score
        clarity_score = min(1.0, base_score + type_bonus + role_bonus)
        
        # Calculate confidence based on entity confidences
        avg_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # Categorize clarity
        categorization = "undefined"
        recommendation = ""
        
        if clarity_score >= 0.8:
            categorization = "well_defined"
            recommendation = "Las responsabilidades institucionales están claramente definidas."
        elif clarity_score >= 0.5:
            categorization = "partially_defined"
            recommendation = "Las responsabilidades institucionales están parcialmente definidas. Se recomienda especificar mejor los roles y responsabilidades."
        elif clarity_score >= 0.2:
            categorization = "poorly_defined"
            recommendation = "Las responsabilidades institucionales están pobremente definidas. Se recomienda incluir entidades específicas y sus roles."
        else:
            categorization = "undefined"
            recommendation = "Las responsabilidades institucionales no están definidas. Se recomienda incluir secciones específicas de responsabilidades."
        
        return {
            "clarity_score": clarity_score,
            "confidence": avg_confidence,
            "has_government_entities": gov_count > 0,
            "has_positions": position_count > 0,
            "has_explicit_roles": explicit_roles > 0,
            "entity_count": len(entities),
            "categorization": categorization,
            "recommendation": recommendation
        }

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Perform complete responsibility analysis on a document.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Complete analysis results including entities and evaluation
        """
        # Detect entities
        entities = self.detect_entities(text)
        
        # Evaluate responsibility clarity
        evaluation = self.evaluate_responsibility_clarity(entities)
        
        # Return comprehensive results
        return {
            "entities": [e.to_dict() for e in entities],
            "evaluation": evaluation,
            "answer_de1_q2": evaluation["categorization"] in ["well_defined", "partially_defined"],
            "confidence": evaluation["confidence"],
            "response_text": evaluation["recommendation"]
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect institutional responsibilities with NORMALIZED OUTPUT SCHEMA.

        Returns:
            List of dicts with strict schema:
            {
                "entity": str,
                "entity_type": str,  # EntityType.value
                "responsibility": str,
                "confidence": float,  # 0.0-1.0
                "context": str,
                "source_sentence": str,
                "applicable_questions": List[str]
            }
        """
        doc = self.nlp(text)
        responsibilities = []

        for ent in doc.ents:
            if ent.label_ in ["ORG", "MISC"]:
                # Buscar contexto de responsabilidad
                context = self._extract_responsibility_context(ent, doc)

                if context:
                    # Calcular confidence determinista
                    confidence = self._calculate_confidence(ent, context)

                    # Mapear a preguntas aplicables
                    applicable_qs = self._map_to_questions(ent, context)

                    responsibility = {
                        "entity": ent.text,
                        "entity_type": self._classify_entity_type(ent),
                        "responsibility": context["action"],
                        "confidence": confidence,
                        "context": context["full_context"],
                        "source_sentence": context["sentence"],
                        "applicable_questions": applicable_qs
                    }

                    responsibilities.append(responsibility)

                    # Registrar automáticamente si hay registry
                    if self.evidence_registry:
                        self.evidence_registry.register(
                            source_component="responsibility_detector",
                            evidence_type="institutional_responsibility",
                            content=responsibility,
                            confidence=confidence,
                            applicable_questions=applicable_qs
                        )

        return responsibilities

    @staticmethod
    def _calculate_confidence(ent, context: Dict) -> float:
        """Calcular confidence DETERMINISTA basado en señales"""
        confidence = 0.5

        # Boost por verbos de acción fuertes
        strong_verbs = ["lidera", "coordina", "ejecuta", "implementa", "gestiona"]
        if any(v in context["action"].lower() for v in strong_verbs):
            confidence += 0.2

        # Boost por contexto explícito
        if "responsable" in context["full_context"].lower():
            confidence += 0.15

        # Boost por tipo de entidad
        if context.get("entity_type") == "GOVERNMENT":
            confidence += 0.1

        return min(1.0, confidence)

    @staticmethod
    def _map_to_questions(ent, context: Dict) -> List[str]:
        """Mapear detección a preguntas específicas del cuestionario"""
        questions = []

        # Preguntas sobre responsabilidades institucionales (D4)
        questions.extend([f"D4-Q{i}" for i in [1, 5, 10, 15, 20]])

        # Si es presupuesto, añadir D3
        if "presupuesto" in context["full_context"].lower():
            questions.extend([f"D3-Q{i}" for i in [5, 10]])

        # Si es coordinación, añadir D1
        if "coordinación" in context["full_context"].lower():
            questions.extend([f"D1-Q{i}" for i in [3, 8]])

        return questions

    # ...existing code...
def create_responsibility_detector(model_name: str = "es_core_news_sm") -> ResponsibilityDetector:
    """
    Factory function to create a responsibility detector.
    
    Args:
        model_name: Name of spaCy model to use
        
    Returns:
        Initialized responsibility detector
    """
    return ResponsibilityDetector(spacy_model=model_name)


# Simple usage example
if __name__ == "__main__":
    detector = create_responsibility_detector()
    
    # Example text with responsibility entities
    sample_text = """
    La Secretaría de Educación Municipal será responsable de implementar el programa de mejoramiento educativo,
    en coordinación con el Instituto Colombiano de Bienestar Familiar. El alcalde supervisará el proceso,
    mientras que el Director de Planeación estará a cargo de la asignación de recursos.
    """
    
    # Detect entities
    detected_entities = detector.detect_entities(sample_text)
    
    # Print results
    print(f"Found {len(detected_entities)} responsibility entities:")
    for detected_entity in detected_entities:
        print(f"- {detected_entity.text} ({detected_entity.entity_type.value}, confidence: {detected_entity.confidence:.2f})")
        if detected_entity.has_explicit_role:
            print(f"  Role: {detected_entity.role_description}")
        if detected_entity.parent_entity:
            print(f"  Part of: {detected_entity.parent_entity}")
    
    # Complete document analysis
    analysis = detector.analyze_document(sample_text)
    print(f"\nResponsibility clarity score: {analysis['evaluation']['clarity_score']:.2f}")
    print(f"Categorization: {analysis['evaluation']['categorization']}")
    print(f"Recommendation: {analysis['evaluation']['recommendation']}")

    # Normalized detection example
    normalized_results = detector.detect(sample_text)
    print("\nNormalized detection results:")
    for result in normalized_results:
        print(result)
