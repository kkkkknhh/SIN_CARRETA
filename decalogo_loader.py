"""
DECALOGO_INDUSTRIAL Template Loading Module

Provides atomic file operations with fallback template loading for DECALOGO components.
Ensures reliable access to decalogo templates even in restricted environments.

Features:
- Atomic file operations for deployment safety
- Fallback to in-memory template when file operations fail
- Thread-safe caching for efficient access
- Comprehensive error handling for deployment scenarios
"""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton pattern
_DECALOGO_CACHE = {}
_CACHE_LOCK = threading.RLock()

# Default paths
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "decalogo-industrial.latest.clean.json"

# Fallback template in case file access fails
DECALOGO_INDUSTRIAL_TEMPLATE = {
    "version": "1.0.0",
    "metadata": {
        "name": "DECALOGO_INDUSTRIAL",
        "description": "Template for industrial-grade decalogo evaluation",
        "created": "2024-01-01",
        "updated": "2024-01-01"
    },
    "dimensions": [
        {
            "id": "DE-1",
            "name": "Coherencia Estratégica",
            "description": "Evalúa la articulación lógica entre diagnóstico, estrategias y resultados",
            "weight": 0.30,
            "questions": [
                {
                    "id": "1",
                    "text": "¿El diagnóstico identifica brechas y retos?",
                    "weight": 0.15
                },
                {
                    "id": "2",
                    "text": "¿Las estrategias responden a los retos identificados?",
                    "weight": 0.20
                },
                {
                    "id": "3",
                    "text": "¿Los resultados tienen líneas base y metas?",
                    "weight": 0.25
                },
                {
                    "id": "4",
                    "text": "¿Existe un encadenamiento lógico en la cadena de valor?",
                    "weight": 0.20
                },
                {
                    "id": "5",
                    "text": "¿Los indicadores son relevantes y específicos?",
                    "weight": 0.10
                },
                {
                    "id": "6",
                    "text": "¿Existe un marco lógico completo?",
                    "weight": 0.10
                }
            ]
        },
        {
            "id": "DE-2",
            "name": "Inclusión Temática",
            "description": "Evalúa la incorporación de temas clave en el plan de desarrollo",
            "weight": 0.25,
            "questions": [
                {
                    "id": "1",
                    "text": "¿Se articulan con Plan Nacional de Desarrollo?",
                    "weight": 0.30
                },
                {
                    "id": "2",
                    "text": "¿Se incorpora presupuesto para cada componente?",
                    "weight": 0.40
                },
                {
                    "id": "3",
                    "text": "¿Se incorporan los ODS en objetivos/indicadores?",
                    "weight": 0.30
                }
            ]
        },
        {
            "id": "DE-3",
            "name": "Proceso Participativo",
            "description": "Evalúa la inclusión de la ciudadanía en la formulación del plan",
            "weight": 0.20,
            "questions": [
                {
                    "id": "1",
                    "text": "¿Hay evidencia de participación ciudadana?",
                    "weight": 0.60
                },
                {
                    "id": "2",
                    "text": "¿Se identifican grupos sociales específicos?",
                    "weight": 0.40
                }
            ]
        },
        {
            "id": "DE-4",
            "name": "Orientación a Resultados",
            "description": "Evalúa la factibilidad y medición de resultados del plan",
            "weight": 0.25,
            "questions": [
                {
                    "id": "1",
                    "text": "¿Los productos tienen KPI medibles?",
                    "weight": 0.20
                },
                {
                    "id": "2",
                    "text": "¿Los resultados tienen líneas base?",
                    "weight": 0.15
                },
                {
                    "id": "3",
                    "text": "¿Existen entidades responsables por resultado?",
                    "weight": 0.15
                },
                {
                    "id": "4",
                    "text": "¿Los recursos son suficientes para resultados?",
                    "weight": 0.30
                },
                {
                    "id": "5",
                    "text": "¿Se articula con planes de largo plazo?",
                    "weight": 0.20
                }
            ]
        }
    ]
}


def _atomic_write_json(target_path: Path, data: Any) -> bool:
    """
    Write JSON data to file using atomic operations for safety.
    
    Args:
        target_path: Path where the JSON file should be written
        data: Data to serialize as JSON
        
    Returns:
        Success status (True if successful, False otherwise)
    """
    # Create parent directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use a temporary file for atomic write
        with tempfile.NamedTemporaryFile(
            mode='w', 
            encoding='utf-8',
            dir=target_path.parent,
            delete=False,
            suffix='.tmp'
        ) as temp_file:
            # Write data to temporary file
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Ensure data is written to disk
            temp_path = Path(temp_file.name)
        
        # Perform atomic rename
        os.replace(temp_path, target_path)
        logger.debug(f"Successfully wrote template to {target_path}")
        return True
    
    except (PermissionError, OSError) as e:
        logger.warning(f"Failed to write template to {target_path}: {e}")
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def load_decalogo_industrial(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the DECALOGO_INDUSTRIAL template with fallback mechanism.
    
    Args:
        path: Optional custom path to template file
        
    Returns:
        DECALOGO_INDUSTRIAL template data
    """
    template_path = Path(path) if path else DEFAULT_TEMPLATE_PATH
    
    try:
        # Try to load from file
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
                logger.debug(f"Loaded template from {template_path}")
                return template_data
        
        # If file doesn't exist, try to create it with default template
        logger.info(f"Template not found at {template_path}, creating with default")
        if _atomic_write_json(template_path, DECALOGO_INDUSTRIAL_TEMPLATE):
            return DECALOGO_INDUSTRIAL_TEMPLATE
    
    except (PermissionError, OSError, IOError, json.JSONDecodeError) as e:
        # Log detailed error but continue with fallback
        logger.warning(f"Error accessing template at {template_path}: {e}")
    
    # Return in-memory fallback template
    logger.info("Using in-memory fallback template")
    return DECALOGO_INDUSTRIAL_TEMPLATE


def get_decalogo_industrial(path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Get DECALOGO_INDUSTRIAL template with caching for efficiency.
    
    Args:
        path: Optional custom path to template file
        force_reload: Force reload from disk, bypassing cache
        
    Returns:
        DECALOGO_INDUSTRIAL template data
    """
    cache_key = path or str(DEFAULT_TEMPLATE_PATH)

    with _CACHE_LOCK:
        if force_reload or cache_key not in _DECALOGO_CACHE:
            _DECALOGO_CACHE[cache_key] = load_decalogo_industrial(path)
        return _DECALOGO_CACHE[cache_key]


if __name__ == "__main__":
    # Example usage
    template = get_decalogo_industrial()
    print(f"Loaded DECALOGO_INDUSTRIAL template version: {template.get('version', 'unknown')}")
    print(f"Dimensions: {len(template.get('dimensions', []))}")
    
    # Display dimension information
    for dim in template.get('dimensions', []):
        print(f"- {dim.get('id')}: {dim.get('name')} ({len(dim.get('questions', []))} questions)")


def load_dnp_standards(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DNP_STANDARDS.json configuration.
    
    Args:
        path: Optional custom path to DNP standards file
        
    Returns:
        DNP standards data
    """
    standards_path = Path(path) if path else (Path(__file__).parent / "dnp-standards.latest.clean.json")

    try:
        if standards_path.exists():
            with open(standards_path, 'r', encoding='utf-8') as f:
                standards_data = json.load(f)
                logger.debug(f"Loaded DNP standards from {standards_path}")
                return standards_data
    except (PermissionError, OSError, IOError, json.JSONDecodeError) as e:
        logger.warning(f"Error accessing DNP standards at {standards_path}: {e}")
    logger.warning("DNP standards file not found, returning empty structure")
    return {
        "metadata": {"version": "2.0.0"},
        "decalogo_dimension_mapping": {}
    }


def ensure_aligned_templates() -> Dict[str, Any]:
    """
    Ensure all critical templates are loaded.
    
    NOTE: This system uses THREE canonical JSON files:
    - decalogo_industrial.json: 300 questions for evaluation
    - DNP_STANDARDS.json: dimension mapping and weights
    - RUBRIC_SCORING.json: scoring modalities and weights
    
    Returns:
        Dictionary with all loaded templates
    """
    decalogo_industrial = get_decalogo_industrial()
    dnp_standards = load_dnp_standards()
    
    # Verify basic structure
    decalogo_questions = len(decalogo_industrial.get("questions", []))
    dnp_dimensions = len(dnp_standards.get("decalogo_dimension_mapping", {}))
    
    logger.info(f"Loaded templates: {decalogo_questions} questions, {dnp_dimensions} dimension mappings")
    
    alignment_status = "verified" if decalogo_questions == 300 else "incomplete"
    
    return {
        "decalogo_industrial": decalogo_industrial,
        "dnp_standards": dnp_standards,
        "alignment": {
            "status": alignment_status,
            "questions_found": decalogo_questions,
            "dimensions_mapped": dnp_dimensions
        }
    }


def get_question_by_id(
    question_id: str,
    decalogo_industrial: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get specific question details by question ID.
    
    Args:
        question_id: Question ID (e.g., 'D1-Q1')
        decalogo_industrial: Optional pre-loaded decalogo_industrial
        
    Returns:
        Dictionary with question details or empty dict if not found
    """
    if decalogo_industrial is None:
        decalogo_industrial = get_decalogo_industrial()
    
    questions = decalogo_industrial.get("questions", [])
    
    for question in questions:
        if question.get("id") == question_id:
            return question
    
    return {}


def get_dimension_weight(
    point_code: str,
    dimension: str,
    dnp_standards: Optional[Dict[str, Any]] = None
) -> float:
    """
    Get weight for a specific dimension within a point.
    
    Args:
        point_code: Point code (e.g., 'P1')
        dimension: Dimension ID (e.g., 'D1')
        dnp_standards: Optional pre-loaded DNP standards
        
    Returns:
        Weight for the dimension (0.0 if not found)
    """
    if dnp_standards is None:
        dnp_standards = load_dnp_standards()
    
    mapping = dnp_standards.get("decalogo_dimension_mapping", {})
    point_config = mapping.get(point_code, {})
    
    weight_key = f"{dimension}_weight"
    return point_config.get(weight_key, 0.0)


if __name__ == "__main__":
    # Example usage
    print("DECALOGO Loader - Diagnostic tool")
    print("=" * 50)
    
    # Load templates
    templates = ensure_aligned_templates()
    
    # Show alignment status
    alignment = templates["alignment"]
    print(f"\nAlignment status: {alignment['status']}")
    print(f"Questions found: {alignment['questions_found']}")
    print(f"Dimensions mapped: {alignment['dimensions_mapped']}")
    
    # Show sample questions
    industrial = templates["decalogo_industrial"]
    questions = industrial.get("questions", [])
    if questions:
        print(f"\nTotal questions: {len(questions)}")
        print("\nSample questions:")
        for q in questions[:3]:
            print(f"- {q.get('id')}: {q.get('point_title')} (Dimension {q.get('dimension')})")
    
    # Show DNP standards sample
    dnp = templates["dnp_standards"]
    mapping = dnp.get("decalogo_dimension_mapping", {})
    if mapping:
        print(f"\nDNP Standards - Points mapped: {len(mapping)}")
        print("\nSample dimension weights (P1):")
        p1_config = mapping.get("P1", {})
        for key, value in list(p1_config.items())[:6]:
            if key.endswith("_weight"):
                print(f"- {key}: {value}")
