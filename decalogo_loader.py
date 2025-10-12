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

# Import canonical path resolver
from repo_paths import get_decalogo_path, get_dnp_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton pattern
_DECALOGO_CACHE = {}
_CACHE_LOCK = threading.RLock()

# Default paths - now using central resolver
DEFAULT_TEMPLATE_PATH = get_decalogo_path(os.getenv("DECALOGO_PATH_OVERRIDE"))

# Fallback template in case file access fails
# Minimal template matching the actual file structure
DECALOGO_INDUSTRIAL_TEMPLATE = {
    "version": "1.0",
    "schema": "decalogo_causal_questions_v1",
    "total": 6,
    "questions": [
        {
            "id": "D1-Q1",
            "dimension": "D1",
            "question_no": 1,
            "point_code": "P1",
            "point_title": "Sample Policy Point 1",
            "prompt": "Sample diagnostic question with baseline data requirements",
            "hints": ["sample hint 1", "sample hint 2"]
        },
        {
            "id": "D1-Q2",
            "dimension": "D1",
            "question_no": 2,
            "point_code": "P1",
            "point_title": "Sample Policy Point 1",
            "prompt": "Sample question about problem magnitude and data quality",
            "hints": ["sample hint 1", "sample hint 2"]
        },
        {
            "id": "D2-Q6",
            "dimension": "D2",
            "question_no": 6,
            "point_code": "P1",
            "point_title": "Sample Policy Point 1",
            "prompt": "Sample question about activities and instruments",
            "hints": ["sample hint 1", "sample hint 2"]
        },
        {
            "id": "D1-Q1",
            "dimension": "D1",
            "question_no": 1,
            "point_code": "P2",
            "point_title": "Sample Policy Point 2",
            "prompt": "Sample diagnostic question with baseline data requirements",
            "hints": ["sample hint 1", "sample hint 2"]
        },
        {
            "id": "D1-Q2",
            "dimension": "D1",
            "question_no": 2,
            "point_code": "P2",
            "point_title": "Sample Policy Point 2",
            "prompt": "Sample question about problem magnitude and data quality",
            "hints": ["sample hint 1", "sample hint 2"]
        },
        {
            "id": "D2-Q6",
            "dimension": "D2",
            "question_no": 6,
            "point_code": "P2",
            "point_title": "Sample Policy Point 2",
            "prompt": "Sample question about activities and instruments",
            "hints": ["sample hint 1", "sample hint 2"]
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
        logger.debug("Successfully wrote template to %s", target_path)
        return True
    
    except (PermissionError, OSError) as e:
        logger.warning("Failed to write template to %s: %s", target_path, e)
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
                logger.debug("Loaded template from %s", template_path)
                return template_data
        
        # If file doesn't exist, try to create it with default template
        logger.info("Template not found at %s, creating with default", template_path)
        if _atomic_write_json(template_path, DECALOGO_INDUSTRIAL_TEMPLATE):
            return DECALOGO_INDUSTRIAL_TEMPLATE
    
    except (PermissionError, OSError, IOError, json.JSONDecodeError) as e:
        # Log detailed error but continue with fallback
        logger.warning("Error accessing template at %s: %s", template_path, e)
    
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
    print(f"Schema: {template.get('schema', 'unknown')}")
    print(f"Total questions: {template.get('total', 0)}")
    print(f"Actual questions: {len(template.get('questions', []))}")
    
    # Display sample question information
    if template.get('questions'):
        sample_q = template['questions'][0]
        print(f"\nSample question structure:")
        print(f"- ID: {sample_q.get('id')}")
        print(f"- Dimension: {sample_q.get('dimension')}")
        print(f"- Point: {sample_q.get('point_code')} - {sample_q.get('point_title')}")
        print(f"- Prompt: {sample_q.get('prompt', '')[:60]}...")


def load_dnp_standards(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DNP_STANDARDS.json configuration.
    
    Args:
        path: Optional custom path to DNP standards file
        
    Returns:
        DNP standards data
    """
    # Use central path resolver with optional override
    standards_path = Path(path) if path else get_dnp_path(os.getenv("DNP_PATH_OVERRIDE"))

    try:
        if standards_path.exists():
            with open(standards_path, 'r', encoding='utf-8') as f:
                standards_data = json.load(f)
                logger.debug("Loaded DNP standards from %s", standards_path)
                return standards_data
    except (PermissionError, OSError, IOError, json.JSONDecodeError) as e:
        logger.warning("Error accessing DNP standards at %s: %s", standards_path, e)
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
    
    logger.info("Loaded templates: %s questions, %s dimension mappings", decalogo_questions, dnp_dimensions)
    
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
    
    all_questions = decalogo_industrial.get("questions", [])
    
    for question in all_questions:
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
    dimension_mapping = dnp.get("decalogo_dimension_mapping", {})
    if dimension_mapping:
        print(f"\nDNP Standards - Points mapped: {len(dimension_mapping)}")
        print("\nSample dimension weights (P1):")
        p1_config = dimension_mapping.get("P1", {})
        for key, value in list(p1_config.items())[:6]:
            if key.endswith("_weight"):
                print(f"- {key}: {value}")
