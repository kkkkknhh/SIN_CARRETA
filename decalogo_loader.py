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
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton pattern
_DECALOGO_CACHE = {}
_CACHE_LOCK = threading.RLock()

# Default paths
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "decalogo_industrial.json"

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
    cache_key = str(path) if path else "default"
    
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


def ensure_aligned_templates() -> Dict[str, Any]:
    """
    Ensure all templates are loaded and aligned.
    
    Returns:
        Dictionary with all loaded templates and alignment status
    """
    # Load all templates
    decalogo_industrial = get_decalogo_industrial()
    decalogo_full = load_decalogo_full()
    dnp_standards = load_dnp_standards()
    
    # Verify alignment
    alignment = verify_alignment(decalogo_industrial, decalogo_full, dnp_standards)
    
    # Log alignment status
    if alignment["status"] == "verified":
        logger.info("All templates are properly aligned")
    else:
        logger.warning(f"Template alignment issues detected: {alignment['issues']}")
    
    return {
        "decalogo_industrial": decalogo_industrial,
        "decalogo_full": decalogo_full,
        "dnp_standards": dnp_standards,
        "alignment": alignment
    }


def get_cluster_metadata(
    cluster_id: str, 
    decalogo_full: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get metadata for a specific cluster.
    
    Args:
        cluster_id: ID of the cluster to retrieve
        decalogo_full: Optional pre-loaded DECALOGO_FULL
        
    Returns:
        Dictionary with cluster metadata or empty dict if not found
    """
    if decalogo_full is None:
        decalogo_full = load_decalogo_full()
    
    clusters = decalogo_full.get("clusters_politica", [])
    
    for cluster in clusters:
        if cluster.get("id") == cluster_id:
            return cluster
    
    return {}


def map_punto_to_cluster(
    punto_id: int, 
    decalogo_full: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Map a decalogo point to its associated clusters.
    
    Args:
        punto_id: Decalogo point ID to map
        decalogo_full: Optional pre-loaded DECALOGO_FULL
        
    Returns:
        List of cluster IDs that include this point
    """
    if decalogo_full is None:
        decalogo_full = load_decalogo_full()
    
    clusters = decalogo_full.get("clusters_politica", [])
    associated_clusters = []
    
    for cluster in clusters:
        if punto_id in cluster.get("puntos_decalogo", []):
            associated_clusters.append(cluster.get("id"))
    
    return associated_clusters


def get_question_by_id(
    dimension_id: str,
    question_id: str,
    decalogo_full: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get specific question details by dimension and question ID.
    
    Args:
        dimension_id: Dimension ID (e.g., 'DE-1')
        question_id: Question ID (e.g., 'Q1')
        decalogo_full: Optional pre-loaded DECALOGO_FULL
        
    Returns:
        Dictionary with question details or empty dict if not found
    """
    if decalogo_full is None:
        decalogo_full = load_decalogo_full()
    
    dimensions = decalogo_full.get("dimensiones", [])
    
    for dimension in dimensions:
        if dimension.get("id") == dimension_id:
            for question in dimension.get("preguntas", []):
                if question.get("id") == question_id:
                    return question
    
    return {}


if __name__ == "__main__":
    # Example usage
    print("DECALOGO Loader - Diagnostic tool")
    print("================================")
    
    # Load templates
    templates = ensure_aligned_templates()
    
    # Show alignment status
    alignment = templates["alignment"]
    print(f"\nAlignment status: {alignment['status']}")
    
    if alignment["issues"]:
        print("\nDetected issues:")
        for issue in alignment["issues"]:
            print(f"- {issue}")
    
    # Show dimensions
    industrial = templates["decalogo_industrial"]
    print("\nDECALOGO_INDUSTRIAL dimensions:")
    for dim in industrial.get("dimensions", []):
        print(f"- {dim['id']}: {dim['name']} (weight: {dim['weight']})")
    
    # Show policy clusters if available
    full = templates["decalogo_full"]
    clusters = full.get("clusters_politica", [])
    if clusters:
        print("\nPolicy clusters:")
        for cluster in clusters:
            points = cluster.get("puntos_decalogo", [])
            print(f"- {cluster.get('id')}: {cluster.get('nombre')} ({len(points)} points)")
