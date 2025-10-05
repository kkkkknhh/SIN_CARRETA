"""
Demonstration of plan name sanitization and JSON key standardization functionality.
"""

import os
import tempfile
import json
from plan_sanitizer import PlanSanitizer, sanitize_plan_name, standardize_json_keys


def demo_plan_name_sanitization():
    """Demonstrate plan name sanitization with various problematic characters."""
    print("=== PLAN NAME SANITIZATION DEMO ===\n")
    
    test_cases = [
        "Plan: Meta/Objetivo 2024*",
        "Plan <urgente> | Desarrollo Social", 
        'Plan "Nacional" ¿Educación?',
        "Plan\\Desarrollo\\2024\\Meta",
        "CON",  # Windows reserved name
        "Plan: Línea/Base*Evaluación?",
        "Plan\t\ncon\x00caracteres\x01de\x02control",
        "Plan 🚀 Meta ⭐ 2024 💫",
        "/<>:\"|*?\\",  # All forbidden characters
        "Plan de Desarrollo Nacional Integral Sostenible " * 5,  # Very long name
    ]
    
    for i, original_name in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Original: {repr(original_name)}")
        
        sanitized = sanitize_plan_name(original_name)
        print(f"  Sanitized: {repr(sanitized)}")
        print(f"  Length: {len(sanitized)} chars")
        
        # Test directory creation
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                safe_dir = os.path.join(temp_dir, sanitized)
                os.makedirs(safe_dir, exist_ok=True)
                print("  ✓ Directory creation successful")
        except Exception as e:
            print(f"  ✗ Directory creation failed: {e}")
        
        print()


def demo_json_key_standardization():
    """Demonstrate JSON key standardization."""
    print("=== JSON KEY STANDARDIZATION DEMO ===\n")
    
    # Complex nested JSON with Spanish tildes and mixed formats
    test_data = {
        "información_general": {
            "línea_base": "Situación inicial 2023",
            "número_página": 42,
            "fecha_creación": "2024-01-15T10:30:00Z",
            "tipoDocumento": "plan_nacional",
            "evaluación-técnica": "Completa"
        },
        "metas_específicas": [
            {
                "descripción": "Reducir pobreza extrema",
                "línea_temporal": "2024-2026", 
                "población_objetivo": "Rural",
                "implementación_esperada": {
                    "número_beneficiarios": 50000,
                    "duración_meses": 24,
                    "región_geográfica": "Norte"
                }
            },
            {
                "descripción": "Mejorar educación básica",
                "evaluación_inicial": "Deficiente",
                "situación_actual": "En progreso"
            }
        ],
        "indicadores_clave": {
            "línea_base_pobreza": 25.5,
            "meta_reducción": 15.0,
            "población_total": 1200000
        }
    }
    
    print("Original JSON structure:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")
    
    # Standardize with display keys preserved
    standardized_with_display = standardize_json_keys(test_data, preserve_display=True)
    print("Standardized JSON (with display keys):")
    print(json.dumps(standardized_with_display, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")
    
    # Standardize without display keys
    standardized_clean = standardize_json_keys(test_data, preserve_display=False)
    print("Standardized JSON (clean):")
    print(json.dumps(standardized_clean, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")
    
    # Show key transformations
    print("Key transformations:")
    original_keys = extract_all_keys(test_data)
    standardized_keys = extract_all_keys(standardized_clean)
    
    for orig, std in zip(sorted(original_keys), sorted(standardized_keys)):
        if orig != std:
            print(f"  '{orig}' → '{std}'")


def demo_markdown_display_integration():
    """Demonstrate how to use display keys for Markdown output."""
    print("=== MARKDOWN DISPLAY INTEGRATION DEMO ===\n")
    
    # Sample data with standardized keys and display versions
    plan_data = {
        "informacion_general": {
            "linea_base": "2023 baseline data",
            "linea_base_display": "línea_base",
            "numero_pagina": 15,
            "numero_pagina_display": "número_página",
            "evaluacion": "Complete assessment", 
            "evaluacion_display": "evaluación"
        }
    }
    
    # Generate Markdown using display keys for human readability
    print("Generated Markdown using display keys:")
    print("```markdown")
    print("# Plan Information")
    print()
    
    info_general = plan_data["informacion_general"]
    for key, value in info_general.items():
        if not key.endswith("_display"):
            # Get display version if available
            display_key = PlanSanitizer.get_markdown_display_key(key, info_general)
            print(f"- **{display_key}**: {value}")
    
    print("```")
    print()
    
    # Show the difference
    print("Without display keys (programmatic keys only):")
    print("```markdown")
    print("# Plan Information")
    print()
    
    for key, value in info_general.items():
        if not key.endswith("_display"):
            print(f"- **{key.replace('_', ' ')}**: {value}")
    
    print("```")


def demo_complete_workflow():
    """Demonstrate complete workflow from raw plan to processed output."""
    print("=== COMPLETE WORKFLOW DEMO ===\n")
    
    # Raw plan with problematic name and data
    raw_plan_name = "Plan: Desarrollo/Social*2024 <Urgente>"
    raw_plan_data = {
        "información_básica": {
            "línea_base": "Pobreza: 25%, Educación: 60% cobertura",
            "número_elementos": 147,
            "fecha_creación": "2024-01-15",
            "situación_actual": "En desarrollo"
        },
        "metas_principales": [
            {
                "descripción": "Reducir pobreza extrema al 15%",
                "línea_temporal": "2024-2026",
                "población_objetivo": 500000,
                "implementación": "Gradual por regiones"
            }
        ]
    }
    
    print("1. Raw plan name:", repr(raw_plan_name))
    print("2. Raw JSON keys:", list(raw_plan_data.keys()))
    print()
    
    # Process plan name  
    sanitized_name = sanitize_plan_name(raw_plan_name)
    print("3. Sanitized name:", repr(sanitized_name))
    
    # Process JSON data
    processed_data = standardize_json_keys(raw_plan_data, preserve_display=True)
    print("4. Processed JSON keys:", list(processed_data.keys()))
    print()
    
    # Create safe directory
    with tempfile.TemporaryDirectory() as temp_dir:
        plan_dir = PlanSanitizer.create_safe_directory(raw_plan_name, temp_dir)
        print("5. Created directory:", plan_dir)
        
        # Save processed data
        data_file = os.path.join(plan_dir, "plan_data.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print("6. Saved processed data to:", data_file)
        
        # Verify directory and file creation
        print("7. Directory contents:", os.listdir(plan_dir))
        print("8. File size:", os.path.getsize(data_file), "bytes")


def extract_all_keys(obj, keys=None):
    """Helper function to extract all keys from nested JSON object."""
    if keys is None:
        keys = set()
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            extract_all_keys(value, keys)
    elif isinstance(obj, list):
        for item in obj:
            extract_all_keys(item, keys)
    
    return keys


if __name__ == "__main__":
    print("PLAN SANITIZER DEMO")
    print("=" * 80)
    print()
    
    demo_plan_name_sanitization()
    print("\n" + "=" * 80 + "\n")
    
    demo_json_key_standardization()
    print("\n" + "=" * 80 + "\n")
    
    demo_markdown_display_integration()  
    print("\n" + "=" * 80 + "\n")
    
    demo_complete_workflow()
    print("\n" + "=" * 80)
    print("\nDemo completed successfully! ✓")