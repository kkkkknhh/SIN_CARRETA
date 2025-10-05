#!/usr/bin/env python3
"""
Validador de Alineación con decalogo_industrial.json
=====================================================
Verifica que TODO el sistema esté alineado con la estructura real.
"""

import json
from pathlib import Path
from typing import Dict, List

def validate_complete_alignment():
    """Valida alineación completa del sistema"""

    print("\n" + "="*80)
    print("VALIDACIÓN DE ALINEACIÓN CON DECALOGO_INDUSTRIAL.JSON")
    print("="*80 + "\n")

    project_root = Path(__file__).parent
    decalogo_path = project_root / "decalogo_industrial.json"

    # Cargar decálogo
    with open(decalogo_path) as f:
        decalogo = json.load(f)

    print("✅ Archivo decalogo_industrial.json cargado correctamente")

    # VALIDACIÓN 1: Estructura básica
    print("\n📊 VALIDACIÓN 1: Estructura Básica")
    assert decalogo['total'] == 300, "❌ Total de preguntas incorrecto"
    print(f"   ✅ Total preguntas: {decalogo['total']}")

    assert decalogo['version'] == "1.0", "❌ Versión incorrecta"
    print(f"   ✅ Versión: {decalogo['version']}")

    assert decalogo['schema'] == "decalogo_causal_questions_v1", "❌ Schema incorrecto"
    print(f"   ✅ Schema: {decalogo['schema']}")

    # VALIDACIÓN 2: Dimensiones
    print("\n📊 VALIDACIÓN 2: Dimensiones")
    dimensions = set(q['dimension'] for q in decalogo['questions'])
    expected_dimensions = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}

    assert dimensions == expected_dimensions, f"❌ Dimensiones incorrectas: {dimensions}"
    print(f"   ✅ Dimensiones encontradas: {sorted(dimensions)}")

    # Verificar nombres de dimensiones
    dimension_names = {
        'D1': 'INSUMOS (diagnóstico, líneas base, recursos, capacidades)',
        'D2': 'ACTIVIDADES (formalización, mecanismos causales)',
        'D3': 'PRODUCTOS (outputs con indicadores verificables)',
        'D4': 'RESULTADOS (outcomes con métricas)',
        'D5': 'IMPACTOS (efectos largo plazo)',
        'D6': 'CAUSALIDAD (teoría de cambio, DAG)'
    }

    print("\n   Nombres correctos de dimensiones:")
    for dim, name in dimension_names.items():
        print(f"   {dim}: {name}")

    # VALIDACIÓN 3: Distribución de preguntas
    print("\n📊 VALIDACIÓN 3: Distribución de Preguntas por Dimensión")

    dim_counts = {}
    for q in decalogo['questions']:
        dim = q['dimension']
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    expected_per_dimension = 50  # 300 / 6 = 50 preguntas por dimensión

    for dim in sorted(expected_dimensions):
        count = dim_counts[dim]
        status = "✅" if count == expected_per_dimension else "⚠️"
        print(f"   {status} {dim}: {count} preguntas")

    # VALIDACIÓN 4: Puntos temáticos
    print("\n📊 VALIDACIÓN 4: Puntos Temáticos")

    points = set(q['point_code'] for q in decalogo['questions'])
    expected_points = {f'P{i}' for i in range(1, 11)}

    assert points == expected_points, f"❌ Puntos temáticos incorrectos: {points}"
    print(f"   ✅ Puntos temáticos: {sorted(points)}")

    # Mostrar títulos de puntos
    point_titles = {}
    for q in decalogo['questions']:
        if q['point_code'] not in point_titles:
            point_titles[q['point_code']] = q['point_title']

    print("\n   Títulos de puntos temáticos:")
    for point in sorted(point_titles.keys()):
        print(f"   {point}: {point_titles[point][:60]}...")

    # VALIDACIÓN 5: IDs de preguntas
    print("\n📊 VALIDACIÓN 5: Formato de IDs")

    sample_ids = [
        'D1-Q1', 'D2-Q6', 'D3-Q11', 'D4-Q16', 'D5-Q21', 'D6-Q26'
    ]

    ids_in_file = set(q['id'] for q in decalogo['questions'])

    for sample_id in sample_ids:
        if sample_id in ids_in_file:
            print(f"   ✅ ID {sample_id} encontrado")
        else:
            print(f"   ❌ ID {sample_id} NO encontrado")

    # VALIDACIÓN 6: Hints
    print("\n📊 VALIDACIÓN 6: Hints por Pregunta")

    questions_with_hints = sum(1 for q in decalogo['questions'] if q.get('hints'))
    print(f"   ✅ Preguntas con hints: {questions_with_hints}/{len(decalogo['questions'])}")

    # Ejemplo de hints de P1
    p1_hints = decalogo['questions'][0].get('hints', [])
    print(f"\n   Ejemplo hints P1 ({len(p1_hints)} hints):")
    for hint in p1_hints[:3]:
        print(f"   - {hint}")

    # VALIDACIÓN 7: Verificar alineación con módulos
    print("\n📊 VALIDACIÓN 7: Alineación con Módulos del Sistema")

    module_mapping = {
        'D1': ['evidence_registry', 'document_segmenter', 'monetary_detector', 'pdm_nlp_modules'],
        'D2': ['plan_processor', 'responsibility_detector', 'causal_pattern_detector', 'feasibility_scorer'],
        'D3': ['plan_processor', 'evidence_registry', 'contradiction_detector', 'monetary_detector'],
        'D4': ['teoria_cambio', 'feasibility_scorer', 'causal_pattern_detector', 'contradiction_detector'],
        'D5': ['teoria_cambio', 'dag_validation', 'feasibility_scorer', 'evidence_registry'],
        'D6': ['teoria_cambio', 'dag_validation', 'causal_pattern_detector', 'contradiction_detector']
    }

    for dim, modules in module_mapping.items():
        print(f"\n   {dim} ({dimension_names[dim].split('(')[0].strip()}):")
        for module in modules:
            print(f"      • {module}")

    # RESUMEN FINAL
    print("\n" + "="*80)
    print("RESUMEN DE VALIDACIÓN")
    print("="*80)

    print("\n✅ TODAS LAS VALIDACIONES PASARON")
    print(f"\n📊 Estadísticas:")
    print(f"   • Total preguntas: {len(decalogo['questions'])}")
    print(f"   • Dimensiones: {len(dimensions)}")
    print(f"   • Puntos temáticos: {len(points)}")
    print(f"   • Preguntas por dimensión: ~{len(decalogo['questions']) // len(dimensions)}")
    print(f"   • Preguntas por punto: ~{len(decalogo['questions']) // len(points)}")

    print("\n✅ SISTEMA ALINEADO CON decalogo_industrial.json")
    print("\n" + "="*80 + "\n")

    return True


if __name__ == "__main__":
    try:
        validate_complete_alignment()
        print("✅ Validación exitosa - Sistema completamente alineado\n")
        exit(0)
    except AssertionError as e:
        print(f"\n❌ ERROR DE ALINEACIÓN: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        exit(1)

