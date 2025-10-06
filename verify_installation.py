#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Verificación Post-Instalación
Verifica que todos los módulos críticos estén correctamente instalados
"""

import sys

print("=" * 70)
print("VERIFICACIÓN DE INSTALACIÓN - MINIMINIMOON v2.0")
print("=" * 70)
print()

modules_ok = []
modules_fail = []

# Test 1: miniminimoon_orchestrator
print("Testing miniminimoon_orchestrator...", end=" ")
try:
    from miniminimoon_orchestrator import (
        CanonicalDeterministicOrchestrator,
        PipelineStage,
        EvidenceRegistry,
        UnifiedEvaluationPipeline
    )
    stages = len(list(PipelineStage))
    print(f"✓ OK ({stages} stages)")
    modules_ok.append("miniminimoon_orchestrator")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("miniminimoon_orchestrator", str(e)))

# Test 2: plan_processor
print("Testing plan_processor...", end=" ")
try:
    from plan_processor import PlanProcessor
    processor = PlanProcessor()
    print("✓ OK")
    modules_ok.append("plan_processor")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("plan_processor", str(e)))

# Test 3: document_segmenter
print("Testing document_segmenter...", end=" ")
try:
    from document_segmenter import DocumentSegmenter
    segmenter = DocumentSegmenter()
    print("✓ OK")
    modules_ok.append("document_segmenter")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("document_segmenter", str(e)))

# Test 4: plan_sanitizer
print("Testing plan_sanitizer...", end=" ")
try:
    from plan_sanitizer import PlanSanitizer
    sanitizer = PlanSanitizer()
    print("✓ OK")
    modules_ok.append("plan_sanitizer")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("plan_sanitizer", str(e)))

# Test 5: embedding_model
print("Testing embedding_model...", end=" ")
try:
    from embedding_model import EmbeddingModel
    print("✓ OK")
    modules_ok.append("embedding_model")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("embedding_model", str(e)))

# Test 6: responsibility_detector
print("Testing responsibility_detector...", end=" ")
try:
    from responsibility_detector import ResponsibilityDetector
    print("✓ OK")
    modules_ok.append("responsibility_detector")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("responsibility_detector", str(e)))

# Test 7: contradiction_detector
print("Testing contradiction_detector...", end=" ")
try:
    from contradiction_detector import ContradictionDetector
    print("✓ OK")
    modules_ok.append("contradiction_detector")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("contradiction_detector", str(e)))

# Test 8: monetary_detector
print("Testing monetary_detector...", end=" ")
try:
    from monetary_detector import MonetaryDetector
    print("✓ OK")
    modules_ok.append("monetary_detector")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("monetary_detector", str(e)))

# Test 9: feasibility_scorer
print("Testing feasibility_scorer...", end=" ")
try:
    from feasibility_scorer import FeasibilityScorer
    print("✓ OK")
    modules_ok.append("feasibility_scorer")
except Exception as e:
    print(f"✗ FAIL: {e}")
    modules_fail.append(("feasibility_scorer", str(e)))

# Test 10: Dependencies externas
print("\nTesting external dependencies...")
deps_ok = []
deps_fail = []

for dep in ["numpy", "pandas", "torch", "spacy", "networkx"]:
    try:
        __import__(dep)
        print(f"  ✓ {dep}")
        deps_ok.append(dep)
    except Exception as e:
        print(f"  ✗ {dep}: {e}")
        deps_fail.append((dep, str(e)))

# Resumen
print()
print("=" * 70)
print("RESUMEN DE VERIFICACIÓN")
print("=" * 70)
print(f"\nMódulos del Sistema:")
print(f"  ✓ OK: {len(modules_ok)}/9")
print(f"  ✗ FAIL: {len(modules_fail)}/9")

print(f"\nDependencias Externas:")
print(f"  ✓ OK: {len(deps_ok)}/5")
print(f"  ✗ FAIL: {len(deps_fail)}/5")

if modules_fail:
    print("\nMódulos con problemas:")
    for mod, err in modules_fail:
        print(f"  - {mod}: {err[:100]}")

if deps_fail:
    print("\nDependencias con problemas:")
    for dep, err in deps_fail:
        print(f"  - {dep}: {err[:100]}")

total_ok = len(modules_ok) + len(deps_ok)
total_fail = len(modules_fail) + len(deps_fail)
total = total_ok + total_fail

print(f"\nTOTAL: {total_ok}/{total} componentes funcionando")

if total_fail == 0:
    print("\n" + "=" * 70)
    print("✓✓✓ INSTALACIÓN COMPLETADA EXITOSAMENTE ✓✓✓")
    print("=" * 70)
    print("\nTodos los módulos críticos están funcionando correctamente.")
    print("\nPróximos pasos:")
    print("1. Congelar configuración: python miniminimoon_orchestrator.py freeze ./config/")
    print("2. Ejecutar evaluación: python miniminimoon_orchestrator.py evaluate ./config/ plan.pdf ./output/")
    print()
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print(f"⚠ ATENCIÓN: {total_fail} componentes requieren revisión")
    print("=" * 70)
    print("\nRevisa los errores arriba y consulta TROUBLESHOOTING_ESPACIO.md")
    print()
    sys.exit(1)

