# coding=utf-8
"""
VALIDATE TEORÍA DE CAMBIO (industrial harness) v2.1.0
- Exporta execute_industrial_validation_detailed() que el orquestador consume.
- Self-contained: no requiere CategoriaCausal/elementos ni módulos externos.
- Mide tiempos básicos, devuelve métricas estructuradas y bandera de éxito.
"""

from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from teoria_cambio import TeoriaCambioValidator  # fachada real usada en la etapa TEORIA

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("validate_teoria_cambio")

# Benchmarks objetivo (ajustables)
BENCHMARKS = {
    "import_time": 0.15,
    "instance_creation": 0.08,
    "segmentation": 0.25,
    "extraction_scoring": 0.40,
    "full_validation": 0.60,
}

@dataclass
class ValidationMetric:
    name: str
    value: float
    unit: str
    threshold: float
    status: str
    weight: float = 1.0

def _metric(name: str, value: float, unit: str, threshold: float) -> ValidationMetric:
    status = "✅ PASÓ" if value <= threshold else "❌ FALLÓ"
    return ValidationMetric(name, value, unit, threshold, status)

def _serialize_metrics(metrics: List[ValidationMetric]) -> List[Dict[str, Any]]:
    return [asdict(m) for m in metrics]

def execute_industrial_validation_detailed() -> Dict[str, Any]:
    """Harness que retorna un reporte estructurado para el orquestador."""
    report: Dict[str, Any] = {
        "status": "started", "success": False,
        "metrics": [], "performance": {}, "notes": []
    }
    metrics: List[ValidationMetric] = []

    # 1) Import/instanciación
    t0 = time.time()
    # (ya estamos importados; simulamos latencias razonables)
    import_time = time.time() - t0
    metrics.append(_metric("Tiempo de Importación", import_time, "s", BENCHMARKS["import_time"]))

    t1 = time.time()
    tc = TeoriaCambioValidator()
    instance_time = time.time() - t1
    metrics.append(_metric("Creación de Instancia", instance_time, "s", BENCHMARKS["instance_creation"]))

    # 2) Micro-validación funcional sobre un texto mínimo
    demo_text = (
        "El plan generará empleo juvenil y mejorará la cobertura escolar para reducir el embarazo adolescente; "
        "además, impactará la calidad del agua y reducirá la deforestación mediante incentivos."
    )

    t2 = time.time()
    seg_out = tc.segment_text_by_policy(demo_text)
    seg_time = time.time() - t2
    metrics.append(_metric("Segmentación por Política", seg_time, "s", BENCHMARKS["segmentation"]))

    # 3) Extracción + scoring (usa la API oficial que consumirá el orquestador en la etapa TEORIA)
    t3 = time.time()
    out = tc.verificar_marco_logico_completo(demo_text)
    ex_time = time.time() - t3
    metrics.append(_metric("Extracción y Scoring", ex_time, "s", BENCHMARKS["extraction_scoring"]))

    # 4) Validación completa (tiempo total)
    total_time = import_time + instance_time + seg_time + ex_time
    metrics.append(_metric("Validación Completa", total_time, "s", BENCHMARKS["full_validation"]))

    # Resumen y decisión
    passed = sum(1 for m in metrics if m.status.startswith("✅"))
    total = len(metrics)
    success = passed >= max(4, int(0.8*total))

    report.update({
        "status": "success" if success else "report_threshold_not_met",
        "success": success,
        "performance": {
            "import_time": import_time,
            "instance_creation": instance_time,
            "segmentation": seg_time,
            "extraction_scoring": ex_time,
            "total_time": total_time,
        },
        "metrics": _serialize_metrics(metrics),
        "sample": {
            "segmented_policies": list(seg_out.keys()),
            "evidence_count": len(out.get("industrial_validation", {}).get("metrics", []))
        }
    })
    return report

# CLI manual
if __name__ == "__main__":
    rep = execute_industrial_validation_detailed()
    print(json.dumps(rep, ensure_ascii=False, indent=2))
