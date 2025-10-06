# DEPRECATIONS - Sistema MINIMINIMOON

**Versión:** 2.0.0  
**Fecha de emisión:** 6 de octubre de 2025  
**Estado:** VIGENTE

---

## Módulos Deprecados (PROHIBIDO SU USO)

### 1. `decalogo_pipeline_orchestrator.py` ❌ DEPRECATED

**Estado:** PROHIBIDO - RuntimeError al importar  
**Fecha de corte:** 6 de octubre de 2025  
**Razón:** Crea rutas de ejecución paralelas que violan:
- Gate #6: No deprecated orchestrator usage
- Unicidad del evidence registry (single source of truth)
- Enforcement del orden canónico de flujo
- Integridad del audit trail

**Reemplazo obligatorio:**
```python
# ❌ PROHIBIDO (lanza RuntimeError)
from decalogo_pipeline_orchestrator import DecalogoPipelineOrchestrator

# ✅ CORRECTO
from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
```

**Mapa de reemplazo de flujos:**

| Flujo Deprecado | Reemplazo Canónico |
|----------------|-------------------|
| `decalogo_pipeline_orchestrator → monetary_detector` | `miniminimoon_orchestrator → monetary_detector` |
| `decalogo_pipeline_orchestrator → causal_pattern_detector` | `miniminimoon_orchestrator → causal_pattern_detector` |
| `decalogo_pipeline_orchestrator → teoria_cambio` | `miniminimoon_orchestrator → teoria_cambio` |
| `decalogo_pipeline_orchestrator → feasibility_scorer` | `miniminimoon_orchestrator → feasibility_scorer` |
| `decalogo_pipeline_orchestrator → responsibility_detector` | `miniminimoon_orchestrator → responsibility_detector` |
| `decalogo_pipeline_orchestrator → contradiction_detector` | `miniminimoon_orchestrator → contradiction_detector` |
| `decalogo_pipeline_orchestrator → document_segmenter` | `miniminimoon_orchestrator → document_segmenter` |

---

## Enforcement

El archivo `decalogo_pipeline_orchestrator.py` contiene un **RuntimeError obligatorio** que impide su uso:

```python
raise RuntimeError(
    "CRITICAL: decalogo_pipeline_orchestrator is DEPRECATED and FORBIDDEN.\n"
    "\n"
    "This module creates parallel execution paths that violate:\n"
    "  - Gate #6: No deprecated orchestrator usage\n"
    "  - Evidence registry uniqueness (single source of truth)\n"
    "  - Deterministic flow order enforcement\n"
    "  - Audit trail integrity\n"
    "\n"
    "REQUIRED ACTION:\n"
    "  Use: from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator\n"
)
```

---

## Gates de Aceptación (Verificación Automática)

Todos los sistemas de CI/CD deben verificar:

1. **Gate #1:** `verify_frozen_config() == True` antes de ejecución
2. **Gate #2:** `flow_runtime.json` idéntico a `flow_doc.json` + contratos OK
3. **Gate #3:** `evidence_hash` estable con mismo input
4. **Gate #4:** Cobertura `answers_report.summary.total_questions ≥ 300`
5. **Gate #5:** `rubric_check.py` sin missing/extra
6. **Gate #6:** ⚠️ **Ningún uso de orquestador deprecado**

---

## Migración Obligatoria

### Antes (❌ Deprecado):
```python
from decalogo_pipeline_orchestrator import DecalogoPipelineOrchestrator

orchestrator = DecalogoPipelineOrchestrator(config_path)
results = orchestrator.run_pipeline(plan_path)
```

### Después (✅ Canónico):
```python
from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator

orchestrator = CanonicalDeterministicOrchestrator(
    config_dir=".",
    enable_validation=True,
    flow_doc_path="flow_doc.json"
)
results = orchestrator.process_plan_deterministic(plan_path)
```

---

## Beneficios del Orquestador Canónico

1. **Determinismo garantizado:** Seeds fijos (random=42, numpy=42, torch=42)
2. **Trazabilidad completa:** EvidenceRegistry único con provenance
3. **Validación automática:** Pre/post checks con gates duros
4. **Reproducibilidad:** `evidence_hash` y `flow_hash` deterministas
5. **Cobertura garantizada:** 300/300 preguntas con evidencia ≥1
6. **Alineación rubrica:** 1:1 preguntas↔pesos verificado

---

## Documentación de Referencia

- **Flujos críticos:** Ver `FLUJOS_CRITICOS_GARANTIZADOS.md`
- **Arquitectura:** Ver `ARCHITECTURE.md`
- **Deployment:** Ver `DEPLOYMENT_INFRASTRUCTURE.md`
- **Orden canónico:** Ver `flow_doc.json`
- **Verificación:** Ejecutar `rubric_check.py` y `verify_critical_flows.py`

---

## Soporte

Para dudas sobre migración:
- Revisar: `miniminimoon_orchestrator.py` (docstrings completos)
- Ejecutar: `python miniminimoon_cli.py --help`
- Verificar: `python verify_critical_flows.py`

**Nota:** No hay marcha atrás. El sistema deprecado está deshabilitado permanentemente por integridad arquitectónica.

