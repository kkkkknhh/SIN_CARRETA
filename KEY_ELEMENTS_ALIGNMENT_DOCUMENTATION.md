# KEY_ELEMENTS Alignment - Implementation Documentation

## Summary

The `KEY_ELEMENTS` dictionary in `plan_sanitizer.py` has been updated to align with the canonical 6-dimension structure defined in `decalogo-industrial.latest.clean.json`.

## Problem Statement

The original `KEY_ELEMENTS` had **4 dimensions** using old **DE-** nomenclature:
- `indicators` (DE-1: Logical Intervention Framework)
- `diagnostics` (DE-2: Thematic Inclusion)
- `participation` (DE-3: Participation and Governance)
- `monitoring` (DE-4: Results Orientation)

This was misaligned with the canonical structure which has **6 dimensions (D1-D6)**.

## Solution

Updated `KEY_ELEMENTS` to match the 6-dimension structure:

### D1: INSUMOS (Inputs)
**Key:** `insumos`  
**Focus:** Diagnóstico, líneas base, recursos, capacidades institucionales  
**Patterns (8):**
- diagnóstico
- línea base
- recursos
- capacidades institucionales
- situación actual
- problemática
- brechas
- coherencia

### D2: ACTIVIDADES (Activities)
**Key:** `actividades`  
**Focus:** Formalización, mecanismos causales, teoría de intervención  
**Patterns (7):**
- actividades
- mecanismo causal
- intervención
- responsable
- instrumento
- población diana
- riesgo de implementación

### D3: PRODUCTOS (Products/Outputs)
**Key:** `productos`  
**Focus:** Outputs con indicadores verificables, trazabilidad  
**Patterns (7):**
- productos
- outputs
- indicadores verificables
- trazabilidad
- cobertura
- dosificación
- entregables

### D4: RESULTADOS (Results/Outcomes)
**Key:** `resultados`  
**Focus:** Outcomes con métricas, encadenamiento causal  
**Patterns (7):**
- resultados
- outcomes
- métricas
- metas
- encadenamiento causal
- ventana de maduración
- nivel de ambición

### D5: IMPACTOS (Impacts)
**Key:** `impactos`  
**Focus:** Efectos largo plazo, proxies, alineación marcos  
**Patterns (8):**
- impactos
- efectos de largo plazo
- proxies
- transmisión
- rezagos
- PND (Plan Nacional de Desarrollo)
- ODS (Objetivos de Desarrollo Sostenible)
- marcos nacional/global

### D6: CAUSALIDAD (Causality)
**Key:** `causalidad`  
**Focus:** Teoría de cambio explícita, DAG, validación lógica  
**Patterns (12):**
- teoría de cambio
- diagrama causal
- DAG (Directed Acyclic Graph)
- cadena causal
- lógica causal
- supuestos verificables
- mediadores
- moderadores
- validación lógica
- seguimiento
- monitoreo
- evaluación

## Canonical Source

The alignment is based on:
- **Primary:** `decalogo-industrial.latest.clean.json` (300 questions, 6 dimensions × 50 questions)
- **Documentation:** `ALINEACION_DECALOGO_CORRECCION_CRITICA.txt`
- **Standards:** `dnp-standards.latest.clean.json`

## Validation

### Structure Verification
```
✓ 6 dimensions (D1-D6)
✓ 49 total regex patterns
✓ All patterns compile successfully
✓ No old DE-X nomenclature remaining
```

### Canonical JSON Verification
```
✓ JSON has 300 questions
✓ 6 dimensions: D1, D2, D3, D4, D5, D6
✓ Each dimension has exactly 50 questions
✓ D1: INSUMOS (Q1-Q5 per point × 10 points)
✓ D2: ACTIVIDADES (Q6-Q10 per point × 10 points)
✓ D3: PRODUCTOS (Q11-Q15 per point × 10 points)
✓ D4: RESULTADOS (Q16-Q20 per point × 10 points)
✓ D5: IMPACTOS (Q21-Q25 per point × 10 points)
✓ D6: CAUSALIDAD (Q26-Q30 per point × 10 points)
```

### Pattern Matching Test
All 49 patterns successfully match their intended keywords in Spanish policy documents.

## Impact

### Files Modified
1. **plan_sanitizer.py** - Updated KEY_ELEMENTS dictionary
2. **test_key_elements_alignment.py** - New validation test suite

### Backward Compatibility
The change is **backward compatible** because:
- The code uses `KEY_ELEMENTS.items()` and `KEY_ELEMENTS.keys()` dynamically
- No hardcoded references to old dimension names in logic
- Only the dictionary keys changed, not the structure or API

### Downstream Effects
- `PlanSanitizer` class continues to work without modification
- `key_element_patterns` automatically adapts to 6 dimensions
- Statistics tracking now reports 6 dimensions instead of 4

## Module Mappings

Based on `ALINEACION_DECALOGO_CORRECCION_CRITICA.txt`, dimensions map to modules:

- **D1 (INSUMOS)** → evidence_registry, monetary_detector, pdm_nlp_modules
- **D2 (ACTIVIDADES)** → plan_processor, responsibility_detector, causal_pattern_detector
- **D3 (PRODUCTOS)** → plan_processor, evidence_registry, contradiction_detector
- **D4 (RESULTADOS)** → teoria_cambio, feasibility_scorer, causal_pattern_detector
- **D5 (IMPACTOS)** → teoria_cambio, dag_validation, feasibility_scorer
- **D6 (CAUSALIDAD)** → teoria_cambio, dag_validation, causal_pattern_detector

## Testing

Run the alignment test:
```bash
python3 test_key_elements_alignment.py
```

Expected output:
```
✓ KEY_ELEMENTS has 6 dimensions
✓ Dimension names are correct
✓ No old DE-X nomenclature found
✓ All dimensions have regex patterns
✓ Canonical JSON has correct 6-dimension structure
✓ All tests passed!
```

## References

- Issue: "THESE ARE NOT THE DIMENSIONS! Check the Decalogo JSON"
- Canonical structure: D1-D6 (INSUMOS, ACTIVIDADES, PRODUCTOS, RESULTADOS, IMPACTOS, CAUSALIDAD)
- Total questions: 300 (50 per dimension)
- 10 policy points × 30 questions each (5 questions × 6 dimensions)

## Date
October 10, 2025

## Status
✅ COMPLETED - Fully aligned with canonical structure
