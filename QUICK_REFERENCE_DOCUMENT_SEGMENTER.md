# Quick Reference: Document Segmenter Refactoring

## What Changed?

The Document Segmenter now uses the **official DECALOGO structure** from `decalogo-industrial.latest.clean.json` v1.0.

## Dimension Structure

### OLD (Incorrect) ❌
```python
DE-1: Logic Intervention Framework
DE-2: Thematic Inclusion
DE-3: Participation and Governance
DE-4: Results Orientation
```

### NEW (Correct) ✅
```python
D1: INSUMOS (diagnóstico, líneas base, recursos, capacidades) - 55 questions
D2: ACTIVIDADES (formalización, mecanismos causales) - 60 questions
D3: PRODUCTOS (outputs con indicadores verificables) - 60 questions
D4: RESULTADOS (outcomes con métricas) - 60 questions
D5: IMPACTOS (efectos largo plazo) - 57 questions
D6: CAUSALIDAD (teoría de cambio explícita) - 55 questions
```

## Usage Examples

### Basic Usage (Backward Compatible)
```python
from document_segmenter import DocumentSegmenter

segmenter = DocumentSegmenter()

# String input (works with existing orchestrator code)
segments = segmenter.segment("Diagnóstico con líneas base...")
```

### New Format (Recommended)
```python
# Dict input with structured document
doc_struct = {
    "full_text": "Diagnóstico con líneas base...",
    "sections": {...}
}
segments = segmenter.segment(doc_struct)
```

### Accessing Dimensions
```python
for segment in segments:
    print(f"Text: {segment.text[:50]}...")
    print(f"Type: {segment.section_type.value}")
    print(f"Dimensions: {segment.decalogo_dimensions}")  # ['D1', 'D2', etc.]
```

## Section Types → Dimensions Mapping

### D1: INSUMOS
- `DIAGNOSTIC` - Diagnóstico, problemática
- `BASELINE` - Líneas base, series temporales
- `RESOURCES` - Recursos asignados, PPI
- `CAPACITY` - Capacidades institucionales
- `BUDGET` - Presupuesto financiero
- `PARTICIPATION` - Gobernanza

### D2: ACTIVIDADES
- `ACTIVITY` - Actividades formalizadas
- `MECHANISM` - Mecanismos causales
- `INTERVENTION` - Teoría de intervención
- `TIMELINE` - Cronograma, calendario

### D3: PRODUCTOS
- `PRODUCT` - Productos, bienes y servicios
- `OUTPUT` - Outputs verificables

### D4: RESULTADOS
- `RESULT` - Resultados esperados
- `OUTCOME` - Outcomes con métricas
- `INDICATOR` - Indicadores
- `MONITORING` - Seguimiento

### D5: IMPACTOS
- `IMPACT` - Impactos largo plazo
- `LONG_TERM_EFFECT` - Efectos duraderos

### D6: CAUSALIDAD
- `CAUSAL_THEORY` - Teoría de cambio
- `CAUSAL_LINK` - Encadenamiento causal

### Multi-dimensional (Legacy)
- `VISION` - D1+D6
- `OBJECTIVE` - D4+D6
- `STRATEGY` - D2+D6
- `RESPONSIBILITY` - D1+D2

## Testing

### Run Unit Tests
```bash
python3 test_document_segmenter_decalogo_alignment.py
# Expected: 11/11 tests passing
```

### Run Verification
```bash
python3 verify_document_segmenter_refactoring.py
# Expected: 5/5 tests passing
```

### Run Demo
```bash
python3 demo_document_segmenter_refactored.py
# Shows dimension mapping in action
```

## Migration Guide

### If you were using old dimension codes:

❌ **Don't do this:**
```python
if "DE-1" in segment.decalogo_dimensions:
    # This will never match!
```

✅ **Do this instead:**
```python
if "D1" in segment.decalogo_dimensions:
    # Correct!
```

### If you need to map old to new:

```python
OLD_TO_NEW = {
    "DE-1": ["D1", "D2", "D4", "D6"],  # Logic Intervention → various
    "DE-2": ["D1", "D3"],               # Thematic Inclusion → INSUMOS, PRODUCTOS
    "DE-3": ["D1"],                     # Participation → INSUMOS (capacidades)
    "DE-4": ["D4"],                     # Results Orientation → RESULTADOS
}
```

## Common Questions

**Q: Will my existing code break?**  
A: No. The refactoring maintains backward compatibility. String inputs still work.

**Q: Do I need to update my code?**  
A: Only if you were explicitly checking for `DE-*` dimensions. Otherwise, no changes needed.

**Q: How do I know which dimension a segment belongs to?**  
A: Check `segment.decalogo_dimensions` - it's a list of dimension codes like `['D1', 'D4']`.

**Q: Can a segment belong to multiple dimensions?**  
A: Yes! Some section types (like OBJECTIVE) map to multiple dimensions (D4+D6).

## Documentation

- Full details: `DOCUMENT_SEGMENTER_REFACTORING_SUMMARY.md`
- Demo: `demo_document_segmenter_refactored.py`
- Tests: `test_document_segmenter_decalogo_alignment.py`
- Verification: `verify_document_segmenter_refactoring.py`

## References

- Official structure: `decalogo-industrial.latest.clean.json` v1.0
- Dimension definitions: `ALINEACION_DECALOGO_CORRECCION_CRITICA.txt`
- Flow specs: `FLUJOS_CRITICOS_GARANTIZADOS.md` (Flow #3)
