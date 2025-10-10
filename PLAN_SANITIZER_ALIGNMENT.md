# Plan Sanitizer - DECALOGUE Alignment

## Summary

The `plan_sanitizer.py` module has been refactored to align with the JSON DECALOGUE structure, implementing a frozen contract design that ensures deterministic behavior and maintainability.

## Changes Made

### 1. Frozen Contract Implementation

**Zero-arg constructor:**
```python
sanitizer = PlanSanitizer()  # No kwargs backdoors
```

**Factory methods:**
```python
# Explicit configuration
config = PlanSanitizerConfig(preserve_structure=True, ...)
sanitizer = PlanSanitizer.from_config(config)

# Legacy keyword arguments (strict validation)
sanitizer = PlanSanitizer.legacy(preserve_structure=True, ...)

# Module-level convenience
sanitizer = create_plan_sanitizer(preserve_structure=True, ...)
```

### 2. Configuration System

- **Immutable dataclass:** `PlanSanitizerConfig` (frozen=True)
- **Legacy aliases:** Support for historical parameter names
  - `keep_structure` → `preserve_structure`
  - `tag_elements` → `tag_key_elements`
  - `aggressive_mode` → `aggressive_cleaning`
- **Fail-closed validation:** Unknown flags raise `ValueError`

### 3. KEY_ELEMENTS Alignment

The KEY_ELEMENTS mapping has been verified to align with DECALOGUE evaluation needs:

```python
KEY_ELEMENTS = {
    "indicators": [        # Metrics, metas, líneas base, resultados esperados
        r"(?i)indicador(?:es)?",
        r"(?i)meta(?:s)?",
        r"(?i)l[ií]nea(?:s)?\s+base",
        r"(?i)resultado(?:s)?\s+esperado(?:s)?",
    ],
    "diagnostics": [      # Diagnóstico, situación actual, problemática
        r"(?i)diagn[óo]stico",
        r"(?i)situaci[óo]n\s+actual",
        r"(?i)problem[áa]tica",
        r"(?i)contexto",
    ],
    "participation": [    # Participación, consultas, mesas técnicas
        r"(?i)participaci[óo]n",
        r"(?i)consulta(?:s)?",
        r"(?i)mesa(?:s)?\s+t[ée]cnica(?:s)?",
        r"(?i)concertaci[óo]n",
    ],
    "monitoring": [       # Seguimiento, monitoreo, evaluación
        r"(?i)seguimiento",
        r"(?i)monitoreo",
        r"(?i)evaluaci[óo]n",
        r"(?i)control",
        r"(?i)tablero(?:s)?",
    ],
}
```

These elements support evaluation across all 6 DECALOGUE dimensions (D1-D6):
- **D1:** Diagnóstico / líneas base
- **D2:** Actividades
- **D3:** Productos
- **D4:** Resultados
- **D5:** Impactos
- **D6:** Teoría de cambio

### 4. Enhanced Text Processing

- **Key element tagging:** Wraps detected elements with `<KEY:type>...</KEY:type>`
- **Structure preservation:** Maintains headings, lists, bullets
- **Special character handling:** Normalizes OCR artifacts and ligatures
- **Aggressive cleaning mode:** Optional parameter for `clean_policy_text()`

### 5. Supporting Changes

**text_processor.py:**
- Added `aggressive` parameter to `clean_policy_text()` function
- Backward compatible (defaults to False)

## Verification

All core functionality has been verified:

✓ Zero-arg constructor  
✓ Factory methods (from_config, legacy)  
✓ Legacy aliases  
✓ Fail-closed validation  
✓ KEY_ELEMENTS alignment  
✓ Text sanitization  
✓ File sanitization  
✓ Configuration exposure  
✓ Statistics tracking  

## API Contract (Frozen)

```python
class PlanSanitizer:
    __init__(self)  # ZERO-ARG only
    sanitize_text(self, text: str) -> str
    sanitize_file(self, input_path: Union[str, Path], 
                  output_path: Optional[Union[str, Path]] = None) -> str
    get_sanitization_stats(self) -> Dict[str, Any]
    
    # Factory methods (do NOT widen __init__)
    @classmethod
    from_config(cls, cfg: PlanSanitizerConfig) -> PlanSanitizer
    
    @classmethod
    legacy(cls, **legacy_flags) -> PlanSanitizer
    
    # Property for audits
    @property
    resolved_config(self) -> PlanSanitizerConfig
```

## Notes

- No import-time side effects (beyond logger configuration)
- Deterministic behavior guaranteed
- Thread-safe configuration (frozen dataclass)
- Full backward compatibility with legacy code via factory methods
- Aligns with decalogo-industrial.latest.clean.json structure (300 questions, 6 dimensions)
