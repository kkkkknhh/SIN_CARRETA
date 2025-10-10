# ScoreBand Enum and QuestionnaireEngine Initialization Fix

## Problem Statement

The system was failing to start with the following error:

```
TypeError: ScoreBand.__init__() takes 3 positional arguments but 5 were given
```

This error occurred during the import of `questionnaire_engine.py` when the `ScoreBand` Enum was being instantiated.

## Root Cause Analysis

### Issue 1: ScoreBand Enum Pattern

The `ScoreBand` Enum was incorrectly implemented using a custom `__init__` method:

```python
class ScoreBand(Enum):
    EXCELENTE = (85, 100, "🟢", "Diseño causal robusto")
    # ...
    
    def __init__(self, min_score, max_score, color, description):
        self.min_score = min_score
        self.max_score = max_score
        self.color = color
        self.description = description
```

**Problem**: When Python creates Enum members with tuple values, it doesn't automatically unpack the tuple into the `__init__` parameters. Instead, it passes the entire tuple as a single value to `__init__`. The custom `__init__` expected 4 unpacked arguments (min_score, max_score, color, description) plus self, but received self + the tuple value, causing a signature mismatch.

### Issue 2: QuestionnaireEngine Initialization

The orchestrator was attempting to instantiate `QuestionnaireEngine` with parameters:

```python
self.questionnaire_engine = QuestionnaireEngine(
    evidence_registry=self.evidence_registry,
    rubric_path=rubric_path
)
```

But the `__init__` method only accepted `self`:

```python
def __init__(self):
    """Initialize with complete question library"""
```

## Solution Implemented

### Fix 1: ScoreBand Enum - Use @property Pattern

Changed the Enum to use Python's proper pattern for multi-value enums:

```python
class ScoreBand(Enum):
    """Score interpretation bands"""
    EXCELENTE = (85, 100, "🟢", "Diseño causal robusto")
    BUENO = (70, 84, "🟡", "Diseño sólido con vacíos menores")
    SATISFACTORIO = (55, 69, "🟠", "Cumple mínimos, requiere mejoras")
    INSUFICIENTE = (40, 54, "🔴", "Vacíos críticos")
    DEFICIENTE = (0, 39, "⚫", "Ausencia de diseño causal")

    @property
    def min_score(self):
        return self.value[0]

    @property
    def max_score(self):
        return self.value[1]

    @property
    def color(self):
        return self.value[2]

    @property
    def description(self):
        return self.value[3]

    @classmethod
    def classify(cls, score_percentage: float) -> 'ScoreBand':
        """Classify score into band"""
        for band in cls:
            if band.min_score <= score_percentage <= band.max_score:
                return band
        return cls.DEFICIENTE
```

**Key changes**:
- Removed custom `__init__` method
- Added `@property` decorators for each attribute
- Values accessed via `self.value[index]` tuple unpacking
- Maintains same API for consumers (e.g., `band.min_score` still works)

### Fix 2: QuestionnaireEngine - Accept Required Parameters

Updated `__init__` to accept the parameters that the orchestrator passes:

```python
def __init__(self, evidence_registry=None, rubric_path=None):
    """Initialize with complete question library"""
    self.evidence_registry = evidence_registry
    self.rubric_path = rubric_path
    
    # ... existing initialization code ...
    
    # Load rubric if path provided
    if rubric_path:
        self._load_rubric()
    
    logger.info("✅ QuestionnaireEngine v2.0 initialized with COMPLETE 30×10 structure")
```

Added supporting method:

```python
def _load_rubric(self):
    """Load rubric data from the provided path."""
    if not self.rubric_path:
        return
    
    try:
        with open(self.rubric_path, 'r', encoding='utf-8') as f:
            self.rubric_data = json.load(f)
        logger.info(f"✓ Rubric loaded from {self.rubric_path}")
    except Exception as e:
        logger.warning(f"Could not load rubric from {self.rubric_path}: {e}")
        self.rubric_data = None
```

**Key changes**:
- Added `evidence_registry` and `rubric_path` parameters with default values (`None`)
- Store parameters as instance attributes
- Call `_load_rubric()` if rubric_path is provided
- Maintain backward compatibility (existing tests use `QuestionnaireEngine()` with no args)

## Backward Compatibility

Both fixes maintain full backward compatibility:

1. **ScoreBand**: The API remains identical. Code accessing `band.min_score`, `band.max_score`, etc. continues to work.

2. **QuestionnaireEngine**: Default parameter values allow existing code to continue using:
   ```python
   engine = QuestionnaireEngine()  # Still works
   ```
   While also supporting the new pattern:
   ```python
   engine = QuestionnaireEngine(
       evidence_registry=registry,
       rubric_path=path
   )
   ```

## Testing

Created comprehensive test suite in `test_questionnaire_enum_fix.py` that verifies:

1. ✅ ScoreBand uses @property pattern (no custom __init__)
2. ✅ ScoreBand has all required properties (min_score, max_score, color, description)
3. ✅ QuestionnaireEngine.__init__ accepts evidence_registry and rubric_path
4. ✅ Both parameters have default values
5. ✅ Orchestrator calls QuestionnaireEngine with correct keyword arguments
6. ✅ Python syntax is valid in both modified files

All existing tests continue to pass without modification.

## Files Modified

1. **questionnaire_engine.py**:
   - `ScoreBand` enum class (lines 36-66)
   - `QuestionnaireEngine.__init__` method (lines 1141-1166)
   - Added `QuestionnaireEngine._load_rubric` method (lines 1168-1179)

## Impact

This fix allows the MINIMINIMOON evaluation system to:
- Successfully import the questionnaire_engine module
- Instantiate QuestionnaireEngine with the evidence registry and rubric path
- Properly classify scores using the ScoreBand enum
- Maintain compatibility with all existing code and tests

The system can now proceed past the initialization phase and execute the full evaluation pipeline.
