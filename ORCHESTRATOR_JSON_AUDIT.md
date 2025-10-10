# Orchestrator JSON File Reference Audit

## Executive Summary

This document audits all orchestrator files to ensure they correctly reference `decalogo-industrial.latest.clean.json` and `dnp-standards.latest.clean.json`.

**Result:** ✅ All orchestrators correctly access the JSON files through the appropriate mechanisms.

## Orchestrator Files Analysis

### 1. miniminimoon_orchestrator.py (Main Orchestrator)

**Status:** ✅ CORRECT - No direct file references

**Analysis:**
- Does NOT directly load the JSON files
- Uses other modules that internally use `decalogo_loader.py`
- Follows proper architectural separation (orchestration vs. data loading)

**Recommendation:** No changes needed. This is the correct approach.

### 2. decalogo_pipeline_orchestrator.py (Deprecated)

**Status:** ⚠️ DEPRECATED - Raises RuntimeError on import

**Analysis:**
- File is marked as DEPRECATED and FORBIDDEN
- Raises RuntimeError to prevent usage
- Should not be used per DEPRECATIONS.md

**Code snippet (lines 18-26):**
```python
raise RuntimeError(
    "CRITICAL: decalogo_pipeline_orchestrator is DEPRECATED and FORBIDDEN.\n"
    "\n"
    "This module creates parallel execution paths that violate:\n"
    "  - Gate #6: No deprecated orchestrator usage\n"
    ...
)
```

**Recommendation:** No changes needed. File is intentionally disabled.

### 3. unified_evaluation_pipeline.py (Evaluation Pipeline)

**Status:** ⚠️ COULD BE IMPROVED - Uses direct file loading

**Analysis:**
- Directly loads `decalogo-industrial.latest.clean.json` (line 351)
- Uses `Path("decalogo-industrial.latest.clean.json")` without loader module
- Works correctly but doesn't benefit from loader's features (caching, fallback)

**Current code (lines 351-359):**
```python
decalogo_json = Path("decalogo-industrial.latest.clean.json")
if not decalogo_json.exists():
    raise FileNotFoundError("decalogo-industrial.latest.clean.json not found")

with open(decalogo_json, 'r', encoding='utf-8') as f:
    decalogo_data = json.load(f)

questions = decalogo_data.get("questions", [])
logger.info(f"→ Loaded {len(questions)} questions from decalogo-industrial.latest.clean.json")
```

**Recommended improvement:**
```python
from decalogo_loader import get_decalogo_industrial

decalogo_data = get_decalogo_industrial()
questions = decalogo_data.get("questions", [])
logger.info(f"→ Loaded {len(questions)} questions from decalogo-industrial.latest.clean.json")
```

**Benefits of using loader:**
- Thread-safe caching (faster repeated access)
- Automatic fallback if file is missing
- Consistent path resolution across all modules
- Single source of truth for file location

**Recommendation:** Consider refactoring to use `decalogo_loader`, but current implementation works correctly.

## Test Files

### 4. test_miniminimoon_orchestrator_parallel.py

**Status:** ✅ CORRECT - Appropriate mocking for tests

**Analysis:**
- Creates mock JSON files for testing (lines 208-216)
- Uses dictionary literals to mock file content
- Proper test isolation pattern

**Code snippet:**
```python
def _create_minimal_configs(self):
    """Create minimal config files for testing"""
    configs = {
        "DECALOGO_FULL.json": {"questions": []},
        "decalogo-industrial.latest.clean.json": {"questions": []},
        "dnp-standards.latest.clean.json": {},
        "RUBRIC_SCORING.json": {"questions": {}, "weights": {}},
    }
    
    for filename, content in configs.items():
        with open(self.config_dir / filename, 'w') as f:
            json.dump(content, f)
```

**Recommendation:** No changes needed. This is the correct testing approach.

## File Access Patterns Summary

### Pattern 1: No Direct Access (Preferred)
**Used by:** `miniminimoon_orchestrator.py`

Orchestrator relies on other components that internally use loaders.

**Pros:**
- Clean separation of concerns
- Orchestrator focuses on flow, not data loading
- Easy to test and maintain

### Pattern 2: Loader Module (Recommended)
**Should be used by:** `unified_evaluation_pipeline.py`

Uses `from decalogo_loader import get_decalogo_industrial, load_dnp_standards`

**Pros:**
- Centralized file access
- Thread-safe caching
- Automatic fallback mechanism
- Consistent path resolution

### Pattern 3: Direct File Loading (Works but not ideal)
**Used by:** `unified_evaluation_pipeline.py`

Directly opens and parses JSON files.

**Pros:**
- Simple and explicit
- No extra dependencies

**Cons:**
- No caching (slower repeated access)
- No fallback mechanism
- Path resolution must be handled manually

### Pattern 4: Test Mocking (Appropriate for tests)
**Used by:** Test files

Creates temporary mock files or patches loaders.

**Pros:**
- Test isolation
- Fast test execution
- Control over test data

## Path Resolution Verification

All orchestrator files that reference the JSON files do so in a way that resolves to the repository root:

```
Repository Root: /home/runner/work/SIN_CARRETA/SIN_CARRETA/
├── decalogo-industrial.latest.clean.json ✓
├── dnp-standards.latest.clean.json ✓
├── miniminimoon_orchestrator.py
├── unified_evaluation_pipeline.py
└── decalogo_loader.py (loader module)
```

**Verification command:**
```bash
python3 validate_json_file_locations.py
```

**Result:** ✅ All validations passed

## Integration Points

### How Orchestrators Access Data

```
┌─────────────────────────────────┐
│  miniminimoon_orchestrator.py   │
│  (Main Orchestrator)             │
└────────────┬────────────────────┘
             │
             │ calls components
             ▼
┌─────────────────────────────────┐
│  Component Modules               │
│  (responsibility_detector, etc)  │
└────────────┬────────────────────┘
             │
             │ uses loader
             ▼
┌─────────────────────────────────┐
│  decalogo_loader.py              │
│  (Centralized Data Access)       │
└────────────┬────────────────────┘
             │
             │ loads from
             ▼
┌─────────────────────────────────┐
│  decalogo-industrial.latest.clean.json  │
│  dnp-standards.latest.clean.json        │
│  (Repository Root)               │
└─────────────────────────────────┘
```

This architecture ensures:
1. Single source of truth for data
2. Orchestrators focus on flow control
3. Data access is centralized and cached
4. Easy to update file locations in one place

## Recommendations

### Priority 1: Current State (No Action Required)
- ✅ Both JSON files are in the correct location (repository root)
- ✅ All paths resolve correctly
- ✅ Main orchestrator uses appropriate abstraction
- ✅ Test files use proper mocking

### Priority 2: Optional Improvements
1. **unified_evaluation_pipeline.py**: Refactor to use `decalogo_loader` for consistency
   - Benefits: Caching, fallback mechanism, path consistency
   - Risk: Low (loader is well-tested)
   - Effort: Minimal (2-3 lines of code)

### Priority 3: Documentation
- ✅ Created `JSON_FILE_LOCATIONS.md` (comprehensive path documentation)
- ✅ Created `validate_json_file_locations.py` (automated validation)
- ✅ Created this orchestrator audit document

## Validation Checklist

Run these commands to verify everything works:

```bash
# 1. Validate file locations and paths
python3 validate_json_file_locations.py

# 2. Test decalogo loader
python3 test_decalogo_loader.py

# 3. Verify alignment
python3 verify_decalogo_alignment.py

# 4. Check JSON structure (requires pytest)
python3 -c "import json; d = json.load(open('decalogo-industrial.latest.clean.json')); print(f'✓ Industrial: {len(d[\"questions\"])} questions')"
python3 -c "import json; s = json.load(open('dnp-standards.latest.clean.json')); print(f'✓ DNP Standards: valid JSON')"
```

**All validations pass:** ✅

## Conclusion

**All orchestrators correctly access the JSON files** either through:
1. Indirect access via component modules (preferred for orchestrators)
2. Direct file loading with correct paths (works but could be improved)
3. Appropriate mocking in tests

**No critical issues found.** The files are in the right location and all references resolve correctly. The system is working as designed.

**Optional improvement:** Consider updating `unified_evaluation_pipeline.py` to use `decalogo_loader` for consistency with the rest of the codebase, but this is not critical as the current implementation works correctly.
