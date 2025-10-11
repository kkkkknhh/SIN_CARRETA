# Canonical Path Migration - Summary

## Overview

This migration establishes a hermetic, enforceable system for managing the two canonical JSON artifacts required by the MINIMINIMOON evaluation system. All path references have been standardized and centralized to prevent drift and ensure reproducibility.

## What Changed

### 1. File Locations (MOVED)

**Before:**
```
/home/runner/work/SIN_CARRETA/SIN_CARRETA/
├── decalogo-industrial.latest.clean.json  (root directory)
└── dnp-standards.latest.clean.json        (root directory)
```

**After:**
```
/home/runner/work/SIN_CARRETA/SIN_CARRETA/
├── bundles/
│   └── decalogo-industrial.latest.clean.json  ✓ CANONICAL
└── standards/
    └── dnp-standards.latest.clean.json        ✓ CANONICAL
```

### 2. Central Path Resolver (NEW)

**Created:** `repo_paths.py`

This module provides the single source of truth for all path resolution:

```python
from repo_paths import get_decalogo_path, get_dnp_path

# Always returns canonical paths with validation
decalogo_path = get_decalogo_path()  # /bundles/decalogo-industrial.latest.clean.json
dnp_path = get_dnp_path()            # /standards/dnp-standards.latest.clean.json
```

**Features:**
- Runtime validation of canonical filenames
- FileNotFoundError if files missing
- Support for overrides via environment variables
- Thread-safe, immutable paths

### 3. Updated Files (REFACTORED)

**Core Modules:**
- ✅ `decalogo_loader.py` - Now uses `repo_paths.get_decalogo_path()` and `get_dnp_path()`
- ✅ `unified_evaluation_pipeline.py` - Uses central resolver
- ✅ `questionnaire_engine.py` - Uses central resolver
- ✅ `demo_questionnaire_driven_system.py` - Uses central resolver

**Configuration:**
- ✅ `pdm_contra/config/decalogo.yaml` - Updated to `../../bundles/` and `../../standards/`

**Test Files:**
- ✅ `test_decalogo_alignment_fix.py` - Uses `get_decalogo_path()` and `get_dnp_path()`
- ✅ `test_dnp_standards_json.py` - Uses `get_dnp_path()`
- ✅ `test_key_elements_alignment.py` - Uses `get_decalogo_path()`
- ✅ `tests/test_loader_compat.py` - Updated with canonical paths

**Validation:**
- ✅ `validate_json_file_locations.py` - Refactored to use `repo_paths`
- ✅ `verify_decalogo_alignment.py` - Uses `get_decalogo_path()`
- ✅ `verify_alignment_comprehensive.py` - Uses central resolver

**Documentation:**
- ✅ `JSON_FILE_LOCATIONS.md` - Fully updated with new structure
- ✅ `README_JSON_LOCATIONS.md` - Updated with canonical paths
- ✅ `QUICKREF_JSON_LOCATIONS.md` - Updated with resolver usage

### 4. Validation Guards (NEW)

**Created:** `tools/check_canonical_paths.py`

Scans the entire repository for non-canonical path references:
- Detects old path patterns
- Reports file, line number, and context
- Used in CI to prevent regressions

**Pre-commit Hook:**
Added to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: check-canonical-paths
      name: Check canonical artifact paths
      entry: python3 tools/check_canonical_paths.py
      language: system
      pass_filenames: false
      always_run: true
```

**CI Integration:**
Added to `.github/workflows/canonical-integration.yml`:
```yaml
- name: Validate canonical artifact paths
  run: |
    source venv/bin/activate
    python tools/check_canonical_paths.py
```

## Migration Pattern

### Before (DEPRECATED)
```python
# ❌ Old pattern - DO NOT USE
from pathlib import Path
decalogo_path = Path(__file__).parent / "decalogo-industrial.latest.clean.json"
dnp_path = Path(__file__).parent / "dnp-standards.latest.clean.json"
```

### After (REQUIRED)
```python
# ✅ New pattern - REQUIRED
from repo_paths import get_decalogo_path, get_dnp_path

decalogo_path = get_decalogo_path()
dnp_path = get_dnp_path()

# Optional: Environment variable override (filename must still be canonical)
import os
decalogo_path = get_decalogo_path(os.getenv("DECALOGO_PATH_OVERRIDE"))
dnp_path = get_dnp_path(os.getenv("DNP_PATH_OVERRIDE"))
```

## Benefits

1. **Single Source of Truth**: All paths resolved through one module
2. **Validation at Runtime**: Ensures filenames are exactly correct
3. **Prevention of Drift**: Pre-commit hooks and CI block non-canonical paths
4. **Easy Migration**: Future moves only require updating `repo_paths.py`
5. **Environment Override**: Supports testing with alternative files (must have canonical names)
6. **Reproducibility**: Deterministic paths across all environments

## Validation

### Manual Checks
```bash
# Verify files exist in canonical locations
ls -lh bundles/decalogo-industrial.latest.clean.json
ls -lh standards/dnp-standards.latest.clean.json

# Test path resolver
python3 -c "from repo_paths import get_decalogo_path, get_dnp_path; print(get_decalogo_path()); print(get_dnp_path())"

# Test loader
python3 -c "from decalogo_loader import get_decalogo_industrial, load_dnp_standards; print('Questions:', len(get_decalogo_industrial()['questions']))"

# Run path validation
python3 tools/check_canonical_paths.py

# Run full validation
python3 validate_json_file_locations.py
```

### Expected Results
```
✅ /home/runner/work/SIN_CARRETA/SIN_CARRETA/bundles/decalogo-industrial.latest.clean.json
✅ /home/runner/work/SIN_CARRETA/SIN_CARRETA/standards/dnp-standards.latest.clean.json
Questions: 300
✅ OK: All paths are canonical
✅ ALL VALIDATIONS PASSED
```

## Remaining Items

Most remaining references are in:
- **Documentation comments** - Informational only, not executable
- **Test mock data** - Uses string keys, not actual file paths  
- **Markdown files** - Historical context and examples

These do not affect runtime behavior and can be updated over time.

## Critical Paths to Validate

When testing the system, ensure these critical flows work:

1. **Decalogo Loading**: `from decalogo_loader import get_decalogo_industrial`
2. **DNP Standards Loading**: `from decalogo_loader import load_dnp_standards`
3. **Path Resolution**: `from repo_paths import get_decalogo_path, get_dnp_path`
4. **Pipeline Execution**: `unified_evaluation_pipeline.py` runs without errors
5. **Test Suite**: Core tests pass (especially `test_decalogo_loader.py`, `test_dnp_standards_json.py`)

## Rollback Instructions

If needed, rollback is straightforward:

1. Move files back to root:
   ```bash
   mv bundles/decalogo-industrial.latest.clean.json .
   mv standards/dnp-standards.latest.clean.json .
   ```

2. Revert commits:
   ```bash
   git revert <commit-hash>
   ```

3. Delete new files:
   ```bash
   rm repo_paths.py
   rm tools/check_canonical_paths.py
   ```

However, this is **not recommended** as the new system provides significant benefits for reproducibility and validation.

## Future Enhancements

Potential future improvements:
1. Add file content validation (checksums/hashes)
2. Version pinning with automatic validation
3. Automated migration tools for future path changes
4. Integration with artifact versioning system

## Contact

For questions or issues:
- Check `JSON_FILE_LOCATIONS.md` for detailed documentation
- Check `QUICKREF_JSON_LOCATIONS.md` for quick reference
- Run validation scripts: `tools/check_canonical_paths.py` and `validate_json_file_locations.py`

---

**Status**: ✅ Migration Complete  
**Date**: 2025-10-10  
**Files Moved**: 2  
**Files Updated**: 15+  
**Validation**: Active (pre-commit + CI)
