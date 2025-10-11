# Quick Reference: JSON File Locations

## TL;DR

✅ **Files are in the RIGHT canonical locations** (`/bundles/` and `/standards/`)  
✅ **All references use central path resolver** (`repo_paths.py`)  
✅ **No changes needed** unless migrating paths

## File Locations

```
/home/runner/work/SIN_CARRETA/SIN_CARRETA/
├── bundles/
│   └── decalogo-industrial.latest.clean.json  (210KB, 300 questions)
└── standards/
    └── dnp-standards.latest.clean.json        (79KB)
```

## How to Use

### ✅ CORRECT: Use the central path resolver
```python
from repo_paths import get_decalogo_path, get_dnp_path

decalogo_path = get_decalogo_path()
dnp_path = get_dnp_path()

# Or use the high-level loader
from decalogo_loader import get_decalogo_industrial, load_dnp_standards

industrial_data = get_decalogo_industrial()
dnp_data = load_dnp_standards()
```

### ❌ INCORRECT: Direct path construction
```python
# DON'T DO THIS - Will fail validation
from pathlib import Path
path = Path("decalogo-industrial.latest.clean.json")  # WRONG
path = Path(__file__).parent / "decalogo-industrial.json"  # WRONG
```

## Validate Everything Works

```bash
# Check canonical paths
python3 tools/check_canonical_paths.py

# Validate file locations
python3 validate_json_file_locations.py
```

Expected output:
```
✅ OK: All paths are canonical
✅ ALL VALIDATIONS PASSED
```

## Quick Checks

```bash
# Check files exist in canonical locations
ls -lh bundles/decalogo-industrial.latest.clean.json
ls -lh standards/dnp-standards.latest.clean.json

# Test path resolver
python3 -c "from repo_paths import get_decalogo_path; print(get_decalogo_path())"

# Test loader works
python3 -c "from decalogo_loader import get_decalogo_industrial; print(f'{len(get_decalogo_industrial()[\"questions\"])} questions loaded')"

# Verify JSON is valid
python3 -c "import json; from repo_paths import get_decalogo_path; json.load(open(get_decalogo_path())); print('✓ Valid JSON')"
```

## Key Files That Use These JSON Files

| File | How It Accesses | Status |
|------|-----------------|--------|
| `repo_paths.py` | Central path resolver | ✅ Source of truth |
| `decalogo_loader.py` | Uses repo_paths | ✅ Core module |
| `pdm_contra/config/decalogo.yaml` | Relative paths to canonical dirs | ✅ Working |
| `unified_evaluation_pipeline.py` | Uses repo_paths | ✅ Updated |
| Test files | Use repo_paths | ✅ Updated |

## If You Need to Move Files

1. Update `repo_paths.py` (BUNDLES_DIR, STANDARDS_DIR, paths)
2. Update `pdm_contra/config/decalogo.yaml` paths section
3. Run: `python3 tools/check_canonical_paths.py`
4. Run: `python3 validate_json_file_locations.py`
5. Update this documentation

**Note:** All other files automatically use the new paths via the central resolver.

## Documentation

- 📄 `JSON_FILE_LOCATIONS.md` - Comprehensive path documentation
- 📄 `repo_paths.py` - Central path resolver (source of truth)
- 📄 `tools/check_canonical_paths.py` - Path validation tool
- 📄 This file - Quick reference

## Troubleshooting

**Problem:** File not found error

**Solution:**
```bash
# Check files are in canonical locations
ls bundles/decalogo-industrial.latest.clean.json
ls standards/dnp-standards.latest.clean.json

# Test path resolver
python3 -c "from repo_paths import get_decalogo_path; print(get_decalogo_path())"
```

**Problem:** Module can't import files

**Solution:**
```python
# Use the central path resolver instead of direct paths
from repo_paths import get_decalogo_path, get_dnp_path
```

## Summary

✅ Files in canonical locations (`/bundles/` and `/standards/`)  
✅ All modules use central path resolver (`repo_paths.py`)  
✅ Validation scripts confirm everything works  
✅ Pre-commit hooks prevent regressions  
✅ CI validates on every push

**No action needed!** Everything is working as designed.
