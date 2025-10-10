# Quick Reference: JSON File Locations

## TL;DR

✅ **Files are in the RIGHT location** (repository root)
✅ **All references work correctly**
✅ **No changes needed**

## File Locations

```
/home/runner/work/SIN_CARRETA/SIN_CARRETA/
├── decalogo-industrial.latest.clean.json  (210KB, 300 questions)
└── dnp-standards.latest.clean.json        (79KB)
```

## How to Use

### ✅ CORRECT: Use the loader module
```python
from decalogo_loader import get_decalogo_industrial, load_dnp_standards

industrial_data = get_decalogo_industrial()
dnp_data = load_dnp_standards()
```

### ⚠️ WORKS BUT NOT RECOMMENDED: Direct loading
```python
from pathlib import Path
import json

# This works but doesn't benefit from caching/fallback
with open("decalogo-industrial.latest.clean.json", 'r') as f:
    data = json.load(f)
```

## Validate Everything Works

```bash
python3 validate_json_file_locations.py
```

Expected output:
```
✅ ALL VALIDATIONS PASSED
```

## Quick Checks

```bash
# Check files exist
ls -lh decalogo-industrial.latest.clean.json dnp-standards.latest.clean.json

# Test loader works
python3 -c "from decalogo_loader import get_decalogo_industrial; print(f'{len(get_decalogo_industrial()[\"questions\"])} questions loaded')"

# Verify JSON is valid
python3 -c "import json; json.load(open('decalogo-industrial.latest.clean.json')); print('✓ Valid JSON')"
```

## Key Files That Use These JSON Files

| File | How It Accesses | Status |
|------|-----------------|--------|
| `decalogo_loader.py` | Primary loader (Path(__file__).parent) | ✅ Core module |
| `pdm_contra/config/decalogo.yaml` | Config with relative paths | ✅ Working |
| `pdm_contra/bridges/decatalogo_provider.py` | Via config | ✅ Working |
| `unified_evaluation_pipeline.py` | Direct Path() | ⚠️ Could use loader |
| Test files | Various patterns | ✅ Appropriate |

## If You Need to Move Files

1. Update `decalogo_loader.py` lines 31 and 224
2. Update `pdm_contra/config/decalogo.yaml` paths section
3. Run: `python3 validate_json_file_locations.py`
4. Update this documentation

## Documentation

- 📄 `JSON_FILE_LOCATIONS.md` - Comprehensive path documentation
- 📄 `ORCHESTRATOR_JSON_AUDIT.md` - Orchestrator file analysis
- 📄 This file - Quick reference

## Troubleshooting

**Problem:** File not found error

**Solution:**
```bash
# Check current directory
pwd

# Should be repository root
# /home/runner/work/SIN_CARRETA/SIN_CARRETA

# If not, cd to repo root
cd /home/runner/work/SIN_CARRETA/SIN_CARRETA
```

**Problem:** Module can't import files

**Solution:**
```python
# Use the loader module instead of direct paths
from decalogo_loader import get_decalogo_industrial
```

## Summary

✅ Files in correct location
✅ All 12 modules can access them
✅ All 33 references resolve correctly
✅ Validation scripts confirm everything works

**No action needed!** Everything is working as designed.
