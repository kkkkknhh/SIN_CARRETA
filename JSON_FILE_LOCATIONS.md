# JSON File Locations and References

## Canonical File Locations

The following JSON files are the canonical sources of truth for the system:

### Canonical Directories
```
/home/runner/work/SIN_CARRETA/SIN_CARRETA/
├── bundles/
│   └── decalogo-industrial.latest.clean.json  (210,775 bytes, 300 questions)
└── standards/
    └── dnp-standards.latest.clean.json        (79,737 bytes)
```

**These files MUST remain in these canonical locations** (`/bundles/` and `/standards/`) to ensure all modules can reference them correctly through the central path resolver.

## Path Resolution Pattern

### Central Path Resolver (`repo_paths.py`)

All code MUST use the central path resolver to access these files:

```python
# Import the resolver
from repo_paths import get_decalogo_path, get_dnp_path

# Get canonical paths
decalogo_path = get_decalogo_path()  # Returns /bundles/decalogo-industrial.latest.clean.json
dnp_path = get_dnp_path()            # Returns /standards/dnp-standards.latest.clean.json

# Optional: Override with environment variables (filename must still be canonical)
import os
decalogo_path = get_decalogo_path(os.getenv("DECALOGO_PATH_OVERRIDE"))
dnp_path = get_dnp_path(os.getenv("DNP_PATH_OVERRIDE"))
```

**Direct path construction is forbidden.** All code must use the resolver functions.

### Legacy Pattern (DEPRECATED)

The old pattern of using `Path(__file__).parent / "decalogo-industrial.latest.clean.json"` has been replaced by the central resolver.

### Configuration-Based Loading (`pdm_contra/config/decalogo.yaml`)

The PDM configuration uses relative paths from the config directory:

```yaml
# pdm_contra/config/decalogo.yaml
paths:
  full: "../../bundles/decalogo-industrial.latest.clean.json"
  industrial: "../../bundles/decalogo-industrial.latest.clean.json"
  dnp: "../../standards/dnp-standards.latest.clean.json"
```

**Resolution:**
- Config directory: `pdm_contra/config/`
- `../../bundles/` navigates up two levels then into bundles/
- `../../standards/` navigates up two levels then into standards/

### Test Files

Test files should use the central resolver:

```python
# Recommended pattern
from repo_paths import get_decalogo_path, get_dnp_path

class TestDecalogoLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.industrial_path = get_decalogo_path()
        cls.dnp_path = get_dnp_path()
```

**Legacy patterns (DEPRECATED):** Direct path construction has been replaced by the central resolver.

### Pipeline and Orchestrator References

Pipelines and orchestrators should use the central resolver:

```python
# unified_evaluation_pipeline.py
from repo_paths import get_decalogo_path

def run_evaluation(self):
    decalogo_json = get_decalogo_path()
    with open(decalogo_json, 'r', encoding='utf-8') as f:
        decalogo_data = json.load(f)
```

## File Reference Audit

### Files That Load JSON Data (12 files, 33 references)

1. **Core Loaders:**
   - `decalogo_loader.py` - Primary loader with fallback mechanism
   - `pdm_contra/bridges/decatalogo_provider.py` - Provider using config
   - `pdm_contra/bridges/decalogo_loader_adapter.py` - Schema references

2. **Alignment and Processing:**
   - `pdm_contra/decalogo_alignment.py` - Generates versioned outputs
   - `unified_evaluation_pipeline.py` - Main evaluation pipeline

3. **Test Files:**
   - `test_decalogo_loader.py`
   - `test_decalogo_alignment_fix.py`
   - `test_dnp_standards_json.py`
   - `test_miniminimoon_orchestrator_parallel.py`
   - `verify_decalogo_alignment.py`

4. **Test Suites (pdm_contra):**
   - `tests/test_loader_compat.py`
   - `tests/test_crosswalk_isomorphism.py`
   - `tests/test_schema_validation.py`

## Validation

Run the validation script to verify all paths resolve correctly:

```bash
python3 validate_json_file_locations.py
```

This script validates:
- ✓ Files exist in repository root
- ✓ JSON structure is valid
- ✓ Module imports work correctly
- ✓ Config paths resolve correctly
- ✓ Test paths are correct

## Orchestrator Integration

The main orchestrator does not directly reference these JSON files. Instead, it uses the loader modules:

```python
# In any orchestrator or pipeline
from decalogo_loader import get_decalogo_industrial, load_dnp_standards

# Load the canonical data
industrial_data = get_decalogo_industrial()  # Loads from /bundles/
dnp_data = load_dnp_standards()              # Loads from /standards/
```

**Benefits:**
1. **Centralized loading** - Single source of truth via `repo_paths.py`
2. **Fallback mechanism** - In-memory templates if files unavailable
3. **Thread-safe caching** - Efficient repeated access
4. **Path abstraction** - Modules don't need to know absolute paths
5. **Validation guards** - Runtime checks ensure canonical filenames

## Version Management

The system maintains two types of files:

### Latest (Working Copies)
- `/bundles/decalogo-industrial.latest.clean.json`
- `/standards/dnp-standards.latest.clean.json`

These are in the **canonical directories** and are the canonical working versions.

### Versioned (Generated Outputs)
- `pdm_contra/config/out/decalogo-industrial.v1.0.0.clean.json`
- `pdm_contra/config/out/dnp-standards.v1.0.0.clean.json`

These are generated by `pdm_contra/decalogo_alignment.py` and represent frozen versions.

## Migration Notes

**All path resolution now goes through the central resolver (`repo_paths.py`)**

If the files need to be moved in the future:

1. **Update `repo_paths.py`:**
   - Change `BUNDLES_DIR` and `STANDARDS_DIR` constants
   - Change `DECALOGO_PATH` and `DNP_PATH` assignments

2. **Update `pdm_contra/config/decalogo.yaml`:**
   - Adjust relative paths in `paths` section

3. **No changes needed in other files** - they use the central resolver

4. **Run validation:**
   ```bash
   python3 tools/check_canonical_paths.py
   python3 validate_json_file_locations.py
   ```

## Best Practices

1. **Always use the central path resolver:**
   ```python
   from repo_paths import get_decalogo_path, get_dnp_path
   ```
   
   Or use the high-level loader:
   ```python
   from decalogo_loader import get_decalogo_industrial, load_dnp_standards
   ```

2. **Never hardcode paths** - Use the resolver functions

3. **Run validation after changes** - Ensure all references still work

4. **Keep files in canonical locations** - `/bundles/` and `/standards/`

5. **Use pre-commit hooks** - Automatically validate paths before commits

## Validation

### Pre-commit Hook

The repository includes a pre-commit hook that validates canonical paths:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run check-canonical-paths --all-files
```

### CI Validation

GitHub Actions automatically validates paths on every push and PR via the `canonical-integration.yml` workflow.

## Troubleshooting

### Issue: "File not found" errors

**Check:**
1. Are the files in `/bundles/` and `/standards/`?
2. Are you using the central resolver (`repo_paths.py`)?
3. Is the filename exactly correct (case-sensitive)?

**Solution:**
```python
from repo_paths import get_decalogo_path, get_dnp_path
print(f"Decalogo path: {get_decalogo_path()}")
print(f"DNP path: {get_dnp_path()}")
```

### Issue: "Non-canonical reference" errors

**Check:**
2. Are there syntax errors?

**Solution:**
```bash
python3 -c "import json; json.load(open('decalogo-industrial.latest.clean.json'))"
```

### Issue: Module can't find files after moving them

**Fix:**
1. Update loader paths
2. Update config file
3. Run validation
4. Update documentation

## Summary

✅ **Current Status:** All files are in the correct location (repository root)

✅ **All References:** Working correctly and validated

✅ **Orchestrator Integration:** Uses loader modules (no direct file references)

✅ **Path Resolution:** All patterns resolve correctly to repository root

The system is correctly configured and all invocations match the available version in the current path.
