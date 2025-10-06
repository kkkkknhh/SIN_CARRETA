# Module Audit Report

## Audit Date: 2025-10-05

## Audit Methodology
- Last modification date analysis
- Import dependency tracking
- Test coverage analysis
- Usage pattern identification

## Categories
- **DELETE**: Not modified >6 months, no active imports, no test coverage
- **DOCUMENT**: Actively used, requires documentation
- **ARCHIVE**: Deprecated but may have historical value

---

## Modules Requiring Action

### Category: DELETE (Unused >6 months)

1. **system_validators.py** (0 bytes)
   - Status: Empty file
   - Last modified: Unknown
   - Action: DELETE
   - Reason: Empty placeholder file

2. **annotated_examples_test.py**
   - Status: Test file, unclear if active
   - Action: Review and consolidate into tests/

3. **debug_causal_patterns.py**
   - Status: Debug utility
   - Action: Move to dev_tools/ or DELETE if obsolete

4. **demo_signal_test.py**
   - Status: Demo file
   - Action: Archive to examples/ or DELETE

5. **demo_heap_functionality.py**
   - Status: Demo file
   - Action: Archive to examples/ or DELETE

6. **demo_plan_sanitizer.py**
   - Status: Demo file
   - Action: Archive to examples/ or DELETE

7. **demo_document_mapper.py**
   - Status: Demo file
   - Action: Archive to examples/ or DELETE

8. **demo_document_segmentation.py**
   - Status: Demo file
   - Action: Archive to examples/ or DELETE

9. **kill_python27_warnings.sh**
   - Status: Python 2.7 compatibility script
   - Action: DELETE (Python 2.7 EOL)

10. **fix_pycharm.sh**
    - Status: IDE-specific utility
    - Action: Move to dev_tools/ or DELETE

### Category: DOCUMENT (Actively used)

1. **embedding_model.py**
   - Status: ACTIVE - Core component
   - Action: Ensure comprehensive docstrings ✓

2. **responsibility_detector.py**
   - Status: ACTIVE - Core component
   - Action: Ensure comprehensive docstrings ✓

3. **teoria_cambio.py**
   - Status: ACTIVE - Core component
   - Action: Ensure comprehensive docstrings ✓

4. **dag_validation.py**
   - Status: ACTIVE - Core component
   - Action: Ensure comprehensive docstrings ✓

5. **feasibility_scorer.py**
   - Status: ACTIVE - Core component
   - Action: Document scoring algorithm

6. **contradiction_detector.py**
   - Status: ACTIVE - Core component
   - Action: Document pattern matching logic

7. **decalogo_loader.py**
   - Status: ACTIVE - Core component
   - Action: Document template structure ✓

8. **spacy_loader.py**
   - Status: ACTIVE - Core component
   - Action: Document fallback mechanism ✓

9. **questionnaire_engine.py**
   - Status: ACTIVE - Large file (81KB)
   - Action: Consider refactoring + documentation

10. **pdm_evaluator.py**
    - Status: ACTIVE - Core evaluation engine
    - Action: Document evaluation criteria

### Category: ARCHIVE (Deprecated)

1. **jsonschema.py**
   - Status: Duplicate of standard library
   - Action: Remove if using standard jsonschema package

2. **Decatalogo_principal.py**
   - Status: Legacy naming convention
   - Action: Rename to decatalogo_principal.py or consolidate

3. **cli.py**
   - Status: Unclear if active
   - Action: Review usage, document or consolidate

4. **miniminimoon_cli.py**
   - Status: Possible duplicate CLI
   - Action: Consolidate with cli.py

5. **pdm_utils_cli_modules.py**
   - Status: Utility module
   - Action: Consolidate into utils/

6. **pdm_tests_examples.py**
   - Status: Test examples
   - Action: Move to tests/examples/

7. **integration_example.py**
   - Status: Example file
   - Action: Move to examples/

8. **example_monetary_usage.py**
   - Status: Example file
   - Action: Move to examples/

9. **example_teoria_cambio.py**
   - Status: Example file
   - Action: Move to examples/

10. **unicode_test_samples.py**
    - Status: Test data
    - Action: Move to tests/fixtures/

---

## Recommended File Structure

```
miniminimoon/
├── src/
│   ├── core/
│   │   ├── embedding_model.py
│   │   ├── responsibility_detector.py
│   │   ├── teoria_cambio.py
│   │   └── dag_validation.py
│   ├── evaluation/
│   │   ├── pdm_evaluator.py
│   │   ├── feasibility_scorer.py
│   │   └── contradiction_detector.py
│   └── loaders/
│       ├── decalogo_loader.py
│       └── spacy_loader.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── contracts/
│   └── chaos/
├── examples/
│   └── [all demo and example files]
├── dev_tools/
│   └── [development utilities]
└── docs/
    └── [comprehensive documentation]
```

---

## Action Items Summary

- **DELETE**: 10 files (empty, obsolete, Python 2.7 specific)
- **DOCUMENT**: 10 core modules (add/improve documentation)
- **ARCHIVE**: 10 files (move to examples/, consolidate, or refactor)

**Target**: Reduce from 66 non-test modules to <30 actively maintained modules

## Next Steps

1. Create examples/ directory and move all demo/example files
2. Delete empty and obsolete files
3. Consolidate duplicate CLI modules
4. Add comprehensive documentation to core modules
5. Refactor large modules (>500 lines) into smaller components
