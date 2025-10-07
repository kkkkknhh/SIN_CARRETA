# Test Suite Systematic Update Plan

## Critical Issues Found (Exit Code 1)

### Files with Deprecated Orchestrator References (3 files)
1. **test_audit_comprehensive.py** - ✅ FIXED (self-reference check added)
2. **test_coverage_analyzer.py** - String reference only, no actual import
3. **test_suite_audit_runner.py** - String reference only, no actual import

## Warning Issues Found

### Missing RUBRIC_SCORING.json Validation (5 files)
1. **test_e2e_unified_pipeline_mock.py** - ✅ FIXED (complete rewrite with validation)
2. **test_enhanced_orchestrator.py** - Needs: rubric validation checks
3. **test_orchestrator_instrumentation.py** - Needs: rubric validation checks
4. **test_orchestrator_modifications.py** - Needs: rubric validation checks
5. **test_orchestrator_syntax.py** - Needs: rubric validation checks

### Missing AnswerAssembler Component (1 file)
1. **test_e2e_unified_pipeline_mock.py** - ✅ FIXED

### Missing system_validators Pre/Post Execution (6 files)
1. **test_critical_flows.py** - Needs: pre/post execution gate tests
2. **test_e2e_unified_pipeline_mock.py** - ✅ FIXED
3. **test_enhanced_orchestrator.py** - Needs: validator integration
4. **test_orchestrator_instrumentation.py** - Needs: validator integration
5. **test_orchestrator_modifications.py** - Needs: validator integration
6. **test_orchestrator_syntax.py** - Needs: validator integration

### Missing Determinism Checks (7 files)
1. **test_critical_flows.py** - Needs: hash/frozen/evidence_id checks
2. **test_deterministic_seeding.py** - Has determinism tests but missing frozen/evidence_id
3. **test_e2e_unified_pipeline_mock.py** - ✅ FIXED
4. **test_enhanced_orchestrator.py** - Needs: full determinism suite
5. **test_orchestrator_instrumentation.py** - Needs: full determinism suite
6. **test_orchestrator_modifications.py** - Needs: full determinism suite
7. **test_orchestrator_syntax.py** - Needs: full determinism suite

### Missing EvidenceRegistry Usage (2 files)
1. **test_evidence_quality.py** - Needs: EvidenceRegistry integration
2. **test_zero_evidence.py** - Needs: EvidenceRegistry integration

## Update Strategy

### Phase 1: Fix Critical String References (Non-functional)
- test_coverage_analyzer.py - Update documentation strings
- test_suite_audit_runner.py - Update documentation strings

### Phase 2: Add Missing Test Coverage
For each orchestrator test file, add:
```python
from system_validators import SystemHealthValidator
from evidence_registry import EvidenceRegistry
import json

def test_rubric_scoring_validation(self):
    \"\"\"Validate RUBRIC_SCORING.json as single source of truth\"\"\"
    rubric_path = Path("RUBRIC_SCORING.json")
    self.assertTrue(rubric_path.exists())
    with open(rubric_path) as f:
        rubric = json.load(f)
    self.assertIn("weights", rubric)
    self.assertIn("questions", rubric)

def test_system_validators_pre_execution(self):
    \"\"\"Test pre-execution validation gates\"\"\"
    validator = SystemHealthValidator(".")
    result = validator.validate_pre_execution()
    self.assertTrue(result.get("ok"))

def test_deterministic_hash_reproducibility(self):
    \"\"\"Test deterministic hash consistency\"\"\"
    registry1 = EvidenceRegistry()
    registry1.register("comp", "type", {"val": 42}, 0.8, ["D1-Q1"])
    hash1 = registry1.deterministic_hash()
    
    registry2 = EvidenceRegistry()
    registry2.register("comp", "type", {"val": 42}, 0.8, ["D1-Q1"])
    hash2 = registry2.deterministic_hash()
    
    self.assertEqual(hash1, hash2)
```

### Phase 3: Evidence Registry Integration
- test_evidence_quality.py - Add EvidenceRegistry-based quality checks
- test_zero_evidence.py - Use EvidenceRegistry for zero-evidence scenarios

## Execution Order
1. ✅ Fix test_audit_comprehensive.py (DONE)
2. ✅ Fix test_e2e_unified_pipeline_mock.py (DONE)
3. Fix documentation strings in coverage_analyzer and audit_runner
4. Add missing tests to orchestrator files
5. Integrate EvidenceRegistry in evidence quality tests
6. Run full test suite validation

## Success Criteria
- Zero deprecated orchestrator imports
- All orchestrator tests validate RUBRIC_SCORING.json
- All orchestrator tests include system_validators
- All orchestrator tests check determinism
- All evidence tests use EvidenceRegistry
- test_suite_audit_runner.py shows zero critical issues
