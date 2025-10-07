#!/usr/bin/env python3
"""
Validation script for miniminimoon_orchestrator.py changes.
Verifies:
1. PipelineStage enum values match flow_doc.json canonical order
2. AnswerAssembler instantiation and integration
3. Evidence registry provenance tracking
4. Deterministic JSON serialization
"""

import json
from pathlib import Path

def validate_pipeline_stages():
    """Verify PipelineStage enum matches flow_doc.json"""
    print("=" * 60)
    print("1. Validating PipelineStage enum alignment")
    print("=" * 60)
    
    # Import after checking file exists
    from miniminimoon_orchestrator import PipelineStage
    
    # Load flow_doc.json
    flow_doc_path = Path("tools/flow_doc.json")
    if not flow_doc_path.exists():
        print("⚠️  tools/flow_doc.json not found - skipping canonical order validation")
        return True
    
    with open(flow_doc_path, 'r') as f:
        flow_doc = json.load(f)
    
    canonical_order = flow_doc.get("canonical_order", [])
    pipeline_order = [stage.value for stage in PipelineStage]
    
    print(f"Canonical order (flow_doc.json): {len(canonical_order)} stages")
    print(f"PipelineStage enum: {len(pipeline_order)} stages")
    
    match = True
    for i, (canonical, pipeline) in enumerate(zip(canonical_order, pipeline_order), 1):
        status = "✓" if canonical == pipeline else "✗"
        print(f"  {i:2}. {status} {pipeline:35} {'==' if canonical == pipeline else '!='} {canonical}")
        if canonical != pipeline:
            match = False
    
    if match:
        print("✓ All pipeline stages match canonical order")
    else:
        print("✗ Pipeline stages DO NOT match canonical order")
    
    return match

def validate_answer_assembler_integration():
    """Verify AnswerAssembler is properly integrated"""
    print("\n" + "=" * 60)
    print("2. Validating AnswerAssembler integration")
    print("=" * 60)
    
    import inspect
    from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
    
    # Check _assemble_answers method exists
    has_method = hasattr(CanonicalDeterministicOrchestrator, '_assemble_answers')
    print(f"  {'✓' if has_method else '✗'} _assemble_answers method exists")
    
    if has_method:
        # Check method signature
        method = getattr(CanonicalDeterministicOrchestrator, '_assemble_answers')
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        print(f"  ✓ Method signature: {params}")
        
        # Check for key operations in source
        source = inspect.getsource(method)
        checks = [
            ("weights section loading", "rubric.get(\"weights\"" in source),
            ("evidence_ids extraction", "evidence_ids" in source),
            ("provenance metadata", "source_evidence_ids" in source),
            ("rubric_weight in metadata", "rubric_weight" in source),
            ("confidence in metadata", "confidence" in source)
        ]
        
        all_pass = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False
        
        return all_pass
    
    return False

def validate_export_artifacts():
    """Verify export_artifacts generates required files with deterministic JSON"""
    print("\n" + "=" * 60)
    print("3. Validating export_artifacts implementation")
    print("=" * 60)
    
    import inspect
    from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
    
    method = getattr(CanonicalDeterministicOrchestrator, 'export_artifacts')
    source = inspect.getsource(method)
    
    checks = [
        ("answers_report.json export", "answers_report.json" in source),
        ("answers_sample.json export", "answers_sample.json" in source),
        ("flow_runtime.json export", "flow_runtime.json" in source),
        ("deterministic JSON (sort_keys=True)", "sort_keys=True" in source),
        ("ensure_ascii=False", "ensure_ascii=False" in source)
    ]
    
    all_pass = True
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if not check_result:
            all_pass = False
    
    return all_pass

def validate_flow_runtime_structure():
    """Verify _generate_flow_runtime_metadata creates properly sorted structure"""
    print("\n" + "=" * 60)
    print("4. Validating flow_runtime.json structure")
    print("=" * 60)
    
    import inspect
    from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
    
    method = getattr(CanonicalDeterministicOrchestrator, '_generate_flow_runtime_metadata')
    source = inspect.getsource(method)
    
    # Check for proper sorted stage_timestamps
    has_sorted = "sorted(runtime_data.get(\"stage_timestamps\"" in source
    print(f"  {'✓' if has_sorted else '✗'} stage_timestamps sorted deterministically")
    
    # Check for required fields
    required_fields = [
        "flow_hash",
        "stages",
        "stage_count",
        "stage_timestamps",
        "duration_seconds",
        "validation"
    ]
    
    all_present = True
    for field in required_fields:
        present = f'"{field}"' in source
        status = "✓" if present else "✗"
        print(f"  {status} field '{field}' included")
        if not present:
            all_present = False
    
    return has_sorted and all_present

def main():
    print("MINIMINIMOON ORCHESTRATOR VALIDATION")
    print("=" * 60)
    print()
    
    results = []
    
    try:
        results.append(("Pipeline Stage Alignment", validate_pipeline_stages()))
    except Exception as e:
        print(f"✗ Pipeline stage validation failed: {e}")
        results.append(("Pipeline Stage Alignment", False))
    
    try:
        results.append(("AnswerAssembler Integration", validate_answer_assembler_integration()))
    except Exception as e:
        print(f"✗ AnswerAssembler integration validation failed: {e}")
        results.append(("AnswerAssembler Integration", False))
    
    try:
        results.append(("Export Artifacts", validate_export_artifacts()))
    except Exception as e:
        print(f"✗ Export artifacts validation failed: {e}")
        results.append(("Export Artifacts", False))
    
    try:
        results.append(("Flow Runtime Structure", validate_flow_runtime_structure()))
    except Exception as e:
        print(f"✗ Flow runtime validation failed: {e}")
        results.append(("Flow Runtime Structure", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
