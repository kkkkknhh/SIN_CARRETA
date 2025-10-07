#!/usr/bin/env python3
"""
Structural validation for miniminimoon_orchestrator.py changes.
Validates code structure without importing full module.
"""

import json
import re
from pathlib import Path


def validate_pipeline_stages_in_source():
    """Validate PipelineStage enum matches flow_doc.json by parsing source"""
    print("=" * 70)
    print("1. Validating PipelineStage Enum Alignment")
    print("=" * 70)
    
    # Read flow_doc.json
    flow_doc_path = Path("tools/flow_doc.json")
    if not flow_doc_path.exists():
        print("⚠️  tools/flow_doc.json not found")
        return False
    
    with open(flow_doc_path, 'r') as f:
        flow_doc = json.load(f)
    canonical_order = flow_doc.get("canonical_order", [])
    
    # Read miniminimoon_orchestrator.py
    with open("miniminimoon_orchestrator.py", 'r') as f:
        source = f.read()
    
    # Extract PipelineStage enum values
    enum_pattern = r'class PipelineStage\(Enum\):(.*?)(?=\n\n|\nclass|\n@dataclass)'
    enum_match = re.search(enum_pattern, source, re.DOTALL)
    
    if not enum_match:
        print("✗ Could not find PipelineStage enum definition")
        return False
    
    enum_text = enum_match.group(1)
    stage_pattern = r'(\w+)\s*=\s*"([^"]+)"'
    stages = re.findall(stage_pattern, enum_text)
    
    pipeline_order = [value for _, value in stages]
    
    print(f"Canonical order: {len(canonical_order)} stages")
    print(f"Pipeline order:  {len(pipeline_order)} stages")
    print()
    
    match = True
    for i, (canonical, pipeline) in enumerate(zip(canonical_order, pipeline_order), 1):
        status = "✓" if canonical == pipeline else "✗"
        print(f"  {i:2}. {status} {pipeline:35} {'==' if canonical == pipeline else '!='} {canonical}")
        if canonical != pipeline:
            match = False
    
    if match:
        print("\n✓ All pipeline stages match canonical order")
    else:
        print("\n✗ Pipeline stages DO NOT match canonical order")
    
    return match


def validate_assemble_answers_implementation():
    """Validate _assemble_answers method implementation"""
    print("\n" + "=" * 70)
    print("2. Validating _assemble_answers Implementation")
    print("=" * 70)
    
    with open("miniminimoon_orchestrator.py", 'r') as f:
        source = f.read()
    
    # Find _assemble_answers method
    method_pattern = r'def _assemble_answers\(self.*?\):(.*?)(?=\n    def |\nclass |\Z)'
    method_match = re.search(method_pattern, source, re.DOTALL)
    
    if not method_match:
        print("✗ Could not find _assemble_answers method")
        return False
    
    method_source = method_match.group(0)
    
    checks = [
        ("Method definition exists", "def _assemble_answers" in source),
        ("Loads RUBRIC_SCORING.json", 'RUBRIC_SCORING.json' in method_source),
        ("Loads weights section", 'rubric.get("weights"' in method_source or 'rubric.get(\'weights\'' in method_source),
        ("Extracts evidence_ids", "evidence_ids" in method_source),
        ("Creates source_evidence_ids metadata", "source_evidence_ids" in method_source),
        ("Includes rubric_weight in metadata", "rubric_weight" in method_source),
        ("Includes confidence in metadata", "confidence" in method_source),
        ("Includes rationale in metadata", "rationale" in method_source),
        ("Includes scoring_modality in metadata", "scoring_modality" in method_source),
        ("Registers answers to evidence registry", "self.evidence_registry.register" in method_source)
    ]
    
    all_pass = True
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if not check_result:
            all_pass = False
    
    if all_pass:
        print("\n✓ _assemble_answers implementation complete")
    else:
        print("\n✗ _assemble_answers missing required components")
    
    return all_pass


def validate_export_artifacts_implementation():
    """Validate export_artifacts method implementation"""
    print("\n" + "=" * 70)
    print("3. Validating export_artifacts Implementation")
    print("=" * 70)
    
    with open("miniminimoon_orchestrator.py", 'r') as f:
        source = f.read()
    
    # Find export_artifacts method
    method_pattern = r'def export_artifacts\(self.*?\):(.*?)(?=\n    def |\nclass |\Z)'
    method_match = re.search(method_pattern, source, re.DOTALL)
    
    if not method_match:
        print("✗ Could not find export_artifacts method")
        return False
    
    method_source = method_match.group(0)
    
    checks = [
        ("Exports answers_report.json", "answers_report.json" in method_source),
        ("Exports answers_sample.json", "answers_sample.json" in method_source),
        ("Exports flow_runtime.json", "flow_runtime.json" in method_source),
        ("Uses sort_keys=True", "sort_keys=True" in method_source),
        ("Uses ensure_ascii=False", "ensure_ascii=False" in method_source),
        ("Sample has max 10 questions", "[:10]" in method_source),
        ("Calls _generate_flow_runtime_metadata", "_generate_flow_runtime_metadata" in method_source)
    ]
    
    all_pass = True
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if not check_result:
            all_pass = False
    
    if all_pass:
        print("\n✓ export_artifacts implementation complete")
    else:
        print("\n✗ export_artifacts missing required components")
    
    return all_pass


def validate_flow_runtime_metadata():
    """Validate _generate_flow_runtime_metadata implementation"""
    print("\n" + "=" * 70)
    print("4. Validating _generate_flow_runtime_metadata Implementation")
    print("=" * 70)
    
    with open("miniminimoon_orchestrator.py", 'r') as f:
        source = f.read()
    
    # Find method
    method_pattern = r'def _generate_flow_runtime_metadata\(self.*?\):(.*?)(?=\n    def |\nclass |\Z)'
    method_match = re.search(method_pattern, source, re.DOTALL)
    
    if not method_match:
        print("✗ Could not find _generate_flow_runtime_metadata method")
        return False
    
    method_source = method_match.group(0)
    
    required_fields = [
        "flow_hash",
        "stages",
        "stage_count",
        "stage_timestamps",
        "duration_seconds",
        "validation",
        "evidence_hash",
        "orchestrator_version",
        "plan_path"
    ]
    
    checks = [
        ("Method definition exists", True),
        ("Sorts stage_timestamps", "sorted(runtime_data.get" in method_source or "dict(sorted" in method_source)
    ]
    
    for field in required_fields:
        checks.append((f"Includes field '{field}'", f'"{field}"' in method_source))
    
    all_pass = True
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if not check_result:
            all_pass = False
    
    if all_pass:
        print("\n✓ _generate_flow_runtime_metadata implementation complete")
    else:
        print("\n✗ _generate_flow_runtime_metadata missing required components")
    
    return all_pass


def validate_python_syntax():
    """Validate Python syntax using py_compile"""
    print("\n" + "=" * 70)
    print("5. Validating Python Syntax")
    print("=" * 70)
    
    import py_compile
    import tempfile
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pyc', delete=True) as tmp:
            py_compile.compile('miniminimoon_orchestrator.py', cfile=tmp.name, doraise=True)
        print("  ✓ miniminimoon_orchestrator.py syntax valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def main():
    print("MINIMINIMOON ORCHESTRATOR STRUCTURAL VALIDATION")
    print("=" * 70)
    print()
    
    results = []
    
    try:
        results.append(("Pipeline Stage Alignment", validate_pipeline_stages_in_source()))
    except Exception as e:
        print(f"✗ Pipeline stage validation failed: {e}")
        results.append(("Pipeline Stage Alignment", False))
    
    try:
        results.append(("_assemble_answers Implementation", validate_assemble_answers_implementation()))
    except Exception as e:
        print(f"✗ _assemble_answers validation failed: {e}")
        results.append(("_assemble_answers Implementation", False))
    
    try:
        results.append(("export_artifacts Implementation", validate_export_artifacts_implementation()))
    except Exception as e:
        print(f"✗ export_artifacts validation failed: {e}")
        results.append(("export_artifacts Implementation", False))
    
    try:
        results.append(("_generate_flow_runtime_metadata Implementation", validate_flow_runtime_metadata()))
    except Exception as e:
        print(f"✗ flow_runtime_metadata validation failed: {e}")
        results.append(("_generate_flow_runtime_metadata Implementation", False))
    
    try:
        results.append(("Python Syntax", validate_python_syntax()))
    except Exception as e:
        print(f"✗ Syntax validation failed: {e}")
        results.append(("Python Syntax", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        return 0
    else:
        failed_count = sum(1 for _, passed in results if not passed)
        print(f"✗ {failed_count} VALIDATION(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
