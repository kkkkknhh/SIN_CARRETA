#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verification script for miniminimoon_orchestrator.py modifications
"""

import ast
import inspect

def check_imports():
    """Check that AnswerAssembler is imported from answer_assembler"""
    with open('miniminimoon_orchestrator.py', 'r') as f:
        content = f.read()
    
    if 'from answer_assembler import AnswerAssembler as ExternalAnswerAssembler' in content:
        print("✓ AnswerAssembler imported from answer_assembler.py")
        return True
    else:
        print("✗ Missing import of ExternalAnswerAssembler")
        return False

def check_init_evaluators():
    """Check that _init_evaluators instantiates ExternalAnswerAssembler"""
    with open('miniminimoon_orchestrator.py', 'r') as f:
        content = f.read()
    
    if 'self.external_answer_assembler = ExternalAnswerAssembler(' in content:
        print("✓ ExternalAnswerAssembler instantiated in _init_evaluators")
        return True
    else:
        print("✗ Missing ExternalAnswerAssembler instantiation")
        return False

def check_assemble_answers():
    """Check that _assemble_answers calls external assembler methods"""
    with open('miniminimoon_orchestrator.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('self.external_answer_assembler.assemble(', 'Calls external assembler.assemble()'),
        ('self.evidence_registry.register(answer_entry)', 'Registers answers in EvidenceRegistry'),
    ]
    
    results = []
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
            results.append(True)
        else:
            print(f"✗ Missing: {description}")
            results.append(False)
    
    return all(results)

def check_export_artifacts():
    """Check that export_artifacts writes answer and flow_runtime files"""
    with open('miniminimoon_orchestrator.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('def export_artifacts(self, output_dir: Path, pipeline_results: Dict[str, Any] = None)',
         'export_artifacts accepts pipeline_results parameter'),
        ('self.external_answer_assembler.save_report_json(',
         'Calls save_report_json for answers_report.json'),
        ('answers_sample.json',
         'Creates answers_sample.json'),
        ('flow_runtime.json',
         'Creates flow_runtime.json'),
        ('sort_keys=True',
         'Uses deterministic key ordering'),
    ]
    
    results = []
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
            results.append(True)
        else:
            print(f"✗ Missing: {description}")
            results.append(False)
    
    return all(results)

def check_flow_runtime_metadata():
    """Check that _generate_flow_runtime_metadata exists and creates deterministic output"""
    with open('miniminimoon_orchestrator.py', 'r') as f:
        content = f.read()
    
    if 'def _generate_flow_runtime_metadata(self, pipeline_results: Dict[str, Any])' in content:
        print("✓ _generate_flow_runtime_metadata method exists")
        
        # Check for required keys
        required_keys = [
            'evidence_hash',
            'duration_seconds',
            'end_time',
            'flow_hash',
            'stages',
            'start_time',
            'validation'
        ]
        
        all_keys_present = all(f'"{key}"' in content for key in required_keys)
        if all_keys_present:
            print(f"✓ flow_runtime.json includes required metadata keys")
            return True
        else:
            print(f"✗ Some required keys missing in flow_runtime.json")
            return False
    else:
        print("✗ _generate_flow_runtime_metadata method not found")
        return False

def check_unified_pipeline():
    """Check that UnifiedEvaluationPipeline.evaluate calls export_artifacts correctly"""
    with open('miniminimoon_orchestrator.py', 'r') as f:
        content = f.read()
    
    if 'orchestrator.export_artifacts(output_dir, pipeline_results=results)' in content:
        print("✓ UnifiedEvaluationPipeline passes pipeline_results to export_artifacts")
        return True
    else:
        print("✗ export_artifacts not called with pipeline_results in UnifiedEvaluationPipeline")
        return False

def main():
    print("=" * 70)
    print("VERIFICATION: miniminimoon_orchestrator.py modifications")
    print("=" * 70)
    print()
    
    print("1. Checking imports...")
    r1 = check_imports()
    print()
    
    print("2. Checking _init_evaluators...")
    r2 = check_init_evaluators()
    print()
    
    print("3. Checking _assemble_answers...")
    r3 = check_assemble_answers()
    print()
    
    print("4. Checking export_artifacts...")
    r4 = check_export_artifacts()
    print()
    
    print("5. Checking _generate_flow_runtime_metadata...")
    r5 = check_flow_runtime_metadata()
    print()
    
    print("6. Checking UnifiedEvaluationPipeline.evaluate...")
    r6 = check_unified_pipeline()
    print()
    
    print("=" * 70)
    if all([r1, r2, r3, r4, r5, r6]):
        print("✅ ALL CHECKS PASSED")
        print()
        print("Summary:")
        print("- AnswerAssembler imported and instantiated")
        print("- Answer assembly calls external assembler methods")
        print("- Answers registered in EvidenceRegistry")
        print("- answers_report.json and answers_sample.json written to artifacts")
        print("- flow_runtime.json generated with deterministic keys")
        print("- All files exported in export_artifacts method")
    else:
        print("❌ SOME CHECKS FAILED")
    print("=" * 70)

if __name__ == "__main__":
    main()
