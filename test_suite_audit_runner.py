#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite Audit Runner - Comprehensive validation of all test files
=====================================================================

Systematically audits every test file to verify alignment with:
1. miniminimoon_orchestrator (not deprecated components)
2. RUBRIC_SCORING.json as single source of truth for weights
3. AnswerAssembler and unified_evaluation_pipeline components
4. system_validators pre/post execution gates
5. Correct artifacts directory structure (answers_report.json, flow_runtime.json)
6. Deterministic behavior (frozen configs, consistent evidence_ids, reproducible hashes)
7. EvidenceRegistry deterministic hashing
8. tools/rubric_check.py validation

NOTE: This auditor checks for deprecated imports but does not itself import deprecated code.
"""

import pathlib
import json
from collections import defaultdict


def audit_test_files():
    """Audit all test files for alignment issues"""
    repo_root = pathlib.Path(".")
    test_files = sorted([f for f in repo_root.glob("test_*.py") if "venv" not in str(f)])
    
    print(f"🔍 Auditing {len(test_files)} test files...\n")
    
    issues = defaultdict(list)
    stats = {
        "total_files": len(test_files),
        "deprecated_orchestrator": 0,
        "missing_rubric_validation": 0,
        "missing_answer_assembler": 0,
        "missing_system_validators": 0,
        "missing_artifacts_checks": 0,
        "missing_determinism": 0,
        "missing_evidence_registry": 0,
        "missing_rubric_check_tool": 0,
    }
    
    for test_file in test_files:
        try:
            content = test_file.read_text(encoding='utf-8', errors='ignore')
            file_issues = []
            
            # Check 1: Deprecated orchestrator (skip self-referencing)
            if test_file.name not in ['test_audit_comprehensive.py', 'test_coverage_analyzer.py', 'test_suite_audit_runner.py']:
                if 'from decalogo_pipeline_orchestrator import' in content:
                    file_issues.append("❌ Uses deprecated decalogo_pipeline_orchestrator import")
                    stats["deprecated_orchestrator"] += 1
            
            # Check 2: RUBRIC_SCORING.json validation (for relevant tests)
            if any(x in test_file.stem for x in ['answer_assembler', 'evaluation', 'pipeline', 
                                                   'unified', 'orchestrator', 'rubric', 'e2e']):
                if 'RUBRIC_SCORING' not in content and 'rubric_scoring' not in content:
                    file_issues.append("⚠️  Missing RUBRIC_SCORING.json validation")
                    stats["missing_rubric_validation"] += 1
            
            # Check 3: AnswerAssembler (for pipeline tests)
            if any(x in test_file.stem for x in ['evaluation', 'pipeline', 'unified', 'e2e', 'answer']):
                if 'AnswerAssembler' not in content and 'answer_assembler' not in content:
                    if 'answer' not in test_file.stem or 'assembler' in test_file.stem:
                        file_issues.append("⚠️  Missing AnswerAssembler component")
                        stats["missing_answer_assembler"] += 1
            
            # Check 4: system_validators (for orchestrator tests)
            if any(x in test_file.stem for x in ['orchestrator', 'pipeline', 'e2e', 'unified', 
                                                   'critical', 'validator']):
                if 'system_validators' not in content and 'SystemHealthValidator' not in content:
                    file_issues.append("⚠️  Missing system_validators pre/post execution")
                    stats["missing_system_validators"] += 1
            
            # Check 5: Artifact structure (for e2e tests)
            if any(x in test_file.stem for x in ['e2e', 'pipeline', 'orchestrator', 'unified']):
                if 'artifact' in content.lower():
                    missing = []
                    if 'answers_report.json' not in content:
                        missing.append('answers_report.json')
                    if 'flow_runtime.json' not in content:
                        missing.append('flow_runtime.json')
                    if missing:
                        file_issues.append(f"⚠️  Missing artifact checks: {', '.join(missing)}")
                        stats["missing_artifacts_checks"] += 1
            
            # Check 6: Determinism (for orchestrator and pipeline tests)
            if any(x in test_file.stem for x in ['orchestrator', 'pipeline', 'e2e', 'deterministic',
                                                   'unified', 'critical', 'reproducibility']):
                missing_det = []
                if 'deterministic' not in content.lower() and 'hash' not in content.lower():
                    missing_det.append('deterministic hash')
                if 'frozen' not in content and 'freeze' not in content:
                    missing_det.append('frozen config')
                if 'evidence_id' not in content:
                    missing_det.append('evidence_ids')
                
                if missing_det and len(missing_det) >= 2:
                    file_issues.append(f"⚠️  Missing determinism: {', '.join(missing_det)}")
                    stats["missing_determinism"] += 1
            
            # Check 7: EvidenceRegistry (for evidence tests)
            if 'evidence' in test_file.stem:
                if 'EvidenceRegistry' not in content and 'evidence_registry' not in content:
                    file_issues.append("⚠️  Missing EvidenceRegistry usage")
                    stats["missing_evidence_registry"] += 1
            
            # Check 8: tools/rubric_check.py (for rubric validator tests)
            if 'rubric' in test_file.stem and 'check' in test_file.stem:
                if 'rubric_check' not in content:
                    file_issues.append("⚠️  Missing tools/rubric_check.py integration")
                    stats["missing_rubric_check_tool"] += 1
            
            if file_issues:
                issues[test_file.name] = file_issues
        
        except Exception as e:
            issues[test_file.name] = [f"❌ Parse error: {e}"]
    
    # Print results
    print("=" * 80)
    print("TEST SUITE AUDIT RESULTS")
    print("=" * 80)
    print("\n📊 STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("ISSUES BY FILE:")
    print("=" * 80)
    
    if not issues:
        print("\n✅ No issues found! All tests are aligned.\n")
    else:
        for filename, file_issues in sorted(issues.items()):
            print(f"\n📄 {filename}")
            for issue in file_issues:
                print(f"    {issue}")
    
    print("\n" + "=" * 80)
    
    # Save report
    report_path = pathlib.Path("TEST_SUITE_AUDIT_RESULTS.md")
    with open(report_path, 'w') as f:
        f.write("# Test Suite Audit Results\n\n")
        f.write("## Statistics\n\n")
        for key, value in stats.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n## Issues by File\n\n")
        if not issues:
            f.write("✅ No issues found!\n")
        else:
            for filename, file_issues in sorted(issues.items()):
                f.write(f"\n### {filename}\n\n")
                for issue in file_issues:
                    f.write(f"- {issue}\n")
    
    print(f"\n📝 Report saved to: {report_path}\n")
    
    return issues, stats


if __name__ == "__main__":
    issues, stats = audit_test_files()
    
    # Determine exit code
    critical_issues = stats["deprecated_orchestrator"]
    if critical_issues > 0:
        print(f"❌ Found {critical_issues} critical issue(s)")
        exit(1)
    elif len(issues) > 0:
        print(f"⚠️  Found {len(issues)} file(s) with warnings")
        exit(0)
    else:
        print("✅ All tests pass audit")
        exit(0)
