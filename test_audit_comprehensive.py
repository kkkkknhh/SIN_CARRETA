#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite Audit
===============================

Systematically validates all test files for alignment with:
1. Unified miniminimoon_orchestrator (no deprecated decalogo_pipeline_orchestrator)
2. RUBRIC_SCORING.json as single source of truth
3. AnswerAssembler and unified_evaluation_pipeline integration
4. system_validators pre/post execution gates
5. Correct artifacts directory structure (answers_report.json, flow_runtime.json)
6. Deterministic behavior (frozen configs, consistent evidence_ids, reproducible hashes)
7. EvidenceRegistry deterministic hashing
8. tools/rubric_check.py validation
"""

import ast
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class TestAuditor:
    """Auditor for test file alignment with current architecture"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = pathlib.Path(repo_root).resolve()
        self.issues: Dict[str, List[str]] = defaultdict(list)
        self.stats = {
            "total_test_files": 0,
            "deprecated_references": 0,
            "missing_rubric_validation": 0,
            "missing_answer_assembler": 0,
            "missing_unified_pipeline": 0,
            "missing_system_validators": 0,
            "missing_artifact_checks": 0,
            "missing_determinism_checks": 0,
            "missing_evidence_registry": 0,
            "missing_rubric_check_tool": 0,
        }
    
    def find_test_files(self) -> List[pathlib.Path]:
        """Find all test files in the repository"""
        test_files = []
        
        # Find test_*.py files
        for pattern in ["test_*.py", "tests/**/*.py"]:
            test_files.extend(self.repo_root.glob(pattern))
        
        # Filter out venv
        test_files = [f for f in test_files if "venv" not in str(f)]
        return sorted(set(test_files))
    
    def check_deprecated_imports(self, file_path: pathlib.Path) -> List[str]:
        """Check for deprecated references (now only checks others, not self)"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Skip self-referencing check
        if file_path.name == 'test_audit_comprehensive.py':
            return issues
        
        # Check for deprecated decalogo_pipeline_orchestrator in other files
        if 'from decalogo_pipeline_orchestrator import' in content:
            issues.append(f"Uses deprecated decalogo_pipeline_orchestrator import")
            self.stats["deprecated_references"] += 1
        
        return issues
    
    def check_rubric_validation(self, file_path: pathlib.Path) -> List[str]:
        """Check if test validates against RUBRIC_SCORING.json"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Skip non-evaluator tests
        if any(x in file_path.stem for x in ['model', 'detector', 'segmenter', 'loader', 
                                               'embedding', 'spacy', 'deployment', 'canary',
                                               'monitoring', 'instrumentation']):
            return issues
        
        # Check for rubric validation in evaluator/assembler/pipeline tests
        if any(x in file_path.stem for x in ['answer_assembler', 'evaluation', 'pipeline',
                                               'unified', 'orchestrator', 'rubric']):
            if 'RUBRIC_SCORING' not in content and 'rubric_scoring' not in content:
                issues.append("Missing RUBRIC_SCORING.json validation")
                self.stats["missing_rubric_validation"] += 1
        
        return issues
    
    def check_answer_assembler(self, file_path: pathlib.Path) -> List[str]:
        """Check for AnswerAssembler usage in relevant tests"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check pipeline/evaluation tests
        if any(x in file_path.stem for x in ['evaluation', 'pipeline', 'unified', 'e2e']):
            if 'AnswerAssembler' not in content and 'answer_assembler' not in content:
                issues.append("Missing AnswerAssembler component")
                self.stats["missing_answer_assembler"] += 1
        
        return issues
    
    def check_unified_pipeline(self, file_path: pathlib.Path) -> List[str]:
        """Check for unified_evaluation_pipeline usage"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check e2e and integration tests
        if 'e2e' in file_path.stem or 'integration' in str(file_path):
            if 'unified_evaluation_pipeline' not in content and 'UnifiedEvaluationPipeline' not in content:
                if 'pipeline' in file_path.stem or 'unified' in file_path.stem:
                    issues.append("Missing unified_evaluation_pipeline import")
                    self.stats["missing_unified_pipeline"] += 1
        
        return issues
    
    def check_system_validators(self, file_path: pathlib.Path) -> List[str]:
        """Check for system_validators pre/post execution gates"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check orchestrator and pipeline tests
        if any(x in file_path.stem for x in ['orchestrator', 'pipeline', 'e2e', 'unified', 'critical']):
            if 'system_validators' not in content and 'SystemHealthValidator' not in content:
                issues.append("Missing system_validators pre/post execution checks")
                self.stats["missing_system_validators"] += 1
        
        return issues
    
    def check_artifact_structure(self, file_path: pathlib.Path) -> List[str]:
        """Check for correct artifacts directory validation"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check e2e and pipeline tests
        if any(x in file_path.stem for x in ['e2e', 'pipeline', 'orchestrator', 'unified']):
            missing_artifacts = []
            if 'answers_report.json' not in content:
                missing_artifacts.append('answers_report.json')
            if 'flow_runtime.json' not in content:
                missing_artifacts.append('flow_runtime.json')
            
            if missing_artifacts and 'artifact' in content.lower():
                issues.append(f"Missing artifact checks: {', '.join(missing_artifacts)}")
                self.stats["missing_artifact_checks"] += 1
        
        return issues
    
    def check_determinism(self, file_path: pathlib.Path) -> List[str]:
        """Check for deterministic behavior validation"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check pipeline and orchestrator tests
        if any(x in file_path.stem for x in ['orchestrator', 'pipeline', 'e2e', 'deterministic',
                                               'reproducibility', 'unified', 'critical']):
            missing_checks = []
            if 'deterministic' not in content.lower() and 'hash' not in content.lower():
                missing_checks.append('deterministic hash validation')
            if 'frozen' not in content and 'freeze' not in content:
                missing_checks.append('frozen configuration')
            if 'evidence_id' not in content and 'evidence-id' not in content:
                missing_checks.append('consistent evidence_ids')
            
            if missing_checks and 'test' in file_path.stem:
                issues.append(f"Missing determinism checks: {', '.join(missing_checks)}")
                self.stats["missing_determinism_checks"] += 1
        
        return issues
    
    def check_evidence_registry(self, file_path: pathlib.Path) -> List[str]:
        """Check for EvidenceRegistry deterministic hashing tests"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check evidence and pipeline tests
        if 'evidence' in file_path.stem or any(x in file_path.stem for x in ['pipeline', 'e2e', 'unified']):
            if 'EvidenceRegistry' not in content and 'evidence_registry' not in content:
                if 'evidence' in file_path.stem:
                    issues.append("Missing EvidenceRegistry usage/testing")
                    self.stats["missing_evidence_registry"] += 1
        
        return issues
    
    def check_rubric_check_tool(self, file_path: pathlib.Path) -> List[str]:
        """Check for tools/rubric_check.py validation"""
        issues = []
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check rubric and validation tests
        if 'rubric' in file_path.stem or 'validator' in file_path.stem:
            if 'rubric_check' not in content and 'tools/rubric_check.py' not in content:
                if 'validation' in content.lower():
                    issues.append("Missing tools/rubric_check.py integration test")
                    self.stats["missing_rubric_check_tool"] += 1
        
        return issues
    
    def audit_file(self, file_path: pathlib.Path) -> Dict[str, List[str]]:
        """Perform complete audit on a single test file"""
        file_issues = {}
        
        all_checks = [
            self.check_deprecated_imports,
            self.check_rubric_validation,
            self.check_answer_assembler,
            self.check_unified_pipeline,
            self.check_system_validators,
            self.check_artifact_structure,
            self.check_determinism,
            self.check_evidence_registry,
            self.check_rubric_check_tool,
        ]
        
        for check_func in all_checks:
            try:
                issues = check_func(file_path)
                if issues:
                    check_name = check_func.__name__.replace('check_', '')
                    file_issues[check_name] = issues
            except Exception as e:
                file_issues["parse_error"] = [str(e)]
        
        return file_issues
    
    def generate_report(self) -> str:
        """Generate comprehensive audit report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TEST SUITE AUDIT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 80)
        for key, value in self.stats.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # Issues by file
        report.append("ISSUES BY FILE:")
        report.append("-" * 80)
        if not self.issues:
            report.append("  ✅ No issues found!")
        else:
            for file_path, file_issues in sorted(self.issues.items()):
                report.append(f"\n📄 {file_path}")
                for category, issues in file_issues.items():
                    report.append(f"  ⚠️  {category}:")
                    for issue in issues:
                        report.append(f"      - {issue}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_audit(self) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, int]]:
        """Run complete audit on all test files"""
        test_files = self.find_test_files()
        self.stats["total_test_files"] = len(test_files)
        
        print(f"Found {len(test_files)} test files to audit...")
        print("")
        
        for test_file in test_files:
            relative_path = str(test_file.relative_to(self.repo_root))
            file_issues = self.audit_file(test_file)
            
            if file_issues:
                self.issues[relative_path] = file_issues
                print(f"⚠️  {relative_path}: {len(file_issues)} categories with issues")
            else:
                print(f"✅ {relative_path}: No issues")
        
        print("")
        return self.issues, self.stats


def main():
    """Main audit execution"""
    print("Starting Comprehensive Test Suite Audit...")
    print("")
    
    auditor = TestAuditor()
    issues, stats = auditor.run_audit()
    
    # Generate and print report
    report = auditor.generate_report()
    print(report)
    
    # Save report to file
    report_path = pathlib.Path("test_audit_report.txt")
    report_path.write_text(report, encoding='utf-8')
    print(f"\n📝 Report saved to: {report_path}")
    
    # Exit with appropriate code
    if any(stats[k] > 0 for k in stats if k != "total_test_files"):
        print("\n❌ Audit found issues that need attention")
        return 1
    else:
        print("\n✅ All tests pass audit checks")
        return 0


if __name__ == "__main__":
    sys.exit(main())
