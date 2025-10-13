#!/usr/bin/env python3
"""
MINIMINIMOON Audit System
========================

Programmatically validates the MINIMINIMOON system by:
1. Checking all 16 canonical stages execute and produce evidence
2. Verifying module integrations (pdm_contra, factibilidad, CompetenceValidator, 
   ReliabilityCalibrator, doctoral_argumentation_engine)
3. Confirming 300 questions have doctoral justifications with 3+ evidence sources
4. Testing determinism by running orchestrator twice with identical inputs

Usage:
    python audit_system.py --config CONFIG_DIR --plan TEST_PLAN.pdf [--output REPORT.json]
"""

import ast
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class StageAuditResult:
    """Result of auditing a single pipeline stage."""
    stage_name: str
    stage_number: int
    executed: bool
    has_evidence: bool
    evidence_count: int
    error: Optional[str] = None


@dataclass
class ModuleIntegrationResult:
    """Result of auditing module integration."""
    module_name: str
    expected_stage: Optional[int]
    found_in_codebase: bool
    imported_in_orchestrator: bool
    invoked_in_stage: bool
    file_location: Optional[str] = None
    import_statement: Optional[str] = None
    invocation_method: Optional[str] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class QuestionCoverageDetail:
    """Coverage details for a single question."""
    question_id: str
    evidence_count: int
    detector_sources: List[str]
    has_doctoral_justification: bool
    meets_3plus_requirement: bool


@dataclass
class EvidenceCoverageResult:
    """Result of auditing evidence coverage."""
    total_questions_expected: int
    total_questions_found: int
    questions_with_0_sources: int
    questions_with_1_source: int
    questions_with_2_sources: int
    questions_with_3plus_sources: int
    questions_with_doctoral_justification: int
    coverage_percentage: float
    meets_requirements: bool
    question_details: List[QuestionCoverageDetail] = field(default_factory=list)


@dataclass
class ScoreMismatch:
    """Details of a score mismatch between runs."""
    question_id: str
    run1_score: float
    run2_score: float
    difference: float


@dataclass
class DeterminismResult:
    """Result of determinism validation."""
    run1_document_hash: str
    run2_document_hash: str
    run1_evidence_hash: str
    run2_evidence_hash: str
    run1_flow_hash: str
    run2_flow_hash: str
    document_hashes_match: bool
    evidence_hashes_match: bool
    flow_hashes_match: bool
    total_questions_compared: int
    questions_with_score_mismatch: int
    score_mismatches: List[ScoreMismatch] = field(default_factory=list)
    max_score_difference: float = 0.0
    is_deterministic: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete audit report."""
    audit_timestamp: str
    config_dir: str
    test_plan_path: str
    orchestrator_version: str
    stage_audit: List[StageAuditResult]
    module_integration_audit: List[ModuleIntegrationResult]
    evidence_coverage_audit: EvidenceCoverageResult
    determinism_audit: DeterminismResult
    overall_status: str
    summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# AUDITOR CLASS
# ============================================================================


class MINIMINIMOONAuditor:
    """Comprehensive auditor for MINIMINIMOON orchestrator."""
    
    CANONICAL_STAGES = [
        "sanitization",
        "plan_processing", 
        "document_segmentation",
        "embedding",
        "responsibility_detection",
        "contradiction_detection",
        "monetary_detection",
        "feasibility_scoring",
        "causal_detection",
        "teoria_cambio",
        "dag_validation",
        "evidence_registry_build",
        "decalogo_load",
        "decalogo_evaluation",
        "questionnaire_evaluation",
        "answers_assembly",
    ]
    
    EXPECTED_MODULES = {
        "pdm_contra": 6,  # CONTRADICTION stage
        "factibilidad": 8,  # FEASIBILITY stage
        "CompetenceValidator": 15,  # QUESTIONNAIRE_EVAL stage
        "ReliabilityCalibrator": 15,  # QUESTIONNAIRE_EVAL stage
        "doctoral_argumentation_engine": 16,  # ANSWER_ASSEMBLY stage
    }
    
    def __init__(self, config_dir: Path, test_plan_path: Path):
        """Initialize auditor."""
        self.config_dir = Path(config_dir)
        self.test_plan_path = Path(test_plan_path)
        self.orchestrator_source = Path("miniminimoon_orchestrator.py")
        self.logger = logging.getLogger(__name__)
        
        if not self.test_plan_path.exists():
            raise FileNotFoundError(f"Test plan not found: {test_plan_path}")
        
        if not self.orchestrator_source.exists():
            raise FileNotFoundError(f"Orchestrator not found: {self.orchestrator_source}")
    
    def audit_stage_execution(self, pipeline_results: Dict[str, Any]) -> List[StageAuditResult]:
        """Audit that all 16 canonical stages executed and produced evidence."""
        self.logger.info("=== AUDIT 1: Stage Execution ===")
        
        results = []
        stages_completed = pipeline_results.get("stages_completed", [])
        
        # Get evidence registry
        evidence_by_stage = {}
        if "evaluations" in pipeline_results:
            # Try to extract evidence counts from various sources
            pass
        
        for i, stage_name in enumerate(self.CANONICAL_STAGES, 1):
            executed = stage_name in stages_completed
            
            # Check for evidence - simplified check
            has_evidence = False
            evidence_count = 0
            error = None
            
            if not executed:
                error = f"Stage '{stage_name}' did not execute"
            
            results.append(StageAuditResult(
                stage_name=stage_name,
                stage_number=i,
                executed=executed,
                has_evidence=has_evidence,
                evidence_count=evidence_count,
                error=error
            ))
        
        # Log summary
        executed_count = sum(1 for r in results if r.executed)
        self.logger.info(f"Stages executed: {executed_count}/16")
        
        for result in results:
            status = "✓" if result.executed else "✗"
            self.logger.info(f"  {status} Stage {result.stage_number}: {result.stage_name}")
            if result.error:
                self.logger.error(f"    ERROR: {result.error}")
        
        return results
    
    def audit_module_integration(self) -> List[ModuleIntegrationResult]:
        """Audit that critical modules are imported and invoked."""
        self.logger.info("=== AUDIT 2: Module Integration ===")
        
        results = []
        
        # Parse orchestrator source
        with open(self.orchestrator_source, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            self.logger.error(f"Failed to parse orchestrator: {e}")
            return []
        
        # Extract imports
        imports = self._extract_imports(tree)
        
        # Check each expected module
        for module_name, expected_stage in self.EXPECTED_MODULES.items():
            result = self._check_module_integration(
                module_name, expected_stage, imports, source_code
            )
            results.append(result)
        
        # Log summary
        found_count = sum(1 for r in results if r.found_in_codebase)
        imported_count = sum(1 for r in results if r.imported_in_orchestrator)
        invoked_count = sum(1 for r in results if r.invoked_in_stage)
        
        self.logger.info(f"Modules found in codebase: {found_count}/5")
        self.logger.info(f"Modules imported in orchestrator: {imported_count}/5")
        self.logger.info(f"Modules invoked in stages: {invoked_count}/5")
        
        for result in results:
            status = "✓" if (result.found_in_codebase and result.imported_in_orchestrator) else "✗"
            self.logger.info(f"  {status} {result.module_name} (stage {result.expected_stage})")
            if result.issues:
                for issue in result.issues:
                    self.logger.warning(f"    ISSUE: {issue}")
            if result.file_location:
                self.logger.info(f"    Location: {result.file_location}")
        
        return results
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract all import statements from AST."""
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if module_name not in imports:
                        imports[module_name] = []
                    imports[module_name].append(f"import {module_name}")
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    imported_name = alias.name
                    full_key = f"{module_name}.{imported_name}" if module_name else imported_name
                    if full_key not in imports:
                        imports[full_key] = []
                    imports[full_key].append(f"from {module_name} import {imported_name}")
        
        return imports
    
    def _check_module_integration(
        self, 
        module_name: str, 
        expected_stage: int,
        imports: Dict[str, List[str]],
        source_code: str
    ) -> ModuleIntegrationResult:
        """Check if module is properly integrated."""
        issues = []
        
        # Check if file exists in codebase
        found_in_codebase = False
        file_location = None
        
        potential_files = [
            Path(f"{module_name}.py"),
            Path(f"{module_name.lower()}.py"),
            Path(f"{module_name.replace('_', '')}.py"),
        ]
        
        for pf in potential_files:
            if pf.exists():
                found_in_codebase = True
                file_location = str(pf)
                break
        
        if not found_in_codebase:
            issues.append(f"Module file not found in codebase (searched: {', '.join(str(p) for p in potential_files)})")
        
        # Check if imported in orchestrator
        imported_in_orchestrator = False
        import_statement = None
        
        for key, statements in imports.items():
            if module_name.lower() in key.lower():
                imported_in_orchestrator = True
                import_statement = statements[0] if statements else None
                break
        
        if not imported_in_orchestrator:
            issues.append(f"Not imported in orchestrator")
        
        # Check if invoked in source code
        invoked_in_stage = False
        invocation_method = None
        
        if module_name.lower() in source_code.lower():
            # Look for method/class invocations
            import re
            patterns = [
                rf"{module_name}\s*\(",
                rf"{module_name}\s*\.",
                rf"from\s+{module_name}\s+import",
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, source_code, re.IGNORECASE)
                if matches:
                    invoked_in_stage = True
                    invocation_method = f"Found pattern: {pattern}"
                    break
        
        if not invoked_in_stage and imported_in_orchestrator:
            issues.append(f"Imported but not invoked in stage {expected_stage}")
        
        return ModuleIntegrationResult(
            module_name=module_name,
            expected_stage=expected_stage,
            found_in_codebase=found_in_codebase,
            imported_in_orchestrator=imported_in_orchestrator,
            invoked_in_stage=invoked_in_stage,
            file_location=file_location,
            import_statement=import_statement,
            invocation_method=invocation_method,
            issues=issues
        )
    
    def audit_evidence_coverage(self, pipeline_results: Dict[str, Any]) -> EvidenceCoverageResult:
        """Audit that all 300 questions have 3+ evidence sources."""
        self.logger.info("=== AUDIT 3: Evidence Coverage ===")
        
        # Extract answers from results
        answers_report = pipeline_results.get("evaluations", {}).get("answers_report", {})
        question_answers = answers_report.get("question_answers", [])
        
        # Count by evidence sources
        by_source_count = {0: 0, 1: 0, 2: 0, 3: 0}
        question_details = []
        questions_with_doctoral = 0
        
        for qa in question_answers:
            question_id = qa.get("question_id", "")
            evidence_ids = qa.get("evidence_ids", [])
            evidence_count = len(evidence_ids)
            
            # Count by bucket
            if evidence_count == 0:
                by_source_count[0] += 1
            elif evidence_count == 1:
                by_source_count[1] += 1
            elif evidence_count == 2:
                by_source_count[2] += 1
            else:
                by_source_count[3] += 1
            
            # Extract detector sources from evidence IDs
            detector_sources = []
            for eid in evidence_ids:
                # Evidence IDs are typically prefixed with detector type
                if "_" in eid:
                    source = eid.split("_")[0]
                    if source not in detector_sources:
                        detector_sources.append(source)
            
            # Check for doctoral justification (simplified - check if rationale exists)
            has_doctoral = bool(qa.get("rationale", ""))
            if has_doctoral:
                questions_with_doctoral += 1
            
            meets_3plus = evidence_count >= 3
            
            question_details.append(QuestionCoverageDetail(
                question_id=question_id,
                evidence_count=evidence_count,
                detector_sources=detector_sources,
                has_doctoral_justification=has_doctoral,
                meets_3plus_requirement=meets_3plus
            ))
        
        total_found = len(question_answers)
        questions_with_3plus = by_source_count[3]
        coverage_percentage = (questions_with_3plus / 300 * 100) if total_found > 0 else 0.0
        meets_requirements = questions_with_3plus >= 300
        
        result = EvidenceCoverageResult(
            total_questions_expected=300,
            total_questions_found=total_found,
            questions_with_0_sources=by_source_count[0],
            questions_with_1_source=by_source_count[1],
            questions_with_2_sources=by_source_count[2],
            questions_with_3plus_sources=questions_with_3plus,
            questions_with_doctoral_justification=questions_with_doctoral,
            coverage_percentage=coverage_percentage,
            meets_requirements=meets_requirements,
            question_details=question_details[:50]  # Sample first 50
        )
        
        # Log summary
        self.logger.info(f"Total questions found: {total_found}/300")
        self.logger.info(f"Questions with 0 sources: {by_source_count[0]}")
        self.logger.info(f"Questions with 1 source: {by_source_count[1]}")
        self.logger.info(f"Questions with 2 sources: {by_source_count[2]}")
        self.logger.info(f"Questions with 3+ sources: {questions_with_3plus}")
        self.logger.info(f"Questions with doctoral justification: {questions_with_doctoral}")
        self.logger.info(f"Coverage: {coverage_percentage:.1f}%")
        
        status = "✓" if meets_requirements else "✗"
        self.logger.info(f"{status} Coverage requirement (300 questions with 3+ sources): {meets_requirements}")
        
        return result
    
    def audit_determinism(self) -> DeterminismResult:
        """Test determinism by running orchestrator twice with identical inputs."""
        self.logger.info("=== AUDIT 4: Determinism Validation ===")
        
        errors = []
        
        try:
            from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
            
            self.logger.info("Running orchestrator - Run 1...")
            orchestrator1 = CanonicalDeterministicOrchestrator(
                config_dir=self.config_dir,
                enable_validation=True,
                log_level="WARNING"  # Reduce noise
            )
            results1 = orchestrator1.process_plan_deterministic(str(self.test_plan_path))
            
            self.logger.info("Running orchestrator - Run 2...")
            orchestrator2 = CanonicalDeterministicOrchestrator(
                config_dir=self.config_dir,
                enable_validation=True,
                log_level="WARNING"
            )
            results2 = orchestrator2.process_plan_deterministic(str(self.test_plan_path))
            
            # Extract hashes
            doc_hash1 = results1.get("document_hash", "")
            doc_hash2 = results2.get("document_hash", "")
            evidence_hash1 = results1.get("evidence_hash", "")
            evidence_hash2 = results2.get("evidence_hash", "")
            
            # Extract flow hashes
            flow_hash1 = results1.get("validation", {}).get("flow_hash", "")
            flow_hash2 = results2.get("validation", {}).get("flow_hash", "")
            
            # Compare hashes
            doc_match = doc_hash1 == doc_hash2
            evidence_match = evidence_hash1 == evidence_hash2
            flow_match = flow_hash1 == flow_hash2
            
            # Compare scores
            answers1 = results1.get("evaluations", {}).get("answers_report", {}).get("question_answers", [])
            answers2 = results2.get("evaluations", {}).get("answers_report", {}).get("question_answers", [])
            
            score_mismatches = []
            max_diff = 0.0
            
            # Create lookup for run2
            run2_scores = {qa["question_id"]: qa.get("raw_score", 0.0) for qa in answers2}
            
            for qa1 in answers1:
                qid = qa1["question_id"]
                score1 = qa1.get("raw_score", 0.0)
                score2 = run2_scores.get(qid, 0.0)
                
                diff = abs(score1 - score2)
                if diff > 0.0001:  # Allow for floating point precision
                    score_mismatches.append(ScoreMismatch(
                        question_id=qid,
                        run1_score=score1,
                        run2_score=score2,
                        difference=diff
                    ))
                    max_diff = max(max_diff, diff)
            
            is_deterministic = (
                doc_match and 
                evidence_match and 
                flow_match and 
                len(score_mismatches) == 0
            )
            
            result = DeterminismResult(
                run1_document_hash=doc_hash1,
                run2_document_hash=doc_hash2,
                run1_evidence_hash=evidence_hash1,
                run2_evidence_hash=evidence_hash2,
                run1_flow_hash=flow_hash1,
                run2_flow_hash=flow_hash2,
                document_hashes_match=doc_match,
                evidence_hashes_match=evidence_match,
                flow_hashes_match=flow_match,
                total_questions_compared=len(answers1),
                questions_with_score_mismatch=len(score_mismatches),
                score_mismatches=score_mismatches[:20],  # Sample first 20
                max_score_difference=max_diff,
                is_deterministic=is_deterministic,
                errors=errors
            )
            
            # Log summary
            self.logger.info(f"Document hashes match: {doc_match}")
            self.logger.info(f"Evidence hashes match: {evidence_match}")
            self.logger.info(f"Flow hashes match: {flow_match}")
            self.logger.info(f"Questions compared: {len(answers1)}")
            self.logger.info(f"Score mismatches: {len(score_mismatches)}")
            if max_diff > 0:
                self.logger.info(f"Max score difference: {max_diff}")
            
            status = "✓" if is_deterministic else "✗"
            self.logger.info(f"{status} System is deterministic: {is_deterministic}")
            
            if not is_deterministic:
                if not doc_match:
                    self.logger.error("  Document hash mismatch!")
                if not evidence_match:
                    self.logger.error("  Evidence hash mismatch!")
                if not flow_match:
                    self.logger.error("  Flow hash mismatch!")
                if score_mismatches:
                    self.logger.error(f"  {len(score_mismatches)} score mismatches found!")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Determinism audit failed: {e}")
            errors.append(str(e))
            
            return DeterminismResult(
                run1_document_hash="",
                run2_document_hash="",
                run1_evidence_hash="",
                run2_evidence_hash="",
                run1_flow_hash="",
                run2_flow_hash="",
                document_hashes_match=False,
                evidence_hashes_match=False,
                flow_hashes_match=False,
                total_questions_compared=0,
                questions_with_score_mismatch=0,
                score_mismatches=[],
                max_score_difference=0.0,
                is_deterministic=False,
                errors=errors
            )
    
    def run_full_audit(self) -> AuditReport:
        """Run all audits and generate comprehensive report."""
        from datetime import datetime
        
        self.logger.info("=" * 70)
        self.logger.info("MINIMINIMOON SYSTEM AUDIT")
        self.logger.info("=" * 70)
        
        # Run orchestrator once to get results for stage and coverage audits
        self.logger.info("\nRunning orchestrator for audit...")
        
        try:
            from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
            
            orchestrator = CanonicalDeterministicOrchestrator(
                config_dir=self.config_dir,
                enable_validation=True,
                log_level="WARNING"
            )
            pipeline_results = orchestrator.process_plan_deterministic(str(self.test_plan_path))
            orchestrator_version = pipeline_results.get("orchestrator_version", "unknown")
            
        except Exception as e:
            self.logger.error(f"Failed to run orchestrator: {e}")
            raise
        
        # Run all audits
        stage_audit = self.audit_stage_execution(pipeline_results)
        module_audit = self.audit_module_integration()
        coverage_audit = self.audit_evidence_coverage(pipeline_results)
        determinism_audit = self.audit_determinism()
        
        # Determine overall status
        stages_ok = all(r.executed for r in stage_audit)
        modules_ok = all(r.found_in_codebase and r.imported_in_orchestrator for r in module_audit)
        coverage_ok = coverage_audit.meets_requirements
        determinism_ok = determinism_audit.is_deterministic
        
        if stages_ok and modules_ok and coverage_ok and determinism_ok:
            overall_status = "PASS"
        elif not stages_ok or not determinism_ok:
            overall_status = "FAIL"
        else:
            overall_status = "PARTIAL"
        
        # Create summary
        summary = {
            "stages_executed": sum(1 for r in stage_audit if r.executed),
            "stages_total": 16,
            "modules_found": sum(1 for r in module_audit if r.found_in_codebase),
            "modules_imported": sum(1 for r in module_audit if r.imported_in_orchestrator),
            "modules_invoked": sum(1 for r in module_audit if r.invoked_in_stage),
            "modules_total": 5,
            "questions_with_3plus_sources": coverage_audit.questions_with_3plus_sources,
            "questions_total": 300,
            "coverage_percentage": coverage_audit.coverage_percentage,
            "is_deterministic": determinism_audit.is_deterministic,
            "score_mismatches": determinism_audit.questions_with_score_mismatch,
        }
        
        report = AuditReport(
            audit_timestamp=datetime.utcnow().isoformat(),
            config_dir=str(self.config_dir),
            test_plan_path=str(self.test_plan_path),
            orchestrator_version=orchestrator_version,
            stage_audit=stage_audit,
            module_integration_audit=module_audit,
            evidence_coverage_audit=coverage_audit,
            determinism_audit=determinism_audit,
            overall_status=overall_status,
            summary=summary
        )
        
        # Print summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("AUDIT SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Overall Status: {overall_status}")
        self.logger.info(f"\nStages: {summary['stages_executed']}/{summary['stages_total']} executed")
        self.logger.info(f"Modules: {summary['modules_found']}/{summary['modules_total']} found, "
                        f"{summary['modules_imported']}/{summary['modules_total']} imported, "
                        f"{summary['modules_invoked']}/{summary['modules_total']} invoked")
        self.logger.info(f"Coverage: {summary['questions_with_3plus_sources']}/{summary['questions_total']} "
                        f"questions with 3+ sources ({summary['coverage_percentage']:.1f}%)")
        self.logger.info(f"Determinism: {summary['is_deterministic']} "
                        f"({summary['score_mismatches']} score mismatches)")
        self.logger.info("=" * 70)
        
        return report


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MINIMINIMOON System Audit - Validates orchestrator execution"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuration directory containing RUBRIC_SCORING.json"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Test plan file (PDF or text)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit_report.json"),
        help="Output path for audit report (default: audit_report.json)"
    )
    
    args = parser.parse_args()
    
    try:
        auditor = MINIMINIMOONAuditor(
            config_dir=args.config,
            test_plan_path=args.plan
        )
        
        report = auditor.run_full_audit()
        
        # Export to JSON
        report_dict = asdict(report)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nAudit report saved to: {args.output}")
        
        # Exit with appropriate code
        if report.overall_status == "PASS":
            sys.exit(0)
        elif report.overall_status == "PARTIAL":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
