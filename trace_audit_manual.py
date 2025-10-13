#!/usr/bin/env python3.10
"""Manual trace audit without running orchestrator - analyzes code structure."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def analyze_orchestrator_code() -> Dict[str, Any]:
    """Analyze orchestrator code structure to identify stage definitions and flows."""

    orchestrator_path = Path("miniminimoon_orchestrator.py")

    if not orchestrator_path.exists():
        return {"error": "Orchestrator file not found"}

    code = orchestrator_path.read_text(encoding="utf-8")

    # Find stage definitions
    stage_enum_match = re.search(
        r"class PipelineStage\(Enum\):(.*?)(?=\n\nclass|\nclass [A-Z])", code, re.DOTALL
    )

    stages_defined = []
    if stage_enum_match:
        enum_body = stage_enum_match.group(1)
        stage_matches = re.findall(r'(\w+)\s*=\s*["\']([^"\']+)["\']', enum_body)
        stages_defined = [(name, value) for name, value in stage_matches]

    # Find process_plan_deterministic method
    process_method_match = re.search(
        r"def process_plan_deterministic\(self.*?\):(.*?)(?=\n    def\s)",
        code,
        re.DOTALL,
    )

    stages_invoked = []
    evidence_registrations = []

    if process_method_match:
        method_body = process_method_match.group(1)

        # Find _run_stage calls
        run_stage_calls = re.findall(
            r"self\._run_stage\(\s*PipelineStage\.(\w+)", method_body
        )
        stages_invoked = run_stage_calls

        # Find evidence registration calls
        evidence_calls = re.findall(
            r"evidence_registry\.register|self\.evidence_registry\.register",
            method_body,
        )
        evidence_registrations = evidence_calls

    # Find _build_evidence_registry method
    build_evidence_match = re.search(
        r"def _build_evidence_registry\(self.*?\):(.*?)(?=\n    def\s)", code, re.DOTALL
    )

    evidence_stages = []
    if build_evidence_match:
        build_body = build_evidence_match.group(1)
        # Find register_evidence calls
        register_calls = re.findall(
            r"register_evidence\(\s*PipelineStage\.(\w+)", build_body
        )
        evidence_stages = register_calls

    # Analyze each stage enum vs invocation
    stage_analysis = {}

    for stage_name, stage_value in stages_defined:
        invoked = stage_name in stages_invoked
        has_evidence = stage_name in evidence_stages

        stage_analysis[stage_value] = {
            "enum_name": stage_name,
            "invoked_in_flow": invoked,
            "has_evidence_registration": has_evidence,
            "stage_number": stages_defined.index((stage_name, stage_value)) + 1,
        }

    return {
        "total_stages_defined": len(stages_defined),
        "total_stages_invoked": len(stages_invoked),
        "stages_defined": [v for _, v in stages_defined],
        "stages_invoked": stages_invoked,
        "evidence_stages": evidence_stages,
        "stage_analysis": stage_analysis,
    }


def identify_code_issues(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify potential code issues."""

    issues = []

    stage_analysis = analysis.get("stage_analysis", {})

    for stage_value, info in stage_analysis.items():
        stage_num = info["stage_number"]

        # Issue 1: Stage defined but never invoked
        if not info["invoked_in_flow"]:
            issues.append(
                {
                    "stage": stage_value,
                    "stage_number": stage_num,
                    "severity": "HIGH",
                    "issue_type": "UNREACHABLE_CODE",
                    "description": f"Stage {stage_num} ({stage_value}) defined in enum but never invoked in process_plan_deterministic",
                    "line_indicators": "PipelineStage enum vs process_plan_deterministic",
                    "remediation": "Add _run_stage call for this stage or remove from enum if obsolete",
                }
            )

        # Issue 2: Detector stages without evidence registration (stages 1-12 only)
        if stage_num <= 12:
            detection_stages = [
                "responsibility_detection",
                "contradiction_detection",
                "monetary_detection",
                "feasibility_scoring",
                "causal_detection",
                "teoria_cambio",
            ]

            if (
                stage_value in detection_stages
                and not info["has_evidence_registration"]
            ):
                issues.append(
                    {
                        "stage": stage_value,
                        "stage_number": stage_num,
                        "severity": "MEDIUM",
                        "issue_type": "MISSING_EVIDENCE_INTEGRATION",
                        "description": f"Detection stage {stage_num} ({stage_value}) lacks evidence registry integration",
                        "line_indicators": "_build_evidence_registry method",
                        "remediation": "Add register_evidence call in _build_evidence_registry for this stage output",
                    }
                )

    return issues


def generate_code_audit_report(
    analysis: Dict[str, Any], issues: List[Dict[str, Any]]
) -> str:
    """Generate audit report from code analysis."""

    lines = []
    lines.append("=" * 80)
    lines.append("ORCHESTRATOR CODE STRUCTURE AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Stages Defined in Enum: {analysis['total_stages_defined']}")
    lines.append(f"Stages Invoked in Flow: {analysis['total_stages_invoked']}")
    lines.append(f"Evidence Stages: {len(analysis['evidence_stages'])}")
    lines.append(f"Issues Identified: {len(issues)}")
    lines.append("")

    lines.append("SECTION 1: STAGE FLOW DEFINITION")
    lines.append("=" * 80)
    lines.append("")

    stage_analysis = analysis.get("stage_analysis", {})

    for i, stage_value in enumerate(analysis["stages_defined"], 1):
        info = stage_analysis[stage_value]

        invoked_symbol = "✓" if info["invoked_in_flow"] else "✗"
        evidence_symbol = "✓" if info["has_evidence_registration"] else "-"

        lines.append(f"Stage {i:2d}: {stage_value:30s}")
        lines.append(f"  Enum Name: {info['enum_name']}")
        lines.append(f"  Invoked in Flow: {invoked_symbol}")
        lines.append(f"  Evidence Registration: {evidence_symbol}")
        lines.append("")

    lines.append("")
    lines.append("SECTION 2: CODE ISSUES & REMEDIATION")
    lines.append("=" * 80)
    lines.append("")

    if not issues:
        lines.append("✓ No code issues detected")
    else:
        # Group by severity
        high_issues = [i for i in issues if i["severity"] == "HIGH"]
        medium_issues = [i for i in issues if i["severity"] == "MEDIUM"]

        if high_issues:
            lines.append("HIGH SEVERITY ISSUES:")
            lines.append("-" * 80)
            for issue in high_issues:
                lines.append(f"\nStage {issue['stage_number']}: {issue['stage']}")
                lines.append(f"  Type: {issue['issue_type']}")
                lines.append(f"  Description: {issue['description']}")
                lines.append(f"  Location: {issue['line_indicators']}")
                lines.append(f"  Remediation: {issue['remediation']}")
            lines.append("")

        if medium_issues:
            lines.append("\nMEDIUM SEVERITY ISSUES:")
            lines.append("-" * 80)
            for issue in medium_issues:
                lines.append(f"\nStage {issue['stage_number']}: {issue['stage']}")
                lines.append(f"  Type: {issue['issue_type']}")
                lines.append(f"  Description: {issue['description']}")
                lines.append(f"  Location: {issue['line_indicators']}")
                lines.append(f"  Remediation: {issue['remediation']}")
            lines.append("")

    lines.append("")
    lines.append("SECTION 3: EVIDENCE REGISTRATION MAP")
    lines.append("=" * 80)
    lines.append("")

    for stage_name in analysis["evidence_stages"]:
        stage_value = None
        for sv, info in stage_analysis.items():
            if info["enum_name"] == stage_name:
                stage_value = sv
                break

        if stage_value:
            info = stage_analysis[stage_value]
            lines.append(
                f"  Stage {info['stage_number']:2d} ({stage_value:30s}) → Evidence registered"
            )
        else:
            lines.append(f"  {stage_name:30s} → Evidence registered (unknown stage)")

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"AUDIT COMPLETE - {len(issues)} issues requiring attention")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 80)
    print("ORCHESTRATOR CODE STRUCTURE AUDIT")
    print("=" * 80)
    print()

    print("[1/3] Analyzing orchestrator code structure...")
    analysis = analyze_orchestrator_code()

    if "error" in analysis:
        print(f"  ✗ Error: {analysis['error']}")
        return 1

    print(f"  ✓ Found {analysis['total_stages_defined']} stage definitions")
    print(f"  ✓ Found {analysis['total_stages_invoked']} stage invocations")
    print()

    print("[2/3] Identifying code issues...")
    issues = identify_code_issues(analysis)
    print(f"  ✓ Identified {len(issues)} potential issues")
    print()

    print("[3/3] Generating audit report...")
    report = generate_code_audit_report(analysis, issues)

    # Save report
    output_dir = Path("trace_output")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "CODE_STRUCTURE_AUDIT.txt"
    report_path.write_text(report, encoding="utf-8")

    analysis_path = output_dir / "code_analysis.json"
    analysis_path.write_text(
        json.dumps({"analysis": analysis, "issues": issues}, indent=2), encoding="utf-8"
    )

    print(f"  ✓ Report saved: {report_path}")
    print(f"  ✓ Analysis saved: {analysis_path}")
    print()

    # Print report
    print(report)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
