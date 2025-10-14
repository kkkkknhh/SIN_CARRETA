#!/usr/bin/env python3
"""
Trace Execution Test & Audit Report Generator
Runs orchestrator against test PDM and analyzes execution trace
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Configure structured logging to capture stage traces
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("execution_trace.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)


def create_test_pdm(output_path: Path):
    """Create a minimal test PDM document."""
    test_content = """
PLAN MUNICIPAL DE DESARROLLO 2024-2027
Municipio de San Pedro

I. DIAGNÓSTICO

El municipio de San Pedro cuenta con 50,000 habitantes. Se identifican las siguientes 
problemáticas prioritarias:

1. Infraestructura educativa insuficiente: Solo 15 escuelas para una población escolar 
   de 12,000 estudiantes.

2. Acceso limitado a servicios de salud: Una clínica para toda la población.

3. Desempleo juvenil del 35%, principalmente por falta de capacitación técnica.

II. VISIÓN Y OBJETIVOS

Visión: Convertir a San Pedro en un municipio próspero y sostenible para 2027.

Objetivos estratégicos:
- Reducir la brecha educativa en 40%
- Mejorar el acceso a salud pública
- Crear 2,000 empleos para jóvenes

III. ESTRATEGIAS Y METAS

Eje 1: Educación de Calidad

Meta 1.1: Construir 5 nuevas escuelas en zonas rurales
- Responsable: Secretaría de Educación Municipal
- Presupuesto: $15,000,000 MXN
- Plazo: 24 meses
- Indicadores: Número de escuelas construidas, estudiantes atendidos

Meta 1.2: Equipar bibliotecas digitales en 10 escuelas existentes
- Responsable: Coordinación de Tecnología Educativa
- Presupuesto: $3,500,000 MXN
- Plazo: 12 meses

Eje 2: Salud para Todos

Meta 2.1: Ampliar clínica municipal con 20 consultorios adicionales
- Responsable: Dirección de Salud Municipal
- Presupuesto: $8,000,000 MXN
- Plazo: 18 meses
- Indicadores: Consultas mensuales, tiempo de espera

Meta 2.2: Programa móvil de salud para comunidades remotas
- Responsable: Coordinación de Salud Comunitaria
- Presupuesto: $2,000,000 MXN
- Plazo: 6 meses

Eje 3: Desarrollo Económico

Meta 3.1: Centro de capacitación técnica para jóvenes
- Responsable: Secretaría de Desarrollo Económico
- Presupuesto: $5,000,000 MXN
- Plazo: 15 meses
- Esperamos que esto reduzca el desempleo juvenil en 20%
- Si capacitamos a 1,000 jóvenes, entonces mejoraremos sus oportunidades laborales

Meta 3.2: Fondo de apoyo a microempresas locales
- Responsable: Coordinación de Fomento Empresarial  
- Presupuesto: $10,000,000 MXN
- Plazo: 36 meses
- Beneficiarios esperados: 300 microempresarios

IV. MECANISMOS DE SEGUIMIENTO

Se realizarán evaluaciones trimestrales con la participación de:
- Comité de Planeación Municipal
- Consejos ciudadanos sectoriales
- Contraloría Social

Los indicadores se reportarán en el Sistema Nacional de Información Municipal.

V. PRESUPUESTO CONSOLIDADO

Total del Plan: $43,500,000 MXN

Distribución por eje:
- Educación: $18,500,000 (42.5%)
- Salud: $10,000,000 (23.0%)
- Desarrollo Económico: $15,000,000 (34.5%)

Fuentes de financiamiento:
- Recursos propios: $15,000,000
- Participaciones federales: $20,000,000
- Financiamiento estatal: $8,500,000

VI. TEORÍA DEL CAMBIO

Si invertimos en educación y capacitación técnica,
entonces mejoraremos las competencias de los jóvenes,
lo que resultará en mayor empleabilidad y reducción de la pobreza.

Si ampliamos la infraestructura de salud,
entonces aumentará el acceso a servicios médicos,
lo que mejorará la calidad de vida de la población.

La inversión en microempresas generará un efecto multiplicador en la economía local.
"""

    output_path.write_text(test_content, encoding="utf-8")
    return output_path


def parse_execution_trace(log_path: Path) -> Dict[str, Any]:
    """Parse execution trace from log file."""
    traces = {"stages": [], "entries": [], "exits": [], "errors": []}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "STAGE_ENTRY:" in line:
                json_part = line.split("STAGE_ENTRY:", 1)[1].strip()
                try:
                    entry = json.loads(json_part)
                    traces["entries"].append(entry)
                except json.JSONDecodeError:
                    pass

            elif "STAGE_EXIT:" in line:
                json_part = line.split("STAGE_EXIT:", 1)[1].strip()
                try:
                    exit_data = json.loads(json_part)
                    traces["exits"].append(exit_data)
                    if exit_data.get("status") == "success":
                        traces["stages"].append(exit_data["stage_name"])
                    else:
                        traces["errors"].append(exit_data)
                except json.JSONDecodeError:
                    pass

    return traces


def analyze_stage_execution(traces: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze stage execution to identify issues."""

    expected_stages = [
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

    executed_stages = set(traces["stages"])
    missing_stages = [s for s in expected_stages if s not in executed_stages]

    stage_analysis = {}

    # Analyze each stage
    for stage_name in expected_stages:
        exits = [e for e in traces["exits"] if e["stage_name"] == stage_name]

        if not exits:
            stage_analysis[stage_name] = {
                "executed": False,
                "status": "NOT_EXECUTED",
                "reason": "Stage never reached or bypassed",
                "evidence_count": 0,
                "output_valid": False,
                "issues": ["Stage did not execute"],
            }
        else:
            exit_data = exits[0]
            output_summary = exit_data.get("output_summary", {})

            issues = []

            # Check for empty output
            if output_summary.get("is_empty", False):
                issues.append("Empty output")

            # Check for malformed output
            if output_summary.get("is_malformed", False):
                issues.append("Malformed output")

            # Check validation errors
            validation_errors = output_summary.get("validation_errors", [])
            issues.extend(validation_errors)

            # Check evidence registration
            evidence_count = exit_data.get("evidence_registered", 0)
            if evidence_count == 0 and stage_name in [
                "responsibility_detection",
                "contradiction_detection",
                "monetary_detection",
                "feasibility_scoring",
                "causal_detection",
                "teoria_cambio",
            ]:
                issues.append("No evidence registered")

            stage_analysis[stage_name] = {
                "executed": True,
                "status": exit_data.get("status", "unknown"),
                "duration": exit_data.get("duration_seconds", 0),
                "evidence_count": evidence_count,
                "output_valid": len(issues) == 0,
                "output_summary": output_summary,
                "issues": issues,
            }

    return {
        "expected_stages": expected_stages,
        "executed_stages": list(executed_stages),
        "missing_stages": missing_stages,
        "stage_details": stage_analysis,
        "total_errors": len(traces["errors"]),
        "error_details": traces["errors"],
    }


def identify_dead_code(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify potential dead code in stages 1-12."""
    dead_code_findings = []

    stages_1_12 = [
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
    ]

    for stage_name in stages_1_12:
        stage_info = analysis["stage_details"].get(stage_name, {})

        # Stage never executed = potential dead code
        if not stage_info.get("executed", False):
            dead_code_findings.append(
                {
                    "stage": stage_name,
                    "issue": "UNREACHABLE_CODE",
                    "description": "Stage code exists but never executes",
                    "remediation": "Check conditional logic or remove if obsolete",
                }
            )

        # Empty output = potential dead code path
        elif stage_info.get("output_summary", {}).get("is_empty", False):
            dead_code_findings.append(
                {
                    "stage": stage_name,
                    "issue": "DEAD_CODE_PATH",
                    "description": "Stage executes but produces no output",
                    "remediation": "Verify implementation or remove unused branches",
                }
            )

        # No evidence when expected = integration missing
        elif stage_info.get(
            "evidence_count", 0
        ) == 0 and "No evidence registered" in stage_info.get("issues", []):
            dead_code_findings.append(
                {
                    "stage": stage_name,
                    "issue": "MISSING_INTEGRATION",
                    "description": "Stage runs but evidence registration never called",
                    "remediation": "Add evidence_registry.register() calls",
                }
            )

    return dead_code_findings


def generate_audit_report(
    analysis: Dict[str, Any], dead_code: List[Dict[str, Any]]
) -> str:
    """Generate structured audit report."""

    report_lines = [
        "=" * 80,
        "ORCHESTRATOR EXECUTION TRACE AUDIT REPORT",
        "=" * 80,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 80,
        f"Total Stages Expected: {len(analysis['expected_stages'])}",
        f"Total Stages Executed: {len(analysis['executed_stages'])}",
        f"Missing Stages: {len(analysis['missing_stages'])}",
        f"Total Errors: {analysis['total_errors']}",
        f"Dead Code Issues: {len(dead_code)}",
        "",
    ]

    # Section 1: Stage-by-Stage Execution Status
    report_lines.extend(["SECTION 1: STAGE EXECUTION STATUS", "=" * 80, ""])

    for i, stage_name in enumerate(analysis["expected_stages"], 1):
        stage_info = analysis["stage_details"][stage_name]

        report_lines.append(f"Stage {i}: {stage_name}")
        report_lines.append("-" * 80)
        report_lines.append(f"  Executed: {'✓' if stage_info['executed'] else '✗'}")
        report_lines.append(f"  Status: {stage_info['status']}")

        if stage_info["executed"]:
            report_lines.append(f"  Duration: {stage_info['duration']:.3f}s")
            report_lines.append(
                f"  Evidence Registered: {stage_info['evidence_count']}"
            )
            report_lines.append(
                f"  Output Valid: {'✓' if stage_info['output_valid'] else '✗'}"
            )

            if stage_info["output_summary"]:
                os = stage_info["output_summary"]
                report_lines.append(f"  Output Type: {os.get('type', 'unknown')}")
                report_lines.append(f"  Output Size: {os.get('size', 0)}")

                if os.get("is_empty"):
                    report_lines.append("  ⚠ WARNING: Empty output")
                if os.get("is_malformed"):
                    report_lines.append("  ⚠ WARNING: Malformed output")

            if stage_info["issues"]:
                report_lines.append("  Issues Found:")
                for issue in stage_info["issues"]:
                    report_lines.append(f"    - {issue}")
        else:
            report_lines.append(f"  Reason: {stage_info.get('reason', 'Unknown')}")

        report_lines.append("")

    # Section 2: Evidence Registration Verification
    report_lines.extend(
        ["", "SECTION 2: EVIDENCE REGISTRATION VERIFICATION", "=" * 80, ""]
    )

    evidence_stages = [
        "responsibility_detection",
        "contradiction_detection",
        "monetary_detection",
        "feasibility_scoring",
        "causal_detection",
        "teoria_cambio",
    ]

    for stage_name in evidence_stages:
        stage_info = analysis["stage_details"].get(stage_name, {})
        count = stage_info.get("evidence_count", 0)
        status = "✓ OK" if count > 0 else "✗ MISSING"
        report_lines.append(f"  {stage_name:30s} → {count:3d} entries  {status}")

    report_lines.append("")

    # Section 3: Dead Code and Missing Integration Points
    report_lines.extend(
        [
            "",
            "SECTION 3: DEAD CODE & MISSING INTEGRATION POINTS (Stages 1-12)",
            "=" * 80,
            "",
        ]
    )

    if not dead_code:
        report_lines.append("  ✓ No dead code or missing integration points detected")
    else:
        for finding in dead_code:
            report_lines.append(f"Stage: {finding['stage']}")
            report_lines.append(f"  Issue Type: {finding['issue']}")
            report_lines.append(f"  Description: {finding['description']}")
            report_lines.append(f"  Remediation: {finding['remediation']}")
            report_lines.append("")

    # Section 4: Error Details
    if analysis["error_details"]:
        report_lines.extend(["", "SECTION 4: ERROR DETAILS", "=" * 80, ""])

        for error in analysis["error_details"]:
            report_lines.append(f"Stage: {error['stage_name']}")
            report_lines.append(f"  Error Type: {error.get('error_type', 'Unknown')}")
            report_lines.append(f"  Error Message: {error.get('error', 'No message')}")
            report_lines.append(f"  Duration: {error.get('duration_seconds', 0):.3f}s")
            report_lines.append("")

    # Section 5: Recommendations
    report_lines.extend(["", "SECTION 5: REMEDIATION RECOMMENDATIONS", "=" * 80, ""])

    if analysis["missing_stages"]:
        report_lines.append("Missing Stages:")
        for stage in analysis["missing_stages"]:
            report_lines.append(
                f"  - {stage}: Ensure stage is invoked in orchestrator flow"
            )
        report_lines.append("")

    if dead_code:
        report_lines.append("Dead Code Remediation:")
        seen_issues = set()
        for finding in dead_code:
            key = (finding["issue"], finding["remediation"])
            if key not in seen_issues:
                report_lines.append(f"  - {finding['issue']}: {finding['remediation']}")
                seen_issues.add(key)
        report_lines.append("")

    # Count issues
    total_issues = (
        len(analysis["missing_stages"])
        + len(dead_code)
        + analysis["total_errors"]
        + sum(
            1
            for s in analysis["stage_details"].values()
            if not s.get("output_valid", True)
        )
    )

    report_lines.extend(
        [
            "",
            "AUDIT SUMMARY",
            "-" * 80,
            f"Total Issues Identified: {total_issues}",
            f"  - Missing Stages: {len(analysis['missing_stages'])}",
            f"  - Dead Code Issues: {len(dead_code)}",
            f"  - Execution Errors: {analysis['total_errors']}",
            f"  - Invalid Outputs: {sum(1 for s in analysis['stage_details'].values() if not s.get('output_valid', True))}",
            "",
            "=" * 80,
        ]
    )

    return "\n".join(report_lines)


def main():
    """Main execution function."""
    print("=" * 80)
    print("ORCHESTRATOR TRACE EXECUTION & AUDIT")
    print("=" * 80)
    print()

    # Setup paths
    test_pdm_path = Path("test_pdm_trace.txt")
    output_dir = Path("trace_output")
    output_dir.mkdir(exist_ok=True)

    # Create test PDM
    print("[1/5] Creating test PDM...")
    create_test_pdm(test_pdm_path)
    print(f"  ✓ Test PDM created: {test_pdm_path}")
    print()

    # Run orchestrator
    print("[2/5] Running orchestrator with structured logging...")
    try:
        from miniminimoon_orchestrator import UnifiedEvaluationPipeline

        config_dir = Path("config")
        if not config_dir.exists():
            config_dir.mkdir()
            # Create minimal rubric
            rubric = {"questions": {}, "weights": {}}
            (config_dir / "RUBRIC_SCORING.json").write_text(
                json.dumps(rubric, indent=2)
            )

        pipeline = UnifiedEvaluationPipeline(config_dir=config_dir)

        try:
            results = pipeline.evaluate(str(test_pdm_path), output_dir)
            print("  ✓ Orchestrator execution completed")
        except Exception as e:
            print(f"  ⚠ Orchestrator execution encountered errors: {e}")
            print("  Continuing with trace analysis...")

    except ImportError as e:
        print(f"  ✗ Failed to import orchestrator: {e}")
        return 1

    print()

    # Parse execution trace
    print("[3/5] Parsing execution trace...")
    log_path = Path("execution_trace.log")
    if not log_path.exists():
        print(f"  ✗ Trace log not found: {log_path}")
        return 1

    traces = parse_execution_trace(log_path)
    print(f"  ✓ Parsed {len(traces['entries'])} stage entries")
    print(f"  ✓ Parsed {len(traces['exits'])} stage exits")
    print()

    # Analyze execution
    print("[4/5] Analyzing stage execution...")
    analysis = analyze_stage_execution(traces)
    print(
        f"  ✓ Executed: {len(analysis['executed_stages'])}/{len(analysis['expected_stages'])} stages"
    )
    print(f"  ✓ Identified {analysis['total_errors']} errors")
    print()

    # Identify dead code
    print("[5/5] Identifying dead code and missing integrations...")
    dead_code = identify_dead_code(analysis)
    print(f"  ✓ Found {len(dead_code)} dead code issues")
    print()

    # Generate audit report
    print("Generating audit report...")
    report = generate_audit_report(analysis, dead_code)

    # Save reports
    report_path = output_dir / "AUDIT_REPORT.txt"
    report_path.write_text(report, encoding="utf-8")

    analysis_path = output_dir / "trace_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {"analysis": analysis, "dead_code": dead_code}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print(f"✓ Audit report saved: {report_path}")
    print(f"✓ Trace analysis saved: {analysis_path}")
    print()

    # Print report to console
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
