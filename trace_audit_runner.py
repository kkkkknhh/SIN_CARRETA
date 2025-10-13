#!/usr/bin/env python3.10
"""
Direct trace audit execution without subprocess
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("execution_trace.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def create_test_pdm():
    """Create minimal test PDM."""
    content = """PLAN MUNICIPAL 2024
    
I. DIAGNÓSTICO
Población: 50,000 habitantes
Problemática: Infraestructura educativa insuficiente

II. OBJETIVOS
- Reducir brecha educativa
- Mejorar acceso a salud

III. METAS
Meta 1: Construir 5 escuelas
- Responsable: Secretaría de Educación
- Presupuesto: $15,000,000 MXN
- Indicador: Escuelas construidas

Meta 2: Ampliar clínica municipal
- Responsable: Dirección de Salud
- Presupuesto: $8,000,000 MXN

Si invertimos en educación, entonces mejorarán las competencias estudiantiles.
"""

    path = Path("test_pdm_minimal.txt")
    path.write_text(content, encoding="utf-8")
    return path


def analyze_logs():
    """Analyze execution logs."""
    log_path = Path("execution_trace.log")

    if not log_path.exists():
        print("No execution trace found")
        return

    stages_executed = []
    stage_data = {}
    errors = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "STAGE_ENTRY:" in line:
                try:
                    json_part = line.split("STAGE_ENTRY:", 1)[1].strip()
                    data = json.loads(json_part)
                    stage_name = data["stage_name"]
                    if stage_name not in stage_data:
                        stage_data[stage_name] = {"entries": [], "exits": []}
                    stage_data[stage_name]["entries"].append(data)
                except:
                    pass

            elif "STAGE_EXIT:" in line:
                try:
                    json_part = line.split("STAGE_EXIT:", 1)[1].strip()
                    data = json.loads(json_part)
                    stage_name = data["stage_name"]
                    if stage_name not in stage_data:
                        stage_data[stage_name] = {"entries": [], "exits": []}
                    stage_data[stage_name]["exits"].append(data)

                    if data.get("status") == "success":
                        stages_executed.append(stage_name)
                    elif data.get("status") == "failure":
                        errors.append(data)
                except:
                    pass

    # Generate report
    report = generate_quick_report(stage_data, stages_executed, errors)

    # Save report
    Path("trace_output").mkdir(exist_ok=True)
    Path("trace_output/QUICK_AUDIT.txt").write_text(report)

    print(report)


def generate_quick_report(stage_data, executed, errors):
    """Generate quick audit report."""
    lines = []
    lines.append("=" * 80)
    lines.append("ORCHESTRATOR TRACE AUDIT - QUICK REPORT")
    lines.append("=" * 80)
    lines.append("")

    expected = [
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

    lines.append(f"Expected Stages: {len(expected)}")
    lines.append(f"Executed Stages: {len(set(executed))}")
    lines.append(f"Errors: {len(errors)}")
    lines.append("")

    lines.append("STAGE EXECUTION DETAILS:")
    lines.append("-" * 80)

    for i, stage in enumerate(expected, 1):
        data = stage_data.get(stage, {})
        exits = data.get("exits", [])

        if not exits:
            lines.append(f"{i:2d}. {stage:30s} ✗ NOT_EXECUTED")
        else:
            exit_info = exits[0]
            status = exit_info.get("status", "unknown")
            duration = exit_info.get("duration_seconds", 0)
            evidence = exit_info.get("evidence_registered", 0)
            output = exit_info.get("output_summary", {})

            status_symbol = "✓" if status == "success" else "✗"
            lines.append(
                f"{i:2d}. {stage:30s} {status_symbol} {status.upper():10s} ({duration:.3f}s)"
            )
            lines.append(
                f"     Evidence: {evidence:3d} | Output: {output.get('type', 'unknown'):10s} size={output.get('size', 0)}"
            )

            if output.get("is_empty"):
                lines.append(f"     ⚠ EMPTY OUTPUT")
            if output.get("is_malformed"):
                lines.append(f"     ⚠ MALFORMED OUTPUT")
            if output.get("validation_errors"):
                for err in output["validation_errors"]:
                    lines.append(f"     ⚠ {err}")

        lines.append("")

    if errors:
        lines.append("")
        lines.append("ERRORS:")
        lines.append("-" * 80)
        for err in errors:
            lines.append(f"Stage: {err['stage_name']}")
            lines.append(
                f"  {err.get('error_type', 'Unknown')}: {err.get('error', 'No details')}"
            )
            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Main execution."""
    try:
        print("=" * 80)
        print("ORCHESTRATOR TRACE AUDIT")
        print("=" * 80)
        print()

        # Create test PDM
        print("[1/3] Creating test PDM...")
        pdm_path = create_test_pdm()
        print(f"  ✓ Created: {pdm_path}")
        print()

        # Setup config
        print("[2/3] Setting up configuration...")
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        rubric_path = config_dir / "RUBRIC_SCORING.json"
        if not rubric_path.exists():
            rubric = {"questions": {}, "weights": {}}
            rubric_path.write_text(json.dumps(rubric, indent=2))
        print(f"  ✓ Config ready")
        print()

        # Run orchestrator
        print("[3/3] Running orchestrator with trace logging...")
        print("-" * 80)

        try:
            from miniminimoon_orchestrator import UnifiedEvaluationPipeline

            output_dir = Path("trace_output")
            output_dir.mkdir(exist_ok=True)

            pipeline = UnifiedEvaluationPipeline(config_dir=config_dir)
            results = pipeline.evaluate(str(pdm_path), output_dir)

            print("-" * 80)
            print("  ✓ Orchestrator completed")

        except Exception as e:
            print("-" * 80)
            print(f"  ⚠ Orchestrator error: {type(e).__name__}: {e}")
            print("  Continuing with trace analysis...")

        print()
        print("=" * 80)
        print("TRACE ANALYSIS")
        print("=" * 80)
        print()

        # Analyze logs
        analyze_logs()

        return 0

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
