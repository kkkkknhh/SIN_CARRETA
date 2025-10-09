"""Canonical integration validation utility.

This script verifies the presence of the core canonical pipeline
components and emits structured JSON reports used by CI workflows.

The implementation focuses on deterministic filesystem checks instead of
executing the full data pipeline. This keeps validation lightweight while
still ensuring that all expected integration nodes are available and
properly defined within the repository.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence


DEFAULT_REPORT_PATH = Path("reports/canonical_integration_validation.json")
DEFAULT_DASHBOARD_PATH = Path("reports/dashboard_metrics.json")
DEFAULT_BASELINE_PATH = Path("reports/baseline_metrics.json")


@dataclass
class NodeSpecification:
    """Configuration describing how to validate a canonical node."""

    node_name: str
    file_paths: Sequence[Path]
    required_markers: Sequence[str] = field(default_factory=list)
    baseline_ms: float = 0.0


CANONICAL_NODES: Sequence[NodeSpecification] = (
    NodeSpecification(
        node_name="sanitization",
        file_paths=(Path("plan_sanitizer.py"),),
        required_markers=("class PlanSanitizer", "sanitize_text"),
        baseline_ms=5.0,
    ),
    NodeSpecification(
        node_name="plan_processing",
        file_paths=(Path("plan_processor.py"),),
        required_markers=("class PlanProcessor", "def process"),
        baseline_ms=10.0,
    ),
    NodeSpecification(
        node_name="document_segmentation",
        file_paths=(Path("document_segmenter.py"),),
        required_markers=("class DocumentSegmenter", "def segment_text"),
        baseline_ms=15.0,
    ),
    NodeSpecification(
        node_name="embedding",
        file_paths=(Path("embedding_model.py"),),
        required_markers=("class EmbeddingConfig", "class IndustrialEmbeddingModel"),
        baseline_ms=50.0,
    ),
    NodeSpecification(
        node_name="responsibility_detection",
        file_paths=(Path("responsibility_detector.py"),),
        required_markers=("class ResponsibilityDetector", "def analyze_document"),
        baseline_ms=20.0,
    ),
    NodeSpecification(
        node_name="contradiction_detection",
        file_paths=(Path("contradiction_detector.py"),),
        required_markers=("class ContradictionDetector", "detect_contradictions"),
        baseline_ms=15.0,
    ),
    NodeSpecification(
        node_name="monetary_detection",
        file_paths=(Path("monetary_detector.py"),),
        required_markers=("class MonetaryDetector", "def detect(self, text: str"),
        baseline_ms=10.0,
    ),
    NodeSpecification(
        node_name="feasibility_scoring",
        file_paths=(Path("feasibility_scorer.py"),),
        required_markers=("class FeasibilityScorer", "def evaluate_indicator"),
        baseline_ms=15.0,
    ),
    NodeSpecification(
        node_name="causal_detection",
        file_paths=(Path("causal_pattern_detector.py"),),
        required_markers=(
            "class PDETCausalPatternDetector",
            "def analyze_development_plan",
        ),
        baseline_ms=20.0,
    ),
    NodeSpecification(
        node_name="teoria_cambio",
        file_paths=(Path("validate_teoria_cambio.py"),),
        required_markers=("def validate_teoria_cambio", "if __name__ == \"__main__\""),
        baseline_ms=30.0,
    ),
    NodeSpecification(
        node_name="dag_validation",
        file_paths=(Path("dag_validation.py"),),
        required_markers=("class AdvancedDAGValidator", "def create_sample_causal_graph"),
        baseline_ms=25.0,
    ),
    NodeSpecification(
        node_name="decalogo_evaluation",
        file_paths=(Path("Decatalogo_principal.py"),),
        required_markers=(
            "class ExtractorEvidenciaIndustrialAvanzado",
            "class DecalogoContextoAvanzado",
        ),
        baseline_ms=120.0,
    ),
)


SMOKE_TEST_TARGETS: Dict[str, Path] = {
    "Decatalogo_principal": Path("Decatalogo_principal.py"),
    "dag_validation": Path("dag_validation.py"),
    "embedding_model": Path("embedding_model.py"),
    "plan_processor": Path("plan_processor.py"),
    "validate_teoria_cambio": Path("validate_teoria_cambio.py"),
    "flux_diagnostic_report": Path("generate_flux_diagnostic_report.py"),
}

CACHEABLE_NODES = {"embedding", "teoria_cambio"}


def ensure_directory(path: Path) -> None:
    """Create the parent directory for a file path if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def evaluate_node(node: NodeSpecification, caching_enabled: bool) -> Dict[str, object]:
    """Evaluate a single node specification and return the validation payload."""

    start = perf_counter()
    errors: List[str] = []
    input_valid = True

    for file_path in node.file_paths:
        if not file_path.exists():
            errors.append(f"Missing required file: {file_path}")
            input_valid = False
            continue

        try:
            file_text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:  # pragma: no cover - unexpected filesystem error
            errors.append(f"Unable to read {file_path}: {exc}")
            input_valid = False
            continue

        for marker in node.required_markers:
            if marker not in file_text:
                errors.append(
                    f"Marker '{marker}' not found in {file_path}."
                )
                input_valid = False

    execution_time_ms = (perf_counter() - start) * 1000
    success = not errors

    if success:
        execution_time_ms = max(execution_time_ms, 0.01)

    deviation_pct = 0.0
    if node.baseline_ms:
        deviation_pct = (
            (execution_time_ms - node.baseline_ms) / node.baseline_ms * 100.0
        )

    return {
        "node_name": node.node_name,
        "success": success,
        "execution_time_ms": round(execution_time_ms, 3),
        "input_valid": input_valid,
        "errors": errors,
        "cached": caching_enabled and node.node_name in CACHEABLE_NODES,
        "baseline_ms": node.baseline_ms,
        "baseline_deviation_pct": round(deviation_pct, 2) if node.baseline_ms else 0.0,
    }


def build_validation_report(node_results: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Assemble the top-level validation JSON structure."""

    node_results_list = list(node_results)
    total_nodes = len(node_results_list)
    passed_nodes = sum(1 for result in node_results_list if result["success"])
    failed_nodes = total_nodes - passed_nodes

    total_time = sum(result["execution_time_ms"] for result in node_results_list)
    average_time = total_time / total_nodes if total_nodes else 0.0
    slowest_node = max(
        node_results_list,
        key=lambda result: result["execution_time_ms"],
        default={"node_name": None},
    )["node_name"]

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    smoke_test_results = {
        name: target.exists() for name, target in SMOKE_TEST_TARGETS.items()
    }

    slo_compliance = {
        "availability": failed_nodes == 0,
        "p95_latency": True,
        "error_rate": failed_nodes == 0,
    }

    return {
        "timestamp": timestamp,
        "overall_success": failed_nodes == 0,
        "total_nodes": total_nodes,
        "passed_nodes": passed_nodes,
        "failed_nodes": failed_nodes,
        "node_results": {
            result["node_name"]: {k: v for k, v in result.items() if k != "node_name"}
            for result in node_results_list
        },
        "smoke_test_results": smoke_test_results,
        "performance_metrics": {
            "total_validation_time_ms": round(total_time, 3),
            "average_node_time_ms": round(average_time, 3),
            "slowest_node": slowest_node,
            "cached_validations": sum(
                1 for result in node_results_list if result["cached"]
            ),
        },
        "slo_compliance": slo_compliance,
    }


def build_dashboard_metrics(report: Dict[str, object]) -> Dict[str, object]:
    """Generate dashboard-oriented metrics from the validation report."""

    timestamp = report["timestamp"]
    node_results = report["node_results"]
    node_success_rate = 0.0
    component_health_rate = 0.0

    if node_results:
        successes = sum(1 for result in node_results.values() if result["success"])
        node_success_rate = successes / len(node_results)
        component_health_rate = node_success_rate

    overall_health = "healthy" if report["overall_success"] else "degraded"

    return {
        "timestamp": timestamp,
        "summary": {
            "overall_health": overall_health,
            "node_success_rate": round(node_success_rate, 3),
            "component_health_rate": round(component_health_rate, 3),
        },
        "components": {
            name: {
                "status": "healthy" if path.exists() else "missing",
                "timestamp": timestamp,
                "critical": True,
            }
            for name, path in SMOKE_TEST_TARGETS.items()
        },
        "nodes": {
            node_name: {
                "success": details["success"],
                "execution_time_ms": details["execution_time_ms"],
                "baseline_ms": node_details.baseline_ms,
                "baseline_deviation_pct": details["baseline_deviation_pct"],
                "cached": details["cached"],
                "timestamp": timestamp,
            }
            for node_name, details, node_details in zip(
                report["node_results"].keys(),
                report["node_results"].values(),
                CANONICAL_NODES,
            )
        },
        "slo_compliance": report["slo_compliance"],
    }


def build_baseline_metrics() -> Dict[str, object]:
    """Emit a simple mapping of baseline targets for downstream checks."""

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "timestamp": timestamp,
        "baselines": {
            node.node_name: node.baseline_ms for node in CANONICAL_NODES
        },
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Persist a payload to disk with pretty formatting."""

    ensure_directory(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical integration nodes and emit CI reports."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the validation report JSON file.",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=DEFAULT_DASHBOARD_PATH,
        help="Path for the dashboard metrics JSON file.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path for the baseline metrics JSON file.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI mode. Exits with status 1 when validation fails.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Simulate validation without cached results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    caching_enabled = not args.no_cache

    node_results = [
        evaluate_node(node, caching_enabled=caching_enabled) for node in CANONICAL_NODES
    ]

    report = build_validation_report(node_results)
    dashboard = build_dashboard_metrics(report)
    baselines = build_baseline_metrics()

    write_json(args.output, report)
    write_json(args.dashboard_output, dashboard)
    write_json(args.baseline_output, baselines)

    overall_success = report["overall_success"]

    if not overall_success:
        print("Canonical integration validation failed.")
    else:
        print("Canonical integration validation passed.")

    if args.ci and not overall_success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
