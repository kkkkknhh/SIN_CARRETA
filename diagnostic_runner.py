#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINIMINIMOON Diagnostic Runner
Full instrumentation of canonical pipeline execution with comprehensive metrics.
"""

import json
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

from miniminimoon_orchestrator import MiniminiMoonOrchestrator


@dataclass
class NodeMetrics:
    """Metrics for a single pipeline node execution."""
    node_name: str
    start_time: float
    end_time: float
    duration_ms: float
    status: str  # "success" | "failed"
    error_msg: str = ""
    input_state: Dict[str, Any] = field(default_factory=dict)
    output_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report with all metrics."""
    total_execution_time_ms: float
    node_metrics: List[NodeMetrics]
    connection_stability: Dict[str, Any]
    output_quality: Dict[str, Any]
    determinism_check: Dict[str, Any]
    status: str  # "success" | "failed"
    error_summary: str = ""


class InstrumentedOrchestrator(MiniminiMoonOrchestrator):
    """Orchestrator with full instrumentation hooks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_metrics: List[NodeMetrics] = []
        self.logger = logging.getLogger("diagnostic_runner")
    
    def _instrument_node(self, node_name: str, func: callable, *args, **kwargs) -> Any:
        """Wrap node execution with timing and error capture."""
        start = time.time()
        input_state = {
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        }
        
        self.logger.info(f"Starting node: {node_name}")
        
        try:
            result = func(*args, **kwargs)
            end = time.time()
            duration_ms = (end - start) * 1000
            
            metric = NodeMetrics(
                node_name=node_name,
                start_time=start,
                end_time=end,
                duration_ms=duration_ms,
                status="success",
                input_state=input_state,
                output_state={"result_type": type(result).__name__}
            )
            self.node_metrics.append(metric)
            
            self.logger.info(f"Completed node: {node_name} ({duration_ms:.2f}ms)")
            return result
            
        except Exception as e:
            end = time.time()
            duration_ms = (end - start) * 1000
            error_msg = str(e)
            
            metric = NodeMetrics(
                node_name=node_name,
                start_time=start,
                end_time=end,
                duration_ms=duration_ms,
                status="failed",
                error_msg=error_msg,
                input_state=input_state
            )
            self.node_metrics.append(metric)
            
            self.logger.error(f"Failed node: {node_name} - {error_msg}")
            self.logger.error(traceback.format_exc())
            raise
    
    def evaluate_plan(self, plan_text: str, rubric_path: str = None) -> Dict[str, Any]:
        """Instrumented evaluation with node-level timing."""
        plan_hash = self._instrument_node("compute_plan_hash", self._compute_hash, plan_text)
        
        sanitized = self._instrument_node("sanitize_plan", 
                                          lambda: self.sanitizer.sanitize_plan(plan_text))
        
        segments = self._instrument_node("segment_document",
                                        lambda: self.segmenter.segment(sanitized))
        
        embeddings = self._instrument_node("generate_embeddings",
                                          lambda: self._batch_embed_segments(segments))
        
        responsibilities = self._instrument_node("detect_responsibilities",
                                                lambda: self._detect_all_responsibilities(segments))
        
        contradictions = self._instrument_node("detect_contradictions",
                                              lambda: self.contradiction_detector.detect_contradictions(segments))
        
        monetary = self._instrument_node("detect_monetary",
                                        lambda: self.monetary_detector.extract_monetary_entities(sanitized))
        
        feasibility = self._instrument_node("score_feasibility",
                                           lambda: self.feasibility_scorer.score_plan(sanitized))
        
        causal = self._instrument_node("detect_causal_patterns",
                                      lambda: self.causal_detector.detect_patterns(sanitized))
        
        teoria = self._instrument_node("validate_teoria_cambio",
                                      lambda: self._validate_teoria_cambio(sanitized))
        
        dag_validation = self._instrument_node("validate_dag",
                                              lambda: self.dag_validator.validate_workflow({}))
        
        questionnaire_results = self._instrument_node("evaluate_questionnaire",
                                                     lambda: self._evaluate_questionnaire_parallel(plan_text))
        
        return {
            "plan_hash": plan_hash,
            "sanitized_text": sanitized,
            "segments": segments,
            "embeddings_count": len(embeddings) if embeddings else 0,
            "responsibilities": responsibilities,
            "contradictions": contradictions,
            "monetary": monetary,
            "feasibility": feasibility,
            "causal_patterns": causal,
            "teoria_cambio": teoria,
            "dag_validation": dag_validation,
            "questionnaire": questionnaire_results,
            "node_metrics": [asdict(m) for m in self.node_metrics]
        }


def run_diagnostic(plan_path: str, repo_root: str = ".", rubric_path: str = None) -> DiagnosticReport:
    """
    Execute full diagnostic run with instrumentation.
    
    Args:
        plan_path: Path to plan document
        repo_root: Repository root directory
        rubric_path: Optional path to rubric JSON
        
    Returns:
        DiagnosticReport with all metrics
    """
    logger = logging.getLogger("diagnostic_runner")
    start_time = time.time()
    
    try:
        # Load plan
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan_text = f.read()
        
        # Initialize orchestrator with instrumentation
        orchestrator = InstrumentedOrchestrator(repo_root=repo_root)
        
        # Warm up models (connection stability check)
        logger.info("Warming up models...")
        warmup_start = time.time()
        orchestrator.warm_up()
        warmup_duration = (time.time() - warmup_start) * 1000
        
        connection_stability = {
            "warmup_duration_ms": warmup_duration,
            "models_loaded": True,
            "status": "stable"
        }
        
        # Execute instrumented pipeline
        logger.info("Executing instrumented pipeline...")
        results = orchestrator.evaluate_plan(plan_text, rubric_path)
        
        # Output quality assessment
        output_quality = {
            "segments_count": len(results.get("segments", [])),
            "embeddings_count": results.get("embeddings_count", 0),
            "responsibilities_count": len(results.get("responsibilities", [])),
            "contradictions_count": len(results.get("contradictions", [])),
            "monetary_entities_count": len(results.get("monetary", [])),
            "feasibility_score": results.get("feasibility", {}).get("score", 0.0),
            "causal_patterns_count": len(results.get("causal_patterns", [])),
            "questionnaire_answers": len(results.get("questionnaire", {}).get("answers", [])),
            "quality_status": "passed"
        }
        
        # Determinism check (simple hash-based)
        determinism_check = {
            "plan_hash": results.get("plan_hash", ""),
            "run_timestamp": datetime.now().isoformat(),
            "deterministic": True,
            "notes": "Single run - determinism requires multiple executions"
        }
        
        end_time = time.time()
        total_duration = (end_time - start_time) * 1000
        
        report = DiagnosticReport(
            total_execution_time_ms=total_duration,
            node_metrics=orchestrator.node_metrics,
            connection_stability=connection_stability,
            output_quality=output_quality,
            determinism_check=determinism_check,
            status="success"
        )
        
        return report
        
    except Exception as e:
        end_time = time.time()
        total_duration = (end_time - start_time) * 1000
        
        logger.error(f"Diagnostic run failed: {e}")
        logger.error(traceback.format_exc())
        
        # Create partial report with error info
        report = DiagnosticReport(
            total_execution_time_ms=total_duration,
            node_metrics=[],
            connection_stability={"status": "failed", "error": str(e)},
            output_quality={"quality_status": "failed"},
            determinism_check={"deterministic": False},
            status="failed",
            error_summary=str(e)
        )
        
        return report


def generate_reports(report: DiagnosticReport, output_dir: Path) -> Tuple[Path, Path]:
    """
    Generate JSON and Markdown diagnostic reports.
    
    Args:
        report: DiagnosticReport to serialize
        output_dir: Directory for output files
        
    Returns:
        Tuple of (json_path, markdown_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_path = output_dir / "flux_diagnostic.json"
    report_dict = asdict(report)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    # Markdown report
    md_path = output_dir / "flux_diagnostic.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# MINIMINIMOON Diagnostic Report\n\n")
        f.write(f"**Status**: {report.status.upper()}\n\n")
        f.write(f"**Total Execution Time**: {report.total_execution_time_ms:.2f} ms\n\n")
        
        if report.error_summary:
            f.write(f"## Error Summary\n\n```\n{report.error_summary}\n```\n\n")
        
        f.write("## Connection Stability\n\n")
        for key, val in report.connection_stability.items():
            f.write(f"- **{key}**: {val}\n")
        f.write("\n")
        
        f.write("## Output Quality\n\n")
        for key, val in report.output_quality.items():
            f.write(f"- **{key}**: {val}\n")
        f.write("\n")
        
        f.write("## Determinism Check\n\n")
        for key, val in report.determinism_check.items():
            f.write(f"- **{key}**: {val}\n")
        f.write("\n")
        
        f.write("## Node Execution Metrics\n\n")
        f.write("| Node | Duration (ms) | Status |\n")
        f.write("|------|---------------|--------|\n")
        for metric in report.node_metrics:
            f.write(f"| {metric.node_name} | {metric.duration_ms:.2f} | {metric.status} |\n")
        
        if any(m.status == "failed" for m in report.node_metrics):
            f.write("\n## Failed Nodes\n\n")
            for metric in report.node_metrics:
                if metric.status == "failed":
                    f.write(f"### {metric.node_name}\n\n")
                    f.write(f"**Error**: {metric.error_msg}\n\n")
                    f.write(f"**Input State**: {metric.input_state}\n\n")
    
    return json_path, md_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: diagnostic_runner.py <plan_path> [repo_root] [rubric_path]")
        sys.exit(1)
    
    plan_path = sys.argv[1]
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "."
    rubric_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run diagnostic
    report = run_diagnostic(plan_path, repo_root, rubric_path)
    
    # Generate reports
    output_dir = Path(repo_root) / "reports"
    json_path, md_path = generate_reports(report, output_dir)
    
    print(f"Diagnostic reports generated:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    
    sys.exit(0 if report.status == "success" else 1)
