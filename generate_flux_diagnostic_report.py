#!/usr/bin/env python3
"""
Flux Diagnostic Report Generator
Consumes reports/flux_diagnostic.json and produces reports/flux_diagnostic.md
with comprehensive pipeline health analysis.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human-readable format."""
    if bytes_val < 1024:
        return f"{bytes_val:.1f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f} MB"
    else:
        return f"{bytes_val/1024**3:.2f} GB"


def format_latency(seconds: float) -> str:
    """Format latency with appropriate unit."""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f} μs"
    elif seconds < 1.0:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def format_throughput(items_per_sec: float) -> str:
    """Format throughput."""
    if items_per_sec >= 1000:
        return f"{items_per_sec/1000:.1f}k items/s"
    else:
        return f"{items_per_sec:.1f} items/s"


def assess_node_health(node: Dict[str, Any]) -> Tuple[str, str]:
    """
    Assess node health and return (status, reason).
    Returns: ('PASS'|'WARN'|'FAIL', reason_string)
    """
    latency = node.get("latency_ms", 0)
    memory = node.get("peak_memory_mb", 0)
    throughput = node.get("throughput", 0)
    error_rate = node.get("error_rate", 0)
    
    # Failure criteria
    if error_rate > 0.05:
        return "FAIL", f"High error rate: {error_rate*100:.1f}%"
    if latency > 5000:
        return "FAIL", f"Excessive latency: {format_latency(latency/1000)}"
    if memory > 2048:
        return "FAIL", f"Memory overload: {memory:.0f} MB"
    
    # Warning criteria
    if latency > 2000:
        return "WARN", f"High latency: {format_latency(latency/1000)}"
    if memory > 1024:
        return "WARN", f"High memory: {memory:.0f} MB"
    if throughput < 1.0 and throughput > 0:
        return "WARN", f"Low throughput: {format_throughput(throughput)}"
    
    return "PASS", "Nominal"


def generate_executive_summary(data: Dict[str, Any]) -> str:
    """Generate executive summary section (≤200 words)."""
    nodes = data.get("nodes", {})
    connections = data.get("connections", {})
    output_quality = data.get("output_quality", {})
    
    total_nodes = len(nodes)
    pass_count = sum(1 for n in nodes.values() if assess_node_health(n)[0] == "PASS")
    fail_count = sum(1 for n in nodes.values() if assess_node_health(n)[0] == "FAIL")
    
    stable_connections = sum(1 for c in connections.values() if c.get("stability", 0) >= 0.95)
    total_connections = len(connections)
    
    determinism = output_quality.get("determinism_verified", False)
    coverage = output_quality.get("question_coverage", 0)
    rubric_aligned = output_quality.get("rubric_aligned", False)
    gates_passed = output_quality.get("all_gates_passed", False)
    
    health_status = "HEALTHY" if fail_count == 0 and gates_passed else "DEGRADED" if fail_count < 3 else "CRITICAL"
    
    summary = f"""## Executive Summary

**Pipeline Health:** {health_status}  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The evaluation pipeline processed {total_nodes} stages with {pass_count}/{total_nodes} passing nodes and {fail_count} failures. Inter-node connectivity shows {stable_connections}/{total_connections} stable data flows (≥95% reliability).

**Output Quality:** {'✓' if determinism else '✗'} Deterministic execution, {coverage}/300 questions covered, {'✓' if rubric_aligned else '✗'} rubric alignment verified, {'✓' if gates_passed else '✗'} all acceptance gates passed.

**Critical Findings:** """
    
    if fail_count > 0:
        summary += f"{fail_count} node(s) failing performance thresholds. "
    if not rubric_aligned:
        summary += "Rubric alignment check failed. "
    if not gates_passed:
        summary += "One or more acceptance gates not passed. "
    if fail_count == 0 and rubric_aligned and gates_passed:
        summary += "All systems nominal. No critical issues detected."
    
    return summary


def generate_performance_table(nodes: Dict[str, Any]) -> str:
    """Generate node-by-node performance table."""
    table = """## Node-by-Node Performance Analysis

| Stage | Latency | Peak Memory | Throughput | Status | Notes |
|-------|---------|-------------|------------|--------|-------|
"""
    
    # Sort nodes by stage order (assumes node names are numbered or ordered)
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
    
    for node_name, metrics in sorted_nodes:
        latency = metrics.get("latency_ms", 0)
        memory = metrics.get("peak_memory_mb", 0)
        throughput = metrics.get("throughput", 0)
        
        status, reason = assess_node_health(metrics)
        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
        
        table += f"| {node_name} | {format_latency(latency/1000)} | {format_bytes(memory*1024**2)} | "
        if throughput > 0:
            table += f"{format_throughput(throughput)} "
        else:
            table += "N/A "
        table += f"| {status_icon} {status} | {reason} |\n"
    
    return table


def generate_connection_assessment(connections: Dict[str, Any]) -> str:
    """Generate connection assessment section."""
    section = """## Inter-Node Connection Assessment

Evaluates data flow stability and type compatibility between pipeline stages.

| Connection | Stability | Verdict | Notes |
|------------|-----------|---------|-------|
"""
    
    for conn_name, metrics in sorted(connections.items()):
        stability = metrics.get("stability", 0)
        mismatches = metrics.get("type_mismatches", [])
        
        if stability >= 0.99:
            verdict = "✓ EXCELLENT"
        elif stability >= 0.95:
            verdict = "✓ GOOD"
        elif stability >= 0.85:
            verdict = "⚠ ACCEPTABLE"
        else:
            verdict = "✗ UNSTABLE"
        
        notes = ""
        if mismatches:
            examples = ", ".join(mismatches[:3])
            if len(mismatches) > 3:
                examples += f", +{len(mismatches)-3} more"
            notes = f"Type mismatches: {examples}"
        else:
            notes = "No type mismatches detected"
        
        section += f"| {conn_name} | {stability*100:.1f}% | {verdict} | {notes} |\n"
    
    return section


def generate_output_quality_section(output_quality: Dict[str, Any]) -> str:
    """Generate final output quality assessment."""
    section = """## Final Output Quality Assessment

Verifies pipeline output integrity and acceptance criteria.

"""
    
    determinism = output_quality.get("determinism_verified", False)
    determinism_runs = output_quality.get("determinism_runs", {})
    coverage = output_quality.get("question_coverage", 0)
    rubric_status = output_quality.get("rubric_check_exit_code", None)
    rubric_aligned = output_quality.get("rubric_aligned", False)
    gates_passed = output_quality.get("all_gates_passed", False)
    failed_gates = output_quality.get("failed_gates", [])
    
    # Determinism
    section += "### Determinism Verification\n"
    section += f"**Status:** {'✓ PASS' if determinism else '✗ FAIL'}\n\n"
    if determinism_runs:
        section += "Multiple runs produced identical outputs:\n"
        for run_id, hash_val in determinism_runs.items():
            section += f"- {run_id}: `{hash_val}`\n"
    else:
        section += "No run data available for determinism check.\n"
    section += "\n"
    
    # Coverage
    section += "### Question Coverage\n"
    section += f"**Status:** {coverage}/300 questions {'✓ COMPLETE' if coverage == 300 else '⚠ PARTIAL'}\n\n"
    
    # Rubric alignment
    section += "### Rubric Alignment\n"
    section += f"**Status:** {'✓ PASS' if rubric_aligned else '✗ FAIL'}\n"
    if rubric_status is not None:
        section += f"**Exit Code:** {rubric_status} (tools/rubric_check.py)\n"
        if rubric_status == 0:
            section += "1:1 alignment verified between answers and rubric scoring.\n"
        elif rubric_status == 3:
            section += "Alignment mismatch detected. Check rubric_check output for missing/extra question IDs.\n"
        else:
            section += f"Rubric check encountered errors (exit code {rubric_status}).\n"
    section += "\n"
    
    # Gate passage
    section += "### Acceptance Gates\n"
    section += f"**Status:** {'✓ ALL GATES PASSED' if gates_passed else '✗ GATES FAILED'}\n\n"
    if not gates_passed and failed_gates:
        section += "Failed gates:\n"
        for gate in failed_gates:
            section += f"- {gate}\n"
    elif gates_passed:
        section += "All 6 acceptance gates passed successfully.\n"
    
    return section


def identify_top_risks(nodes: Dict[str, Any], connections: Dict[str, Any]) -> List[Tuple[int, str, str, str]]:
    """
    Identify top 5 risks ranked by severity.
    Returns: [(severity_score, risk_type, node/connection_name, description)]
    """
    risks = []
    
    # Node-level risks
    for node_name, metrics in nodes.items():
        status, reason = assess_node_health(metrics)
        
        if status == "FAIL":
            latency = metrics.get("latency_ms", 0)
            memory = metrics.get("peak_memory_mb", 0)
            error_rate = metrics.get("error_rate", 0)
            
            severity = 0
            if error_rate > 0.10:
                severity = 100
                risks.append((severity, "Critical Error Rate", node_name, 
                            f"Error rate {error_rate*100:.1f}% exceeds 10% threshold"))
            elif latency > 10000:
                severity = 90
                risks.append((severity, "Excessive Latency", node_name,
                            f"Latency {format_latency(latency/1000)} exceeds 10s threshold"))
            elif memory > 4096:
                severity = 85
                risks.append((severity, "Memory Exhaustion", node_name,
                            f"Peak memory {format_bytes(memory*1024**2)} exceeds 4GB"))
            else:
                severity = 70
                risks.append((severity, "Performance Degradation", node_name, reason))
        
        elif status == "WARN":
            risks.append((50, "Performance Warning", node_name, reason))
    
    # Connection-level risks
    for conn_name, metrics in connections.items():
        stability = metrics.get("stability", 0)
        mismatches = metrics.get("type_mismatches", [])
        
        if stability < 0.85:
            severity = 80
            risks.append((severity, "Unstable Data Flow", conn_name,
                        f"Stability {stability*100:.1f}% below 85% threshold"))
        
        if len(mismatches) > 5:
            severity = 75
            risks.append((severity, "Type Contract Violations", conn_name,
                        f"{len(mismatches)} type mismatches detected"))
    
    # Sort by severity and return top 5
    risks.sort(key=lambda x: x[0], reverse=True)
    return risks[:5]


def generate_top_risks_section(risks: List[Tuple[int, str, str, str]]) -> str:
    """Generate top 5 risks section."""
    section = """## Top 5 Risks

Critical bottlenecks and failures ranked by severity.

"""
    
    if not risks:
        section += "✓ No critical risks identified. All systems operating within acceptable parameters.\n"
        return section
    
    for i, (severity, risk_type, location, description) in enumerate(risks, 1):
        severity_label = "CRITICAL" if severity >= 80 else "HIGH" if severity >= 60 else "MEDIUM"
        section += f"### {i}. {risk_type} — {severity_label}\n"
        section += f"**Location:** `{location}`  \n"
        section += f"**Severity Score:** {severity}/100  \n"
        section += f"**Description:** {description}\n\n"
    
    return section


def generate_top_fixes_section(risks: List[Tuple[int, str, str, str]], nodes: Dict[str, Any]) -> str:
    """Generate top 5 fixes section with actionable recommendations."""
    section = """## Top 5 Recommended Fixes

Actionable recommendations to address identified risks.

"""
    
    if not risks:
        section += "✓ No fixes required. Consider performance optimization for future scalability.\n\n"
        section += "**Proactive Recommendations:**\n"
        section += "1. Implement response time budget alerts for early degradation detection\n"
        section += "2. Add capacity headroom monitoring (suggest scaling at 70% utilization)\n"
        section += "3. Set up synthetic monitoring to catch regressions pre-production\n"
        return section
    
    for i, (_severity, risk_type, location, _description) in enumerate(risks, 1):
        section += f"### {i}. Fix {risk_type}\n"
        section += f"**Target:** `{location}`\n\n"
        
        # Generate specific recommendations based on risk type
        if "Error Rate" in risk_type:
            section += "**Recommended Actions:**\n"
            section += f"- Review error logs for {location} to identify root cause\n"
            section += "- Add circuit breaker pattern with 3-retry limit and exponential backoff\n"
            section += "- Implement input validation to catch malformed data upstream\n"
            section += "- Add health check endpoint with automatic failover\n"
        
        elif "Latency" in risk_type:
            section += "**Recommended Actions:**\n"
            section += f"- Profile {location} to identify bottleneck operations\n"
            section += "- Implement caching layer for repeated computations (LRU cache, TTL=1800s)\n"
            section += "- Consider batching/parallelization for independent operations\n"
            section += "- Add timeout enforcement (e.g., 5s timeout with graceful degradation)\n"
        
        elif "Memory" in risk_type:
            section += "**Recommended Actions:**\n"
            section += f"- Implement streaming/chunked processing in {location}\n"
            section += "- Add memory profiling to identify leak sources\n"
            section += "- Set memory limits via resource constraints (e.g., 1GB soft limit, 2GB hard limit)\n"
            section += "- Consider offloading large data structures to disk/cache\n"
        
        elif "Unstable Data Flow" in risk_type:
            section += "**Recommended Actions:**\n"
            section += f"- Add retry logic with exponential backoff for {location}\n"
            section += "- Implement message queue (e.g., Celery/RabbitMQ) for reliable delivery\n"
            section += "- Add dead letter queue for failed messages\n"
            section += "- Monitor connection health with automatic reconnection\n"
        
        elif "Type Contract" in risk_type:
            section += "**Recommended Actions:**\n"
            section += f"- Enforce strict schema validation at {location} input/output\n"
            section += "- Add Pydantic models or dataclass contracts\n"
            section += "- Implement type checking in CI/CD pipeline (mypy --strict)\n"
            section += "- Add contract tests to catch mismatches early\n"
        
        else:
            section += "**Recommended Actions:**\n"
            section += f"- Investigate {location} performance metrics\n"
            section += "- Add monitoring and alerting for threshold violations\n"
            section += "- Review component configuration and scaling parameters\n"
            section += "- Implement graceful degradation fallback\n"
        
        section += "\n"
    
    return section


def generate_report(json_path: Path, output_path: Path) -> bool:
    """
    Generate Markdown report from JSON diagnostic data.
    Returns True if successful, False otherwise.
    """
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract sections
        nodes = data.get("nodes", {})
        connections = data.get("connections", {})
        output_quality = data.get("output_quality", {})
        
        # Build report sections
        report_sections = []
        
        # Title and metadata
        report_sections.append("# Pipeline Flux Diagnostic Report\n")
        report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_sections.append(f"**Source:** `{json_path}`\n\n")
        report_sections.append("---\n\n")
        
        # 1. Executive Summary
        report_sections.append(generate_executive_summary(data))
        report_sections.append("\n---\n\n")
        
        # 2. Node-by-Node Performance
        report_sections.append(generate_performance_table(nodes))
        report_sections.append("\n---\n\n")
        
        # 3. Connection Assessment
        report_sections.append(generate_connection_assessment(connections))
        report_sections.append("\n---\n\n")
        
        # 4. Output Quality
        report_sections.append(generate_output_quality_section(output_quality))
        report_sections.append("\n---\n\n")
        
        # 5. Top 5 Risks
        risks = identify_top_risks(nodes, connections)
        report_sections.append(generate_top_risks_section(risks))
        report_sections.append("\n---\n\n")
        
        # 6. Top 5 Fixes
        report_sections.append(generate_top_fixes_section(risks, nodes))
        
        # Write to file
        report_content = "".join(report_sections)
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"✓ Report generated: {output_path}")
        return True
        
    except FileNotFoundError:
        print(f"✗ Error: JSON file not found: {json_path}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON format: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Error generating report: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent
    json_path = repo_root / "reports" / "flux_diagnostic.json"
    output_path = repo_root / "reports" / "flux_diagnostic.md"
    
    # Allow command-line override
    if len(sys.argv) >= 2:
        json_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    
    success = generate_report(json_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
