#!/usr/bin/env python3
"""
Flux Diagnostic Report Generator

Consumes instrumentation data from diagnostic_runner.py and connection_stability_analyzer.py
to produce a structured machine-readable diagnostic report.

Exit codes:
    0: Report generated successfully
    1: Instrumentation data incomplete or malformed
    2: Validation failed (missing stages or contracts)
    3: Environment detection failed
"""

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FluxDiagnosticGenerator:
    """Generates comprehensive diagnostic reports from pipeline instrumentation data"""
    
    # Expected pipeline stages from flow_doc.json
    EXPECTED_STAGES = [
        "document_segmentation",
        "embedding_generation",
        "semantic_search",
        "evidence_extraction",
        "responsibility_detection",
        "temporal_analysis",
        "causal_graph_construction",
        "intervention_identification",
        "outcome_mapping",
        "rubric_alignment",
        "scoring_computation",
        "contract_validation",
        "determinism_verification",
        "coverage_validation",
        "report_generation"
    ]
    
    # Expected flow contracts (subset of 72 documented)
    EXPECTED_CONTRACT_CATEGORIES = [
        "interface_contracts",
        "data_flow_contracts",
        "stability_contracts",
        "determinism_contracts",
        "performance_contracts"
    ]
    
    def __init__(self, 
                 diagnostic_data: Optional[Dict[str, Any]] = None,
                 connection_data: Optional[Dict[str, Any]] = None,
                 determinism_data: Optional[Dict[str, Any]] = None,
                 rubric_exit_code: Optional[int] = None):
        """
        Initialize generator with instrumentation data
        
        Args:
            diagnostic_data: Per-stage metrics from diagnostic_runner.py
            connection_data: Inter-node flow data from connection_stability_analyzer.py
            determinism_data: Verification results from determinism_verifier.py
            rubric_exit_code: Exit code from rubric_check.py
        """
        self.diagnostic_data = diagnostic_data or {}
        self.connection_data = connection_data or {}
        self.determinism_data = determinism_data or {}
        self.rubric_exit_code = rubric_exit_code
        self.errors = []
        
    def validate_instrumentation_data(self) -> bool:
        """Validate that instrumentation data is complete and well-formed"""
        
        # Validate pipeline stages
        if "pipeline_stages" not in self.diagnostic_data:
            self.errors.append("Missing 'pipeline_stages' in diagnostic data")
            return False
            
        stages = self.diagnostic_data.get("pipeline_stages", [])
        if not isinstance(stages, list):
            self.errors.append("'pipeline_stages' must be a list")
            return False
            
        # Check for required stage fields
        for stage in stages:
            required_fields = ["stage_name", "latency_ms", "cpu_ms", "peak_memory_mb", 
                             "throughput", "cache_hits", "contract_checks"]
            missing = [f for f in required_fields if f not in stage]
            if missing:
                self.errors.append(f"Stage '{stage.get('stage_name', 'unknown')}' missing fields: {missing}")
                return False
                
            # Validate contract_checks structure
            if not isinstance(stage.get("contract_checks"), dict):
                self.errors.append(f"Stage '{stage['stage_name']}' contract_checks must be a dict")
                return False
                
            if "pass" not in stage["contract_checks"] or "fail" not in stage["contract_checks"]:
                self.errors.append(f"Stage '{stage['stage_name']}' contract_checks missing pass/fail")
                return False
        
        # Validate all expected stages are present
        stage_names = {s["stage_name"] for s in stages}
        missing_stages = set(self.EXPECTED_STAGES) - stage_names
        if missing_stages:
            self.errors.append(f"Missing expected stages: {sorted(missing_stages)}")
            return False
            
        # Validate connections
        if "connections" not in self.connection_data:
            self.errors.append("Missing 'connections' in connection data")
            return False
            
        connections = self.connection_data.get("connections", [])
        if not isinstance(connections, list):
            self.errors.append("'connections' must be a list")
            return False
            
        for conn in connections:
            required_fields = ["from_node", "to_node", "interface_check", 
                             "stability_rate", "suitability_verdict", "mismatch_examples"]
            missing = [f for f in required_fields if f not in conn]
            if missing:
                self.errors.append(f"Connection '{conn.get('from_node', '?')}->{conn.get('to_node', '?')}' missing: {missing}")
                return False
                
            # Validate verdict enum
            valid_verdicts = ["SUITABLE", "UNSTABLE", "INCOMPATIBLE", "DEGRADED", "UNKNOWN"]
            if conn.get("suitability_verdict") not in valid_verdicts:
                self.errors.append(f"Invalid suitability_verdict: {conn.get('suitability_verdict')}")
                return False
                
        # Validate connection coverage (72 documented flow contracts)
        if len(connections) < 72:
            self.errors.append(f"Expected 72 flow contracts, found {len(connections)}")
            return False
            
        return True
        
    @staticmethod
    def collect_environment_metadata() -> Dict[str, Any]:
        """Collect environment information including git hash, Python version, etc."""
        env = {}
        
        try:
            # Get git commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env["repo_commit_hash"] = result.stdout.strip()
            else:
                env["repo_commit_hash"] = "unknown"
        except Exception as e:
            env["repo_commit_hash"] = f"error: {str(e)}"
            
        # Python version
        env["python_version"] = platform.python_version()
        
        # OS platform
        env["os_platform"] = platform.platform()
        
        # Key library versions
        env["library_versions"] = {}
        key_libraries = [
            "sentence-transformers", "scikit-learn", "numpy", 
            "spacy", "pytest", "torch", "transformers"
        ]
        
        for lib in key_libraries:
            try:
                version = importlib.metadata.version(lib)
                env["library_versions"][lib] = version
            except importlib.metadata.PackageNotFoundError:
                env["library_versions"][lib] = "not_installed"
                
        # Execution timestamp
        env["execution_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return env
        
    def generate_final_output_section(self) -> Dict[str, Any]:
        """Generate final_output section aggregating validation results"""
        
        # Determinism verification
        determinism_verified = self.determinism_data.get("determinism_verified", False)
        
        # Coverage validation (300 rule requirement)
        coverage_300 = self.diagnostic_data.get("coverage_300", {
            "met": False,
            "actual_count": 0,
            "required_count": 300
        })
        
        # Rubric alignment status from exit code
        rubric_status_map = {
            0: "ALIGNED",
            1: "ERROR",
            2: "MISSING_FILES",
            3: "MISALIGNED"
        }
        rubric_alignment = rubric_status_map.get(
            self.rubric_exit_code if self.rubric_exit_code is not None else -1,
            "UNKNOWN"
        )
        
        # Quality metrics
        quality_metrics = self.diagnostic_data.get("quality_metrics", {
            "confidence_scores": {
                "overall": 0.0,
                "evidence_extraction": 0.0,
                "responsibility_detection": 0.0,
                "intervention_mapping": 0.0
            },
            "evidence_completeness": {
                "percentage": 0.0,
                "missing_categories": []
            }
        })
        
        return {
            "determinism_verified": determinism_verified,
            "coverage_300": coverage_300,
            "rubric_alignment": rubric_alignment,
            "quality_metrics": quality_metrics
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete diagnostic report"""
        
        # Validate input data
        if not self.validate_instrumentation_data():
            raise ValueError(f"Instrumentation data validation failed: {'; '.join(self.errors)}")
            
        # Build report structure
        report = {
            "pipeline_stages": self.diagnostic_data.get("pipeline_stages", []),
            "connections": self.connection_data.get("connections", []),
            "final_output": self.generate_final_output_section(),
            "environment": self.collect_environment_metadata()
        }
        
        return report
        
    def save_report(self, output_path: Path) -> None:
        """Generate and save report to JSON file"""
        report = self.generate_report()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


def load_diagnostic_data() -> Dict[str, Any]:
    """Load pipeline diagnostic data from diagnostic_runner.py output"""
    repo_root = Path(__file__).parent.parent
    diagnostic_file = repo_root / "artifacts" / "diagnostic_data.json"
    
    if not diagnostic_file.exists():
        return {}
        
    with open(diagnostic_file) as f:
        return json.load(f)


def load_connection_data() -> Dict[str, Any]:
    """Load connection stability data from connection_stability_analyzer.py output"""
    repo_root = Path(__file__).parent.parent
    connection_file = repo_root / "artifacts" / "connection_data.json"
    
    if not connection_file.exists():
        return {}
        
    with open(connection_file) as f:
        return json.load(f)


def load_determinism_data() -> Dict[str, Any]:
    """Load determinism verification data from determinism_verifier.py output"""
    repo_root = Path(__file__).parent.parent
    determinism_file = repo_root / "artifacts" / "determinism_data.json"
    
    if not determinism_file.exists():
        return {}
        
    with open(determinism_file) as f:
        return json.load(f)


def get_rubric_exit_code() -> int:
    """Execute rubric_check.py and capture exit code"""
    repo_root = Path(__file__).parent.parent
    rubric_script = repo_root / "tools" / "rubric_check.py"
    
    if not rubric_script.exists():
        return -1
        
    try:
        result = subprocess.run(
            [sys.executable, str(rubric_script)],
            capture_output=True,
            timeout=30
        )
        return result.returncode
    except Exception:
        return -1


def main():
    """Main entry point for diagnostic report generation"""
    try:
        # Load all instrumentation data
        diagnostic_data = load_diagnostic_data()
        connection_data = load_connection_data()
        determinism_data = load_determinism_data()
        rubric_exit_code = get_rubric_exit_code()
        
        # Generate report
        generator = FluxDiagnosticGenerator(
            diagnostic_data=diagnostic_data,
            connection_data=connection_data,
            determinism_data=determinism_data,
            rubric_exit_code=rubric_exit_code
        )
        
        # Save to output file
        output_path = Path(__file__).parent / "flux_diagnostic.json"
        generator.save_report(output_path)
        
        print(json.dumps({
            "status": "success",
            "output_file": str(output_path),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        
        return 0
        
    except ValueError as e:
        print(json.dumps({
            "status": "validation_failed",
            "error": str(e)
        }), file=sys.stderr)
        return 2
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
