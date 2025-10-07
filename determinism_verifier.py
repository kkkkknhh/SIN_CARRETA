#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determinism Verifier - Standalone Utility for MINIMINIMOON Reproducibility Testing
==================================================================================

Executes miniminimoon_orchestrator.py twice on the same input PDF, captures 
artifacts from both runs into separate timestamped directories, and performs
comprehensive determinism validation including:

- SHA-256 hashing of evidence_registry deterministic state
- SHA-256 hashing of all JSON artifacts (flow_runtime, answers_report, coverage_report)
- Byte-level comparison with sorted key normalization for JSON objects
- Line-level diffs for non-matching outputs
- Execution order validation
- Evidence hash comparison
- Answer content comparison

Exit Codes:
- 0: Perfect reproducibility (all hashes match)
- 4: Determinism violations detected (discrepancies found)
- 1: Execution errors (orchestrator failures, missing artifacts, etc.)

Directory Structure:
  artifacts/
    determinism_run_<timestamp>/
      run1/
        evidence_registry.json
        flow_runtime.json
        answers_report.json
        coverage_report.json
      run2/
        evidence_registry.json
        flow_runtime.json
        answers_report.json
        coverage_report.json
      determinism_report.json
      determinism_report.txt

Usage:
  python3 determinism_verifier.py <input_pdf_path> [--output-dir <dir>]
"""

import sys
import json
import hashlib
import logging
import shutil
import difflib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ArtifactComparison:
    """Comparison result for a single artifact file"""
    artifact_name: str
    run1_hash: str
    run2_hash: str
    match: bool
    run1_size: int
    run2_size: int
    diff_lines: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeterminismReport:
    """Complete determinism verification report"""
    timestamp: str
    input_pdf: str
    run1_dir: str
    run2_dir: str
    perfect_match: bool
    artifact_comparisons: List[ArtifactComparison]
    evidence_hash_run1: str
    evidence_hash_run2: str
    evidence_hash_match: bool
    flow_hash_run1: str
    flow_hash_run2: str
    flow_hash_match: bool
    execution_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "input_pdf": self.input_pdf,
            "run1_dir": self.run1_dir,
            "run2_dir": self.run2_dir,
            "perfect_match": self.perfect_match,
            "artifact_comparisons": [ac.to_dict() for ac in self.artifact_comparisons],
            "evidence_hash_run1": self.evidence_hash_run1,
            "evidence_hash_run2": self.evidence_hash_run2,
            "evidence_hash_match": self.evidence_hash_match,
            "flow_hash_run1": self.flow_hash_run1,
            "flow_hash_run2": self.flow_hash_run2,
            "flow_hash_match": self.flow_hash_match,
            "execution_errors": self.execution_errors
        }


class DeterminismVerifier:
    """Main determinism verification utility"""
    
    REQUIRED_ARTIFACTS = [
        "evidence_registry.json",
        "flow_runtime.json",
        "answers_report.json",
        "coverage_report.json"
    ]
    
    def __init__(self, input_pdf: Path, output_dir: Optional[Path] = None):
        """
        Initialize verifier with input PDF and output directory.
        
        Args:
            input_pdf: Path to input PDF file
            output_dir: Optional output directory (default: artifacts/determinism_run_<timestamp>)
        """
        self.input_pdf = Path(input_pdf).resolve()
        if not self.input_pdf.exists():
            raise FileNotFoundError(f"Input PDF not found: {self.input_pdf}")
        
        # Create timestamped output directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            artifacts_dir = Path("artifacts").resolve()
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = artifacts_dir / f"determinism_run_{timestamp}"
        
        self.run1_dir = self.output_dir / "run1"
        self.run2_dir = self.output_dir / "run2"
        
        logger.info(f"Determinism Verifier initialized")
        logger.info(f"  Input PDF: {self.input_pdf}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def run_orchestrator(self, run_dir: Path) -> Tuple[bool, str]:
        """
        Execute miniminimoon_orchestrator.py and capture output to run_dir.
        
        Args:
            run_dir: Directory to store orchestrator output
            
        Returns:
            (success: bool, error_message: str)
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Executing miniminimoon_orchestrator.py -> {run_dir}")
        
        # Construct command
        cmd = [
            sys.executable,
            "miniminimoon_orchestrator.py",
            str(self.input_pdf),
            "--output-dir", str(run_dir)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"Orchestrator exited with code {result.returncode}\nSTDERR:\n{result.stderr}"
                logger.error(error_msg)
                return False, error_msg
            
            # Verify required artifacts exist
            missing = []
            for artifact in self.REQUIRED_ARTIFACTS:
                artifact_path = run_dir / artifact
                if not artifact_path.exists():
                    missing.append(artifact)
            
            if missing:
                error_msg = f"Missing required artifacts: {missing}"
                logger.error(error_msg)
                return False, error_msg
            
            logger.info(f"✓ Orchestrator execution completed successfully")
            return True, ""
            
        except subprocess.TimeoutExpired:
            error_msg = "Orchestrator execution timed out (600s)"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Orchestrator execution failed: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file contents.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 hex digest
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def normalize_json(self, json_path: Path) -> bytes:
        """
        Load JSON, normalize with sorted keys, and return canonical bytes.
        
        This ensures that JSON files with identical content but different
        key ordering or formatting are recognized as identical.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Canonical JSON bytes (sorted keys, no whitespace)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Remove non-deterministic fields
        self._remove_nondeterministic_fields(data)
        
        # Serialize with deterministic ordering
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
        return canonical.encode('utf-8')
    
    def _remove_nondeterministic_fields(self, obj: Any) -> None:
        """
        Recursively remove non-deterministic fields from JSON object.
        
        Removes:
        - Timestamps (except structural ones needed for ordering)
        - Execution durations
        - Absolute file paths
        - Runtime-specific metadata
        
        Modifies obj in-place.
        """
        if isinstance(obj, dict):
            # Remove non-deterministic keys
            nondeterministic_keys = [
                'timestamp', 'execution_time', 'duration_seconds',
                'start_time', 'end_time', 'stage_timestamps',
                'creation_time', 'absolute_path'
            ]
            for key in nondeterministic_keys:
                obj.pop(key, None)
            
            # Recursively process nested objects
            for value in obj.values():
                self._remove_nondeterministic_fields(value)
        
        elif isinstance(obj, list):
            for item in obj:
                self._remove_nondeterministic_fields(item)
    
    def compute_normalized_hash(self, json_path: Path) -> str:
        """
        Compute SHA-256 hash of normalized JSON.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            SHA-256 hex digest of normalized JSON
        """
        normalized = self.normalize_json(json_path)
        return hashlib.sha256(normalized).hexdigest()
    
    def compare_artifacts(self, artifact_name: str) -> ArtifactComparison:
        """
        Compare a single artifact between run1 and run2.
        
        Args:
            artifact_name: Name of artifact file (e.g., "evidence_registry.json")
            
        Returns:
            ArtifactComparison object with detailed comparison results
        """
        run1_path = self.run1_dir / artifact_name
        run2_path = self.run2_dir / artifact_name
        
        # Compute normalized hashes
        run1_hash = self.compute_normalized_hash(run1_path)
        run2_hash = self.compute_normalized_hash(run2_path)
        
        match = (run1_hash == run2_hash)
        
        # Get file sizes
        run1_size = run1_path.stat().st_size
        run2_size = run2_path.stat().st_size
        
        # Generate diff if not matching
        diff_lines = []
        if not match:
            diff_lines = self.generate_json_diff(run1_path, run2_path)
        
        comparison = ArtifactComparison(
            artifact_name=artifact_name,
            run1_hash=run1_hash,
            run2_hash=run2_hash,
            match=match,
            run1_size=run1_size,
            run2_size=run2_size,
            diff_lines=diff_lines
        )
        
        if match:
            logger.info(f"  ✓ {artifact_name}: MATCH (hash={run1_hash[:16]}...)")
        else:
            logger.error(f"  ✗ {artifact_name}: MISMATCH")
            logger.error(f"    Run1 hash: {run1_hash}")
            logger.error(f"    Run2 hash: {run2_hash}")
        
        return comparison
    
    def generate_json_diff(self, file1: Path, file2: Path, context_lines: int = 5) -> List[str]:
        """
        Generate line-level diff for JSON files.
        
        Args:
            file1: First JSON file
            file2: Second JSON file
            context_lines: Number of context lines around differences
            
        Returns:
            List of diff lines
        """
        # Load and pretty-print both files for readable diff
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        # Remove non-deterministic fields
        self._remove_nondeterministic_fields(data1)
        self._remove_nondeterministic_fields(data2)
        
        # Pretty-print with sorted keys
        json1_lines = json.dumps(data1, indent=2, sort_keys=True, ensure_ascii=False).splitlines()
        json2_lines = json.dumps(data2, indent=2, sort_keys=True, ensure_ascii=False).splitlines()
        
        # Generate unified diff
        diff = difflib.unified_diff(
            json1_lines,
            json2_lines,
            fromfile=f"run1/{file1.name}",
            tofile=f"run2/{file2.name}",
            lineterm='',
            n=context_lines
        )
        
        return list(diff)
    
    def extract_evidence_hash(self, evidence_registry_path: Path) -> str:
        """
        Extract deterministic_hash from evidence_registry.json.
        
        Args:
            evidence_registry_path: Path to evidence_registry.json
            
        Returns:
            Deterministic hash value
        """
        with open(evidence_registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("deterministic_hash", "")
    
    def extract_flow_hash(self, flow_runtime_path: Path) -> str:
        """
        Extract flow_hash from flow_runtime.json.
        
        Args:
            flow_runtime_path: Path to flow_runtime.json
            
        Returns:
            Flow hash value
        """
        with open(flow_runtime_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("flow_hash", "")
    
    def verify_determinism(self) -> DeterminismReport:
        """
        Execute both runs and perform comprehensive determinism verification.
        
        Returns:
            DeterminismReport with complete comparison results
        """
        logger.info("=" * 80)
        logger.info("DETERMINISM VERIFICATION STARTED")
        logger.info("=" * 80)
        
        execution_errors = []
        
        # Run 1
        logger.info("\n--- RUN 1 ---")
        success1, error1 = self.run_orchestrator(self.run1_dir)
        if not success1:
            execution_errors.append(f"Run 1 failed: {error1}")
        
        # Run 2
        logger.info("\n--- RUN 2 ---")
        success2, error2 = self.run_orchestrator(self.run2_dir)
        if not success2:
            execution_errors.append(f"Run 2 failed: {error2}")
        
        # If either run failed, return early with error report
        if execution_errors:
            logger.error("\n" + "=" * 80)
            logger.error("EXECUTION ERRORS DETECTED")
            logger.error("=" * 80)
            for error in execution_errors:
                logger.error(f"  {error}")
            
            return DeterminismReport(
                timestamp=datetime.utcnow().isoformat(),
                input_pdf=str(self.input_pdf),
                run1_dir=str(self.run1_dir),
                run2_dir=str(self.run2_dir),
                perfect_match=False,
                artifact_comparisons=[],
                evidence_hash_run1="",
                evidence_hash_run2="",
                evidence_hash_match=False,
                flow_hash_run1="",
                flow_hash_run2="",
                flow_hash_match=False,
                execution_errors=execution_errors
            )
        
        # Compare artifacts
        logger.info("\n--- ARTIFACT COMPARISON ---")
        comparisons = []
        for artifact in self.REQUIRED_ARTIFACTS:
            comparison = self.compare_artifacts(artifact)
            comparisons.append(comparison)
        
        # Extract evidence hashes
        evidence_hash_run1 = self.extract_evidence_hash(self.run1_dir / "evidence_registry.json")
        evidence_hash_run2 = self.extract_evidence_hash(self.run2_dir / "evidence_registry.json")
        evidence_hash_match = (evidence_hash_run1 == evidence_hash_run2)
        
        logger.info(f"\n--- EVIDENCE HASH COMPARISON ---")
        if evidence_hash_match:
            logger.info(f"  ✓ Evidence hashes MATCH: {evidence_hash_run1}")
        else:
            logger.error(f"  ✗ Evidence hashes MISMATCH")
            logger.error(f"    Run1: {evidence_hash_run1}")
            logger.error(f"    Run2: {evidence_hash_run2}")
        
        # Extract flow hashes
        flow_hash_run1 = self.extract_flow_hash(self.run1_dir / "flow_runtime.json")
        flow_hash_run2 = self.extract_flow_hash(self.run2_dir / "flow_runtime.json")
        flow_hash_match = (flow_hash_run1 == flow_hash_run2)
        
        logger.info(f"\n--- FLOW HASH COMPARISON ---")
        if flow_hash_match:
            logger.info(f"  ✓ Flow hashes MATCH: {flow_hash_run1}")
        else:
            logger.error(f"  ✗ Flow hashes MISMATCH")
            logger.error(f"    Run1: {flow_hash_run1}")
            logger.error(f"    Run2: {flow_hash_run2}")
        
        # Determine overall result
        all_artifacts_match = all(c.match for c in comparisons)
        perfect_match = all_artifacts_match and evidence_hash_match and flow_hash_match
        
        # Create report
        report = DeterminismReport(
            timestamp=datetime.utcnow().isoformat(),
            input_pdf=str(self.input_pdf),
            run1_dir=str(self.run1_dir),
            run2_dir=str(self.run2_dir),
            perfect_match=perfect_match,
            artifact_comparisons=comparisons,
            evidence_hash_run1=evidence_hash_run1,
            evidence_hash_run2=evidence_hash_run2,
            evidence_hash_match=evidence_hash_match,
            flow_hash_run1=flow_hash_run1,
            flow_hash_run2=flow_hash_run2,
            flow_hash_match=flow_hash_match,
            execution_errors=execution_errors
        )
        
        # Log summary
        logger.info("\n" + "=" * 80)
        if perfect_match:
            logger.info("✓ PERFECT REPRODUCIBILITY - All artifacts match")
        else:
            logger.error("✗ DETERMINISM VIOLATIONS DETECTED")
            mismatches = [c.artifact_name for c in comparisons if not c.match]
            if mismatches:
                logger.error(f"  Mismatched artifacts: {mismatches}")
            if not evidence_hash_match:
                logger.error(f"  Evidence hash mismatch")
            if not flow_hash_match:
                logger.error(f"  Flow hash mismatch")
        logger.info("=" * 80)
        
        return report
    
    def export_report(self, report: DeterminismReport):
        """
        Export determinism report to JSON and human-readable text.
        
        Args:
            report: DeterminismReport to export
        """
        # Export JSON report
        json_path = self.output_dir / "determinism_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"\n✓ JSON report exported: {json_path}")
        
        # Export human-readable text report
        txt_path = self.output_dir / "determinism_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            self._write_text_report(f, report)
        logger.info(f"✓ Text report exported: {txt_path}")
    
    def _write_text_report(self, f, report: DeterminismReport):
        """Write human-readable text report"""
        f.write("=" * 80 + "\n")
        f.write("DETERMINISM VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {report.timestamp}\n")
        f.write(f"Input PDF: {report.input_pdf}\n")
        f.write(f"Run 1 Directory: {report.run1_dir}\n")
        f.write(f"Run 2 Directory: {report.run2_dir}\n\n")
        
        # Overall result
        f.write("=" * 80 + "\n")
        if report.perfect_match:
            f.write("RESULT: ✓ PERFECT REPRODUCIBILITY\n")
        else:
            f.write("RESULT: ✗ DETERMINISM VIOLATIONS DETECTED\n")
        f.write("=" * 80 + "\n\n")
        
        # Execution errors
        if report.execution_errors:
            f.write("EXECUTION ERRORS:\n")
            for error in report.execution_errors:
                f.write(f"  - {error}\n")
            f.write("\n")
        
        # Evidence hash comparison
        f.write("EVIDENCE HASH COMPARISON:\n")
        f.write(f"  Run 1: {report.evidence_hash_run1}\n")
        f.write(f"  Run 2: {report.evidence_hash_run2}\n")
        if report.evidence_hash_match:
            f.write(f"  Status: ✓ MATCH\n\n")
        else:
            f.write(f"  Status: ✗ MISMATCH\n\n")
        
        # Flow hash comparison
        f.write("FLOW HASH COMPARISON:\n")
        f.write(f"  Run 1: {report.flow_hash_run1}\n")
        f.write(f"  Run 2: {report.flow_hash_run2}\n")
        if report.flow_hash_match:
            f.write(f"  Status: ✓ MATCH\n\n")
        else:
            f.write(f"  Status: ✗ MISMATCH\n\n")
        
        # Artifact comparisons
        f.write("ARTIFACT COMPARISONS:\n")
        f.write("-" * 80 + "\n")
        for comp in report.artifact_comparisons:
            f.write(f"\nArtifact: {comp.artifact_name}\n")
            f.write(f"  Run 1 Hash: {comp.run1_hash}\n")
            f.write(f"  Run 2 Hash: {comp.run2_hash}\n")
            f.write(f"  Run 1 Size: {comp.run1_size} bytes\n")
            f.write(f"  Run 2 Size: {comp.run2_size} bytes\n")
            if comp.match:
                f.write(f"  Status: ✓ MATCH\n")
            else:
                f.write(f"  Status: ✗ MISMATCH\n")
                if comp.diff_lines:
                    f.write(f"\n  Diff (first 100 lines):\n")
                    for line in comp.diff_lines[:100]:
                        f.write(f"    {line}\n")
                    if len(comp.diff_lines) > 100:
                        f.write(f"    ... ({len(comp.diff_lines) - 100} more lines)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def main():
    """Main entry point for determinism verifier"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify deterministic execution of miniminimoon_orchestrator.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify determinism for a PDF (output to artifacts/determinism_run_<timestamp>)
  python3 determinism_verifier.py input.pdf
  
  # Verify with custom output directory
  python3 determinism_verifier.py input.pdf --output-dir /tmp/determinism_test

Exit Codes:
  0 - Perfect reproducibility (all artifacts match)
  4 - Determinism violations detected
  1 - Execution errors (orchestrator failures, missing artifacts, etc.)
        """
    )
    
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to input PDF file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for comparison artifacts (default: artifacts/determinism_run_<timestamp>)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create verifier
        verifier = DeterminismVerifier(args.input_pdf, args.output_dir)
        
        # Run verification
        report = verifier.verify_determinism()
        
        # Export report
        verifier.export_report(report)
        
        # Determine exit code
        if report.execution_errors:
            logger.error("\nExiting with code 1 (execution errors)")
            sys.exit(1)
        elif report.perfect_match:
            logger.info("\nExiting with code 0 (perfect reproducibility)")
            sys.exit(0)
        else:
            logger.error("\nExiting with code 4 (determinism violations)")
            sys.exit(4)
    
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
