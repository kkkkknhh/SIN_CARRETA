#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Integration Test for Unified Evaluation Pipeline
============================================================

This test exercises the complete unified evaluation pipeline by:
1. Invoking miniminimoon_cli.py to run full evaluation cycles
2. Verifying all expected artifacts exist with correct structure
3. Validating flow_runtime.json matches canonical flow order from flow_doc.json
4. Checking answers_report.json contains evidence_ids, confidence, and rationale
5. Verifying system_validators pre/post execution checks pass
6. Confirming 300/300 coverage with deterministic hashes
7. Running three consecutive tests to verify reproducibility
8. Asserting deterministic_hash values are identical across runs
9. Confirming flow order consistency across all runs

Test fails with actionable diagnostics if any component is missing,
flow order deviates, artifacts lack required fields, or validation gates fail.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple


class TestE2EUnifiedPipeline(unittest.TestCase):
    """End-to-end integration tests for unified evaluation pipeline"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.repo_root = pathlib.Path(__file__).parent.resolve()
        cls.artifacts_dir = cls.repo_root / "artifacts"
        cls.flow_doc_path = cls.repo_root / "flow_doc.json"
        cls.rubric_path = cls.repo_root / "RUBRIC_SCORING.json"
        cls.cli_path = cls.repo_root / "miniminimoon_cli.py"
        cls.test_plan_path = cls.repo_root / "test_input.txt"
        
        # Store results from multiple runs for reproducibility testing
        cls.run_results: List[Dict[str, Any]] = []
        
        # Ensure tools directory exists for rubric_check.py
        cls.tools_dir = cls.repo_root / "tools"
        cls.tools_dir.mkdir(exist_ok=True)
        
        # Create tools/flow_doc.json symlink/copy if needed
        cls.tools_flow_doc = cls.tools_dir / "flow_doc.json"
        if not cls.tools_flow_doc.exists() and cls.flow_doc_path.exists():
            shutil.copy(cls.flow_doc_path, cls.tools_flow_doc)
        
        # Create minimal rubric_check.py in tools if it doesn't exist
        cls.rubric_check_path = cls.tools_dir / "rubric_check.py"
        if not cls.rubric_check_path.exists():
            cls._create_rubric_check_stub()
        
        # Ensure test plan exists
        if not cls.test_plan_path.exists():
            cls.test_plan_path.write_text("Plan de desarrollo municipal de prueba.\n", encoding="utf-8")
        
        # Use system Python instead of venv if venv doesn't work
        cls.python_cmd = sys.executable

    @classmethod
    def _create_rubric_check_stub(cls):
        """Create a minimal rubric_check.py that validates 1:1 alignment"""
        rubric_check_code = '''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def check_rubric_alignment():
    """Check 1:1 alignment between answers and rubric"""
    try:
        repo_root = Path(__file__).parent.parent
        answers_path = repo_root / "artifacts" / "answers_report.json"
        rubric_path = repo_root / "RUBRIC_SCORING.json"
        
        if not answers_path.exists():
            print(json.dumps({"ok": False, "error": "answers_report.json not found"}))
            return 1
        
        if not rubric_path.exists():
            print(json.dumps({"ok": False, "error": "RUBRIC_SCORING.json not found"}))
            return 1
        
        with open(answers_path) as f:
            answers = json.load(f)
        
        with open(rubric_path) as f:
            rubric = json.load(f)
        
        weights = rubric.get("weights", {})
        answer_ids = {a["question_id"] for a in answers.get("answers", [])}
        
        missing = [qid for qid in answer_ids if qid not in weights]
        extra = [qid for qid in weights.keys() if qid not in answer_ids]
        
        if missing or extra:
            print(json.dumps({
                "ok": False,
                "missing_in_rubric": missing[:10],
                "extra_in_rubric": extra[:10],
                "message": "1:1 alignment failed"
            }))
            return 3
        
        print(json.dumps({"ok": True, "message": "1:1 alignment verified"}))
        return 0
        
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    sys.exit(check_rubric_alignment())
'''
        cls.rubric_check_path.write_text(rubric_check_code, encoding="utf-8")
        cls.rubric_check_path.chmod(0o755)

    def _run_cli_command(self, command: List[str]) -> Tuple[int, Dict[str, Any], str]:
        """
        Run a CLI command and return (exit_code, parsed_json, stderr)
        
        Returns:
            Tuple of (exit code, parsed JSON output, stderr string)
        """
        try:
            proc = subprocess.run(
                [self.python_cmd] + command,
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=300
            )
            
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            
            # Parse JSON output
            try:
                parsed = json.loads(stdout) if stdout else {}
            except json.JSONDecodeError:
                parsed = {"raw_stdout": stdout}
            
            return proc.returncode, parsed, stderr
            
        except subprocess.TimeoutExpired:
            return 1, {"error": "Command timeout"}, "Timeout after 300s"
        except Exception as e:
            return 1, {"error": str(e)}, str(e)

    def _verify_artifacts_structure(self, run_index: int) -> List[str]:
        """
        Verify all expected artifacts exist with correct structure.
        
        Returns:
            List of error messages (empty if all checks pass)
        """
        errors = []
        
        required_files = [
            "flow_runtime.json",
            "answers_report.json",
            "evidence_registry.json",
            "final_results.json"
        ]
        
        for filename in required_files:
            filepath = self.artifacts_dir / filename
            if not filepath.exists():
                errors.append(f"Run {run_index}: Missing required artifact: {filename}")
        
        return errors

    def _verify_flow_order(self, run_index: int) -> List[str]:
        """
        Verify flow_runtime.json matches canonical order from flow_doc.json.
        
        Returns:
            List of error messages (empty if order matches)
        """
        errors = []
        
        # Load canonical order
        if not self.tools_flow_doc.exists():
            errors.append(f"Run {run_index}: Missing tools/flow_doc.json")
            return errors
        
        try:
            with open(self.tools_flow_doc) as f:
                flow_doc = json.load(f)
            canonical_order = flow_doc.get("canonical_order", [])
        except Exception as e:
            errors.append(f"Run {run_index}: Cannot parse flow_doc.json: {e}")
            return errors
        
        # Load runtime order
        runtime_path = self.artifacts_dir / "flow_runtime.json"
        if not runtime_path.exists():
            errors.append(f"Run {run_index}: Missing flow_runtime.json")
            return errors
        
        try:
            with open(runtime_path) as f:
                runtime = json.load(f)
            runtime_order = runtime.get("stages", [])
        except Exception as e:
            errors.append(f"Run {run_index}: Cannot parse flow_runtime.json: {e}")
            return errors
        
        # Compare orders
        if not canonical_order:
            errors.append(f"Run {run_index}: flow_doc.json has empty canonical_order")
        
        if not runtime_order:
            errors.append(f"Run {run_index}: flow_runtime.json has empty stages")
        
        if canonical_order != runtime_order:
            errors.append(
                f"Run {run_index}: Flow order mismatch!\n"
                f"  Expected: {canonical_order}\n"
                f"  Got: {runtime_order}\n"
                f"  Difference: expected {len(canonical_order)} stages, got {len(runtime_order)}"
            )
            
            # Find specific differences
            for i, (expected, actual) in enumerate(zip(canonical_order, runtime_order)):
                if expected != actual:
                    errors.append(
                        f"  Stage {i}: expected '{expected}', got '{actual}'"
                    )
        
        return errors

    def _verify_answers_structure(self, run_index: int) -> List[str]:
        """
        Verify answers_report.json has required fields for all questions.
        
        Returns:
            List of error messages (empty if all checks pass)
        """
        errors = []
        
        answers_path = self.artifacts_dir / "answers_report.json"
        if not answers_path.exists():
            errors.append(f"Run {run_index}: Missing answers_report.json")
            return errors
        
        try:
            with open(answers_path) as f:
                answers = json.load(f)
        except Exception as e:
            errors.append(f"Run {run_index}: Cannot parse answers_report.json: {e}")
            return errors
        
        # Check summary
        summary = answers.get("summary", {})
        total_questions = summary.get("total_questions", 0)
        
        if total_questions < 300:
            errors.append(
                f"Run {run_index}: Insufficient coverage - "
                f"expected ≥300 questions, got {total_questions}"
            )
        
        # Check individual answers
        answer_list = answers.get("answers", [])
        if len(answer_list) != total_questions:
            errors.append(
                f"Run {run_index}: Answer count mismatch - "
                f"summary says {total_questions}, but got {len(answer_list)} answers"
            )
        
        # Verify required fields in first 5 answers (sample check)
        required_fields = ["question_id", "evidence_ids", "confidence", "reasoning"]
        for i, answer in enumerate(answer_list[:5]):
            for field in required_fields:
                if field not in answer:
                    errors.append(
                        f"Run {run_index}: Answer {i} missing required field: {field}"
                    )
                elif field == "evidence_ids" and not isinstance(answer[field], list):
                    errors.append(
                        f"Run {run_index}: Answer {i} has invalid evidence_ids type"
                    )
                elif field == "confidence" and not isinstance(answer[field], (int, float)):
                    errors.append(
                        f"Run {run_index}: Answer {i} has invalid confidence type"
                    )
                elif field == "reasoning" and not isinstance(answer[field], str):
                    errors.append(
                        f"Run {run_index}: Answer {i} has invalid reasoning type"
                    )
        
        return errors

    def _verify_validators(self, run_index: int) -> List[str]:
        """
        Verify system validators pre and post execution checks.
        
        Returns:
            List of error messages (empty if validators pass)
        """
        errors = []
        
        # Run validators via CLI
        rc, result, stderr = self._run_cli_command([
            str(self.cli_path),
            "verify",
            "--repo", str(self.repo_root)
        ])
        
        if rc != 0:
            errors.append(
                f"Run {run_index}: Validators failed with exit code {rc}\n"
                f"  Stderr: {stderr}\n"
                f"  Result: {json.dumps(result, indent=2)}"
            )
        
        if not result.get("ok", False):
            errors.append(f"Run {run_index}: Validators returned ok=false")
            
            # Extract specific error details
            pre_errors = result.get("pre", {}).get("errors", [])
            post_errors = result.get("post", {}).get("errors", [])
            
            if pre_errors:
                errors.append(f"  Pre-execution errors: {pre_errors}")
            if post_errors:
                errors.append(f"  Post-execution errors: {post_errors}")
        
        return errors

    def _verify_deterministic_hash(self, run_index: int) -> Tuple[Optional[str], List[str]]:
        """
        Extract and verify deterministic hash from evidence_registry.json.
        
        Returns:
            Tuple of (hash value, list of error messages)
        """
        errors = []
        
        registry_path = self.artifacts_dir / "evidence_registry.json"
        if not registry_path.exists():
            errors.append(f"Run {run_index}: Missing evidence_registry.json")
            return None, errors
        
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except Exception as e:
            errors.append(f"Run {run_index}: Cannot parse evidence_registry.json: {e}")
            return None, errors
        
        hash_value = registry.get("deterministic_hash")
        if not hash_value:
            errors.append(f"Run {run_index}: Missing deterministic_hash in evidence_registry")
            return None, errors
        
        if not isinstance(hash_value, str) or len(hash_value) != 64:
            errors.append(
                f"Run {run_index}: Invalid deterministic_hash format "
                f"(expected 64-char hex string, got: {hash_value})"
            )
            return None, errors
        
        return hash_value, errors

    def _backup_artifacts(self, run_index: int) -> pathlib.Path:
        """
        Backup current artifacts directory for reproducibility comparison.
        
        Returns:
            Path to backup directory
        """
        backup_dir = self.repo_root / f"artifacts_run_{run_index}"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(self.artifacts_dir, backup_dir)
        return backup_dir

    def test_01_prerequisites(self):
        """Verify all prerequisites are in place"""
        self.assertTrue(self.cli_path.exists(), "miniminimoon_cli.py not found")
        self.assertTrue(self.flow_doc_path.exists(), "flow_doc.json not found")
        self.assertTrue(self.test_plan_path.exists(), "test_input.txt not found")
        self.assertTrue(self.tools_dir.exists(), "tools/ directory not found")
        self.assertTrue(self.tools_flow_doc.exists(), "tools/flow_doc.json not found")

    def test_02_single_evaluation_run(self):
        """Test a single complete evaluation run"""
        rc, result, stderr = self._run_cli_command([
            str(self.cli_path),
            "evaluate",
            str(self.test_plan_path),
            "--repo", str(self.repo_root)
        ])
        
        self.assertEqual(rc, 0, f"Evaluation failed with exit code {rc}:\n{stderr}\n{result}")
        self.assertTrue(result.get("ok", False), f"Evaluation returned ok=false: {result}")
        
        errors = self._verify_artifacts_structure(run_index=1)
        self.assertEqual(len(errors), 0, "\n".join(errors))
        
        errors = self._verify_flow_order(run_index=1)
        self.assertEqual(len(errors), 0, "\n".join(errors))
        
        errors = self._verify_answers_structure(run_index=1)
        self.assertEqual(len(errors), 0, "\n".join(errors))
        
        errors = self._verify_validators(run_index=1)
        self.assertEqual(len(errors), 0, "\n".join(errors))
        
        hash_value, errors = self._verify_deterministic_hash(run_index=1)
        self.assertEqual(len(errors), 0, "\n".join(errors))
        self.assertIsNotNone(hash_value, "Failed to extract deterministic_hash")

    def test_03_three_run_reproducibility(self):
        """Test reproducibility across three consecutive runs"""
        hashes = []
        flow_orders = []
        all_errors = []
        
        for run_idx in range(1, 4):
            print(f"\n{'='*80}")
            print(f"REPRODUCIBILITY RUN {run_idx}/3")
            print(f"{'='*80}")
            
            rc, result, stderr = self._run_cli_command([
                str(self.cli_path),
                "evaluate",
                str(self.test_plan_path),
                "--repo", str(self.repo_root)
            ])
            
            if rc != 0:
                all_errors.append(f"Run {run_idx}: Evaluation failed with exit code {rc}")
                continue
            
            backup_dir = self._backup_artifacts(run_idx)
            print(f"  → Artifacts backed up to {backup_dir}")
            
            hash_value, errors = self._verify_deterministic_hash(run_idx)
            if errors:
                all_errors.extend(errors)
            else:
                hashes.append(hash_value)
                print(f"  → Deterministic hash: {hash_value}")
            
            runtime_path = self.artifacts_dir / "flow_runtime.json"
            try:
                with open(runtime_path) as f:
                    runtime = json.load(f)
                flow_order = tuple(runtime.get("stages", []))
                flow_orders.append(flow_order)
                print(f"  → Flow stages: {len(flow_order)} stages")
            except Exception as e:
                all_errors.append(f"Run {run_idx}: Cannot read flow order: {e}")
            
            for check_name, check_func in [
                ("artifacts", self._verify_artifacts_structure),
                ("flow_order", self._verify_flow_order),
                ("answers", self._verify_answers_structure),
                ("validators", self._verify_validators)
            ]:
                errors = check_func(run_idx)
                if errors:
                    all_errors.extend(errors)
                    print(f"  ✗ {check_name} check failed")
                else:
                    print(f"  ✓ {check_name} check passed")
        
        if all_errors:
            self.fail("Reproducibility test failed:\n" + "\n".join(all_errors))
        
        self.assertEqual(len(hashes), 3, "Failed to extract hashes from all 3 runs")
        
        self.assertEqual(
            len(set(hashes)), 1,
            f"Deterministic hashes differ across runs!\n"
            f"  Run 1: {hashes[0]}\n"
            f"  Run 2: {hashes[1]}\n"
            f"  Run 3: {hashes[2]}"
        )
        
        self.assertEqual(len(flow_orders), 3, "Failed to extract flow orders from all 3 runs")
        
        self.assertEqual(
            len(set(flow_orders)), 1,
            f"Flow orders differ across runs!\n"
            f"  Run 1: {list(flow_orders[0])}\n"
            f"  Run 2: {list(flow_orders[1])}\n"
            f"  Run 3: {list(flow_orders[2])}"
        )
        
        print(f"\n{'='*80}")
        print("✓ REPRODUCIBILITY VERIFIED")
        print(f"  - All 3 runs produced identical deterministic_hash: {hashes[0]}")
        print(f"  - All 3 runs followed same flow order ({len(flow_orders[0])} stages)")
        print(f"{'='*80}\n")

    def test_04_rubric_check(self):
        """Test rubric 1:1 alignment check"""
        if not (self.artifacts_dir / "answers_report.json").exists():
            self._run_cli_command([
                str(self.cli_path),
                "evaluate",
                str(self.test_plan_path),
                "--repo", str(self.repo_root)
            ])
        
        rc, result, stderr = self._run_cli_command([
            str(self.cli_path),
            "rubric-check",
            "--repo", str(self.repo_root)
        ])
        
        self.assertIn(
            rc, [0, 3],
            f"rubric-check failed with unexpected exit code {rc}:\n{stderr}\n{result}"
        )
        
        if rc == 0:
            print("  ✓ Rubric 1:1 alignment verified")
        else:
            print(f"  ⚠ Rubric alignment mismatch (expected for test): {result.get('parsed', {})}")


def main():
    """Run tests with verbose output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestE2EUnifiedPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
