#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for system_validators.py rubric_check integration.

Tests the subprocess invocation of tools/rubric_check.py and proper
handling of exit codes 2 (missing files) and 3 (mismatch).
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from system_validators import SystemHealthValidator


class TestRubricCheckIntegration(unittest.TestCase):
    """Test rubric check subprocess integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.tools_dir = self.tmpdir / "tools"
        self.tools_dir.mkdir()
        self.artifacts_dir = self.tmpdir / "artifacts"
        self.artifacts_dir.mkdir()
        
        # Create minimal required files
        (self.tmpdir / "RUBRIC_SCORING.json").write_text('{"weights": {"Q1": 0.5}}')
        (self.tmpdir / "tools" / "flow_doc.json").write_text('{"canonical_order": ["step1"]}')
        (self.artifacts_dir / "flow_runtime.json").write_text('{"order": ["step1"]}')
        (self.artifacts_dir / "answers_report.json").write_text(
            '{"answers": [{"question_id": "Q1"}], "summary": {"total_questions": 300}}'
        )
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rubric_check_exit_code_0_success(self):
        """Test that exit code 0 from rubric_check indicates success."""
        # Create a mock rubric_check.py that exits with 0
        rubric_check_script = self.tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print(json.dumps({"match": True}))\n'
            'sys.exit(0)\n'
        )
        
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        self.assertTrue(result["ok_rubric_1to1"])
        self.assertNotIn("Rubric mismatch", str(result.get("errors", [])))

    def test_rubric_check_exit_code_3_mismatch(self):
        """Test that exit code 3 from rubric_check indicates mismatch."""
        # Create a mock rubric_check.py that exits with 3
        rubric_check_script = self.tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print(json.dumps({"match": False, "missing_weights": ["Q2"], "extra_weights": []}))\n'
            'sys.exit(3)\n'
        )
        
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        self.assertFalse(result["ok_rubric_1to1"])
        self.assertFalse(result["ok"])
        # Check that error message mentions mismatch
        errors_str = " ".join(result.get("errors", []))
        self.assertIn("exit code 3", errors_str)
        self.assertIn("mismatch", errors_str.lower())

    def test_rubric_check_exit_code_2_missing_files(self):
        """Test that exit code 2 from rubric_check indicates missing files."""
        # Create a mock rubric_check.py that exits with 2
        rubric_check_script = self.tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print(json.dumps({"error": "file_read_error"}), file=sys.stderr)\n'
            'sys.exit(2)\n'
        )
        
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        self.assertFalse(result["ok_rubric_1to1"])
        self.assertFalse(result["ok"])
        # Check that error message mentions missing files
        errors_str = " ".join(result.get("errors", []))
        self.assertIn("exit code 2", errors_str)
        self.assertIn("Missing input files", errors_str)

    def test_rubric_check_not_invoked_when_strict_false(self):
        """Test that rubric_check is not invoked when check_rubric_strict=False."""
        # Don't create rubric_check.py at all
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=False
        )
        
        # Should pass since we don't check rubric when strict is False
        self.assertTrue(result["ok_rubric_1to1"])

    def test_rubric_check_missing_script(self):
        """Test that missing rubric_check.py script is handled gracefully."""
        # Don't create rubric_check.py
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        self.assertFalse(result["ok_rubric_1to1"])
        self.assertFalse(result["ok"])
        # Check that error message mentions either missing script or file error
        errors_str = " ".join(result.get("errors", []))
        self.assertTrue(
            "rubric_check.py not found" in errors_str.lower() or
            ("exit code 2" in errors_str and "no such file" in errors_str.lower())
        )

    def test_rubric_check_stdout_captured(self):
        """Test that stdout from rubric_check is captured in error messages."""
        # Create a mock rubric_check.py with diagnostic output
        rubric_check_script = self.tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print(json.dumps({"match": False, "missing_weights": ["Q-DIAGNOSTIC"]}))\n'
            'sys.exit(3)\n'
        )
        
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        self.assertFalse(result["ok_rubric_1to1"])
        errors_str = " ".join(result.get("errors", []))
        # The JSON output should be included in the error message
        self.assertIn("Q-DIAGNOSTIC", errors_str)

    def test_rubric_check_stderr_captured(self):
        """Test that stderr from rubric_check is captured in error messages."""
        # Create a mock rubric_check.py with stderr output
        rubric_check_script = self.tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print("STDERR_DIAGNOSTIC_MESSAGE", file=sys.stderr)\n'
            'sys.exit(2)\n'
        )
        
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        self.assertFalse(result["ok_rubric_1to1"])
        errors_str = " ".join(result.get("errors", []))
        # The stderr output should be included in the error message
        self.assertIn("STDERR_DIAGNOSTIC_MESSAGE", errors_str)

    def test_existing_checks_still_execute(self):
        """Test that existing post-execution checks continue to run."""
        # Create a valid rubric_check.py
        rubric_check_script = self.tools_dir / "rubric_check.py"
        rubric_check_script.write_text(
            '#!/usr/bin/env python3\n'
            'import sys, json\n'
            'print(json.dumps({"match": True}))\n'
            'sys.exit(0)\n'
        )
        
        # But create an invalid flow order to trigger existing validation
        (self.artifacts_dir / "flow_runtime.json").write_text('{"order": ["wrong_step"]}')
        
        validator = SystemHealthValidator(str(self.tmpdir))
        result = validator.validate_post_execution(
            artifacts_dir=str(self.artifacts_dir.relative_to(self.tmpdir)),
            check_rubric_strict=True
        )
        
        # Rubric check should pass
        self.assertTrue(result["ok_rubric_1to1"])
        # But order check should fail
        self.assertFalse(result["ok_order"])
        # Overall should fail
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
