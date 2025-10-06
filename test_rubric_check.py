#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for rubric_check.py validation tool.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest


class TestRubricCheckTool(unittest.TestCase):
    """Test suite for rubric_check.py GATE #5 validation tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = pathlib.Path.cwd()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validation_passes_with_real_files(self):
        """Test that validation passes with actual project files."""
        result = subprocess.run(
            ['python3', 'rubric_check.py'],
            cwd=self.original_dir,
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, 
                        f"Validation should pass. Output: {result.stdout}\n{result.stderr}")
        self.assertIn("GATE #5 VALIDATION PASSED", result.stdout)
        self.assertIn("300/300 questions have weights", result.stdout)

    def test_validation_fails_with_missing_weights(self):
        """Test that validation fails when weights are missing."""
        # Create test files in temporary directory
        rubric = {
            "questions": [{"id": "D1-Q1", "scoring_modality": "TYPE_A"}],
            "weights": {}  # Empty weights
        }
        decalogo = {
            "questions": [{"id": "D1-Q1", "point_code": "P1"}]
        }
        
        rubric_path = pathlib.Path(self.test_dir) / "rubric_scoring.json"
        decalogo_path = pathlib.Path(self.test_dir) / "DECALOGO_FULL.json"
        
        with open(rubric_path, 'w') as f:
            json.dump(rubric, f)
        with open(decalogo_path, 'w') as f:
            json.dump(decalogo, f)
        
        result = subprocess.run(
            ['python3', str(self.original_dir / 'rubric_check.py')],
            cwd=self.test_dir,
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 1, 
                        "Validation should fail with missing weights")
        self.assertIn("GATE #5 VALIDATION FAILED", result.stdout)
        self.assertIn("Missing weights", result.stdout)

    def test_validation_shows_alignment_info(self):
        """Test that validation output shows weight and question counts."""
        result = subprocess.run(
            ['python3', 'rubric_check.py'],
            cwd=self.original_dir,
            capture_output=True,
            text=True
        )
        
        self.assertIn("Found 300 weights", result.stdout)
        self.assertIn("Found 300 questions", result.stdout)


if __name__ == "__main__":
    unittest.main()
