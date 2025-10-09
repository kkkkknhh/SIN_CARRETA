#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for pdm_contra/decalogo_alignment.py fix

This test verifies that pdm_contra/decalogo_alignment.py correctly
parses and uses the structure from decalogo-industrial.latest.clean.json.

BEFORE THE FIX:
- _load_decalogo_industrial expected a list at root level
- Code ignored the loaded data and always created placeholders
- No real questions were used from the industrial file

AFTER THE FIX:
- _load_decalogo_industrial correctly handles dict with 'questions' key
- _build_industrial_clusters properly builds clusters from real data
- align_decalogos uses actual questions instead of placeholders
- All 347 questions are correctly loaded and processed

Run: python3 test_decalogo_alignment_fix.py
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add repo to path
sys.path.insert(0, '/home/runner/work/SIN_CARRETA/SIN_CARRETA')

from pdm_contra.decalogo_alignment import (
    _load_decalogo_industrial,
    _build_industrial_clusters,
    align_decalogos,
    AuditIssue
)


class TestDecalogoIndustrialLoading(unittest.TestCase):
    """Test that decalogo-industrial.latest.clean.json is correctly loaded."""
    
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path('/home/runner/work/SIN_CARRETA/SIN_CARRETA')
        cls.industrial_path = cls.repo_root / 'decalogo-industrial.latest.clean.json'
        
        # Load the original file for comparison
        with open(cls.industrial_path, 'r') as f:
            cls.original_data = json.load(f)
    
    def test_file_structure(self):
        """Verify the file has expected structure."""
        self.assertIsInstance(self.original_data, dict)
        self.assertIn('version', self.original_data)
        self.assertIn('schema', self.original_data)
        self.assertIn('total', self.original_data)
        self.assertIn('questions', self.original_data)
        self.assertIsInstance(self.original_data['questions'], list)
    
    def test_load_function_returns_dict(self):
        """Test that _load_decalogo_industrial returns a dict."""
        audit = []
        data = _load_decalogo_industrial(self.industrial_path, audit)
        
        self.assertIsInstance(data, dict, "Should return a dict, not a list")
        self.assertIn('questions', data, "Should have 'questions' key")
        self.assertEqual(len(audit), 0, "Should not have audit issues on success")
    
    def test_load_function_preserves_all_data(self):
        """Test that _load_decalogo_industrial preserves all data."""
        audit = []
        data = _load_decalogo_industrial(self.industrial_path, audit)
        
        self.assertEqual(data['version'], self.original_data['version'])
        self.assertEqual(data['schema'], self.original_data['schema'])
        self.assertEqual(data['total'], self.original_data['total'])
        self.assertEqual(len(data['questions']), len(self.original_data['questions']))


class TestIndustrialClusterBuilding(unittest.TestCase):
    """Test that clusters are correctly built from industrial data."""
    
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path('/home/runner/work/SIN_CARRETA/SIN_CARRETA')
        cls.industrial_path = cls.repo_root / 'decalogo-industrial.latest.clean.json'
        
        # Load the data
        audit = []
        cls.industrial_data = _load_decalogo_industrial(cls.industrial_path, audit)
        cls.clusters = _build_industrial_clusters(cls.industrial_data, audit)
    
    def test_correct_number_of_clusters(self):
        """Test that 10 clusters are created (P1-P10)."""
        self.assertEqual(len(self.clusters), 10)
    
    def test_all_clusters_have_data(self):
        """Test that no cluster is empty."""
        for cluster in self.clusters:
            self.assertGreater(len(cluster.points), 0)
            self.assertGreater(len(cluster.points[0].questions), 0)
    
    def test_total_questions_match(self):
        """Test that total questions match the source file."""
        total = sum(len(c.points[0].questions) for c in self.clusters)
        self.assertEqual(total, 347)
    
    def test_no_placeholders(self):
        """Test that clusters are not placeholders."""
        for cluster in self.clusters:
            # Check that status is not a placeholder status
            if cluster.status:
                self.assertNotIn('faltante', cluster.status.lower())
            
            # Check that questions have real data
            for point in cluster.points:
                if point.questions:
                    first_q = point.questions[0]
                    self.assertNotEqual(first_q.q_label, "evidencia_insuficiente")
    
    def test_question_structure(self):
        """Test that questions have proper structure."""
        first_cluster = self.clusters[0]
        first_question = first_cluster.points[0].questions[0]
        
        # Check required fields are present
        self.assertTrue(first_question.q_id)
        self.assertTrue(first_question.q_code)
        self.assertTrue(first_question.q_label)
        self.assertIsInstance(first_question.aliases, list)
        self.assertIsInstance(first_question.refs, list)


class TestFullAlignment(unittest.TestCase):
    """Test the full align_decalogos function."""
    
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path('/home/runner/work/SIN_CARRETA/SIN_CARRETA')
        cls.industrial_path = cls.repo_root / 'decalogo-industrial.latest.clean.json'
        cls.dnp_path = cls.repo_root / 'dnp-standards.latest.clean.json'
    
    def test_align_decalogos_success(self):
        """Test that align_decalogos completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'output'
            
            result = align_decalogos(
                full_path=self.industrial_path,
                industrial_path=self.industrial_path,
                dnp_path=self.dnp_path,
                out_dir=out_dir
            )
            
            self.assertEqual(len(result.clusters), 10)
            self.assertGreater(len(result.audit), 0)
    
    def test_output_files_created(self):
        """Test that output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'output'
            
            align_decalogos(
                full_path=self.industrial_path,
                industrial_path=self.industrial_path,
                dnp_path=self.dnp_path,
                out_dir=out_dir
            )
            
            # Check expected files exist
            expected_files = [
                'decalogo-industrial.v1.0.0.clean.json',
                'crosswalk.v1.0.0.json',
            ]
            
            for filename in expected_files:
                filepath = out_dir / filename
                self.assertTrue(filepath.exists(), f"{filename} should be created")
    
    def test_audit_shows_real_data_used(self):
        """Test that audit trail shows real data was used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'output'
            
            result = align_decalogos(
                full_path=self.industrial_path,
                industrial_path=self.industrial_path,
                dnp_path=self.dnp_path,
                out_dir=out_dir
            )
            
            # Find the industrial audit message
            industrial_audit = [
                issue for issue in result.audit
                if issue.source == 'decalogo-industrial'
            ]
            
            self.assertGreater(len(industrial_audit), 0)
            
            # Check that it mentions building clusters (not placeholders)
            message = industrial_audit[0].message
            self.assertIn('clusters', message.lower())
            self.assertIn('347', message)  # Should mention the correct number


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDecalogoIndustrialLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestIndustrialClusterBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestFullAlignment))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Testing pdm_contra/decalogo_alignment.py fix")
    print("=" * 80 + "\n")
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("  pdm_contra/decalogo_alignment.py correctly handles")
        print("  decalogo-industrial.latest.clean.json structure")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        sys.exit(1)
