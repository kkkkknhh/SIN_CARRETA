"""
Test suite for reliability calibration.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
from evaluation.reliability_calibration import (
    ReliabilityCalibrator,
    reliability_weighted_score,
    CalibratorManager,
    calibration_workflow
)


class TestReliabilityCalibrator(unittest.TestCase):
    """Tests for the ReliabilityCalibrator class."""
    
    def test_initialization(self):
        """Test calibrator initialization with default priors."""
        cal = ReliabilityCalibrator(detector_name="test_detector")
        
        self.assertEqual(cal.detector_name, "test_detector")
        self.assertEqual(cal.precision_a, 1.0)
        self.assertEqual(cal.precision_b, 1.0)
        self.assertEqual(cal.recall_a, 1.0)
        self.assertEqual(cal.recall_b, 1.0)
        self.assertEqual(cal.n_updates, 0)
        
        # Default prior mean is 0.5
        self.assertAlmostEqual(cal.expected_precision, 0.5, places=5)
        self.assertAlmostEqual(cal.expected_recall, 0.5, places=5)
    
    def test_perfect_detector(self):
        """Test calibrator with perfect predictions."""
        cal = ReliabilityCalibrator(detector_name="perfect")
        
        # Perfect predictions
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        cal.update(y_true, y_pred)
        
        # Should have high precision and recall (with prior, 3 TP + 1 prior = 4/5 = 0.8)
        self.assertGreater(cal.expected_precision, 0.75)
        self.assertGreater(cal.expected_recall, 0.75)
        self.assertGreater(cal.expected_f1, 0.75)
        self.assertEqual(cal.n_updates, 5)
    
    def test_poor_detector(self):
        """Test calibrator with poor predictions."""
        cal = ReliabilityCalibrator(detector_name="poor")
        
        # Poor predictions (inverted)
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1])
        cal.update(y_true, y_pred)
        
        # Should have low precision and recall
        self.assertLess(cal.expected_precision, 0.5)
        self.assertLess(cal.expected_recall, 0.5)
    
    def test_high_precision_low_recall(self):
        """Test detector with high precision but low recall."""
        cal = ReliabilityCalibrator(detector_name="conservative")
        
        # Only predicts positive when very confident (few positives)
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2 TP, 0 FP, 3 FN
        cal.update(y_true, y_pred)
        
        # High precision (no false positives)
        self.assertGreater(cal.expected_precision, 0.6)
        # Low recall (many false negatives)
        self.assertLess(cal.expected_recall, 0.5)
    
    def test_precision_update(self):
        """Test precision-only update."""
        cal = ReliabilityCalibrator(detector_name="precision_test")
        
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0])  # 2 TP, 1 FP
        
        initial_a = cal.precision_a
        initial_b = cal.precision_b
        
        cal.update_precision(y_true, y_pred)
        
        # Should add 2 to a (TP) and 1 to b (FP)
        self.assertEqual(cal.precision_a, initial_a + 2)
        self.assertEqual(cal.precision_b, initial_b + 1)
    
    def test_recall_update(self):
        """Test recall-only update."""
        cal = ReliabilityCalibrator(detector_name="recall_test")
        
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])  # 2 TP, 1 FN
        
        initial_a = cal.recall_a
        initial_b = cal.recall_b
        
        cal.update_recall(y_true, y_pred)
        
        # Should add 2 to a (TP) and 1 to b (FN)
        self.assertEqual(cal.recall_a, initial_a + 2)
        self.assertEqual(cal.recall_b, initial_b + 1)
    
    def test_credible_intervals(self):
        """Test credible interval calculation."""
        cal = ReliabilityCalibrator(detector_name="interval_test")
        
        # Add some data
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        cal.update(y_true, y_pred)
        
        # Get intervals
        prec_interval = cal.precision_credible_interval(level=0.95)
        rec_interval = cal.recall_credible_interval(level=0.95)
        
        # Should be tuples
        self.assertIsInstance(prec_interval, tuple)
        self.assertIsInstance(rec_interval, tuple)
        
        # Lower bound should be less than upper bound
        self.assertLess(prec_interval[0], prec_interval[1])
        self.assertLess(rec_interval[0], rec_interval[1])
        
        # Intervals should contain expected values
        self.assertLess(prec_interval[0], cal.expected_precision)
        self.assertGreater(prec_interval[1], cal.expected_precision)
    
    def test_f1_calculation(self):
        """Test F1 score calculation."""
        cal = ReliabilityCalibrator(detector_name="f1_test")
        
        # Add data with known precision/recall
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])  # 3 TP, 0 FP, 1 FN
        cal.update(y_true, y_pred)
        
        # Manual F1 calculation
        precision = cal.expected_precision
        recall = cal.expected_recall
        expected_f1 = 2 * precision * recall / (precision + recall)
        
        self.assertAlmostEqual(cal.expected_f1, expected_f1, places=5)
    
    def test_get_stats(self):
        """Test stats dictionary generation."""
        cal = ReliabilityCalibrator(detector_name="stats_test")
        
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        cal.update(y_true, y_pred)
        
        stats = cal.get_stats()
        
        # Check structure
        self.assertIn('detector', stats)
        self.assertIn('n_updates', stats)
        self.assertIn('precision', stats)
        self.assertIn('recall', stats)
        self.assertIn('f1', stats)
        self.assertIn('beta_params', stats)
        
        # Check values
        self.assertEqual(stats['detector'], 'stats_test')
        self.assertEqual(stats['n_updates'], 4)
        self.assertIsInstance(stats['precision']['mean'], float)
        self.assertIsInstance(stats['precision']['interval'], tuple)
    
    def test_save_load(self):
        """Test calibrator persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = ReliabilityCalibrator(detector_name="persistence_test")
            
            # Add some data
            y_true = np.array([1, 1, 1, 0, 0])
            y_pred = np.array([1, 1, 0, 0, 0])
            cal.update(y_true, y_pred)
            
            # Save
            save_path = Path(tmpdir) / "calibrator.json"
            cal.save(save_path)
            
            # Load
            loaded_cal = ReliabilityCalibrator.load(save_path)
            
            # Check equality
            self.assertEqual(loaded_cal.detector_name, cal.detector_name)
            self.assertEqual(loaded_cal.precision_a, cal.precision_a)
            self.assertEqual(loaded_cal.precision_b, cal.precision_b)
            self.assertEqual(loaded_cal.recall_a, cal.recall_a)
            self.assertEqual(loaded_cal.recall_b, cal.recall_b)
            self.assertEqual(loaded_cal.n_updates, cal.n_updates)
            self.assertAlmostEqual(loaded_cal.expected_precision, cal.expected_precision)
            self.assertAlmostEqual(loaded_cal.expected_recall, cal.expected_recall)


class TestReliabilityWeightedScore(unittest.TestCase):
    """Tests for reliability_weighted_score function."""
    
    def test_weighting_with_precision(self):
        """Test score weighting with precision metric."""
        cal = ReliabilityCalibrator(detector_name="weight_test")
        
        # Create a high-precision detector
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        cal.update(y_true, y_pred)
        
        raw_score = 0.8
        weighted = reliability_weighted_score(raw_score, cal, metric='precision')
        
        # Weighted score should be reasonable for good detector (with prior: 4/5 = 0.8)
        self.assertGreater(weighted, 0.6)
    
    def test_weighting_with_recall(self):
        """Test score weighting with recall metric."""
        cal = ReliabilityCalibrator(detector_name="recall_weight_test")
        
        # Create a high-recall detector
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        cal.update(y_true, y_pred)
        
        raw_score = 0.8
        weighted = reliability_weighted_score(raw_score, cal, metric='recall')
        
        self.assertGreater(weighted, 0.6)
    
    def test_weighting_with_f1(self):
        """Test score weighting with F1 metric."""
        cal = ReliabilityCalibrator(detector_name="f1_weight_test")
        
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        cal.update(y_true, y_pred)
        
        raw_score = 0.8
        weighted = reliability_weighted_score(raw_score, cal, metric='f1')
        
        self.assertGreater(weighted, 0.0)
        self.assertLessEqual(weighted, raw_score)
    
    def test_good_vs_bad_detector(self):
        """Test that good detectors get higher weights than bad ones."""
        # Good detector
        good_cal = ReliabilityCalibrator(detector_name="good")
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        good_cal.update(y_true, y_pred)
        
        # Bad detector
        bad_cal = ReliabilityCalibrator(detector_name="bad")
        y_pred_bad = np.array([0, 0, 0, 1, 1])
        bad_cal.update(y_true, y_pred_bad)
        
        raw_score = 0.8
        good_weighted = reliability_weighted_score(raw_score, good_cal)
        bad_weighted = reliability_weighted_score(raw_score, bad_cal)
        
        self.assertGreater(good_weighted, bad_weighted)
    
    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        cal = ReliabilityCalibrator(detector_name="invalid_test")
        
        with self.assertRaises(ValueError):
            reliability_weighted_score(0.8, cal, metric='invalid')


class TestCalibratorManager(unittest.TestCase):
    """Tests for CalibratorManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            self.assertEqual(manager.storage_dir, Path(tmpdir))
            self.assertEqual(len(manager.calibrators), 0)
    
    def test_get_calibrator_new(self):
        """Test getting a new calibrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            cal = manager.get_calibrator("new_detector")
            
            self.assertIsInstance(cal, ReliabilityCalibrator)
            self.assertEqual(cal.detector_name, "new_detector")
            self.assertIn("new_detector", manager.calibrators)
    
    def test_get_calibrator_existing(self):
        """Test getting an existing calibrator from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)
            
            # Create and save a calibrator
            cal = ReliabilityCalibrator(detector_name="existing")
            y_true = np.array([1, 1, 0, 0])
            y_pred = np.array([1, 1, 0, 0])
            cal.update(y_true, y_pred)
            cal.save(storage_dir / "existing_calibrator.json")
            
            # Create manager and load
            manager = CalibratorManager(storage_dir)
            loaded_cal = manager.get_calibrator("existing")
            
            self.assertEqual(loaded_cal.detector_name, "existing")
            self.assertEqual(loaded_cal.n_updates, 4)
    
    def test_update_from_ground_truth(self):
        """Test updating calibrator from ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            y_true = np.array([1, 1, 0, 0])
            y_pred = np.array([1, 0, 0, 1])
            
            manager.update_from_ground_truth("test_detector", y_true, y_pred)
            
            cal = manager.get_calibrator("test_detector")
            self.assertEqual(cal.n_updates, 4)
    
    def test_save_calibrator(self):
        """Test saving individual calibrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            # Get and update calibrator
            cal = manager.get_calibrator("save_test")
            y_true = np.array([1, 1, 0, 0])
            y_pred = np.array([1, 1, 0, 0])
            cal.update(y_true, y_pred)
            
            # Save
            manager.save_calibrator("save_test")
            
            # Check file exists
            save_path = Path(tmpdir) / "save_test_calibrator.json"
            self.assertTrue(save_path.exists())
    
    def test_save_all(self):
        """Test saving all calibrators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            # Create multiple calibrators
            for name in ["detector1", "detector2", "detector3"]:
                cal = manager.get_calibrator(name)
                y_true = np.array([1, 0])
                y_pred = np.array([1, 0])
                cal.update(y_true, y_pred)
            
            # Save all
            manager.save_all()
            
            # Check all files exist
            for name in ["detector1", "detector2", "detector3"]:
                save_path = Path(tmpdir) / f"{name}_calibrator.json"
                self.assertTrue(save_path.exists())
    
    def test_get_all_stats(self):
        """Test getting stats for all calibrators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            # Create calibrators
            for name in ["det1", "det2"]:
                cal = manager.get_calibrator(name)
                y_true = np.array([1, 1, 0, 0])
                y_pred = np.array([1, 1, 0, 0])
                cal.update(y_true, y_pred)
            
            stats = manager.get_all_stats()
            
            self.assertEqual(len(stats), 2)
            self.assertIn("det1", stats)
            self.assertIn("det2", stats)
            self.assertIn("precision", stats["det1"])


class TestCalibrationWorkflow(unittest.TestCase):
    """Tests for calibration workflow."""
    
    def test_workflow_execution(self):
        """Test complete calibration workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CalibratorManager(Path(tmpdir))
            
            # Prepare labeled data
            labeled_data = {
                "detector1": [(1, 1), (1, 1), (0, 0), (0, 0)],
                "detector2": [(1, 1), (1, 0), (0, 0), (0, 1)]
            }
            
            # Run workflow
            stats = calibration_workflow(manager, labeled_data)
            
            # Check results
            self.assertEqual(len(stats), 2)
            self.assertIn("detector1", stats)
            self.assertIn("detector2", stats)
            
            # detector1 should have better metrics (perfect predictions)
            self.assertGreater(stats["detector1"]["precision"]["mean"], 
                             stats["detector2"]["precision"]["mean"])


if __name__ == '__main__':
    unittest.main()
