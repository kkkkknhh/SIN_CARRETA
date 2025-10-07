"""
Test suite for ground truth collection.
"""

import unittest
import tempfile
from pathlib import Path
import json
from evaluation.ground_truth_collector import (
    GroundTruthCollector,
    create_ground_truth_collector
)


class TestGroundTruthCollector(unittest.TestCase):
    """Tests for the GroundTruthCollector class."""
    
    def test_initialization(self):
        """Test collector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "gt_storage"
            collector = GroundTruthCollector(storage_path)
            
            self.assertEqual(collector.storage_path, storage_path)
            self.assertTrue(storage_path.exists())
            self.assertEqual(len(collector.pending_labels), 0)
    
    def test_add_prediction(self):
        """Test adding predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(Path(tmpdir))
            
            context = {"text": "Sample text", "confidence": 0.85}
            collector.add_prediction(
                detector_name="test_detector",
                evidence_id="evidence_001",
                prediction=1,
                confidence=0.85,
                context=context
            )
            
            self.assertEqual(len(collector.pending_labels), 1)
            
            item = collector.pending_labels[0]
            self.assertEqual(item['detector'], "test_detector")
            self.assertEqual(item['evidence_id'], "evidence_001")
            self.assertEqual(item['prediction'], 1)
            self.assertEqual(item['confidence'], 0.85)
            self.assertIsNone(item['ground_truth'])
            self.assertIn('timestamp', item)
    
    def test_add_multiple_predictions(self):
        """Test adding multiple predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(Path(tmpdir))
            
            for i in range(5):
                collector.add_prediction(
                    detector_name=f"detector_{i}",
                    evidence_id=f"evidence_{i}",
                    prediction=i % 2,
                    confidence=0.5 + i * 0.1,
                    context={"index": i}
                )
            
            self.assertEqual(len(collector.pending_labels), 5)
    
    def test_get_pending_count(self):
        """Test getting pending count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(Path(tmpdir))
            
            self.assertEqual(collector.get_pending_count(), 0)
            
            collector.add_prediction("det1", "ev1", 1, 0.8, {})
            self.assertEqual(collector.get_pending_count(), 1)
            
            collector.add_prediction("det2", "ev2", 0, 0.6, {})
            self.assertEqual(collector.get_pending_count(), 2)
    
    def test_clear_pending(self):
        """Test clearing pending labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(Path(tmpdir))
            
            collector.add_prediction("det1", "ev1", 1, 0.8, {})
            collector.add_prediction("det2", "ev2", 0, 0.6, {})
            self.assertEqual(len(collector.pending_labels), 2)
            
            collector.clear_pending()
            self.assertEqual(len(collector.pending_labels), 0)
    
    def test_export_for_labeling_json(self):
        """Test exporting for labeling to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GroundTruthCollector(Path(tmpdir))
            
            # Add some predictions
            for i in range(3):
                collector.add_prediction(
                    detector_name="test_detector",
                    evidence_id=f"ev_{i}",
                    prediction=i % 2,
                    confidence=0.7 + i * 0.1,
                    context={"index": i}
                )
            
            # Export
            output_file = Path(tmpdir) / "to_label.json"
            collector.export_for_labeling(output_file)
            
            # Check file exists
            self.assertTrue(output_file.exists())
            
            # Load and verify
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]['detector'], "test_detector")
    
    def test_import_labeled_data_json(self):
        """Test importing labeled data from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create labeled data file
            labeled_data = [
                {
                    'detector': 'detector1',
                    'evidence_id': 'ev1',
                    'prediction': 1,
                    'confidence': 0.8,
                    'context': '{}',
                    'timestamp': '2024-01-01T00:00:00',
                    'ground_truth': 1
                },
                {
                    'detector': 'detector1',
                    'evidence_id': 'ev2',
                    'prediction': 0,
                    'confidence': 0.6,
                    'context': '{}',
                    'timestamp': '2024-01-01T00:00:01',
                    'ground_truth': 0
                },
                {
                    'detector': 'detector2',
                    'evidence_id': 'ev3',
                    'prediction': 1,
                    'confidence': 0.9,
                    'context': '{}',
                    'timestamp': '2024-01-01T00:00:02',
                    'ground_truth': 0
                },
                {
                    'detector': 'detector1',
                    'evidence_id': 'ev4',
                    'prediction': 1,
                    'confidence': 0.7,
                    'context': '{}',
                    'timestamp': '2024-01-01T00:00:03',
                    'ground_truth': None  # Should be filtered out
                }
            ]
            
            input_file = Path(tmpdir) / "labeled.json"
            with open(input_file, 'w') as f:
                json.dump(labeled_data, f)
            
            # Import
            collector = GroundTruthCollector(Path(tmpdir))
            grouped = collector.import_labeled_data(input_file)
            
            # Verify
            self.assertEqual(len(grouped), 2)
            self.assertIn('detector1', grouped)
            self.assertIn('detector2', grouped)
            
            # detector1 should have 2 labeled items
            self.assertEqual(len(grouped['detector1']), 2)
            self.assertEqual(grouped['detector1'][0], (1, 1))
            self.assertEqual(grouped['detector1'][1], (0, 0))
            
            # detector2 should have 1 labeled item
            self.assertEqual(len(grouped['detector2']), 1)
            self.assertEqual(grouped['detector2'][0], (0, 1))
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "gt_storage"
            collector = GroundTruthCollector(storage_path)
            
            # Add predictions
            for i in range(3):
                collector.add_prediction(
                    detector_name="test_detector",
                    evidence_id=f"ev_{i}",
                    prediction=i % 2,
                    confidence=0.8,
                    context={"index": i}
                )
            
            # Save checkpoint
            checkpoint_file = storage_path / "checkpoint.json"
            collector.save_checkpoint(checkpoint_file)
            self.assertTrue(checkpoint_file.exists())
            
            # Create new collector and load
            collector2 = GroundTruthCollector(storage_path)
            self.assertEqual(len(collector2.pending_labels), 0)
            
            collector2.load_checkpoint(checkpoint_file)
            self.assertEqual(len(collector2.pending_labels), 3)
            self.assertEqual(collector2.pending_labels[0]['evidence_id'], 'ev_0')
    
    def test_checkpoint_default_path(self):
        """Test checkpoint with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "gt_storage"
            collector = GroundTruthCollector(storage_path)
            
            collector.add_prediction("det1", "ev1", 1, 0.8, {})
            
            # Save with default path
            collector.save_checkpoint()
            default_checkpoint = storage_path / "pending_checkpoint.json"
            self.assertTrue(default_checkpoint.exists())
            
            # Load with default path
            collector2 = GroundTruthCollector(storage_path)
            collector2.load_checkpoint()
            self.assertEqual(len(collector2.pending_labels), 1)
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "gt_storage"
            collector = GroundTruthCollector(storage_path)
            
            nonexistent = storage_path / "nonexistent.json"
            collector.load_checkpoint(nonexistent)
            
            # Should not raise exception
            self.assertEqual(len(collector.pending_labels), 0)


class TestCreateGroundTruthCollector(unittest.TestCase):
    """Tests for create_ground_truth_collector factory."""
    
    def test_factory_default_path(self):
        """Test factory with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                collector = create_ground_truth_collector("test_detector")
                
                self.assertIsInstance(collector, GroundTruthCollector)
                expected_path = Path("ground_truth") / "test_detector"
                self.assertEqual(collector.storage_path, expected_path)
            finally:
                os.chdir(original_cwd)
    
    def test_factory_custom_path(self):
        """Test factory with custom base path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "custom_gt"
            collector = create_ground_truth_collector("test_detector", base_path)
            
            expected_path = base_path / "test_detector"
            self.assertEqual(collector.storage_path, expected_path)


if __name__ == '__main__':
    unittest.main()
