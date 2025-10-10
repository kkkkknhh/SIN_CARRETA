"""
Example integration of reliability calibration with ResponsibilityDetector.

Demonstrates how to modify an existing detector to:
1. Use reliability calibration for weighted scoring
2. Collect ground truth for future calibration
3. Load/save calibrator state
"""

from pathlib import Path
from typing import Dict, Any
from evaluation.reliability_calibration import CalibratorManager, reliability_weighted_score
from evaluation.ground_truth_collector import create_ground_truth_collector


class ResponsibilityDetectorWithCalibration:
    """
    Enhanced ResponsibilityDetector that uses reliability calibration.
    
    This is a simplified example showing integration patterns.
    """
    
    def __init__(self, 
                 calibrator_manager: CalibratorManager = None,
                 enable_ground_truth_collection: bool = True):
        """
        Initialize detector with calibration support.
        
        Args:
            calibrator_manager: Manager for calibrators (creates default if None)
            enable_ground_truth_collection: Whether to collect predictions for labeling
        """
        # Initialize base detector (would normally load spaCy model, etc.)
        self.detector_name = "responsibility_detector"
        
        # Initialize calibration infrastructure
        if calibrator_manager is None:
            calibrator_manager = CalibratorManager(Path("calibrators"))
        
        self.calibrator_manager = calibrator_manager
        self.calibrator = calibrator_manager.get_calibrator(self.detector_name)
        
        # Initialize ground truth collector
        self.enable_gt_collection = enable_ground_truth_collection
        if enable_ground_truth_collection:
            self.gt_collector = create_ground_truth_collector(self.detector_name)
    
    @staticmethod
    def _detect_raw(text: str) -> Dict[str, Any]:
        """
        Base detection method (simplified for example).
        
        In reality, this would use spaCy NER, pattern matching, etc.
        """
        # Simplified detection logic
        # In real implementation, this would return actual entity detection results
        
        # Simulate detection
        has_responsibility = any(word in text.lower() for word in 
                               ["ministerio", "alcaldía", "secretaría", "director"])
        
        raw_confidence = 0.85 if has_responsibility else 0.15
        
        return {
            "detected": has_responsibility,
            "raw_confidence": raw_confidence,
            "entities": ["Ministerio"] if has_responsibility else []
        }
    
    def detect(self, text: str, evidence_id: str = None) -> Dict[str, Any]:
        """
        Detect responsibilities with calibrated scoring.
        
        Args:
            text: Text to analyze
            evidence_id: Optional evidence ID for ground truth collection
        
        Returns:
            Detection results with both raw and calibrated scores
        """
        # Perform base detection
        raw_results = self._detect_raw(text)
        
        # Calculate calibrated score
        raw_confidence = raw_results["raw_confidence"]
        calibrated_confidence = reliability_weighted_score(
            raw_confidence,
            self.calibrator,
            metric='f1'
        )
        
        # Prepare binary prediction for ground truth collection
        prediction = 1 if raw_results["detected"] else 0
        
        # Collect for future labeling if enabled
        if self.enable_gt_collection and evidence_id:
            context = {
                'text': text[:200],  # First 200 chars for review
                'entities': raw_results["entities"],
                'raw_confidence': raw_confidence
            }
            self.gt_collector.add_prediction(
                detector_name=self.detector_name,
                evidence_id=evidence_id,
                prediction=prediction,
                confidence=raw_confidence,
                context=context
            )
        
        # Return enhanced results
        return {
            'entities': raw_results["entities"],
            'detected': raw_results["detected"],
            'raw_confidence': raw_confidence,
            'calibrated_confidence': calibrated_confidence,
            'detector_reliability': self.calibrator.expected_f1,
            'calibration_stats': {
                'precision': self.calibrator.expected_precision,
                'recall': self.calibrator.expected_recall,
                'f1': self.calibrator.expected_f1,
                'n_updates': self.calibrator.n_updates
            }
        }
    
    def export_for_labeling(self, output_file: Path):
        """Export collected predictions for manual labeling."""
        if self.enable_gt_collection:
            self.gt_collector.export_for_labeling(output_file)
            print(f"Exported {self.gt_collector.get_pending_count()} predictions to {output_file}")
        else:
            print("Ground truth collection not enabled")
    
    def update_from_labeled_data(self, labeled_file: Path):
        """
        Update calibrator from labeled data.
        
        Args:
            labeled_file: Path to file with ground truth labels
        """
        if not self.enable_gt_collection:
            print("Ground truth collection not enabled")
            return
        
        # Import labeled data
        labeled_data = self.gt_collector.import_labeled_data(labeled_file)
        
        # Update calibrator
        if self.detector_name in labeled_data:
            import numpy as np
            pairs = labeled_data[self.detector_name]
            y_true = np.array([p[0] for p in pairs])
            y_pred = np.array([p[1] for p in pairs])
            
            self.calibrator_manager.update_from_ground_truth(
                self.detector_name,
                y_true,
                y_pred
            )
            
            print(f"Updated calibrator with {len(pairs)} labels")
            print(f"New F1: {self.calibrator.expected_f1:.2%}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic_usage():
    """Example of basic detector usage with calibration."""
    print("=" * 70)
    print("EXAMPLE: Responsibility Detector with Calibration")
    print("=" * 70)
    
    # Create detector with calibration
    detector = ResponsibilityDetectorWithCalibration()
    
    # Test texts
    test_cases = [
        ("El Ministerio de Educación implementará el programa.", "ev_001"),
        ("La alcaldía supervisará el cumplimiento.", "ev_002"),
        ("El proyecto incluye mejoras en infraestructura.", "ev_003"),
        ("La Secretaría de Salud coordinará las actividades.", "ev_004"),
    ]
    
    print("\nProcessing texts with calibrated scoring:")
    print("-" * 70)
    
    for text, ev_id in test_cases:
        result = detector.detect(text, evidence_id=ev_id)
        
        print(f"\nText: {text[:50]}...")
        print(f"  Detected: {result['detected']}")
        print(f"  Raw Confidence: {result['raw_confidence']:.2%}")
        print(f"  Calibrated Confidence: {result['calibrated_confidence']:.2%}")
        print(f"  Detector Reliability (F1): {result['detector_reliability']:.2%}")
    
    # Export for labeling
    print("\n" + "-" * 70)
    output_file = Path("ground_truth/responsibility_detector/to_label.json")
    detector.export_for_labeling(output_file)


def example_with_labeled_data():
    """Example showing the complete workflow with labeled data."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE: Complete Workflow with Labeled Data")
    print("=" * 70)
    
    # Create detector
    detector = ResponsibilityDetectorWithCalibration()
    
    # Process some texts
    test_texts = [
        ("El Ministerio implementará", "ev_101"),
        ("La alcaldía supervisará", "ev_102"),
        ("El proyecto mejorará", "ev_103"),
        ("La Secretaría coordinará", "ev_104"),
    ]
    
    print("\nProcessing texts...")
    for text, ev_id in test_texts:
        detector.detect(text, evidence_id=ev_id)
    
    # Export for labeling
    export_file = Path("ground_truth_demo/to_label_batch1.json")
    detector.export_for_labeling(export_file)
    
    # Simulate labeling (in reality, human would do this)
    print("\n[Simulating manual labeling...]")
    import json
    
    with open(export_file, 'r') as f:
        items = json.load(f)
    
    # Add ground truth (simulating human labels)
    items[0]['ground_truth'] = 1  # Correct
    items[1]['ground_truth'] = 1  # Correct
    items[2]['ground_truth'] = 0  # Correct
    items[3]['ground_truth'] = 1  # Correct
    
    labeled_file = Path("ground_truth_demo/labeled_batch1.json")
    with open(labeled_file, 'w') as f:
        json.dump(items, f, indent=2)
    
    print(f"Labeled data saved to {labeled_file}")
    
    # Update calibrator
    print("\nUpdating calibrator from labeled data...")
    detector.update_from_labeled_data(labeled_file)
    
    # Show updated stats
    print("\nUpdated calibration stats:")
    stats = detector.calibrator.get_stats()
    print(f"  Precision: {stats['precision']['mean']:.2%}")
    print(f"  Recall: {stats['recall']['mean']:.2%}")
    print(f"  F1: {stats['f1']:.2%}")
    print(f"  Updates: {stats['n_updates']}")


def example_ensemble_with_calibration():
    """Example of ensemble scoring with multiple calibrated detectors."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE: Ensemble with Multiple Calibrated Detectors")
    print("=" * 70)
    
    # Create manager for all detectors
    manager = CalibratorManager(Path("calibrators"))
    
    # Simulate detectors with different reliabilities
    import numpy as np
    
    # Update calibrators with simulated historical performance
    manager.update_from_ground_truth(
        "responsibility_detector",
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),  # Good precision/recall
        np.array([1, 1, 1, 0, 0, 0, 1, 0])
    )
    
    manager.update_from_ground_truth(
        "contradiction_detector",
        np.array([1, 1, 1, 0, 0, 0]),  # Medium precision
        np.array([1, 1, 0, 0, 0, 1])
    )
    
    manager.update_from_ground_truth(
        "monetary_detector",
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),  # High precision/recall
        np.array([1, 1, 1, 1, 0, 0, 0, 0])
    )
    
    # Simulate raw scores from detectors
    raw_scores = {
        "responsibility_detector": 0.80,
        "contradiction_detector": 0.75,
        "monetary_detector": 0.85,
    }
    
    print("\nDetector Scores (Raw vs Calibrated):")
    print("-" * 70)
    
    weighted_scores = {}
    for detector_name, raw_score in raw_scores.items():
        calibrator = manager.get_calibrator(detector_name)
        weighted = reliability_weighted_score(raw_score, calibrator, metric='f1')
        weighted_scores[detector_name] = weighted
        
        print(f"\n{detector_name}:")
        print(f"  Raw Score: {raw_score:.2%}")
        print(f"  Reliability (F1): {calibrator.expected_f1:.2%}")
        print(f"  Calibrated Score: {weighted:.2%}")
    
    # Compare ensemble approaches
    simple_avg = np.mean(list(raw_scores.values()))
    weighted_avg = np.mean(list(weighted_scores.values()))
    
    print("\n" + "-" * 70)
    print("Ensemble Comparison:")
    print(f"  Simple Average (unweighted): {simple_avg:.2%}")
    print(f"  Weighted Average (calibrated): {weighted_avg:.2%}")
    print(f"  Adjustment: {weighted_avg - simple_avg:+.2%}")


if __name__ == "__main__":
    example_basic_usage()
    example_with_labeled_data()
    example_ensemble_with_calibration()
    
    print("\n\n" + "=" * 70)
    print("Integration examples completed successfully!")
    print("=" * 70)
    print("\nKey integration points:")
    print("1. Initialize detector with CalibratorManager")
    print("2. Use reliability_weighted_score() for calibrated outputs")
    print("3. Collect predictions with GroundTruthCollector")
    print("4. Periodically update calibrators from labeled data")
    print("5. Use calibrated scores in ensemble evaluation")
