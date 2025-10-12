"""
Example usage of reliability calibration system.

Demonstrates:
1. Creating calibrators for detectors
2. Simulating detector predictions
3. Updating calibrators with ground truth
4. Weighted scoring
5. Saving/loading calibrators
"""

from pathlib import Path

import numpy as np

from evaluation.ground_truth_collector import GroundTruthCollector
from evaluation.reliability_calibration import (
    CalibratorManager,
    ReliabilityCalibrator,
    calibration_workflow,
    reliability_weighted_score,
)


def example_basic_calibration():
    """Basic calibration example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Calibration")
    print("=" * 60)

    # Create a calibrator for a detector
    calibrator = ReliabilityCalibrator(detector_name="responsibility_detector")

    # Simulate some predictions and ground truth
    # In reality, these would come from actual detector output and manual labeling
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])  # Ground truth
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1])  # Detector predictions

    print("\nInitial state (uniform prior):")
    print(f"  Expected Precision: {calibrator.expected_precision:.2%}")
    print(f"  Expected Recall: {calibrator.expected_recall:.2%}")
    print(f"  Expected F1: {calibrator.expected_f1:.2%}")

    # Update calibrator with ground truth
    calibrator.update(y_true, y_pred)

    print(f"\nAfter {len(y_true)} updates:")
    print(f"  Expected Precision: {calibrator.expected_precision:.2%}")
    print(f"  Expected Recall: {calibrator.expected_recall:.2%}")
    print(f"  Expected F1: {calibrator.expected_f1:.2%}")

    # Get credible intervals
    prec_interval = calibrator.precision_credible_interval(level=0.95)
    rec_interval = calibrator.recall_credible_interval(level=0.95)

    print("\n95% Credible Intervals:")
    print(f"  Precision: [{prec_interval[0]:.2%}, {prec_interval[1]:.2%}]")
    print(f"  Recall: [{rec_interval[0]:.2%}, {rec_interval[1]:.2%}]")

    # Use weighted scoring
    raw_score = 0.85
    weighted_score = reliability_weighted_score(raw_score, calibrator, metric="f1")

    print("\nScore Weighting:")
    print(f"  Raw Score: {raw_score:.2%}")
    print(f"  Weighted Score: {weighted_score:.2%}")
    print(f"  Adjustment Factor: {calibrator.expected_f1:.2%}")


def example_multiple_detectors():
    """Example with multiple detectors using CalibratorManager."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: Multiple Detectors with Manager")
    print("=" * 60)

    # Create manager
    manager = CalibratorManager(Path("calibrators_demo"))

    # Simulate data for multiple detectors
    detectors_data = {
        "responsibility_detector": {
            "y_true": np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1]),
            "y_pred": np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1]),
        },
        "contradiction_detector": {
            "y_true": np.array([1, 1, 1, 0, 0, 0, 1, 1]),
            "y_pred": np.array([1, 1, 0, 0, 0, 1, 1, 0]),
        },
        "monetary_detector": {
            "y_true": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            "y_pred": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        },
        "feasibility_scorer": {
            "y_true": np.array([1, 1, 0, 0, 1, 1, 0, 0]),
            "y_pred": np.array([1, 0, 0, 1, 1, 0, 0, 1]),
        },
    }

    # Update calibrators
    for detector_name, data in detectors_data.items():
        manager.update_from_ground_truth(detector_name, data["y_true"], data["y_pred"])

    # Get all stats
    all_stats = manager.get_all_stats()

    print("\nCalibration Results for All Detectors:")
    print("-" * 60)
    for detector, stats in all_stats.items():
        print(f"\n{detector}:")
        print(f"  Updates: {stats['n_updates']}")
        print(f"  Precision: {stats['precision']['mean']:.2%}")
        print(f"  Recall: {stats['recall']['mean']:.2%}")
        print(f"  F1: {stats['f1']:.2%}")

    # Demonstrate ensemble with weighted scores
    print("\n\nEnsemble Scoring Example:")
    print("-" * 60)
    raw_scores = {
        "responsibility_detector": 0.85,
        "contradiction_detector": 0.72,
        "monetary_detector": 0.90,
        "feasibility_scorer": 0.68,
    }

    weighted_scores = {}
    for detector_name, raw_score in raw_scores.items():
        calibrator = manager.get_calibrator(detector_name)
        weighted = reliability_weighted_score(raw_score, calibrator, metric="f1")
        weighted_scores[detector_name] = weighted
        print(f"{detector_name}:")
        print(
            f"  Raw: {raw_score:.2%} → Weighted: {weighted:.2%} "
            f"(reliability: {calibrator.expected_f1:.2%})"
        )

    # Simple vs weighted ensemble
    simple_avg = np.mean(list(raw_scores.values()))
    weighted_avg = np.mean(list(weighted_scores.values()))

    print("\nEnsemble Scores:")
    print(f"  Simple Average: {simple_avg:.2%}")
    print(f"  Weighted Average: {weighted_avg:.2%}")
    print(f"  Difference: {weighted_avg - simple_avg:+.2%}")


def example_ground_truth_collection():
    """Example of ground truth collection workflow."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 3: Ground Truth Collection Workflow")
    print("=" * 60)

    # Create collector
    collector = GroundTruthCollector(Path("ground_truth_demo"))

    # Simulate adding predictions during processing
    print("\nCollecting predictions for labeling...")

    predictions = [
        (
            "responsibility_detector",
            "ev_001",
            1,
            0.85,
            {"text": "El Ministerio implementará..."},
        ),
        (
            "responsibility_detector",
            "ev_002",
            1,
            0.72,
            {"text": "La alcaldía supervisará..."},
        ),
        (
            "responsibility_detector",
            "ev_003",
            0,
            0.45,
            {"text": "El proyecto incluye..."},
        ),
        (
            "contradiction_detector",
            "ev_004",
            1,
            0.90,
            {"text": "Contradicción detectada"},
        ),
        ("monetary_detector", "ev_005", 1, 0.95, {"text": "$1.5 millones detectados"}),
    ]

    for detector, ev_id, pred, conf, ctx in predictions:
        collector.add_prediction(detector, ev_id, pred, conf, ctx)

    print(f"  Collected {collector.get_pending_count()} predictions")

    # Export for labeling
    export_file = Path("ground_truth_demo/to_label.json")
    collector.export_for_labeling(export_file)
    print(f"  Exported to {export_file}")

    # Simulate manual labeling (in reality, human would label)
    print("\n  [Simulating manual labeling...]")
    import json

    with open(export_file, "r") as f:
        items = json.load(f)

    # Add ground truth labels (simulate human labeling)
    items[0]["ground_truth"] = 1  # Correct
    items[1]["ground_truth"] = 1  # Correct
    items[2]["ground_truth"] = 1  # Wrong (detector said 0, but true was 1)
    items[3]["ground_truth"] = 0  # Wrong (detector said 1, but true was 0)
    items[4]["ground_truth"] = 1  # Correct

    labeled_file = Path("ground_truth_demo/labeled.json")
    with open(labeled_file, "w") as f:
        json.dump(items, f, indent=2)

    print(f"  Labeled data saved to {labeled_file}")

    # Import and update calibrators
    print("\nUpdating calibrators with labeled data...")
    labeled_data = collector.import_labeled_data(labeled_file)

    manager = CalibratorManager(Path("calibrators_demo"))
    for detector_name, pairs in labeled_data.items():
        y_true = np.array([p[0] for p in pairs])
        y_pred = np.array([p[1] for p in pairs])
        manager.update_from_ground_truth(detector_name, y_true, y_pred)

        cal = manager.get_calibrator(detector_name)
        print(f"  {detector_name}: F1 = {cal.expected_f1:.2%}")


def example_calibration_workflow():
    """Example using the complete calibration workflow."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 4: Complete Calibration Workflow")
    print("=" * 60)

    # Prepare labeled data (simulating multiple labeling batches)
    labeled_data = {
        "responsibility_detector": [
            (1, 1),
            (1, 1),
            (1, 0),
            (0, 0),
            (0, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 1),
        ],
        "contradiction_detector": [
            (1, 1),
            (1, 0),
            (0, 0),
            (0, 0),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 0),
        ],
        "monetary_detector": [
            (1, 1),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ],
    }

    # Create manager
    manager = CalibratorManager(Path("calibrators_demo"))

    # Run workflow
    _stats = calibration_workflow(manager, labeled_data)

    print("\nCalibration metrics successfully updated and saved!")


def example_persistence():
    """Example of saving and loading calibrators."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 5: Persistence")
    print("=" * 60)

    # Create and train a calibrator
    cal = ReliabilityCalibrator(detector_name="test_detector")
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 1])
    cal.update(y_true, y_pred)

    print("\nOriginal calibrator:")
    print(f"  Precision: {cal.expected_precision:.2%}")
    print(f"  Recall: {cal.expected_recall:.2%}")

    # Save
    save_path = Path("calibrators_demo/test_detector_calibrator.json")
    cal.save(save_path)
    print(f"\nSaved to {save_path}")

    # Load
    loaded_cal = ReliabilityCalibrator.load(save_path)
    print("\nLoaded calibrator:")
    print(f"  Precision: {loaded_cal.expected_precision:.2%}")
    print(f"  Recall: {loaded_cal.expected_recall:.2%}")
    print(
        f"  Match: {abs(cal.expected_precision - loaded_cal.expected_precision) < 1e-10}"
    )


if __name__ == "__main__":
    example_basic_calibration()
    example_multiple_detectors()
    example_ground_truth_collection()
    example_calibration_workflow()
    example_persistence()

    print("\n\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate calibrators into existing detectors")
    print("2. Collect real ground truth data")
    print("3. Run calibration workflow periodically")
    print("4. Monitor F1 improvements in ensemble")
