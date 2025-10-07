# Reliability Calibration System

## Overview

The Reliability Calibration System implements Bayesian reliability calibration using Beta distribution to track and update the expected precision of detectors based on their real-world performance. This allows the system to automatically weight detector outputs by their historical reliability, improving overall ensemble accuracy.

## Motivation

Currently, MINIMINIMOON has ~7 detectors (responsibility_detector, contradiction_detector, monetary_detector, feasibility_scorer, etc.) that generate scores or binary classifications. Without reliability tracking, all detectors are weighted equally in the final ensemble, even though some may be more accurate than others.

The calibration system addresses this by:
- Tracking precision, recall, and F1 for each detector using Bayesian methods
- Weighting detector outputs by their expected reliability
- Providing uncertainty estimates through credible intervals
- Enabling continuous improvement through ground truth collection

## Architecture

### Core Components

1. **ReliabilityCalibrator** (`evaluation/reliability_calibration.py`)
   - Maintains Beta(a, b) posterior distribution for precision and recall
   - Updates with (y_true, y_pred) binary pairs
   - Calculates expected precision, recall, and F1
   - Produces 95% credible intervals
   - Supports persistence (save/load state)

2. **CalibratorManager** (`evaluation/reliability_calibration.py`)
   - Manages calibrators for all detectors
   - Handles loading/saving from disk
   - Provides batch update operations
   - Returns aggregate statistics

3. **GroundTruthCollector** (`evaluation/ground_truth_collector.py`)
   - Collects predictions during pipeline execution
   - Exports to CSV/JSON for manual labeling
   - Imports labeled data
   - Groups by detector for calibrator updates

4. **reliability_weighted_score()** (`evaluation/reliability_calibration.py`)
   - Weights detector scores by reliability
   - Supports precision, recall, or F1 as weighting metric

## Mathematical Foundation

### Beta-Bernoulli Conjugate Prior

The system uses Beta distribution as a conjugate prior for Bernoulli/binomial likelihood:

```
Prior: Beta(a, b)
Likelihood: Bernoulli(p)
Posterior: Beta(a + successes, b + failures)
```

**Precision**: Beta(a_p, b_p) where:
- a_p increases with True Positives (TP)
- b_p increases with False Positives (FP)
- E[precision] = a_p / (a_p + b_p)

**Recall**: Beta(a_r, b_r) where:
- a_r increases with True Positives (TP)
- b_r increases with False Negatives (FN)
- E[recall] = a_r / (a_r + b_r)

**F1 Score**: Harmonic mean of expected precision and recall:
```
F1 = 2 * E[precision] * E[recall] / (E[precision] + E[recall])
```

### Credible Intervals

95% credible intervals are computed using the Beta distribution quantile function:
```
[Beta_ppf(0.025, a, b), Beta_ppf(0.975, a, b)]
```

## Usage

### Basic Usage

```python
from evaluation.reliability_calibration import ReliabilityCalibrator, reliability_weighted_score
import numpy as np

# Create calibrator
calibrator = ReliabilityCalibrator(detector_name="responsibility_detector")

# Update with ground truth
y_true = np.array([1, 1, 1, 0, 0])
y_pred = np.array([1, 1, 0, 0, 0])
calibrator.update(y_true, y_pred)

# Get reliability metrics
print(f"Expected F1: {calibrator.expected_f1:.2%}")

# Weight a score
raw_score = 0.85
weighted_score = reliability_weighted_score(raw_score, calibrator, metric='f1')
print(f"Raw: {raw_score:.2%} → Weighted: {weighted_score:.2%}")
```

### Managing Multiple Detectors

```python
from evaluation.reliability_calibration import CalibratorManager
from pathlib import Path

# Create manager
manager = CalibratorManager(Path("calibrators/"))

# Update detectors
manager.update_from_ground_truth(
    "responsibility_detector",
    y_true=np.array([1, 1, 0, 0]),
    y_pred=np.array([1, 0, 0, 1])
)

# Get all statistics
stats = manager.get_all_stats()
for detector, stat in stats.items():
    print(f"{detector}: F1={stat['f1']:.2%}")

# Save all calibrators
manager.save_all()
```

### Ground Truth Collection

```python
from evaluation.ground_truth_collector import GroundTruthCollector
from pathlib import Path

# Create collector
collector = GroundTruthCollector(Path("ground_truth/"))

# Collect predictions during processing
collector.add_prediction(
    detector_name="responsibility_detector",
    evidence_id="ev_001",
    prediction=1,
    confidence=0.85,
    context={"text": "El Ministerio implementará..."}
)

# Export for labeling
collector.export_for_labeling(Path("to_label.csv"))

# After human labeling, import and update
labeled_data = collector.import_labeled_data(Path("labeled.csv"))
# labeled_data = {"detector_name": [(y_true, y_pred), ...]}
```

### Complete Workflow

```python
from evaluation.reliability_calibration import calibration_workflow, CalibratorManager
from pathlib import Path

# Create manager
manager = CalibratorManager(Path("calibrators/"))

# Prepare labeled data
labeled_data = {
    "responsibility_detector": [(1, 1), (1, 0), (0, 0), (0, 1)],
    "contradiction_detector": [(1, 1), (1, 1), (0, 0), (0, 0)]
}

# Run workflow (updates and reports)
stats = calibration_workflow(manager, labeled_data)
```

## Integration with Detectors

### Recommended Integration Pattern

```python
class EnhancedDetector:
    def __init__(self, calibrator_manager: CalibratorManager):
        self.calibrator = calibrator_manager.get_calibrator("detector_name")
        self.gt_collector = GroundTruthCollector(Path("ground_truth/detector_name"))
    
    def detect(self, text: str, evidence_id: str = None) -> Dict:
        # Perform detection
        raw_result = self._detect_raw(text)
        raw_score = raw_result["confidence"]
        
        # Calculate calibrated score
        calibrated_score = reliability_weighted_score(
            raw_score,
            self.calibrator,
            metric='f1'
        )
        
        # Collect for future labeling
        if evidence_id:
            self.gt_collector.add_prediction(
                detector_name="detector_name",
                evidence_id=evidence_id,
                prediction=1 if raw_result["detected"] else 0,
                confidence=raw_score,
                context={"text": text[:200]}
            )
        
        return {
            "raw_confidence": raw_score,
            "calibrated_confidence": calibrated_score,
            "detector_reliability": self.calibrator.expected_f1
        }
```

## Calibration Workflow

### Step 1: Initial Deployment

```bash
# Detectors start with uniform prior Beta(1, 1)
# This represents 50% expected precision/recall with high uncertainty
```

### Step 2: Collect Predictions

```python
# During pipeline execution, collect predictions
detector.detect(text, evidence_id="ev_001")
# Predictions are stored with context for review
```

### Step 3: Export for Labeling

```bash
# Export accumulated predictions
detector.export_for_labeling(Path("to_label_batch1.csv"))
# Produces CSV with columns: detector, evidence_id, prediction, confidence, context, ground_truth
```

### Step 4: Manual Labeling

```
# Human reviewers fill in ground_truth column
# ground_truth: 1 (correct), 0 (incorrect), or blank (skip)
```

### Step 5: Update Calibrators

```python
# Import labeled data and update calibrators
detector.update_from_labeled_data(Path("labeled_batch1.csv"))
# Calibrators automatically update Beta posteriors
```

### Step 6: Monitor Improvements

```python
# Check updated statistics
stats = detector.calibrator.get_stats()
print(f"F1: {stats['f1']:.2%}")
print(f"95% CI: {stats['precision']['interval']}")
```

## Expected Improvements

Based on the problem statement and Bayesian calibration theory:

1. **Detector Calibration**: After 200+ labels per detector:
   - Average precision > 75%
   - Credible interval width < 0.15
   - Reliable uncertainty estimates

2. **Ensemble Improvement**: 
   - Expected +8-12% F1 improvement over simple voting
   - Better handling of unreliable detectors
   - Automatic down-weighting of noisy detectors

3. **Operational Benefits**:
   - Continuous learning from labeled data
   - Quantified uncertainty for each detector
   - Transparent reliability metrics
   - Persistent calibration state

## Testing

### Run Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run calibration tests
python3 -m pytest test_reliability_calibration.py -v

# Run ground truth collector tests
python3 -m pytest test_ground_truth_collector.py -v

# Run all tests
python3 -m pytest test_reliability_calibration.py test_ground_truth_collector.py -v
```

### Example Scripts

```bash
# Basic usage examples
python3 example_reliability_calibration.py

# Detector integration examples
python3 example_detector_integration.py
```

## Files

### Core Implementation
- `evaluation/reliability_calibration.py` - Core calibration classes
- `evaluation/ground_truth_collector.py` - Ground truth collection
- `evaluation/__init__.py` - Package initialization

### Tests
- `test_reliability_calibration.py` - 23 tests for calibration
- `test_ground_truth_collector.py` - 12 tests for GT collection

### Examples
- `example_reliability_calibration.py` - Usage examples
- `example_detector_integration.py` - Integration patterns

### Storage Directories
- `calibrators/` - Persisted calibrator state (JSON)
- `ground_truth/` - Collected predictions and labels

## Future Enhancements

1. **Multi-class Calibration**: Extend beyond binary to multi-class problems
2. **Temporal Calibration**: Track reliability over time windows
3. **Stratified Calibration**: Separate calibrators by data characteristics
4. **Automated Labeling**: Integration with active learning for efficient labeling
5. **Calibration Monitoring**: Dashboard for tracking reliability metrics

## References

- Beta-Bernoulli conjugate prior: Murphy, K. (2012). Machine Learning: A Probabilistic Perspective
- Calibration theory: Guo, C., et al. (2017). On Calibration of Modern Neural Networks
- Bayesian inference: Gelman, A., et al. (2013). Bayesian Data Analysis

## Support

For issues or questions about the calibration system:
1. Check the example scripts for usage patterns
2. Review test files for edge cases
3. Consult the inline documentation in source files
