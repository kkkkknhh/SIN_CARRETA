"""
Reliability Calibration Module

Implements Bayesian reliability calibration using Beta distribution to update
expected precision of detectors based on real performance data.

Features:
- ReliabilityCalibrator class with Beta(a, b) posterior
- Precision, recall, and F1 tracking
- Credible interval calculation
- Persistence (save/load state)
- reliability_weighted_score function for score weighting

Usage:
    calibrator = ReliabilityCalibrator(detector_name="responsibility_detector")
    calibrator.update(y_true, y_pred)
    weighted_score = reliability_weighted_score(raw_score, calibrator, metric='f1')
"""

import numpy as np
from scipy.stats import beta
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityCalibrator:
    """
    Calibra fiabilidad de un detector usando Beta(a, b).
    Trackea precisión, recall, y F1 de manera bayesiana.
    """
    # Parámetros Beta para precisión
    precision_a: float = 1.0
    precision_b: float = 1.0
    
    # Parámetros Beta para recall
    recall_a: float = 1.0
    recall_b: float = 1.0
    
    # Metadata
    detector_name: str = "unknown"
    n_updates: int = 0
    
    def update_precision(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Actualiza posterior de precisión.
        Precisión = TP / (TP + FP)
        
        Args:
            y_true: Ground truth binario {0, 1}
            y_pred: Predicciones binarias {0, 1}
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        
        # Contar TP y FP
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        
        # Actualizar Beta (conjugado Bernoulli)
        self.precision_a += tp
        self.precision_b += fp
        self.n_updates += len(y_true)
        
        logger.debug(f"{self.detector_name}: Precision updated TP={tp}, FP={fp}")
    
    def update_recall(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Actualiza posterior de recall.
        Recall = TP / (TP + FN)
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        
        self.recall_a += tp
        self.recall_b += fn
        
        logger.debug(f"{self.detector_name}: Recall updated TP={tp}, FN={fn}")
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Actualiza ambos: precisión y recall."""
        self.update_precision(y_true, y_pred)
        self.update_recall(y_true, y_pred)
    
    @property
    def expected_precision(self) -> float:
        """Media posterior E[precision] = a/(a+b)"""
        return self.precision_a / (self.precision_a + self.precision_b)
    
    @property
    def expected_recall(self) -> float:
        """Media posterior E[recall]"""
        return self.recall_a / (self.recall_a + self.recall_b)
    
    @property
    def expected_f1(self) -> float:
        """F1 aproximado usando medias posteriores"""
        p = self.expected_precision
        r = self.expected_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def precision_credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Intervalo de credibilidad para precisión"""
        alpha = (1 - level) / 2
        lo = beta.ppf(alpha, self.precision_a, self.precision_b)
        hi = beta.ppf(1 - alpha, self.precision_a, self.precision_b)
        return (lo, hi)
    
    def recall_credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Intervalo de credibilidad para recall"""
        alpha = (1 - level) / 2
        lo = beta.ppf(alpha, self.recall_a, self.recall_b)
        hi = beta.ppf(1 - alpha, self.recall_a, self.recall_b)
        return (lo, hi)
    
    def get_stats(self) -> Dict:
        """Retorna todas las estadísticas"""
        return {
            'detector': self.detector_name,
            'n_updates': self.n_updates,
            'precision': {
                'mean': self.expected_precision,
                'interval': self.precision_credible_interval()
            },
            'recall': {
                'mean': self.expected_recall,
                'interval': self.recall_credible_interval()
            },
            'f1': self.expected_f1,
            'beta_params': {
                'precision': (self.precision_a, self.precision_b),
                'recall': (self.recall_a, self.recall_b)
            }
        }
    
    def save(self, path: Path):
        """Persiste calibrador a JSON"""
        data = {
            'precision_a': self.precision_a,
            'precision_b': self.precision_b,
            'recall_a': self.recall_a,
            'recall_b': self.recall_b,
            'detector_name': self.detector_name,
            'n_updates': self.n_updates
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibrador guardado en {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ReliabilityCalibrator':
        """Carga calibrador desde JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def reliability_weighted_score(raw_score: float, calibrator: ReliabilityCalibrator,
                               metric: str = 'precision') -> float:
    """
    Pondera score por fiabilidad esperada del detector.
    
    Args:
        raw_score: Score original del detector (0-1)
        calibrator: Calibrador del detector
        metric: 'precision', 'recall', o 'f1'
    
    Returns:
        Score calibrado
    """
    if metric == 'precision':
        weight = calibrator.expected_precision
    elif metric == 'recall':
        weight = calibrator.expected_recall
    elif metric == 'f1':
        weight = calibrator.expected_f1
    else:
        raise ValueError(f"Métrica desconocida: {metric}")
    
    return raw_score * weight


# ============================================================================
# MANAGER DE CALIBRADORES
# ============================================================================

class CalibratorManager:
    """Gestiona calibradores de todos los detectores"""
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.calibrators = {}
    
    def get_calibrator(self, detector_name: str) -> ReliabilityCalibrator:
        """Obtiene o crea calibrador para un detector"""
        if detector_name not in self.calibrators:
            # Intentar cargar desde disco
            path = self.storage_dir / f"{detector_name}_calibrator.json"
            if path.exists():
                self.calibrators[detector_name] = ReliabilityCalibrator.load(path)
                logger.info(f"Calibrador cargado desde {path}")
            else:
                self.calibrators[detector_name] = ReliabilityCalibrator(
                    detector_name=detector_name
                )
                logger.info(f"Calibrador nuevo creado para {detector_name}")
        return self.calibrators[detector_name]
    
    def update_from_ground_truth(self, detector_name: str, 
                                y_true: np.ndarray, y_pred: np.ndarray):
        """Actualiza calibrador con nuevos datos"""
        calibrator = self.get_calibrator(detector_name)
        calibrator.update(y_true, y_pred)
        self.save_calibrator(detector_name)
    
    def save_calibrator(self, detector_name: str):
        """Persiste calibrador a disco"""
        path = self.storage_dir / f"{detector_name}_calibrator.json"
        self.calibrators[detector_name].save(path)
    
    def save_all(self):
        """Persiste todos los calibradores"""
        for name in self.calibrators:
            self.save_calibrator(name)
    
    def get_all_stats(self) -> Dict:
        """Retorna stats de todos los calibradores"""
        return {name: cal.get_stats() 
                for name, cal in self.calibrators.items()}


# ============================================================================
# WORKFLOW DE CALIBRACIÓN
# ============================================================================

def calibration_workflow(manager: CalibratorManager, 
                        labeled_data: Dict[str, List[Tuple[int, int]]]):
    """
    Workflow completo de calibración:
    1. Importar labels y actualizar calibradores
    2. Validar mejora en métricas
    3. Reportar estadísticas
    
    Args:
        manager: CalibratorManager instance
        labeled_data: Dict[detector_name] = [(y_true, y_pred), ...]
    """
    for detector_name, pairs in labeled_data.items():
        y_true = np.array([p[0] for p in pairs])
        y_pred = np.array([p[1] for p in pairs])
        manager.update_from_ground_truth(detector_name, y_true, y_pred)
    
    # Reportar estadísticas
    stats = manager.get_all_stats()
    print("=== CALIBRACIÓN ACTUALIZADA ===")
    for detector, stat in stats.items():
        print(f"\n{detector}:")
        print(f"  Precisión: {stat['precision']['mean']:.2%} "
              f"{stat['precision']['interval']}")
        print(f"  Recall: {stat['recall']['mean']:.2%} "
              f"{stat['recall']['interval']}")
        print(f"  F1: {stat['f1']:.2%}")
        print(f"  Actualizaciones: {stat['n_updates']}")
    
    return stats
