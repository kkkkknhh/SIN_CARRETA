"""
Effects Logger for Causal Data Accumulation

Accumulates observational data from multiple plans to enable causal effect estimation.
Supports:
- Data accumulation across multiple plans
- Persistence to disk
- Threshold checking for minimum observations
- Multiple effect types management

Author: MINIMINIMOON Team
Date: 2025
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EffectsLogger:
    """
    Acumula pares (X, Y) de múltiples planes para análisis causal.
    
    Attributes:
        storage_path: Directory donde se guardan los datos
        effects_db: Dict que mapea effect_name → lista de observaciones
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize effects logger.
        
        Args:
            storage_path: Directory path for storing accumulated data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.effects_db = {}  # {effect_name: [(x, y), ...]}
    
    def log_effect(self, effect_name: str, x_value: float, y_value: float, 
                   plan_id: str):
        """
        Registra observación de un efecto causal.
        
        Args:
            effect_name: Nombre del efecto (e.g., "responsibility_to_feasibility")
            x_value: Variable independiente (causa)
            y_value: Variable dependiente (efecto)
            plan_id: Identificador del plan
        """
        if effect_name not in self.effects_db:
            self.effects_db[effect_name] = []
        
        # Use timestamp if pandas available, otherwise use simple counter
        if PANDAS_AVAILABLE:
            import pandas as pd
            timestamp = pd.Timestamp.now().isoformat()
        else:
            timestamp = f"observation_{len(self.effects_db[effect_name])}"
        
        self.effects_db[effect_name].append({
            'x': x_value,
            'y': y_value,
            'plan_id': plan_id,
            'timestamp': timestamp
        })
        
        logger.debug(f"Logged effect {effect_name}: x={x_value:.3f}, y={y_value:.3f}, plan={plan_id}")
    
    def get_effect_data(self, effect_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna arrays (X, Y) para un efecto.
        
        Args:
            effect_name: Nombre del efecto a recuperar
            
        Returns:
            Tuple (x_array, y_array) con las observaciones
        """
        if effect_name not in self.effects_db:
            return np.array([]), np.array([])
        
        data = self.effects_db[effect_name]
        x = np.array([d['x'] for d in data])
        y = np.array([d['y'] for d in data])
        return x, y
    
    def has_sufficient_data(self, effect_name: str, min_obs: int = 30) -> bool:
        """
        Verifica si hay suficientes observaciones para análisis causal.
        
        Args:
            effect_name: Nombre del efecto
            min_obs: Número mínimo de observaciones requeridas
            
        Returns:
            True si hay suficientes datos, False en caso contrario
        """
        if effect_name not in self.effects_db:
            return False
        return len(self.effects_db[effect_name]) >= min_obs
    
    def get_observation_count(self, effect_name: str) -> int:
        """
        Get the number of observations for an effect.
        
        Args:
            effect_name: Name of the effect
            
        Returns:
            Number of observations
        """
        if effect_name not in self.effects_db:
            return 0
        return len(self.effects_db[effect_name])
    
    def save(self):
        """Persiste base de datos a archivos JSON."""
        for effect_name, data in self.effects_db.items():
            path = self.storage_path / f"{effect_name}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        logger.info(f"Efectos guardados: {len(self.effects_db)} tipos en {self.storage_path}")
    
    def load(self):
        """Carga base de datos desde archivos JSON."""
        for json_file in self.storage_path.glob("*.json"):
            effect_name = json_file.stem
            with open(json_file, 'r') as f:
                self.effects_db[effect_name] = json.load(f)
        logger.info(f"Efectos cargados: {len(self.effects_db)} tipos desde {self.storage_path}")
    
    def get_all_effects(self) -> List[str]:
        """
        Get list of all effect names with data.
        
        Returns:
            List of effect names
        """
        return list(self.effects_db.keys())
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about accumulated data.
        
        Returns:
            Dict with statistics for each effect
        """
        stats = {}
        for effect_name in self.effects_db:
            x, y = self.get_effect_data(effect_name)
            if len(x) > 0:
                stats[effect_name] = {
                    'n_observations': len(x),
                    'x_mean': float(np.mean(x)),
                    'x_std': float(np.std(x)),
                    'y_mean': float(np.mean(y)),
                    'y_std': float(np.std(y)),
                    'correlation': float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
                }
        return stats


class CausalEffectsManager:
    """
    Gestiona estimadores de efectos causales.
    Solo activa estimación cuando hay suficientes datos.
    
    Attributes:
        logger: EffectsLogger instance para datos acumulados
        min_obs: Número mínimo de observaciones requeridas
        estimators: Dict de estimadores ajustados por efecto
    """
    
    def __init__(self, logger: EffectsLogger, min_obs: int = 30):
        """
        Initialize causal effects manager.
        
        Args:
            logger: EffectsLogger instance with accumulated data
            min_obs: Minimum number of observations required for estimation
        """
        self.logger = logger
        self.min_obs = min_obs
        self.estimators = {}
    
    def estimate_effect(self, effect_name: str) -> 'Optional[BayesLinearEffect]':
        """
        Estima efecto causal si hay suficientes datos.
        
        Args:
            effect_name: Nombre del efecto a estimar
        
        Returns:
            Estimador ajustado o None si datos insuficientes
        """
        # Import here to avoid circular dependency
        from evaluation.causal_effect_estimator import BayesLinearEffect
        
        if not self.logger.has_sufficient_data(effect_name, self.min_obs):
            n_obs = self.logger.get_observation_count(effect_name)
            logger.warning(f"Datos insuficientes para {effect_name}: {n_obs}/{self.min_obs} observaciones")
            return None
        
        x, y = self.logger.get_effect_data(effect_name)
        
        model = BayesLinearEffect()
        model.fit(x, y)
        
        self.estimators[effect_name] = model
        
        summary = model.get_summary()
        logger.info(f"Efecto {effect_name} estimado: β₁={summary['beta1_mean']:.3f} "
                   f"{summary['beta1_interval_95']}")
        
        return model
    
    def get_all_effects(self) -> Dict[str, Dict]:
        """
        Retorna resumen de todos los efectos estimados.
        
        Returns:
            Dict mapping effect_name → summary dict
        """
        return {name: model.get_summary() 
                for name, model in self.estimators.items()}
    
    def estimate_all_available(self) -> Dict[str, 'Optional[BayesLinearEffect]']:
        """
        Estimate all effects that have sufficient data.
        
        Returns:
            Dict mapping effect_name → fitted model (or None if insufficient data)
        """
        results = {}
        for effect_name in self.logger.get_all_effects():
            results[effect_name] = self.estimate_effect(effect_name)
        return results


def periodic_causal_analysis(effects_logger: EffectsLogger, min_obs: int = 30) -> Dict:
    """
    Ejecutar periódicamente para actualizar efectos causales.
    
    Args:
        effects_logger: EffectsLogger con datos acumulados
        min_obs: Número mínimo de observaciones para estimación
    
    Returns:
        Dict con resultados del análisis
    """
    manager = CausalEffectsManager(effects_logger, min_obs=min_obs)
    
    effects_to_estimate = [
        "responsibility_to_feasibility",
        "monetary_to_feasibility",
        "contradictions_to_coherence",
        "teoria_cambio_to_kpi"
    ]
    
    results = {}
    for effect_name in effects_to_estimate:
        if effect_name in effects_logger.get_all_effects():
            model = manager.estimate_effect(effect_name)
            if model:
                results[effect_name] = model.get_summary()
    
    # Reportar
    print("=== ANÁLISIS DE EFECTOS CAUSALES ===")
    for effect, summary in results.items():
        if summary.get('fitted'):
            print(f"\n{effect}:")
            print(f"  β₁ (efecto): {summary['beta1_mean']:.3f} "
                  f"{summary['beta1_interval_95']}")
            print(f"  Significativo: {summary['effect_significant']}")
            print(f"  P(efecto > 0): {summary['prob_positive_effect']:.1%}")
    
    return results
