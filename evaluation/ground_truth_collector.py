"""
Ground Truth Collection Module

Provides infrastructure for collecting ground truth labels and updating
reliability calibrators.

Features:
- GroundTruthCollector for tracking predictions
- Export for manual labeling
- Import labeled data
- Integration with CalibratorManager

Usage:
    collector = GroundTruthCollector(Path("ground_truth/"))
    collector.add_prediction("detector_name", "evidence_id", prediction=1, confidence=0.8, context={})
    collector.export_for_labeling(Path("to_label.csv"))
    # ... human labeling ...
    labeled_data = collector.import_labeled_data(Path("labeled.csv"))
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Check if pandas is available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Ground truth collector will use JSON fallback.")


class GroundTruthCollector:
    """
    Sistema para colectar labels verdaderos y actualizar calibradores.
    En producción, esto vendría de auditoría manual.
    """
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.pending_labels = []
    
    def add_prediction(self, detector_name: str, evidence_id: str, 
                      prediction: int, confidence: float, context: Dict):
        """
        Registra predicción para futura labeling.
        
        Args:
            detector_name: Nombre del detector
            evidence_id: ID único de la evidencia
            prediction: Predicción binaria {0, 1}
            confidence: Confianza del detector
            context: Contexto adicional para revisión humana
        """
        item = {
            'detector': detector_name,
            'evidence_id': evidence_id,
            'prediction': prediction,
            'confidence': confidence,
            'context': json.dumps(context) if isinstance(context, dict) else str(context),
            'timestamp': datetime.now().isoformat(),
            'ground_truth': None  # A llenar manualmente
        }
        self.pending_labels.append(item)
    
    def export_for_labeling(self, output_file: Path):
        """Exporta items pendientes a CSV/JSON para labeling manual"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if PANDAS_AVAILABLE and output_file.suffix == '.csv':
            df = pd.DataFrame(self.pending_labels)
            df.to_csv(output_file, index=False)
            logger.info(f"{len(df)} items exportados para labeling: {output_file}")
        else:
            # Fallback to JSON
            json_file = output_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(self.pending_labels, f, indent=2)
            logger.info(f"{len(self.pending_labels)} items exportados a JSON: {json_file}")
    
    @staticmethod
    def import_labeled_data(input_file: Path) -> Dict[str, List[Tuple]]:
        """
        Importa datos labeleados y agrupa por detector.
        
        Returns:
            Dict[detector_name] = [(y_true, y_pred), ...]
        """
        if PANDAS_AVAILABLE and input_file.suffix == '.csv':
            df = pd.read_csv(input_file)
            
            # Filtrar solo items labeleados
            df = df[df['ground_truth'].notna()]
            
            # Agrupar por detector
            grouped = {}
            for detector in df['detector'].unique():
                subset = df[df['detector'] == detector]
                pairs = list(zip(subset['ground_truth'].astype(int), 
                               subset['prediction'].astype(int)))
                grouped[detector] = pairs
            
            logger.info(f"Importados {len(df)} labels de {len(grouped)} detectores")
            return grouped
        else:
            # Fallback to JSON
            json_file = input_file.with_suffix('.json')
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Filter only labeled items
            labeled = [item for item in data if item.get('ground_truth') is not None]
            
            # Group by detector
            grouped = {}
            for item in labeled:
                detector = item['detector']
                if detector not in grouped:
                    grouped[detector] = []
                grouped[detector].append((int(item['ground_truth']), int(item['prediction'])))
            
            logger.info(f"Importados {len(labeled)} labels de {len(grouped)} detectores desde JSON")
            return grouped
    
    def clear_pending(self):
        """Limpia los items pendientes después de exportar"""
        self.pending_labels = []
        logger.info("Pending labels cleared")
    
    def get_pending_count(self) -> int:
        """Retorna el número de items pendientes de labelear"""
        return len(self.pending_labels)
    
    def save_checkpoint(self, checkpoint_file: Path = None):
        """Guarda checkpoint de items pendientes"""
        if checkpoint_file is None:
            checkpoint_file = self.storage_path / "pending_checkpoint.json"
        
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(self.pending_labels, f, indent=2)
        logger.info(f"Checkpoint guardado: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_file: Path = None):
        """Carga checkpoint de items pendientes"""
        if checkpoint_file is None:
            checkpoint_file = self.storage_path / "pending_checkpoint.json"
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                self.pending_labels = json.load(f)
            logger.info(f"Checkpoint cargado: {checkpoint_file} ({len(self.pending_labels)} items)")
        else:
            logger.warning(f"No checkpoint encontrado en {checkpoint_file}")


# ============================================================================
# UTILIDADES PARA INTEGRACIÓN CON DETECTORES
# ============================================================================

def create_ground_truth_collector(detector_name: str, base_path: Path = None) -> GroundTruthCollector:
    """
    Factory para crear colector específico de un detector.
    
    Args:
        detector_name: Nombre del detector
        base_path: Path base (default: ground_truth/)
    
    Returns:
        GroundTruthCollector configurado
    """
    if base_path is None:
        base_path = Path("ground_truth")
    
    storage_path = base_path / detector_name
    return GroundTruthCollector(storage_path)
