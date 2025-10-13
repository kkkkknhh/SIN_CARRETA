"""Mock Decatalogo_principal for testing orchestrator trace."""


BUNDLE = {
    "version": "9.0-mock",
    "categories": [
        {"id": "D1", "name": "INSUMOS"},
        {"id": "D2", "name": "ACTIVIDADES"},
        {"id": "D3", "name": "PRODUCTOS"},
        {"id": "D4", "name": "RESULTADOS"},
        {"id": "D5", "name": "IMPACTOS"},
        {"id": "D6", "name": "CAUSALIDAD"}
    ],
    "questions": []
}


class ExtractorEvidenciaIndustrialAvanzado:
    """Mock extractor for testing."""
    
    def __init__(self, bundle):
        self.bundle = bundle
    
    def evaluate_from_evidence(self, evidence_registry):
        """Mock evaluation from evidence."""
        return {
            "evaluation_type": "decalogo",
            "total_questions": 300,
            "evaluated": 0,
            "mock": True
        }
