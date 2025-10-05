import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _install_stubs():
    # Install minimal stubs for heavy external deps used by Decatalogo_evaluador
    for module_name in ["spacy", "sentence_transformers", "pdfplumber"]:
        sys.modules.pop(module_name, None)

    class _DummyDoc:
        ents = []

    class _DummyNLP:
        def __call__(self, _text):
            return _DummyDoc()

    dummy_spacy = ModuleType("spacy")
    dummy_spacy.load = lambda *args, **kwargs: _DummyNLP()
    dummy_spacy.tokens = SimpleNamespace(Doc=_DummyDoc)
    sys.modules["spacy"] = dummy_spacy

    class _DummySentenceModel:
        @staticmethod
        def encode(*args, **kwargs):
            return [0]

        def to(self, *args, **kwargs):
            return self

    dummy_sentence = ModuleType("sentence_transformers")
    dummy_sentence.SentenceTransformer = lambda *args, **kwargs: _DummySentenceModel()
    dummy_sentence.util = SimpleNamespace(
        pytorch_cos_sim=lambda *args, **kwargs: 0)
    sys.modules["sentence_transformers"] = dummy_sentence

    sys.modules["pdfplumber"] = MagicMock()


class TestDecatalogoEvaluatorIntegrationExtra(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        # Ensure fresh import
        sys.modules.pop("Decatalogo_evaluador", None)
        import Decatalogo_evaluador as de

        importlib.reload(de)
        self.module = de
        self.evaluator = de.IndustrialDecatalogoEvaluatorFull()

    @staticmethod
    def _sample_evidence():
        return {
            "indicadores": [
                {
                    "texto": "La línea base actual es 50 beneficiarios y la meta es 120 en 2025."
                },
            ],
            "metas": [
                {
                    "texto": "Meta transformadora alcanzar 120 familias antes de diciembre de 2025."
                }
            ],
            "recursos": [{"texto": "Se asignan $500 millones para el programa."}],
            "plazos": [{"texto": "Plan plurianual 2024-2027"}],
            "riesgos": [
                {"texto": "Riesgo de retraso por capacidad operativa limitada."}
            ],
            "responsables": [{"texto": "El Ministerio de Salud será responsable."}],
        }

    def test_evaluator_has_detector_and_uses_it(self):
        self.assertTrue(hasattr(self.evaluator, "contradiction_detector"))
        cd = self.evaluator.contradiction_detector
        self.assertIsNotNone(cd)
        # Run a quick detection
        analysis = cd.detect_contradictions(
            "La meta es 95%, sin embargo no hay presupuesto"
        )
        self.assertIsInstance(analysis, self.module.ContradictionAnalysis)

    def test_evaluar_punto_completo_returns_expected_tuple(self):
        evidencia = self._sample_evidence()
        result = self.evaluator.evaluar_punto_completo(evidencia, punto_id=1)
        # Expect a tuple of (EvaluacionPuntoCompleto, AnalisisEvidenciaDecalogo, ResultadoDimensionIndustrial|None)
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 2)
        evaluacion_punto, analisis = result[0], result[1]
        self.assertTrue(hasattr(evaluacion_punto, "puntaje_agregado_punto"))
        self.assertTrue(hasattr(analisis, "contradicciones"))

    def test_generar_reporte_final_structure(self):
        evidencias_por_punto = {
            1: self._sample_evidence(), 2: self._sample_evidence()}
        reporte = self.evaluator.generar_reporte_final(
            evidencias_por_punto, nombre_plan="Plan Extra"
        )
        self.assertIsInstance(reporte, self.module.ReporteFinalDecatalogo)
        self.assertIn("resultados_industriales", reporte.anexos_serializables)


if __name__ == "__main__":
    unittest.main()
