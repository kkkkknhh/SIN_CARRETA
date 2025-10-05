import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _load_evaluator_with_stubs():
    for module_name in [
        "Decatalogo_evaluador",
        "Decatalogo_principal",
        "responsibility_detector",
        "spacy",
        "sentence_transformers",
        "pdfplumber",
    ]:
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
            return MagicMock()

        def to(self, *args, **kwargs):
            return self

    dummy_sentence = ModuleType("sentence_transformers")
    dummy_sentence.SentenceTransformer = lambda *args, **kwargs: _DummySentenceModel()
    dummy_sentence.util = SimpleNamespace(
        pytorch_cos_sim=lambda *args, **kwargs: MagicMock()
    )
    sys.modules["sentence_transformers"] = dummy_sentence

    sys.modules["pdfplumber"] = MagicMock()

    module = importlib.import_module("Decatalogo_evaluador")
    importlib.reload(module)
    return module


def _build_sample_evidence():
    return {
        "indicadores": [
            {
                "texto": "La línea base actual es 50 beneficiarios y la meta es 120 en 2025."
            },
            {
                "texto": "Objetivo específico incrementar cobertura 30% con horizonte temporal 2024."
            },
        ],
        "metas": [
            {
                "texto": "Meta transformadora alcanzar 120 familias antes de diciembre de 2025."
            }
        ],
        "recursos": [
            {
                "texto": "Se asignan $500 millones para el programa, codificado en el plan plurianual."
            }
        ],
        "plazos": [{"texto": "Plan plurianual 2024-2027 con hitos trimestrales."}],
        "riesgos": [{"texto": "Riesgo de retraso por capacidad operativa limitada."}],
        "responsables": [
            {"texto": "El Ministerio de Salud será responsable directo del programa."}
        ],
    }


def test_evaluador_genera_resultado_industrial():
    module = _load_evaluator_with_stubs()
    evaluator = module.IndustrialDecatalogoEvaluatorFull()
    evidencia = _build_sample_evidence()

    evaluacion_punto, analisis, resultado_industrial = evaluator.evaluar_punto_completo(
        evidencia, punto_id=1
    )

    assert evaluacion_punto.puntaje_agregado_punto > 0
    assert analisis.indicador_scores
    assert isinstance(resultado_industrial,
                      module.ResultadoDimensionIndustrial)
    assert resultado_industrial.recomendaciones
    assert resultado_industrial.evaluacion_causal.factibilidad_operativa > 0


def test_generar_reporte_final_contiene_trazabilidad():
    module = _load_evaluator_with_stubs()
    evaluator = module.IndustrialDecatalogoEvaluatorFull()
    evidencias_por_punto = {
        1: _build_sample_evidence(), 2: _build_sample_evidence()}

    reporte = evaluator.generar_reporte_final(
        evidencias_por_punto, nombre_plan="Plan Piloto"
    )

    assert isinstance(reporte, module.ReporteFinalDecatalogo)
    assert reporte.resultados_dimension
    assert reporte.anexos_serializables["resultados_industriales"]
    assert len(reporte.anexos_serializables["resultados_industriales"]) == len(
        reporte.resultados_dimension
    )
    assert "brechas_globales" in reporte.reporte_macro
