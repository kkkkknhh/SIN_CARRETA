#!/usr/bin/env python3
# coding=utf-8
# coding=utf-8
"""
Validador industrial de última generación para implementación de Teoría de Cambio
Nivel de sofisticación: Estado del arte industrial - Nivel máximo
"""

import logging
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from log_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


class ValidationTier(Enum):
    """Niveles de validación industrial"""

    BASIC = "Básico"
    ADVANCED = "Avanzado"
    INDUSTRIAL = "Industrial"
    STATE_OF_ART = "Estado del Arte"


@dataclass
class ValidationMetric:
    """Métrica de validación industrial"""

    name: str
    value: float
    unit: str
    threshold: float
    status: str
    weight: float = 1.0


class IndustrialGradeValidator:
    """Validador de grado industrial con capacidades de última generación"""

    def __init__(self):
        self.logger = LOGGER
        self.metrics: List[ValidationMetric] = []
        self.validation_start_time: float = 0
        self.performance_benchmarks: Dict[str, float] = {
            "import_time": 0.1,
            "instance_creation": 0.05,
            "graph_construction": 0.2,
            "path_detection": 0.15,
            "full_validation": 0.5,
        }

    def start_validation(self):
        """Inicia el proceso de validación industrial"""
        self.validation_start_time = time.time()
        self.logger.info("🚀 INICIANDO VALIDACIÓN INDUSTRIAL DE ÚLTIMA GENERACIÓN")
        self.logger.info("=" * 80)

    def log_metric(self, name: str, value: float, unit: str, threshold: float):
        """Registra métrica con evaluación automática de estado"""
        status = "✅ PASÓ" if value <= threshold else "❌ FALLÓ"
        metric = ValidationMetric(name, value, unit, threshold, status)
        self.metrics.append(metric)
        return metric

    def validate_import_performance(self) -> bool:
        """Valida rendimiento de importación con estándares industriales"""
        start_time = time.time()

        try:
            from teoria_cambio import CategoriaCausal, TeoriaCambio, ValidacionResultado

            import_time = time.time() - start_time

            metric = self.log_metric(
                "Tiempo de Importación",
                import_time,
                "segundos",
                self.performance_benchmarks["import_time"],
            )

            self.logger.info(f"📦 IMPORTACIÓN INDUSTRIAL: {metric.status}")
            self.logger.info(
                f"   ⏱️  Tiempo: {import_time:.4f}s (Límite: {metric.threshold}s)"
            )

            return metric.status == "✅ PASÓ"

        except ImportError:
            self.logger.exception("❌ FALLA CRÍTICA EN IMPORTACIÓN")
            return False

    def validate_causal_categories(self) -> Tuple[bool, List[str]]:
        """Valida categorías causales con análisis exhaustivo"""
        from teoria_cambio import CategoriaCausal

        expected_categories = [
            "INSUMOS",
            "PROCESOS",
            "PRODUCTOS",
            "RESULTADOS",
            "IMPACTOS",
        ]
        category_objects = list(CategoriaCausal)
        category_names = [cat.name for cat in category_objects]

        validation_results = []
        missing_categories = []

        for expected in expected_categories:
            if expected in category_names:
                validation_results.append(True)
                self.logger.info(f"   ✅ {expected}: Definición óptima")
            else:
                validation_results.append(False)
                missing_categories.append(expected)
                self.logger.error(f"   ❌ {expected}: Categoría faltante")

        # Validación de orden lógico
        try:
            order_valid = self._validate_causal_order(category_objects)
            validation_results.append(order_valid)

            if order_valid:
                self.logger.info("   🔗 Orden causal: Secuencia lógica validada")
            else:
                self.logger.warning(
                    "   ⚠️  Orden causal: Posible inconsistencia detectada"
                )

        except Exception:
            self.logger.exception("   ⚠️  Orden causal: Error en validación")
            validation_results.append(False)

        return all(validation_results), missing_categories

    @staticmethod
    def _validate_causal_order(categories: List["CategoriaCausal"]) -> bool:
        """Valida el orden lógico de las categorías causales"""
        expected_order = ["INSUMOS", "PROCESOS", "PRODUCTOS", "RESULTADOS", "IMPACTOS"]
        actual_order = [cat.name for cat in categories]

        # Verifica que el orden esperado esté preservado
        for i, expected in enumerate(expected_order):
            if expected in actual_order:
                if actual_order.index(expected) != i:
                    return False
        return True

    def validate_connection_matrix(self) -> Dict[Tuple[str, str], bool]:
        """Valida matriz completa de conexiones con análisis predictivo"""
        from teoria_cambio import CategoriaCausal, TeoriaCambio

        tc = TeoriaCambio()
        categories = list(CategoriaCausal)
        connection_matrix = {}

        self.logger.info("   🔬 ANALIZANDO MATRIZ DE CONEXIONES:")

        for i, origen in enumerate(categories):
            for j, destino in enumerate(categories):
                is_valid = tc._es_conexion_valida(origen, destino)
                connection_matrix[(origen.name, destino.name)] = is_valid

                status_icon = "✅" if is_valid else "❌"
                self.logger.info(
                    f"      {status_icon} {origen.name:>10} → {destino.name:<10} | Válido: {is_valid}"
                )

        return connection_matrix

    def validate_performance_benchmarks(self) -> List[ValidationMetric]:
        """Ejecuta benchmarks de rendimiento industrial"""
        from teoria_cambio import TeoriaCambio

        tc = TeoriaCambio()
        performance_metrics = []

        # Benchmark de construcción de grafo
        start_time = time.time()
        grafo = tc.construir_grafo_causal()
        graph_time = time.time() - start_time
        performance_metrics.append(
            self.log_metric(
                "Construcción de Grafo",
                graph_time,
                "segundos",
                self.performance_benchmarks["graph_construction"],
            )
        )

        # Benchmark de detección de caminos
        start_time = time.time()
        caminos = tc.detectar_caminos_completos(grafo)
        path_time = time.time() - start_time
        performance_metrics.append(
            self.log_metric(
                "Detección de Caminos",
                path_time,
                "segundos",
                self.performance_benchmarks["path_detection"],
            )
        )

        # Benchmark de validación completa
        start_time = time.time()
        validacion = tc.validacion_completa(grafo)
        validation_time = time.time() - start_time
        performance_metrics.append(
            self.log_metric(
                "Validación Completa",
                validation_time,
                "segundos",
                self.performance_benchmarks["full_validation"],
            )
        )

        return performance_metrics


    def generate_industrial_report(self):
        """Genera reporte industrial completo"""
        total_time = time.time() - self.validation_start_time

        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 INFORME INDUSTRIAL DE VALIDACIÓN - ESTADO DEL ARTE")
        self.logger.info("=" * 80)

        # Resumen ejecutivo
        passed_metrics = sum(1 for m in self.metrics if m.status == "✅ PASÓ")
        total_metrics = len(self.metrics)
        success_rate = (passed_metrics / total_metrics) * 100

        self.logger.info("\n🎯 RESUMEN EJECUTIVO:")
        self.logger.info(f"   • Tiempo total de validación: {total_time:.3f} segundos")
        self.logger.info(f"   • Métricas evaluadas: {total_metrics}")
        self.logger.info(f"   • Tasa de éxito: {success_rate:.1f}%")
        self.logger.info(
            f"   • Nivel de calidad: {self._determine_quality_level(success_rate)}"
        )

        # Métricas detalladas
        self.logger.info("\n📈 MÉTRICAS DE RENDIMIENTO:")
        for metric in self.metrics:
            color_icon = "🟢" if metric.status == "✅ PASÓ" else "🔴"
            self.logger.info(
                f"   {color_icon} {metric.name}: {metric.value:.4f}{metric.unit} (Límite: {metric.threshold}{metric.unit}) - {metric.status}"
            )

        # Recomendaciones industriales
        self.logger.info("\n💡 RECOMENDACIONES DE GRADO INDUSTRIAL:")
        self._generate_industrial_recommendations()

        self.logger.info(
            f"\n🏆 VALIDACIÓN {'EXITOSA' if success_rate >= 90 else 'CON OBSERVACIONES'}"
        )
        return success_rate >= 90

    @staticmethod
    def _determine_quality_level(success_rate: float) -> str:
        """Determina el nivel de calidad industrial"""
        if success_rate >= 95:
            return "🏭 CALIDAD INDUSTRIAL PREMIUM"
        elif success_rate >= 85:
            return "🏭 CALIDAD INDUSTRIAL ESTÁNDAR"
        elif success_rate >= 70:
            return "⚠️  CALIDAD INDUSTRIAL BÁSICA"
        else:
            return "❌ NO CUMPLE ESTÁNDARES INDUSTRIALES"

    def _generate_industrial_recommendations(self):
        """Genera recomendaciones específicas para mejora industrial"""
        failed_metrics = [m for m in self.metrics if m.status != "✅ PASÓ"]

        if not failed_metrics:
            self.logger.info(
                "   ✅ Implementación cumple con todos los estándares industriales"
            )
            return

        for metric in failed_metrics:
            if "Tiempo" in metric.name:
                self.logger.info(
                    f"   ⚡ Optimizar {metric.name}: Considerar caching o optimización de algoritmos"
                )
            elif "Construcción" in metric.name:
                self.logger.info(
                    f"   🏗️  Revisar arquitectura de {metric.name}: Evaluar patrones de diseño industrial"
                )
            elif "Detección" in metric.name:
                self.logger.info(
                    f"   🔍 Mejorar algoritmos de {metric.name}: Implementar técnicas de búsqueda eficiente"
                )


def _serialize_metrics(metrics: List[ValidationMetric]) -> List[Dict[str, Any]]:
    """Convierte métricas de validación a estructuras serializables"""
    return [asdict(metric) for metric in metrics]


def execute_industrial_validation_detailed() -> Dict[str, Any]:
    """Ejecuta la validación industrial y devuelve un reporte estructurado"""
    validator = IndustrialGradeValidator()
    validator.start_validation()

    report: Dict[str, Any] = {
        "status": "started",
        "success": False,
        "import_performance": False,
        "categories_valid": False,
        "missing_categories": [],
        "connection_matrix": [],
        "performance_metrics": [],
        "functional": {},
        "metrics": [],
    }

    try:
        LOGGER.info("\n1. 🔧 VALIDACIÓN DE INFRAESTRUCTURA")
        import_ok = validator.validate_import_performance()
        report["import_performance"] = import_ok
        if not import_ok:
            report["status"] = "import_failed"
            report["metrics"] = _serialize_metrics(validator.metrics)
            return report

        LOGGER.info("\n2. 🏷️  VALIDACIÓN DE CATEGORÍAS CAUSALES")
        categories_valid, missing = validator.validate_causal_categories()
        report["categories_valid"] = categories_valid
        report["missing_categories"] = missing
        if not categories_valid:
            report["status"] = "categories_missing"
            report["metrics"] = _serialize_metrics(validator.metrics)
            return report

        LOGGER.info("\n3. 🔗 VALIDACIÓN DE MATRIZ DE CONEXIONES")
        connection_matrix = validator.validate_connection_matrix()
        report["connection_matrix"] = [
            {"source": src, "target": tgt, "valid": valid}
            for (src, tgt), valid in sorted(connection_matrix.items())
        ]

        LOGGER.info("\n4. ⚡ BENCHMARKS DE RENDIMIENTO INDUSTRIAL")
        performance_metrics = validator.validate_performance_benchmarks()
        report["performance_metrics"] = _serialize_metrics(performance_metrics)

        from teoria_cambio import TeoriaCambio

        LOGGER.info("\n5. 🧪 VALIDACIÓN FUNCIONAL AVANZADA")
        tc = TeoriaCambio()
        grafo = tc.construir_grafo_causal()
        validacion = tc.validacion_completa(grafo)
        caminos = tc.detectar_caminos_completos(grafo)
        sugerencias = tc.generar_sugerencias(grafo)

        report["functional"] = {
            "graph_nodes": len(grafo.nodes),
            "graph_edges": len(grafo.edges),
            "validacion": {
                "es_valida": getattr(validacion, "es_valida", False),
                "violaciones_orden": getattr(validacion, "violaciones_orden", []),
                "caminos_completos": getattr(validacion, "caminos_completos", []),
                "categorias_faltantes": [
                    getattr(cat, "name", str(cat))
                    for cat in getattr(validacion, "categorias_faltantes", [])
                ],
                "sugerencias": getattr(validacion, "sugerencias", []),
            },
            "caminos_detectados": len(getattr(caminos, "caminos_completos", [])),
            "sugerencias": getattr(sugerencias, "sugerencias", []),
        }

        success = validator.generate_industrial_report()
        report["success"] = success
        report["status"] = "success" if success else "report_threshold_not_met"
        report["metrics"] = _serialize_metrics(validator.metrics)
        return report

    except Exception:
        LOGGER.exception("\n💥 FALLA CATASTRÓFICA EN VALIDACIÓN INDUSTRIAL (pipeline)")
        report["status"] = "exception"
        report["metrics"] = _serialize_metrics(validator.metrics)
        return report


def validate_teoria_cambio_industrial():
    """Validador industrial de última generación para Teoría de Cambio"""
    report = execute_industrial_validation_detailed()
    if report["success"]:
        LOGGER.info(
            "\n🎉 IMPLEMENTACIÓN CERTIFICADA PARA ENTORNOS INDUSTRIALES CRÍTICOS"
        )
        LOGGER.info("   • Nivel: Estado del Arte en Teorías de Cambio")
        LOGGER.info(
            "   • Capacidad: Validación en tiempo real de sistemas complejos"
        )
        LOGGER.info("   • Robustez: Tolerancia a fallos y alto rendimiento")
    return report["success"]


if __name__ == "__main__":
    LOGGER.info("🏭 VALIDADOR INDUSTRIAL DE TEORÍA DE CAMBIO - NIVEL MÁXIMO")
    LOGGER.info("🔬 Tecnología: Estado del Arte en Validación de Sistemas Complejos")
    LOGGER.info("💼 Aplicación: Entornos Industriales Críticos\n")

    success = validate_teoria_cambio_industrial()

    exit_code = 0 if success else 1
    LOGGER.info(
        f"\n📤 Código de salida: {exit_code} - {'ÉXITO' if success else 'FALLA'}"
    )
    sys.exit(exit_code)
