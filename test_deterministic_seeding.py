#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script para verificar comportamiento determinista del sistema de evaluación industrial.
Ejecuta el mismo plan dos veces con semilla idéntica y compara outputs numéricos.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from log_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def extract_numerical_data(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae datos numéricos del reporte JSON para comparación determinista.

    Args:
        report_data: Datos del reporte JSON

    Returns:
        Diccionario con solo los valores numéricos comparables
    """
    numerical_data = {}

    # Extraer métricas del resumen ejecutivo
    if "resumen_ejecutivo" in report_data:
        resumen = report_data["resumen_ejecutivo"]
        numerical_data["puntaje_global"] = resumen.get("puntaje_global", 0)
        numerical_data["dimensiones_evaluadas"] = resumen.get(
            "dimensiones_evaluadas", 0
        )

    # Extraer métricas por dimensión
    if "dimensiones" in report_data:
        numerical_data["dimensiones"] = []
        for dim in report_data["dimensiones"]:
            dim_numerical = {
                "id": dim.get("id", 0),
                "puntaje_final": dim.get("puntaje_final", 0),
                "consistencia_logica": dim.get("evaluacion_causal", {}).get(
                    "consistencia_logica", 0
                ),
                "identificabilidad_causal": dim.get("evaluacion_causal", {}).get(
                    "identificabilidad_causal", 0
                ),
                "factibilidad_operativa": dim.get("evaluacion_causal", {}).get(
                    "factibilidad_operativa", 0
                ),
                "certeza_probabilistica": dim.get("evaluacion_causal", {}).get(
                    "certeza_probabilistica", 0
                ),
                "robustez_causal": dim.get("evaluacion_causal", {}).get(
                    "robustez_causal", 0
                ),
                "evidencia_soporte": dim.get("evaluacion_causal", {}).get(
                    "evidencia_soporte", 0
                ),
            }

            # Extraer scores de evidencia si están disponibles
            if "evidencia" in dim:
                for tipo_evidencia, evidencias in dim["evidencia"].items():
                    if isinstance(evidencias, list):
                        scores = [
                            ev.get("score_final", 0)
                            for ev in evidencias
                            if isinstance(ev, dict)
                        ]
                        if scores:
                            dim_numerical[f"{tipo_evidencia}_scores"] = sorted(scores)

            numerical_data["dimensiones"].append(dim_numerical)

    return numerical_data


def compare_numerical_data(
    data1: Dict[str, Any], data2: Dict[str, Any], tolerance: float = 1e-10
) -> Dict[str, Any]:
    """
    Compara dos conjuntos de datos numéricos con una tolerancia especificada.

    Args:
        data1, data2: Datos numéricos a comparar
        tolerance: Tolerancia para comparaciones de punto flotante

    Returns:
        Diccionario con resultados de la comparación
    """
    differences = []

    # Comparar métricas globales
    for key in ["puntaje_global", "dimensiones_evaluadas"]:
        if key in data1 and key in data2:
            val1, val2 = data1[key], data2[key]
            if abs(val1 - val2) > tolerance:
                differences.append(
                    f"{key}: {val1} vs {val2} (diff: {abs(val1 - val2)})"
                )

    # Comparar dimensiones
    if "dimensiones" in data1 and "dimensiones" in data2:
        dims1, dims2 = data1["dimensiones"], data2["dimensiones"]
        if len(dims1) != len(dims2):
            differences.append(f"Número de dimensiones: {len(dims1)} vs {len(dims2)}")
        else:
            for i, (dim1, dim2) in enumerate(zip(dims1, dims2)):
                for key in dim1.keys():
                    if key in dim2 and key.endswith("_scores"):
                        # Comparar listas de scores
                        if dim1[key] != dim2[key]:
                            differences.append(
                                f"Dimensión {i}, {key}: {dim1[key]} vs {dim2[key]}"
                            )
                    elif key in dim2 and isinstance(dim1[key], (int, float)):
                        # Comparar valores numéricos
                        if abs(dim1[key] - dim2[key]) > tolerance:
                            differences.append(
                                f"Dimensión {i}, {key}: {dim1[key]} vs {dim2[key]} (diff: {abs(dim1[key] - dim2[key])})"
                            )

    return {
        "identical": len(differences) == 0,
        "differences_count": len(differences),
        "differences": differences[
            :10
        ],  # Limitar a primeras 10 diferencias para legibilidad
    }


def create_test_pdf() -> Path:
    """
    Crea un PDF de prueba simple para testing.

    Returns:
        Path al archivo PDF creado
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        # Crear PDF temporal
        temp_dir = Path(tempfile.mkdtemp())
        pdf_path = temp_dir / "plan_desarrollo_test.pdf"

        # Generar contenido básico
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, "Plan de Desarrollo Municipal de Prueba")
        c.drawString(
            100,
            720,
            "1. Prevención de la violencia y protección frente al conflicto armado",
        )
        c.drawString(
            100,
            690,
            "Se implementarán estrategias integrales para reducir los índices de violencia",
        )
        c.drawString(
            100,
            660,
            "mediante fortalecimiento institucional y participación ciudadana.",
        )
        c.drawString(
            100, 630, "2. Equidad de género en acceso a servicios y oportunidades"
        )
        c.drawString(
            100,
            600,
            "Promoción de políticas públicas con enfoque diferencial de género",
        )
        c.drawString(100, 570, "para garantizar igualdad de oportunidades.")
        c.drawString(100, 540, "3. Desarrollo económico sostenible")
        c.drawString(
            100,
            510,
            "Fomento del emprendimiento y fortalecimiento del tejido empresarial local.",
        )
        c.save()

        logger.info(f"✅ PDF de prueba creado: {pdf_path}")
        return pdf_path

    except ImportError:
        logger.warning("⚠️  reportlab no disponible, creando PDF simple")
        # Crear un PDF básico usando información de texto simple
        temp_dir = Path(tempfile.mkdtemp())
        pdf_path = temp_dir / "plan_desarrollo_test.pdf"

        # Crear un contenido mínimo para PDF que pdfplumber pueda leer
        content = """%PDF-1.3
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 200
>>
stream
BT
/F1 12 Tf
100 750 Td
(Plan de Desarrollo Municipal de Prueba) Tj
0 -30 Td
(1. Prevencion de la violencia y proteccion) Tj
0 -30 Td
(2. Equidad de genero en acceso a servicios) Tj
0 -30 Td
(3. Desarrollo economico sostenible) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
458
%%EOF"""

        with open(pdf_path, "w", encoding="latin1") as f:
            f.write(content)

        logger.info(f"✅ PDF básico de prueba creado: {pdf_path}")
        return pdf_path


def run_evaluation_with_seed(pdf_path: Path, seed: int) -> Dict[str, Any]:
    """
    Ejecuta evaluación con semilla específica y retorna datos JSON.

    Args:
        pdf_path: Ruta al archivo PDF de prueba
        seed: Semilla para reproducibilidad

    Returns:
        Datos del reporte JSON generado
    """
    logger.info(f"🔄 Ejecutando evaluación con semilla {seed}...")

    # Ejecutar evaluación
    cmd = [
        sys.executable,
        "Decatalogo_principal.py",
        "--seed",
        str(seed),
        str(pdf_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), check=True)

    if result.returncode != 0:
        logger.error(f"❌ Error ejecutando evaluación: {result.stderr}")
        raise RuntimeError(f"Falló evaluación con código {result.returncode}")

    # Buscar archivo JSON generado
    plan_name = pdf_path.stem
    output_dir = Path("resultados_evaluacion_industrial") / plan_name
    json_files = list(output_dir.glob("*_evaluacion_industrial.json"))

    if not json_files:
        logger.error(f"❌ No se encontró archivo JSON en: {output_dir}")
        raise FileNotFoundError("Archivo JSON de resultado no encontrado")

    json_path = json_files[0]
    logger.info(f"📄 Leyendo reporte JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_deterministic_behavior():
    """
    Test principal para verificar comportamiento determinista.
    """
    logger.info("🧪 Iniciando test de comportamiento determinista")

    try:
        # Crear PDF de prueba
        pdf_path = create_test_pdf()

        # Semilla de prueba
        test_seed = 42

        # Ejecutar dos veces con la misma semilla
        logger.info("🔄 Ejecutando primera evaluación...")
        data1 = run_evaluation_with_seed(pdf_path, test_seed)

        logger.info("🔄 Ejecutando segunda evaluación...")
        data2 = run_evaluation_with_seed(pdf_path, test_seed)

        # Extraer datos numéricos
        numerical_data1 = extract_numerical_data(data1)
        numerical_data2 = extract_numerical_data(data2)

        # Comparar resultados
        comparison = compare_numerical_data(numerical_data1, numerical_data2)

        # Reportar resultados
        logger.info("📊 Resultados del test de determinismo:")
        logger.info(f"¿Son idénticos?: {comparison['identical']}")
        logger.info(f"Diferencias encontradas: {comparison['differences_count']}")

        if comparison["differences"]:
            logger.warning("⚠️  Diferencias detectadas:")
            for diff in comparison["differences"]:
                logger.warning(f"   - {diff}")

        if comparison["identical"]:
            logger.info("✅ TEST EXITOSO: El comportamiento es determinista")
            return True
        else:
            logger.error("❌ TEST FALLIDO: Se detectaron diferencias")
            return False

    except Exception as e:
        logger.error(f"❌ Error durante el test: {e}")
        return False

    finally:
        # Cleanup
        try:
            if "pdf_path" in locals():
                pdf_path.unlink(missing_ok=True)
                pdf_path.parent.rmdir()
        except OSError as cleanup_error:
            logger.debug("Cleanup failed: %s", cleanup_error)


if __name__ == "__main__":
    success = test_deterministic_behavior()
    sys.exit(0 if success else 1)
