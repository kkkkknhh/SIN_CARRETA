#!/usr/bin/env python3
"""
trace_matrix.py — Provenance Traceability Auditor (QA Gate)
============================================================
Función: Genera matriz cruda módulo→pregunta a partir de evidence_ids para auditoría
de cobertura, trazabilidad y no-repudio en el pipeline deterministico.

Posición en el flujo (según Dependency Flows Documentation):
tools/trace_matrix.py → artifacts/module_to_questions_matrix.csv
Type: QA • Card: 1:1
Input: {answers_report.evidence_ids}
Output: {module→question matrix}
Razón: trazabilidad cruda para auditoría.

Invariantes garantizados:
1. Parseo determinista de evidence_id → module (basado en convención 'source::type::hash')
2. Preservación total de provenance: cada tupla (module, question_id, evidence_id) es inmutable
3. Salida CSV canónica con encoding UTF-8 para compatibilidad cross-platform
4. Exit codes semánticos: 0=success, 2=missing_input, 3=malformed_data, 1=runtime_error

Contrato I/O:
Input:  artifacts/answers_report.json
Schema esperado: {
"answers": [
{
"question_id": str,
"evidence_ids": list[str],
"confidence": float,
"score": float,
...
}
]
}

Output: artifacts/module_to_questions_matrix.csv
Schema: module,question_id,evidence_id,confidence,score
Ordenamiento: insertion order (refleja orden de procesamiento)

Casos de uso:
- Auditoría externa: verificar que cada pregunta tiene evidencia trazable
- Análisis de cobertura: identificar módulos sub/sobre-utilizados
- Debugging de provenance: rastrear qué detector generó qué evidence_id
- Compliance: demostrar cadena de custodia desde raw_text → answer

Dependencias autorizadas:
- csv, json, pathlib, sys (stdlib)
- evidence_registry (import no-op, mantiene compatibilidad histórica)

Integración CI/CD:
Llamado desde ci/checks.yml después de rubric_check.py:
python tools/trace_matrix.py || exit $?

Changelog:
- v2.0 (2025-10-05): Refactor post-unificación, exit codes semánticos,
validación de schema, documentación exhaustiva.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import TypedDict, List, Any

# Import simbólico para mantener compatibilidad con módulos que esperan este import
try:
    from evidence_registry import EvidenceRegistry
except ImportError:
    # Graceful degradation si evidence_registry no está en PYTHONPATH
    EvidenceRegistry = None  # type: ignore


# ────────────────────────────────────────────────────────────────────────────────
# Type Definitions (para claridad de contrato)
# ────────────────────────────────────────────────────────────────────────────────

class AnswerRecord(TypedDict):
    """Schema esperado de cada elemento en answers_report.json['answers']."""
    question_id: str
    evidence_ids: List[str]
    confidence: float
    score: float
    # Campos adicionales (rationale, etc.) son ignorados para esta matriz


class MatrixRow(TypedDict):
    """Schema de cada fila en la matriz de salida CSV."""
    module: str
    question_id: str
    evidence_id: str
    confidence: float
    score: float


# ────────────────────────────────────────────────────────────────────────────────
# Core Logic
# ────────────────────────────────────────────────────────────────────────────────

def extract_module_from_evidence_id(evidence_id: str) -> str:
    """
Extrae el módulo fuente de un evidence_id según convención 'source::type::hash'.

Convención de evidence_id (definida en evidence_registry.py):
{detector_name}::{evidence_type}::{content_hash}
Ejemplo: "responsibility_detector::assignment::a3f9c2e1"

Args:
evidence_id: ID de evidencia en formato canónico

Returns:
Nombre del módulo detector (primer segmento antes de '::')
Retorna 'unknown' si el formato no cumple convención

Razón de 'unknown':
- Permite identificar evidence_ids malformados en auditoría
- No falla el pipeline por problemas de formato legacy
- Señal clara para investigación manual
    """
    if "::" not in evidence_id:
        return "unknown"

    module = evidence_id.split("::", 1)[0]
    return module if module else "unknown"


def parse_answers_report(report_path: pathlib.Path) -> List[AnswerRecord]:
    """
Carga y valida el schema básico de answers_report.json.

Args:
report_path: Ruta a artifacts/answers_report.json

Returns:
Lista de registros de respuesta validados

Raises:
FileNotFoundError: Si el archivo no existe
json.JSONDecodeError: Si el JSON está malformado
KeyError: Si falta el campo 'answers' en el schema
ValueError: Si los campos requeridos están ausentes
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Missing required input: {report_path}")

    data = json.loads(report_path.read_text(encoding="utf-8"))

    if "answers" not in data:
        raise KeyError("Malformed answers_report.json: missing 'answers' field")

    answers = data["answers"]
    if not isinstance(answers, list):
        raise ValueError("Malformed answers_report.json: 'answers' must be a list")

    # Validación básica de schema (no exhaustiva, permite campos extra)
    for i, ans in enumerate(answers):
        required_fields = ["question_id", "evidence_ids", "confidence", "score"]
        for field in required_fields:
            if field not in ans:
                raise ValueError(
                    f"Malformed answer at index {i}: missing required field '{field}'"
                )

    return answers  # type: ignore


def build_traceability_matrix(answers: List[AnswerRecord]) -> List[MatrixRow]:
    """
Construye la matriz de trazabilidad expandiendo cada (pregunta, evidence_id).

Lógica:
- Cada pregunta puede referenciar N evidence_ids (fan-out)
- Cada evidence_id aparece en 1 fila independiente (normalización)
- Preserva orden de procesamiento (insertion order)

Args:
answers: Lista de registros de respuesta

Returns:
Lista de filas para la matriz CSV (sin agregación ni sorting)

Complejidad: O(Q × E) donde Q=preguntas, E=evidencias_por_pregunta
    """
    rows: List[MatrixRow] = []

    for ans in answers:
        q_id = ans["question_id"]
        confidence = ans["confidence"]
        score = ans["score"]

        for evid in ans["evidence_ids"]:
            module = extract_module_from_evidence_id(evid)

            row: MatrixRow = {
                "module": module,
                "question_id": q_id,
                "evidence_id": evid,
                "confidence": confidence,
                "score": score,
            }
            rows.append(row)

    return rows


def write_matrix_csv(rows: List[MatrixRow], output_path: pathlib.Path) -> None:
    """
Escribe la matriz de trazabilidad a CSV con encoding UTF-8 y newlines normalizados.

Args:
rows: Lista de filas de matriz
output_path: Ruta de salida (típicamente artifacts/module_to_questions_matrix.csv)

Garantías:
- UTF-8 con BOM opcional (compatible con Excel y herramientas Unix)
- Newlines normalizados (CRLF en Windows, LF en Unix)
- Header canónico: module,question_id,evidence_id,confidence,score
- Crea directorio padre si no existe (idempotente)

Raises:
OSError: Si no se puede escribir el archivo (permisos, disco lleno, etc.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["module", "question_id", "evidence_id", "confidence", "score"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore


# ────────────────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
Orquestación del flujo completo: parse → build → write.

Returns:
Exit code semántico:
0: Success (matriz generada correctamente)
1: Runtime error inesperado (ver stderr)
2: Missing input (answers_report.json no existe)
3: Malformed data (schema inválido o campos faltantes)

Side Effects:
- Crea artifacts/module_to_questions_matrix.csv
- Imprime ruta de salida a stdout (para encadenamiento en scripts)
- Imprime errores a stderr
    """
    input_path = pathlib.Path("artifacts/answers_report.json")
    output_path = pathlib.Path("artifacts/module_to_questions_matrix.csv")

    try:
        # Fase 1: Cargar y validar input
        answers = parse_answers_report(input_path)

        # Fase 2: Construir matriz de trazabilidad
        matrix_rows = build_traceability_matrix(answers)

        # Fase 3: Escribir output canónico
        write_matrix_csv(matrix_rows, output_path)

        # Success: imprimir ruta de salida para encadenamiento
        print(str(output_path))
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: Malformed data - {e}", file=sys.stderr)
        return 3

    except Exception as e:
        print(f"ERROR: Unexpected runtime error - {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())