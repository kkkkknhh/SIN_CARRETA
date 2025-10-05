#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validador de Sincronización entre el Decálogo Canónico y la Rúbrica de Puntuación.

Descripción:
    Este script actúa como una compuerta de gobernanza crítica ("governance gate")
    para asegurar la integridad y consistencia entre la lista maestra de preguntas
    (DECALOGO_FULL.json) y las reglas de puntuación definidas (rubric_scoring.json).

    Verifica que exista una correspondencia exacta y unívoca (1:1) entre los
    identificadores de las plantillas de preguntas en ambos artefactos. Cualquier
    divergencia (preguntas sin regla de puntuación o reglas para preguntas
    inexistentes) es reportada y causa que el script falle.

Funcionamiento:
    1.  Carga el archivo canónico de preguntas del decálogo.
    2.  Carga el archivo de la rúbrica de puntuación.
    3.  Extrae los conjuntos de identificadores de plantillas de preguntas de ambos.
    4.  Compara los conjuntos para identificar discrepancias.
    5.  Imprime un reporte en formato JSON con los resultados.
    6.  Termina con un código de salida específico para reflejar el estado de la
        validación, permitiendo su integración en pipelines de CI/CD.

Argumentos:
    --decalogo-file: Ruta al archivo JSON que contiene la lista completa de las 300 preguntas.
    --rubric-file: Ruta al archivo JSON que define la rúbrica de puntuación y sus 30 plantillas base.
    --verbose, -v: (Opcional) Muestra en la salida estándar los IDs específicos de las discrepancias.

Salida (stdout):
    Un objeto JSON con la siguiente estructura:
    {
      "ok": bool,      // True si no hay discrepancias, False en caso contrario.
      "missing_in_rubric": list[str], // IDs presentes en el decálogo pero ausentes en la rúbrica.
      "extra_in_rubric": list[str]    // IDs presentes en la rúbrica pero ausentes en el decálogo.
    }

Códigos de Salida (Exit Codes):
    - 0: Éxito. La validación pasó sin errores; los archivos están sincronizados.
    - 1: Error de Archivo. Uno o ambos archivos de entrada no se encontraron.
    - 2: Error de Formato. Uno o ambos archivos no son JSON válidos o carecen de la estructura esperada.
    - 3: Fallo de Validación. Los archivos están desincronizados; se encontraron discrepancias.
"""

import json
import sys
import argparse
import pathlib
from typing import Set, Dict, Any, List

# --- Constantes para códigos de salida y formato ---

# Códigos de salida semánticos para automatización
EXIT_CODE_SUCCESS = 0
EXIT_CODE_FILE_NOT_FOUND = 1
EXIT_CODE_FORMAT_ERROR = 2
EXIT_CODE_VALIDATION_FAILED = 3

# Secuencias de escape ANSI para una salida de terminal más clara
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"

def load_and_validate_json(file_path: pathlib.Path, expected_key: str) -> List[Dict[str, Any]]:
    """Carga y valida la estructura básica de un archivo JSON."""
    if not file_path.exists():
        print(f"{COLOR_RED}Error: El archivo no fue encontrado en la ruta especificada: {file_path}{COLOR_RESET}", file=sys.stderr)
        sys.exit(EXIT_CODE_FILE_NOT_FOUND)

    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or expected_key not in data:
            raise KeyError
        return data[expected_key]
    except json.JSONDecodeError:
        print(f"{COLOR_RED}Error: El archivo no es un JSON válido: {file_path}{COLOR_RESET}", file=sys.stderr)
        sys.exit(EXIT_CODE_FORMAT_ERROR)
    except KeyError:
        print(f"{COLOR_RED}Error: El archivo JSON no tiene la clave esperada ('{expected_key}'): {file_path}{COLOR_RESET}", file=sys.stderr)
        sys.exit(EXIT_CODE_FORMAT_ERROR)

def extract_question_ids(questions_data: List[Dict[str, Any]]) -> Set[str]:
    """Extrae un conjunto de identificadores de pregunta de una lista de objetos de pregunta."""
    try:
        return {item['id'] for item in questions_data}
    except KeyError:
        print(f"{COLOR_RED}Error: Se encontró un objeto de pregunta sin la clave 'id' requerida.{COLOR_RESET}", file=sys.stderr)
        sys.exit(EXIT_CODE_FORMAT_ERROR)

def main():
    """Punto de entrada principal para la ejecución del script."""
    parser = argparse.ArgumentParser(
        description="Valida la consistencia entre el decálogo de preguntas y la rúbrica de puntuación.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Ejemplo de uso:\n"
               "  python3 rubric_check.py --decalogo-file DECALOGO_FULL.json --rubric-file rubric_scoring.json -v"
    )
    parser.add_argument(
        "--decalogo-file",
        type=pathlib.Path,
        required=True,
        help="Ruta al archivo JSON canónico con la lista completa de preguntas (e.g., DECALOGO_FULL.json)."
    )
    parser.add_argument(
        "--rubric-file",
        type=pathlib.Path,
        required=True,
        help="Ruta al archivo JSON con la rúbrica de puntuación y sus plantillas (e.g., rubric_scoring.json)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Muestra detalles de las discrepancias en la consola."
    )
    args = parser.parse_args()

    # Cargar y validar ambos archivos
    decalogo_questions = load_and_validate_json(args.decalogo_file, 'questions')
    rubric_questions = load_and_validate_json(args.rubric_file, 'questions')

    # Extraer los conjuntos de IDs de las plantillas de preguntas
    canonical_qids = extract_question_ids(decalogo_questions)
    rubric_qids = extract_question_ids(rubric_questions)

    # Realizar la comparación lógica
    missing_in_rubric = canonical_qids - rubric_qids
    extra_in_rubric = rubric_qids - canonical_qids

    is_ok = not missing_in_rubric and not extra_in_rubric
    
    # Reporte detallado opcional para el usuario
    if args.verbose:
        if is_ok:
            print(f"{COLOR_GREEN}✔ Verificación exitosa: La rúbrica y el decálogo están perfectamente sincronizados.{COLOR_RESET}")
            print(f"  - Total de plantillas de preguntas validadas: {len(canonical_qids)}")
        else:
            print(f"{COLOR_RED}✘ Fallo de validación: Se encontraron discrepancias.{COLOR_RESET}")
            if missing_in_rubric:
                print(f"{COLOR_YELLOW}  - {len(missing_in_rubric)} pregunta(s) existe(n) en el decálogo pero no tienen regla en la rúbrica:{COLOR_RESET}")
                for qid in sorted(list(missing_in_rubric)):
                    print(f"    - {qid}")
            if extra_in_rubric:
                print(f"{COLOR_YELLOW}  - {len(extra_in_rubric)} regla(s) en la rúbrica no corresponde(n) a ninguna pregunta del decálogo:{COLOR_RESET}")
                for qid in sorted(list(extra_in_rubric)):
                    print(f"    - {qid}")

    # Salida JSON determinista para la automatización
    result_data = {
        "ok": is_ok,
        "missing_in_rubric": sorted(list(missing_in_rubric)),
        "extra_in_rubric": sorted(list(extra_in_rubric)),
    }
    
    print(json.dumps(result_data, indent=2, ensure_ascii=False))

    # Salir con el código apropiado
    sys.exit(EXIT_CODE_SUCCESS if is_ok else EXIT_CODE_VALIDATION_FAILED)

if __name__ == "__main__":
    main()