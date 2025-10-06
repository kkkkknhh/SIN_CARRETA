#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system_validators.py — Gates de salud previos y posteriores a la evaluación MINIMINIMOON.

Checks principales:
  PRE:
    - Inmutabilidad congelada (EnhancedImmutabilityContract.verify_frozen_config()).
    - Presencia de RUBRIC_SCORING.json.
    - Documentación de flujo canónico presente (tools/flow_doc.json).

  POST:
    - Existencia de artifacts/flow_runtime.json y artifacts/answers_report.json.
    - Igualdad exacta de orden doc↔runtime (tools/flow_doc.json vs flow_runtime.json).
    - Validación de contratos/orden con CanonicalFlowValidator (deterministic_pipeline_validator).
    - Cobertura ≥ 300 preguntas en answers_report.json.
    - (Opcional) Verificación 1:1 answers↔pesos de rúbrica.

Salida:
  - Métodos devuelven dict con {ok: bool, errors: [str], ...}
  - CLI devuelve JSON y exit codes (0 OK, 2 fallo de validación, 1 error interno).
"""

from __future__ import annotations
import json
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Tuple

# Dependencias internas
try:
    from miniminimoon_immutability import EnhancedImmutabilityContract
except Exception as _e:
    EnhancedImmutabilityContract = None  # type: ignore

# El validador canónico debe exponer las utilidades abajo:
#   - CanonicalFlowValidator(contracts, order).validate_flow(runtime_trace) -> {"ok": bool, "errors": [...]}
#   - _contracts_and_order() -> (contracts: dict, order: list[str])
try:
    from deterministic_pipeline_validator import CanonicalFlowValidator, _contracts_and_order
except Exception as _e:
    CanonicalFlowValidator = None  # type: ignore

# Utilidad: lectura de JSON con mensajes claros
def _read_json(path: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    if not path.exists():
        return {}, f"Missing JSON file: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, ""
    except Exception as e:
        return {}, f"Cannot parse JSON {path}: {e}"

class SystemHealthValidator:
    def __init__(self, repo_root: str = ".") -> None:
        self.repo_root = str(pathlib.Path(repo_root).resolve())
        self.repo = pathlib.Path(self.repo_root)

        # Cargar orden/contratos canónicos si el módulo lo expone
        self.contracts = {}
        self.order: List[str] = []
        if CanonicalFlowValidator is not None:
            try:
                c, o = _contracts_and_order()  # type: ignore
                self.contracts = c or {}
                self.order = list(o or [])
            except Exception:
                # Se verificará igualmente por doc↔runtime contra tools/flow_doc.json
                self.contracts = {}
                self.order = []

    # ---------- PRE ----------
    def validate_pre_execution(self) -> Dict[str, Any]:
        errors: List[str] = []

        # 1) Freeze/inmutabilidad
        if EnhancedImmutabilityContract is None:
            errors.append("EnhancedImmutabilityContract not importable")
            immut_ok = False
        else:
            try:
                immut = EnhancedImmutabilityContract(self.repo_root)  # type: ignore
                immut_ok = bool(immut.verify_frozen_config())
                if not immut_ok:
                    errors.append("Frozen config missing or drift detected")
            except Exception as e:
                immut_ok = False
                errors.append(f"immutability check failed: {e}")

        # 2) Rúbrica presente
        rubric_path = self.repo / "RUBRIC_SCORING.json"
        if not rubric_path.exists():
            errors.append("RUBRIC_SCORING.json missing")

        # 3) Doc de flujo canónico presente
        flow_doc_path = self.repo / "tools" / "flow_doc.json"
        if not flow_doc_path.exists():
            errors.append("tools/flow_doc.json missing (canonical order doc)")

        ok = len(errors) == 0
        return {
            "ok": ok,
            "errors": errors,
            "checks": {
                "immutability_ok": immut_ok if EnhancedImmutabilityContract else False,
                "rubric_present": rubric_path.exists(),
                "flow_doc_present": flow_doc_path.exists(),
            }
        }

    # ---------- POST ----------
    def validate_post_execution(self, artifacts_dir: str = "artifacts", check_rubric_strict: bool = False) -> Dict[str, Any]:
        errors: List[str] = []
        artifacts = self.repo / artifacts_dir

        # 1) Existen artefactos mínimos
        runtime_path = artifacts / "flow_runtime.json"
        answers_path = artifacts / "answers_report.json"
        if not runtime_path.exists():
            errors.append(f"{runtime_path} missing")
        if not answers_path.exists():
            errors.append(f"{answers_path} missing")

        # Si faltan artefactos, retornamos con errores
        if errors:
            return {"ok": False, "errors": errors, "ok_order": False, "ok_coverage": False, "ok_rubric_1to1": (not check_rubric_strict)}

        # 2) Igualdad exacta de orden doc↔runtime
        flow_doc_path = self.repo / "tools" / "flow_doc.json"
        flow_doc, e1 = _read_json(flow_doc_path)
        if e1:
            errors.append(e1)
            return {"ok": False, "errors": errors, "ok_order": False, "ok_coverage": False, "ok_rubric_1to1": (not check_rubric_strict)}

        runtime_trace, e2 = _read_json(runtime_path)
        if e2:
            errors.append(e2)
            return {"ok": False, "errors": errors, "ok_order": False, "ok_coverage": False, "ok_rubric_1to1": (not check_rubric_strict)}

        doc_order = list(flow_doc.get("canonical_order", []))
        rt_order = list(runtime_trace.get("order", []))
        if not doc_order:
            errors.append("tools/flow_doc.json has empty canonical_order")
        if not rt_order:
            errors.append("flow_runtime.json has empty order")

        ok_order_doc = (doc_order == rt_order) and len(doc_order) > 0
        if not ok_order_doc:
            errors.append("Canonical order mismatch between tools/flow_doc.json and flow_runtime.json")

        # 3) Validación de contratos/orden via CanonicalFlowValidator (si está disponible)
        ok_order_contracts = True
        contract_errors: List[str] = []
        if CanonicalFlowValidator is not None and self.contracts and self.order:
            try:
                res = CanonicalFlowValidator(self.contracts, self.order).validate_flow(runtime_trace)  # type: ignore
                ok_order_contracts = bool(res.get("ok", False))
                if not ok_order_contracts:
                    ce = res.get("errors", [])
                    if ce:
                        contract_errors.extend(ce if isinstance(ce, list) else [str(ce)])
            except Exception as e:
                ok_order_contracts = False
                contract_errors.append(f"contract validation error: {e}")
        else:
            # Si no hay contratos expuestos, mantenemos la verificación doc↔runtime como gate principal
            ok_order_contracts = True

        if not ok_order_contracts:
            errors.append("CanonicalFlowValidator: order/schema validation failed")
            errors.extend(contract_errors[:5])  # limitar ruido

        # 4) Cobertura ≥ 300
        answers, e3 = _read_json(answers_path)
        if e3:
            errors.append(e3)
            return {
                "ok": False,
                "errors": errors,
                "ok_order": ok_order_doc and ok_order_contracts,
                "ok_coverage": False,
                "ok_rubric_1to1": (not check_rubric_strict)
            }

        total_questions = int(answers.get("summary", {}).get("total_questions", 0))
        ok_coverage = total_questions >= 300
        if not ok_coverage:
            errors.append(f"Coverage below 300 questions (got {total_questions})")

        # 5) (Opcional) Rúbrica 1:1: todas las preguntas reportadas deben existir en pesos
        ok_rubric_1to1 = True
        if check_rubric_strict:
            rubric_path = self.repo / "RUBRIC_SCORING.json"
            rubric_check_script = self.repo / "tools" / "rubric_check.py"
            
            # Try invoking tools/rubric_check.py as a subprocess
            try:
                result = subprocess.run(
                    [sys.executable, str(rubric_check_script), str(answers_path), str(rubric_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Handle exit codes
                if result.returncode == 3:
                    # Mismatch error: missing_weights and extra_weights diff
                    ok_rubric_1to1 = False
                    error_msg = f"Rubric mismatch detected (exit code 3): Questions in answers_report.json do not align with RUBRIC_SCORING.json weights section"
                    if result.stdout.strip():
                        error_msg += f"\n{result.stdout.strip()}"
                    if result.stderr.strip():
                        error_msg += f"\n{result.stderr.strip()}"
                    errors.append(error_msg)
                elif result.returncode == 2:
                    # Missing input files
                    ok_rubric_1to1 = False
                    error_msg = f"Rubric check failed (exit code 2): Missing input files for rubric validation"
                    if result.stdout.strip():
                        error_msg += f"\n{result.stdout.strip()}"
                    if result.stderr.strip():
                        error_msg += f"\n{result.stderr.strip()}"
                    errors.append(error_msg)
                elif result.returncode != 0:
                    # Other non-zero exit codes
                    ok_rubric_1to1 = False
                    error_msg = f"Rubric check failed with exit code {result.returncode}"
                    if result.stdout.strip():
                        error_msg += f"\n{result.stdout.strip()}"
                    if result.stderr.strip():
                        error_msg += f"\n{result.stderr.strip()}"
                    errors.append(error_msg)
                # Exit code 0 means success, ok_rubric_1to1 remains True
                    
            except FileNotFoundError:
                # Handle missing rubric_check.py script gracefully
                ok_rubric_1to1 = False
                errors.append(f"Rubric validation failed: tools/rubric_check.py not found")
            except subprocess.TimeoutExpired:
                ok_rubric_1to1 = False
                errors.append("Rubric validation timed out after 30 seconds")
            except Exception as e:
                ok_rubric_1to1 = False
                errors.append(f"Rubric validation error: {e}")

        ok_all = (ok_order_doc and ok_order_contracts and ok_coverage and ok_rubric_1to1 and len(errors) == 0)
        return {
            "ok": ok_all,
            "errors": errors,
            "ok_order": ok_order_doc and ok_order_contracts,
            "ok_coverage": ok_coverage,
            "ok_rubric_1to1": ok_rubric_1to1
        }


# ---------------- CLI mínima ----------------
def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True))


def _main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="MINIMINIMOON System Validators")
    ap.add_argument("--repo", default=".", help="Repo root")
    ap.add_argument("--pre", action="store_true", help="Run pre-execution checks")
    ap.add_argument("--post", action="store_true", help="Run post-execution checks")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory for post checks")
    ap.add_argument("--rubric-strict", action="store_true", help="Enforce 1:1 rubric vs answers in post checks")
    args = ap.parse_args(argv)

    try:
        v = SystemHealthValidator(args.repo)
        result: Dict[str, Any] = {"repo": str(pathlib.Path(args.repo).resolve())}

        rc = 0
        if args.pre:
            pre = v.validate_pre_execution()
            result["pre"] = pre
            if not pre.get("ok", False):
                rc = 2
        if args.post:
            post = v.validate_post_execution(artifacts_dir=args.artifacts, check_rubric_strict=args.rubric_strict)
            result["post"] = post
            if not post.get("ok", False):
                rc = 2

        if not args.pre and not args.post:
            # Por defecto ejecuta ambos
            pre = v.validate_pre_execution()
            post = v.validate_post_execution(artifacts_dir=args.artifacts, check_rubric_strict=args.rubric_strict)
            result["pre"] = pre
            result["post"] = post
            if not (pre.get("ok", False) and post.get("ok", False)):
                rc = 2

        result["ok"] = (rc == 0)
        _print_json(result)
        return rc
    except Exception as e:
        _print_json({"ok": False, "error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
