#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINIMINIMOON Unified CLI
Comandos:
  - freeze        : Congela configuración crítica (snapshot SHA-256).
  - evaluate      : Ejecuta el flujo canónico determinista (pre/post validation).
  - verify        : Verifica artefactos post-ejecución (orden, contratos, cobertura).
  - rubric-check  : Valida 1:1 preguntas↔rubrica usando tools/rubric_check.py.
  - trace-matrix  : Genera matriz módulo→pregunta (artifacts/module_to_questions_matrix.csv).
  - version       : Muestra versión y estado mínimo.

Convenciones de salida:
  - Éxito        → exit 0
  - Error “esperado” de validación (gates) → exit 2 o 3 (rubric)
  - Error interno (excepción) → exit 1

Salida: JSON imprimible en stdout (salvo comandos auxiliares que también devuelven JSON).
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import subprocess
import sys
import traceback
from typing import Any, Dict

CLI_VERSION = "1.0.0"


def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True))


def cmd_freeze(args: argparse.Namespace) -> int:
    from miniminimoon_immutability import EnhancedImmutabilityContract
    repo = str(pathlib.Path(args.repo).resolve())
    try:
        immut = EnhancedImmutabilityContract(repo)
        out = immut.freeze_configuration()
        _print_json({"ok": True, "action": "freeze", "repo": repo, "snapshot": out})
        return 0
    except Exception as e:
        _print_json({"ok": False, "action": "freeze", "repo": repo, "error": str(e)})
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    from unified_evaluation_pipeline import UnifiedEvaluationPipeline
    repo = str(pathlib.Path(args.repo).resolve())
    rubric = args.rubric
    plan_path = str(pathlib.Path(args.plan_path).resolve())
    strict = bool(args.strict)
    try:
        pipe = UnifiedEvaluationPipeline(repo_root=repo, rubric_path=rubric)
        bundle = pipe.evaluate(plan_path)
        result = {
            "ok": True,
            "action": "evaluate",
            "repo": repo,
            "plan_path": plan_path,
            "artifacts_dir": str(pathlib.Path(repo) / "artifacts"),
            "results": bundle.get("results", {}),
            "pre_validation": bundle.get("pre_validation", {}),
            "post_validation": bundle.get("post_validation", {})
        }
        _print_json(result)
        # Si strict está activo, reforzamos fallos por gates incompletos (aunque Unified ya lanza)
        if strict:
            post = bundle.get("post_validation", {})
            if not post.get("ok", False):
                return 2
        return 0
    except Exception as e:
        _print_json({
            "ok": False,
            "action": "evaluate",
            "repo": repo,
            "plan_path": plan_path,
            "error": str(e),
            "traceback": traceback.format_exc().splitlines()[-10:]
        })
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    from system_validators import SystemHealthValidator
    repo = str(pathlib.Path(args.repo).resolve())
    artifacts = str(pathlib.Path(repo) / "artifacts")
    try:
        v = SystemHealthValidator(repo)
        pre = v.validate_pre_execution()
        post = v.validate_post_execution(artifacts_dir="artifacts")
        _print_json({"ok": pre.get("ok", False) and post.get("ok", False),
                     "action": "verify",
                     "repo": repo,
                     "artifacts_dir": artifacts,
                     "pre": pre, "post": post})
        return 0 if pre.get("ok", False) and post.get("ok", False) else 2
    except Exception as e:
        _print_json({"ok": False, "action": "verify", "repo": repo, "error": str(e)})
        return 1


def _run_tool(pyfile_rel: str, repo: str, tool_args: list = None) -> Dict[str, Any]:
    """
    Ejecuta un script Python del árbol del repo y retorna su salida parseada si es JSON,
    o bien incluye stdout/stderr crudos.
    """
    py = pathlib.Path(repo) / pyfile_rel
    if not py.exists():
        return {"ok": False, "error": f"Tool not found: {py}"}
    try:
        cmd = [sys.executable, str(py)]
        if tool_args:
            cmd.extend(tool_args)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        # Intentar parsear JSON si corresponde
        parsed = None
        try:
            parsed = json.loads(stdout) if stdout else None
        except Exception:
            parsed = None
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": parsed
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def cmd_rubric_check(args: argparse.Namespace) -> int:
    answers_path = str(pathlib.Path(args.answers_report).resolve())
    rubric_path = str(pathlib.Path(args.rubric_scoring).resolve())
    
    tool_path = pathlib.Path(__file__).parent / "tools" / "rubric_check.py"
    if not tool_path.exists():
        print(f"Error: rubric_check.py not found at {tool_path}", file=sys.stderr)
        return 1
    
    try:
        cmd = [sys.executable, str(tool_path), answers_path, rubric_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        if proc.stdout:
            print(proc.stdout, end='')
        if proc.stderr:
            print(proc.stderr, end='', file=sys.stderr)
        
        return proc.returncode
    except Exception as e:
        print(f"Error executing rubric-check: {e}", file=sys.stderr)
        return 1


def cmd_trace_matrix(args: argparse.Namespace) -> int:
    repo = str(pathlib.Path(args.repo).resolve())
    res = _run_tool("tools/trace_matrix.py", repo)
    payload = {"action": "trace-matrix", "repo": repo, **res}
    _print_json(payload)
    return 0 if res.get("ok", False) else 1


def cmd_version(_args: argparse.Namespace) -> int:
    _print_json({"ok": True, "action": "version", "cli_version": CLI_VERSION})
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MINIMINIMOON Unified CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # freeze
    p_fr = sub.add_parser("freeze", help="Freeze configuration snapshot for immutable config")
    p_fr.add_argument("--repo", default=".", help="Repo root")
    p_fr.set_defaults(func=cmd_freeze)

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Evaluate a PDM with the canonical deterministic flow")
    p_ev.add_argument("plan_path", help="Path to plan/PDM text file")
    p_ev.add_argument("--repo", default=".", help="Repo root")
    p_ev.add_argument("--rubric", default="RUBRIC_SCORING.json", help="Path to rubric JSON")
    p_ev.add_argument("--strict", action="store_true", help="Fail on any post-validation error (exit 2)")
    p_ev.set_defaults(func=cmd_evaluate)

    # verify
    p_vf = sub.add_parser("verify", help="Verify post-execution artifacts (order/contracts/coverage)")
    p_vf.add_argument("--repo", default=".", help="Repo root")
    p_vf.set_defaults(func=cmd_verify)

    # rubric-check
    p_rc = sub.add_parser("rubric-check", help="Run rubric 1:1 check (answers vs weights)")
    p_rc.add_argument("answers_report", help="Path to answers_report.json")
    p_rc.add_argument("rubric_scoring", help="Path to RUBRIC_SCORING.json")
    p_rc.add_argument("--repo", default=".", help="Repo root")
    p_rc.set_defaults(func=cmd_rubric_check)

    # trace-matrix
    p_tm = sub.add_parser("trace-matrix", help="Produce module→question trace matrix CSV")
    p_tm.add_argument("--repo", default=".", help="Repo root")
    p_tm.set_defaults(func=cmd_trace_matrix)

    # version
    p_vs = sub.add_parser("version", help="Show CLI version and minimal status")
    p_vs.set_defaults(func=cmd_version)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rc = args.func(args)
        sys.exit(rc)
    except AttributeError:
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        _print_json({"ok": False, "error": str(e), "traceback": traceback.format_exc().splitlines()[-10:]})
        sys.exit(1)


if __name__ == "__main__":
    main()
