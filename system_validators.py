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
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Custom exception for validation failures
class ValidationError(Exception):
    pass


# psutil is optional for batch validation - handle gracefully if not available
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

# Dependencias internas
try:
    from miniminimoon_immutability import EnhancedImmutabilityContract
except Exception as _e:
    EnhancedImmutabilityContract = None  # type: ignore

# El validador canónico debe exponer las utilidades abajo:
#   - CanonicalFlowValidator(contracts, order).validate_flow(runtime_trace) -> {"ok": bool, "errors": [...]}
#   - _contracts_and_order() -> (contracts: dict, order: list[str])
try:
    from deterministic_pipeline_validator import (
        CanonicalFlowValidator,
        _contracts_and_order,
    )
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


@dataclass
class BatchValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)

    # Pre-execution resource checks
    memory_available_gb: Optional[float] = None
    memory_ok: bool = False
    disk_available_gb: Optional[float] = None
    disk_ok: bool = False
    redis_ok: bool = False
    workers_available: int = 0
    workers_ok: bool = False

    # Post-execution quality metrics
    total_documents: int = 0
    coverage_passed: int = 0
    coverage_failed: int = 0
    hash_consistency_ok: bool = False
    error_rate_percent: float = 0.0
    throughput_docs_per_hour: float = 0.0
    processing_time_stats: Optional[Dict[str, float]] = None


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

    def validate_question_id_format(self) -> None:
        rubric_path = self.repo / "RUBRIC_SCORING.json"

        if not rubric_path.exists():
            raise ValidationError(
                "RUBRIC_SCORING.json missing - cannot validate question ID format"
            )

        try:
            rubric_data = json.loads(rubric_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValidationError(f"Failed to parse RUBRIC_SCORING.json: {e}")

        weights = rubric_data.get("weights", {})
        if not weights:
            raise ValidationError(
                "RUBRIC_SCORING.json 'weights' section is empty or missing"
            )

        # Pattern: P{1-10}-D{1-6}-Q{1-30}
        question_id_pattern = re.compile(r"^P([1-9]|10)-D[1-6]-Q([1-9]|[12][0-9]|30)$")

        malformed_ids: List[str] = []
        for question_id in weights.keys():
            if not question_id_pattern.match(question_id):
                malformed_ids.append(question_id)

        total_count = len(weights)

        error_messages: List[str] = []
        if malformed_ids:
            error_messages.append(
                f"Found {len(malformed_ids)} malformed question ID(s) in RUBRIC_SCORING.json weights section. "
                f"Expected format: P{{1-10}}-D{{1-6}}-Q{{1-30}}. Malformed IDs: {malformed_ids[:10]}"
                + (
                    f" (showing first 10 of {len(malformed_ids)})"
                    if len(malformed_ids) > 10
                    else ""
                )
            )

        if total_count != 300:
            error_messages.append(
                f"Question count mismatch in RUBRIC_SCORING.json: found {total_count} questions, expected exactly 300"
            )

        if error_messages:
            raise ValidationError(
                "Rubric structure validation failed:\n"
                + "\n".join(f"  - {msg}" for msg in error_messages)
            )

    # ---------- PRE ----------
    def validate_pre_execution(self) -> Dict[str, Any]:
        errors: List[str] = []

        # 0) Validate question ID format in RUBRIC_SCORING.json
        try:
            self.validate_question_id_format()
        except ValidationError as e:
            errors.append(str(e))

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
            },
        }

    # ---------- POST ----------
    def validate_post_execution(
        self, artifacts_dir: str = "artifacts", check_rubric_strict: bool = False
    ) -> Dict[str, Any]:
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
            return {
                "ok": False,
                "errors": errors,
                "ok_order": False,
                "ok_coverage": False,
                "ok_rubric_1to1": (not check_rubric_strict),
            }

        # 2) Igualdad exacta de orden doc↔runtime
        flow_doc_path = self.repo / "tools" / "flow_doc.json"
        flow_doc, e1 = _read_json(flow_doc_path)
        if e1:
            errors.append(e1)
            return {
                "ok": False,
                "errors": errors,
                "ok_order": False,
                "ok_coverage": False,
                "ok_rubric_1to1": (not check_rubric_strict),
            }

        runtime_trace, e2 = _read_json(runtime_path)
        if e2:
            errors.append(e2)
            return {
                "ok": False,
                "errors": errors,
                "ok_order": False,
                "ok_coverage": False,
                "ok_rubric_1to1": (not check_rubric_strict),
            }

        doc_order = list(flow_doc.get("canonical_order", []))
        rt_order = list(runtime_trace.get("order", []))
        if not doc_order:
            errors.append("tools/flow_doc.json has empty canonical_order")
        if not rt_order:
            errors.append("flow_runtime.json has empty order")

        ok_order_doc = (doc_order == rt_order) and len(doc_order) > 0
        if not ok_order_doc:
            errors.append(
                "Canonical order mismatch between tools/flow_doc.json and flow_runtime.json"
            )

        # 3) Validación de contratos/orden via CanonicalFlowValidator (si está disponible)
        ok_order_contracts = True
        contract_errors: List[str] = []
        if CanonicalFlowValidator is not None and self.contracts and self.order:
            try:
                res = CanonicalFlowValidator(self.contracts, self.order).validate_flow(
                    runtime_trace
                )  # type: ignore
                ok_order_contracts = bool(res.get("ok", False))
                if not ok_order_contracts:
                    ce = res.get("errors", [])
                    if ce:
                        contract_errors.extend(
                            ce if isinstance(ce, list) else [str(ce)]
                        )
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
                "ok_rubric_1to1": (not check_rubric_strict),
            }

        total_questions = int(answers.get("summary", {}).get("total_questions", 0))
        ok_coverage = total_questions >= 300
        if not ok_coverage:
            errors.append(f"Coverage below 300 questions (got {total_questions})")

        # 5) (Opcional) Rúbrica 1:1: todas las preguntas reportadas deben existir en pesos
        ok_rubric_1to1 = True
        if check_rubric_strict:
            # Resolve absolute paths relative to project root
            rubric_path_abs = (self.repo / "RUBRIC_SCORING.json").resolve()
            rubric_check_script_abs = (
                self.repo / "tools" / "rubric_check.py"
            ).resolve()
            answers_path_abs = answers_path.resolve()

            # Try invoking tools/rubric_check.py as a subprocess
            exit_code = None
            stdout_output = ""
            stderr_output = ""

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(rubric_check_script_abs),
                        str(answers_path_abs),
                        str(rubric_path_abs),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                exit_code = result.returncode
                stdout_output = result.stdout.strip()
                stderr_output = result.stderr.strip()

            except FileNotFoundError as fnf_error:
                # Handle missing rubric_check.py script gracefully - treat as exit code 2
                exit_code = 2
                stderr_output = f"FileNotFoundError: {fnf_error}"

            except subprocess.TimeoutExpired:
                ok_rubric_1to1 = False
                errors.append("Rubric validation timed out after 30 seconds")
                exit_code = None  # Mark as handled

            except Exception as e:
                ok_rubric_1to1 = False
                errors.append(f"Rubric validation error: {e}")
                exit_code = None  # Mark as handled

            # Handle exit codes
            if exit_code == 3:
                # Mismatch error: missing_weights and extra_weights diff
                ok_rubric_1to1 = False
                error_msg = "Rubric mismatch (exit code 3): Questions in answers_report.json do not align with RUBRIC_SCORING.json weights"
                if stdout_output:
                    error_msg += f"\nDiff output: {stdout_output}"
                if stderr_output:
                    error_msg += f"\nError details: {stderr_output}"
                errors.append(error_msg)

            elif exit_code == 2:
                # Missing input files
                ok_rubric_1to1 = False
                error_msg = "Rubric check failed (exit code 2): Missing input file(s) - artifacts/answers_report.json or RUBRIC_SCORING.json not found"
                if stdout_output:
                    error_msg += f"\nDetails: {stdout_output}"
                if stderr_output:
                    error_msg += f"\nError: {stderr_output}"
                errors.append(error_msg)

            elif exit_code is not None and exit_code != 0:
                # Other non-zero exit codes
                ok_rubric_1to1 = False
                error_msg = f"Rubric check failed with exit code {exit_code}"
                if stdout_output:
                    error_msg += f"\nOutput: {stdout_output}"
                if stderr_output:
                    error_msg += f"\nError: {stderr_output}"
                errors.append(error_msg)
            # Exit code 0 means success, ok_rubric_1to1 remains True

        # 6) Trace matrix generation: validate provenance traceability
        ok_trace_matrix = True
        trace_matrix_script = self.repo / "tools" / "trace_matrix.py"
        if trace_matrix_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(trace_matrix_script)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self.repo),
                )

                # Handle exit codes (per trace_matrix.py spec)
                if result.returncode == 2:
                    # Missing input files
                    ok_trace_matrix = False
                    error_msg = "Trace matrix generation failed (exit code 2): Missing input files (answers_report.json)"
                    if result.stdout.strip():
                        error_msg += f"\n{result.stdout.strip()}"
                    if result.stderr.strip():
                        error_msg += f"\n{result.stderr.strip()}"
                    errors.append(error_msg)
                elif result.returncode == 3:
                    # Malformed data
                    ok_trace_matrix = False
                    error_msg = "Trace matrix generation failed (exit code 3): Malformed data in answers_report.json"
                    if result.stdout.strip():
                        error_msg += f"\n{result.stdout.strip()}"
                    if result.stderr.strip():
                        error_msg += f"\n{result.stderr.strip()}"
                    errors.append(error_msg)
                elif result.returncode != 0:
                    # Other non-zero exit codes (runtime errors)
                    ok_trace_matrix = False
                    error_msg = f"Trace matrix generation failed with exit code {result.returncode}"
                    if result.stdout.strip():
                        error_msg += f"\n{result.stdout.strip()}"
                    if result.stderr.strip():
                        error_msg += f"\n{result.stderr.strip()}"
                    errors.append(error_msg)
                # Exit code 0 means success, ok_trace_matrix remains True

            except FileNotFoundError:
                # Handle missing trace_matrix.py script gracefully
                ok_trace_matrix = False
                errors.append(
                    "Trace matrix validation failed: tools/trace_matrix.py not found"
                )
            except subprocess.TimeoutExpired:
                ok_trace_matrix = False
                errors.append("Trace matrix generation timed out after 60 seconds")
            except Exception as e:
                ok_trace_matrix = False
                errors.append(f"Trace matrix generation error: {e}")

        ok_all = (
            ok_order_doc
            and ok_order_contracts
            and ok_coverage
            and ok_rubric_1to1
            and ok_trace_matrix
            and len(errors) == 0
        )
        return {
            "ok": ok_all,
            "errors": errors,
            "ok_order": ok_order_doc and ok_order_contracts,
            "ok_coverage": ok_coverage,
            "ok_rubric_1to1": ok_rubric_1to1,
            "ok_trace_matrix": ok_trace_matrix,
        }


def validate_batch_pre_execution() -> BatchValidationResult:
    errors: List[str] = []

    # Memory check (8GB threshold)
    memory_available_gb = None
    memory_ok = False

    if not PSUTIL_AVAILABLE:
        errors.append("psutil not available - skipping resource checks")
        memory_ok = True  # Allow test to proceed
    else:
        try:
            mem = psutil.virtual_memory()  # type: ignore
            memory_available_gb = mem.available / (1024**3)
            memory_ok = memory_available_gb >= 8.0
            if not memory_ok:
                errors.append(
                    f"Insufficient memory: {memory_available_gb:.2f}GB available, 8GB required"
                )
        except Exception as e:
            memory_available_gb = None
            memory_ok = False
            errors.append(f"Memory check failed: {e}")

    # Disk space check (10GB threshold)
    try:
        disk = shutil.disk_usage(".")
        disk_available_gb = disk.free / (1024**3)
        disk_ok = disk_available_gb >= 10.0
        if not disk_ok:
            errors.append(
                f"Insufficient disk space: {disk_available_gb:.2f}GB available, 10GB required"
            )
    except Exception as e:
        disk_available_gb = None
        disk_ok = False
        errors.append(f"Disk space check failed: {e}")

    # Redis connectivity check
    redis_ok = False
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=5)
        redis_ok = r.ping()
        if not redis_ok:
            errors.append("Redis ping failed")
    except ImportError:
        errors.append("Redis library not available")
    except Exception as e:
        errors.append(f"Redis connectivity check failed: {e}")

    # Celery worker availability check
    workers_available = 0
    workers_ok = False
    try:
        from celery import Celery

        app = Celery("batch_validator")
        app.config_from_object("celeryconfig", silent=True)

        inspect = app.control.inspect()
        stats = inspect.stats()
        if stats:
            workers_available = len(stats)
            workers_ok = workers_available > 0

        if not workers_ok:
            errors.append("No Celery workers available")
    except ImportError:
        errors.append("Celery library not available")
    except Exception as e:
        errors.append(f"Celery worker check failed: {e}")

    ok = memory_ok and disk_ok and redis_ok and workers_ok

    result = BatchValidationResult(
        ok=ok,
        errors=errors,
        memory_available_gb=memory_available_gb,
        memory_ok=memory_ok,
        disk_available_gb=disk_available_gb,
        disk_ok=disk_ok,
        redis_ok=redis_ok,
        workers_available=workers_available,
        workers_ok=workers_ok,
    )

    # Raise ValidationError if any check fails
    if not ok:
        error_msg = "Batch pre-execution validation failed:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise ValidationError(error_msg)

    return result


def validate_batch_post_execution(
    batch_results: List[Dict[str, Any]], artifacts_base_dir: str = "artifacts"
) -> Dict[str, Any]:
    errors: List[str] = []
    base_path = pathlib.Path(artifacts_base_dir)

    total_documents = len(batch_results)
    coverage_passed = 0
    coverage_failed = 0
    processing_times: List[float] = []
    error_count = 0
    hashes: List[str] = []
    per_document_status: List[Dict[str, Any]] = []

    # Iterate through batch job results
    for result in batch_results:
        doc_id = result.get("document_id", "unknown")
        doc_status: Dict[str, Any] = {
            "document_id": doc_id,
            "status": "unknown",
            "coverage_passed": False,
            "hash_verified": False,
            "processing_time": None,
            "errors": [],
        }

        # Check for errors
        if result.get("error") or result.get("status") == "failed":
            error_count += 1
            error_msg = result.get("error", "unknown error")
            errors.append(f"Document {doc_id} failed: {error_msg}")
            doc_status["status"] = "failed"
            doc_status["errors"].append(error_msg)
            per_document_status.append(doc_status)
            continue

        doc_status["status"] = "processing"

        # Extract processing time if available
        if "processing_time" in result:
            try:
                proc_time = float(result["processing_time"])
                processing_times.append(proc_time)
                doc_status["processing_time"] = proc_time
            except (ValueError, TypeError):
                pass

        # Check coverage_report.json for 300/300 coverage
        doc_artifacts_dir = base_path / doc_id if doc_id != "unknown" else base_path
        coverage_path = doc_artifacts_dir / "coverage_report.json"

        if coverage_path.exists():
            try:
                coverage_data = json.loads(coverage_path.read_text(encoding="utf-8"))
                total_questions = coverage_data.get("summary", {}).get(
                    "total_questions", 0
                )

                if total_questions >= 300:
                    coverage_passed += 1
                    doc_status["coverage_passed"] = True
                    doc_status["coverage_count"] = total_questions
                else:
                    coverage_failed += 1
                    err_msg = f"Document {doc_id} has insufficient coverage: {total_questions}/300"
                    errors.append(err_msg)
                    doc_status["errors"].append(err_msg)
                    doc_status["coverage_count"] = total_questions
            except Exception as e:
                coverage_failed += 1
                err_msg = f"Document {doc_id} coverage_report.json parsing failed: {e}"
                errors.append(err_msg)
                doc_status["errors"].append(str(e))
        else:
            coverage_failed += 1
            err_msg = f"Document {doc_id} missing coverage_report.json"
            errors.append(err_msg)
            doc_status["errors"].append("Missing coverage_report.json")

        # Extract deterministic hash from evidence_registry.json
        evidence_path = doc_artifacts_dir / "evidence_registry.json"
        if evidence_path.exists():
            try:
                evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
                doc_hash = evidence_data.get("deterministic_hash")
                if doc_hash:
                    hashes.append(doc_hash)
                    doc_status["hash"] = doc_hash
                    doc_status["hash_verified"] = True
            except Exception as e:
                err_msg = (
                    f"Document {doc_id} evidence_registry.json parsing failed: {e}"
                )
                errors.append(err_msg)
                doc_status["errors"].append(str(e))

        # Update document status
        if doc_status["coverage_passed"] and doc_status["hash_verified"]:
            doc_status["status"] = "success"
        elif len(doc_status["errors"]) == 0:
            doc_status["status"] = "partial"
        else:
            doc_status["status"] = "failed"

        per_document_status.append(doc_status)

    # Validate hash consistency
    hash_consistency_ok = True
    hash_verification_details: Dict[str, Any] = {
        "total_hashes": len(hashes),
        "unique_hashes": 0,
        "consistent": False,
    }

    if len(hashes) > 0:
        unique_hashes = set(hashes)
        hash_verification_details["unique_hashes"] = len(unique_hashes)
        if len(unique_hashes) > 1:
            hash_consistency_ok = False
            err_msg = f"Hash inconsistency detected: {len(unique_hashes)} unique hashes across {len(hashes)} documents"
            errors.append(err_msg)
            hash_verification_details["consistent"] = False
        else:
            hash_verification_details["consistent"] = True
    else:
        hash_consistency_ok = False
        errors.append("No deterministic hashes found for validation")
        hash_verification_details["consistent"] = False

    # Calculate quality metrics
    error_rate_percent = (
        (error_count / total_documents * 100) if total_documents > 0 else 0.0
    )

    # Calculate throughput (documents per hour)
    throughput_docs_per_hour = 0.0
    if processing_times:
        total_time_hours = sum(processing_times) / 3600.0
        if total_time_hours > 0:
            throughput_docs_per_hour = total_documents / total_time_hours

    # Calculate processing time statistics (mean, p50, p95)
    processing_time_stats: Dict[str, float] = {}
    if processing_times:
        sorted_times = sorted(processing_times)
        processing_time_stats = {
            "mean": statistics.mean(processing_times),
            "p50": statistics.median(processing_times),
            "p95": sorted_times[int(len(sorted_times) * 0.95)]
            if len(sorted_times) > 0
            else 0.0,
            "min": min(processing_times),
            "max": max(processing_times),
            "count": len(processing_times),
        }

    # Aggregate coverage statistics
    aggregate_coverage_stats = {
        "total_documents": total_documents,
        "coverage_passed": coverage_passed,
        "coverage_failed": coverage_failed,
        "pass_rate_percent": (coverage_passed / total_documents * 100)
        if total_documents > 0
        else 0.0,
    }

    # Performance metrics
    performance_metrics = {
        "error_rate_percent": error_rate_percent,
        "throughput_docs_per_hour": throughput_docs_per_hour,
        "processing_time_distribution": processing_time_stats,
    }

    # Structured report
    ok = coverage_failed == 0 and hash_consistency_ok and total_documents > 0

    report = {
        "ok": ok,
        "errors": errors,
        "per_document_status": per_document_status,
        "aggregate_coverage_statistics": aggregate_coverage_stats,
        "hash_verification": hash_verification_details,
        "performance_metrics": performance_metrics,
    }

    # Raise ValidationError if any document fails coverage or hash consistency checks
    if coverage_failed > 0 or not hash_consistency_ok:
        failure_details = []
        if coverage_failed > 0:
            failure_details.append(
                f"{coverage_failed} document(s) failed coverage validation (300/300 required)"
            )
        if not hash_consistency_ok:
            failure_details.append("Hash consistency check failed")

        error_msg = "Batch post-execution validation failed:\n" + "\n".join(
            f"  - {detail}" for detail in failure_details
        )
        if errors:
            error_msg += "\n\nDetailed errors:\n" + "\n".join(
                f"  - {err}" for err in errors[:10]
            )
        raise ValidationError(error_msg)

    return report


# ---------------- CLI mínima ----------------
def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True))


def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="MINIMINIMOON System Validators")
    ap.add_argument("--repo", default=".", help="Repo root")
    ap.add_argument("--pre", action="store_true", help="Run pre-execution checks")
    ap.add_argument("--post", action="store_true", help="Run post-execution checks")
    ap.add_argument(
        "--artifacts", default="artifacts", help="Artifacts directory for post checks"
    )
    ap.add_argument(
        "--rubric-strict",
        action="store_true",
        help="Enforce 1:1 rubric vs answers in post checks",
    )
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
            post = v.validate_post_execution(
                artifacts_dir=args.artifacts, check_rubric_strict=args.rubric_strict
            )
            result["post"] = post
            if not post.get("ok", False):
                rc = 2

        if not args.pre and not args.post:
            # Por defecto ejecuta ambos
            pre = v.validate_pre_execution()
            post = v.validate_post_execution(
                artifacts_dir=args.artifacts, check_rubric_strict=args.rubric_strict
            )
            result["pre"] = pre
            result["post"] = post
            if not (pre.get("ok", False) and post.get("ok", False)):
                rc = 2

        result["ok"] = rc == 0
        _print_json(result)
        return rc
    except Exception as e:
        _print_json({"ok": False, "error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
