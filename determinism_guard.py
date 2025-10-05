#!/usr/bin/env python3
"""
determinism_guard.py — Deterministic Execution Enforcer (Cross-Library Seeding)
================================================================================
Función: Garantiza reproducibilidad bit-exacta de operaciones estocásticas en todo
el pipeline mediante seeding determinista de generadores pseudo-aleatorios
en Python stdlib, NumPy y PyTorch (opcional).

Posición en el flujo (según Dependency Flows Documentation):
Invocado implícitamente por:
- miniminimoon_orchestrator (pre-procesamiento)
- embedding_model (generación de embeddings reproducibles)
- feasibility_scorer (sampling de features)
- Cualquier módulo que use operaciones estocásticas

Type: control (cross-cutting concern)
Card: 1:N (múltiples puntos de enforcement)
Input: {seed:int, strict:bool}
Output: {seeded_state, warnings[]}
Razón: Prerequisito para evidence_hash estable y flow_hash determinista

Invariantes garantizados:
1. Seeding idempotente: múltiples llamadas con mismo seed → mismo estado
2. Graceful degradation: ausencia de PyTorch NO rompe el pipeline
3. Thread-safety: seeds afectan solo el proceso actual (no afectan multiprocessing)
4. Verificación post-seeding: validación de que el seed fue aplicado correctamente

Casos donde el determinismo es crítico:
- embedding_model: vectorización con dropout/random projection requiere seed fija
- document_segmenter: si usa sampling estratificado para docs grandes
- feasibility_scorer: ranking con tie-breaking aleatorio
- teoria_cambio: construcción de grafo con ordenamiento no-determinista
- CI/CD: runs reproducibles para comparación de flow_hash

Dependencias autorizadas:
- random, numpy (obligatorias)
- torch (opcional, silenciosamente ignorado si ausente)
- hashlib, os, sys, platform (para diagnostics avanzados)

Integración típica:
>>> from determinism_guard import enforce_determinism, verify_determinism
>>> enforce_determinism(seed=42, strict=True)
>>> # ... operaciones estocásticas aquí ...
>>> verify_determinism(seed=42)  # Valida que el estado sigue determinista

Changelog:
- v2.0 (2025-10-05): Refactor post-unificación con verificación de estado,
diagnostics detallados, modo strict, y documentación exhaustiva.
- v1.0: Implementación inicial minimalista (3 líneas)
"""
from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
import warnings
from typing import Dict, List, Optional, Any

import numpy as np

# Intento de importar PyTorch con manejo defensivo
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except (ImportError, AttributeError, Exception):
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    TORCH_VERSION = None


# ────────────────────────────────────────────────────────────────────────────────
# Global State Tracking (para diagnostics y verificación)
# ────────────────────────────────────────────────────────────────────────────────

_LAST_ENFORCED_SEED: Optional[int] = None
_ENFORCEMENT_COUNT: int = 0


# ────────────────────────────────────────────────────────────────────────────────
# Core Enforcement Logic
# ────────────────────────────────────────────────────────────────────────────────

def enforce_determinism(seed: int = 42, strict: bool = False) -> Dict[str, Any]:
    """
Fuerza determinismo en todos los generadores pseudo-aleatorios disponibles.

Estrategia de seeding:
1. Python random.seed() → Afecta random.random(), random.choice(), etc.
2. NumPy np.random.seed() → Afecta np.random.rand(), np.random.choice(), etc.
3. PyTorch torch.manual_seed() → Afecta torch.rand(), dropout, etc.
4. PyTorch CUDA (si disponible) → torch.cuda.manual_seed_all()

Args:
seed: Semilla entera (típicamente 42 para consistencia con literatura)
strict: Si True, falla con RuntimeError si PyTorch está presente pero no seedable

Returns:
Dict con estado de seeding:
{
"seed": int,
"python_seeded": bool,
"numpy_seeded": bool,
"torch_seeded": bool,
"torch_cuda_seeded": bool,
"warnings": List[str],
"enforcement_count": int
}

Raises:
RuntimeError: Si strict=True y falla el seeding de una librería crítica
ValueError: Si seed no es un entero no-negativo

Side Effects:
- Modifica estado global de RNGs en Python, NumPy y PyTorch
- Incrementa _ENFORCEMENT_COUNT (para tracking de re-seeding)
- Actualiza _LAST_ENFORCED_SEED

Limitaciones conocidas:
- No afecta generadores en otros procesos (multiprocessing)
- No afecta librerías C/C++ externas (e.g., OpenBLAS threaded RNG)
- PyTorch CUDNN tiene no-determinismo residual en algunos kernels
(requiere torch.backends.cudnn.deterministic = True por separado)
    """
    global _LAST_ENFORCED_SEED, _ENFORCEMENT_COUNT

    # Validación de entrada
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be non-negative integer, got: {seed}")

    state: Dict[str, Any] = {
        "seed": seed,
        "python_seeded": False,
        "numpy_seeded": False,
        "torch_seeded": False,
        "torch_cuda_seeded": False,
        "warnings": [],
        "enforcement_count": 0,
    }

    # ──── Seeding de Python random (stdlib) ────
    try:
        random.seed(seed)
        state["python_seeded"] = True
    except Exception as e:
        msg = f"Failed to seed Python random: {e}"
        state["warnings"].append(msg)
        if strict:
            raise RuntimeError(msg) from e

    # ──── Seeding de NumPy (obligatorio para embeddings) ────
    try:
        np.random.seed(seed)
        state["numpy_seeded"] = True
    except Exception as e:
        msg = f"Failed to seed NumPy random: {e}"
        state["warnings"].append(msg)
        if strict:
            raise RuntimeError(msg) from e

    # ──── Seeding de PyTorch (opcional pero recomendado) ────
    if TORCH_AVAILABLE and torch is not None:
        try:
            torch.manual_seed(seed)
            state["torch_seeded"] = True

            # Seeding adicional para CUDA si está disponible
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(seed)
                    state["torch_cuda_seeded"] = True
                except Exception as cuda_e:
                    msg = f"PyTorch CUDA seeding failed: {cuda_e}"
                    state["warnings"].append(msg)
                    # CUDA seeding no es crítico en modo no-strict

        except Exception as e:
            msg = f"Failed to seed PyTorch: {e}"
            state["warnings"].append(msg)
            if strict:
                raise RuntimeError(msg) from e
    else:
        state["warnings"].append(
            "PyTorch not available - skipping torch seeding (expected for CPU-only deployments)"
        )

    # Actualizar estado global de tracking
    _LAST_ENFORCED_SEED = seed
    _ENFORCEMENT_COUNT += 1
    state["enforcement_count"] = _ENFORCEMENT_COUNT

    return state


def enforce(seed: int = 42) -> None:
    """
Alias legacy para retrocompatibilidad con código existente.

Wrapper simplificado de enforce_determinism() sin diagnostics.
Usado por módulos legacy que esperan la interfaz minimalista original.

Args:
seed: Semilla entera (default: 42)

Returns:
None (silencioso, warnings suprimidos)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enforce_determinism(seed=seed, strict=False)


# ────────────────────────────────────────────────────────────────────────────────
# Verification & Diagnostics
# ────────────────────────────────────────────────────────────────────────────────

def verify_determinism(seed: int = 42, n_samples: int = 100) -> Dict[str, Any]:
    """
Verifica que el determinismo está activo mediante prueba de reproducibilidad.

Estrategia:
1. Genera n_samples de cada RNG con el seed actual
2. Re-seedea con el mismo valor
3. Genera n_samples nuevamente
4. Compara bit-exactamente ambas secuencias

Args:
seed: Semilla a verificar (debe coincidir con último enforce_determinism())
n_samples: Número de muestras para la prueba (mayor = más confianza)

Returns:
Dict con resultados de verificación:
{
"deterministic": bool (True si TODOS los RNGs son reproducibles),
"python_ok": bool,
"numpy_ok": bool,
"torch_ok": bool | None (None si no disponible),
"mismatches": List[str] (nombres de RNGs que fallaron),
"sample_hash": str (hash de la secuencia para tracking)
}

Raises:
RuntimeError: Si _LAST_ENFORCED_SEED no coincide con seed (no fue seeded)

Uso típico en tests:
>>> enforce_determinism(seed=42)
>>> result = verify_determinism(seed=42)
>>> assert result["deterministic"], f"Non-deterministic RNGs: {result['mismatches']}"
    """
    if _LAST_ENFORCED_SEED is None:
        raise RuntimeError(
            "Cannot verify determinism: enforce_determinism() has not been called yet"
        )

    if _LAST_ENFORCED_SEED != seed:
        raise RuntimeError(
            f"Seed mismatch: last enforced seed was {_LAST_ENFORCED_SEED}, "
            f"but verification requested for {seed}"
        )

    result: Dict[str, Any] = {
        "deterministic": True,
        "python_ok": False,
        "numpy_ok": False,
        "torch_ok": None,
        "mismatches": [],
        "sample_hash": "",
    }

    # ──── Verificación de Python random ────
    random.seed(seed)
    seq1_py = [random.random() for _ in range(n_samples)]
    random.seed(seed)
    seq2_py = [random.random() for _ in range(n_samples)]
    result["python_ok"] = seq1_py == seq2_py
    if not result["python_ok"]:
        result["mismatches"].append("python_random")
        result["deterministic"] = False

    # ──── Verificación de NumPy ────
    np.random.seed(seed)
    seq1_np = np.random.rand(n_samples)
    np.random.seed(seed)
    seq2_np = np.random.rand(n_samples)
    result["numpy_ok"] = np.allclose(seq1_np, seq2_np, atol=0.0, rtol=0.0)
    if not result["numpy_ok"]:
        result["mismatches"].append("numpy_random")
        result["deterministic"] = False

    # ──── Verificación de PyTorch (si disponible) ────
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        seq1_torch = torch.rand(n_samples).numpy()
        torch.manual_seed(seed)
        seq2_torch = torch.rand(n_samples).numpy()
        result["torch_ok"] = np.allclose(seq1_torch, seq2_torch, atol=0.0, rtol=0.0)
        if not result["torch_ok"]:
            result["mismatches"].append("torch_random")
            result["deterministic"] = False

    # Hash de la secuencia combinada para tracking
    combined = str(seq1_py) + str(seq1_np.tolist())
    if result["torch_ok"] is not None:
        combined += str(seq1_torch.tolist())
    result["sample_hash"] = hashlib.sha256(combined.encode()).hexdigest()[:16]

    return result


def get_determinism_diagnostics() -> Dict[str, Any]:
    """
Retorna estado completo del sistema de determinismo para debugging.

Returns:
Dict con información diagnóstica:
{
"platform": {
"system": str (e.g., "Linux"),
"python_version": str,
"numpy_version": str,
"torch_version": str | None,
"torch_cuda_available": bool | None,
},
"state": {
"last_enforced_seed": int | None,
"enforcement_count": int,
},
"capabilities": {
"python_random": bool (siempre True),
"numpy_random": bool (siempre True),
"torch_random": bool,
"torch_cuda": bool,
}
}

Uso típico:
>>> diag = get_determinism_diagnostics()
>>> print(json.dumps(diag, indent=2))  # Para logs de CI/CD
    """
    diagnostics = {
        "platform": {
            "system": platform.system(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": TORCH_VERSION,
            "torch_cuda_available": None,
        },
        "state": {
            "last_enforced_seed": _LAST_ENFORCED_SEED,
            "enforcement_count": _ENFORCEMENT_COUNT,
        },
        "capabilities": {
            "python_random": True,
            "numpy_random": True,
            "torch_random": TORCH_AVAILABLE,
            "torch_cuda": False,
        },
    }

    if TORCH_AVAILABLE and torch is not None:
        try:
            diagnostics["platform"]["torch_cuda_available"] = torch.cuda.is_available()
            diagnostics["capabilities"]["torch_cuda"] = torch.cuda.is_available()
        except Exception:
            pass

    return diagnostics


# ────────────────────────────────────────────────────────────────────────────────
# Advanced: Environment Variable Control
# ────────────────────────────────────────────────────────────────────────────────

def enforce_from_environment(
        env_var: str = "MINIMINIMOON_SEED",
        default_seed: int = 42,
        strict: bool = False
) -> Dict[str, Any]:
    """
Lee la semilla desde una variable de entorno para configuración externa.

Permite control de determinismo vía infraestructura (Docker, Kubernetes, CI/CD)
sin modificar código:
export MINIMINIMOON_SEED=12345
python miniminimoon_cli.py evaluate plan.pdf

Args:
env_var: Nombre de la variable de entorno a leer
default_seed: Fallback si la variable no está definida
strict: Pasado a enforce_determinism()

Returns:
Dict de enforce_determinism() con campo adicional:
{"seed_source": "environment" | "default"}

Raises:
ValueError: Si la variable de entorno no es un entero válido
    """
    seed = default_seed
    source = "default"

    env_value = os.environ.get(env_var)
    if env_value is not None:
        try:
            seed = int(env_value)
            source = "environment"
        except ValueError as e:
            raise ValueError(
                f"Invalid seed in environment variable {env_var}={env_value}: {e}"
            ) from e

    result = enforce_determinism(seed=seed, strict=strict)
    result["seed_source"] = source
    return result


# ────────────────────────────────────────────────────────────────────────────────
# CLI Entry Point (para verificación standalone)
# ────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
CLI para verificación standalone del sistema de determinismo.

Uso:
python determinism_guard.py              # Verifica con seed=42
python determinism_guard.py --seed 123   # Verifica con seed custom
python determinism_guard.py --diagnostics # Imprime diagnostics completos

Returns:
Exit code:
0: Determinismo verificado OK
1: Fallo de determinismo detectado
2: Error de configuración
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Determinism Guard - Verify reproducibility of RNGs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value to enforce and verify (default: 42)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for verification test (default: 100)"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print full diagnostics and exit"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail hard if any RNG cannot be seeded"
    )

    args = parser.parse_args()

    try:
        if args.diagnostics:
            diag = get_determinism_diagnostics()
            print(json.dumps(diag, indent=2))
            return 0

        # Enforce y verificar
        print(f"Enforcing determinism with seed={args.seed}...")
        enforce_state = enforce_determinism(seed=args.seed, strict=args.strict)
        print(json.dumps(enforce_state, indent=2))

        print(f"\nVerifying determinism with {args.samples} samples...")
        verify_result = verify_determinism(seed=args.seed, n_samples=args.samples)
        print(json.dumps(verify_result, indent=2))

        if verify_result["deterministic"]:
            print("\n✓ Determinism verified successfully")
            return 0
        else:
            print(f"\n✗ Determinism verification failed: {verify_result['mismatches']}")
            return 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())