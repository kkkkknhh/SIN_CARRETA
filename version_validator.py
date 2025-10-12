# sin_carreta/env_validation.py
from __future__ import annotations
import sys
import importlib
from dataclasses import dataclass
from typing import List, Tuple

REQUIRED_PYTHON: Tuple[int, int] = (3, 10)
NUMPY_MIN: str = "1.21.0"      # first with Py3.10 wheels
NUMPY_MAX_EXCL: str | None = None  # set e.g. "1.25.0" if you want a hard ceiling

@dataclass(frozen=True)
class ValidationIssue:
    component: str
    found: str
    required: str
    hint: str

class EnvironmentValidationError(RuntimeError):
    def __init__(self, issues: List[ValidationIssue]):
        lines = ["Environment validation failed:"]
        for it in issues:
            lines.append(f"- {it.component}: found {it.found}; require {it.required}. {it.hint}")
        super().__init__("\n".join(lines))
        self.issues = issues

def _parse_ver(ver: str) -> Tuple[int, int, int]:
    parts = ver.split(".")
    ints = []
    for p in parts[:3]:
        n = ""
        for ch in p:
            if ch.isdigit(): n += ch
            else: break
        ints.append(int(n or 0))
    while len(ints) < 3:
        ints.append(0)
    return tuple(ints)  # type: ignore[return-value]

def _cmp(a: str, b: str) -> int:
    A, B = _parse_ver(a), _parse_ver(b)
    return (A > B) - (A < B)

def validate_environment(strict: bool = True) -> None:
    issues: List[ValidationIssue] = []

    py = sys.version_info
    req = f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x"
    if not (py.major == REQUIRED_PYTHON[0] and py.minor == REQUIRED_PYTHON[1]):
        issues.append(ValidationIssue(
            "Python",
            f"{py.major}.{py.minor}.{py.micro}",
            req,
            "Use a Python 3.10 interpreter (e.g., pyenv or a 3.10-pinned venv)."
        ))

    try:
        numpy = importlib.import_module("numpy")
        nv = getattr(numpy, "__version__", "0.0.0")
        if _cmp(nv, NUMPY_MIN) < 0:
            issues.append(ValidationIssue(
                "NumPy",
                nv,
                f">= {NUMPY_MIN}",
                "Upgrade: pip install 'numpy>="+NUMPY_MIN+"'"
            ))
        if NUMPY_MAX_EXCL and _cmp(nv, NUMPY_MAX_EXCL) >= 0:
            issues.append(ValidationIssue(
                "NumPy",
                nv,
                f"< {NUMPY_MAX_EXCL}",
                f"Downgrade: pip install 'numpy<{NUMPY_MAX_EXCL}'"
            ))
    except ModuleNotFoundError:
        issues.append(ValidationIssue(
            "NumPy",
            "not installed",
            f">= {NUMPY_MIN}"+(f", < {NUMPY_MAX_EXCL}" if NUMPY_MAX_EXCL else ""),
            "Install: pip install 'numpy>="+NUMPY_MIN+"'"
        ))

    if issues and strict:
        raise EnvironmentValidationError(issues)

    if issues and not strict:
        # Return soft-fail signal for callers that want to *log* and proceed.
        # Importantly, still no side effects at import.
        msg = EnvironmentValidationError(issues)
        # Let caller decide logging; just raise for visibility if they ignore return.
        raise msg  # remove this line if you truly want non-throwing soft mode

def cli() -> int:
    try:
        validate_environment(strict=True)
        print("Environment validation: OK")
        return 0
    except EnvironmentValidationError as e:
        print(e, file=sys.stderr)
        return 2

# No import-time execution. Call validate_environment() or cli() explicitly.
