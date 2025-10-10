"""Test suite for environment validation contracts."""

from __future__ import annotations

import sys
import types
from typing import List

import pytest

from version_validator import (
    EnvironmentValidationError,
    ValidationDiagnostic,
    get_python_version_info,
    validate_environment,
    validate_numpy_compatibility,
    validate_python_310,
)


def _make_numpy_module(version: str) -> types.ModuleType:
    module = types.ModuleType("numpy")
    module.__version__ = version
    return module


class TestVersionValidator:
    """Test version validation functionality."""

    def test_get_python_version_info(self) -> None:
        """Ensure interpreter metadata is returned as a tuple of integers."""

        version_info = get_python_version_info()
        assert isinstance(version_info, tuple)
        assert len(version_info) == 3
        assert all(isinstance(part, int) for part in version_info)

    def test_validate_python_310_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Validation emits a diagnostic when the interpreter is compliant."""

        monkeypatch.setattr("version_validator.get_python_version_info", lambda: (3, 10, 1))
        diagnostic = validate_python_310()
        assert isinstance(diagnostic, ValidationDiagnostic)
        assert diagnostic.passed is True
        assert diagnostic.details["actual"] == "3.10.1"

    def test_validate_python_310_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-compliant interpreters raise an EnvironmentValidationError."""

        monkeypatch.setattr("version_validator.get_python_version_info", lambda: (3, 9, 0))
        with pytest.raises(EnvironmentValidationError) as exc_info:
            validate_python_310()
        diagnostic = exc_info.value.diagnostic
        assert diagnostic.check == "python_310"
        assert diagnostic.passed is False
        assert diagnostic.details["actual"] == "3.9.0"

    def test_validate_python_310_wrong_major(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Major version mismatches also surface through the diagnostic payload."""

        monkeypatch.setattr("version_validator.get_python_version_info", lambda: (2, 7, 18))
        with pytest.raises(EnvironmentValidationError) as exc_info:
            validate_python_310()
        assert exc_info.value.diagnostic.details["actual"] == "2.7.18"

    def test_validate_numpy_compatibility_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Import failures propagate as deterministic validation errors."""

        original_import = __import__

        def fake_import(name: str, *args, **kwargs):
            if name == "numpy":
                raise ImportError("No module named 'numpy'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(EnvironmentValidationError) as exc_info:
            validate_numpy_compatibility()
        diagnostic = exc_info.value.diagnostic
        assert diagnostic.check == "numpy_import"
        assert diagnostic.details["actual"] == "missing"

    def test_validate_numpy_compatibility_old_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Older NumPy versions violate the compatibility contract."""

        module = _make_numpy_module("1.20.0")
        monkeypatch.setitem(sys.modules, "numpy", module)
        with pytest.raises(EnvironmentValidationError) as exc_info:
            validate_numpy_compatibility()
        diagnostic = exc_info.value.diagnostic
        assert diagnostic.check == "numpy_version"
        assert diagnostic.details["actual"] == "1.20.0"

    def test_validate_numpy_compatibility_new_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Too-new NumPy versions are blocked to preserve determinism."""

        module = _make_numpy_module("1.25.0")
        monkeypatch.setitem(sys.modules, "numpy", module)
        with pytest.raises(EnvironmentValidationError):
            validate_numpy_compatibility()

    def test_validate_numpy_compatibility_good_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Supported NumPy releases pass validation and yield diagnostics."""

        module = _make_numpy_module("1.24.0")
        monkeypatch.setitem(sys.modules, "numpy", module)
        diagnostic = validate_numpy_compatibility()
        assert diagnostic.passed is True
        assert diagnostic.details["actual"] == "1.24.0"

    def test_validate_environment_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_environment aggregates diagnostics for all checks."""

        monkeypatch.setattr("version_validator.get_python_version_info", lambda: (3, 10, 8))
        module = _make_numpy_module("1.24.1")
        monkeypatch.setitem(sys.modules, "numpy", module)
        calls: List[ValidationDiagnostic] = []

        diagnostics = validate_environment(observer=calls.append)

        assert len(diagnostics) == 2
        assert calls == diagnostics
        assert all(diag.passed for diag in diagnostics)

    def test_validate_environment_failure_reports_observer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Observers receive diagnostics for both successes and failures."""

        monkeypatch.setattr("version_validator.get_python_version_info", lambda: (3, 10, 2))
        module = _make_numpy_module("1.26.0")
        monkeypatch.setitem(sys.modules, "numpy", module)
        calls: List[ValidationDiagnostic] = []

        with pytest.raises(EnvironmentValidationError) as exc_info:
            validate_environment(observer=calls.append)

        diagnostic = exc_info.value.diagnostic
        assert diagnostic.check == "numpy_version"
        assert diagnostic.passed is False
        assert len(calls) == 2  # python success + numpy failure
        assert calls[0].check == "python_310"
        assert calls[1] == diagnostic


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    pytest.main([__file__])
