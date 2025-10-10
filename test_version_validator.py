# test_env_validation.py

from __future__ import annotations

import sys
import types
from typing import List, Tuple

import pytest

# Target under test: sin_carreta/env_validation.py
from sin_carreta.env_validation import (
    EnvironmentValidationError,
    ValidationIssue,
    validate_environment,
    cli,
    _parse_ver,   # internal helpers are tested lightly for determinism
    _cmp,
    REQUIRED_PYTHON,
    NUMPY_MIN,
    NUMPY_MAX_EXCL,
)


def _fake_numpy(version: str) -> types.ModuleType:
    m = types.ModuleType("numpy")
    m.__version__ = version
    return m


class TestInternalHelpers:
    def test_parse_ver_basic(self):
        assert _parse_ver("1.2.3") == (1, 2, 3)

    def test_parse_ver_extra_tags(self):
        assert _parse_ver("1.21.0rc1") == (1, 21, 0)
        assert _parse_ver("2.0.0+cpu") == (2, 0, 0)
        assert _parse_ver("0.9") == (0, 9, 0)

    def test_cmp(self):
        assert _cmp("1.2.3", "1.2.3") == 0
        assert _cmp("1.2.4", "1.2.3") == 1
        assert _cmp("1.2.3", "1.3.0") == -1


class TestPythonGate:
    def test_python_gate_matches_runtime(self):
        major, minor = REQUIRED_PYTHON
        # This test passes in environments that actually match REQUIRED_PYTHON.
        if sys.version_info.major == major and sys.version_info.minor == minor:
            validate_environment(strict=True)  # should not raise

    def test_python_gate_mismatch_raises(self, monkeypatch):
        major, minor = REQUIRED_PYTHON
        wrong_minor = 9 if minor != 9 else 8
        class FakeVersion:
            major = major
            minor = wrong_minor
            micro = 0
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)
        with pytest.raises(EnvironmentValidationError) as exc:
            validate_environment(strict=True)
        msg = str(exc.value)
        assert "Python" in msg
        assert f"{major}.{wrong_minor}.0" in msg
        assert f"{major}.{minor}.x" in msg


class TestNumpyGate:
    def test_numpy_not_installed_raises(self, monkeypatch):
        # Remove numpy from sys.modules and make import fail
        monkeypatch.dict(sys.modules, {"numpy": None}, clear=False)
        with pytest.raises(EnvironmentValidationError) as exc:
            validate_environment(strict=True)
        err: EnvironmentValidationError = exc.value
        comps = [i.component for i in err.issues]
        assert "NumPy" in comps
        assert any("not installed" in i.found for i in err.issues if i.component == "NumPy")
        # Ensure message carries actionable hint
        assert "pip install" in str(err)

    def test_numpy_old_version_raises(self, monkeypatch):
        old = "1.20.0"
        monkeypatch.dict(sys.modules, {"numpy": _fake_numpy(old)}, clear=False)
        # Force Python gate to pass by mirroring current runtime major/minor
        class FakeVersion:
            major = sys.version_info.major
            minor = sys.version_info.minor
            micro = sys.version_info.micro
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)

        with pytest.raises(EnvironmentValidationError) as exc:
            validate_environment(strict=True)
        err: EnvironmentValidationError = exc.value
        issue = next(i for i in err.issues if i.component == "NumPy")
        assert issue.found == old
        assert f">= {NUMPY_MIN}" in issue.required

    def test_numpy_good_version_passes(self, monkeypatch):
        # Pick a version that satisfies >= NUMPY_MIN and (if set) < NUMPY_MAX_EXCL
        good = "1.23.0"
        if NUMPY_MAX_EXCL is not None and _cmp(good, NUMPY_MAX_EXCL) >= 0:
            good = "1.24.0" if _cmp("1.24.0", NUMPY_MAX_EXCL) < 0 else "1.22.0"

        monkeypatch.dict(sys.modules, {"numpy": _fake_numpy(good)}, clear=False)
        # Mirror current Python so Python gate doesn't fail
        class FakeVersion:
            major = sys.version_info.major
            minor = sys.version_info.minor
            micro = sys.version_info.micro
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)

        # Should not raise in strict mode if only NumPy gate is relevant and satisfied
        validate_environment(strict=True)

    def test_numpy_upper_bound_enforced_when_configured(self, monkeypatch):
        if NUMPY_MAX_EXCL is None:
            pytest.skip("Upper bound not configured; skip test.")
        bad = NUMPY_MAX_EXCL
        monkeypatch.dict(sys.modules, {"numpy": _fake_numpy(bad)}, clear=False)
        class FakeVersion:
            major = sys.version_info.major
            minor = sys.version_info.minor
            micro = sys.version_info.micro
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)

        with pytest.raises(EnvironmentValidationError) as exc:
            validate_environment(strict=True)
        err: EnvironmentValidationError = exc.value
        issue = next(i for i in err.issues if i.component == "NumPy")
        assert issue.found == bad
        assert f"< {NUMPY_MAX_EXCL}" in issue.required


class TestAggregatedReporting:
    def test_multiple_issues_aggregate(self, monkeypatch):
        # Force wrong Python minor AND missing numpy
        class FakeVersion:
            major = sys.version_info.major
            minor = (sys.version_info.minor + 1) % 11  # likely different
            micro = 0
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)
        monkeypatch.dict(sys.modules, {"numpy": None}, clear=False)

        with pytest.raises(EnvironmentValidationError) as exc:
            validate_environment(strict=True)
        err: EnvironmentValidationError = exc.value
        assert isinstance(err.issues, list) and len(err.issues) >= 2
        comps = {i.component for i in err.issues}
        assert {"Python", "NumPy"}.issubset(comps)


class TestCLI:
    def test_cli_ok(self, monkeypatch, capsys):
        # Make both gates pass
        class FakeVersion:
            major = REQUIRED_PYTHON[0]
            minor = REQUIRED_PYTHON[1]
            micro = 5
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)
        monkeypatch.dict(sys.modules, {"numpy": _fake_numpy(max(NUMPY_MIN, "1.23.0"))}, clear=False)

        rc = cli()
        captured = capsys.readouterr()
        assert rc == 0
        assert "Environment validation: OK" in captured.out

    def test_cli_fail(self, monkeypatch, capsys):
        # Force failure
        class FakeVersion:
            major = REQUIRED_PYTHON[0]
            minor = REQUIRED_PYTHON[1] - 1 if REQUIRED_PYTHON[1] > 0 else 0
            micro = 0
        monkeypatch.setattr(sys, "version_info", FakeVersion, raising=False)
        monkeypatch.dict(sys.modules, {"numpy": None}, clear=False)

        rc = cli()
        captured = capsys.readouterr()
        assert rc == 2
        assert "Environment validation failed:" in captured.err


if __name__ == "__main__":
    pytest.main([__file__])
