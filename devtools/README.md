# DevTools - Comprehensive Verification Suite

This directory contains the master verification script and related tools for ensuring code quality, compilation, testing, and deployment readiness in the MINIMINIMOON project.

## Overview

The verification suite provides:
- **Local development checks** via `run_all_checks.sh`
- **CI/CD integration** via `.github/workflows/full-checks.yml`
- **Machine-readable logs** (JSON format) for all steps
- **Automated remediation** for common issues
- **Comprehensive exit codes** for precise error identification

## Quick Start

### Local Usage

```bash
# Run all checks
./devtools/run_all_checks.sh

# Run with automatic venv repair
./devtools/run_all_checks.sh --auto-repair

# Or use environment variable
AUTO_REPAIR_VENV=1 ./devtools/run_all_checks.sh
```

### What Gets Checked

1. **Environment Verification**
   - Python 3.10+ availability
   - Virtual environment validity
   - Optional: protoc and buf tools

2. **Code Quality**
   - Python bytecode compilation
   - Type checking (mypy)
   - Linting (flake8/ruff)
   - Code formatting (black)

3. **Proto Generation**
   - Protocol buffer schema generation (if applicable)
   - Verification of generated files

4. **Testing**
   - pytest test suite
   - Contract tests (producer/consumer)

5. **Registry & Determinism**
   - Evidence registry verification
   - Canonical JSON determinism checks

6. **Security**
   - Bandit security scanning
   - Safety dependency vulnerability checks

## Exit Codes Reference

### Success
- **0** - All checks passed successfully

### Code Quality Failures (11-14)
- **11** - Python compilation failure (syntax errors)
- **12** - Type checking failure (mypy)
- **13** - Linting failure (flake8/ruff)
- **14** - Code formatting failure (black)

### Test Failures (2, 22-23)
- **2** - pytest test failures
- **21** - Proto generation failure
- **22** - Producer contract test failure
- **23** - Consumer contract test failure

### Environment Issues (31, 41-42)
- **31** - Broken or invalid virtual environment
- **41** - protoc (Protocol Buffers compiler) not installed
- **42** - buf (proto management tool) not installed

### Registry & Determinism (51, 61)
- **51** - Evidence registry verification failure
- **61** - Determinism check failure (canonical_json)

### Security (71)
- **71** - Critical security vulnerability detected

## Remediation Guide

### Exit Code 11: Python Compilation Failure

**Symptom**: `bash: .venv/bin/python: No such file or directory` or syntax errors in Python files

**Cause**: Python files have syntax errors

**Remediation**:
```bash
# View the failing file (shown in error output)
# Fix syntax errors manually
python -m py_compile path/to/failing/file.py
```

### Exit Code 12: Type Checking Failure

**Symptom**: mypy reports type errors

**Cause**: Type annotations are incorrect or missing

**Remediation**:
```bash
# Run mypy with verbose output
source .venv/bin/activate
mypy python_package/ --pretty --show-error-codes

# Fix reported type errors
# Or add type: ignore comments for false positives
```

### Exit Code 13: Linting Failure

**Symptom**: flake8 reports code style violations

**Cause**: Code doesn't follow PEP 8 style guidelines

**Remediation**:
```bash
# View specific violations
source .venv/bin/activate
flake8 . --show-source

# Auto-fix some issues
autopep8 --in-place --recursive .

# Or update .flake8 config to ignore specific rules
```

### Exit Code 14: Code Formatting Failure

**Symptom**: black reports formatting issues

**Cause**: Code is not formatted according to black standards

**Remediation**:
```bash
# Auto-format all code
source .venv/bin/activate
black .

# Or check what would be changed
black --diff .
```

### Exit Code 2: Test Failures

**Symptom**: pytest reports test failures

**Cause**: Tests are failing due to code changes

**Remediation**:
```bash
# Run tests with verbose output
source .venv/bin/activate
pytest -v

# Run specific failing test
pytest path/to/test_file.py::test_name -v

# View pytest report
cat devtools/reports/pytest-report.json | jq .
```

### Exit Code 21: Proto Generation Failure

**Symptom**: Protocol buffer generation fails

**Cause**: Proto definitions have errors or protoc is misconfigured

**Remediation**:
```bash
# Check proto files for syntax errors
protoc --proto_path=protos --python_out=. protos/*.proto

# Install/update protoc
# Ubuntu/Debian:
sudo apt-get install protobuf-compiler

# macOS:
brew install protobuf
```

### Exit Code 22/23: Contract Test Failures

**Symptom**: Producer or consumer contract tests fail

**Cause**: Component interfaces don't match expected contracts

**Remediation**:
```bash
# Run contract tests individually
source .venv/bin/activate
python tests/contracts/test_embedding_model_contract.py
python tests/contracts/test_responsibility_detector_contract.py

# Review contract violations in output
# Update implementation to match contract
```

### Exit Code 31: Broken Virtual Environment

**Symptom**: `.venv/bin/python` not found or not executable

**Cause**: Virtual environment was not created properly or is corrupted

**Remediation Option 1 (Automated)**:
```bash
# Use auto-repair flag
./devtools/run_all_checks.sh --auto-repair

# Or use environment variable
AUTO_REPAIR_VENV=1 ./devtools/run_all_checks.sh
```

**Remediation Option 2 (Manual)**:
```bash
# Remove broken venv
rm -rf .venv

# Recreate from scratch
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify it works
python --version
pip freeze > .venv/pip-freeze.txt
```

**Remediation Option 3 (Using setup script)**:
```bash
# Use existing setup script
bash setup_environment.sh
```

### Exit Code 41: protoc Not Installed

**Symptom**: `protoc not found`

**Cause**: Protocol Buffers compiler is not installed

**Remediation**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install protobuf-compiler

# macOS
brew install protobuf

# Or download from GitHub releases
wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip
unzip protoc-21.12-linux-x86_64.zip -d ~/.local
export PATH="$PATH:$HOME/.local/bin"
```

### Exit Code 42: buf Not Installed

**Symptom**: `buf not found`

**Cause**: buf proto management tool is not installed (optional)

**Remediation**:
```bash
# Install buf (optional, non-critical)
curl -sSL "https://github.com/bufbuild/buf/releases/download/v1.28.1/buf-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/buf
chmod +x /usr/local/bin/buf

# Verify installation
buf --version
```

### Exit Code 51: Registry Verification Failure

**Symptom**: Evidence registry verification fails

**Cause**: Registry data is corrupted or signatures don't match

**Remediation**:
```bash
# Check registry integrity
source .venv/bin/activate
python -m evidence_registry verify

# Review registry data
cat evidence_registry.json | jq .

# Regenerate registry if needed
python -m evidence_registry rebuild
```

### Exit Code 61: Determinism Check Failure

**Symptom**: canonical_json produces different output for same input

**Cause**: JSON serialization is not deterministic

**Remediation**:
```bash
# Review canonical_json implementation
# Ensure:
# 1. Keys are always sorted
# 2. No random elements
# 3. Consistent whitespace

# Test manually
source .venv/bin/activate
python -c "
from json_utils import canonical_json
sample = {'z': 1, 'a': 2}
print(canonical_json(sample))
"
```

### Exit Code 71: Security Vulnerability

**Symptom**: bandit or safety reports critical vulnerabilities

**Cause**: Code has security issues or dependencies have known vulnerabilities

**Remediation**:
```bash
# Review bandit report
source .venv/bin/activate
cat devtools/reports/bandit-report.json | jq .

# Review safety report
cat devtools/reports/safety-report.json | jq .

# Update vulnerable dependencies
pip list --outdated
pip install --upgrade <package-name>

# Or use safety to fix
safety check --apply-fixes
```

## Reports

All checks produce detailed reports in `devtools/reports/`:

- **checks-summary.json** - Overall summary with all step results
- **pytest-report.json** - Detailed pytest results
- **bandit-report.json** - Security scan results
- **safety-report.json** - Dependency vulnerability scan
- **\*_stdout.log** - Standard output for each step
- **\*_stderr.log** - Error output for each step

### Example Summary Structure

```json
{
  "run_timestamp": "2025-10-13T08:00:00Z",
  "hostname": "runner-xyz",
  "status": "success",
  "end_timestamp": "2025-10-13T08:05:00Z",
  "steps": [
    {
      "name": "Python Compilation",
      "status": "success",
      "exit_code": 0,
      "start_time": "2025-10-13T08:01:00Z",
      "end_time": "2025-10-13T08:01:30Z",
      "stdout": "...",
      "stderr": ""
    }
  ]
}
```

## CI Integration

The suite runs automatically in CI via `.github/workflows/full-checks.yml`:

- **Runs on**: Push to main/master, all pull requests
- **Python versions**: 3.10, 3.11
- **Fail-fast**: Stops on first failure
- **Artifacts**: Uploads reports on failure
- **PR Comments**: Posts results as PR comment

### CI Features

1. **Pip caching** - Speeds up dependency installation
2. **Matrix builds** - Tests multiple Python versions
3. **Artifact upload** - Preserves reports for debugging
4. **PR comments** - Automatic feedback on pull requests

## Best Practices

### Before Committing

```bash
# Run checks locally
./devtools/run_all_checks.sh

# Fix any issues before pushing
```

### In CI/CD Pipeline

The checks run automatically, but you can:

1. Review the PR comment for quick summary
2. Download artifacts for detailed analysis
3. Fix issues and push again

### Continuous Improvement

- Add new checks to `run_all_checks.sh` as needed
- Update exit codes when adding new check types
- Document remediation steps for new failures
- Keep this README up to date

## Troubleshooting

### Script Won't Run

```bash
# Make sure script is executable
chmod +x devtools/run_all_checks.sh

# Check bash is available
which bash
```

### Missing Dependencies

```bash
# Recreate environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CI Failures

1. Check the summary in PR comment
2. Download artifacts from failed run
3. Review specific error logs
4. Reproduce locally with same Python version
5. Fix and push again

## Advanced Usage

### Custom Check Configuration

Set environment variables before running:

```bash
# Skip certain checks
export SKIP_SECURITY_SCAN=1

# Use specific Python
export PYTHON_BIN=/usr/bin/python3.10

# Custom reports directory
export REPORTS_DIR=/tmp/my-reports
```

### Integration with Pre-commit

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: full-checks
        name: Full verification suite
        entry: devtools/run_all_checks.sh
        language: system
        pass_filenames: false
```

### IDE Integration

Configure your IDE to run checks:

- **VS Code**: Add task in `.vscode/tasks.json`
- **PyCharm**: Add External Tool
- **Vim/Neovim**: Add keybinding

## Support

For issues or questions:

1. Check this README for remediation steps
2. Review reports in `devtools/reports/`
3. Check existing issues in the repository
4. Create a new issue with:
   - Exit code
   - Full error output
   - Summary JSON
   - Steps to reproduce

## Version History

- **v1.0.0** (2025-10-13) - Initial comprehensive verification suite
  - All basic checks implemented
  - CI integration complete
  - Full documentation
