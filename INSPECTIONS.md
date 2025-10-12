# Code Quality Inspections

## Overview

The inspection system provides comprehensive code quality checks including:
- **Python Bytecode Compilation**: Validates syntax errors
- **Flake8 Linting**: PEP 8 style checking
- **Mypy Type Checking**: Static type analysis
- **Ruff Linting**: Fast Python linter (optional)

## Usage

### Run All Inspections

```bash
# Default mode (warnings only, exits 0)
python run_inspections.py

# Strict mode (fails on any error)
python run_inspections.py --strict

# Fail-fast mode (stop on first failure)
python run_inspections.py --fail-fast
```

### Run Individual Inspections

```bash
# Bytecode compilation
python -m compileall -q .

# Flake8 linting
flake8 .

# Mypy type checking
mypy . --config-file pyproject.toml

# Ruff linting (if installed)
ruff check .
```

## Configuration

### Flake8 (.flake8)
- Max line length: 88 (Black-compatible)
- Ignores: E203 (whitespace before ':'), W503 (line break before binary operator)
- Excludes: build artifacts, virtual environments, output directories

### Mypy (pyproject.toml)
- Python version: 3.10
- Ignores missing imports
- Pretty error messages with error codes
- Excludes: virtual environments, output directories

### Ruff (pyproject.toml)
- Line length: 88
- Target version: Python 3.10
- Selects: E (errors), F (pyflakes), W (warnings), I (isort)

## CI Integration

Inspections are run automatically in GitHub Actions CI:
- Bytecode compilation (blocking)
- Flake8 linting (non-blocking, continue-on-error)
- Mypy type checking (non-blocking, continue-on-error)
- Pytest tests (blocking)

See `.github/workflows/ci.yml` for details.

## Development Workflow

1. Make code changes
2. Run inspections locally:
   ```bash
   python run_inspections.py
   ```
3. Fix any critical errors (compilation, syntax)
4. Address style/type issues as needed
5. Commit and push (CI will run inspections)

## Exit Codes

- **0**: All checks passed OR warning mode (default)
- **1**: One or more checks failed in strict mode
- **Command exit codes**: Individual tools may return their own codes

## Testing

Test the inspection system:
```bash
python test_run_inspections.py
```

This validates:
- Script exists and is executable
- Help flag works
- Default mode exits cleanly
- Strict mode can fail
- Individual commands work
