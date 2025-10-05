#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDM Contradiction Detection Package
====================================
Detección de contradicciones, incompatibilidades y riesgos de gobernanza
en Planes de Desarrollo Municipal (PDM) de Colombia.

@novelty_manifest:
- sentence-transformers v3.3.1 (2024-11): SOTA multilingual embeddings, release: https://github.com/UKPLab/sentence-transformers/releases/tag/v3.3.1
- transformers v4.46.0 (2024-10): Latest NLI models for Spanish, release: https://github.com/huggingface/transformers/releases/tag/v4.46.0
- typer v0.12.5 (2024-08): Modern CLI framework, release: https://github.com/tiangolo/typer/releases/tag/0.12.5
- pydantic v2.10.0 (2024-11): Data validation with strict typing, release: https://github.com/pydantic/pydantic/releases/tag/v2.10.0
- polars v1.15.0 (2024-11): High-performance dataframes, release: https://github.com/pola-rs/polars/releases/tag/py-1.15.0
- uv v0.5.10 (2024-11): Fast Python package installer, release: https://github.com/astral-sh/uv/releases/tag/0.5.10
- ruff v0.8.0 (2024-11): Fast Python linter/formatter, release: https://github.com/astral-sh/ruff/releases/tag/v0.8.0
- pypdf v5.1.0 (2024-10): Modern PDF processing, release: https://github.com/py-pdf/pypdf/releases/tag/5.1.0
- python-docx v1.1.2 (2024-06): DOCX processing, release: https://github.com/python-openxml/python-docx/releases/tag/v1.1.2
- mapie v0.9.1 (2024-09): Conformal prediction for uncertainty, release: https://github.com/scikit-learn-contrib/MAPIE/releases/tag/v0.9.1
- torch v2.5.0 (2024-10): Deep learning backend, release: https://github.com/pytorch/pytorch/releases/tag/v2.5.0
"""

from pdm_contra.core import ContradictionDetector
from pdm_contra.explain.tracer import ExplanationTracer
from pdm_contra.models import (
    CompetenceValidation,
    ContradictionAnalysis,
    ContradictionMatch,
    PDMDocument,
)
from pdm_contra.nlp.nli import SpanishNLIDetector
from pdm_contra.nlp.patterns import PatternMatcher
from pdm_contra.policy.competence import CompetenceValidator
from pdm_contra.scoring.risk import RiskScorer
from pdm_contra.utils.guard_novelty import check_dependencies

__version__ = "1.0.0"
__author__ = "PDM Analysis Team"

# pyproject.toml
PYPROJECT_TOML = """
[build-system]
requires = ["setuptools>=70.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pdm_contra"
version = "1.0.0"
description = "Detección de contradicciones en Planes de Desarrollo Municipal de Colombia"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "PDM Analysis Team", email = "pdm-analysis@example.org"}
]
keywords = ["nlp", "policy-analysis", "spanish", "municipal-planning", "colombia"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Government",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Natural Language :: Spanish"
]

dependencies = [
    "sentence-transformers>=3.3.1",
    "transformers>=4.46.0",
    "torch>=2.5.0",
    "typer>=0.12.5",
    "pydantic>=2.10.0",
    "polars>=1.15.0",
    "pypdf>=5.1.0",
    "python-docx>=1.1.2",
    "mapie>=0.9.1",
    "scikit-learn>=1.5.0",
    "numpy>=1.26.0",
    "rich>=13.9.0",
    "httpx>=0.27.0",
    "orjson>=3.10.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "ipykernel>=6.29.0"
]

[project.scripts]
pdm-contradict = "pdm_contra.cli:app"

[tool.setuptools]
packages = ["pdm_contra"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "N", "W", "UP", "B", "C90", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=pdm_contra --cov-report=term-missing"
"""
