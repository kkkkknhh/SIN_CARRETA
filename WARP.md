# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

MINIMINIMOON is an **automated questionnaire answering system** designed to evaluate Municipal Development Plans (PDMs) against a structured 300-question framework for human rights compliance. The system's primary purpose is to automate the answering of a standardized questionnaire with exactly:

- **10 Policy Domains (P1-P10)**: Each corresponding to a human rights catalog item
- **30 Questions per Domain**: Standardized across 6 dimensions (D1-D6, 5 questions each)
- **Total: 300 Questions (10 × 30)**: Complete automated evaluation framework

The system combines advanced NLP techniques including causal inference, contradiction detection, responsibility assignment, and feasibility scoring to automatically provide evidence-based responses to each of the 300 standardized questions.

**Core Mission**: Preserve the exact structure and wording of the original 300-question questionnaire while automating the evidence detection and response generation process.

## Common Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download required spaCy Spanish model
python3 -m spacy download es_core_news_sm
```

### Building and Validation
```bash
# Check component compilation (build verification)
python3 -c "import embedding_model, responsibility_detector, dag_validation; print('Core components imported successfully')"
python3 -m py_compile teoria_cambio.py
python3 -m py_compile decalogo_loader.py

# Lint code (follows PEP 8)
python3 -m py_compile embedding_model.py test_embedding_model.py
python3 -m py_compile responsibility_detector.py test_responsibility_detector.py
```

### Running Tests
```bash
# Run core component tests
python3 -m pytest test_embedding_model.py -v
python3 -m pytest test_responsibility_detector.py -v

# Run all tests
python3 run_all_tests.py

# Run specific test suites
python3 test_factibilidad.py
python3 test_dag_validation.py 2>/dev/null || echo "DAG validation optional"
```

### System Execution
```bash
# CLI interface for policy analysis
python cli.py --input ./documents --outdir ./results
python cli.py --input ./documents --workers 8 --device cuda --precision float32

# System runner for comprehensive plan analysis
python run_system.py analyze path/to/plan.txt
python run_system.py batch path/to/plans_directory/
python run_system.py interactive

# Core Decatalogo Evaluation System (MAIN ENTRY POINTS)
python Decatalogo_evaluador.py  # Industrial decalogo evaluator - 4 dimensions analysis
python Decatalogo_principal.py  # Principal evaluation system with advanced mathematical innovations

# Demo modes
python example_usage.py  # Embedding model demo
python responsibility_detector.py  # Responsibility detection demo
```

### Single Test Execution
```bash
# Run a single test file
python3 -m pytest test_embedding_model.py::TestSotaEmbedding::test_initialization_creates_calibration_card -v

# Run specific test method patterns
python3 -m pytest -k "test_detect_government" -v
```

## Architecture Overview

### Core Analysis Engine - Decatalogo Industrial System
The system is built around the **Decatalogo Industrial Evaluation System**, the primary orchestrator for comprehensive policy analysis:

- **IndustrialDecatalogoEvaluatorFull** (`Decatalogo_evaluador.py`): **MAIN EVALUATOR** - Industrial-grade evaluator implementing the complete Human Rights Decalogue assessment across 4 dimensions (DE-1 to DE-4) with rigorous scoring methodology for 117 development plans
- **SistemaEvaluacionIndustrial** (`Decatalogo_principal.py`): **CORE SYSTEM** - Principal evaluation system with advanced mathematical innovations, causal graph analysis, fuzzy logic aggregation, and frontier AI capabilities
- **ContradictionDetector** (`pdm_contra/core.py`): Hybrid heuristic engine combining lexical pattern matching, Spanish NLI, competence validation, and risk scoring
- **PDMEvaluator** (`pdm_evaluator.py`): Implements the 30-question evaluation framework across 6 dimensions for Municipal Development Plans
- **ResponsibilityDetector** (`responsibility_detector.py`): Uses spaCy NER + lexical patterns to detect institutional responsibilities with hierarchical confidence scoring

### Multi-Component Analysis Suite
The system integrates multiple analysis capabilities:

- **Embedding Models** (`embedding_model.py`): Resilient multilingual embeddings with automatic fallback (multilingual-e5-base → all-MiniLM-L6-v2)
- **Causal Validation** (`dag_validation.py`): Monte Carlo deterministic sampling for causal graph validation
- **Feasibility Scoring** (`factibilidad/`): Pattern detection for baseline indicators, goals, and timelines with proximity analysis
- **Theory of Change** (`teoria_cambio.py`): Cached causal graph construction with invalidation mechanisms

### Knowledge Base System
The system operates on standardized evaluation frameworks:

- **Decalogo Provider** (`pdm_contra/bridges/decatalogo_provider.py`): Loads normalized templates from YAML configuration
- **Industrial Templates**: Three primary catalogs (DECALOGO_FULL.json, decalogo_industrial.json, DNP_STANDARDS.json) with clean versions in `out/`
- **Schema Validation**: JSON schemas in `schemas/` ensure integrity before bundle exposure

### Processing Pipeline
```
Text Input → Document Segmentation → Multi-Modal Analysis → Risk Assessment → Explanation Generation → Report Output
     ↓              ↓                      ↓                    ↓                  ↓                  ↓
Document      Section Detection     Pattern Matching      Risk Scoring     Traceable       Industrial
Parsing     + Metadata Extraction  + NLI + Responsibility  + Confidence    Explanations     Reports
```

## Key Configuration Files

- `config/embedding.yaml`: Embedding model configuration (BAAI/bge-m3, batch_size, device settings)
- `pyproject.toml`: Development tools configuration (Black, isort, flake8, ruff, mypy)
- `requirements.txt`: Python dependencies with version pinning
- `pdm_contra/config/decalogo.yaml`: Decalogo bundle configuration

## Development Patterns

### Component Integration
- All major components support graceful degradation (fallback modes when dependencies unavailable)
- Atomic file operations with temporary file + rename pattern for deployment safety
- Comprehensive error handling without SystemExit calls
- Cache invalidation mechanisms for expensive operations (theory of change graph construction)

### Testing Strategy
- Deterministic stubs for external dependencies (SentenceTransformer, spaCy models)
- Mock-based testing for NLP components to ensure reproducibility
- Monte Carlo validation with fixed seeds for statistical reproducibility
- Comprehensive edge case coverage (empty inputs, malformed data, network failures)

### Code Quality Standards
- PEP 8 compliance with 88-character line length
- Type hints throughout using typing module
- Comprehensive docstrings with usage examples
- Pre-commit hooks for automated code quality (black, isort, flake8, ruff)

### Performance Considerations
- Device auto-detection (CUDA, MPS, CPU) with manual override support
- Batch processing optimization with configurable batch sizes
- Memory watchdog for large document processing
- Parallel processing support with configurable worker counts

## Specialized Analysis Components

### Contradiction Detection Engine
Combines four detection methods in hybrid approach:
1. **Lexical Pattern Matching**: Spanish adversative connectors ("sin embargo", "no obstante")
2. **NLI Inference**: Spanish natural language inference for semantic contradictions
3. **Competence Validation**: Cross-reference against institutional competence matrices
4. **Risk Assessment**: Statistical confidence intervals with conformal prediction

### Evaluation Framework
Six-dimensional assessment structure:
1. **D1**: Diagnóstico y Recursos (Diagnosis & Resources)
2. **D2**: Diseño de Intervención (Intervention Design) 
3. **D3**: Eje Transversal (Cross-cutting Themes)
4. **D4**: Implementación (Implementation)
5. **D5**: Resultados (Results)
6. **D6**: Sostenibilidad (Sustainability)

### Evidence Processing
- **Feasibility Patterns**: Baseline detection, goal identification, timeline extraction
- **Responsibility Assignment**: Government entities > official positions > generic institutions
- **Monetary Detection**: Budget allocation patterns and financial sustainability indicators
- **Causal Chain Validation**: Theory of change verification with DAG analysis

## Data Flow Architecture

```
Input Documents (PDF/TXT)
    ↓
Document Segmentation (section detection, metadata extraction)
    ↓
Parallel Analysis Pipeline:
├── Contradiction Detection (lexical + NLI + competence)
├── Responsibility Detection (spaCy NER + patterns)  
├── Feasibility Scoring (proximity patterns)
├── Causal Validation (DAG structure)
└── Embedding Analysis (semantic similarity)
    ↓
Risk Assessment & Confidence Calculation
    ↓
Explanation Generation (traceable narratives)
    ↓
Industrial Report Output (JSON + summaries)
```

## Device and Performance Configuration

The system supports multiple computation backends:
- **CPU**: Default fallback, reliable for all operations
- **CUDA**: GPU acceleration for embedding models (when available)
- **MPS**: Apple Silicon acceleration (when available)
- **Auto-detection**: Intelligent device selection based on hardware availability

Precision options: `float16` (memory efficient), `float32` (balanced), `float64` (high precision)

## Error Handling Philosophy

All components implement graceful degradation:
- Missing spaCy models → degraded mode with pattern-only detection
- Network failures → local fallback models
- GPU unavailable → CPU computation
- Missing dependencies → reduced functionality with clear status reporting

The system never fails completely; it adapts capability based on available resources while maintaining core functionality.