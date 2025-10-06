@echo off
REM MINIMINIMOON v2.0 - Environment Setup Script (Windows)
REM Updated: 2025-10-05
REM Usage: setup_environment.bat

echo ==========================================
echo MINIMINIMOON v2.0 - Environment Setup
echo ==========================================
echo.

REM Check Python version
echo Verificando version de Python...
python --version 2>NUL
if errorlevel 1 (
    echo Error: Python no encontrado
    echo Por favor instala Python 3.11 o superior
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist venv (
    echo Virtual environment 'venv' ya existe
    set /p RECREATE="¿Deseas recrearlo? (y/n): "
    if /i "%RECREATE%"=="y" (
        echo Eliminando entorno existente...
        rmdir /s /q venv
    ) else (
        echo Usando entorno existente
        call venv\Scripts\activate.bat
        set SKIP_VENV=true
    )
)

REM Create virtual environment
if not defined SKIP_VENV (
    echo.
    echo Creando entorno virtual...
    python -m venv venv
    echo Entorno virtual creado

    REM Activate virtual environment
    call venv\Scripts\activate.bat
    echo Entorno virtual activado
)

REM Upgrade pip
echo.
echo Actualizando pip...
python -m pip install --upgrade pip setuptools wheel
echo pip actualizado

REM Install requirements
echo.
echo Instalando dependencias base...
pip install -r requirements.txt
echo Dependencias base instaladas

REM Ask about development dependencies
echo.
set /p INSTALL_DEV="¿Deseas instalar dependencias de desarrollo? (y/n): "
if /i "%INSTALL_DEV%"=="y" (
    echo Instalando dependencias de desarrollo...
    pip install -r requirements-dev.txt
    echo Dependencias de desarrollo instaladas
)

REM Download Spacy models
echo.
echo Descargando modelos de Spacy...
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md
echo Modelos de Spacy descargados

REM Download NLTK data
echo.
echo Descargando datos de NLTK...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

REM Create necessary directories
echo.
echo Creando directorios necesarios...
if not exist artifacts mkdir artifacts
if not exist config mkdir config
if not exist logs mkdir logs
if not exist output mkdir output
if not exist data mkdir data
echo Directorios creados

REM Check for configuration files
echo.
echo Verificando archivos de configuracion...
set MISSING=0
if not exist "DECALOGO_FULL.json" if not exist "config\DECALOGO_FULL.json" (
    echo   - DECALOGO_FULL.json FALTANTE
    set MISSING=1
)
if not exist "decalogo_industrial.json" if not exist "config\decalogo_industrial.json" (
    echo   - decalogo_industrial.json FALTANTE
    set MISSING=1
)
if not exist "DNP_STANDARDS.json" if not exist "config\DNP_STANDARDS.json" (
    echo   - DNP_STANDARDS.json FALTANTE
    set MISSING=1
)
if not exist "RUBRIC_SCORING.json" if not exist "config\RUBRIC_SCORING.json" (
    echo   - RUBRIC_SCORING.json FALTANTE
    set MISSING=1
)

if %MISSING%==1 (
    echo Deberas agregar los archivos faltantes antes de ejecutar el sistema
) else (
    echo Todos los archivos de configuracion presentes
)

REM Run basic tests
echo.
set /p RUN_TESTS="¿Deseas ejecutar tests basicos de verificacion? (y/n): "
if /i "%RUN_TESTS%"=="y" (
    echo Ejecutando tests basicos...
    python test_critical_flows.py
)

REM Summary
echo.
echo ==========================================
echo Setup completado exitosamente!
echo ==========================================
echo.
echo Para activar el entorno en el futuro:
echo   venv\Scripts\activate.bat
echo.
echo Para ejecutar el sistema:
echo   1. Congelar configuracion:
echo      python miniminimoon_orchestrator.py freeze .\config\
echo.
echo   2. Ejecutar evaluacion:
echo      python miniminimoon_orchestrator.py evaluate .\config\ plan.pdf .\output\
echo.
echo   3. Verificar reproducibilidad:
echo      python miniminimoon_orchestrator.py verify .\config\ plan.pdf --runs 3
echo.
echo Documentacion completa en:
echo   - FLUJOS_CRITICOS_GARANTIZADOS.md
echo   - ARCHITECTURE.md
echo.
pause
# MINIMINIMOON v2.0 - Development Dependencies
# Updated: 2025-10-05

# Incluir dependencias base
-r requirements.txt

# ============================================================================
# DEVELOPMENT TOOLS
# ============================================================================

# Interactive Development
ipython>=8.0.0
jupyter>=1.0.0
jupyterlab>=3.6.0

# Debugging
ipdb>=0.13.0
pdbpp>=0.10.0

# ============================================================================
# ADDITIONAL TESTING TOOLS
# ============================================================================

# Test coverage and reporting
pytest-html>=3.1.0
pytest-xdist>=3.0.0  # Parallel test execution
pytest-mock>=3.10.0

# Load testing
locust>=2.14.0

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Documentation generation
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0
sphinx-autodoc-typehints>=1.22.0

# Markdown support
myst-parser>=1.0.0

# ============================================================================
# CODE ANALYSIS
# ============================================================================

# Security scanning
bandit>=1.7.0
safety>=2.3.0

# Complexity analysis
radon>=5.1.0
mccabe>=0.7.0

# Import sorting
isort>=5.12.0

# Dead code detection
vulture>=2.7.0

# ============================================================================
# PRE-COMMIT HOOKS
# ============================================================================

pre-commit>=3.0.0

# ============================================================================
# PROFILING & BENCHMARKING
# ============================================================================

# Performance profiling
py-spy>=0.3.0
scalene>=1.5.0

# Benchmarking
pytest-benchmark>=4.0.0

