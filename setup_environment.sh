#!/bin/bash
# MINIMINIMOON v2.0 - Environment Setup Script (macOS/Linux)
# Updated: 2025-10-06
# Usage: ./setup_environment.sh

echo "=========================================="
echo "MINIMINIMOON v2.0 - Environment Setup"
echo "=========================================="
echo

# Check Python version
echo "Verificando versión de Python..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python no encontrado."
    echo "Por favor, instala una versión de Python compatible (se recomienda 3.8, 3.9 o 3.10)."
    exit 1
fi

PY_VERSION=$(python3 --version 2>&1)
echo "Python encontrado: $PY_VERSION"
if [[ "$PY_VERSION" == *"Python 3.11"* || "$PY_VERSION" == *"Python 3.12"* ]]; then
    echo "Advertencia: Tus dependencias parecen requerir una versión de Python anterior a la 3.11. Podrías encontrar errores."
    read -p "¿Continuar de todas formas? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi


# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "El entorno virtual 'venv' ya existe."
    read -p "¿Deseas recrearlo? (y/n): " RECREATE
    if [ "$RECREATE" = "y" ]; then
        echo "Eliminando entorno existente..."
        rm -rf venv
        echo "Creando entorno virtual..."
        python3 -m venv venv
    else
        echo "Usando entorno existente."
    fi
else
    echo "Creando entorno virtual..."
    python3 -m venv venv
    echo "Entorno virtual creado."
fi

# Activate virtual environment
source venv/bin/activate
echo "Entorno virtual activado."

# Upgrade pip
echo
echo "Actualizando pip..."
python -m pip install --upgrade pip setuptools wheel
echo "pip actualizado."

# Install requirements
echo
echo "Instalando dependencias base..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error instalando dependencias. Revisa los mensajes de error de arriba."
    echo "Es posible que necesites usar una versión de Python diferente (e.g., 3.10)."
    exit 1
fi
echo "Dependencias base instaladas."

# Ask about development dependencies
echo
read -p "¿Deseas instalar dependencias de desarrollo? (y/n): " INSTALL_DEV
if [ "$INSTALL_DEV" = "y" ]; then
    echo "Instalando dependencias de desarrollo..."
    pip install -r requirements-dev.txt
    echo "Dependencias de desarrollo instaladas."
fi

# Download Spacy models
echo
echo "Descargando modelos de Spacy..."
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md
echo "Modelos de Spacy descargados."

# Download NLTK data
echo
echo "Descargando datos de NLTK..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Create necessary directories
echo
echo "Creando directorios necesarios..."
mkdir -p artifacts config logs output data
echo "Directorios creados."

# Check for configuration files
echo
echo "Verificando archivos de configuración..."
MISSING=0
if [ ! -f "DECALOGO_FULL.json" ] && [ ! -f "config/DECALOGO_FULL.json" ]; then
    echo "  - DECALOGO_FULL.json FALTANTE"
    MISSING=1
fi
if [ ! -f "decalogo_industrial.json" ] && [ ! -f "config/decalogo_industrial.json" ]; then
    echo "  - decalogo_industrial.json FALTANTE"
    MISSING=1
fi
if [ ! -f "DNP_STANDARDS.json" ] && [ ! -f "config/DNP_STANDARDS.json" ]; then
    echo "  - DNP_STANDARDS.json FALTANTE"
    MISSING=1
fi
if [ ! -f "RUBRIC_SCORING.json" ] && [ ! -f "config/RUBRIC_SCORING.json" ]; then
    echo "  - RUBRIC_SCORING.json FALTANTE"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo "Deberás agregar los archivos faltantes antes de ejecutar el sistema."
else
    echo "Todos los archivos de configuración presentes."
fi

# Run basic tests
echo
read -p "¿Deseas ejecutar tests básicos de verificación? (y/n): " RUN_TESTS
if [ "$RUN_TESTS" = "y" ]; then
    echo "Ejecutando tests básicos..."
    python test_critical_flows.py
fi

# Summary
echo
echo "=========================================="
echo "Setup completado exitosamente!"
echo "=========================================="
echo
echo "Para activar el entorno en el futuro:"
echo "  source venv/bin/activate"
echo
echo "Para ejecutar el sistema:"
echo "  1. Congelar configuración:"
echo "     python miniminimoon_orchestrator.py freeze ./config/"
echo
echo "  2. Ejecutar evaluación:"
echo "     python miniminimoon_orchestrator.py evaluate ./config/ plan.pdf ./output/"
echo
echo "  3. Verificar reproducibilidad:"
echo "     python miniminimoon_orchestrator.py verify ./config/ plan.pdf --runs 3"
echo
echo "Documentación completa en:"
echo "  - FLUJOS_CRITICOS_GARANTIZADOS.md"
echo "  - ARCHITECTURE.md"
echo

