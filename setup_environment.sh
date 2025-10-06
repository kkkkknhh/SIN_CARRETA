#!/bin/bash
# MINIMINIMOON v2.0 - Environment Setup Script
# Updated: 2025-10-05
# Usage: bash setup_environment.sh

set -e  # Exit on error

echo "=========================================="
echo "MINIMINIMOON v2.0 - Environment Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Verificando versión de Python...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}✗ Error: Se requiere Python 3.11 o superior${NC}"
    echo -e "${RED}  Versión actual: $PYTHON_VERSION${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Python $PYTHON_VERSION detectado${NC}"
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment 'venv' ya existe${NC}"
    read -p "¿Deseas recrearlo? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Eliminando entorno existente...${NC}"
        rm -rf venv
    else
        echo -e "${BLUE}Usando entorno existente${NC}"
        source venv/bin/activate
        SKIP_VENV=true
    fi
fi

# Create virtual environment
if [ -z "$SKIP_VENV" ]; then
    echo ""
    echo -e "${BLUE}Creando entorno virtual...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Entorno virtual creado${NC}"

    # Activate virtual environment
    source venv/bin/activate
    echo -e "${GREEN}✓ Entorno virtual activado${NC}"
fi

# Upgrade pip
echo ""
echo -e "${BLUE}Actualizando pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip actualizado${NC}"

# Install requirements
echo ""
echo -e "${BLUE}Instalando dependencias base...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencias base instaladas${NC}"

# Ask about development dependencies
echo ""
read -p "¿Deseas instalar dependencias de desarrollo? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Instalando dependencias de desarrollo...${NC}"
    pip install -r requirements-dev.txt
    echo -e "${GREEN}✓ Dependencias de desarrollo instaladas${NC}"
fi

# Download Spacy models
echo ""
echo -e "${BLUE}Descargando modelos de Spacy...${NC}"
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md
echo -e "${GREEN}✓ Modelos de Spacy descargados${NC}"

# Download NLTK data
echo ""
echo -e "${BLUE}Descargando datos de NLTK...${NC}"
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
print('✓ Datos de NLTK descargados')
"

# Create necessary directories
echo ""
echo -e "${BLUE}Creando directorios necesarios...${NC}"
mkdir -p artifacts
mkdir -p config
mkdir -p logs
mkdir -p output
mkdir -p data
echo -e "${GREEN}✓ Directorios creados${NC}"

# Check for configuration files
echo ""
echo -e "${BLUE}Verificando archivos de configuración...${NC}"
CONFIG_FILES=("DECALOGO_FULL.json" "decalogo_industrial.json" "DNP_STANDARDS.json" "RUBRIC_SCORING.json")
MISSING_CONFIGS=()

for config in "${CONFIG_FILES[@]}"; do
    if [ ! -f "$config" ] && [ ! -f "config/$config" ]; then
        MISSING_CONFIGS+=("$config")
    fi
done

if [ ${#MISSING_CONFIGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Archivos de configuración faltantes:${NC}"
    for config in "${MISSING_CONFIGS[@]}"; do
        echo -e "${YELLOW}  - $config${NC}"
    done
    echo -e "${YELLOW}  Deberás agregarlos antes de ejecutar el sistema${NC}"
else
    echo -e "${GREEN}✓ Todos los archivos de configuración presentes${NC}"
fi

# Run basic tests
echo ""
read -p "¿Deseas ejecutar tests básicos de verificación? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Ejecutando tests básicos...${NC}"
    python3 test_critical_flows.py || echo -e "${YELLOW}⚠ Algunos tests fallaron (esperado si faltan configuraciones)${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}${BOLD}Setup completado exitosamente!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Para activar el entorno en el futuro:${NC}"
echo "  source venv/bin/activate"
echo ""
echo -e "${BLUE}Para ejecutar el sistema:${NC}"
echo "  1. Congelar configuración:"
echo "     python miniminimoon_orchestrator.py freeze ./config/"
echo ""
echo "  2. Ejecutar evaluación:"
echo "     python miniminimoon_orchestrator.py evaluate ./config/ plan.pdf ./output/"
echo ""
echo "  3. Verificar reproducibilidad:"
echo "     python miniminimoon_orchestrator.py verify ./config/ plan.pdf --runs 3"
echo ""
echo -e "${BLUE}Documentación completa en:${NC}"
echo "  - FLUJOS_CRITICOS_GARANTIZADOS.md"
echo "  - ARCHITECTURE.md"
echo ""

