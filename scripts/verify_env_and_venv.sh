#!/usr/bin/env bash
# verify_env_and_venv.sh - Verify and setup Python virtual environment
# =====================================================================
#
# This script ensures a working Python virtual environment exists with all
# required dependencies. It detects common issues and provides remediation.
#
# Exit codes:
#   0 - Success, environment ready
#   1 - Python version incompatible
#   2 - venv creation failed
#   3 - pip installation failed
#   4 - Requirements installation failed
#   5 - Environment activation failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"
REQUIREMENTS_DEV="${REPO_ROOT}/requirements-dev.txt"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Python Environment Verification and Setup"
echo "======================================================================"
echo ""

# Function to print colored messages
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
    echo "  $1"
}

# Step 1: Check system Python version
echo "[1/6] Checking system Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found in PATH"
    echo ""
    echo "Remediation:"
    echo "  - Ubuntu/Debian: sudo apt-get install -y python3.10 python3.10-venv"
    echo "  - macOS: brew install python@3.10"
    echo "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

print_info "Found Python $PYTHON_VERSION"

# Check minimum Python version (3.10+)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    echo ""
    echo "This project requires Python 3.10 or later due to:"
    echo "  - NumPy compatibility requirements"
    echo "  - Modern type hints (PEP 604)"
    echo "  - Pydantic v2 requirements"
    echo ""
    echo "Remediation:"
    echo "  - Ubuntu/Debian: sudo apt-get install -y python3.10"
    echo "  - macOS: brew install python@3.10"
    echo "  - Then create venv with: python3.10 -m venv .venv"
    exit 1
fi

print_success "Python version check passed"
echo ""

# Step 2: Check if venv exists and is valid
echo "[2/6] Checking virtual environment..."
VENV_NEEDS_CREATION=false

if [ ! -d "$VENV_DIR" ]; then
    print_warning "Virtual environment not found at $VENV_DIR"
    VENV_NEEDS_CREATION=true
elif [ ! -f "$VENV_PYTHON" ]; then
    print_warning "Python executable not found in venv: $VENV_PYTHON"
    VENV_NEEDS_CREATION=true
else
    # Check if venv Python is working
    if ! "$VENV_PYTHON" --version &> /dev/null; then
        print_warning "Virtual environment Python is broken"
        VENV_NEEDS_CREATION=true
    else
        VENV_PYTHON_VERSION=$("$VENV_PYTHON" --version | awk '{print $2}')
        print_info "Found venv with Python $VENV_PYTHON_VERSION"
        print_success "Virtual environment is valid"
    fi
fi

if [ "$VENV_NEEDS_CREATION" = true ]; then
    print_info "Creating new virtual environment..."
    
    # Remove old venv if it exists
    if [ -d "$VENV_DIR" ]; then
        print_info "Removing broken venv..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create new venv
    if ! python3 -m venv "$VENV_DIR"; then
        print_error "Failed to create virtual environment"
        echo ""
        echo "Remediation:"
        echo "  - Ubuntu/Debian: sudo apt-get install -y python3-venv"
        echo "  - Ensure you have write permissions to: $REPO_ROOT"
        echo "  - Try manually: python3 -m venv $VENV_DIR"
        exit 2
    fi
    
    print_success "Virtual environment created"
fi

echo ""

# Step 3: Activate venv and verify
echo "[3/6] Activating virtual environment..."
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    print_error "Activation script not found: ${VENV_DIR}/bin/activate"
    exit 5
fi

# Source activation script
source "${VENV_DIR}/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    print_error "Virtual environment activation failed"
    exit 5
fi

print_success "Virtual environment activated"
print_info "VIRTUAL_ENV=$VIRTUAL_ENV"
echo ""

# Step 4: Upgrade pip
echo "[4/6] Upgrading pip..."
if ! python -m pip install --upgrade pip setuptools wheel; then
    print_error "Failed to upgrade pip"
    echo ""
    echo "Remediation:"
    echo "  - Try: python -m ensurepip --upgrade"
    echo "  - Or: curl https://bootstrap.pypa.io/get-pip.py | python"
    exit 3
fi

PIP_VERSION=$(python -m pip --version | awk '{print $2}')
print_success "pip upgraded to version $PIP_VERSION"
echo ""

# Step 5: Install requirements
echo "[5/6] Installing requirements..."

if [ ! -f "$REQUIREMENTS" ]; then
    print_error "requirements.txt not found: $REQUIREMENTS"
    exit 4
fi

print_info "Installing from requirements.txt..."
if ! python -m pip install -r "$REQUIREMENTS"; then
    print_error "Failed to install requirements.txt"
    echo ""
    echo "Remediation:"
    echo "  - Check $REQUIREMENTS for syntax errors"
    echo "  - Try: python -m pip install -r $REQUIREMENTS --verbose"
    echo "  - Check network connectivity"
    echo "  - Check for conflicting packages"
    exit 4
fi

print_success "Requirements installed successfully"

# Install dev requirements if requested
if [ "${INSTALL_DEV:-false}" = "true" ] && [ -f "$REQUIREMENTS_DEV" ]; then
    print_info "Installing from requirements-dev.txt..."
    if ! python -m pip install -r "$REQUIREMENTS_DEV"; then
        print_warning "Failed to install requirements-dev.txt (non-critical)"
    else
        print_success "Dev requirements installed successfully"
    fi
fi

echo ""

# Step 6: Verify critical packages
echo "[6/6] Verifying critical packages..."

CRITICAL_PACKAGES=(
    "pydantic"
    "protobuf"
    "pytest"
)

ALL_INSTALLED=true
for pkg in "${CRITICAL_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        print_success "$pkg ($VERSION)"
    else
        print_error "$pkg not installed"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = false ]; then
    print_error "Some critical packages are missing"
    exit 4
fi

echo ""
echo "======================================================================"
print_success "Environment setup complete!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  Python version: $(python --version | awk '{print $2}')"
echo "  Virtual env:    $VENV_DIR"
echo "  pip version:    $(python -m pip --version | awk '{print $2}')"
echo ""
echo "To activate the environment manually:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To generate proto code:"
echo "  ./scripts/generate_proto.sh"
echo ""

exit 0
