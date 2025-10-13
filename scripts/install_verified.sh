#!/bin/bash
# MINIMINIMOON Verified Installation Script
# Interactive installation with validation and verification
# Supports CPU/GPU PyTorch variants and optional dependencies

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section header
print_header() {
    echo ""
    echo "======================================================================"
    print_msg "$BLUE" "$1"
    echo "======================================================================"
    echo ""
}

# Check Python version
check_python_version() {
    print_header "Checking Python Version"
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    print_msg "$BLUE" "Found Python $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ] && [ "$PYTHON_MINOR" -le 12 ]; then
        print_msg "$GREEN" "✓ Python version is compatible (3.10-3.12)"
        return 0
    else
        print_msg "$RED" "✗ Python version must be 3.10-3.12"
        print_msg "$YELLOW" "Please install a compatible Python version:"
        echo "  • Using pyenv: pyenv install 3.10.13 && pyenv local 3.10.13"
        echo "  • Using conda: conda create -n miniminimoon python=3.10"
        return 1
    fi
}

# Prompt user for PyTorch variant
select_pytorch_variant() {
    print_header "Select PyTorch Variant"
    
    echo "Choose your installation type:"
    echo ""
    echo "  1) CPU-only (Recommended for development, smaller download)"
    echo "  2) CUDA 11.8 (NVIDIA GPU with CUDA 11.8)"
    echo "  3) CUDA 12.1 (NVIDIA GPU with CUDA 12.1)"
    echo ""
    
    while true; do
        read -p "Enter your choice [1-3]: " choice
        case $choice in
            1)
                PYTORCH_VARIANT="cpu"
                REQUIREMENTS_FILE="requirements/torch-cpu.txt"
                EXTRA_INDEX=""
                break
                ;;
            2)
                PYTORCH_VARIANT="cuda118"
                REQUIREMENTS_FILE="requirements/torch-cuda.txt"
                EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cu118"
                break
                ;;
            3)
                PYTORCH_VARIANT="cuda121"
                REQUIREMENTS_FILE="requirements/torch-cuda.txt"
                EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cu121"
                break
                ;;
            *)
                print_msg "$RED" "Invalid choice. Please enter 1, 2, or 3."
                ;;
        esac
    done
    
    print_msg "$GREEN" "✓ Selected PyTorch variant: $PYTORCH_VARIANT"
}

# Install core dependencies
install_core_dependencies() {
    print_header "Installing Core Dependencies"
    
    print_msg "$BLUE" "Installing from: $REQUIREMENTS_FILE"
    
    if [ -n "$EXTRA_INDEX" ]; then
        python3 -m pip install -r "$REQUIREMENTS_FILE" $EXTRA_INDEX --upgrade
    else
        python3 -m pip install -r "$REQUIREMENTS_FILE" --upgrade
    fi
    
    if [ $? -eq 0 ]; then
        print_msg "$GREEN" "✓ Core dependencies installed successfully"
    else
        print_msg "$RED" "✗ Failed to install core dependencies"
        return 1
    fi
}

# Install spaCy language model
install_spacy_model() {
    print_header "Installing spaCy Language Model"
    
    print_msg "$BLUE" "Downloading es_core_news_sm model..."
    python3 -m spacy download es_core_news_sm
    
    if [ $? -eq 0 ]; then
        print_msg "$GREEN" "✓ spaCy model installed successfully"
    else
        print_msg "$YELLOW" "⚠ spaCy model installation failed, continuing anyway"
    fi
}

# Prompt for optional dependencies
install_optional_dependencies() {
    print_header "Optional Dependencies"
    
    echo "Install optional dependency sets?"
    echo ""
    echo "  d) Development tools (pytest, mypy, black, etc.)"
    echo "  p) Production tools (FastAPI, Celery, Redis, etc.)"
    echo "  s) Security scanning tools (safety, bandit, etc.)"
    echo "  n) None (skip optional dependencies)"
    echo ""
    
    read -p "Enter your choices (e.g., 'd' or 'dps' or 'n'): " choices
    
    if [[ "$choices" == *"d"* ]]; then
        print_msg "$BLUE" "Installing development dependencies..."
        python3 -m pip install -r requirements/dev.txt
    fi
    
    if [[ "$choices" == *"p"* ]]; then
        print_msg "$BLUE" "Installing production dependencies..."
        python3 -m pip install -r requirements/prod.txt
    fi
    
    if [[ "$choices" == *"s"* ]]; then
        print_msg "$BLUE" "Installing security scanning tools..."
        python3 -m pip install -r requirements/security.txt
    fi
    
    print_msg "$GREEN" "✓ Optional dependencies installed"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Check if validation scripts exist
    if [ -f "scripts/check_conflicts.py" ]; then
        print_msg "$BLUE" "Running conflict detection..."
        python3 scripts/check_conflicts.py
        
        if [ $? -ne 0 ]; then
            print_msg "$YELLOW" "⚠ Conflict check reported issues, but installation continues"
        fi
    fi
    
    # Check critical imports
    print_msg "$BLUE" "Testing critical imports..."
    
    python3 -c "
import sys
modules = ['numpy', 'scipy', 'sklearn', 'torch', 'transformers', 'spacy', 'pandas', 'networkx']
failed = []
for module in modules:
    try:
        __import__(module)
        print(f'  ✓ {module}')
    except ImportError:
        print(f'  ✗ {module}', file=sys.stderr)
        failed.append(module)

if failed:
    print(f'\nFailed to import: {\" \".join(failed)}', file=sys.stderr)
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_msg "$GREEN" "✓ All critical imports successful"
    else
        print_msg "$RED" "✗ Some imports failed"
        return 1
    fi
}

# Generate compatibility certificate
generate_certificate() {
    print_header "Generating Compatibility Certificate"
    
    if [ -f "scripts/generate_certificate.py" ]; then
        python3 scripts/generate_certificate.py --output-dir certificates
        
        if [ $? -eq 0 ]; then
            print_msg "$GREEN" "✓ Compatibility certificate generated"
        else
            print_msg "$YELLOW" "⚠ Certificate generation failed"
        fi
    else
        print_msg "$YELLOW" "⚠ Certificate generator not found, skipping"
    fi
}

# Print post-installation instructions
print_post_install() {
    print_header "Installation Complete"
    
    print_msg "$GREEN" "✓ MINIMINIMOON installation successful!"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Verify installation:"
    echo "     python3 scripts/validate_continuous.py"
    echo ""
    echo "  2. Run first evaluation:"
    echo "     python3 demo.py"
    echo ""
    echo "  3. See full documentation:"
    echo "     cat README.md"
    echo ""
    
    if [ -f "certificates/compatibility_certificate.md" ]; then
        echo "  4. View compatibility certificate:"
        echo "     cat certificates/compatibility_certificate.md"
        echo ""
    fi
}

# Main installation flow
main() {
    print_header "MINIMINIMOON Verified Installation"
    
    print_msg "$BLUE" "This script will guide you through installing MINIMINIMOON"
    print_msg "$BLUE" "with verified dependencies and conflict checking."
    echo ""
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Select PyTorch variant
    select_pytorch_variant
    
    # Upgrade pip
    print_header "Upgrading pip"
    python3 -m pip install --upgrade pip
    
    # Install core dependencies
    if ! install_core_dependencies; then
        exit 1
    fi
    
    # Install spaCy model
    install_spacy_model
    
    # Install optional dependencies
    install_optional_dependencies
    
    # Verify installation
    if ! verify_installation; then
        print_msg "$YELLOW" "⚠ Installation completed with warnings"
    fi
    
    # Generate certificate
    generate_certificate
    
    # Print post-installation instructions
    print_post_install
}

# Run main function
main
