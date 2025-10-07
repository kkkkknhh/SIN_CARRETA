#!/bin/bash
# -*- coding: utf-8 -*-
#
# Unified Flow Certification Execution Script
# ============================================
# 
# This script executes the comprehensive end-to-end unified flow certification test
# which validates the complete MINIMINIMOON evaluation pipeline.
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "UNIFIED FLOW CERTIFICATION TEST EXECUTION"
echo "================================================================================"
echo ""
echo "Repository: $SCRIPT_DIR"
echo "Python: $(which python3)"
echo "Python Version: $(python3 --version)"
echo "Timestamp: $(date)"
echo ""
echo "================================================================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "→ Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
fi

# Verify prerequisites
echo "→ Verifying prerequisites..."
echo ""

# Check for required files
REQUIRED_FILES=(
    "miniminimoon_cli.py"
    "system_validators.py"
    "unified_evaluation_pipeline.py"
    "tools/flow_doc.json"
    "tools/rubric_check.py"
    "data/florencia_plan_texto.txt"
    "tests/test_unified_flow_certification.py"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
        echo "  ✗ Missing: $file"
    else
        echo "  ✓ Found: $file"
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo ""
    echo "ERROR: Missing required files. Cannot proceed."
    exit 1
fi

echo ""
echo "✓ All prerequisites verified"
echo ""

# Clean artifacts directory
echo "→ Cleaning artifacts directory..."
if [ -d "artifacts" ]; then
    rm -rf artifacts/*
    echo "✓ Artifacts directory cleaned"
else
    mkdir -p artifacts
    echo "✓ Artifacts directory created"
fi
echo ""

# Execute the test
echo "================================================================================"
echo "EXECUTING TEST: test_unified_flow_certification.py"
echo "================================================================================"
echo ""

python3 run_unified_flow_certification.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "TEST EXECUTION COMPLETE"
echo "================================================================================"
echo "Exit Code: $EXIT_CODE"
echo "Timestamp: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 TEST PASSED 🎉"
else
    echo "❌ TEST FAILED ❌"
fi

echo "================================================================================"
echo ""

exit $EXIT_CODE
