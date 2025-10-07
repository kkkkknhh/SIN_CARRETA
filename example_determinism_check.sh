#!/bin/bash
# Example script demonstrating determinism_verifier.py usage

set -e

echo "=========================================="
echo "Determinism Verification Example"
echo "=========================================="
echo ""

# Check if input PDF is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_pdf_path>"
    echo ""
    echo "Example:"
    echo "  $0 test_fixtures/sample_plan.pdf"
    exit 1
fi

INPUT_PDF="$1"

# Verify input PDF exists
if [ ! -f "$INPUT_PDF" ]; then
    echo "Error: Input PDF not found: $INPUT_PDF"
    exit 1
fi

echo "Input PDF: $INPUT_PDF"
echo ""

# Run determinism verifier
echo "Running determinism verification..."
echo "This will execute the orchestrator twice and compare outputs."
echo ""

python3 determinism_verifier.py "$INPUT_PDF"

EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ RESULT: Perfect reproducibility"
    echo "All artifacts match between runs."
    echo "Exit code: 0"
elif [ $EXIT_CODE -eq 4 ]; then
    echo "✗ RESULT: Determinism violations detected"
    echo "Review determinism_report.txt for details."
    echo "Exit code: 4"
else
    echo "✗ RESULT: Execution errors"
    echo "Orchestrator failed or produced incomplete artifacts."
    echo "Exit code: 1"
fi

echo "=========================================="

# Show report location
LATEST_REPORT=$(ls -td artifacts/determinism_run_* 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    echo ""
    echo "Reports available at:"
    echo "  $LATEST_REPORT/determinism_report.json"
    echo "  $LATEST_REPORT/determinism_report.txt"
    echo ""
    echo "Run outputs preserved at:"
    echo "  $LATEST_REPORT/run1/"
    echo "  $LATEST_REPORT/run2/"
fi

exit $EXIT_CODE
