#!/bin/bash
# Quick test execution script
set -e

echo "=========================================="
echo "EXECUTING UNIFIED FLOW CERTIFICATION TEST"
echo "=========================================="
echo ""

# Clean artifacts
echo "→ Cleaning artifacts..."
rm -rf artifacts/*
echo "✓ Artifacts cleaned"
echo ""

# Generate mock artifacts for 3 runs
echo "→ Generating mock artifacts (simulating 3 runs)..."
for i in 1 2 3; do
    echo "  Run $i..."
    python3 test_mock_execution.py > /dev/null 2>&1
done
echo "✓ Mock artifacts generated"
echo ""

# Run the test
echo "→ Running certification test..."
echo ""
python3 quick_test_unified.py
