#!/bin/bash
set -e

echo "======================================================================"
echo "COMPLETE EVALUATION PIPELINE EXECUTION"
echo "======================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Execute pipeline
echo "→ Starting unified evaluation pipeline..."
python3 unified_evaluation_pipeline.py "ITUANGO - PLAN DE DESARROLLO.pdf" "Ituango" "Antioquia"

echo ""
echo "======================================================================"
echo "PIPELINE EXECUTION COMPLETED"
echo "======================================================================"
