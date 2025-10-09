#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Canonical MINIMINIMOON evaluation pipeline launcher
# -----------------------------------------------------------------------------
# This script orchestrates the full deterministic pipeline using the
# CanonicalDeterministicOrchestrator v2.2.0 (Ultimate Edition).
# It validates prerequisites, activates the virtual environment when available,
# and exports artifacts to the canonical `artifacts/` directory.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

printf '%s\n' "======================================================================"
printf '%s\n' "CANONICAL MINIMINIMOON PIPELINE EXECUTION"
printf '%s\n' "======================================================================"
printf '\n'

declare -r DEFAULT_PLAN="data/florencia_plan_texto.txt"
declare -r DEFAULT_CONFIG_DIR="config"
declare -r DEFAULT_OUTPUT_DIR="artifacts"
declare -r FLOW_DOC_PATH="tools/flow_doc.json"

PLAN_PATH="${1:-$DEFAULT_PLAN}"
CONFIG_DIR="${2:-$DEFAULT_CONFIG_DIR}"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

printf 'Repository: %s\n' "$SCRIPT_DIR"
printf 'Python: %s\n' "$(which python3)"
printf 'Python Version: %s\n' "$(python3 --version)"
printf 'Timestamp: %s\n' "$(date)"

# Display orchestrator version
ORCHESTRATOR_VERSION="$(python3 - <<'PY'
from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator
print(CanonicalDeterministicOrchestrator.VERSION)
PY
)"
printf 'Orchestrator Version: %s\n' "$ORCHESTRATOR_VERSION"

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo "→ Activating virtual environment..."
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    echo
fi

# Validate required files
MISSING_FILES=()
if [ ! -f "$PLAN_PATH" ]; then
    MISSING_FILES+=("$PLAN_PATH")
fi
if [ ! -f "miniminimoon_orchestrator.py" ]; then
    MISSING_FILES+=("miniminimoon_orchestrator.py")
fi
if [ ! -f "$FLOW_DOC_PATH" ]; then
    MISSING_FILES+=("$FLOW_DOC_PATH")
fi

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "ERROR: Missing required files:"
    for missing in "${MISSING_FILES[@]}"; do
        echo "  ✗ $missing"
    done
    exit 1
fi

# Ensure configuration directory exists
mkdir -p "$CONFIG_DIR"

# Ensure RUBRIC_SCORING.json is present in the configuration directory
if [ ! -f "$CONFIG_DIR/RUBRIC_SCORING.json" ]; then
    if [ -f "RUBRIC_SCORING.json" ]; then
        cp "RUBRIC_SCORING.json" "$CONFIG_DIR/RUBRIC_SCORING.json"
        echo "→ Copied RUBRIC_SCORING.json to $CONFIG_DIR/"
    elif [ -f "rubric_scoring.json" ]; then
        cp "rubric_scoring.json" "$CONFIG_DIR/RUBRIC_SCORING.json"
        echo "→ Copied rubric_scoring.json to $CONFIG_DIR/RUBRIC_SCORING.json"
    else
        echo "ERROR: RUBRIC_SCORING.json not found in repository or $CONFIG_DIR/."
        exit 1
    fi
fi

# Prepare output directory
if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "${OUTPUT_DIR:?}"/*
else
    mkdir -p "$OUTPUT_DIR"
fi

echo
printf '%s\n' "→ Starting canonical orchestrator run..."
printf 'Plan: %s\n' "$PLAN_PATH"
printf 'Config Dir: %s\n' "$CONFIG_DIR"
printf 'Output Dir: %s\n' "$OUTPUT_DIR"
printf 'Flow Doc: %s\n' "$FLOW_DOC_PATH"

python3 miniminimoon_orchestrator.py \
    --plan "$PLAN_PATH" \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --flow-doc "$FLOW_DOC_PATH"

EXEC_EXIT=$?

echo
printf '%s\n' "======================================================================"
printf '%s\n' "PIPELINE EXECUTION COMPLETED"
printf '%s\n' "======================================================================"
printf 'Exit Code: %s\n' "$EXEC_EXIT"
printf 'Artifacts generated in %s:\n' "$OUTPUT_DIR"
ls -1 "$OUTPUT_DIR" || true
printf '%s\n' "======================================================================"

echo
exit "$EXEC_EXIT"
