#!/usr/bin/env bash
# generate_proto.sh - Generate Python code from protobuf schema
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROTO_DIR="${REPO_ROOT}/proto"
OUTPUT_DIR="${REPO_ROOT}/python_package/contract"

echo "======================================================================"
echo "Protobuf Code Generation Script"
echo "======================================================================"
echo "Proto dir:  ${PROTO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "ERROR: protoc not found. Please install Protocol Buffers compiler."
    echo ""
    echo "Installation options:"
    echo "  - Ubuntu/Debian: sudo apt-get install -y protobuf-compiler"
    echo "  - macOS: brew install protobuf"
    echo "  - Or download from: https://github.com/protocolbuffers/protobuf/releases"
    echo ""
    exit 1
fi

# Check protoc version
PROTOC_VERSION=$(protoc --version | awk '{print $2}')
echo "Found protoc version: ${PROTOC_VERSION}"

# Check if grpcio-tools is installed (for Python plugin)
if ! python3 -c "import grpc_tools" 2>/dev/null; then
    echo "WARNING: grpcio-tools not found. Attempting to install..."
    pip install grpcio-tools
fi

echo ""
echo "Generating Python code from evidence.proto..."
echo ""

# Generate Python code using protoc
python3 -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUTPUT_DIR}" \
    --pyi_out="${OUTPUT_DIR}" \
    "${PROTO_DIR}/evidence.proto"

if [ $? -eq 0 ]; then
    echo "✓ Python code generated successfully!"
    echo ""
    echo "Generated files:"
    ls -lh "${OUTPUT_DIR}"/evidence_pb2.py* 2>/dev/null || echo "  (files will appear after first run)"
    echo ""
else
    echo "✗ Code generation failed!"
    exit 1
fi

# Create __init__.py if it doesn't exist
if [ ! -f "${OUTPUT_DIR}/__init__.py" ]; then
    cat > "${OUTPUT_DIR}/__init__.py" << 'EOF'
"""
Contract package for EvidencePacket protocol buffer definitions.
"""
from .evidence_pb2 import (
    EvidencePacket,
    EvidencePacketBatch,
    RegistryEntry,
    PipelineStage,
)

__all__ = [
    'EvidencePacket',
    'EvidencePacketBatch',
    'RegistryEntry',
    'PipelineStage',
]
EOF
    echo "✓ Created __init__.py"
fi

echo ""
echo "======================================================================"
echo "Code generation complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Run 'buf lint proto/' to check schema style"
echo "  2. Run 'buf breaking proto/ --against .git#branch=main' to check breaking changes"
echo "  3. Import in Python: from python_package.contract import EvidencePacket"
echo ""
