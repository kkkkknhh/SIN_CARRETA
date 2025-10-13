# EvidencePacket Contract Documentation

## Overview

This directory contains the complete contract stack for `EvidencePacket` - the canonical data structure used throughout the MINIMINIMOON Decatalogo pipeline for evidence transmission and storage.

## Architecture

The contract stack consists of:

1. **Protobuf Schema** (`proto/evidence.proto`) - Wire format definition
2. **Pydantic Models** (`python_package/contract/`) - Runtime validation and serialization
3. **Append-Only Registry** (`python_package/registry/`) - Tamper-evident storage
4. **OPA Policies** (`python_package/policies/opa/`) - Business rule enforcement
5. **Contract Tests** (`tests/test_contract_*.py`) - Producer and consumer validation
6. **CI Pipeline** (`.github/workflows/contract-ci.yml`) - Automated verification

## Key Features

### 1. Deterministic Serialization

All EvidencePackets use canonical JSON with:
- Sorted keys (alphabetically)
- No whitespace variance (compact separators: `,` and `:`)
- ASCII encoding (Unicode escaped)
- Sorted applicable_questions array

```python
from python_package.contract.factory import create_evidence_packet

packet = create_evidence_packet(
    stage=8,
    source_component='feasibility_scorer',
    evidence_type='baseline_presence',
    content={'finding': 'baseline detected'},
    confidence=0.85,
    applicable_questions=['D1-Q1', 'D1-Q2'],
)

canonical = packet.canonical_json()
# Output: {"applicable_questions":["D1-Q1","D1-Q2"],"confidence":0.85,...}
```

### 2. Immutability

All models are frozen (Pydantic `frozen=True`):

```python
packet = create_evidence_packet(...)
packet.confidence = 0.9  # ❌ Raises ValidationError
```

### 3. Extra Fields Rejection

Unknown fields are rejected (`extra="forbid"`):

```python
EvidencePacketModel(
    ...,
    unknown_field='value'  # ❌ Raises ValidationError
)
```

### 4. HMAC-SHA256 Signing

Packets are signed with HMAC-SHA256 using a secret from environment:

```python
import os
os.environ['EVIDENCE_HMAC_SECRET'] = 'your_secret_key_minimum_32_chars'

packet = create_evidence_packet(...)
# Packet is automatically signed

secret = os.environ['EVIDENCE_HMAC_SECRET']
is_valid = packet.verify_signature(secret)  # True
```

**Security Notes:**
- Secret must be at least 32 characters
- Use cryptographically random strings
- Rotate secrets periodically
- Store in secure environment variables or secret managers

### 5. Append-Only Registry with Chained Hashes

Evidence can be stored in an append-only registry with cryptographic chaining:

```python
from python_package.registry.append_only_registry import AppendOnlyRegistry

registry = AppendOnlyRegistry('evidence_registry.json')

# Append evidence
entry = registry.append(packet)
print(f"Entry hash: {entry.entry_hash}")
print(f"Previous hash: {entry.prev_hash}")

# Verify chain integrity
is_valid, error = registry.verify_chain()
if is_valid:
    print("✓ Registry integrity verified")
else:
    print(f"✗ Chain broken: {error}")
```

**Chain Properties:**
- First entry has `prev_hash = "0000..."`
- Each entry's hash depends on previous entry's hash
- Tampering with any entry breaks the chain
- Similar to blockchain but file-based

### 6. OPA Policy Enforcement

Business rules are enforced via Open Policy Agent:

```rego
# Reject unsigned packets
is_signed {
    input.signature != null
    input.signature != ""
}

# Reject low confidence
has_valid_confidence {
    input.confidence >= 0.2
}
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install Buf (for proto linting)
brew install bufbuild/buf/buf  # macOS
# or download from: https://github.com/bufbuild/buf/releases

# Install OPA (for policy checking)
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
chmod +x opa
sudo mv opa /usr/local/bin/
```

### 2. Generate Protobuf Code

```bash
./scripts/generate_proto.sh
```

This generates:
- `python_package/contract/evidence_pb2.py` - Protobuf classes
- `python_package/contract/evidence_pb2.pyi` - Type stubs

### 3. Run Buf Checks

```bash
# Lint protobuf schema
buf lint proto/

# Check for breaking changes (against main branch)
buf breaking proto/ --against '.git#branch=main'

# Format check
buf format -d --exit-code
```

### 4. Run Tests

```bash
# Set HMAC secret for testing
export EVIDENCE_HMAC_SECRET="test_secret_key_minimum_32_characters_long_12345"

# Run contract tests
pytest tests/test_contract_producer.py -v
pytest tests/test_contract_consumer.py -v

# Run with coverage
pytest tests/test_contract_*.py --cov=python_package/contract --cov=python_package/registry
```

### 5. Verify Registry

```bash
# Create test registry
python -c "
from python_package.contract.factory import create_evidence_packet
from python_package.registry.append_only_registry import AppendOnlyRegistry

registry = AppendOnlyRegistry('test_registry.json')
packet = create_evidence_packet(
    stage=1,
    source_component='test',
    evidence_type='test',
    content={'data': 'value'},
    confidence=0.5,
    applicable_questions=['Q1'],
)
registry.append(packet)
"

# Verify integrity
python python_package/registry/append_only_registry.py test_registry.json
```

### 6. Run OPA Policy Check

```bash
# Create test packet
python -c "
import json
from python_package.contract.factory import create_evidence_packet

packet = create_evidence_packet(
    stage=8,
    source_component='feasibility_scorer',
    evidence_type='baseline_presence',
    content={'finding': 'test'},
    confidence=0.8,
    applicable_questions=['Q1'],
)

with open('test_packet.json', 'w') as f:
    json.dump(json.loads(packet.model_dump_json()), f)
"

# Evaluate policy
opa eval -i test_packet.json \
    -d python_package/policies/opa/policy.rego \
    'data.evidence.validation.allow'
```

## CI/CD Pipeline

The CI pipeline (`.github/workflows/contract-ci.yml`) runs:

1. **Buf Lint & Breaking** - Schema validation and breaking change detection
2. **Build & Test** - Generate proto code, run mypy, pytest, flake8
3. **Registry Verification** - Create test registry and verify integrity
4. **OPA Policy Check** - Test policies against valid/invalid packets
5. **Signature Verification** - Test HMAC signing stability

**Environment Variables Required:**
- `EVIDENCE_HMAC_SECRET` - HMAC signing secret (set in CI secrets)

## Usage Examples

### Creating Evidence

```python
from python_package.contract.factory import (
    create_feasibility_evidence,
    create_decalogo_evidence,
)

# Create feasibility evidence
feasibility = create_feasibility_evidence(
    content={
        'finding': 'baseline study found',
        'location': 'section 3.2',
        'evidence_text': 'The plan includes baseline indicators...',
    },
    confidence=0.85,
    applicable_questions=['D1-Q1', 'D1-Q2'],
    metadata={'version': '1.0', 'extractor': 'rule_based'},
)

# Create decalogo evidence
decalogo = create_decalogo_evidence(
    content={
        'criterion': 'D1',
        'score': 8.5,
        'findings': ['baseline present', 'indicators defined'],
    },
    confidence=0.9,
    applicable_questions=['D1-Q1', 'D1-Q2', 'D1-Q3'],
)
```

### Validating Evidence

```python
from python_package.contract.factory import validate_packet
from python_package.contract.evidence_proto_gen import get_hmac_secret

# Validate packet
is_valid = validate_packet(packet)

# Verify signature explicitly
secret = get_hmac_secret()
is_valid = packet.verify_signature(secret)
```

### Building Registry

```python
from python_package.registry.append_only_registry import AppendOnlyRegistry

registry = AppendOnlyRegistry('evidence_registry.json')

# Append multiple packets
for packet in packets:
    entry = registry.append(packet)
    print(f"Appended: {entry.sequence_number}")

# Verify integrity
is_valid, error = registry.verify_chain()
assert is_valid, f"Registry corrupted: {error}"

# Get statistics
stats = registry.get_stats()
print(f"Total entries: {stats['entry_count']}")
print(f"Latest hash: {stats['latest_hash']}")
```

## Contract Compatibility Matrix

See `contracts/compatibility_matrix.yaml` for version compatibility rules.

## Troubleshooting

### Proto Generation Fails

```bash
# Install protoc
sudo apt-get install -y protobuf-compiler

# Install grpcio-tools
pip install grpcio-tools

# Regenerate
./scripts/generate_proto.sh
```

### HMAC Secret Not Set

```bash
export EVIDENCE_HMAC_SECRET="your_secret_minimum_32_chars_long_12345"
```

### Registry Verification Fails

Check for:
1. File tampering
2. Missing entries
3. Incorrect hash computation
4. Signature verification failures

### Buf Not Found

```bash
# Ubuntu/Debian
curl -sSL "https://github.com/bufbuild/buf/releases/download/v1.28.1/buf-Linux-x86_64" -o /usr/local/bin/buf
chmod +x /usr/local/bin/buf

# macOS
brew install bufbuild/buf/buf
```

## References

- [Protobuf Language Guide](https://developers.google.com/protocol-buffers/docs/proto3)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Buf Documentation](https://buf.build/docs/)
- [OPA Documentation](https://www.openpolicyagent.org/docs/latest/)
- [HMAC-SHA256 RFC](https://tools.ietf.org/html/rfc2104)

## Support

For questions or issues:
1. Check test files for examples
2. Review CI logs for error details
3. Consult `NOTE.md` for limitations
4. Check compatibility matrix for version rules
