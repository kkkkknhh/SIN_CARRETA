# Contract Stack Limitations and Future Improvements

## Executive Summary

This document explains the **inherent limitations** of the EvidencePacket contract stack and why **absolute 100% guarantees are impossible** in any software system. It also outlines what additional steps (formal verification, external audits, TLA+) would provide stronger assurances.

## Current Guarantees

The contract stack provides:

✅ **Strong guarantees:**
- Deterministic serialization (canonical JSON with sorted keys)
- Immutability enforcement at runtime (Pydantic frozen models)
- HMAC-SHA256 cryptographic signing
- Append-only registry with chained hashes
- Extra field rejection (Pydantic `extra="forbid"`)
- Type safety (Pydantic + Protobuf validation)
- CI pipeline with automated contract testing

✅ **Good practices:**
- Semantic versioning
- Breaking change detection (Buf)
- Policy enforcement (OPA)
- Contract tests (producer/consumer)
- Comprehensive test coverage

## Fundamental Limitations

### 1. Runtime Validation Cannot Prevent All Logic Errors

**Limitation:** Pydantic validates structure and types at runtime, but cannot prove absence of logic bugs.

**Example:**
```python
# Valid according to contract, but semantically wrong
packet = create_evidence_packet(
    stage=PipelineStage.FEASIBILITY,
    source_component='feasibility_scorer',
    evidence_type='baseline_presence',
    content={'finding': 'NO_BASELINE'},  # Contradictory!
    confidence=0.9,  # High confidence in absence?
    applicable_questions=['D1-Q1'],
)
```

The packet passes all validation but the logic is flawed. **No runtime type system can prevent this.**

**Mitigation:**
- Add semantic validation rules in OPA policies
- Use property-based testing (Hypothesis)
- Add domain-specific validators

**What would help:**
- Dependent types (Idris, Agda) to encode semantic constraints
- Formal specification of business rules
- Runtime monitoring and anomaly detection

### 2. Deterministic Serialization Does Not Guarantee Deterministic Computation

**Limitation:** Canonical JSON ensures consistent serialization, but upstream computation may be non-deterministic.

**Example:**
```python
import time

# Non-deterministic timestamp
packet = create_evidence_packet(
    ...,
    metadata={'timestamp': time.time()},  # Different every time!
)
```

The serialization is deterministic, but the input varies. **Determinism requires discipline at all levels.**

**Mitigation:**
- Use fixed seeds for random operations
- Avoid timestamps in content (use explicit timestamp field)
- Document non-deterministic sources

**What would help:**
- Pure functional programming (Haskell)
- Explicit effect tracking (IO monad)
- Deterministic runtime (Deno)

### 3. HMAC Signing Does Not Prevent Key Compromise

**Limitation:** HMAC-SHA256 is cryptographically secure, but if the secret key leaks, all bets are off.

**Threat model:**
- Attacker gains access to `EVIDENCE_HMAC_SECRET` environment variable
- Attacker can forge valid signatures
- No detection mechanism

**Mitigation:**
- Rotate keys regularly
- Use hardware security modules (HSM)
- Implement key management best practices
- Add cosign/sigstore attestation (see below)

**What would help:**
- Public key cryptography (RSA, Ed25519)
- Hardware-backed keys (TPM, Yubikey)
- Transparency logs (Certificate Transparency)

### 4. Append-Only Registry Is Vulnerable to File System Attacks

**Limitation:** The registry detects tampering but cannot prevent it if attacker has file system access.

**Threat model:**
- Attacker modifies `evidence_registry.json` file
- Verification fails, but data is corrupted
- No rollback mechanism

**Mitigation:**
- Use read-only file systems
- Store registry in immutable storage (WORM drives)
- Replicate registry to multiple locations
- Add write-ahead log (WAL)

**What would help:**
- Blockchain or distributed ledger (Hyperledger, Ethereum)
- Content-addressed storage (IPFS)
- Byzantine fault tolerance (BFT consensus)

### 5. OPA Policies Are Not Formally Verified

**Limitation:** OPA policies are written in Rego, which is Turing-complete. Logic errors are possible.

**Example:**
```rego
# Bug: Should be >= but wrote >
has_valid_confidence {
    input.confidence > 0.2  # Rejects 0.2 exactly!
}
```

The policy compiles but has a bug. **No static analysis can catch all policy bugs.**

**Mitigation:**
- Write property-based tests for policies
- Use OPA test framework
- Code review policies carefully

**What would help:**
- Formal verification of Rego policies (using SMT solvers)
- Policy synthesis from specifications
- Bounded model checking

### 6. Protobuf/Pydantic Type Systems Are Not Sound

**Limitation:** Python's type system is gradual and can be circumvented. Protobuf has no runtime enforcement in Python.

**Example:**
```python
packet: EvidencePacketModel = ...
# Type checker says this is safe, but it's not!
packet.__dict__['confidence'] = 'invalid'  # Bypasses frozen check
```

**Mitigation:**
- Don't use `__dict__` or other escape hatches
- Add runtime assertions
- Use stricter type checkers (Pyright in strict mode)

**What would help:**
- Sound type systems (OCaml, Rust)
- Proof-carrying code
- Runtime contract verification

### 7. CI Tests Can Have Blind Spots

**Limitation:** Test coverage measures lines executed, not correctness. Missing test cases = missing guarantees.

**Example:**
- Tests cover signature verification
- But don't test signature replay attacks
- Attacker reuses old valid signature

**Mitigation:**
- Add security-focused test cases
- Use mutation testing to find weak tests
- Perform penetration testing

**What would help:**
- Formal proof of test coverage (coverage criteria)
- Exhaustive testing (limited to small inputs)
- Model-based testing (generate tests from models)

## What Would Stronger Guarantees Require?

### Level 1: Property-Based Testing (Moderate Improvement)

**Tools:** Hypothesis (Python), QuickCheck (Haskell)

**What it provides:**
- Automated generation of test inputs
- Shrinking to minimal failing cases
- Properties as specifications

**Example:**
```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.0, max_value=1.0))
def test_confidence_bounds(confidence):
    packet = create_evidence_packet(..., confidence=confidence)
    assert 0.0 <= packet.confidence <= 1.0
```

**Cost:** Low (weeks to add)

### Level 2: Formal Specification (Significant Improvement)

**Tools:** TLA+ (Temporal Logic of Actions), Alloy, Z notation

**What it provides:**
- Mathematical model of system behavior
- Model checking for invariant violations
- Proof of safety properties

**Example TLA+ spec:**
```tla
EXTENDS Naturals, Sequences

VARIABLE registry

Init == registry = <<>>

AppendEvidence(evidence) ==
    registry' = Append(registry, evidence)

Invariant ==
    \A i \in DOMAIN registry :
        registry[i].prev_hash = 
            IF i = 1 THEN genesis_hash 
            ELSE registry[i-1].entry_hash
```

**Cost:** High (months of expert time)

### Level 3: Formal Verification (Strong Improvement)

**Tools:** Coq, Isabelle/HOL, Lean, F*

**What it provides:**
- Machine-checked proofs of correctness
- Verified extraction to executable code
- Proof of functional correctness

**Example (pseudocode):**
```coq
Theorem signature_unique : forall packet1 packet2 secret,
  packet1 <> packet2 ->
  compute_signature packet1 secret <> compute_signature packet2 secret.
Proof.
  (* Proof by properties of HMAC-SHA256 *)
Qed.
```

**Cost:** Very high (person-years of expert time)

### Level 4: External Security Audit (Independent Validation)

**What it provides:**
- Independent review by security experts
- Penetration testing
- Threat modeling
- Compliance verification

**Deliverables:**
- Vulnerability report
- Remediation recommendations
- Security certification

**Cost:** High ($20k-$100k+ depending on scope)

### Level 5: Hardware-Backed Security (Infrastructure)

**Technologies:**
- Hardware Security Modules (HSM)
- Trusted Platform Modules (TPM)
- Secure Enclaves (Intel SGX, ARM TrustZone)

**What it provides:**
- Key material never leaves hardware
- Attestation of code integrity
- Side-channel resistance

**Cost:** High (hardware + integration)

## Recommended Incremental Improvements

### Short Term (1-3 months)

1. **Add Hypothesis property tests** for all validation logic
2. **Implement cosign/sigstore attestation** (sketch provided below)
3. **Add mutation testing** to verify test quality
4. **Perform threat modeling** session
5. **Add security linters** (Bandit, Safety)

### Medium Term (3-6 months)

1. **Write TLA+ spec** for append-only registry
2. **Implement replay attack prevention** (nonces, timestamps)
3. **Add distributed registry replication**
4. **Implement key rotation mechanism**
5. **Add monitoring and alerting** for anomalies

### Long Term (6-12 months)

1. **Formal verification** of core algorithms
2. **External security audit**
3. **Migrate to HSM** for key storage
4. **Implement blockchain backend** option
5. **Add Byzantine consensus** for distributed deployment

## Cosign/Sigstore Attestation Sketch

```python
# python_package/contract/cosign_attestation.py
"""
Optional cosign/sigstore attestation for EvidencePacket.

Requires: cosign CLI (https://docs.sigstore.dev/cosign/installation/)
"""

import subprocess
import json
from pathlib import Path

def attest_evidence_packet(packet: EvidencePacketModel, key_path: str) -> str:
    """
    Create a cosign attestation for an evidence packet.
    
    Args:
        packet: Evidence packet to attest
        key_path: Path to cosign private key
        
    Returns:
        Attestation signature
    """
    # Write packet to temp file
    temp_file = Path('/tmp/evidence_packet.json')
    temp_file.write_text(packet.model_dump_json())
    
    # Run cosign attest
    result = subprocess.run([
        'cosign', 'attest-blob',
        '--key', key_path,
        '--type', 'application/vnd.miniminimoon.evidence+json',
        temp_file
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Cosign attestation failed: {result.stderr}")
    
    return result.stdout

def verify_attestation(packet: EvidencePacketModel, signature: str, 
                       public_key_path: str) -> bool:
    """
    Verify a cosign attestation.
    
    Args:
        packet: Evidence packet
        signature: Attestation signature
        public_key_path: Path to cosign public key
        
    Returns:
        True if valid
    """
    # Implementation left as exercise
    # See: https://docs.sigstore.dev/cosign/verify/
    pass
```

## Conclusion

The current contract stack provides **strong practical guarantees** for a production system, but **absolute guarantees are impossible** due to:

1. Turing-complete languages have undecidable properties
2. Physical systems (hardware, network) can fail
3. Human operators can make mistakes
4. Adversaries can compromise secrets

The cost-benefit tradeoff heavily depends on:
- **Threat model:** Who are the attackers?
- **Risk tolerance:** What are consequences of failure?
- **Budget:** How much can we invest?

For most use cases, the **current contract stack is sufficient**. For high-security applications (financial, healthcare, defense), invest in formal methods and external audits.

## References

- Lamport, L. (2002). "Specifying Systems: The TLA+ Language"
- Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel"
- Sigstore Documentation: https://docs.sigstore.dev/
- NIST Cryptographic Standards: https://csrc.nist.gov/
- Schneier, B. (2015). "Applied Cryptography"

---

**Remember:** Security is a process, not a product. Stay vigilant, keep improving, and always question your assumptions.
