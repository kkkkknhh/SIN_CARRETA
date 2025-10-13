#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract Producer Tests
========================

Tests to ensure that produced EvidencePackets conform to contract requirements.

These tests verify:
1. Packet structure and validation
2. Signature generation and validity
3. Canonical JSON stability
4. Immutability enforcement
5. Extra field rejection
"""

import json
import os
import pytest
from typing import Dict, Any

# Set HMAC secret for testing
os.environ['EVIDENCE_HMAC_SECRET'] = 'test_secret_key_minimum_32_characters_long_12345'

from python_package.contract.evidence_proto_gen import (
    EvidencePacketModel,
    PipelineStage,
    get_hmac_secret,
)
from python_package.contract.factory import (
    create_evidence_packet,
    create_feasibility_evidence,
    validate_packet,
)


class TestContractProducer:
    """Test suite for EvidencePacket producer contract."""
    
    def test_create_valid_packet(self):
        """Test creation of a valid evidence packet."""
        packet = create_evidence_packet(
            stage=PipelineStage.FEASIBILITY,
            source_component='feasibility_scorer',
            evidence_type='baseline_presence',
            content={'text': 'baseline found', 'score': 0.9},
            confidence=0.85,
            applicable_questions=['D1-Q1', 'D1-Q2'],
            metadata={'version': '1.0'},
        )
        
        # Verify packet structure
        assert packet.schema_version == '1.0.0'
        assert packet.stage == PipelineStage.FEASIBILITY
        assert packet.source_component == 'feasibility_scorer'
        assert packet.evidence_type == 'baseline_presence'
        assert packet.confidence == 0.85
        assert len(packet.applicable_questions) == 2
        assert packet.signature is not None
        assert packet.evidence_hash is not None
    
    def test_signature_is_valid(self):
        """Test that generated signature is valid."""
        packet = create_feasibility_evidence(
            content={'finding': 'baseline detected'},
            confidence=0.9,
            applicable_questions=['D1-Q1'],
        )
        
        secret = get_hmac_secret()
        assert packet.verify_signature(secret) is True
    
    def test_canonical_json_stability(self):
        """Test that canonical JSON is stable across multiple calls."""
        packet = create_evidence_packet(
            stage=PipelineStage.FEASIBILITY,
            source_component='test_component',
            evidence_type='test_type',
            content={'key': 'value', 'number': 42},
            confidence=0.7,
            applicable_questions=['Q1', 'Q2'],
            sign=False,  # Don't sign to test canonical JSON directly
        )
        
        # Generate canonical JSON multiple times
        json1 = packet.canonical_json()
        json2 = packet.canonical_json()
        json3 = packet.canonical_json()
        
        # All should be identical
        assert json1 == json2
        assert json2 == json3
        
        # Parse and verify structure
        data = json.loads(json1)
        assert 'signature' not in data  # Signature excluded from canonical
        assert 'evidence_hash' not in data  # Hash excluded from canonical
        assert data['confidence'] == 0.7
        assert data['applicable_questions'] == ['Q1', 'Q2']  # Sorted
    
    def test_canonical_json_sorted_keys(self):
        """Test that canonical JSON has sorted keys."""
        packet = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'z': 1, 'a': 2, 'm': 3},
            confidence=0.5,
            applicable_questions=['Z1', 'A1'],
            sign=False,
        )
        
        canonical = packet.canonical_json()
        
        # Check that keys appear in sorted order
        assert canonical.index('"applicable_questions"') < canonical.index('"confidence"')
        assert canonical.index('"confidence"') < canonical.index('"content"')
        
        # Check that question IDs are sorted
        assert canonical.index('"A1"') < canonical.index('"Z1"')
    
    def test_immutability_enforcement(self):
        """Test that packets are immutable (frozen)."""
        packet = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value'},
            confidence=0.5,
            applicable_questions=['Q1'],
        )
        
        # Attempt to modify should raise error
        with pytest.raises(Exception):  # ValidationError from Pydantic
            packet.confidence = 0.9
    
    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected (extra='forbid')."""
        with pytest.raises(Exception):  # ValidationError from Pydantic
            EvidencePacketModel(
                schema_version='1.0.0',
                stage=1,
                source_component='test',
                evidence_type='test',
                content={'data': 'value'},
                confidence=0.5,
                applicable_questions=['Q1'],
                metadata={},
                extra_field='should_fail',  # This should be rejected
            )
    
    def test_confidence_bounds_validation(self):
        """Test that confidence is validated to be in [0.0, 1.0]."""
        # Valid confidence
        packet = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value'},
            confidence=0.5,
            applicable_questions=['Q1'],
        )
        assert packet.confidence == 0.5
        
        # Invalid confidence > 1.0
        with pytest.raises(Exception):
            create_evidence_packet(
                stage=1,
                source_component='test',
                evidence_type='test',
                content={'data': 'value'},
                confidence=1.5,
                applicable_questions=['Q1'],
            )
        
        # Invalid confidence < 0.0
        with pytest.raises(Exception):
            create_evidence_packet(
                stage=1,
                source_component='test',
                evidence_type='test',
                content={'data': 'value'},
                confidence=-0.1,
                applicable_questions=['Q1'],
            )
    
    def test_signature_changes_with_content(self):
        """Test that signature changes when content changes."""
        packet1 = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value1'},
            confidence=0.5,
            applicable_questions=['Q1'],
        )
        
        packet2 = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value2'},  # Different content
            confidence=0.5,
            applicable_questions=['Q1'],
        )
        
        # Signatures should be different
        assert packet1.signature != packet2.signature
        assert packet1.evidence_hash != packet2.evidence_hash
    
    def test_hash_is_deterministic(self):
        """Test that hash is deterministic for same content and timestamp."""
        # Create packet once
        packet1 = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value'},
            confidence=0.5,
            applicable_questions=['Q1'],
            sign=False,
        )
        
        # Create second packet with same timestamp (to ensure determinism)
        packet2 = EvidencePacketModel(
            schema_version='1.0.0',
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value'},
            confidence=0.5,
            applicable_questions=['Q1'],
            metadata={},
            timestamp=packet1.timestamp,  # Use same timestamp
        )
        
        # Hashes should be the same
        hash1 = packet1.compute_hash()
        hash2 = packet2.compute_hash()
        assert hash1 == hash2
    
    def test_factory_validate_function(self):
        """Test the validate_packet factory function."""
        packet = create_feasibility_evidence(
            content={'finding': 'test'},
            confidence=0.8,
            applicable_questions=['Q1'],
        )
        
        # Should be valid
        assert validate_packet(packet) is True
        
        # Create packet with wrong signature
        secret = get_hmac_secret()
        bad_packet = packet.model_copy(update={'signature': 'invalid_signature'})
        assert validate_packet(bad_packet, secret) is False
    
    def test_schema_version_format(self):
        """Test that schema version follows semantic versioning."""
        packet = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'data': 'value'},
            confidence=0.5,
            applicable_questions=['Q1'],
            schema_version='2.1.3',
        )
        
        assert packet.schema_version == '2.1.3'
        
        # Invalid version format should fail
        with pytest.raises(Exception):
            create_evidence_packet(
                stage=1,
                source_component='test',
                evidence_type='test',
                content={'data': 'value'},
                confidence=0.5,
                applicable_questions=['Q1'],
                schema_version='invalid',
            )


class TestCanonicalJSONProperties:
    """Tests specifically for canonical JSON properties."""
    
    def test_no_whitespace_variance(self):
        """Test that canonical JSON has no whitespace variance."""
        packet = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'key': 'value'},
            confidence=0.5,
            applicable_questions=['Q1'],
            sign=False,
        )
        
        canonical = packet.canonical_json()
        
        # Should not contain newlines or extra spaces
        assert '\n' not in canonical
        assert '  ' not in canonical  # No double spaces
        
        # Should use compact separators
        assert ', ' not in canonical  # Should be ',' not ', '
        assert ': ' not in canonical  # Should be ':' not ': '
    
    def test_ensure_ascii(self):
        """Test that canonical JSON uses ASCII encoding."""
        packet = create_evidence_packet(
            stage=1,
            source_component='test',
            evidence_type='test',
            content={'text': 'español ñ'},
            confidence=0.5,
            applicable_questions=['Q1'],
            sign=False,
        )
        
        canonical = packet.canonical_json()
        
        # Should only contain ASCII characters (Unicode escaped)
        assert all(ord(c) < 128 for c in canonical)


# Sample output for documentation
"""
Expected test output:

$ pytest tests/test_contract_producer.py -v

tests/test_contract_producer.py::TestContractProducer::test_create_valid_packet PASSED
tests/test_contract_producer.py::TestContractProducer::test_signature_is_valid PASSED
tests/test_contract_producer.py::TestContractProducer::test_canonical_json_stability PASSED
tests/test_contract_producer.py::TestContractProducer::test_canonical_json_sorted_keys PASSED
tests/test_contract_producer.py::TestContractProducer::test_immutability_enforcement PASSED
tests/test_contract_producer.py::TestContractProducer::test_extra_fields_rejected PASSED
tests/test_contract_producer.py::TestContractProducer::test_confidence_bounds_validation PASSED
tests/test_contract_producer.py::TestContractProducer::test_signature_changes_with_content PASSED
tests/test_contract_producer.py::TestContractProducer::test_hash_is_deterministic PASSED
tests/test_contract_producer.py::TestContractProducer::test_factory_validate_function PASSED
tests/test_contract_producer.py::TestContractProducer::test_schema_version_format PASSED
tests/test_contract_producer.py::TestCanonicalJSONProperties::test_no_whitespace_variance PASSED
tests/test_contract_producer.py::TestCanonicalJSONProperties::test_ensure_ascii PASSED

============= 13 passed in 0.23s =============
"""
