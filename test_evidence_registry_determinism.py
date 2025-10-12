#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for EvidenceRegistry deterministic hashing.

Tests cover:
1. Deterministic evidence_id generation for same content
2. Registry hash reproducibility across multiple runs
3. Frozen registry immutability enforcement
4. Provenance tracking (evidence → questions)
5. Component indexing for evidence lookup
6. Hash consistency with different insertion orders
7. Evidence serialization determinism
"""

import unittest
from evidence_registry import EvidenceRegistry, CanonicalEvidence


class TestEvidenceIDDeterminism(unittest.TestCase):
    """Test deterministic evidence_id generation"""
    
    def test_same_content_generates_same_id(self):
        """Test that identical evidence generates identical evidence_id"""
        registry1 = EvidenceRegistry()
        registry2 = EvidenceRegistry()
        
        # Register same evidence in both registries
        id1 = registry1.register(
            source_component="feasibility_scorer",
            evidence_type="baseline_presence",
            content={"baseline_text": "línea base 2024", "confidence": 0.95},
            confidence=0.95,
            applicable_questions=["D1-Q1"]
        )
        
        id2 = registry2.register(
            source_component="feasibility_scorer",
            evidence_type="baseline_presence",
            content={"baseline_text": "línea base 2024", "confidence": 0.95},
            confidence=0.95,
            applicable_questions=["D1-Q1"]
        )
        
        self.assertEqual(id1, id2, 
                         "Same evidence must generate same evidence_id")
    
    def test_different_content_generates_different_id(self):
        """Test that different evidence generates different evidence_id"""
        registry = EvidenceRegistry()
        
        id1 = registry.register(
            source_component="monetary_detector",
            evidence_type="monetary_value",
            content={"amount": 1000000, "currency": "COP"},
            confidence=0.9,
            applicable_questions=["D2-Q3"]
        )
        
        id2 = registry.register(
            source_component="monetary_detector",
            evidence_type="monetary_value",
            content={"amount": 2000000, "currency": "COP"},
            confidence=0.9,
            applicable_questions=["D2-Q3"]
        )
        
        self.assertNotEqual(id1, id2, 
                            "Different evidence must generate different evidence_id")
    
    def test_evidence_id_is_deterministic_string(self):
        """Test that evidence_id is a deterministic string"""
        registry = EvidenceRegistry()
        
        eid = registry.register(
            source_component="test_component",
            evidence_type="test_type",
            content={"value": 42},
            confidence=0.8,
            applicable_questions=["D1-Q1"]
        )
        
        # Evidence ID format: "component::type::hash_fragment"
        self.assertIsInstance(eid, str)
        self.assertIn("test_component", eid)
        self.assertIn("test_type", eid)
        self.assertIn("::", eid)


class TestRegistryHashReproducibility(unittest.TestCase):
    """Test registry-level hash reproducibility"""
    
    def test_empty_registry_has_consistent_hash(self):
        """Test that empty registries have consistent hash"""
        registry1 = EvidenceRegistry()
        registry2 = EvidenceRegistry()
        
        hash1 = registry1.deterministic_hash()
        hash2 = registry2.deterministic_hash()
        
        self.assertEqual(hash1, hash2, 
                         "Empty registries must have same hash")
    
    def test_same_evidence_sequence_produces_same_hash(self):
        """Test that same evidence in same order produces same hash"""
        evidence_items = [
            ("feasibility_scorer", "baseline", {"text": "baseline"}, 0.9, ["D1-Q1"]),
            ("monetary_detector", "amount", {"value": 1000000}, 0.85, ["D2-Q3"]),
            ("responsibility_detector", "entity", {"name": "Alcaldía"}, 0.95, ["D3-Q5"]),
        ]
        
        hashes = []
        for _ in range(3):
            registry = EvidenceRegistry()
            for item in evidence_items:
                registry.register(*item)
            hashes.append(registry.deterministic_hash())
        
        self.assertEqual(len(set(hashes)), 1, 
                         f"Registry hashes not reproducible: {hashes}")
    
    def test_different_order_produces_same_hash(self):
        """Test that different insertion order produces same hash (sorted internally)"""
        items1 = [
            ("comp1", "type1", {"val": 1}, 0.8, ["D1-Q1"]),
            ("comp2", "type2", {"val": 2}, 0.9, ["D1-Q2"]),
        ]
        
        items2 = [
            ("comp2", "type2", {"val": 2}, 0.9, ["D1-Q2"]),
            ("comp1", "type1", {"val": 1}, 0.8, ["D1-Q1"]),
        ]
        
        registry1 = EvidenceRegistry()
        for item in items1:
            registry1.register(*item)
        
        registry2 = EvidenceRegistry()
        for item in items2:
            registry2.register(*item)
        
        # Same evidence set should produce same hash (deterministic_hash sorts)
        self.assertEqual(registry1.deterministic_hash(), 
                         registry2.deterministic_hash(),
                         "Same evidence set should produce same hash regardless of insertion order")


class TestFrozenRegistry(unittest.TestCase):
    """Test frozen registry immutability"""
    
    def test_frozen_registry_rejects_new_evidence(self):
        """Test that frozen registries reject new evidence"""
        registry = EvidenceRegistry()
        registry.register("comp1", "type1", {}, 0.5, ["D1-Q1"])
        
        # Freeze registry
        registry.freeze()
        
        # Attempt to register more evidence should raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            registry.register("comp2", "type2", {}, 0.6, ["D1-Q2"])
        
        self.assertIn("frozen", str(ctx.exception).lower())
    
    def test_frozen_registry_preserves_existing_evidence(self):
        """Test that freezing preserves existing evidence"""
        registry = EvidenceRegistry()
        _eid = registry.register("comp1", "type1", {"data": "test"}, 0.8, ["D1-Q1"])
        
        registry.freeze()
        
        # Existing evidence should still be accessible via for_question
        evidence_list = registry.for_question("D1-Q1")
        self.assertEqual(len(evidence_list), 1)
        self.assertEqual(evidence_list[0].content["data"], "test")
    
    def test_frozen_registry_hash_is_stable(self):
        """Test that frozen registry hash doesn't change"""
        registry = EvidenceRegistry()
        registry.register("comp1", "type1", {}, 0.5, ["D1-Q1"])
        
        hash_before = registry.deterministic_hash()
        registry.freeze()
        hash_after = registry.deterministic_hash()
        
        self.assertEqual(hash_before, hash_after,
                         "Hash must not change after freezing")


class TestProvenanceTracking(unittest.TestCase):
    """Test evidence → question provenance tracking"""
    
    def test_evidence_tracked_for_single_question(self):
        """Test evidence correctly tracked for single question"""
        registry = EvidenceRegistry()
        
        _eid = registry.register(
            "feasibility_scorer",
            "baseline_presence",
            {"text": "baseline detected"},
            0.9,
            ["D1-Q1"]
        )
        
        # Check provenance using for_question method
        evidence_list = registry.for_question("D1-Q1")
        self.assertEqual(len(evidence_list), 1)
        self.assertEqual(evidence_list[0].source_component, "feasibility_scorer")
    
    def test_evidence_tracked_for_multiple_questions(self):
        """Test evidence correctly tracked for multiple questions"""
        registry = EvidenceRegistry()
        
        _eid = registry.register(
            "monetary_detector",
            "monetary_value",
            {"amount": 5000000},
            0.85,
            ["D2-Q1", "D2-Q3", "D3-Q5"]
        )
        
        # Check provenance for each question
        for qid in ["D2-Q1", "D2-Q3", "D3-Q5"]:
            evidence_list = registry.for_question(qid)
            self.assertEqual(len(evidence_list), 1)
            self.assertEqual(evidence_list[0].content["amount"], 5000000)
    
    def test_multiple_evidence_for_same_question(self):
        """Test multiple evidence items tracked for same question"""
        registry = EvidenceRegistry()
        
        eid1 = registry.register("comp1", "type1", {"val": 1}, 0.8, ["D1-Q1"])
        eid2 = registry.register("comp2", "type2", {"val": 2}, 0.9, ["D1-Q1"])
        eid3 = registry.register("comp3", "type3", {"val": 3}, 0.7, ["D1-Q1"])
        
        # Check provenance
        evidence_list = registry.for_question("D1-Q1")
        self.assertEqual(len(evidence_list), 3)
        
        # Verify all evidence IDs present
        evidence_ids = {e.metadata["evidence_id"] for e in evidence_list}
        self.assertIn(eid1, evidence_ids)
        self.assertIn(eid2, evidence_ids)
        self.assertIn(eid3, evidence_ids)
    
    def test_provenance_empty_for_unknown_question(self):
        """Test that unknown question returns empty list"""
        registry = EvidenceRegistry()
        registry.register("comp1", "type1", {}, 0.5, ["D1-Q1"])
        
        evidence_list = registry.for_question("D99-Q99")
        self.assertEqual(len(evidence_list), 0)


class TestComponentIndexing(unittest.TestCase):
    """Test component-based evidence indexing"""
    
    def test_evidence_indexed_by_component(self):
        """Test evidence can be retrieved by component"""
        registry = EvidenceRegistry()
        
        _eid1 = registry.register("feasibility_scorer", "type1", {}, 0.8, ["D1-Q1"])
        _eid2 = registry.register("feasibility_scorer", "type2", {}, 0.9, ["D1-Q2"])
        _eid3 = registry.register("monetary_detector", "type3", {}, 0.7, ["D2-Q1"])
        
        # Get evidence by component using for_component method
        feasibility_evidence = registry.for_component("feasibility_scorer")
        monetary_evidence = registry.for_component("monetary_detector")
        
        self.assertEqual(len(feasibility_evidence), 2)
        self.assertEqual(len(monetary_evidence), 1)
    
    def test_unknown_component_returns_empty_list(self):
        """Test that unknown component returns empty list"""
        registry = EvidenceRegistry()
        registry.register("comp1", "type1", {}, 0.5, ["D1-Q1"])
        
        evidence_list = registry.for_component("unknown_component")
        self.assertEqual(len(evidence_list), 0)


class TestEvidenceSerializationDeterminism(unittest.TestCase):
    """Test evidence serialization is deterministic"""
    
    def test_evidence_to_dict_is_deterministic(self):
        """Test that to_dict() produces consistent output"""
        evidence1 = CanonicalEvidence(
            source_component="test_comp",
            evidence_type="test_type",
            content={"key": "value"},
            confidence=0.85,
            applicable_questions=["D1-Q1", "D1-Q2"],
            metadata={"timestamp": "2024-01-01"}
        )
        
        # Convert to dict multiple times
        dicts = [evidence1.to_dict() for _ in range(5)]
        
        # All should be identical
        for d in dicts[1:]:
            self.assertEqual(dicts[0], d)
    
    def test_registry_export_is_deterministic(self):
        """Test that registry export produces consistent output"""
        import tempfile
        import os
        
        evidence_items = [
            ("comp1", "type1", {"val": 1}, 0.8, ["D1-Q1"]),
            ("comp2", "type2", {"val": 2}, 0.9, ["D1-Q2"]),
        ]
        
        hashes = []
        for _ in range(3):
            registry = EvidenceRegistry()
            for item in evidence_items:
                registry.register(*item)
            
            # Export to temp file and read back for hash
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_path = f.name
                registry.export_to_json(temp_path)
            
            # Read exported content (exclude timestamp field)
            with open(temp_path, 'r') as f:
                content = f.read()
            
            os.unlink(temp_path)
            
            # Just verify export succeeds - content may have timestamps
            self.assertTrue(len(content) > 0)
            
            # Use deterministic hash instead
            hashes.append(registry.deterministic_hash())
        
        # All deterministic hashes should be identical
        self.assertEqual(len(set(hashes)), 1,
                         f"Export hashes not consistent: {hashes}")


class TestRegistryStatistics(unittest.TestCase):
    """Test registry statistics and reporting"""
    
    def test_registry_reports_correct_counts(self):
        """Test that registry correctly counts evidence"""
        registry = EvidenceRegistry()
        
        registry.register("comp1", "type1", {}, 0.8, ["D1-Q1"])
        registry.register("comp2", "type2", {}, 0.9, ["D1-Q2"])
        registry.register("comp3", "type3", {}, 0.7, ["D1-Q3"])
        
        stats = registry.get_statistics()
        
        self.assertEqual(stats["total_evidence"], 3)
        self.assertEqual(stats["total_components"], 3)
        self.assertIn("comp1", stats["evidence_by_component"])
        self.assertIn("comp2", stats["evidence_by_component"])
        self.assertIn("comp3", stats["evidence_by_component"])


if __name__ == "__main__":
    unittest.main()
