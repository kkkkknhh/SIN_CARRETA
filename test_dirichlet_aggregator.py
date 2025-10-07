#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for DirichletAggregator Bayesian evidence aggregation.

Tests cover:
1. Basic initialization and parameter storage
2. Discrete label voting (update_from_labels)
3. Continuous weight voting (update_from_weights)
4. Posterior statistics (mean, mode, credible intervals)
5. Utility methods (entropy, max_probability)
6. Integration with EvidenceRegistry
7. Consensus determination with thresholds
8. Edge cases and error handling
"""

import unittest
import numpy as np
from evidence_registry import DirichletAggregator, EvidenceRegistry


class TestDirichletAggregatorBasics(unittest.TestCase):
    """Test basic DirichletAggregator functionality"""
    
    def test_initialization(self):
        """Test aggregator initialization"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        
        self.assertEqual(agg.k, 3)
        self.assertEqual(agg.alpha0, 0.5)
        self.assertEqual(agg.n_updates, 0)
        np.testing.assert_array_almost_equal(agg.alpha, [0.5, 0.5, 0.5])
    
    def test_default_alpha0(self):
        """Test default alpha0 value (Jeffreys prior)"""
        agg = DirichletAggregator(k=5)
        self.assertEqual(agg.alpha0, 0.5)
    
    def test_reset(self):
        """Test reset to prior"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        agg.update_from_labels(np.array([0, 0, 1]))
        
        self.assertGreater(agg.n_updates, 0)
        
        agg.reset()
        
        self.assertEqual(agg.n_updates, 0)
        np.testing.assert_array_almost_equal(agg.alpha, [1.0, 1.0, 1.0])


class TestDiscreteVoting(unittest.TestCase):
    """Test discrete label voting"""
    
    def test_unanimous_votes(self):
        """Test unanimous voting for single category"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        agg.update_from_labels(np.array([0, 0, 0, 0, 0]))  # 5 votos para cat 0
        
        mean = agg.posterior_mean()
        
        # Category 0 should dominate
        self.assertGreater(mean[0], 0.8, "Categoría 0 debe dominar")
        self.assertLess(mean[1], 0.15)
        self.assertLess(mean[2], 0.15)
    
    def test_split_votes(self):
        """Test split voting across categories"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        agg.update_from_labels(np.array([0, 1, 2]))  # 1 voto cada categoría
        
        entropy = agg.entropy()
        
        # High uncertainty with split votes
        self.assertGreater(entropy, 0.5, "Alta incertidumbre con votos divididos")
    
    def test_vote_accumulation(self):
        """Test that votes accumulate correctly"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        
        # Initial state: alpha = [1, 1, 1]
        agg.update_from_labels(np.array([0, 0]))  # +2 votes for cat 0
        np.testing.assert_array_almost_equal(agg.alpha, [3.0, 1.0, 1.0])
        
        agg.update_from_labels(np.array([1]))  # +1 vote for cat 1
        np.testing.assert_array_almost_equal(agg.alpha, [3.0, 2.0, 1.0])
    
    def test_empty_labels(self):
        """Test empty label array"""
        agg = DirichletAggregator(k=3)
        initial_alpha = agg.alpha.copy()
        
        agg.update_from_labels(np.array([]))
        
        np.testing.assert_array_equal(agg.alpha, initial_alpha)
        self.assertEqual(agg.n_updates, 0)
    
    def test_invalid_labels_raises_error(self):
        """Test that out-of-range labels raise ValueError"""
        agg = DirichletAggregator(k=3)
        
        with self.assertRaises(ValueError):
            agg.update_from_labels(np.array([0, 1, 3]))  # 3 >= k
        
        with self.assertRaises(ValueError):
            agg.update_from_labels(np.array([-1, 0, 1]))  # -1 < 0


class TestContinuousWeights(unittest.TestCase):
    """Test continuous weight voting"""
    
    def test_weight_update(self):
        """Test weight-based update"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        
        # Strong vote for category 0
        weights = np.array([0.8, 0.1, 0.1])
        agg.update_from_weights(weights)
        
        mean = agg.posterior_mean()
        self.assertGreater(mean[0], mean[1])
        self.assertGreater(mean[0], mean[2])
    
    def test_weight_normalization(self):
        """Test that weights are normalized internally"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        
        # Non-normalized weights (sum = 2.0)
        weights = np.array([1.0, 0.5, 0.5])
        agg.update_from_weights(weights)
        
        # Should still work (normalized internally)
        mean = agg.posterior_mean()
        np.testing.assert_almost_equal(mean.sum(), 1.0)
    
    def test_invalid_weight_shape_raises_error(self):
        """Test that wrong shape raises ValueError"""
        agg = DirichletAggregator(k=3)
        
        with self.assertRaises(ValueError):
            agg.update_from_weights(np.array([0.5, 0.5]))  # Wrong length
        
        with self.assertRaises(ValueError):
            agg.update_from_weights(np.array([[0.3, 0.3, 0.4]]))  # Wrong shape


class TestPosteriorStatistics(unittest.TestCase):
    """Test posterior statistical methods"""
    
    def test_posterior_mean(self):
        """Test posterior mean calculation"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        agg.update_from_labels(np.array([0, 0, 1]))  # alpha = [3, 2, 1]
        
        mean = agg.posterior_mean()
        
        # Mean = alpha / sum(alpha) = [3, 2, 1] / 6
        np.testing.assert_array_almost_equal(mean, [0.5, 1/3, 1/6], decimal=5)
        np.testing.assert_almost_equal(mean.sum(), 1.0)
    
    def test_posterior_mode(self):
        """Test posterior mode (MAP estimate)"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        # Need all alpha > 1 for mode to be defined
        agg.update_from_labels(np.array([0, 0, 0, 1, 2]))  # alpha = [4, 2, 2]
        
        mode = agg.posterior_mode()
        
        # Mode = (alpha - 1) / (sum(alpha) - k) = [3, 1, 1] / 5
        expected_mode = np.array([3, 1, 1]) / 5.0
        np.testing.assert_array_almost_equal(mode, expected_mode, decimal=5)
        np.testing.assert_almost_equal(mode.sum(), 1.0)
    
    def test_mode_with_alpha_le_1_returns_mean(self):
        """Test that mode returns mean when alpha <= 1"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        
        # alpha = [0.5, 0.5, 0.5], mode not defined
        mode = agg.posterior_mode()
        mean = agg.posterior_mean()
        
        np.testing.assert_array_almost_equal(mode, mean)
    
    def test_credible_interval(self):
        """Test credible interval calculation"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        agg.update_from_labels(np.array([0, 0, 0, 0, 0]))  # Strong evidence for cat 0
        
        intervals = agg.credible_interval(level=0.95)
        
        # Check shape
        self.assertEqual(intervals.shape, (3, 2))
        
        # Check ordering: lo <= hi
        self.assertTrue(np.all(intervals[:, 0] <= intervals[:, 1]))
        
        # Category 0 should have high probability interval
        self.assertGreater(intervals[0, 0], 0.4)  # Lower bound > 0.4
    
    def test_credible_interval_coverage(self):
        """Test that credible intervals have correct coverage"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        agg.update_from_labels(np.array([0, 1, 2]))
        
        intervals_95 = agg.credible_interval(level=0.95)
        intervals_90 = agg.credible_interval(level=0.90)
        
        # 95% intervals should be wider than 90%
        width_95 = intervals_95[:, 1] - intervals_95[:, 0]
        width_90 = intervals_90[:, 1] - intervals_90[:, 0]
        
        self.assertTrue(np.all(width_95 >= width_90))


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods"""
    
    def test_entropy_low_with_consensus(self):
        """Test low entropy with strong consensus"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        agg.update_from_labels(np.array([0] * 10))  # 10 votes for cat 0
        
        entropy = agg.entropy()
        
        # Low uncertainty with consensus
        self.assertLess(entropy, 0.5)
    
    def test_entropy_high_with_split(self):
        """Test high entropy with split votes"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        agg.update_from_labels(np.array([0, 0, 1, 1, 2, 2]))  # Even split
        
        entropy = agg.entropy()
        
        # High uncertainty with split
        self.assertGreater(entropy, 0.8)
    
    def test_max_probability(self):
        """Test max_probability returns correct category"""
        agg = DirichletAggregator(k=3, alpha0=1.0)
        agg.update_from_labels(np.array([1, 1, 1, 1, 0]))  # Cat 1 dominates
        
        max_cat, max_prob = agg.max_probability()
        
        self.assertEqual(max_cat, 1)
        self.assertGreater(max_prob, 0.5)
        self.assertLessEqual(max_prob, 1.0)


class TestEvidenceRegistryIntegration(unittest.TestCase):
    """Test integration with EvidenceRegistry"""
    
    def test_registry_has_aggregators(self):
        """Test that registry initializes aggregators"""
        registry = EvidenceRegistry()
        
        self.assertIsInstance(registry.dimension_aggregators, dict)
        self.assertIsInstance(registry.content_type_aggregator, DirichletAggregator)
        self.assertIsInstance(registry.risk_level_aggregator, DirichletAggregator)
        
        # Check aggregator dimensions
        self.assertEqual(registry.content_type_aggregator.k, 5)
        self.assertEqual(registry.risk_level_aggregator.k, 3)
    
    def test_register_evidence_creates_aggregator(self):
        """Test that register_evidence creates dimension aggregator"""
        registry = EvidenceRegistry()
        
        registry.register_evidence(
            evidence_id="test_001",
            source="test_detector",
            dimension_vote=3,  # D4
            content_type=0,
            risk_level=1,
            confidence=0.9
        )
        
        self.assertIn("test_001", registry.dimension_aggregators)
        self.assertEqual(registry.dimension_aggregators["test_001"].k, 10)
    
    def test_register_evidence_updates_aggregators(self):
        """Test that multiple registrations update aggregators"""
        registry = EvidenceRegistry()
        
        # First vote
        registry.register_evidence(
            evidence_id="test_001",
            source="detector_1",
            dimension_vote=2,
            content_type=0,
            risk_level=1,
            confidence=0.8
        )
        
        # Second vote (same evidence_id, different dimension)
        registry.register_evidence(
            evidence_id="test_001",
            source="detector_2",
            dimension_vote=2,  # Same dimension
            content_type=1,
            risk_level=1,
            confidence=0.9
        )
        
        # Check that aggregator was updated
        agg = registry.dimension_aggregators["test_001"]
        self.assertGreater(agg.n_updates, 0)
    
    def test_get_dimension_distribution(self):
        """Test get_dimension_distribution returns correct structure"""
        registry = EvidenceRegistry()
        
        registry.register_evidence(
            evidence_id="test_002",
            source="detector",
            dimension_vote=5,
            content_type=1,
            risk_level=0,
            confidence=0.85
        )
        
        dist = registry.get_dimension_distribution("test_002")
        
        self.assertIsNotNone(dist)
        self.assertIn('mean', dist)
        self.assertIn('credible_interval', dist)
        self.assertIn('max_category', dist)
        self.assertIn('entropy', dist)
        self.assertIn('n_votes', dist)
        
        # Check types
        self.assertIsInstance(dist['mean'], np.ndarray)
        self.assertIsInstance(dist['credible_interval'], np.ndarray)
        self.assertIsInstance(dist['max_category'], tuple)
        self.assertIsInstance(dist['entropy'], float)
        self.assertIsInstance(dist['n_votes'], int)
    
    def test_get_dimension_distribution_nonexistent(self):
        """Test get_dimension_distribution with nonexistent evidence_id"""
        registry = EvidenceRegistry()
        
        dist = registry.get_dimension_distribution("nonexistent")
        
        self.assertIsNone(dist)
    
    def test_get_consensus_dimension_with_consensus(self):
        """Test consensus detection with strong votes"""
        registry = EvidenceRegistry()
        
        # Multiple strong votes for dimension 3
        for i in range(5):
            registry.register_evidence(
                evidence_id="consensus_test",
                source=f"detector_{i}",
                dimension_vote=3,
                content_type=0,
                risk_level=1,
                confidence=0.9
            )
        
        consensus = registry.get_consensus_dimension("consensus_test", threshold=0.6)
        
        self.assertIsNotNone(consensus)
        self.assertEqual(consensus, 3)
    
    def test_get_consensus_dimension_without_consensus(self):
        """Test no consensus with split votes"""
        registry = EvidenceRegistry()
        
        # Split votes across dimensions
        registry.register_evidence(
            evidence_id="split_test",
            source="detector_1",
            dimension_vote=0,
            content_type=0,
            risk_level=1,
            confidence=0.5
        )
        registry.register_evidence(
            evidence_id="split_test",
            source="detector_2",
            dimension_vote=5,
            content_type=0,
            risk_level=1,
            confidence=0.5
        )
        
        consensus = registry.get_consensus_dimension("split_test", threshold=0.6)
        
        self.assertIsNone(consensus)
    
    def test_get_consensus_dimension_nonexistent(self):
        """Test consensus for nonexistent evidence"""
        registry = EvidenceRegistry()
        
        consensus = registry.get_consensus_dimension("nonexistent", threshold=0.6)
        
        self.assertIsNone(consensus)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_single_category(self):
        """Test aggregator with k=1"""
        agg = DirichletAggregator(k=1, alpha0=1.0)
        agg.update_from_labels(np.array([0, 0, 0]))
        
        mean = agg.posterior_mean()
        np.testing.assert_array_almost_equal(mean, [1.0])
    
    def test_large_k(self):
        """Test aggregator with large k"""
        agg = DirichletAggregator(k=100, alpha0=0.5)
        agg.update_from_labels(np.array([0, 1, 2, 50, 99]))
        
        mean = agg.posterior_mean()
        self.assertEqual(len(mean), 100)
        np.testing.assert_almost_equal(mean.sum(), 1.0)
    
    def test_zero_alpha0(self):
        """Test with zero prior (improper prior)"""
        agg = DirichletAggregator(k=3, alpha0=0.0)
        agg.update_from_labels(np.array([0, 1, 2]))
        
        # Should still work
        mean = agg.posterior_mean()
        np.testing.assert_almost_equal(mean.sum(), 1.0)
    
    def test_very_high_confidence(self):
        """Test with very high confidence votes"""
        agg = DirichletAggregator(k=3, alpha0=0.5)
        
        # Simulate many votes
        agg.update_from_labels(np.array([0] * 1000))
        
        mean = agg.posterior_mean()
        
        # Should converge to category 0
        self.assertGreater(mean[0], 0.99)


class TestMultipleVotingSources(unittest.TestCase):
    """Test realistic scenarios with multiple voting sources"""
    
    def test_detector_consensus_scenario(self):
        """Test scenario with multiple detectors voting"""
        registry = EvidenceRegistry()
        
        # 3 detectors vote for D4, 1 votes for D5
        evidence_id = "multi_detector_evidence"
        
        registry.register_evidence(evidence_id, "responsibility_detector", 3, 0, 1, 0.9)
        registry.register_evidence(evidence_id, "monetary_detector", 3, 1, 1, 0.85)
        registry.register_evidence(evidence_id, "causal_detector", 3, 2, 1, 0.8)
        registry.register_evidence(evidence_id, "feasibility_scorer", 4, 0, 1, 0.7)
        
        dist = registry.get_dimension_distribution(evidence_id)
        max_cat, max_prob = dist['max_category']
        
        # D4 (index 3) should win
        self.assertEqual(max_cat, 3)
        self.assertGreater(max_prob, 0.6)
    
    def test_uncertainty_with_split_detectors(self):
        """Test high uncertainty with completely split votes"""
        registry = EvidenceRegistry()
        
        evidence_id = "uncertain_evidence"
        
        # Each detector votes for different dimension
        for i in range(5):
            registry.register_evidence(
                evidence_id, f"detector_{i}", 
                dimension_vote=i, 
                content_type=0, 
                risk_level=1, 
                confidence=0.8
            )
        
        dist = registry.get_dimension_distribution(evidence_id)
        
        # High entropy (uncertainty)
        self.assertGreater(dist['entropy'], 1.0)
        
        # No strong consensus
        _, max_prob = dist['max_category']
        self.assertLess(max_prob, 0.4)


if __name__ == '__main__':
    unittest.main()
