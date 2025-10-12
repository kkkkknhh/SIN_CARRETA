#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Strategic Decalogo Integrator
============================================

Comprehensive test suite with ZERO TOLERANCE for mediocrity.
ALL tests MUST pass. NO EXCEPTIONS.

Tests cover:
1. Semantic extraction threshold enforcement (BEIR-validated 0.75)
2. Causal graph cycle detection (Pearl's criterion)
3. Bayesian integration conflict detection (Gelman's framework)
4. Dimension coverage completeness (300 questions, 6 dimensions)
5. Deterministic execution across runs
6. Performance metrics validation
"""

import hashlib
import json
import unittest

import networkx as nx

from evidence_registry import EvidenceRegistry
from strategic_decalogo_integrator import (
    SemanticExtractor,
    CausalGraphAnalyzer,
    BayesianEvidenceIntegrator,
    DecalogoEvidenceExtractor,
    StrategicDecalogoIntegrator,
    StructuredEvidence
)


class TestSemanticExtractionThresholdEnforcement(unittest.TestCase):
    """Test semantic extraction respects BEIR-validated threshold"""
    
    def setUp(self):
        """Initialize semantic extractor"""
        self.extractor = SemanticExtractor()
    
    def test_threshold_enforcement(self):
        """Test that extraction respects 0.75 threshold"""
        query = "¿El plan identifica recursos financieros?"
        segments = [
            "El presupuesto total asciende a $500 millones",  # High relevance
            "El clima de la región es cálido",  # Zero relevance
            "Se asignan fondos para infraestructura"  # Medium relevance
        ]
        
        results = self.extractor.extract_evidence(query, segments, top_k=10)
        
        # REQUIRED: All results meet threshold
        for text, score in results:
            self.assertGreaterEqual(
                score, 0.75,
                f"FAILED: Score {score:.3f} below threshold for segment: {text[:50]}"
            )
    
    def test_irrelevant_segment_exclusion(self):
        """Test that irrelevant segments are excluded"""
        query = "¿Se definen indicadores medibles?"
        segments = [
            "Los indicadores incluyen tasa de cobertura y calidad",  # Relevant
            "La geografía presenta montañas",  # Irrelevant
            "Se medirán anualmente usando encuestas"  # Relevant
        ]
        
        results = self.extractor.extract_evidence(query, segments)
        
        # REQUIRED: Irrelevant segment not included
        irrelevant_texts = [text for text, _ in results if "geografía" in text.lower() or "montañas" in text.lower()]
        self.assertEqual(
            len(irrelevant_texts), 0,
            "FAILED: Irrelevant segment not filtered"
        )
    
    def test_deterministic_ordering(self):
        """Test that results are deterministically ordered"""
        query = "¿Hay diagnóstico de línea base?"
        segments = [
            "Línea base documentada con datos 2023",
            "Diagnóstico completo de la situación actual",
            "Fuentes primarias y secundarias utilizadas"
        ]
        
        # Run twice
        results1 = self.extractor.extract_evidence(query, segments)
        results2 = self.extractor.extract_evidence(query, segments)
        
        # REQUIRED: Identical results
        self.assertEqual(
            results1, results2,
            "FAILED: Results not deterministic"
        )
    
    def test_empty_segments_handling(self):
        """Test handling of empty segment list"""
        query = "¿Existen metas?"
        segments = []
        
        results = self.extractor.extract_evidence(query, segments)
        
        self.assertEqual(len(results), 0, "FAILED: Should return empty list for empty input")


class TestCausalGraphCycleDetection(unittest.TestCase):
    """Test causal graph analysis detects cycles correctly"""
    
    def setUp(self):
        """Initialize causal analyzer"""
        self.analyzer = CausalGraphAnalyzer()
    
    def test_cycle_detection(self):
        """Test that cyclic graphs are rejected"""
        # Create graph with cycle: A -> B -> C -> A
        G = nx.DiGraph()
        G.add_edges_from([
            ('A', 'B', {'weight': 0.8}),
            ('B', 'C', {'weight': 0.7}),
            ('C', 'A', {'weight': 0.6})  # Creates cycle
        ])
        
        result = self.analyzer.analyze_dimension(G, 'D1')
        
        # REQUIRED: Must detect cycle
        self.assertFalse(result.valid, "FAILED: Did not reject cyclic graph")
        self.assertIn('CYCLE_DETECTED', result.validation_errors, 
                     "FAILED: No cycle detection message")
        self.assertLess(result.acyclicity_pvalue, 0.95,
                       "FAILED: p-value too high for cyclic graph")
    
    def test_acyclic_graph_acceptance(self):
        """Test that valid acyclic graphs are accepted"""
        # Create valid DAG: A -> B -> C, A -> C
        G = nx.DiGraph()
        G.add_edges_from([
            ('A', 'B', {'weight': 0.8}),
            ('B', 'C', {'weight': 0.7}),
            ('A', 'C', {'weight': 0.5})
        ])
        
        result = self.analyzer.analyze_dimension(G, 'D1')
        
        # REQUIRED: Must accept valid DAG
        self.assertTrue(result.valid, "FAILED: Rejected valid DAG")
        self.assertGreaterEqual(result.acyclicity_pvalue, 0.95,
                               "FAILED: p-value too low for valid DAG")
    
    def test_confounder_identification(self):
        """Test identification of confounders"""
        # Create graph with confounder: Z -> X, Z -> Y, X -> Y
        G = nx.DiGraph()
        G.add_edges_from([
            ('Z', 'X', {'weight': 0.7}),
            ('Z', 'Y', {'weight': 0.6}),
            ('X', 'Y', {'weight': 0.8})
        ])
        
        result = self.analyzer.analyze_dimension(G, 'D1')
        
        # REQUIRED: Must identify Z as confounder
        self.assertGreater(result.confounder_count, 0,
                          "FAILED: No confounders identified")
        self.assertIn('Z', result.confounders,
                     "FAILED: Confounder Z not identified")
    
    def test_mediator_detection(self):
        """Test detection of mediator variables"""
        # Create graph with mediator: X -> M -> Y
        G = nx.DiGraph()
        G.add_edges_from([
            ('X', 'M', {'weight': 0.8}),
            ('M', 'Y', {'weight': 0.7})
        ])
        
        declared_mediators = {'M'}
        result = self.analyzer.analyze_dimension(G, 'D1', declared_mediators)
        
        # REQUIRED: Mediator match
        self.assertTrue(result.mediator_match,
                       "FAILED: Mediator not detected")
        self.assertEqual(len(result.mediator_mismatch), 0,
                        "FAILED: Unexpected mediator mismatch")


class TestBayesianIntegrationConflictDetection(unittest.TestCase):
    """Test Bayesian integration detects evidence conflicts"""
    
    def setUp(self):
        """Initialize Bayesian integrator"""
        self.integrator = BayesianEvidenceIntegrator()
    
    def test_conflicting_evidence_detection(self):
        """Test that conflicting evidence is flagged"""
        # Create strongly conflicting evidence
        evidence = [
            StructuredEvidence(
                question_id="D1-Q1",
                dimension="D1",
                evidence_type="quantitative",
                raw_evidence={"score": 0.95},
                processed_content={"score": 0.95},
                confidence=0.95,
                source_module="monetary_detector"
            ),
            StructuredEvidence(
                question_id="D1-Q1",
                dimension="D1",
                evidence_type="qualitative",
                raw_evidence={"text": "No se identifican recursos"},
                processed_content={"score": 0.10},
                confidence=0.90,
                source_module="plan_processor"
            )
        ]
        
        result = self.integrator.integrate_evidence(evidence, "D1-Q1")
        
        # REQUIRED: Must detect conflict
        self.assertTrue(
            result['evidence_conflict_detected'],
            "FAILED: Did not detect evidence conflict"
        )
        # Wide credible interval or high variance expected
        ci_width = result['credible_interval_95'][1] - result['credible_interval_95'][0]
        self.assertGreater(
            ci_width, 0.4,
            f"FAILED: Credible interval too narrow ({ci_width:.3f}) for conflicting evidence"
        )
    
    def test_consistent_evidence_integration(self):
        """Test integration of consistent evidence"""
        # Create consistent evidence
        evidence = [
            StructuredEvidence(
                question_id="D2-Q5",
                dimension="D2",
                evidence_type="quantitative",
                raw_evidence={"score": 0.80},
                processed_content={"score": 0.80},
                confidence=0.85,
                source_module="causal_detector"
            ),
            StructuredEvidence(
                question_id="D2-Q5",
                dimension="D2",
                evidence_type="qualitative",
                raw_evidence={"present": True},
                processed_content={"score": 0.85},
                confidence=0.80,
                source_module="theory_checker"
            )
        ]
        
        result = self.integrator.integrate_evidence(evidence, "D2-Q5")
        
        # REQUIRED: No conflict for consistent evidence
        self.assertFalse(
            result['evidence_conflict_detected'],
            "FAILED: False conflict detection"
        )
        # Narrow credible interval expected
        ci_width = result['credible_interval_95'][1] - result['credible_interval_95'][0]
        self.assertLess(
            ci_width, 0.6,
            f"FAILED: Credible interval too wide ({ci_width:.3f}) for consistent evidence"
        )
    
    def test_credible_interval_generation(self):
        """Test that 95% credible intervals are generated"""
        evidence = [
            StructuredEvidence(
                question_id="D3-Q10",
                dimension="D3",
                evidence_type="quantitative",
                raw_evidence={"score": 0.70},
                processed_content={"score": 0.70},
                confidence=0.75,
                source_module="budget_analyzer"
            )
        ]
        
        result = self.integrator.integrate_evidence(evidence, "D3-Q10")
        
        # REQUIRED: Credible interval present
        self.assertIn('credible_interval_95', result,
                     "FAILED: No credible interval")
        ci = result['credible_interval_95']
        self.assertEqual(len(ci), 2, "FAILED: Invalid credible interval format")
        self.assertLess(ci[0], ci[1], "FAILED: Invalid interval bounds")
    
    def test_prior_usage_reporting(self):
        """Test that prior parameters are reported"""
        evidence = [
            StructuredEvidence(
                question_id="D4-Q1",
                dimension="D4",
                evidence_type="quantitative",
                raw_evidence={"score": 0.60},
                processed_content={"score": 0.60},
                confidence=0.70,
                source_module="outcome_tracker"
            )
        ]
        
        result = self.integrator.integrate_evidence(evidence, "D4-Q1")
        
        # REQUIRED: Prior documented
        self.assertIn('prior_used', result, "FAILED: Prior not documented")
        prior = result['prior_used']
        self.assertEqual(prior['alpha'], 2.0, "FAILED: Incorrect prior alpha")
        self.assertEqual(prior['beta'], 2.0, "FAILED: Incorrect prior beta")


class TestDimensionCoverageCompleteness(unittest.TestCase):
    """Test complete coverage of 300 questions across 6 dimensions"""
    
    def setUp(self):
        """Initialize mock registry and integrator"""
        self.registry = EvidenceRegistry()
        
        # Add sample evidence for testing
        for dim in range(1, 7):
            for q in range(1, 51):
                q_id = f"D{dim}-Q{q}"
                self.registry.register(
                    source_component="test_component",
                    evidence_type="test_evidence",
                    content={"score": 0.5},
                    confidence=0.7,
                    applicable_questions=[q_id]
                )
    
    def test_all_300_questions_mapped(self):
        """Test that all 300 questions are mapped"""
        extractor = DecalogoEvidenceExtractor(self.registry)
        
        # REQUIRED: 300 questions
        total_questions = len(extractor.mapping['questions'])
        self.assertEqual(
            total_questions, 300,
            f"FAILED: Only {total_questions}/300 questions mapped"
        )
    
    def test_all_6_dimensions_covered(self):
        """Test that all 6 dimensions are covered"""
        extractor = DecalogoEvidenceExtractor(self.registry)
        
        dimensions_covered = set()
        for q_id in extractor.mapping['questions'].keys():
            dim = q_id.split('-')[0]
            dimensions_covered.add(dim)
        
        required_dimensions = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}
        
        # REQUIRED: All 6 dimensions
        self.assertEqual(
            dimensions_covered, required_dimensions,
            f"FAILED: Missing dimensions {required_dimensions - dimensions_covered}"
        )
    
    def test_50_questions_per_dimension(self):
        """Test that each dimension has exactly 50 questions"""
        extractor = DecalogoEvidenceExtractor(self.registry)
        
        for dim in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
            dim_questions = [
                q for q in extractor.mapping['questions'].keys()
                if q.startswith(dim)
            ]
            
            # REQUIRED: 50 questions per dimension
            self.assertEqual(
                len(dim_questions), 50,
                f"FAILED: {dim} has {len(dim_questions)}/50 questions"
            )
    
    def test_complete_integration_execution(self):
        """Test that complete integration executes for all dimensions"""
        integrator = StrategicDecalogoIntegrator(
            evidence_registry=self.registry,
            documento_plan="Plan de prueba",
            nombre_plan="test_plan"
        )
        
        results = integrator.execute_complete_analysis()
        
        # REQUIRED: All dimensions analyzed
        dimensions_analyzed = set(results['dimensions'].keys())
        required_dimensions = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}
        
        self.assertEqual(
            dimensions_analyzed, required_dimensions,
            f"FAILED: Missing dimensions {required_dimensions - dimensions_analyzed}"
        )
        
        # REQUIRED: Each dimension has 50 questions
        for dim, dim_data in results['dimensions'].items():
            self.assertEqual(
                dim_data['questions_analyzed'], 50,
                f"FAILED: {dim} analyzed {dim_data['questions_analyzed']}/50 questions"
            )


class TestDeterministicExecution(unittest.TestCase):
    """Test deterministic execution across multiple runs"""
    
    def setUp(self):
        """Initialize registry with sample data"""
        self.registry = EvidenceRegistry()
        
        # Register deterministic evidence
        for i in range(10):
            self.registry.register(
                source_component="test_module",
                evidence_type="test_type",
                content={"score": 0.5 + i * 0.05},
                confidence=0.8,
                applicable_questions=[f"D1-Q{i+1}"]
            )
    
    def test_results_determinism(self):
        """Test that results are identical across runs"""
        integrator1 = StrategicDecalogoIntegrator(
            evidence_registry=self.registry,
            documento_plan="Test plan",
            nombre_plan="test"
        )
        
        integrator2 = StrategicDecalogoIntegrator(
            evidence_registry=self.registry,
            documento_plan="Test plan",
            nombre_plan="test"
        )
        
        # Run twice
        results1 = integrator1.execute_complete_analysis()
        results2 = integrator2.execute_complete_analysis()
        
        # Remove timestamps for comparison
        for r in [results1, results2]:
            del r['analysis_timestamp']
            for dim in r['dimensions'].values():
                for q_result in dim['question_evidence'].values():
                    if 'timestamp' in q_result:
                        del q_result['timestamp']
        
        # Compute hashes
        hash1 = hashlib.sha256(
            json.dumps(results1, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(results2, sort_keys=True).encode()
        ).hexdigest()
        
        # REQUIRED: Identical hashes
        self.assertEqual(
            hash1, hash2,
            "FAILED: Results not deterministic (hash mismatch)"
        )


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculation and validation"""
    
    def setUp(self):
        """Initialize registry with comprehensive test data"""
        self.registry = EvidenceRegistry()
        
        # Add evidence for all 300 questions
        for dim in range(1, 7):
            for q in range(1, 51):
                q_id = f"D{dim}-Q{q}"
                # Vary confidence and quality
                confidence = 0.6 + (q % 10) * 0.04
                self.registry.register(
                    source_component=f"module_{dim}",
                    evidence_type="metric_test",
                    content={"score": 0.5 + (q % 20) * 0.025},
                    confidence=confidence,
                    applicable_questions=[q_id]
                )
    
    def test_metrics_calculation(self):
        """Test that all required metrics are calculated"""
        integrator = StrategicDecalogoIntegrator(
            evidence_registry=self.registry,
            documento_plan="Test plan",
            nombre_plan="metrics_test"
        )
        
        results = integrator.execute_complete_analysis()
        metrics = results['performance_metrics']
        
        # REQUIRED: All metrics present
        required_metrics = [
            'total_questions',
            'questions_with_evidence',
            'dimensions_fully_analyzed',
            'avg_evidence_confidence',
            'pct_questions_high_confidence',
            'pct_evidence_conflicts',
            'elapsed_time_seconds'
        ]
        
        for metric in required_metrics:
            self.assertIn(
                metric, metrics,
                f"FAILED: Missing metric {metric}"
            )
    
    def test_quality_gates_validation(self):
        """Test quality gates validation"""
        integrator = StrategicDecalogoIntegrator(
            evidence_registry=self.registry,
            documento_plan="Test plan",
            nombre_plan="gates_test"
        )
        
        results = integrator.execute_complete_analysis()
        gates = results['quality_gates_passed']
        
        # REQUIRED: All gates evaluated
        required_gates = [
            'all_300_questions_mapped',
            'all_6_dimensions_analyzed',
            'min_evidence_coverage',
            'avg_confidence_acceptable',
            'low_conflict_rate'
        ]
        
        for gate in required_gates:
            self.assertIn(
                gate, gates,
                f"FAILED: Missing quality gate {gate}"
            )
    
    def test_coverage_metrics_accuracy(self):
        """Test accuracy of coverage metrics"""
        integrator = StrategicDecalogoIntegrator(
            evidence_registry=self.registry,
            documento_plan="Test plan",
            nombre_plan="coverage_test"
        )
        
        results = integrator.execute_complete_analysis()
        metrics = results['performance_metrics']
        
        # REQUIRED: Accurate counts
        self.assertEqual(
            metrics['total_questions'], 300,
            "FAILED: Incorrect total questions count"
        )
        self.assertEqual(
            metrics['dimensions_fully_analyzed'], 6,
            "FAILED: Incorrect dimensions count"
        )
        # With our setup, all questions should have evidence
        self.assertEqual(
            metrics['questions_with_evidence'], 300,
            "FAILED: Incorrect evidence coverage"
        )


class TestAntiPatternDetection(unittest.TestCase):
    """Test detection of anti-patterns and prohibited implementations"""
    
    def test_no_magic_numbers(self):
        """Test that thresholds are documented (not magic numbers)"""
        extractor = SemanticExtractor()
        
        # REQUIRED: Threshold is documented constant
        self.assertEqual(
            extractor.threshold, 0.75,
            "FAILED: Threshold not set to BEIR-validated 0.75"
        )
    
    def test_no_uniform_prior(self):
        """Test that Bayesian integrator doesn't use uniform prior"""
        integrator = BayesianEvidenceIntegrator()
        
        # REQUIRED: Not Beta(1,1)
        self.assertNotEqual(
            (integrator.prior_alpha, integrator.prior_beta), (1.0, 1.0),
            "FAILED: Using uniform prior Beta(1,1) instead of Jeffreys prior"
        )
        
        # REQUIRED: Uses Beta(2,2) Jeffreys prior
        self.assertEqual(
            (integrator.prior_alpha, integrator.prior_beta), (2.0, 2.0),
            "FAILED: Not using Jeffreys prior Beta(2,2)"
        )
    
    def test_no_placeholder_implementations(self):
        """Test that implementations are complete (no placeholders)"""
        # Test semantic extractor
        extractor = SemanticExtractor()
        results = extractor.extract_evidence(
            "test query",
            ["relevant segment about test query"],
            top_k=5
        )
        
        # Should return actual results, not placeholder
        self.assertIsInstance(results, list,
                            "FAILED: Placeholder implementation")
        
        # Test causal analyzer
        analyzer = CausalGraphAnalyzer()
        G = nx.DiGraph()
        G.add_edge('A', 'B')
        result = analyzer.analyze_dimension(G, 'D1')
        
        # Should have actual analysis
        self.assertIsInstance(result.acyclicity_pvalue, float,
                            "FAILED: Placeholder implementation")
        self.assertGreaterEqual(result.acyclicity_pvalue, 0.0,
                               "FAILED: Invalid p-value")


# ============================================================================
# TEST SUITE RUNNER
# ============================================================================

def run_all_tests():
    """Run complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticExtractionThresholdEnforcement))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalGraphCycleDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianIntegrationConflictDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDimensionCoverageCompleteness))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicExecution))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestAntiPatternDetection))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
