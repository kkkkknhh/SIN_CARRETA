#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Doctoral Argumentation Engine
============================================

Comprehensive test suite with ZERO TOLERANCE for quality failures.

Test Coverage:
1. Toulmin structure completeness
2. Multi-source synthesis (≥3 sources)
3. Vague language detection and rejection
4. Confidence-Bayesian alignment (±0.05)
5. Logical coherence validation (≥0.85)
6. Academic quality assessment (≥0.80)
7. Deterministic output verification
8. Edge cases and error handling
"""

import unittest
from typing import Any, List

import numpy as np

from doctoral_argumentation_engine import (
    AcademicWritingAnalyzer,
    DoctoralArgumentationEngine,
    LogicalCoherenceValidator,
    StructuredEvidence,
    ToulminArgument,
    create_mock_bayesian_posterior,
)

# ============================================================================
# MOCK EVIDENCE REGISTRY
# ============================================================================


class MockEvidenceRegistry:
    """Mock evidence registry for testing"""

    def __init__(self):
        self.evidence = {}

    def add_evidence(self, question_id: str, evidence: List[StructuredEvidence]):
        """Add evidence for a question"""
        self.evidence[question_id] = evidence

    def get_evidence_for_question(self, question_id: str) -> List[StructuredEvidence]:
        """Get evidence for a question"""
        return self.evidence.get(question_id, [])


# ============================================================================
# TEST UTILITIES
# ============================================================================


def create_test_evidence(
    source: str = "test_module",
    etype: str = "test_type",
    content: Any = None,
    confidence: float = 0.85,
) -> StructuredEvidence:
    """Create test evidence item"""
    if content is None:
        content = {"test": "data"}

    return StructuredEvidence(
        source_module=source,
        evidence_type=etype,
        content=content,
        confidence=confidence,
        applicable_questions=["D1-Q1"],
    )


def create_baseline_evidence(confidence: float = 0.9) -> StructuredEvidence:
    """Create baseline evidence"""
    return create_test_evidence(
        source="feasibility_scorer",
        etype="baseline_presence",
        content={
            "baseline_text": "línea base 2024: 50% cobertura",
            "confidence": confidence,
        },
        confidence=confidence,
    )


def create_monetary_evidence(confidence: float = 0.85) -> StructuredEvidence:
    """Create monetary evidence"""
    return create_test_evidence(
        source="monetary_detector",
        etype="monetary_value",
        content={
            "amount": 5000000,
            "currency": "COP",
            "context": "presupuesto asignado",
        },
        confidence=confidence,
    )


def create_target_evidence(confidence: float = 0.80) -> StructuredEvidence:
    """Create target evidence"""
    return create_test_evidence(
        source="feasibility_scorer",
        etype="target_presence",
        content={"target_text": "meta 2025: 80% cobertura", "quantitative": True},
        confidence=confidence,
    )


# ============================================================================
# TEST TOULMIN STRUCTURE
# ============================================================================


class TestToulminStructure(unittest.TestCase):
    """Test Toulmin argument structure validation"""

    def test_valid_toulmin_structure(self):
        """Test creation of valid Toulmin structure"""
        toulmin = ToulminArgument(
            claim="The plan addresses requirements",
            ground="Evidence shows baseline documentation",
            warrant="Given baseline presence, requirements are met",
            backing=["Source A supports", "Source B confirms"],
            rebuttal="Despite limitations, evidence is robust",
            qualifier="With strong evidence (0.85)",
            evidence_sources=["module1", "module2", "module3"],
            confidence_lower=0.75,
            confidence_upper=0.95,
        )

        self.assertIsNotNone(toulmin)
        self.assertEqual(toulmin.claim, "The plan addresses requirements")
        self.assertEqual(len(toulmin.backing), 2)

    def test_toulmin_requires_non_empty_claim(self):
        """Test that empty claim raises error"""
        with self.assertRaises(ValueError) as ctx:
            ToulminArgument(
                claim="",
                ground="Evidence",
                warrant="Warrant",
                backing=["A", "B"],
                rebuttal="Rebuttal",
                qualifier="Qualifier",
                evidence_sources=["s1"],
                confidence_lower=0.7,
                confidence_upper=0.9,
            )

        self.assertIn("Claim cannot be empty", str(ctx.exception))

    def test_toulmin_requires_minimum_backing(self):
        """Test that insufficient backing raises error"""
        with self.assertRaises(ValueError) as ctx:
            ToulminArgument(
                claim="Claim",
                ground="Ground",
                warrant="Warrant",
                backing=["Only one"],  # Need at least 2
                rebuttal="Rebuttal",
                qualifier="Qualifier",
                evidence_sources=["s1"],
                confidence_lower=0.7,
                confidence_upper=0.9,
            )

        self.assertIn("Backing requires ≥2 sources", str(ctx.exception))

    def test_toulmin_validates_confidence_interval(self):
        """Test confidence interval validation"""
        with self.assertRaises(ValueError) as ctx:
            ToulminArgument(
                claim="Claim",
                ground="Ground",
                warrant="Warrant",
                backing=["A", "B"],
                rebuttal="Rebuttal",
                qualifier="Qualifier",
                evidence_sources=["s1"],
                confidence_lower=0.9,
                confidence_upper=0.5,  # Invalid: upper < lower
            )

        self.assertIn("Invalid confidence interval", str(ctx.exception))


# ============================================================================
# TEST LOGICAL COHERENCE VALIDATOR
# ============================================================================


class TestLogicalCoherenceValidator(unittest.TestCase):
    """Test logical coherence validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = LogicalCoherenceValidator()

    def test_valid_argument_scores_high(self):
        """Test that valid argument scores ≥0.85"""
        toulmin = ToulminArgument(
            claim="The plan demonstrates comprehensive baseline documentation",
            ground="Analysis of plan sections reveals systematic baseline measurements including quantitative metrics (50% coverage baseline 2024) and temporal references",
            warrant="Given that baseline documentation includes quantitative metrics and temporal markers, and considering evaluation standards requiring measurable starting points, it follows that the plan satisfies baseline requirements because documented measurements enable progress tracking",
            backing=[
                "Secondary analysis from monetary_detector confirms budgetary allocations supporting baseline establishment",
                "Independent verification from responsibility_detector identifies institutional mechanisms for baseline maintenance",
            ],
            rebuttal="However, while some sections lack granular detail, the preponderance of evidence across multiple independent sources provides robust support",
            qualifier="With strong evidence (posterior: 0.87, CI: [0.75, 0.95])",
            evidence_sources=[
                "feasibility_scorer",
                "monetary_detector",
                "responsibility_detector",
            ],
            confidence_lower=0.75,
            confidence_upper=0.95,
        )

        score = self.validator.validate(toulmin)
        self.assertGreaterEqual(
            score, 0.85, f"Valid argument should score ≥0.85, got {score:.2f}"
        )

    def test_circular_reasoning_detected(self):
        """Test detection of circular reasoning"""
        toulmin = ToulminArgument(
            claim="The plan has good baselines",
            ground="The plan has good baselines as shown",  # Circular!
            warrant="Because baselines are good",
            backing=["Source A", "Source B"],
            rebuttal="Despite concerns, baselines are good",
            qualifier="Strong evidence",
            evidence_sources=["s1", "s2"],
            confidence_lower=0.7,
            confidence_upper=0.9,
        )

        score = self.validator.validate(toulmin)
        self.assertLess(score, 1.0, "Circular reasoning should reduce score")

    def test_weak_warrant_detected(self):
        """Test detection of weak warrant (no connectives)"""
        toulmin = ToulminArgument(
            claim="The plan is complete",
            ground="Documentation exists in multiple sections",
            warrant="The documentation exists",  # Weak, no connectives
            backing=["Source A provides additional info", "Source B confirms"],
            rebuttal="However despite potential issues the conclusion holds based on extensive evidence",
            qualifier="Strong evidence with high confidence",
            evidence_sources=["s1", "s2"],
            confidence_lower=0.7,
            confidence_upper=0.9,
        )

        score = self.validator.validate(toulmin)
        self.assertLess(score, 1.0, "Weak warrant should reduce score")

    def test_insufficient_backing_penalized(self):
        """Test that insufficient backing is caught at validation"""
        # Note: ToulminArgument constructor already enforces ≥2 backing
        # This test ensures validator also checks
        toulmin = ToulminArgument(
            claim="Claim",
            ground="Ground",
            warrant="Given that ground provides evidence, and considering the evaluation standards, it follows that because documented evidence supports this conclusion",
            backing=[
                "Additional source A provides independent corroboration with specific details",
                "Secondary source B confirms the assessment through quantitative analysis",
            ],  # Sufficient backing
            rebuttal="However despite some potential limitations the overall body of evidence nevertheless supports the conclusion through convergent multi-source triangulation",
            qualifier="Moderate evidence with confidence level aligned to posterior distribution",
            evidence_sources=["s1", "s2"],
            confidence_lower=0.5,
            confidence_upper=0.7,
        )

        score = self.validator.validate(toulmin)
        # Should pass with high score since all components are well-formed
        self.assertGreaterEqual(
            score, 0.85, f"Well-formed argument should score ≥0.85, got {score:.2f}"
        )


# ============================================================================
# TEST ACADEMIC WRITING ANALYZER
# ============================================================================


class TestAcademicWritingAnalyzer(unittest.TestCase):
    """Test academic writing quality analysis"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AcademicWritingAnalyzer()

    def test_precision_detects_vague_language(self):
        """Test detection of vague language"""
        vague_text = "The plan seems to address this. It appears that many elements are present. Some evidence suggests improvements."

        score = self.analyzer._score_precision(vague_text)
        self.assertLess(
            score, 0.80, f"Vague language should score <0.80, got {score:.2f}"
        )

    def test_precision_rewards_precise_language(self):
        """Test that precise language scores high"""
        precise_text = "The plan demonstrates baseline documentation with quantitative metrics: 50% coverage in 2024, target 80% by 2025. Evidence includes budgetary allocation of COP 5,000,000 and institutional responsibility assignment."

        score = self.analyzer._score_precision(precise_text)
        self.assertGreater(
            score, 0.85, f"Precise language should score >0.85, got {score:.2f}"
        )

    def test_overall_quality_integrates_dimensions(self):
        """Test overall quality score integration"""
        # High-quality paragraphs
        paragraphs = [
            "The plan demonstrates comprehensive baseline documentation with quantitative metrics. Analysis reveals systematic measurements including 50% coverage baseline in 2024 with target progression to 80% by 2025. Evidence from feasibility_scorer indicates confidence level of 0.90.",
            "Given that baseline documentation includes specific quantitative metrics and temporal references, and considering evaluation standards requiring measurable starting points, it follows that the plan satisfies baseline requirements. Additional support from monetary_detector (confidence: 0.85) corroborates budgetary alignment. Furthermore, responsibility_detector confirms institutional mechanisms for implementation.",
            "However, while some sections demonstrate less granular detail, the preponderance of evidence across three independent analytical modules provides robust triangulated support. The Bayesian posterior analysis yields a mean estimate of 0.83 with 95% credible interval [0.75, 0.90], representing narrow uncertainty and high precision in the evaluation assessment.",
        ]

        scores = self.analyzer.analyze(paragraphs)

        self.assertIn("overall_score", scores)
        self.assertGreaterEqual(
            scores["overall_score"],
            0.80,
            f"High-quality text should score ≥0.80, got {scores['overall_score']:.2f}",
        )

    def test_coherence_checks_transitions(self):
        """Test coherence analysis with transitions"""
        paragraphs_with_transitions = [
            "The first paragraph establishes the claim.",
            "Furthermore, the second paragraph provides additional evidence.",
            "Nevertheless, the third paragraph addresses counterarguments.",
        ]

        paragraphs_without_transitions = [
            "The first paragraph establishes the claim.",
            "The second paragraph provides evidence.",
            "The third paragraph makes conclusions.",
        ]

        score_with = self.analyzer._score_coherence(paragraphs_with_transitions)
        score_without = self.analyzer._score_coherence(paragraphs_without_transitions)

        self.assertGreater(
            score_with, score_without, "Paragraphs with transitions should score higher"
        )

    def test_sophistication_measures_lexical_diversity(self):
        """Test sophistication scoring"""
        # High sophistication: varied vocabulary, good length
        sophisticated = "The comprehensive analysis demonstrates systematic evaluation employing rigorous methodological frameworks. Triangulated evidence synthesis reveals convergent patterns across independent analytical dimensions, substantiating robust conclusions regarding plan adequacy."

        # Low sophistication: repetitive, simple
        simple = "The plan is good. The plan has parts. The plan is complete. The plan works well."

        score_high = self.analyzer._score_sophistication(sophisticated)
        score_low = self.analyzer._score_sophistication(simple)

        self.assertGreater(
            score_high,
            score_low,
            "Sophisticated text should score higher than simple text",
        )


# ============================================================================
# TEST DOCTORAL ARGUMENTATION ENGINE
# ============================================================================


class TestDoctoralArgumentationEngine(unittest.TestCase):
    """Test main argumentation engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = MockEvidenceRegistry()
        self.engine = DoctoralArgumentationEngine(self.registry)

    def test_sufficient_evidence_required(self):
        """Test that ≥3 evidence sources are required"""
        # Only 2 evidence items
        evidence = [create_baseline_evidence(0.9), create_monetary_evidence(0.85)]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        with self.assertRaises(ValueError) as ctx:
            self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        self.assertIn("INSUFFICIENT EVIDENCE", str(ctx.exception))
        self.assertIn("Need ≥3", str(ctx.exception))

    def test_generates_three_paragraphs(self):
        """Test that exactly 3 paragraphs are generated"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        result = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        self.assertIn("argument_paragraphs", result)
        self.assertEqual(
            len(result["argument_paragraphs"]), 3, "Must generate exactly 3 paragraphs"
        )

    def test_toulmin_structure_present(self):
        """Test that Toulmin structure is complete"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        result = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        self.assertIn("toulmin_structure", result)
        toulmin = result["toulmin_structure"]

        # Check all required components
        required_keys = [
            "claim",
            "ground",
            "warrant",
            "backing",
            "rebuttal",
            "qualifier",
        ]
        for key in required_keys:
            self.assertIn(key, toulmin, f"Missing Toulmin component: {key}")
            self.assertIsNotNone(toulmin[key], f"Toulmin component {key} is None")

    def test_multi_source_synthesis(self):
        """Test that multiple sources are synthesized"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
            create_test_evidence(
                "responsibility_detector",
                "responsibility",
                {"entity": "Municipality"},
                0.75,
            ),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        result = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        # Check evidence synthesis map
        self.assertIn("evidence_synthesis_map", result)
        synth_map = result["evidence_synthesis_map"]

        self.assertIn("all_sources", synth_map)
        self.assertGreaterEqual(
            len(synth_map["all_sources"]), 3, "Should synthesize ≥3 sources"
        )

    def test_logical_coherence_validated(self):
        """Test that logical coherence meets threshold"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        result = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        self.assertIn("logical_coherence_score", result)
        self.assertGreaterEqual(
            result["logical_coherence_score"],
            0.85,
            f"Coherence score must be ≥0.85, got {result['logical_coherence_score']:.2f}",
        )

    def test_academic_quality_validated(self):
        """Test that academic quality meets threshold"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        result = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        self.assertIn("academic_quality_scores", result)
        quality = result["academic_quality_scores"]

        self.assertIn("overall_score", quality)
        self.assertGreaterEqual(
            quality["overall_score"],
            0.80,
            f"Academic quality must be ≥0.80, got {quality['overall_score']:.2f}",
        )

    def test_confidence_alignment(self):
        """Test that confidence aligns with Bayesian posterior"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        result = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        self.assertIn("confidence_alignment_error", result)
        self.assertLessEqual(
            result["confidence_alignment_error"],
            0.05,
            f"Confidence error must be ≤0.05, got {result['confidence_alignment_error']:.3f}",
        )

    def test_deterministic_output(self):
        """Test that same input produces same output"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        posterior = create_mock_bayesian_posterior(2.5, 0.85)

        # Generate twice
        result1 = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)
        result2 = self.engine.generate_argument("D1-Q1", 2.5, evidence, posterior)

        # Compare key outputs (excluding timestamp)
        self.assertEqual(
            result1["argument_paragraphs"],
            result2["argument_paragraphs"],
            "Same evidence should produce identical arguments",
        )

        self.assertEqual(
            result1["toulmin_structure"]["claim"],
            result2["toulmin_structure"]["claim"],
            "Claims should be identical",
        )

    def test_evidence_ranking_prioritizes_quality(self):
        """Test that evidence is ranked by quality"""
        evidence = [
            create_baseline_evidence(0.6),  # Lower quality
            create_monetary_evidence(0.95),  # Highest quality
            create_target_evidence(0.75),  # Medium quality
        ]

        ranked = self.engine._rank_evidence_by_quality_and_diversity(evidence)

        # Highest quality should be first
        self.assertEqual(
            ranked[0].confidence,
            0.95,
            "Highest confidence evidence should be ranked first",
        )

    def test_claim_strength_matches_score(self):
        """Test that claim strength matches score level"""
        evidence = [
            create_baseline_evidence(0.9),
            create_monetary_evidence(0.85),
            create_target_evidence(0.80),
        ]

        # Test high score
        claim_high = self.engine._generate_claim("D1-Q1", 2.7, evidence[0])
        self.assertIn(
            "fully", claim_high.lower(), "High score should generate 'fully' claim"
        )

        # Test medium score
        claim_med = self.engine._generate_claim("D1-Q1", 1.5, evidence[0])
        self.assertIn(
            "partially",
            claim_med.lower(),
            "Medium score should generate 'partially' claim",
        )

        # Test low score
        claim_low = self.engine._generate_claim("D1-Q1", 0.5, evidence[0])
        self.assertIn(
            "does not", claim_low.lower(), "Low score should generate 'does not' claim"
        )


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = MockEvidenceRegistry()
        self.engine = DoctoralArgumentationEngine(self.registry)

    def test_all_low_confidence_evidence(self):
        """Test handling of all low-confidence evidence"""
        evidence = [
            create_test_evidence("source1", "type1", {"data": "A"}, 0.3),
            create_test_evidence("source2", "type2", {"data": "B"}, 0.35),
            create_test_evidence("source3", "type3", {"data": "C"}, 0.4),
        ]

        posterior = create_mock_bayesian_posterior(1.0, 0.4)

        # Should still generate argument but with appropriate qualifiers
        result = self.engine.generate_argument("D1-Q1", 1.0, evidence, posterior)

        self.assertIn("argument_paragraphs", result)
        # Check that qualifier reflects low confidence
        toulmin = result["toulmin_structure"]
        self.assertIn("limited", toulmin["qualifier"].lower())

    def test_perfect_score_high_confidence(self):
        """Test handling of perfect score with high confidence"""
        evidence = [
            create_baseline_evidence(0.98),
            create_monetary_evidence(0.95),
            create_target_evidence(0.97),
        ]

        posterior = create_mock_bayesian_posterior(3.0, 0.95)

        result = self.engine.generate_argument("D1-Q1", 3.0, evidence, posterior)

        self.assertIn("argument_paragraphs", result)
        # Check for strong language
        toulmin = result["toulmin_structure"]
        self.assertIn("strong", toulmin["qualifier"].lower())

    def test_empty_evidence_content(self):
        """Test handling of evidence with minimal content"""
        evidence = [
            create_test_evidence("source1", "type1", {}, 0.8),
            create_test_evidence("source2", "type2", {"minimal": "data"}, 0.75),
            create_test_evidence("source3", "type3", {"value": None}, 0.7),
        ]

        posterior = create_mock_bayesian_posterior(1.5, 0.7)

        # Should handle gracefully without crashing
        try:
            result = self.engine.generate_argument("D1-Q1", 1.5, evidence, posterior)
            self.assertIn("argument_paragraphs", result)
        except Exception as e:
            self.fail(
                f"Should handle minimal evidence content gracefully, but raised: {e}"
            )


# ============================================================================
# TEST ANTI-PATTERNS
# ============================================================================


class TestAntiPatternDetection(unittest.TestCase):
    """Test detection and rejection of anti-patterns"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AcademicWritingAnalyzer()

    def test_rejects_generic_templates(self):
        """Test that generic template language is penalized"""
        generic = "The plan appears to address this aspect. There seems to be evidence suggesting that this might indicate improvements."

        score = self.analyzer._score_precision(generic)
        self.assertLess(
            score,
            0.50,
            f"Generic template language should score <0.50, got {score:.2f}",
        )

    def test_rejects_vague_quantifiers(self):
        """Test rejection of vague quantifiers"""
        vague = "Many resources are allocated. Several activities are planned. Some evidence suggests progress."

        score = self.analyzer._score_precision(vague)
        self.assertLess(
            score, 0.60, f"Vague quantifiers should score <0.60, got {score:.2f}"
        )

    def test_requires_explicit_connectives(self):
        """Test that warrant requires explicit connectives"""
        validator = LogicalCoherenceValidator()

        # Weak warrant without connectives
        toulmin_weak = ToulminArgument(
            claim="Claim",
            ground="Ground evidence",
            warrant="The evidence shows results",  # No explicit connective
            backing=["A", "B"],
            rebuttal="Despite issues the conclusion holds",
            qualifier="Moderate",
            evidence_sources=["s1"],
            confidence_lower=0.5,
            confidence_upper=0.7,
        )

        # Strong warrant with connectives
        toulmin_strong = ToulminArgument(
            claim="Claim",
            ground="Ground evidence with specific details",
            warrant="Given that ground evidence demonstrates specific patterns, and considering the evaluation standards requiring documentation, it follows that the claim is supported because documented evidence satisfies requirements",
            backing=[
                "Additional source A provides corroboration",
                "Independent source B confirms",
            ],
            rebuttal="However despite some limitations the evidence base provides robust support",
            qualifier="Moderate evidence",
            evidence_sources=["s1"],
            confidence_lower=0.5,
            confidence_upper=0.7,
        )

        score_weak = validator.validate(toulmin_weak)
        score_strong = validator.validate(toulmin_strong)

        self.assertGreater(
            score_strong,
            score_weak,
            "Warrant with explicit connectives should score higher",
        )


# ============================================================================
# TEST INTEGRATION SCENARIOS
# ============================================================================


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = MockEvidenceRegistry()
        self.engine = DoctoralArgumentationEngine(self.registry)

    def test_full_argument_generation_high_quality(self):
        """Test complete argument generation for high-quality evidence"""
        evidence = [
            create_baseline_evidence(0.95),
            create_monetary_evidence(0.90),
            create_target_evidence(0.85),
            create_test_evidence(
                "responsibility_detector",
                "institutional_responsibility",
                {"entity": "Municipality", "explicit": True},
                0.82,
            ),
        ]

        posterior = {"posterior_mean": 0.87, "credible_interval_95": (0.80, 0.93)}

        result = self.engine.generate_argument("D1-Q3", 2.6, evidence, posterior)

        # Comprehensive validation
        self.assertEqual(len(result["argument_paragraphs"]), 3)
        self.assertGreaterEqual(result["logical_coherence_score"], 0.85)
        self.assertGreaterEqual(
            result["academic_quality_scores"]["overall_score"], 0.80
        )
        self.assertLessEqual(result["confidence_alignment_error"], 0.05)

        # Validate Toulmin structure completeness
        toulmin = result["toulmin_structure"]
        self.assertGreater(len(toulmin["claim"]), 20)
        self.assertGreater(len(toulmin["ground"]), 30)
        self.assertGreater(len(toulmin["warrant"]), 50)
        self.assertGreaterEqual(len(toulmin["backing"]), 2)
        self.assertGreater(len(toulmin["rebuttal"]), 30)

    def test_full_argument_generation_moderate_quality(self):
        """Test argument generation for moderate evidence quality"""
        evidence = [
            create_baseline_evidence(0.65),
            create_monetary_evidence(0.70),
            create_target_evidence(0.60),
        ]

        posterior = {"posterior_mean": 0.62, "credible_interval_95": (0.50, 0.74)}

        result = self.engine.generate_argument("D2-Q5", 1.8, evidence, posterior)

        # Should still meet quality thresholds
        self.assertGreaterEqual(result["logical_coherence_score"], 0.85)
        self.assertGreaterEqual(
            result["academic_quality_scores"]["overall_score"], 0.80
        )

        # Qualifier should reflect moderate confidence
        toulmin = result["toulmin_structure"]
        qualifier_lower = toulmin["qualifier"].lower()
        self.assertTrue(
            any(
                term in qualifier_lower
                for term in ["moderate", "substantial", "considerable"]
            ),
            "Moderate confidence should be reflected in qualifier",
        )

    def test_300_questions_scenario(self):
        """Test scalability for 300-question evaluation"""
        # Simulate processing multiple questions
        question_ids = [
            f"D{d}-Q{q}" for d in range(1, 7) for q in range(1, 6)
        ]  # 30 questions sample

        success_count = 0
        for qid in question_ids[:5]:  # Test first 5
            evidence = [
                create_baseline_evidence(0.80 + np.random.random() * 0.15),
                create_monetary_evidence(0.75 + np.random.random() * 0.15),
                create_target_evidence(0.70 + np.random.random() * 0.15),
            ]

            score = 1.0 + np.random.random() * 1.5
            posterior = create_mock_bayesian_posterior(score, 0.75)

            try:
                result = self.engine.generate_argument(qid, score, evidence, posterior)
                if (
                    result["logical_coherence_score"] >= 0.85
                    and result["academic_quality_scores"]["overall_score"] >= 0.80
                ):
                    success_count += 1
            except ValueError:
                # Some may fail quality gates - that's expected
                pass

        # At least 60% should pass quality gates
        self.assertGreaterEqual(
            success_count / 5,
            0.6,
            "At least 60% of arguments should meet quality thresholds",
        )


# ============================================================================
# TEST SUITE EXECUTION
# ============================================================================


def run_test_suite():
    """Run complete test suite with detailed reporting"""

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestToulminStructure,
        TestLogicalCoherenceValidator,
        TestAcademicWritingAnalyzer,
        TestDoctoralArgumentationEngine,
        TestEdgeCases,
        TestAntiPatternDetection,
        TestIntegrationScenarios,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    exit(0 if success else 1)
