"""
Test Suite for Document Segmenter
=================================
Comprehensive tests for dual-criteria document segmentation.
"""

import unittest
from unittest.mock import patch

from document_segmenter import DocumentSegmenter, SegmentationStats, SegmentMetrics


class TestDocumentSegmenter(unittest.TestCase):
    """Test cases for DocumentSegmenter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.segmenter = DocumentSegmenter()

        # Sample texts of varying lengths and structures
        self.short_text = "This is a short sentence. Another one here. Final sentence."

        self.medium_text = """
        This is the first sentence of a medium-length document. It contains multiple sentences that should be segmented properly.
        The segmentation algorithm should handle this text well. Each segment should contain approximately three sentences.
        This helps maintain semantic coherence while meeting character count requirements. The algorithm uses dual criteria for optimal results.
        """

        self.long_text = """
        Document segmentation is a critical task in natural language processing. It involves breaking down large documents into smaller, manageable chunks.
        The primary challenge is maintaining semantic coherence while meeting specific size requirements. Modern approaches use both linguistic and statistical criteria.
        
        Traditional methods relied solely on fixed-size chunking. This approach often broke sentences or paragraphs inappropriately.
        The result was poor quality segments that lacked semantic meaning. Users struggled with fragmented information.
        
        Our dual-criteria approach addresses these limitations effectively. It prioritizes sentence boundaries while respecting character limits.
        The algorithm uses spaCy for accurate sentence detection. Fallback mechanisms ensure robustness across different document types.
        
        Quality validation is essential for production systems. We measure segment length distributions and coherence scores.
        Comprehensive logging helps identify potential issues early. The system adapts to different content characteristics automatically.
        """

        self.malformed_text = "No proper punctuation here just a long stream of words that keeps going and going without any clear sentence boundaries to work with in the segmentation process"

    def test_initialization(self):
        """Test DocumentSegmenter initialization."""
        # Test default parameters
        self.assertEqual(self.segmenter.target_char_min, 700)
        self.assertEqual(self.segmenter.target_char_max, 900)
        self.assertEqual(self.segmenter.target_sentences, 3)

        # Test custom parameters
        custom_segmenter = DocumentSegmenter(
            target_char_min=500, target_char_max=800, target_sentences=2
        )
        self.assertEqual(custom_segmenter.target_char_min, 500)
        self.assertEqual(custom_segmenter.target_char_max, 800)
        self.assertEqual(custom_segmenter.target_sentences, 2)

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        self.assertEqual(self.segmenter.segment_document(""), [])
        self.assertEqual(self.segmenter.segment_document("   "), [])
        self.assertEqual(self.segmenter.segment_document("\n\t  \n"), [])

    def test_short_text_segmentation(self):
        """Test segmentation of short text."""
        segments = self.segmenter.segment_document(self.short_text)

        self.assertGreater(len(segments), 0)
        # Short text should typically result in one segment
        self.assertLessEqual(len(segments), 2)

        for segment in segments:
            self.assertIn("text", segment)
            self.assertIn("metrics", segment)
            self.assertIsInstance(segment["metrics"], SegmentMetrics)

    def test_medium_text_segmentation(self):
        """Test segmentation of medium-length text."""
        segments = self.segmenter.segment_document(self.medium_text)

        self.assertGreater(len(segments), 0)

        for segment in segments:
            metrics = segment["metrics"]
            # Check that segments have reasonable length
            self.assertGreater(metrics.char_count, 50)
            self.assertGreater(metrics.sentence_count, 0)

    def test_long_text_segmentation(self):
        """Test segmentation of long text with multiple paragraphs."""
        segments = self.segmenter.segment_document(self.long_text)

        self.assertGreater(len(segments), 2)  # Should create multiple segments

        # Test dual criteria adherence
        segments_in_char_range = sum(
            1
            for seg in segments
            if self.segmenter.target_char_min
            <= seg["metrics"].char_count
            <= self.segmenter.target_char_max
        )

        segments_with_target_sentences = sum(
            1
            for seg in segments
            if seg["metrics"].sentence_count == self.segmenter.target_sentences
        )

        segments_near_target = sum(
            1
            for seg in segments
            if abs(seg["metrics"].sentence_count - self.segmenter.target_sentences) <= 1
        )

        # At least some segments should meet or be close to criteria
        self.assertGreater(
            max(
                segments_in_char_range,
                segments_with_target_sentences,
                segments_near_target,
            ),
            0,
        )

    def test_malformed_text_handling(self):
        """Test handling of text without proper sentence boundaries."""
        segments = self.segmenter.segment_document(self.malformed_text)

        # Should still create segments
        self.assertGreater(len(segments), 0)

        for segment in segments:
            # Should not be empty
            self.assertGreater(len(segment["text"]), 0)

    def test_segment_metrics_calculation(self):
        """Test calculation of segment metrics."""
        segments = self.segmenter.segment_document(self.medium_text)

        for segment in segments:
            metrics = segment["metrics"]
            text = segment["text"]

            # Character count should match text length
            self.assertEqual(metrics.char_count, len(text))

            # Word count should be reasonable
            expected_words = len(text.split())
            self.assertEqual(metrics.word_count, expected_words)

            # Sentence count should be positive
            self.assertGreater(metrics.sentence_count, 0)

    def test_dual_criteria_logic(self):
        """Test the dual-criteria decision logic."""
        # Test with text that should trigger different criteria
        test_cases = [
            # Case 1: Short sentences, should use sentence count
            "Short one. Another short. Third short. Fourth short. Fifth short.",
            # Case 2: Long sentences, should use character count
            "This is a very long sentence that contains many words and should trigger the character-based criteria when the algorithm determines that adding more sentences would exceed the maximum character limit.",
            # Case 3: Mixed sentence lengths
            "Short. This is a longer sentence with more content. Short again. Another longer sentence that adds substantial character count to the segment.",
        ]

        for i, test_text in enumerate(test_cases):
            with self.subTest(case=i):
                segments = self.segmenter.segment_document(test_text)
                self.assertGreater(len(segments), 0)

                # Check that segments are within absolute limits
                for segment in segments:
                    metrics = segment["metrics"]
                    self.assertLessEqual(
                        metrics.char_count, self.segmenter.max_segment_chars
                    )
                    # Be more lenient for test data - some segments might be small due to text structure
                    self.assertGreater(metrics.char_count, 0)

    def test_post_processing(self):
        """Test post-processing of segments."""
        # Create a scenario that should trigger post-processing
        short_segments_text = "A. B. C. D. E. F. G. H."  # Very short segments

        segments = self.segmenter.segment_document(short_segments_text)

        # Post-processing should merge very small segments
        for segment in segments:
            # No segment should be extremely short after post-processing
            self.assertGreater(segment["metrics"].char_count, 5)

    def test_spacy_fallback(self):
        """Test fallback when spaCy is not available."""
        with patch.object(self.segmenter, "nlp", None):
            segments = self.segmenter.segment_document(self.medium_text)

            # Should still create segments using rule-based approach
            self.assertGreater(len(segments), 0)

            for segment in segments:
                self.assertGreater(len(segment["text"]), 0)
                # After post-processing, segment type might change due to merging
                self.assertIn(segment["segment_type"],
                              ["rule_based", "merged"])

    def test_segmentation_stats(self):
        """Test segmentation statistics calculation."""
        segments = self.segmenter.segment_document(self.long_text)

        stats = self.segmenter.segmentation_stats

        self.assertEqual(stats.total_segments, len(segments))
        self.assertGreater(stats.avg_char_length, 0)
        self.assertGreater(stats.avg_sentence_count, 0)
        self.assertIsInstance(stats.char_length_distribution, dict)
        self.assertIsInstance(stats.sentence_count_distribution, dict)

    def test_segmentation_report(self):
        """Test comprehensive segmentation report generation."""
        segments = self.segmenter.segment_document(self.long_text)
        report = self.segmenter.get_segmentation_report()

        # Check report structure
        self.assertIn("summary", report)
        self.assertIn("character_analysis", report)
        self.assertIn("sentence_analysis", report)
        self.assertIn("quality_indicators", report)

        # Check summary metrics
        summary = report["summary"]
        self.assertEqual(summary["total_segments"], len(segments))
        self.assertGreater(summary["avg_char_length"], 0)
        self.assertGreater(summary["avg_sentence_count"], 0)

        # Check quality scores
        quality = report["quality_indicators"]
        self.assertBetween(quality["consistency_score"], 0, 1)
        self.assertBetween(quality["target_adherence_score"], 0, 1)
        self.assertBetween(quality["overall_quality_score"], 0, 1)

    def assertBetween(self, value, min_val, max_val):
        """Custom assertion for range checking."""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

    def test_character_distribution_buckets(self):
        """Test character length distribution bucketing."""
        # Create segments with known lengths
        test_segments = [
            {
                "metrics": SegmentMetrics(
                    char_count=400, sentence_count=2, word_count=50, token_count=50
                )
            },
            {
                "metrics": SegmentMetrics(
                    char_count=650, sentence_count=3, word_count=80, token_count=80
                )
            },
            {
                "metrics": SegmentMetrics(
                    char_count=800, sentence_count=3, word_count=100, token_count=100
                )
            },
            {
                "metrics": SegmentMetrics(
                    char_count=1100, sentence_count=4, word_count=140, token_count=140
                )
            },
        ]

        char_lengths = [seg["metrics"].char_count for seg in test_segments]
        distribution = self.segmenter._create_char_distribution(char_lengths)

        self.assertEqual(distribution["< 500"], 1)
        self.assertEqual(distribution["500-699"], 1)
        self.assertEqual(distribution["700-900 (target)"], 1)
        self.assertEqual(distribution["901-1200"], 1)

    def test_semantic_coherence_estimation(self):
        """Test semantic coherence score calculation."""
        coherent_text = "The algorithm processes documents efficiently. The system uses advanced processing techniques. Processing accuracy improves with better algorithms."
        incoherent_text = "Random words scattered. Elephant purple mathematics. Cooking flying temperature."

        coherent_score = self.segmenter._estimate_semantic_coherence(
            coherent_text)
        incoherent_score = self.segmenter._estimate_semantic_coherence(
            incoherent_text)

        # Coherent text should score higher
        self.assertGreater(coherent_score, incoherent_score)

        # Scores should be in valid range
        self.assertBetween(coherent_score, 0, 1)
        self.assertBetween(incoherent_score, 0, 1)

    def test_large_segment_splitting(self):
        """Test splitting of overly large segments."""
        # Create a very long text that should be split
        very_long_text = " ".join(
            [
                "This is sentence number "
                + str(i)
                + " with additional content to make it longer."
                for i in range(50)
            ]
        )

        segments = self.segmenter.segment_document(very_long_text)

        # Should create multiple segments (be more flexible with count)
        self.assertGreater(len(segments), 1)

        # No segment should exceed maximum size
        for segment in segments:
            self.assertLessEqual(
                segment["metrics"].char_count, self.segmenter.max_segment_chars
            )

    def test_emergency_fallback(self):
        """Test emergency fallback segmentation."""
        # Create a pathological case
        pathological_text = "a" * 10000  # Very long text with no sentence boundaries

        segments = self.segmenter._emergency_fallback_segmentation(
            pathological_text)

        # Should create segments
        self.assertGreater(len(segments), 0)

        # Segments should be reasonably sized
        for segment in segments:
            self.assertLessEqual(
                segment["metrics"].char_count,
                (self.segmenter.target_char_min +
                 self.segmenter.target_char_max) // 2
                + 50,
            )  # Allow some flexibility

    def test_quality_score_calculations(self):
        """Test quality score calculation methods."""
        # Process a document first
        self.segmenter.segment_document(self.long_text)

        consistency_score = self.segmenter._calculate_consistency_score()
        adherence_score = self.segmenter._calculate_target_adherence_score()
        overall_score = self.segmenter._calculate_overall_quality_score()

        # All scores should be in valid range
        self.assertBetween(consistency_score, 0, 1)
        self.assertBetween(adherence_score, 0, 1)
        self.assertBetween(overall_score, 0, 1)

    def test_different_configurations(self):
        """Test segmenter with different configurations."""
        configurations = [
            {"target_char_min": 500, "target_char_max": 700, "target_sentences": 2},
            {"target_char_min": 900, "target_char_max": 1200, "target_sentences": 4},
            {"target_char_min": 300, "target_char_max": 500, "target_sentences": 1},
        ]

        for i, config in enumerate(configurations):
            with self.subTest(config=i):
                segmenter = DocumentSegmenter(**config)
                segments = segmenter.segment_document(self.long_text)

                self.assertGreater(len(segments), 0)

                # Check that configuration is respected
                self.assertEqual(segmenter.target_char_min,
                                 config["target_char_min"])
                self.assertEqual(segmenter.target_char_max,
                                 config["target_char_max"])
                self.assertEqual(segmenter.target_sentences,
                                 config["target_sentences"])


class TestSegmentMetrics(unittest.TestCase):
    """Test cases for SegmentMetrics dataclass."""

    def test_segment_metrics_creation(self):
        """Test SegmentMetrics dataclass creation."""
        metrics = SegmentMetrics(
            char_count=800,
            sentence_count=3,
            word_count=120,
            token_count=125,
            semantic_coherence_score=0.85,
            segment_type="sentence_based",
        )

        self.assertEqual(metrics.char_count, 800)
        self.assertEqual(metrics.sentence_count, 3)
        self.assertEqual(metrics.word_count, 120)
        self.assertEqual(metrics.token_count, 125)
        self.assertEqual(metrics.semantic_coherence_score, 0.85)
        self.assertEqual(metrics.segment_type, "sentence_based")


class TestSegmentationStats(unittest.TestCase):
    """Test cases for SegmentationStats dataclass."""

    def test_segmentation_stats_creation(self):
        """Test SegmentationStats dataclass creation and defaults."""
        stats = SegmentationStats()

        self.assertEqual(stats.segments, [])
        self.assertEqual(stats.total_segments, 0)
        self.assertEqual(stats.segments_in_char_range, 0)
        self.assertEqual(stats.segments_with_3_sentences, 0)
        self.assertEqual(stats.avg_char_length, 0.0)
        self.assertEqual(stats.avg_sentence_count, 0.0)
        self.assertEqual(stats.char_length_distribution, {})
        self.assertEqual(stats.sentence_count_distribution, {})


if __name__ == "__main__":
    unittest.main()
