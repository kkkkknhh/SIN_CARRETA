"""
Tests for deterministic Monte Carlo DAG validation.
"""

import unittest

from dag_validation import AdvancedDAGValidator, GraphNode


class TestDAGValidator(unittest.TestCase):
    """Test cases for DAG validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = AdvancedDAGValidator()

    def test_seed_generation_deterministic(self):
        """Test that seed generation is deterministic for same plan names."""
        plan_name = "test_plan_123"
        seed1 = self.validator._create_seed_from_plan_name(plan_name)
        seed2 = self.validator._create_seed_from_plan_name(plan_name)

        self.assertEqual(
            seed1, seed2, "Seeds should be identical for same plan name")

    def test_seed_generation_different_plans(self):
        """Test that different plan names produce different seeds."""
        seed1 = self.validator._create_seed_from_plan_name("plan_a")
        seed2 = self.validator._create_seed_from_plan_name("plan_b")

        self.assertNotEqual(
            seed1, seed2, "Different plan names should produce different seeds"
        )

    def test_add_nodes_and_edges(self):
        """Test adding nodes and edges to the graph."""
        self.validator.add_node("A")
        self.validator.add_node("B")
        self.validator.add_edge("A", "B")

        self.assertIn("A", self.validator.graph_nodes)
        self.assertIn("B", self.validator.graph_nodes)
        self.assertIn("A", self.validator.graph_nodes["B"].dependencies)

    def test_acyclicity_check_simple_dag(self):
        """Test acyclicity checking on a simple DAG."""
        nodes = {
            "A": GraphNode("A", set()),
            "B": GraphNode("B", {"A"}),
            "C": GraphNode("C", {"B"}),
        }

        self.assertTrue(
            self.validator._is_acyclic(nodes), "Simple DAG should be acyclic"
        )

    def test_acyclicity_check_cycle(self):
        """Test acyclicity checking on a graph with cycle."""
        nodes = {
            "A": GraphNode("A", {"C"}),
            "B": GraphNode("B", {"A"}),
            "C": GraphNode("C", {"B"}),
        }

        self.assertFalse(
            self.validator._is_acyclic(
                nodes), "Graph with cycle should not be acyclic"
        )

    def test_empty_graph_acyclic(self):
        """Test that empty graph is considered acyclic."""
        self.assertTrue(self.validator._is_acyclic({}),
                        "Empty graph should be acyclic")

    def test_single_node_acyclic(self):
        """Test that single node is acyclic."""
        nodes = {"A": GraphNode("A", set())}
        self.assertTrue(
            self.validator._is_acyclic(nodes), "Single node should be acyclic"
        )

    def test_reproducibility_same_plan(self):
        """Test that same plan name produces identical results."""
        # Create a simple graph
        self.validator.add_node("A")
        self.validator.add_node("B", {"A"})
        self.validator.add_node("C", {"B"})

        plan_name = "reproducibility_test"
        result1 = self.validator.calculate_acyclicity_pvalue(plan_name, 50)
        result2 = self.validator.calculate_acyclicity_pvalue(plan_name, 50)

        self.assertEqual(result1.seed, result2.seed)
        self.assertEqual(result1.acyclic_count, result2.acyclic_count)
        self.assertEqual(result1.p_value, result2.p_value)
        self.assertEqual(result1.subgraph_sizes, result2.subgraph_sizes)

    def test_reproducibility_different_plans(self):
        """Test that different plan names produce different results."""
        # Create a simple graph
        self.validator.add_node("A")
        self.validator.add_node("B", {"A"})
        self.validator.add_node("C", {"B"})

        result1 = self.validator.calculate_acyclicity_pvalue("plan_x", 50)
        result2 = self.validator.calculate_acyclicity_pvalue("plan_y", 50)

        # Seeds should be different
        self.assertNotEqual(result1.seed, result2.seed)
        # Results may be different (though not guaranteed due to randomness)

    def test_monte_carlo_empty_graph(self):
        """Test Monte Carlo on empty graph."""
        result = self.validator.calculate_acyclicity_pvalue("empty_test", 100)

        self.assertEqual(result.total_iterations, 0)
        self.assertEqual(result.acyclic_count, 0)
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(result.subgraph_sizes, [])

    def test_monte_carlo_valid_result(self):
        """Test that Monte Carlo produces valid statistical results."""
        # Create a larger graph for meaningful testing
        for i in range(5):
            self.validator.add_node(f"node_{i}")

        # Add some dependencies
        self.validator.add_edge("node_0", "node_1")
        self.validator.add_edge("node_1", "node_2")
        self.validator.add_edge("node_2", "node_3")

        result = self.validator.calculate_acyclicity_pvalue("valid_test", 100)

        # Basic validation
        self.assertEqual(result.total_iterations, 100)
        self.assertGreaterEqual(result.acyclic_count, 0)
        self.assertLessEqual(result.acyclic_count, 100)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)
        self.assertEqual(len(result.subgraph_sizes), 100)
        self.assertTrue(result.reproducible)

    def test_verify_reproducibility_method(self):
        """Test the verify_reproducibility method."""
        # Create a simple graph
        self.validator.add_node("X")
        self.validator.add_node("Y", {"X"})

        is_reproducible = self.validator.verify_reproducibility(
            "repro_test", 50)
        self.assertTrue(is_reproducible, "Results should be reproducible")

    def test_graph_stats(self):
        """Test graph statistics calculation."""
        # Empty graph
        stats = self.validator.get_graph_stats()
        self.assertEqual(stats["total_nodes"], 0)
        self.assertEqual(stats["total_edges"], 0)

        # Add nodes and edges
        self.validator.add_node("A")
        self.validator.add_node("B", {"A"})
        self.validator.add_node("C", {"A", "B"})

        stats = self.validator.get_graph_stats()
        self.assertEqual(stats["total_nodes"], 3)
        self.assertEqual(stats["total_edges"], 3)  # A->B, A->C, B->C
        self.assertEqual(stats["max_possible_edges"], 6)  # 3 * 2

    def test_subgraph_generation(self):
        """Test random subgraph generation."""
        # Create a graph with enough nodes
        for i in range(10):
            self.validator.add_node(f"node_{i}")

        # Initialize RNG with a test plan
        self.validator._initialize_rng("subgraph_test")

        # Generate subgraph
        subgraph = self.validator._generate_random_subgraph(3, 5)

        # Verify constraints
        self.assertGreaterEqual(len(subgraph), 3)
        self.assertLessEqual(len(subgraph), 5)

        # All nodes should be from original graph
        original_names = set(self.validator.graph_nodes.keys())
        subgraph_names = set(subgraph.keys())
        self.assertTrue(subgraph_names.issubset(original_names))


class TestSampleCausalGraph(unittest.TestCase):
    """Test the sample causal graph creation."""

    def test_sample_graph_creation(self):
        """Test that sample graph is created correctly."""
        from dag_validation import create_sample_causal_graph

        validator = create_sample_causal_graph()
        stats = validator.get_graph_stats()

        self.assertGreater(stats["total_nodes"], 0)
        self.assertGreater(stats["total_edges"], 0)

    def test_sample_graph_acyclicity(self):
        """Test that sample graph is acyclic."""
        from dag_validation import create_sample_causal_graph

        validator = create_sample_causal_graph()
        is_acyclic = validator._is_acyclic(validator.graph_nodes)

        self.assertTrue(is_acyclic, "Sample causal graph should be acyclic")

    def test_sample_graph_reproducibility(self):
        """Test reproducibility on sample graph."""
        from dag_validation import create_sample_causal_graph

        validator = create_sample_causal_graph()
        is_reproducible = validator.verify_reproducibility("sample_test", 50)

        self.assertTrue(
            is_reproducible, "Sample graph should produce reproducible results"
        )


if __name__ == "__main__":
    unittest.main()
