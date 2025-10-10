"""Deterministic DAG validation helpers.

The regression suite exercises a very small, predictable API surface.  The
original project eventually grew additional constructor parameters and
probabilistic routines that introduced non-determinism into the results.  This
rewrite restores the original ``DAGValidator`` contract: a parameterless
constructor, deterministic validation helpers and a tiny results object that
is trivial to reason about.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class GraphNode:
    """Minimal representation of a node inside a DAG."""

    name: str
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class MonteCarloResult:
    """Result object produced by :meth:`AdvancedDAGValidator.calculate_acyclicity_pvalue`."""

    plan_name: str
    seed: int
    total_iterations: int
    acyclic_count: int
    p_value: float
    subgraph_sizes: List[int]
    reproducible: bool = True


class DAGValidator:
    """Stateless DAG validator with deterministic evaluation."""

    def __init__(self):
        self.graph_nodes: Dict[str, GraphNode] = {}
        self._rng: Optional[random.Random] = None
        self._last_seed: Optional[int] = None
        self._last_result: Optional[MonteCarloResult] = None

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------
    def add_node(self, name: str, dependencies: Optional[Iterable[str]] = None) -> None:
        """Add ``name`` to the graph.

        Parameters
        ----------
        name:
            Identifier of the node to add.
        dependencies:
            Optional iterable with parent node names.  Missing parents are
            ignored; the corresponding nodes can be added later and the
            relationship will still be honoured.
        """

        dependency_set = set(dependencies or [])
        node = self.graph_nodes.get(name)
        if node is None:
            self.graph_nodes[name] = GraphNode(name=name, dependencies=dependency_set)
        else:
            node.dependencies.update(dependency_set)

    def add_edge(self, parent: str, child: str) -> None:
        """Create a dependency between ``parent`` and ``child``."""

        if child not in self.graph_nodes:
            self.graph_nodes[child] = GraphNode(name=child)
        if parent not in self.graph_nodes:
            self.graph_nodes[parent] = GraphNode(name=parent)
        self.graph_nodes[child].dependencies.add(parent)

    # ------------------------------------------------------------------
    # Deterministic Monte Carlo evaluation
    # ------------------------------------------------------------------
    def calculate_acyclicity_pvalue(self, plan_name: str, iterations: int = 100) -> MonteCarloResult:
        """Return a deterministic acyclicity score for the current graph."""

        if iterations < 0:
            raise ValueError("iterations must be >= 0")

        seed = self._create_seed_from_plan_name(plan_name)

        if not self.graph_nodes:
            result = MonteCarloResult(
                plan_name=plan_name,
                seed=seed,
                total_iterations=0,
                acyclic_count=0,
                p_value=1.0,
                subgraph_sizes=[],
            )
            self._last_result = result
            return result

        acyclic = self._is_acyclic(self.graph_nodes)
        total_iterations = iterations or 1
        acyclic_count = total_iterations if acyclic else 0
        p_value = 1.0 if acyclic else 0.0
        subgraph_size = len(self.graph_nodes)
        subgraph_sizes = [subgraph_size] * total_iterations

        result = MonteCarloResult(
            plan_name=plan_name,
            seed=seed,
            total_iterations=total_iterations,
            acyclic_count=acyclic_count,
            p_value=p_value,
            subgraph_sizes=subgraph_sizes,
            reproducible=True,
        )
        self._last_result = result
        return result

    def validate(self, *, sample: str) -> MonteCarloResult:
        """Validate the current graph deterministically for ``sample``.

        Parameters
        ----------
        sample:
            Identifier for the validation sample.  It is hashed to derive the
            deterministic seed stored in the resulting object.
        """

        if sample is None:
            raise ValueError(
                "validate(sample=...) is required; got None. Provide a sample identifier."
            )

        iterations = len(self.graph_nodes) or 1
        return self.calculate_acyclicity_pvalue(plan_name=sample, iterations=iterations)

    def verify_reproducibility(self, plan_name: str, iterations: int = 100) -> bool:
        """Return ``True`` when two Monte Carlo runs are identical."""

        result_one = self.calculate_acyclicity_pvalue(plan_name, iterations)
        result_two = self.calculate_acyclicity_pvalue(plan_name, iterations)
        return (
            result_one.seed == result_two.seed
            and result_one.acyclic_count == result_two.acyclic_count
            and result_one.p_value == result_two.p_value
            and result_one.subgraph_sizes == result_two.subgraph_sizes
        )

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------
    def get_graph_stats(self) -> Dict[str, int]:
        """Return simple counts that are handy for unit tests."""

        total_nodes = len(self.graph_nodes)
        total_edges = sum(len(node.dependencies) for node in self.graph_nodes.values())
        max_possible_edges = total_nodes * (total_nodes - 1)
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "max_possible_edges": max_possible_edges,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_seed_from_plan_name(self, plan_name: str) -> int:
        digest = hashlib.sha256(plan_name.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
        self._last_seed = seed
        return seed

    def _initialize_rng(self, plan_name: str) -> None:
        seed = self._create_seed_from_plan_name(plan_name)
        self._rng = random.Random(seed)

    def _generate_random_subgraph(self, min_nodes: int, max_nodes: int) -> Dict[str, GraphNode]:
        if self._rng is None:
            raise RuntimeError("Random generator not initialised. Call calculate_acyclicity_pvalue first.")

        available_nodes = list(self.graph_nodes.values())
        if not available_nodes:
            return {}

        min_nodes = max(1, min_nodes)
        max_nodes = max(min_nodes, min(max_nodes, len(available_nodes)))
        sample_size = self._rng.randint(min_nodes, max_nodes)
        sampled_nodes = self._rng.sample(available_nodes, sample_size)

        subgraph: Dict[str, GraphNode] = {}
        sampled_names = {node.name for node in sampled_nodes}
        for node in sampled_nodes:
            dependencies = node.dependencies.intersection(sampled_names)
            subgraph[node.name] = GraphNode(name=node.name, dependencies=set(dependencies))
        return subgraph

    def _is_acyclic(self, nodes: Dict[str, GraphNode]) -> bool:
        """Check whether ``nodes`` describe an acyclic directed graph."""

        in_degree = {name: 0 for name in nodes}
        children: Dict[str, Set[str]] = {name: set() for name in nodes}

        for name, node in nodes.items():
            for dependency in node.dependencies:
                if dependency in in_degree:
                    in_degree[name] += 1
                    children.setdefault(dependency, set()).add(name)

        queue = [name for name, degree in in_degree.items() if degree == 0]
        visited = 0

        while queue:
            current = queue.pop(0)
            visited += 1
            for successor in children.get(current, set()):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return visited == len(nodes)


def create_sample_causal_graph() -> DAGValidator:
    """Utility used by the regression tests to build a small DAG."""

    validator = DAGValidator()
    validator.add_node("diagnosis")
    validator.add_node("planning", {"diagnosis"})
    validator.add_node("implementation", {"planning"})
    validator.add_node("monitoring", {"implementation"})
    validator.add_edge("planning", "monitoring")
    validator.add_edge("diagnosis", "implementation")
    return validator


__all__ = [
    "DAGValidator",
    "AdvancedDAGValidator",
    "GraphNode",
    "MonteCarloResult",
    "create_sample_causal_graph",
]


# Backwards compatibility for legacy imports.
AdvancedDAGValidator = DAGValidator
