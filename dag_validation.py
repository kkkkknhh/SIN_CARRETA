"""Mock DAG validator for testing."""


class DAGValidator:
    """Mock DAG validator."""

    def calculate_acyclicity_pvalue_advanced(self, plan_name: str):
        """Mock DAG validation."""
        return {"p_value": 0.5, "acyclic": True}
