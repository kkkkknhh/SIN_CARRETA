"""Mock feasibility scorer for testing."""


class FeasibilityScorer:
    """Mock feasibility scorer."""
    
    def evaluate_plan_feasibility(self, text: str):
        """Mock evaluate feasibility."""
        return {"indicators": []}
