"""
Contract Test: TeoriaCambio Interface
Validates that TeoriaCambio adheres to expected interface contract
"""
import pytest
import networkx as nx
from teoria_cambio import TeoriaCambio, CausalElement, CausalElementType


@pytest.mark.contract
class TestTeoriaCambioContract:
    """Contract tests for TeoriaCambio interface."""
    
    @staticmethod
    def test_initialization_contract():
        """Contract: TeoriaCambio must initialize without arguments."""
        tc = TeoriaCambio()
        assert tc is not None
    
    @staticmethod
    def test_construir_grafo_causal_contract():
        """Contract: construir_grafo_causal must accept list of CausalElement."""
        tc = TeoriaCambio()
        
        elementos = [
            CausalElement(
                id="A1",
                text="Test activity",
                element_type=CausalElementType.ACTIVITY
            )
        ]
        
        grafo = tc.construir_grafo_causal(elementos)
        
        assert grafo is not None
        assert isinstance(grafo, nx.DiGraph)
    
    @staticmethod
    def test_invalidar_cache_grafo_contract():
        """Contract: invalidar_cache_grafo must be callable."""
        tc = TeoriaCambio()
        
        elementos = [
            CausalElement(
                id="A1",
                text="Test",
                element_type=CausalElementType.ACTIVITY
            )
        ]
        
        grafo1 = tc.construir_grafo_causal(elementos)
        tc.invalidar_cache_grafo()
        grafo2 = tc.construir_grafo_causal(elementos)
        
        assert grafo1 is not None
        assert grafo2 is not None
    
    @staticmethod
    def test_graph_structure_contract():
        """Contract: Constructed graph must have valid structure."""
        tc = TeoriaCambio()
        
        elementos = [
            CausalElement(
                id="A1",
                text="Activity 1",
                element_type=CausalElementType.ACTIVITY
            ),
            CausalElement(
                id="P1",
                text="Product 1",
                element_type=CausalElementType.OUTPUT,
                preconditions={"A1"}
            )
        ]
        
        grafo = tc.construir_grafo_causal(elementos)
        
        assert len(grafo.nodes) > 0
        assert all(isinstance(node, str) for node in grafo.nodes)
    
    @staticmethod
    def test_multiple_elements_contract():
        """Contract: Must handle multiple interconnected elements."""
        tc = TeoriaCambio()
        
        elementos = [
            CausalElement(id="A1", text="Act 1", element_type=CausalElementType.ACTIVITY),
            CausalElement(id="A2", text="Act 2", element_type=CausalElementType.ACTIVITY),
            CausalElement(id="P1", text="Prod 1", element_type=CausalElementType.OUTPUT, preconditions={"A1", "A2"}),
            CausalElement(id="R1", text="Res 1", element_type=CausalElementType.OUTCOME, preconditions={"P1"})
        ]
        
        grafo = tc.construir_grafo_causal(elementos)
        
        assert len(grafo.nodes) >= 4
        assert len(grafo.edges) >= 3
