"""
Integration test: TeoriaCambio + DAG Validation Flow
Critical Flow #2: Causal graph construction and validation
"""
import pytest
from teoria_cambio import TeoriaCambio, CausalElement, CausalElementType
from dag_validation import validate_dag_acyclicity


@pytest.mark.integration
@pytest.mark.critical_path
class TestTeoriaCambioDAGFlow:
    """Test integration between theory of change and DAG validation."""
    
    def test_teoria_cambio_creates_valid_dag(self):
        """Test that TeoriaCambio generates a valid acyclic graph."""
        tc = TeoriaCambio()
        
        elementos = [
            CausalElement(id="A1", text="Capacitación", element_type=CausalElementType.ACTIVITY),
            CausalElement(id="P1", text="500 capacitados", element_type=CausalElementType.OUTPUT, preconditions={"A1"}),
            CausalElement(id="R1", text="30% aumento TIC", element_type=CausalElementType.OUTCOME, preconditions={"P1"}),
            CausalElement(id="I1", text="Inclusión digital", element_type=CausalElementType.IMPACT, preconditions={"R1"})
        ]
        
        grafo = tc.construir_grafo_causal(elementos)
        
        assert grafo is not None
        assert len(grafo.nodes) > 0
        
        is_acyclic, details = validate_dag_acyclicity(grafo, num_samples=100, seed=42)
        assert is_acyclic, f"Graph should be acyclic: {details}"
    
    def test_complex_teoria_cambio_dag_validation(self):
        """Test DAG validation with complex multi-path theory of change."""
        tc = TeoriaCambio()
        
        elementos = []
        
        for i in range(1, 6):
            elementos.append(
                CausalElement(
                    id=f"A{i}",
                    text=f"Actividad {i}",
                    element_type=CausalElementType.ACTIVITY
                )
            )
        
        for i in range(1, 4):
            deps = {f"A{j}" for j in range(1, i+2)}
            elementos.append(
                CausalElement(
                    id=f"P{i}",
                    text=f"Producto {i}",
                    element_type=CausalElementType.OUTPUT,
                    preconditions=deps
                )
            )
        
        elementos.append(
            CausalElement(
                id="R1",
                text="Resultado integrado",
                element_type=CausalElementType.OUTCOME,
                preconditions={"P1", "P2", "P3"}
            )
        )
        
        elementos.append(
            CausalElement(
                id="I1",
                text="Impacto final",
                element_type=CausalElementType.IMPACT,
                preconditions={"R1"}
            )
        )
        
        grafo = tc.construir_grafo_causal(elementos)
        
        is_acyclic, details = validate_dag_acyclicity(grafo, num_samples=500, seed=42)
        assert is_acyclic, "Complex graph should remain acyclic"
        assert len(grafo.nodes) >= 11
    
    def test_cache_invalidation_preserves_dag_property(self):
        """Test that cache invalidation maintains DAG structure."""
        tc = TeoriaCambio()
        
        elementos1 = [
            CausalElement(id="A1", text="Act 1", element_type=CausalElementType.ACTIVITY),
            CausalElement(id="P1", text="Prod 1", element_type=CausalElementType.OUTPUT, preconditions={"A1"})
        ]
        
        grafo1 = tc.construir_grafo_causal(elementos1)
        is_acyclic1, _ = validate_dag_acyclicity(grafo1, num_samples=50, seed=42)
        
        tc.invalidar_cache_grafo()
        
        elementos2 = elementos1 + [
            CausalElement(id="R1", text="Result 1", element_type=CausalElementType.OUTCOME, preconditions={"P1"})
        ]
        
        grafo2 = tc.construir_grafo_causal(elementos2)
        is_acyclic2, _ = validate_dag_acyclicity(grafo2, num_samples=50, seed=42)
        
        assert is_acyclic1 and is_acyclic2, "DAG property should be preserved"
        assert len(grafo2.nodes) > len(grafo1.nodes)
