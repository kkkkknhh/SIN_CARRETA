"""
Integration test: Feasibility Scoring + Contradiction Detection Flow
Critical Flow #3: Plan quality assessment
"""
import pytest
from feasibility_scorer import FactibilidadScorer
from contradiction_detector import ContradictionDetector


@pytest.mark.integration
@pytest.mark.critical_path
class TestFeasibilityContradictionFlow:
    """Test integration between feasibility scoring and contradiction detection."""
    
    def setup_method(self):
        """Initialize components."""
        self.feasibility_scorer = FactibilidadScorer()
        self.contradiction_detector = ContradictionDetector()
    
    def test_feasibility_reduced_by_contradictions(self):
        """Test that contradictions reduce overall feasibility score."""
        contradictory_text = """
        La línea base actual es 100 hogares atendidos.
        Se plantea alcanzar 500 hogares para diciembre de 2024.
        Sin embargo, no se cuenta con presupuesto asignado.
        El proyecto se ejecutará sin recursos adicionales.
        """
        
        consistent_text = """
        La línea base actual es 100 hogares atendidos.
        Se plantea alcanzar 500 hogares para diciembre de 2024.
        El presupuesto asignado es de $500 millones.
        Se cuenta con equipo técnico capacitado.
        """
        
        feas_contradictory = self.feasibility_scorer.score_text(contradictory_text)
        contradictions_found = self.contradiction_detector.detect_contradictions(contradictory_text)
        
        feas_consistent = self.feasibility_scorer.score_text(consistent_text)
        contradictions_consistent = self.contradiction_detector.detect_contradictions(consistent_text)
        
        assert len(contradictions_found) > len(contradictions_consistent)
        
        adjusted_score_contradictory = feas_contradictory * (1 - 0.1 * len(contradictions_found))
        adjusted_score_consistent = feas_consistent * (1 - 0.1 * len(contradictions_consistent))
        
        assert adjusted_score_consistent > adjusted_score_contradictory
    
    def test_risk_integration_with_feasibility_patterns(self):
        """Test risk assessment integration with feasibility indicators."""
        text = """
        Línea base: 50 familias beneficiadas en 2023.
        Meta: 200 familias beneficiadas para diciembre 2025.
        Plazo: 24 meses de ejecución.
        Presupuesto: $300 millones asignados.
        Pero no hay personal disponible para ejecutar.
        """
        
        feas_score = self.feasibility_scorer.score_text(text)
        contradictions = self.contradiction_detector.detect_contradictions(text)
        
        assert feas_score > 0, "Should detect feasibility indicators"
        assert len(contradictions) > 0, "Should detect contradiction"
        
        risk_adjusted_feasibility = feas_score * (
            1 - sum(c.confidence for c in contradictions) * 0.2
        )
        
        assert 0 <= risk_adjusted_feasibility <= 1
        assert risk_adjusted_feasibility < feas_score
    
    def test_comprehensive_plan_quality_score(self):
        """Test comprehensive quality scoring combining both components."""
        high_quality_text = """
        Línea base 2023: 100 estudiantes graduados.
        Meta 2025: 300 estudiantes graduados (incremento del 200%).
        Plazo: 18 meses para implementación completa.
        Presupuesto: $800 millones aprobados y desembolsados.
        Responsable: Secretaría de Educación Municipal.
        Indicador: Tasa de graduación universitaria.
        """
        
        low_quality_text = """
        Queremos mejorar la educación.
        Se espera tener mejores resultados.
        Trabajaremos con las instituciones.
        """
        
        feas_high = self.feasibility_scorer.score_text(high_quality_text)
        contrad_high = len(self.contradiction_detector.detect_contradictions(high_quality_text))
        
        feas_low = self.feasibility_scorer.score_text(low_quality_text)
        contrad_low = len(self.contradiction_detector.detect_contradictions(low_quality_text))
        
        quality_high = feas_high * (1 - 0.15 * contrad_high)
        quality_low = feas_low * (1 - 0.15 * contrad_low)
        
        assert quality_high > quality_low
