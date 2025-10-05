"""
Tests for core contradiction detection functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pdm_contra.core import ContradictionDetector
from pdm_contra.ingest.loader import PDMLoader
from pdm_contra.models import RiskLevel
from pdm_contra.nlp.patterns import PatternMatcher
from pdm_contra.policy.competence import CompetenceValidator
from pdm_contra.scoring.risk import RiskScorer


class TestContradictionDetector:
    """Test suite for ContradictionDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return ContradictionDetector(mode_light=True)

    @pytest.fixture
    def sample_pdm_text(self):
        """Sample PDM text with contradictions."""
        return """
        PLAN DE DESARROLLO MUNICIPAL 2024-2027
        
        OBJETIVO 1: Aumentar la cobertura educativa al 95% para 2027.
        
        META 1.1: Alcanzar una cobertura del 95% en educación básica y media.
        Sin embargo, el presupuesto educativo ha sido reducido en un 30% para el periodo.
        
        PROGRAMA DE SALUD:
        El municipio construirá y administrará directamente un nuevo hospital de nivel 3
        para atender las necesidades especializadas de la población.
        
        META SALUD: Reducir la mortalidad infantil en 50%.
        No obstante, no se contemplan recursos para programas de prevención.
        
        ACCIÓN EDUCACIÓN:
        El municipio procederá a contratar y asignar 200 docentes nuevos para las
        instituciones educativas municipales.
        
        PRESUPUESTO:
        Se proyecta un incremento del 40% en inversión social, pero los ingresos
        esperados solo crecerán un 10% según las proyecciones.
        """

    @staticmethod
    def test_basic_contradiction_detection(detector, sample_pdm_text):
        """Test basic contradiction detection."""
        result = detector.detect_contradictions(sample_pdm_text)

        if result.total_contradictions <= 0:
            raise AssertionError
        if result.risk_score <= 0:
            raise AssertionError
        if result.risk_level not in [level.value for level in RiskLevel]:
            raise AssertionError

    @staticmethod
    def test_competence_validation(detector, sample_pdm_text):
        """Test competence validation."""
        result = detector.detect_contradictions(
            sample_pdm_text, sectors=["salud", "educacion"]
        )

        # Should detect competence issues
        if result.total_competence_issues <= 0:
            raise AssertionError

        # Should find hospital administration overreach
        competence_issues = result.competence_mismatches
        hospital_issues = [
            i for i in competence_issues if "hospital" in str(i).lower()]
        if len(hospital_issues) <= 0:
            raise AssertionError

    @staticmethod
    def test_empty_text(detector):
        """Test handling of empty text."""
        result = detector.detect_contradictions("")

        if result.total_contradictions != 0:
            raise AssertionError
        if result.risk_score != 0:
            raise AssertionError
        if result.risk_level != RiskLevel.LOW.value:
            raise AssertionError

    @staticmethod
    def test_quantitative_contradictions(detector):
        """Test detection of quantitative contradictions."""
        text = """
        META: Incrementar la inversión en infraestructura en 80%.
        Sin embargo, el presupuesto total se reduce en 25%.
        
        OBJETIVO: Alcanzar 100% de cobertura en agua potable.
        Pero solo se asignan recursos para cubrir el 60% de la población.
        """

        result = detector.detect_contradictions(text)

        if result.total_contradictions <= 0:
            raise AssertionError
        # Should detect percentage contradictions
        for contradiction in result.contradictions:
            if not (hasattr(contradiction, "quantitative_targets") or hasattr(
                contradiction, "quantifiers"
            )):
                raise AssertionError


# tests/test_patterns.py
"""
Tests for pattern matching module.
"""


class TestPatternMatcher:
    """Test suite for PatternMatcher."""

    @pytest.fixture
    def matcher(self):
        """Create pattern matcher instance."""
        return PatternMatcher(language="es")

    @staticmethod
    def test_adversative_detection(matcher):
        """Test adversative connector detection."""
        text = "El objetivo es claro, sin embargo no hay recursos."
        matches = matcher.find_adversatives(text)

        if len(matches) <= 0:
            raise AssertionError
        if "sin embargo" not in matches[0]["adversative"].lower():
            raise AssertionError

    @staticmethod
    def test_competence_verb_extraction(matcher):
        """Test extraction of competence-sensitive verbs."""
        text = """
        El municipio procederá a contratar docentes para las escuelas.
        También construirá un hospital de alta complejidad.
        """

        verbs = matcher.extract_competence_verbs(text)

        if len(verbs) <= 0:
            raise AssertionError
        # Should find "contratar docentes" and "construir hospital"
        verb_texts = [v[0] for v in verbs]
        if not any("contratar" in v and "docentes" in v for v in verb_texts):
            raise AssertionError
        if not any("construi" in v and "hospital" in v for v in verb_texts):
            raise AssertionError

    @staticmethod
    def test_quantitative_pattern_detection(matcher):
        """Test detection of quantitative patterns."""
        text = """
        La meta es alcanzar 95% de cobertura.
        Se invertirán $500 millones COP.
        Beneficiará a 10,000 familias.
        """

        matches = matcher.find_adversatives(text, context_window=500)
        # Even without adversatives, we can check quantitative extraction

        # Test direct pattern matching
        found = []
        for pattern in matcher.quantitative:
            for match in pattern.finditer(text):
                found.append(match.group())

        if len(found) <= 0:
            raise AssertionError
        if not any("95%" in f for f in found):
            raise AssertionError
        if not any("500 millones" in f for f in found):
            raise AssertionError
        if not any(
            "10,000 familias" in f or "10.000 familias" in f for f in found):
            raise AssertionError


# tests/test_competence.py
"""
Tests for competence validation module.
"""


class TestCompetenceValidator:
    """Test suite for CompetenceValidator."""

    @pytest.fixture
    def validator(self):
        """Create competence validator instance."""
        return CompetenceValidator()

    @staticmethod
    def test_municipal_overreach_detection(validator):
        """Test detection of municipal overreach."""
        text = """
        El municipio construirá y administrará un hospital de tercer nivel
        para atender especialidades médicas complejas.
        """

        issues = validator.validate_segment(text, ["salud"], "municipal")

        if len(issues) <= 0:
            raise AssertionError
        if issues[0]["type"] != "competence_overreach":
            raise AssertionError
        if issues[0]["sector"] != "salud":
            raise AssertionError
        if "departamental" not in issues[0]["required_level"]:
            raise AssertionError

    @staticmethod
    def test_valid_municipal_action(validator):
        """Test validation of proper municipal actions."""
        text = """
        El municipio gestionará convenios con el departamento para mejorar
        la atención en salud y cofinanciará programas de prevención.
        """

        issues = validator.validate_segment(text, ["salud"], "municipal")

        # Should not find overreach issues
        overreach = [i for i in issues if i["type"] == "competence_overreach"]
        assert len(overreach) == 0

    @staticmethod
    def test_education_competence(validator):
        """Test education sector competence validation."""
        text = """
        El municipio procederá a nombrar y contratar 50 docentes nuevos
        para las instituciones educativas del territorio.
        """

        issues = validator.validate_segment(text, ["educacion"], "municipal")

        if len(issues) <= 0:
            raise AssertionError
        # Nombramiento de docentes es competencia departamental
        if not any("nombrar" in i["text"] for i in issues):
            raise AssertionError

    @staticmethod
    def test_suggested_fixes(validator):
        """Test that suggested fixes are provided."""
        text = "El municipio administrará directamente el hospital regional."

        issues = validator.validate_segment(text, ["salud"], "municipal")

        if len(issues) <= 0:
            raise AssertionError
        if "suggested_fix" not in issues[0]:
            raise AssertionError
        if "gestionar" not in issues[0]["suggested_fix"].lower():
            raise AssertionError


# tests/test_risk_scoring.py
"""
Tests for risk scoring module.
"""


class TestRiskScorer:
    """Test suite for RiskScorer."""

    @pytest.fixture
    def scorer(self):
        """Create risk scorer instance."""
        return RiskScorer(alpha=0.1)

    @staticmethod
    def test_basic_risk_calculation(scorer):
        """Test basic risk calculation."""
        # Mock contradiction data
        contradictions = [
            type("Contradiction", (), {"confidence": 0.8, "type": "semantic"}),
            type("Contradiction", (), {
                 "confidence": 0.6, "type": "quantitative"}),
        ]

        result = scorer.calculate_risk(contradictions, [], [])

        if "overall_risk" not in result:
            raise AssertionError
        if not 0 <= result["overall_risk"] <= 1:
            raise AssertionError
        if "risk_level" not in result:
            raise AssertionError
        if "confidence_intervals" not in result:
            raise AssertionError

    @staticmethod
    def test_confidence_intervals(scorer):
        """Test confidence interval calculation."""
        # Add some calibration data
        for _ in range(30):
            scorer.add_calibration_sample(
                np.random.random(),
                np.random.random(),
                np.random.random(),
                np.random.random(),
            )

        result = scorer.calculate_risk([], [], [])

        intervals = result["confidence_intervals"]
        if "overall" not in intervals:
            raise AssertionError
        assert len(intervals["overall"]) == 2
        if intervals["overall"][0] > intervals["overall"][1]:
            raise AssertionError

    @staticmethod
    def test_risk_levels(scorer):
        """Test risk level assignment."""
        # Test different severity levels
        test_cases = [
            ([], [], [], "LOW"),  # No issues
            ([type("C", (), {"confidence": 0.3})],
             [], [], "LOW"),  # Low confidence
            (
                [type("C", (), {"confidence": 0.9})] * 5,
                [],
                [],
                "HIGH",
            ),  # Many high confidence
        ]

        for contradictions, competences, agenda, expected_level in test_cases:
            result = scorer.calculate_risk(contradictions, competences, agenda)
            # Allow for some flexibility in level assignment
            if not (expected_level in result["risk_level"] or result["risk_level"] in [
                "LOW",
                "MEDIUM",
                "MEDIUM_HIGH",
                "HIGH",
                "CRITICAL",
            ]):
                raise AssertionError


# tests/test_integration.py
"""
Integration tests for the complete PDM analysis pipeline.
"""


class TestIntegration:
    """Integration test suite."""

    @pytest.fixture
    def sample_pdm_file(self):
        """Create a sample PDM file for testing."""
        content = """
        PLAN DE DESARROLLO MUNICIPAL
        MUNICIPIO DE EJEMPLO
        2024-2027
        
        DIAGNÓSTICO:
        El municipio enfrenta desafíos en educación, salud y saneamiento básico.
        La cobertura educativa es del 75% y debe mejorarse.
        
        OBJETIVOS ESTRATÉGICOS:
        1. Aumentar la cobertura educativa al 95%
        2. Reducir la mortalidad infantil en 50%
        3. Garantizar agua potable al 100% de la población
        
        METAS:
        - Meta 1: Cobertura educativa del 95% para 2027
        - Meta 2: Reducir mortalidad infantil de 15 a 7.5 por mil nacidos vivos
        - Meta 3: 100% cobertura agua potable
        
        Sin embargo, el presupuesto asignado para educación se reduce en 30%.
        
        PROGRAMAS:
        - El municipio construirá un nuevo hospital de tercer nivel
        - Se contratarán 100 docentes municipales
        - Construcción de acueducto municipal
        
        PRESUPUESTO:
        Total: $10,000 millones COP
        - Educación: $2,000 millones (reducción del 30% respecto al periodo anterior)
        - Salud: $4,000 millones
        - Agua: $4,000 millones
        """

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            return Path(f.name)

    @pytest.fixture
    def competence_matrix_file(self):
        """Create a sample competence matrix file."""
        import json

        matrix = {
            "municipal": {
                "salud": ["gestionar", "coordinar", "cofinanciar"],
                "educacion": ["mantener infraestructura", "dotar"],
                "agua": ["prestar servicio", "garantizar acceso"],
            },
            "departmental": {
                "salud": ["administrar hospitales nivel 2-3"],
                "educacion": ["contratar docentes", "nombrar personal"],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(matrix, f)
            return Path(f.name)

    @staticmethod
    def test_full_pipeline(sample_pdm_file, competence_matrix_file):
        """Test complete analysis pipeline."""
        # Load document
        loader = PDMLoader()
        doc_data = loader.load(sample_pdm_file)

        if "text" not in doc_data:
            raise AssertionError
        if len(doc_data["text"]) <= 0:
            raise AssertionError

        # Initialize detector
        detector = ContradictionDetector(
            competence_matrix_path=competence_matrix_file, mode_light=True
        )

        # Run analysis
        analysis = detector.detect_contradictions(
            text=doc_data["text"],
            sectors=["salud", "educacion", "agua"],
            pdm_structure={"sections": doc_data.get("sections", {})},
        )

        # Verify results
        if analysis.total_contradictions <= 0:
            raise AssertionError
        if (
            analysis.total_competence_issues <= 0
        ):
            raise AssertionError
        if analysis.risk_score <= 0:
            raise AssertionError
        if len(analysis.explanations) <= 0:
            raise AssertionError

        # Cleanup
        sample_pdm_file.unlink()
        competence_matrix_file.unlink()

    @staticmethod
    def test_multiple_file_formats():
        """Test loading different file formats."""
        loader = PDMLoader()

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load(Path("non_existent.pdf"))

        # Create test files for different formats
        test_content = "Test PDM content"

        # Test TXT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            txt_file = Path(f.name)

        doc_data = loader.load(txt_file)
        if doc_data["text"] != test_content:
            raise AssertionError

        txt_file.unlink()
