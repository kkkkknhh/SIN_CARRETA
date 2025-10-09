#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Document Segmenter alignment with decalogo-industrial.latest.clean.json

Tests verify:
1. Section types map to correct D1-D6 dimensions
2. Patterns correctly identify section types
3. Dimension inference matches decalogo structure
"""

import unittest
from document_segmenter import DocumentSegmenter, DocumentSegment, SectionType, SegmentationType


class TestDecalogoAlignment(unittest.TestCase):
    """Test Document Segmenter alignment with decalogo-industrial.latest.clean.json structure."""

    def setUp(self):
        """Set up test fixtures."""
        self.segmenter = DocumentSegmenter()

    def test_d1_insumos_mapping(self):
        """Test D1: INSUMOS dimension mapping."""
        # Test diagnostic section
        diagnostic_segment = DocumentSegment(
            text="Diagnóstico de la situación actual",
            start_pos=0,
            end_pos=35,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.DIAGNOSTIC
        )
        self.assertIn("D1", diagnostic_segment.decalogo_dimensions)
        
        # Test baseline section
        baseline_segment = DocumentSegment(
            text="Líneas base con datos verificables",
            start_pos=0,
            end_pos=35,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.BASELINE
        )
        self.assertIn("D1", baseline_segment.decalogo_dimensions)
        
        # Test resources section
        resources_segment = DocumentSegment(
            text="Recursos asignados en el PPI",
            start_pos=0,
            end_pos=29,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.RESOURCES
        )
        self.assertIn("D1", resources_segment.decalogo_dimensions)
        
        # Test capacity section
        capacity_segment = DocumentSegment(
            text="Capacidades institucionales necesarias",
            start_pos=0,
            end_pos=39,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.CAPACITY
        )
        self.assertIn("D1", capacity_segment.decalogo_dimensions)

    def test_d2_actividades_mapping(self):
        """Test D2: ACTIVIDADES dimension mapping."""
        # Test activity section
        activity_segment = DocumentSegment(
            text="Actividades formalizadas en tablas",
            start_pos=0,
            end_pos=35,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.ACTIVITY
        )
        self.assertIn("D2", activity_segment.decalogo_dimensions)
        
        # Test mechanism section
        mechanism_segment = DocumentSegment(
            text="Mecanismo causal pretendido",
            start_pos=0,
            end_pos=28,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.MECHANISM
        )
        self.assertIn("D2", mechanism_segment.decalogo_dimensions)
        
        # Test intervention section
        intervention_segment = DocumentSegment(
            text="Teoría de intervención coherente",
            start_pos=0,
            end_pos=33,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.INTERVENTION
        )
        self.assertIn("D2", intervention_segment.decalogo_dimensions)

    def test_d3_productos_mapping(self):
        """Test D3: PRODUCTOS dimension mapping."""
        # Test product section
        product_segment = DocumentSegment(
            text="Productos con indicadores verificables",
            start_pos=0,
            end_pos=39,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.PRODUCT
        )
        self.assertIn("D3", product_segment.decalogo_dimensions)
        
        # Test output section
        output_segment = DocumentSegment(
            text="Outputs con trazabilidad presupuestal",
            start_pos=0,
            end_pos=38,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.OUTPUT
        )
        self.assertIn("D3", output_segment.decalogo_dimensions)

    def test_d4_resultados_mapping(self):
        """Test D4: RESULTADOS dimension mapping."""
        # Test result section
        result_segment = DocumentSegment(
            text="Resultados con métricas de outcome",
            start_pos=0,
            end_pos=35,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.RESULT
        )
        self.assertIn("D4", result_segment.decalogo_dimensions)
        
        # Test outcome section
        outcome_segment = DocumentSegment(
            text="Outcomes con ventana de maduración",
            start_pos=0,
            end_pos=35,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.OUTCOME
        )
        self.assertIn("D4", outcome_segment.decalogo_dimensions)
        
        # Test indicator section
        indicator_segment = DocumentSegment(
            text="Indicadores de resultado medibles",
            start_pos=0,
            end_pos=34,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.INDICATOR
        )
        self.assertIn("D4", indicator_segment.decalogo_dimensions)

    def test_d5_impactos_mapping(self):
        """Test D5: IMPACTOS dimension mapping."""
        # Test impact section
        impact_segment = DocumentSegment(
            text="Impactos de largo plazo medibles",
            start_pos=0,
            end_pos=33,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.IMPACT
        )
        self.assertIn("D5", impact_segment.decalogo_dimensions)
        
        # Test long term effect section
        long_term_segment = DocumentSegment(
            text="Efectos duraderos con proxies de impacto",
            start_pos=0,
            end_pos=41,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.LONG_TERM_EFFECT
        )
        self.assertIn("D5", long_term_segment.decalogo_dimensions)

    def test_d6_causalidad_mapping(self):
        """Test D6: CAUSALIDAD dimension mapping."""
        # Test causal theory section
        theory_segment = DocumentSegment(
            text="Teoría de cambio explícita con DAG",
            start_pos=0,
            end_pos=35,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.CAUSAL_THEORY
        )
        self.assertIn("D6", theory_segment.decalogo_dimensions)
        
        # Test causal link section
        link_segment = DocumentSegment(
            text="Encadenamiento causal validado",
            start_pos=0,
            end_pos=31,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.CAUSAL_LINK
        )
        self.assertIn("D6", link_segment.decalogo_dimensions)

    def test_multi_dimensional_sections(self):
        """Test sections that map to multiple dimensions."""
        # Vision: D1 + D6
        vision_segment = DocumentSegment(
            text="Visión del futuro deseado",
            start_pos=0,
            end_pos=26,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.VISION
        )
        self.assertIn("D1", vision_segment.decalogo_dimensions)
        self.assertIn("D6", vision_segment.decalogo_dimensions)
        
        # Objective: D4 + D6
        objective_segment = DocumentSegment(
            text="Objetivos estratégicos del plan",
            start_pos=0,
            end_pos=32,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.OBJECTIVE
        )
        self.assertIn("D4", objective_segment.decalogo_dimensions)
        self.assertIn("D6", objective_segment.decalogo_dimensions)
        
        # Strategy: D2 + D6
        strategy_segment = DocumentSegment(
            text="Estrategias de intervención",
            start_pos=0,
            end_pos=28,
            segment_type=SegmentationType.SECTION,
            section_type=SectionType.STRATEGY
        )
        self.assertIn("D2", strategy_segment.decalogo_dimensions)
        self.assertIn("D6", strategy_segment.decalogo_dimensions)

    def test_pattern_recognition_d1(self):
        """Test pattern recognition for D1 sections."""
        text = """
        Diagnóstico de la situación actual
        
        La problemática identificada muestra necesidades críticas en el sector.
        
        Líneas base con series temporales verificables
        
        Los datos base muestran mediciones iniciales de los indicadores clave.
        """
        
        segments = self.segmenter.segment_text(text)
        
        # Should identify diagnostic-related sections
        d1_segments = [s for s in segments if "D1" in s.decalogo_dimensions]
        self.assertGreater(len(d1_segments), 0, "Should identify at least one D1 section")

    def test_pattern_recognition_d2(self):
        """Test pattern recognition for D2 sections."""
        text = """
        Actividades y mecanismos causales
        
        Las actividades están formalizadas en tablas con responsables, insumos y outputs.
        
        Teoría de intervención
        
        La teoría de intervención establece las complementariedades entre actividades.
        """
        
        segments = self.segmenter.segment_text(text)
        
        # Should identify activity-related sections
        d2_segments = [s for s in segments if "D2" in s.decalogo_dimensions]
        self.assertGreater(len(d2_segments), 0, "Should identify at least one D2 section")

    def test_no_old_dimensions(self):
        """Test that old DE- dimensions are not used."""
        text = """
        Diagnóstico con líneas base
        Actividades formalizadas
        Productos verificables
        Resultados medibles
        Impactos de largo plazo
        Teoría de cambio explícita
        """
        
        segments = self.segmenter.segment_text(text)
        
        for segment in segments:
            for dim in segment.decalogo_dimensions:
                # Check no old DE- format dimensions
                self.assertFalse(dim.startswith("DE-"), 
                               f"Segment should not use old DE- format: {dim}")
                # Check only D1-D6 format
                if dim:  # Allow empty for OTHER type
                    self.assertTrue(dim in ["D1", "D2", "D3", "D4", "D5", "D6"],
                                  f"Dimension should be D1-D6 format: {dim}")

    def test_backward_compatibility_string_input(self):
        """Test that segment() accepts both string and dict inputs."""
        text = "Diagnóstico de la situación con líneas base verificables y datos completos"
        
        # Test string input (backward compatibility)
        segments_str = self.segmenter.segment(text)
        self.assertGreater(len(segments_str), 0, "Should segment string input")
        
        # Test dict input (new format)
        segments_dict = self.segmenter.segment({"full_text": text})
        self.assertGreater(len(segments_dict), 0, "Should segment dict input")
        
        # Results should be equivalent
        self.assertEqual(len(segments_str), len(segments_dict),
                        "String and dict inputs should produce same number of segments")
        
        # Test with sections dict (with longer text)
        longer_text = "Esta es la primera parte del diagnóstico con información relevante y detallada."
        segments_sections = self.segmenter.segment({
            "sections": {
                "section1": {"text": longer_text},
                "section2": {"text": "Segunda parte del diagnóstico con más información detallada."}
            }
        })
        self.assertGreater(len(segments_sections), 0, "Should segment from sections")
        
        # Test empty inputs
        self.assertEqual(len(self.segmenter.segment("")), 0)
        self.assertEqual(len(self.segmenter.segment({})), 0)
        self.assertEqual(len(self.segmenter.segment(None)), 0)


if __name__ == '__main__':
    unittest.main()
