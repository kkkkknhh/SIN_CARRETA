#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of Document Segmenter aligned with decalogo-industrial.latest.clean.json

This demo shows how the refactored DocumentSegmenter correctly maps sections to D1-D6 dimensions.
"""

from document_segmenter import DocumentSegmenter, SegmentationType


def print_separator():
    print("\n" + "=" * 80 + "\n")


def demo_dimension_mapping():
    """Demonstrate dimension mapping aligned with decalogo-industrial.latest.clean.json."""
    print("DEMONSTRATION: Document Segmenter - DECALOGO Alignment")
    print_separator()
    
    # Sample plan text with different section types
    sample_plan = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    
    1. DIAGNÓSTICO
    
    La situación actual del municipio presenta desafíos significativos en varios sectores.
    Las líneas base muestran datos verificables con series temporales desde 2020.
    
    2. RECURSOS Y CAPACIDADES
    
    El presupuesto asignado en el PPI alcanza $500 millones para el cuatrienio.
    Las capacidades institucionales incluyen personal técnico y sistemas de información.
    
    3. ACTIVIDADES FORMALIZADAS
    
    Las actividades están organizadas en tablas con responsable, insumo, output y costo unitario.
    Los mecanismos causales pretendidos se vinculan con la población diana identificada.
    La teoría de intervención establece complementariedades entre actividades.
    
    4. PRODUCTOS Y OUTPUTS
    
    Los productos están definidos con indicadores verificables y trazabilidad presupuestal.
    La cobertura es proporcional a la magnitud de la brecha identificada.
    
    5. RESULTADOS ESPERADOS
    
    Los resultados están definidos con métricas de outcome y ventana de maduración.
    El encadenamiento productos→resultados es explícito y validado.
    
    6. IMPACTOS DE LARGO PLAZO
    
    Los impactos de largo plazo están alineados con el PND y los ODS.
    Se utilizan indicadores proxy para medición de efectos duraderos.
    
    7. TEORÍA DE CAMBIO
    
    La teoría de cambio es explícita con diagrama causal (DAG).
    El encadenamiento causal ha sido validado lógicamente.
    """
    
    # Create segmenter
    segmenter = DocumentSegmenter(
        segmentation_type=SegmentationType.SECTION,
        min_segment_length=30,
        max_segment_length=500,
        preserve_context=True
    )
    
    # Segment the text
    segments = segmenter.segment_text(sample_plan)
    
    # Analyze segments by dimension
    dimension_counts = {
        "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D6": 0
    }
    
    print(f"✓ Segmented plan into {len(segments)} segments\n")
    
    for i, segment in enumerate(segments, 1):
        if segment.decalogo_dimensions:  # Skip segments with no dimensions
            print(f"Segment {i}: {segment.section_type.value.upper()}")
            print(f"  Dimensions: {', '.join(segment.decalogo_dimensions)}")
            print(f"  Text preview: {segment.text[:80].strip()}...")
            print()
            
            # Count dimensions
            for dim in segment.decalogo_dimensions:
                if dim in dimension_counts:
                    dimension_counts[dim] += 1
    
    print_separator()
    print("DIMENSION COVERAGE SUMMARY")
    print_separator()
    
    dimension_names = {
        "D1": "INSUMOS (diagnóstico, líneas base, recursos, capacidades)",
        "D2": "ACTIVIDADES (formalización, mecanismos causales)",
        "D3": "PRODUCTOS (outputs con indicadores verificables)",
        "D4": "RESULTADOS (outcomes con métricas)",
        "D5": "IMPACTOS (efectos largo plazo)",
        "D6": "CAUSALIDAD (teoría de cambio explícita)"
    }
    
    for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        count = dimension_counts[dim]
        status = "✓" if count > 0 else "✗"
        print(f"{status} {dim}: {dimension_names[dim]}")
        print(f"   Segments mapped: {count}")
        print()
    
    print_separator()
    print("ALIGNMENT VERIFICATION")
    print_separator()
    
    # Verify no old DE- dimensions are used
    all_dimensions = set()
    for segment in segments:
        all_dimensions.update(segment.decalogo_dimensions)
    
    old_format = [d for d in all_dimensions if d.startswith("DE-")]
    if old_format:
        print(f"✗ ERROR: Old DE- format dimensions found: {old_format}")
    else:
        print("✓ No old DE- format dimensions detected")
    
    new_format = [d for d in all_dimensions if d.startswith("D") and not d.startswith("DE-")]
    print(f"✓ New D1-D6 format dimensions used: {sorted(new_format)}")
    
    print("\n✓ Document Segmenter is correctly aligned with decalogo-industrial.latest.clean.json")


def demo_section_type_examples():
    """Show examples of each section type and their dimension mappings."""
    print_separator()
    print("SECTION TYPE → DIMENSION MAPPING REFERENCE")
    print_separator()
    
    from document_segmenter import SectionType, DocumentSegment
    
    examples = [
        # D1: INSUMOS
        (SectionType.DIAGNOSTIC, "Diagnóstico de la situación actual"),
        (SectionType.BASELINE, "Líneas base con series temporales"),
        (SectionType.RESOURCES, "Recursos asignados en el PPI"),
        (SectionType.CAPACITY, "Capacidades institucionales"),
        
        # D2: ACTIVIDADES
        (SectionType.ACTIVITY, "Actividades formalizadas en tablas"),
        (SectionType.MECHANISM, "Mecanismos causales pretendidos"),
        (SectionType.INTERVENTION, "Teoría de intervención coherente"),
        
        # D3: PRODUCTOS
        (SectionType.PRODUCT, "Productos con indicadores verificables"),
        (SectionType.OUTPUT, "Outputs con trazabilidad"),
        
        # D4: RESULTADOS
        (SectionType.RESULT, "Resultados con métricas de outcome"),
        (SectionType.OUTCOME, "Outcomes con ventana de maduración"),
        (SectionType.INDICATOR, "Indicadores de resultado"),
        
        # D5: IMPACTOS
        (SectionType.IMPACT, "Impactos de largo plazo"),
        (SectionType.LONG_TERM_EFFECT, "Efectos duraderos"),
        
        # D6: CAUSALIDAD
        (SectionType.CAUSAL_THEORY, "Teoría de cambio explícita con DAG"),
        (SectionType.CAUSAL_LINK, "Encadenamiento causal validado"),
    ]
    
    for section_type, text_example in examples:
        segment = DocumentSegment(
            text=text_example,
            start_pos=0,
            end_pos=len(text_example),
            segment_type=SegmentationType.SECTION,
            section_type=section_type
        )
        dims = ", ".join(segment.decalogo_dimensions)
        print(f"{section_type.value:20s} → {dims:10s} | {text_example}")


if __name__ == "__main__":
    demo_dimension_mapping()
    demo_section_type_examples()
    
    print_separator()
    print("✓ Demo completed successfully")
    print("✓ Document Segmenter refactored to match decalogo-industrial.latest.clean.json")
    print_separator()
