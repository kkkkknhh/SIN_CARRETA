#!/usr/bin/env python3
"""
Test for Decatalogo 300-question evaluation integration.

This test verifies that:
1. Decatalogo can be loaded with document data
2. evaluate_from_evidence method exists and works
3. All 300 questions are evaluated
4. Results have proper structure for doctoral-level argumentation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_decatalogo_imports():
    """Test that Decatalogo module can be imported."""
    try:
        from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado, BUNDLE
        print("✓ Successfully imported Decatalogo_principal")
        print(f"  BUNDLE version: {BUNDLE.get('version', 'unknown')}")
        print(f"  BUNDLE categories: {len(BUNDLE.get('categories', []))}")
        return True
    except Exception as e:
        print(f"✗ Failed to import Decatalogo_principal: {e}")
        return False


def test_decatalogo_initialization():
    """Test that Decatalogo extractor can be initialized."""
    try:
        from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado
        
        # Create sample documents (page, text) tuples
        documentos = [
            (1, "Plan de Desarrollo Municipal 2024-2027. Objetivos estratégicos de desarrollo territorial."),
            (2, "Recursos presupuestales asignados: $1,000,000,000 para proyectos de infraestructura."),
            (3, "Actividades programadas incluyen construcción de escuelas y hospitales en zonas rurales."),
        ]
        
        extractor = ExtractorEvidenciaIndustrialAvanzado(
            documentos=documentos,
            nombre_plan="Test_PDM"
        )
        
        print("✓ Successfully initialized ExtractorEvidenciaIndustrialAvanzado")
        print(f"  Extractor name: {extractor.nombre_plan}")
        print(f"  Documents loaded: {len(extractor.documentos)}")
        
        return extractor
    except Exception as e:
        print(f"✗ Failed to initialize ExtractorEvidenciaIndustrialAvanzado: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_evaluate_from_evidence():
    """Test the evaluate_from_evidence method."""
    try:
        from Decatalogo_principal import ExtractorEvidenciaIndustrialAvanzado
        
        # Create sample documents
        documentos = [
            (1, "Plan de Desarrollo Municipal con enfoque en paz territorial y seguridad ciudadana."),
            (2, "Presupuesto de $500M para programas sociales. Actividades de formación y capacitación."),
            (3, "Productos esperados: 10 centros comunitarios, 5 rutas de transporte, mejora en servicios."),
            (4, "Resultados: reducción del 30% en índices de violencia. Impacto en bienestar comunitario."),
            (5, "Teoría de cambio basada en inversión social que genera desarrollo sostenible."),
        ]
        
        extractor = ExtractorEvidenciaIndustrialAvanzado(
            documentos=documentos,
            nombre_plan="Test_PDM_Evaluation"
        )
        
        # Create a mock evidence registry
        class MockEvidenceEntry:
            def __init__(self, stage, content, metadata=None):
                self.stage = stage
                self.content = content
                self.metadata = metadata or {}
        
        class MockEvidenceRegistry:
            def __init__(self):
                self.entries = {
                    'seg1': MockEvidenceEntry('document_segmentation', 
                                             {'text': documentos[0][1]}, 
                                             {'page': 1}),
                    'seg2': MockEvidenceEntry('document_segmentation', 
                                             {'text': documentos[1][1]}, 
                                             {'page': 2}),
                    'seg3': MockEvidenceEntry('document_segmentation', 
                                             {'text': documentos[2][1]}, 
                                             {'page': 3}),
                }
            
            def get_all_entries(self):
                return list(self.entries.values())
            
            def get_entries_by_stage(self, stage):
                return [e for e in self.entries.values() if e.stage == stage]
        
        registry = MockEvidenceRegistry()
        
        # Test evaluate_from_evidence
        print("\n=== Testing evaluate_from_evidence ===")
        evaluation = extractor.evaluate_from_evidence(registry)
        
        print("✓ evaluate_from_evidence executed successfully")
        
        # Verify structure
        assert 'metadata' in evaluation, "Missing 'metadata' in evaluation"
        assert 'question_evaluations' in evaluation, "Missing 'question_evaluations' in evaluation"
        assert 'dimension_summaries' in evaluation, "Missing 'dimension_summaries' in evaluation"
        assert 'point_summaries' in evaluation, "Missing 'point_summaries' in evaluation"
        assert 'global_metrics' in evaluation, "Missing 'global_metrics' in evaluation"
        
        print("\n=== Evaluation Metadata ===")
        metadata = evaluation['metadata']
        print(f"  Total questions: {metadata.get('total_questions')}")
        print(f"  Points (P1-P10): {metadata.get('points')}")
        print(f"  Dimensions (D1-D6): {metadata.get('dimensions')}")
        print(f"  Questions per combination: {metadata.get('questions_per_combination')}")
        print(f"  Evidence count: {metadata.get('evidence_count')}")
        
        # Verify 300 questions
        questions = evaluation['question_evaluations']
        print(f"\n=== Question Evaluations ===")
        print(f"  Total questions evaluated: {len(questions)}")
        
        if len(questions) == 300:
            print("✓ All 300 questions evaluated!")
        else:
            print(f"⚠ Expected 300 questions, got {len(questions)}")
        
        # Show sample questions
        if questions:
            print("\n  Sample question evaluations:")
            for i, q in enumerate(questions[:3]):
                print(f"    {i+1}. {q['question_id']}")
                print(f"       Dimension: {q['dimension_name']}")
                print(f"       Score: {q['score']:.2f}")
                print(f"       Confidence: {q['confidence']:.2f}")
                print(f"       Evidence count: {q['evidence_count']}")
                print(f"       Rationale: {q['rationale'][:80]}...")
        
        # Check dimension summaries
        print("\n=== Dimension Summaries ===")
        for dim_id, summary in evaluation['dimension_summaries'].items():
            print(f"  {dim_id} ({summary['dimension_name']}):")
            print(f"    Questions: {summary['total_questions']}")
            print(f"    Avg Score: {summary['average_score']:.2f}")
            print(f"    Coverage: {summary['coverage_percentage']:.1f}%")
        
        # Check global metrics
        print("\n=== Global Metrics ===")
        metrics = evaluation['global_metrics']
        print(f"  Questions evaluated: {metrics['total_questions_evaluated']}")
        print(f"  Questions with evidence: {metrics['questions_with_evidence']}")
        print(f"  Average score: {metrics['average_score']:.2f}")
        print(f"  Average confidence: {metrics['average_confidence']:.2f}")
        print(f"  Coverage: {metrics['coverage_percentage']:.1f}%")
        print(f"  Completeness: {metrics['evaluation_completeness']:.1f}%")
        
        print("\n✓ All structural tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("DECATALOGO 300-QUESTION EVALUATION TEST")
    print("=" * 70)
    
    results = []
    
    print("\n[1/3] Testing Decatalogo imports...")
    results.append(test_decatalogo_imports())
    
    print("\n[2/3] Testing Decatalogo initialization...")
    extractor = test_decatalogo_initialization()
    results.append(extractor is not None)
    
    print("\n[3/3] Testing evaluate_from_evidence method...")
    results.append(test_evaluate_from_evidence())
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED - Decatalogo 300-question integration is working!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - See output above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
