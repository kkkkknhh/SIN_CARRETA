#!/usr/bin/env python3
"""Simple pipeline execution test"""
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Test if imports work
try:
    from unified_evaluation_pipeline import UnifiedEvaluationPipeline
    print("✅ Unified pipeline imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Initialize pipeline
try:
    pipeline = UnifiedEvaluationPipeline(
        repo_root=".",
        rubric_path="RUBRIC_SCORING.json"
    )
    print("✅ Pipeline initialized")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Run evaluation
try:
    pdf_path = "ITUANGO - PLAN DE DESARROLLO.pdf"
    print(f"\n🚀 Running evaluation on {pdf_path}...")
    
    results = pipeline.evaluate(
        pdm_path=pdf_path,
        municipality="Ituango",
        department="Antioquia",
        export_json=True,
        output_dir="output"
    )
    
    print(f"\n✅ Evaluation completed: {results['status']}")
    print(f"   Execution time: {results['metadata']['execution_time_seconds']:.2f}s")
    print(f"   Evidence items: {results['evidence_registry']['statistics']['total_evidence']}")
    
except Exception as e:
    print(f"\n❌ Execution error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ Pipeline test completed successfully")
