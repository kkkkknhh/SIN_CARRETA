#!/usr/bin/env python3
"""
INTEGRATION EXAMPLE: How to use the Authoritative Questionnaire Engine
======================================================================

This example shows how to integrate the new QuestionnaireEngine into your 
existing evaluation pipeline while maintaining backward compatibility.
"""

from questionnaire_engine import get_questionnaire_engine
from pathlib import Path
import json

def run_decatalogo_evaluation(pdm_path: str, municipality: str = "", department: str = ""):
    """
    MAIN EVALUATION FUNCTION: Enforces the exact 300-question structure
    
    This replaces the main evaluation logic in Decatalogo_evaluador.py
    """
    
    # Get the singleton engine instance
    engine = get_questionnaire_engine()
    
    print(f"🚀 Starting Decatalogo Evaluation")
    print(f"📄 PDM Document: {pdm_path}")
    print(f"🏛️ Municipality: {municipality}")
    print(f"🗺️ Department: {department}")
    print()
    
    # Execute the complete 300-question evaluation
    results = engine.execute_full_evaluation(
        pdm_document=pdm_path,
        municipality=municipality, 
        department=department
    )
    
    # Validate the results follow the exact structure
    validation_passed = engine.validate_execution(results)
    
    if not validation_passed:
        raise RuntimeError("CRITICAL: Evaluation did not follow the required 300-question structure!")
    
    return results

def export_results(results: dict, output_dir: str = "output"):
    """Export results in multiple formats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    evaluation_id = results["metadata"]["evaluation_id"]
    
    # JSON export (detailed results)
    json_file = output_path / f"evaluation_{evaluation_id}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # CSV export (question matrix)
    csv_file = output_path / f"evaluation_{evaluation_id}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("question_id,point_code,point_title,dimension,question_no,score,max_score\n")
        for evaluation in results["evaluation_matrix"]:
            f.write(f"{evaluation.question_id},{evaluation.point_code},"
                   f'"{evaluation.point_title}",{evaluation.dimension},'
                   f"{evaluation.question_no},{evaluation.score},{evaluation.max_score}\n")
    
    # Summary report
    summary_file = output_path / f"summary_{evaluation_id}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("DECATALOGO EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation ID: {results['metadata']['evaluation_id']}\n")
        f.write(f"Municipality: {results['metadata']['municipality']}\n")
        f.write(f"Department: {results['metadata']['department']}\n")
        f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
        f.write(f"Processing Time: {results['metadata']['processing_time_seconds']:.2f} seconds\n\n")
        
        f.write("GLOBAL RESULTS:\n")
        f.write(f"  Global Score: {results['summary']['global_score']:.2f}/3.0\n")
        f.write(f"  Total Evaluations: {results['summary']['total_evaluations']}\n")
        f.write(f"  Structure Validation: {'PASSED' if results['summary']['validation_passed'] else 'FAILED'}\n\n")
        
        f.write("DIMENSION AVERAGES:\n")
        for dim, score in results['summary']['dimension_averages'].items():
            f.write(f"  {dim}: {score:.2f}/3.0\n")
        
        f.write("\nSCORE DISTRIBUTION:\n")
        for level, count in results['summary']['score_distribution'].items():
            f.write(f"  {level.title()}: {count} questions\n")
        
        f.write("\nTHEMATIC POINT SCORES:\n")
        for point in results['thematic_points']:
            f.write(f"  {point['point_id']}: {point['total_score']:.2f}/3.0 - {point['point_title']}\n")
    
    print(f"📁 Results exported to: {output_path}")
    print(f"  📄 JSON: {json_file.name}")
    print(f"  📊 CSV: {csv_file.name}")
    print(f"  📋 Summary: {summary_file.name}")

def main():
    """Example usage of the integration"""
    
    # Example PDM document path (replace with actual path)
    pdm_path = "example_pdm.pdf"
    
    # Check if we have a real PDM to test with
    if not Path(pdm_path).exists():
        print("⚠️ No example PDM found. Using placeholder for demonstration.")
        pdm_path = "placeholder_pdm.pdf"
    
    try:
        # Run the evaluation with the new engine
        results = run_decatalogo_evaluation(
            pdm_path=pdm_path,
            municipality="Medellín",
            department="Antioquia"
        )
        
        # Export results
        export_results(results)
        
        print("\n✅ INTEGRATION COMPLETE")
        print("📊 Evaluation Results:")
        print(f"   🎯 Total Questions: {results['summary']['total_evaluations']}")
        print(f"   📈 Global Score: {results['summary']['global_score']:.2f}/3.0")
        print(f"   ✅ Structure Valid: {results['summary']['validation_passed']}")
        
    except FileNotFoundError:
        print("ℹ️ This is a demonstration - actual PDM file needed for full evaluation")
        
        # Show what the engine structure looks like
        engine = get_questionnaire_engine()
        print("\n📋 ENGINE STRUCTURE:")
        print(f"   🎯 {engine.structure.DOMAINS} thematic points")
        print(f"   📝 {engine.structure.QUESTIONS_PER_DOMAIN} questions per point")
        print(f"   📊 {engine.structure.TOTAL_QUESTIONS} total evaluations")

if __name__ == "__main__":
    main()