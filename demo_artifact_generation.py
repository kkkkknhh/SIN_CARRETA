#!/usr/bin/env python3
"""
Demo Artifact Generation
=========================
Demonstrates that export_artifacts() generates all 5 required artifacts
by mocking the necessary data structures.
"""

import json
import tempfile
from pathlib import Path
from miniminimoon_orchestrator import CanonicalDeterministicOrchestrator

def create_mock_pipeline_results():
    """Create mock pipeline results with all required sections."""
    return {
        "plan_path": "data/test_plan.txt",
        "orchestrator_version": "2.1.0",
        "start_time": "2024-10-07T12:00:00.000000",
        "end_time": "2024-10-07T12:05:00.000000",
        "evidence_hash": "a" * 64,
        "stages_completed": [
            "sanitization",
            "plan_processing",
            "document_segmentation",
            "embedding",
            "responsibility_detection",
            "contradiction_detection",
            "monetary_detection",
            "feasibility_scoring",
            "causal_detection",
            "teoria_cambio",
            "dag_validation",
            "evidence_registry_build",
            "decalogo_evaluation",
            "questionnaire_evaluation",
            "answers_assembly"
        ],
        "evaluations": {
            "answers_report": {
                "metadata": {
                    "version": "2.0",
                    "timestamp": "2024-10-07T12:05:00",
                    "evaluator": "mock_evaluator"
                },
                "global_summary": {
                    "total_questions": 300,
                    "answered_questions": 300,
                    "avg_confidence": 0.78,
                    "avg_score": 0.72
                },
                "question_answers": [
                    {
                        "question_id": f"D{(i//50)+1}-Q{(i%50)+1}",
                        "dimension": f"D{(i//50)+1}",
                        "evidence_ids": [f"ev_{i}"],
                        "confidence": 0.75,
                        "raw_score": 0.70,
                        "rationale": f"Rationale for question {i}",
                        "rubric_weight": 1.0 / 300
                    }
                    for i in range(300)
                ]
            }
        },
        "validation": {
            "flow_valid": True,
            "flow_hash": "b" * 64
        }
    }


def main():
    """Main demo execution."""
    print("="*80)
    print("ARTIFACT GENERATION DEMO")
    print("="*80)
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "demo_artifacts"
        
        print(f"\nGenerating artifacts in temporary directory: {output_dir}\n")
        
        # Create orchestrator instance
        config_dir = Path("config")
        orchestrator = CanonicalDeterministicOrchestrator(
            config_dir=config_dir,
            enable_validation=True
        )
        
        # Create mock pipeline results
        pipeline_results = create_mock_pipeline_results()
        
        # Export artifacts
        try:
            orchestrator.export_artifacts(output_dir, pipeline_results)
            print("\n✓ Artifact export completed")
        except Exception as e:
            print(f"\n⨯ Artifact export failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        # Verify all 5 artifacts were created
        print("\nVerifying generated artifacts:\n")
        
        required_artifacts = [
            "flow_runtime.json",
            "evidence_registry.json",
            "answers_report.json",
            "answers_sample.json",
            "coverage_report.json"
        ]
        
        all_present = True
        for artifact_name in required_artifacts:
            artifact_path = output_dir / artifact_name
            if artifact_path.exists():
                size_kb = artifact_path.stat().st_size / 1024
                print(f"  ✓ {artifact_name} ({size_kb:.1f} KB)")
                
                # Load and validate JSON
                try:
                    with open(artifact_path, 'r') as f:
                        data = json.load(f)
                    print(f"    - Valid JSON with {len(data)} top-level keys")
                except Exception as e:
                    print(f"    ⨯ Invalid JSON: {e}")
                    all_present = False
            else:
                print(f"  ⨯ {artifact_name} - NOT FOUND")
                all_present = False
        
        if all_present:
            print("\n" + "="*80)
            print("✓ SUCCESS: All 5 artifacts generated successfully")
            print("="*80)
            
            # Show sample content from coverage_report
            with open(output_dir / "coverage_report.json", 'r') as f:
                coverage = json.load(f)
            print(f"\nSample coverage_report.json content:")
            print(json.dumps(coverage, indent=2))
            
            return 0
        else:
            print("\n" + "="*80)
            print("⨯ FAILURE: Not all artifacts generated")
            print("="*80)
            return 1


if __name__ == "__main__":
    exit(main())
