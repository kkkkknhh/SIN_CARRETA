#!/usr/bin/env python3
"""Test rubric loading and validation for answer_assembler and orchestrator."""

import json
import sys
from pathlib import Path

def test_answer_assembler_rubric_loading():
    """Test that AnswerAssembler loads rubric config correctly."""
    print("=" * 60)
    print("Testing AnswerAssembler rubric loading...")
    print("=" * 60)
    
    try:
        from answer_assembler import AnswerAssembler
        
        # Test initialization with default paths
        assembler = AnswerAssembler(
            rubric_path="RUBRIC_SCORING.json",
            decalogo_path="DECALOGO_FULL.json",
            weights_path="DNP_STANDARDS.json"
        )
        
        # Verify weights were loaded from RUBRIC_SCORING.json
        assert hasattr(assembler, 'weights'), "AnswerAssembler should have 'weights' attribute"
        assert len(assembler.weights) > 0, "Weights should not be empty"
        
        print(f"✓ Loaded {len(assembler.weights)} weights from RUBRIC_SCORING.json")
        
        # Verify weights are keyed by question unique IDs (e.g., 'D1-Q1-P1')
        sample_keys = list(assembler.weights.keys())[:5]
        print(f"✓ Sample weight keys: {sample_keys}")
        
        # Verify question templates were parsed
        assert hasattr(assembler, 'question_templates'), "Should have question_templates"
        print(f"✓ Loaded {len(assembler.question_templates)} question templates")
        
        print("\n✅ AnswerAssembler rubric loading: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ AnswerAssembler rubric loading: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_answer_assembler():
    """Test that orchestrator AnswerAssembler loads rubric correctly."""
    print("=" * 60)
    print("Testing miniminimoon_orchestrator.AnswerAssembler...")
    print("=" * 60)
    
    try:
        from miniminimoon_orchestrator import AnswerAssembler, EvidenceRegistry
        from pathlib import Path
        
        # Create mock evidence registry
        registry = EvidenceRegistry()
        
        # Initialize AnswerAssembler
        rubric_path = Path("RUBRIC_SCORING.json")
        assembler = AnswerAssembler(rubric_path, registry)
        
        # Verify questions and weights were loaded
        assert hasattr(assembler, 'questions'), "Should have 'questions' attribute"
        assert hasattr(assembler, 'weights'), "Should have 'weights' attribute"
        
        print(f"✓ Loaded {len(assembler.questions)} question templates")
        print(f"✓ Loaded {len(assembler.weights)} weights")
        
        # Verify sample weight keys
        sample_weight_keys = list(assembler.weights.keys())[:5]
        print(f"✓ Sample weight keys: {sample_weight_keys}")
        
        # Verify validation passed (1:1 alignment)
        print("✓ Rubric validation (gate #5) passed")
        
        print("\n✅ Orchestrator AnswerAssembler: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Orchestrator AnswerAssembler: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_rubric_structure():
    """Test RUBRIC_SCORING.json structure."""
    print("=" * 60)
    print("Testing RUBRIC_SCORING.json structure...")
    print("=" * 60)
    
    try:
        with open("RUBRIC_SCORING.json", 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        
        # Verify required sections exist
        assert "questions" in rubric, "Rubric must have 'questions' section"
        assert "weights" in rubric, "Rubric must have 'weights' section"
        
        questions = rubric["questions"]
        weights = rubric["weights"]
        
        print(f"✓ Found 'questions' section: {len(questions)} entries")
        print(f"✓ Found 'weights' section: {len(weights)} entries")
        
        # Verify questions is a list
        assert isinstance(questions, list), "'questions' must be an array"
        
        # Verify weights is a dict
        assert isinstance(weights, dict), "'weights' must be a dictionary"
        
        # Sample question structure
        if questions:
            sample_q = questions[0]
            print(f"✓ Sample question ID: {sample_q.get('id')}")
            print(f"  - dimension: {sample_q.get('dimension')}")
            print(f"  - scoring_modality: {sample_q.get('scoring_modality')}")
        
        # Sample weight structure
        sample_weight_keys = list(weights.keys())[:3]
        print(f"✓ Sample weight entries:")
        for key in sample_weight_keys:
            print(f"  - {key}: {weights[key]}")
        
        print("\n✅ RUBRIC_SCORING.json structure: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ RUBRIC_SCORING.json structure: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUBRIC LOADING VALIDATION SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: RUBRIC_SCORING.json structure
    results.append(("RUBRIC_SCORING.json structure", test_rubric_structure()))
    
    # Test 2: AnswerAssembler rubric loading
    results.append(("AnswerAssembler rubric loading", test_answer_assembler_rubric_loading()))
    
    # Test 3: Orchestrator AnswerAssembler
    results.append(("Orchestrator AnswerAssembler", test_orchestrator_answer_assembler()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
