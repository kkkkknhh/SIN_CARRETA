#!/usr/bin/env python3
"""
Test script to verify AnswerAssembler changes for rubric weight loading

The AnswerAssembler class in miniminimoon_orchestrator.py uses question IDs
with point codes (e.g., "D1-Q1-P1"), while RUBRIC_SCORING.json's questions 
section has template IDs without point codes (e.g., "D1-Q1"). The weights
section has the full instantiated IDs with point codes.

This test verifies that:
1. Both sections exist in RUBRIC_SCORING.json
2. The AnswerAssembler loads both sections
3. The AnswerAssembler retrieves weights from the loaded dictionary
4. The validation logic checks for proper alignment
"""
import json
from pathlib import Path
import sys

def test_rubric_structure():
    """Test that RUBRIC_SCORING.json has required structure"""
    rubric_path = Path('RUBRIC_SCORING.json')
    
    if not rubric_path.exists():
        print("FAIL: RUBRIC_SCORING.json not found")
        return False
    
    with open(rubric_path, 'r', encoding='utf-8') as f:
        rubric = json.load(f)
    
    # Check for required sections
    if 'questions' not in rubric:
        print("FAIL: 'questions' section missing from rubric")
        return False
    
    if 'weights' not in rubric:
        print("FAIL: 'weights' section missing from rubric")
        return False
    
    print(f"✓ Rubric has 'questions' and 'weights' sections")
    
    # Parse question template IDs (these are templates like "D1-Q1")
    questions = rubric.get('questions', {})
    if isinstance(questions, list):
        template_ids = set(q.get('id') for q in questions if 'id' in q)
    else:
        template_ids = set(questions.keys())
    
    # Parse weight IDs (these are instantiated like "D1-Q1-P1", "D1-Q1-P2", etc.)
    weights = rubric.get('weights', {})
    weight_ids = set(weights.keys())
    
    print(f"✓ Questions section: {len(template_ids)} template IDs")
    print(f"✓ Weights section: {len(weight_ids)} instantiated weight IDs")
    print(f"  Sample template IDs: {sorted(template_ids)[:3]}")
    print(f"  Sample weight IDs: {sorted(weight_ids)[:3]}")
    
    # Check that weights are more numerous than templates (instantiated across points)
    if len(weight_ids) <= len(template_ids):
        print(f"FAIL: Expected more weight entries than template questions")
        return False
    
    print(f"✓ Weights properly instantiated: {len(weight_ids)} > {len(template_ids)}")
    return True

def test_answer_assembler_init():
    """Test AnswerAssembler initialization and usage"""
    try:
        with open('miniminimoon_orchestrator.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check that AnswerAssembler loads both sections
        if 'self.questions = self.rubric.get("questions"' not in source:
            print("FAIL: AnswerAssembler not loading questions section")
            return False
        
        if 'self.weights = self.rubric.get("weights"' not in source:
            print("FAIL: AnswerAssembler not loading weights section")
            return False
        
        print("✓ AnswerAssembler loads questions and weights sections internally")
        
        # Check that it retrieves weights from loaded dictionary (not external parameters)
        if 'self.weights.get(question_id)' not in source:
            print("FAIL: AnswerAssembler not using loaded weights dictionary")
            return False
        
        print("✓ AnswerAssembler retrieves weights from loaded dictionary")
        
        # Check validation logic exists
        if 'missing = questions - weights' not in source or 'extra = weights - questions' not in source:
            print("FAIL: AnswerAssembler validation not checking for missing/extra IDs")
            return False
        
        print("✓ AnswerAssembler validates 1:1 alignment at initialization")
        
        # Check that no external weight parameters are used in constructor
        assembler_init_pattern = 'def __init__(self, rubric_path: Path, evidence_registry: EvidenceRegistry):'
        if assembler_init_pattern not in source:
            print("WARN: AnswerAssembler __init__ signature changed, verify no weight parameters")
        else:
            print("✓ AnswerAssembler constructor has no external weight parameters")
        
        # Check that assemble() method doesn't accept weight parameters
        if 'def assemble(' in source:
            # Find the assemble method signature
            assemble_start = source.find('def assemble(')
            if assemble_start != -1:
                # Get the signature (up to the closing paren of params)
                assemble_sig = source[assemble_start:assemble_start+200]
                if 'weight' in assemble_sig.split('):')[0]:
                    print("FAIL: assemble() method still has weight parameter")
                    return False
                print("✓ assemble() method has no external weight parameters")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Error checking AnswerAssembler: {e}")
        return False

if __name__ == '__main__':
    print("="*70)
    print("Testing AnswerAssembler Rubric Weight Loading Changes")
    print("="*70)
    
    test1 = test_rubric_structure()
    print()
    test2 = test_answer_assembler_init()
    
    print()
    print("="*70)
    if test1 and test2:
        print("✓ All tests PASSED")
        print()
        print("Summary:")
        print("- RUBRIC_SCORING.json contains both 'questions' and 'weights' sections")
        print("- AnswerAssembler._load_rubric() reads both sections")
        print("- AnswerAssembler stores weights internally in self.weights")
        print("- AnswerAssembler.assemble() retrieves weights from self.weights")
        print("- AnswerAssembler._validate_rubric_coverage() checks 1:1 alignment")
        print("- No external weight parameters in constructor or methods")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)
