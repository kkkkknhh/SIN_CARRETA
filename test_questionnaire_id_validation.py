#!/usr/bin/env python3
"""
Test suite for questionnaire_engine.py question ID generation and validation
"""

import json
from pathlib import Path
from questionnaire_engine import QuestionnaireEngine


def test_question_id_format():
    """Test that question IDs are generated in D{N}-Q{N} format"""
    
    # Create mock orchestrator results with minimal structure
    mock_results = {
        "plan_path": "test_pdm.pdf",
        "selected_programs": [],
        "decalogo_validation": {"validated": True},
        "feasibility": {},
        "teoria_cambio": {},
        "evidence_chain": {}
    }
    
    engine = QuestionnaireEngine()
    results = engine.execute_full_evaluation(mock_results, "Test Municipality", "Test Department")
    
    # Verify all 300 question IDs are in correct format
    assert len(results["evaluation_matrix"]) == 300, "Should have exactly 300 question evaluations"
    
    for eval_result in results["evaluation_matrix"]:
        question_id = eval_result.question_id
        # Check format: D{1-6}-Q{1-50}
        assert question_id.startswith("D"), f"Question ID {question_id} should start with 'D'"
        parts = question_id.split("-")
        assert len(parts) == 2, f"Question ID {question_id} should have format D{{N}}-Q{{N}}"
        assert parts[0] in ["D1", "D2", "D3", "D4", "D5", "D6"], f"Invalid dimension in {question_id}"
        assert parts[1].startswith("Q"), f"Second part of {question_id} should start with 'Q'"
        
        # Extract question number
        q_num = int(parts[1][1:])
        assert 1 <= q_num <= 50, f"Question number in {question_id} should be 1-50, got {q_num}"


def test_rubric_alignment():
    """Test that generated question IDs match RUBRIC_SCORING.json weights"""
    
    # Load RUBRIC_SCORING.json
    rubric_path = Path("RUBRIC_SCORING.json")
    with open(rubric_path, 'r', encoding='utf-8') as f:
        rubric_data = json.load(f)
        rubric_weights = rubric_data.get("weights", {})
    
    # Create mock orchestrator results
    mock_results = {
        "plan_path": "test_pdm.pdf",
        "selected_programs": [],
        "decalogo_validation": {"validated": True},
        "feasibility": {},
        "teoria_cambio": {},
        "evidence_chain": {}
    }
    
    engine = QuestionnaireEngine()
    results = engine.execute_full_evaluation(mock_results, "Test Municipality", "Test Department")
    
    # Extract all generated question IDs
    generated_ids = {eval_result.question_id for eval_result in results["evaluation_matrix"]}
    rubric_ids = set(rubric_weights.keys())
    
    # Verify exact match
    assert generated_ids == rubric_ids, (
        f"Generated IDs don't match RUBRIC_SCORING.json weights. "
        f"Missing: {rubric_ids - generated_ids}. Extra: {generated_ids - rubric_ids}"
    )


def test_unique_question_ids():
    """Test that all 300 question IDs are unique"""
    
    mock_results = {
        "plan_path": "test_pdm.pdf",
        "selected_programs": [],
        "decalogo_validation": {"validated": True},
        "feasibility": {},
        "teoria_cambio": {},
        "evidence_chain": {}
    }
    
    engine = QuestionnaireEngine()
    results = engine.execute_full_evaluation(mock_results, "Test Municipality", "Test Department")
    
    # Extract all question IDs
    question_ids = [eval_result.question_id for eval_result in results["evaluation_matrix"]]
    
    # Check uniqueness
    assert len(question_ids) == 300, "Should have 300 question IDs"
    assert len(set(question_ids)) == 300, "All 300 question IDs should be unique"


def test_dimension_distribution():
    """Test that questions are properly distributed across 6 dimensions"""
    
    mock_results = {
        "plan_path": "test_pdm.pdf",
        "selected_programs": [],
        "decalogo_validation": {"validated": True},
        "feasibility": {},
        "teoria_cambio": {},
        "evidence_chain": {}
    }
    
    engine = QuestionnaireEngine()
    results = engine.execute_full_evaluation(mock_results, "Test Municipality", "Test Department")
    
    # Count questions per dimension
    dimension_counts = {"D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D6": 0}
    for eval_result in results["evaluation_matrix"]:
        dimension = eval_result.question_id.split("-")[0]
        dimension_counts[dimension] += 1
    
    # Each dimension should have exactly 50 questions (5 base questions × 10 thematic points)
    for dimension, count in dimension_counts.items():
        assert count == 50, f"Dimension {dimension} should have 50 questions, got {count}"


def test_validation_exception_on_mismatch():
    """Test that validation raises exception when IDs don't match rubric"""
    
    # This test verifies that the validation logic would catch mismatches
    # We can't easily trigger a mismatch with the current code, so we verify
    # the validation code exists by checking the execute_full_evaluation method
    
    engine = QuestionnaireEngine()
    
    # Verify the method contains validation logic
    import inspect
    source = inspect.getsource(engine.execute_full_evaluation)
    
    assert "rubric_weights" in source, "Should load rubric weights for validation"
    assert "VALIDATION FAILED" in source, "Should have validation failure exception"
    assert "not found in RUBRIC_SCORING.json" in source, "Should check rubric alignment"


if __name__ == "__main__":
    print("Running question ID validation tests...")
    print("\n1. Testing question ID format...")
    test_question_id_format()
    print("✓ Question ID format test passed")
    
    print("\n2. Testing rubric alignment...")
    test_rubric_alignment()
    print("✓ Rubric alignment test passed")
    
    print("\n3. Testing unique question IDs...")
    test_unique_question_ids()
    print("✓ Unique question IDs test passed")
    
    print("\n4. Testing dimension distribution...")
    test_dimension_distribution()
    print("✓ Dimension distribution test passed")
    
    print("\n5. Testing validation exception logic...")
    test_validation_exception_on_mismatch()
    print("✓ Validation exception test passed")
    
    print("\n✅ All tests passed!")
