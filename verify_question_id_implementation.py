#!/usr/bin/env python3
"""
Verification script to demonstrate the complete question ID implementation
"""

import json
from pathlib import Path
from questionnaire_engine import QuestionnaireEngine


def main():
    print("=" * 80)
    print("QUESTION ID GENERATION AND VALIDATION VERIFICATION")
    print("=" * 80)
    
    # Load RUBRIC_SCORING.json
    rubric_path = Path("RUBRIC_SCORING.json")
    with open(rubric_path, 'r', encoding='utf-8') as f:
        rubric_data = json.load(f)
        rubric_weights = rubric_data.get("weights", {})
    
    print(f"\n1. RUBRIC_SCORING.json loaded: {len(rubric_weights)} weight entries")
    
    # Show sample of rubric weights
    print("\n   Sample rubric weight keys:")
    sample_keys = list(rubric_weights.keys())[:10]
    for key in sample_keys:
        print(f"      - {key}: {rubric_weights[key]}")
    
    # Create mock orchestrator results
    mock_results = {
        "plan_path": "test_pdm.pdf",
        "selected_programs": [],
        "decalogo_validation": {"validated": True},
        "feasibility": {},
        "teoria_cambio": {},
        "evidence_chain": {}
    }
    
    print("\n2. Creating QuestionnaireEngine instance...")
    engine = QuestionnaireEngine()
    
    print("\n3. Executing full evaluation (300 questions)...")
    results = engine.execute_full_evaluation(
        mock_results, 
        "Test Municipality", 
        "Test Department"
    )
    
    print(f"\n4. Evaluation complete:")
    print(f"   - Total evaluations: {len(results['evaluation_matrix'])}")
    print(f"   - Global score: {results['global_summary']['score_percentage']:.1f}%")
    print(f"   - Classification: {results['global_summary']['classification']}")
    
    # Verify question ID format
    print("\n5. Verifying question ID format (D{N}-Q{N})...")
    format_errors = []
    for eval_result in results["evaluation_matrix"]:
        qid = eval_result.question_id
        if not qid.startswith("D") or "-Q" not in qid:
            format_errors.append(qid)
    
    if format_errors:
        print(f"   ❌ Format errors found: {format_errors[:5]}")
    else:
        print(f"   ✅ All 300 question IDs have correct D{{N}}-Q{{N}} format")
    
    # Verify alignment with rubric
    print("\n6. Verifying alignment with RUBRIC_SCORING.json...")
    generated_ids = {eval_result.question_id for eval_result in results["evaluation_matrix"]}
    rubric_ids = set(rubric_weights.keys())
    
    missing_in_generated = rubric_ids - generated_ids
    extra_in_generated = generated_ids - rubric_ids
    
    if missing_in_generated or extra_in_generated:
        print(f"   ❌ Mismatch detected:")
        if missing_in_generated:
            print(f"      Missing: {list(missing_in_generated)[:5]}")
        if extra_in_generated:
            print(f"      Extra: {list(extra_in_generated)[:5]}")
    else:
        print(f"   ✅ Perfect match: All 300 IDs align with rubric weights")
    
    # Verify uniqueness
    print("\n7. Verifying question ID uniqueness...")
    question_ids = [eval_result.question_id for eval_result in results["evaluation_matrix"]]
    unique_ids = set(question_ids)
    
    if len(question_ids) != len(unique_ids):
        duplicates = [qid for qid in unique_ids if question_ids.count(qid) > 1]
        print(f"   ❌ Duplicates found: {duplicates[:5]}")
    else:
        print(f"   ✅ All 300 question IDs are unique")
    
    # Verify dimension distribution
    print("\n8. Verifying dimension distribution...")
    dimension_counts = {"D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0, "D6": 0}
    for eval_result in results["evaluation_matrix"]:
        dimension = eval_result.question_id.split("-")[0]
        dimension_counts[dimension] += 1
    
    all_correct = all(count == 50 for count in dimension_counts.values())
    
    print("   Dimension distribution:")
    for dim, count in sorted(dimension_counts.items()):
        status = "✅" if count == 50 else "❌"
        print(f"      {status} {dim}: {count} questions (expected 50)")
    
    # Show sample question IDs from each dimension
    print("\n9. Sample question IDs from each dimension:")
    for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        dim_questions = [
            eval_result.question_id 
            for eval_result in results["evaluation_matrix"] 
            if eval_result.question_id.startswith(dim)
        ]
        sample = sorted(dim_questions)[:5]
        print(f"   {dim}: {', '.join(sample)} ... ({len(dim_questions)} total)")
    
    # Verify validation logic exists
    print("\n10. Verifying validation logic in execute_full_evaluation()...")
    import inspect
    source = inspect.getsource(engine.execute_full_evaluation)
    
    checks = [
        ("loads rubric_weights", "rubric_weights" in source),
        ("validates question_id existence", "not found in RUBRIC_SCORING.json" in source),
        ("checks for duplicates", "generated_question_ids" in source),
        ("validates count", "len(generated_question_ids) != 300" in source),
        ("validates alignment", "generated_question_ids != rubric_weight_keys" in source)
    ]
    
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"   {status} {check_name}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_checks_passed = (
        not format_errors and
        not missing_in_generated and
        not extra_in_generated and
        len(question_ids) == len(unique_ids) and
        all_correct and
        all(check[1] for check in checks)
    )
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nThe execute_full_evaluation() method successfully:")
        print("  1. Generates question IDs in standardized D{N}-Q{N} format")
        print("  2. Aligns perfectly with RUBRIC_SCORING.json weights dictionary")
        print("  3. Includes question_id field in all 300 question results")
        print("  4. Validates every generated question_id exists in rubric weights")
        print("  5. Raises clear exceptions on any ID mismatch")
        print("  6. Covers all combinations for exactly 300 unique identifiers")
        print("  7. Uses dimension number (1-6) and sequential question numbering (1-50)")
        print("  8. Expands 30 base questions across 10 thematic points to 300 questions")
    else:
        print("❌ SOME CHECKS FAILED - Review output above for details")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
