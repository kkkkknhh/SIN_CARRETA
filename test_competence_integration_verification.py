"""Verification script for CompetenceValidator integration in evaluate_from_evidence."""

def verify_integration():
    """Verify all required integration elements are present."""
    
    with open('Decatalogo_principal.py', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    results = {
        'import_competence_validator': False,
        'instantiate_competence_validator': False,
        'per_question_validation': False,
        'validate_segment_call': False,
        'evidence_registration': False,
        'provenance_metadata': False,
        'error_handling': False,
        'results_in_output': False,
        'proper_ordering': False,
    }
    
    # Check imports
    results['import_competence_validator'] = (
        'from pdm_contra.policy.competence import CompetenceValidator' in content
    )
    
    # Check instantiation
    results['instantiate_competence_validator'] = (
        'competence_validator = CompetenceValidator()' in content
    )
    
    # Check per-question validation
    results['per_question_validation'] = (
        'question_competence_issues' in content and
        'COMPETENCE VALIDATION PER QUESTION' in content
    )
    
    # Check validate_segment call
    results['validate_segment_call'] = (
        'competence_validator.validate_segment(' in content
    )
    
    # Check evidence registration with proper ID
    results['evidence_registration'] = (
        'competence_validation::' in content and
        'evidence_registry.register(' in content
    )
    
    # Check provenance metadata
    results['provenance_metadata'] = (
        '"validator_source": "CompetenceValidator"' in content and
        '"provenance"' in content and
        '"module": "pdm_contra.policy.competence"' in content
    )
    
    # Check error handling
    results['error_handling'] = (
        'except Exception' in content and
        'log_warning_with_text' in content and
        '⚠️ Error validando competencias' in content or 
        '⚠️ Error en validación de competencias' in content
    )
    
    # Check results included in return value
    results['results_in_output'] = (
        '"question_competence_validations":' in content
    )
    
    # Check ordering: contradiction -> competence per question -> scoring
    contradiction_idx = -1
    competence_per_q_idx = -1
    scoring_idx = -1
    
    for i, line in enumerate(lines):
        if 'STEP 1: Run ContradictionDetector' in line:
            contradiction_idx = i
        if 'COMPETENCE VALIDATION PER QUESTION' in line:
            competence_per_q_idx = i
        if 'STEP 8: Return comprehensive evaluation results' in line:
            scoring_idx = i
    
    results['proper_ordering'] = (
        contradiction_idx > 0 and 
        competence_per_q_idx > 0 and 
        scoring_idx > 0 and
        contradiction_idx < competence_per_q_idx < scoring_idx
    )
    
    # Print results
    print("CompetenceValidator Integration Verification")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    print("=" * 60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ All integration checks PASSED")
    else:
        print("❌ Some integration checks FAILED")
        failed = [k for k, v in results.items() if not v]
        print(f"Failed checks: {', '.join(failed)}")
    
    print()
    print("Key Integration Points Found:")
    print(f"  - CompetenceValidator occurrences: {content.count('CompetenceValidator')}")
    print(f"  - question_competence_validations references: {content.count('question_competence_validations')}")
    print(f"  - Error handling blocks: {content.count('log_warning_with_text')}")
    print(f"  - Evidence registration: {content.count('competence_validation::')}")
    
    return all_passed


if __name__ == "__main__":
    success = verify_integration()
    exit(0 if success else 1)
