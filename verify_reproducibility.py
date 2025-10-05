"""
Script to verify reproducibility of the DAG validation system.
"""

from dag_validation import create_sample_causal_graph

def main():
    # Test reproducibility across multiple runs
    validator = create_sample_causal_graph()
    
    plan_name = "reproducibility_verification_test"
    
    print(f"Testing reproducibility for plan: {plan_name}")
    print("=" * 50)
    
    # Run the same test multiple times
    results = []
    for i in range(5):
        result = validator.calculate_acyclicity_pvalue(plan_name, 200)
        results.append(result)
        print(f"Run {i+1}: seed={result.seed}, p-value={result.p_value:.4f}, acyclic_count={result.acyclic_count}")
    
    # Verify all results are identical
    first_result = results[0]
    all_identical = all(
        r.seed == first_result.seed and 
        r.p_value == first_result.p_value and 
        r.acyclic_count == first_result.acyclic_count and
        r.subgraph_sizes == first_result.subgraph_sizes
        for r in results[1:]
    )
    
    print(f"\nAll results identical: {all_identical}")
    
    # Test different plan names produce different results
    print("\nTesting different plan names:")
    plan_names = ["plan_a", "plan_b", "plan_c"]
    different_results = []
    
    for plan in plan_names:
        result = validator.calculate_acyclicity_pvalue(plan, 100)
        different_results.append(result)
        print(f"{plan}: seed={result.seed}, p-value={result.p_value:.4f}")
    
    # Verify seeds are different
    seeds = [r.seed for r in different_results]
    all_seeds_different = len(set(seeds)) == len(seeds)
    
    print(f"All seeds different: {all_seeds_different}")
    
    return all_identical and all_seeds_different

if __name__ == "__main__":
    success = main()
    print(f"\nReproducibility verification: {'PASSED' if success else 'FAILED'}")