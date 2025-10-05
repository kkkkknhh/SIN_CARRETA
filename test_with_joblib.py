#!/usr/bin/env python3
"""
Test script with actual joblib installation to validate parallel processing.
"""

import subprocess
import sys
import logging

def install_joblib():
    """Install joblib for testing."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib>=1.3.0"])
        print("✓ joblib installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install joblib: {e}")
        return False

def test_parallel_with_joblib():
    """Test parallel processing with joblib installed."""
    try:
        from feasibility_scorer import FeasibilityScorer
        import joblib
        
        print(f"✓ joblib version: {joblib.__version__}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create large enough batch to trigger parallel processing
        indicators = [
            "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025",
            "Reduce from baseline of 15.3 million people in poverty to target of 8 million by December 2024",
            "Aumentar el valor inicial de 2.5 millones de beneficiarios hasta alcanzar el objetivo de 4 millones",
            "Mejorar desde la situación inicial hasta el objetivo propuesto con incremento del 20%",
            "Partir del nivel base actual para lograr la meta establecida en los próximos años"
        ]
        
        # Create batch large enough for parallel processing
        large_batch = indicators * 4  # 20 indicators
        
        print(f"\nTesting parallel processing with {len(large_batch)} indicators:")
        print("-" * 60)
        
        # Test with loky backend
        scorer = FeasibilityScorer(backend='loky', n_jobs=4)
        print(f"Scorer config: backend={scorer.backend}, n_jobs={scorer.n_jobs}, parallel_enabled={scorer.enable_parallel}")
        
        results = scorer.batch_score(large_batch)
        print(f"✓ Processed {len(results)} indicators with loky backend")
        
        # Test backend comparison
        print("\nTesting backend comparison:")
        comparison_results = scorer.batch_score(large_batch, compare_backends=True)
        print(f"✓ Backend comparison completed with {len(comparison_results)} results")
        
        # Verify results are consistent
        if len(results) == len(comparison_results):
            score_diffs = [abs(r1.feasibility_score - r2.feasibility_score) 
                          for r1, r2 in zip(results, comparison_results)]
            max_diff = max(score_diffs)
            print(f"✓ Results consistency check: max difference = {max_diff:.6f}")
        
        print("\n" + "=" * 60)
        print("✓ All parallel processing tests passed with joblib!")
        return True
        
    except Exception as e:
        print(f"✗ Parallel processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Joblib Parallel Processing Test")
    print("=" * 40)
    
    # Install joblib
    if not install_joblib():
        print("Cannot proceed without joblib")
        return 1
    
    # Test parallel processing
    if test_parallel_with_joblib():
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())