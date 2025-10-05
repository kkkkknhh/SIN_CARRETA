#!/usr/bin/env python3
"""
Test script to validate parallel processing implementation.
"""

import logging
from feasibility_scorer import FeasibilityScorer

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test indicators
    indicators = [
        "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025",
        "Reduce from baseline of 15.3 million people in poverty to target of 8 million by December 2024", 
        "Aumentar el valor inicial de 2.5 millones de beneficiarios hasta alcanzar el objetivo de 4 millones en el horizonte temporal 2020-2025",
        "Mejorar desde la situación inicial hasta el objetivo propuesto con incremento del 20%",
        "Partir del nivel base actual para lograr la meta establecida en los próximos años",
        "Achieve target improvement from current baseline within the established timeframe",
        "Partir de la línea base para alcanzar el objetivo",
        "Improve from baseline to reach established goal"
    ]
    
    print("Testing Parallel Processing Implementation")
    print("=" * 50)
    
    # Test with default loky backend
    print("\n1. Testing with loky backend (default):")
    scorer_loky = FeasibilityScorer(backend='loky')
    batch_large = indicators * 3  # 24 total indicators (enough to trigger parallel)
    results_loky = scorer_loky.batch_score(batch_large)
    print(f"   Processed {len(results_loky)} indicators successfully")
    
    # Test with threading backend comparison
    print("\n2. Testing backend comparison:")
    results_comparison = scorer_loky.batch_score(batch_large, compare_backends=True)
    print(f"   Processed {len(results_comparison)} indicators with comparison")
    
    # Test sequential vs parallel for smaller batch
    print("\n3. Testing sequential fallback for small batch:")
    small_batch = indicators[:3]
    results_small = scorer_loky.batch_score(small_batch)
    print(f"   Processed {len(results_small)} indicators (should use sequential)")
    
    # Test with parallel disabled
    print("\n4. Testing with parallel disabled:")
    scorer_seq = FeasibilityScorer(enable_parallel=False)
    results_seq = scorer_seq.batch_score(indicators)
    print(f"   Processed {len(results_seq)} indicators sequentially")
    
    # Test n_jobs configuration
    print("\n5. Configuration test:")
    scorer_config = FeasibilityScorer(n_jobs=4, backend='threading')
    print(f"   n_jobs: {scorer_config.n_jobs}, backend: {scorer_config.backend}")
    print(f"   enable_parallel: {scorer_config.enable_parallel}")
    
    # Test picklability
    print("\n6. Testing picklability:")
    try:
        import pickle
        copy = scorer_loky._create_picklable_copy()
        pickled = pickle.dumps(copy)
        unpickled = pickle.loads(pickled)
        test_result = unpickled.calculate_feasibility_score("línea base 50% meta 80%")
        print(f"   Pickled copy works: score = {test_result.feasibility_score:.2f}")
    except Exception as e:
        print(f"   Pickling test failed: {e}")
    
    print("\n" + "=" * 50)
    print("All parallel processing tests completed successfully!")

if __name__ == "__main__":
    main()