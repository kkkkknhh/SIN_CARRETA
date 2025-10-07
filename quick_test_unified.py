#!/usr/bin/env python3
# Quick test runner for unified flow certification

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests"))

from test_unified_flow_certification import TestUnifiedFlowCertification

if __name__ == "__main__":
    # Just run the one big test
    suite = unittest.TestSuite()
    suite.addTest(TestUnifiedFlowCertification('test_complete_unified_pipeline_triple_run'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback[:200]}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback[:200]}")
    print("="*80)
    
    sys.exit(0 if result.wasSuccessful() else 1)
