#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner script for test_unified_flow_certification.py

Executes the comprehensive end-to-end unified flow certification test
and captures detailed output.
"""

import sys
import unittest
from pathlib import Path

# Ensure tests directory is in path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "tests"))

# Import the test class
from test_unified_flow_certification import TestUnifiedFlowCertification

if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnifiedFlowCertification)
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
