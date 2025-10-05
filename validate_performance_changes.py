#!/usr/bin/env python3
"""
Simple validation script for performance optimization changes.
Verifies syntax without requiring full imports.
"""

import py_compile
import sys

files_to_check = [
    'data_flow_contract.py',
    'mathematical_invariant_guards.py',
    'performance_test_suite.py',
    'test_performance_optimizations.py',
]

print("=" * 80)
print("PERFORMANCE OPTIMIZATION VALIDATION")
print("=" * 80)

all_ok = True

for filepath in files_to_check:
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✅ {filepath:40s} - Syntax OK")
    except py_compile.PyCompileError as e:
        print(f"❌ {filepath:40s} - SYNTAX ERROR")
        print(f"   {e}")
        all_ok = False

print("\n" + "=" * 80)

if all_ok:
    print("✅ ALL FILES VALID - Ready for testing")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run tests: pytest test_performance_optimizations.py -v")
    print("  3. Run performance suite: python3 performance_test_suite.py")
    print("=" * 80)
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED - Fix syntax errors above")
    print("=" * 80)
    sys.exit(1)
