#!/usr/bin/env python3
import unittest
import sys

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print('\n=== SUMMARY ===')
    print('Tests run:', result.testsRun)
    print('Failures:', len(result.failures))
    print('Errors:', len(result.errors))
    print('Skipped:', len(result.skipped))
    if result.failures or result.errors:
        print('\nDETAILS OF FAILURES/ERRORS:')
        for name, tb in result.failures + result.errors:
            print('---')
            print(name)
            print(tb)
            print('---')
    sys.exit(0 if (not result.failures and not result.errors) else 2)

