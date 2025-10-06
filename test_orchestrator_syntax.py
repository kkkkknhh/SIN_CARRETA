#!/usr/bin/env python3
"""Quick syntax validation for miniminimoon_orchestrator.py modifications"""

import ast
import sys

def validate_syntax(filename):
    """Parse Python file and check for syntax errors"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"✓ Syntax validation passed for {filename}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filename}: {e}")
        return False

if __name__ == "__main__":
    if validate_syntax("miniminimoon_orchestrator.py"):
        print("✓ All syntax checks passed")
        sys.exit(0)
    else:
        print("✗ Syntax validation failed")
        sys.exit(1)
