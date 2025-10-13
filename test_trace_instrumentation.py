#!/usr/bin/env python3.10
"""Test trace instrumentation additions."""

import ast
import re
from pathlib import Path


def test_syntax():
    """Test that orchestrator has valid Python syntax."""
    try:
        code = Path("miniminimoon_orchestrator.py").read_text()
        ast.parse(code)
        print("✓ Syntax validation: PASS")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax validation: FAIL at line {e.lineno}: {e.msg}")
        return False


def test_stage_entry_logging():
    """Test that stage entry logging is present in _run_stage."""
    code = Path("miniminimoon_orchestrator.py").read_text()
    
    if 'event": "stage_entry"' in code and 'STAGE_ENTRY:' in code:
        print("✓ Stage entry logging: PRESENT")
        return True
    else:
        print("✗ Stage entry logging: MISSING")
        return False


def test_stage_exit_logging():
    """Test that stage exit logging is present in _run_stage."""
    code = Path("miniminimoon_orchestrator.py").read_text()
    
    if 'event": "stage_exit"' in code and 'STAGE_EXIT:' in code:
        print("✓ Stage exit logging: PRESENT")
        return True
    else:
        print("✗ Stage exit logging: MISSING")
        return False


def test_output_analysis():
    """Test that output artifact analysis is present."""
    code = Path("miniminimoon_orchestrator.py").read_text()
    
    if '_analyze_output_artifact' in code:
        print("✓ Output artifact analysis: PRESENT")
        return True
    else:
        print("✗ Output artifact analysis: MISSING")
        return False


def test_evidence_count():
    """Test that evidence counting is present."""
    code = Path("miniminimoon_orchestrator.py").read_text()
    
    if '_count_evidence_for_stage' in code:
        print("✓ Evidence counting: PRESENT")
        return True
    else:
        print("✗ Evidence counting: MISSING")
        return False


def test_contradiction_evidence():
    """Test that contradiction evidence registration was added."""
    code = Path("miniminimoon_orchestrator.py").read_text()
    
    if 'PipelineStage.CONTRADICTION' in code and 'contra"' in code:
        print("✓ Contradiction evidence registration: ADDED")
        return True
    else:
        print("✗ Contradiction evidence registration: MISSING")
        return False


def test_dag_evidence():
    """Test that DAG evidence registration was added."""
    code = Path("miniminimoon_orchestrator.py").read_text()
    
    if 'dag_evidence_id' in code and 'PipelineStage.DAG.value' in code:
        print("✓ DAG evidence registration: ADDED")
        return True
    else:
        print("✗ DAG evidence registration: MISSING")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRACE INSTRUMENTATION VERIFICATION")
    print("=" * 60)
    print()
    
    tests = [
        test_syntax,
        test_stage_entry_logging,
        test_stage_exit_logging,
        test_output_analysis,
        test_evidence_count,
        test_contradiction_evidence,
        test_dag_evidence,
    ]
    
    results = [test() for test in tests]
    
    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"✗ {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
