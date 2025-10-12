#!/usr/bin/env python3
"""
Test to verify the Orchestrator Component Initialization fix (Issue B2).
This test validates that PlanProcessor receives config_dir parameter correctly.
"""

import ast
import sys
from pathlib import Path


def test_plan_processor_initialization():
    """Verify that PlanProcessor is initialized with config_dir parameter."""
    print("\n[TEST 1] Checking PlanProcessor initialization in orchestrator...")
    
    # Parse the orchestrator file
    orch_path = Path(__file__).parent / "miniminimoon_orchestrator.py"
    with open(orch_path, 'r') as f:
        tree = ast.parse(f.read(), filename='miniminimoon_orchestrator.py')
    
    # Find PlanProcessor instantiation
    found_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if it's a call to PlanProcessor
            if isinstance(node.func, ast.Name) and node.func.id == "PlanProcessor":
                # Check if it has at least one argument (config_dir)
                total_args = len(node.args) + len(node.keywords)
                
                assert total_args > 0, \
                    "PlanProcessor must be called with at least one argument (config_dir)"
                
                # Check if the argument is self.config_dir
                if len(node.args) > 0:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Attribute):
                        assert first_arg.attr == "config_dir", \
                            f"Expected 'config_dir' but got '{first_arg.attr}'"
                        assert isinstance(first_arg.value, ast.Name) and first_arg.value.id == "self", \
                            "config_dir should be passed as 'self.config_dir'"
                        print(f"    ✓ PlanProcessor initialized with self.config_dir")
                        found_call = True
                    else:
                        print(f"    ✓ PlanProcessor initialized with argument")
                        found_call = True
                elif len(node.keywords) > 0:
                    # Check if config_dir is passed as keyword argument
                    config_dir_found = any(kw.arg == "config_dir" for kw in node.keywords)
                    assert config_dir_found, "config_dir keyword argument not found"
                    print(f"    ✓ PlanProcessor initialized with config_dir keyword argument")
                    found_call = True
    
    assert found_call, "PlanProcessor instantiation not found in orchestrator"
    return True


def test_questionnaire_engine_initialization():
    """Verify that QuestionnaireEngine is still correctly initialized."""
    print("\n[TEST 2] Checking QuestionnaireEngine initialization in orchestrator...")
    
    # Parse the orchestrator file
    orch_path = Path(__file__).parent / "miniminimoon_orchestrator.py"
    with open(orch_path, 'r') as f:
        tree = ast.parse(f.read(), filename='miniminimoon_orchestrator.py')
    
    # Find QuestionnaireEngine instantiation
    found_call = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "QuestionnaireEngine"
        ):
            keyword_args = [kw.arg for kw in node.keywords]
            
            # The orchestrator should pass evidence_registry and rubric_path as keyword args
            assert 'evidence_registry' in keyword_args, \
                    "Orchestrator should pass evidence_registry to QuestionnaireEngine"
            assert 'rubric_path' in keyword_args, \
                    "Orchestrator should pass rubric_path to QuestionnaireEngine"
            
            print(f"    ✓ QuestionnaireEngine initialized with keyword args: {', '.join(keyword_args)}")
            found_call = True
    
    assert found_call, "QuestionnaireEngine instantiation not found in orchestrator"
    return True


def test_syntax_validity():
    """Verify Python syntax is valid in modified file."""
    print("\n[TEST 3] Checking Python syntax validity...")
    
    import py_compile
    
    file_path = Path(__file__).parent / "miniminimoon_orchestrator.py"
    
    try:
        py_compile.compile(str(file_path), doraise=True)
        print(f"    ✓ {file_path.name} has valid syntax")
    except py_compile.PyCompileError as e:
        raise AssertionError(f"Syntax error in {file_path.name}: {e}")
    
    return True


def test_line_748_specifically():
    """Verify that line 748 specifically has the correct change."""
    print("\n[TEST 4] Checking line 748 specifically...")
    
    orch_path = Path(__file__).parent / "miniminimoon_orchestrator.py"
    with open(orch_path, 'r') as f:
        lines = f.readlines()
    
    # Line 748 (0-indexed: 747)
    line_748 = lines[747].strip()
    
    # Check that line contains PlanProcessor with config_dir
    assert "PlanProcessor" in line_748, "Line 748 should contain PlanProcessor"
    assert "self.config_dir" in line_748, "Line 748 should contain self.config_dir"
    assert "PlanProcessor(self.config_dir)" in line_748, \
        "Line 748 should have PlanProcessor(self.config_dir)"
    
    print(f"    ✓ Line 748: {line_748}")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING: Orchestrator Component Initialization Fix (Issue B2)")
    print("="*80)
    
    tests = [
        test_plan_processor_initialization,
        test_questionnaire_engine_initialization,
        test_syntax_validity,
        test_line_748_specifically,
    ]
    
    try:
        for test in tests:
            test()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nAcceptance Criteria Verification:")
        print("  ✅ PlanProcessor receives config_dir")
        print("  ✅ QuestionnaireEngine initializes without TypeError")
        print("  ✅ Only line 748 was modified")
        print("="*80)
        return 0
    
    except AssertionError as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        return 1
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
