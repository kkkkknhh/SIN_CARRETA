#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLOW #3 Fix Validation Script

This script validates that the FLOW #3 fix has been correctly applied
by checking the source code for the expected changes.

It verifies:
1. doc_struct is captured from plan_processor output
2. doc_struct is passed to document_segmenter (not sanitized_text)
3. Contracts in deterministic_pipeline_validator match implementation
"""

import re
import sys
from pathlib import Path


def validate_flow_3_fix():
    """Validate FLOW #3 fix implementation"""
    
    print("=" * 70)
    print("FLOW #3 FIX VALIDATION")
    print("=" * 70)
    print()
    
    repo_root = Path(__file__).parent
    orchestrator_path = repo_root / "miniminimoon_orchestrator.py"
    validator_path = repo_root / "deterministic_pipeline_validator.py"
    segmenter_path = repo_root / "document_segmenter.py"
    
    # Check 1: Verify doc_struct is captured
    print("✓ Check 1: Verify doc_struct is captured from plan_processor")
    print("-" * 70)
    
    with open(orchestrator_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the pattern where doc_struct is assigned
    pattern1 = r'doc_struct\s*=\s*self\._run_stage\s*\(\s*PipelineStage\.PLAN_PROCESSING'
    match1 = re.search(pattern1, content)
    
    if match1:
        print("✅ PASS: doc_struct is properly captured from plan_processor")
        print(f"   Found at position: {match1.start()}")
    else:
        print("❌ FAIL: doc_struct is NOT captured (still using _)")
        return False
    print()
    
    # Check 2: Verify doc_struct is passed to document_segmenter
    print("✓ Check 2: Verify doc_struct is passed to document_segmenter")
    print("-" * 70)
    
    pattern2 = r'self\.document_segmenter\.segment\s*\(\s*doc_struct\s*\)'
    match2 = re.search(pattern2, content)
    
    if match2:
        print("✅ PASS: doc_struct is passed to document_segmenter.segment()")
        print(f"   Found at position: {match2.start()}")
    else:
        print("❌ FAIL: document_segmenter is NOT receiving doc_struct")
        # Check if it's still receiving sanitized_text (the bug)
        pattern2_bug = r'self\.document_segmenter\.segment\s*\(\s*sanitized_text\s*\)'
        match2_bug = re.search(pattern2_bug, content)
        if match2_bug:
            print("   ERROR: Still passing sanitized_text (bug not fixed!)")
        return False
    print()
    
    # Check 3: Verify contract specification
    print("✓ Check 3: Verify contract specification in validator")
    print("-" * 70)
    
    with open(validator_path, 'r', encoding='utf-8') as f:
        validator_content = f.read()
    
    # Check document_segmentation contract
    pattern3 = r'"document_segmentation".*?input_schema\s*=\s*\{\s*"doc_struct"\s*:\s*"dict"\s*\}'
    match3 = re.search(pattern3, validator_content, re.DOTALL)
    
    if match3:
        print("✅ PASS: Contract specifies doc_struct:dict as input")
    else:
        print("❌ FAIL: Contract does NOT specify correct input type")
        return False
    print()
    
    # Check 4: Verify document_segmenter accepts dict
    print("✓ Check 4: Verify document_segmenter.segment() accepts dict")
    print("-" * 70)
    
    with open(segmenter_path, 'r', encoding='utf-8') as f:
        segmenter_content = f.read()
    
    # Check method signature
    pattern4 = r'def\s+segment\s*\(\s*self\s*,\s*doc_struct\s*:\s*Union\s*\[\s*Dict\s*\[.*?\]\s*,\s*str\s*\]'
    match4 = re.search(pattern4, segmenter_content)
    
    if match4:
        print("✅ PASS: document_segmenter.segment() accepts Union[Dict, str]")
        print("   Backward compatible with both dict and str inputs")
    else:
        print("⚠️  WARNING: Could not verify method signature pattern")
        # Try simpler pattern
        pattern4_simple = r'def\s+segment\s*\(\s*self\s*,\s*doc_struct'
        match4_simple = re.search(pattern4_simple, segmenter_content)
        if match4_simple:
            print("✅ PASS: document_segmenter.segment() has doc_struct parameter")
        else:
            print("❌ FAIL: document_segmenter.segment() signature not found")
            return False
    print()
    
    # Check 5: Verify flow sequence
    print("✓ Check 5: Verify FLOW #2 → FLOW #3 sequence")
    print("-" * 70)
    
    # Look for the sequence of FLOW #2 and FLOW #3
    flow2_pattern = r'# Flow #2.*?PipelineStage\.PLAN_PROCESSING'
    flow3_pattern = r'# Flow #3.*?PipelineStage\.SEGMENTATION'
    
    flow2_match = re.search(flow2_pattern, content, re.DOTALL)
    flow3_match = re.search(flow3_pattern, content, re.DOTALL)
    
    if flow2_match and flow3_match and flow2_match.start() < flow3_match.start():
        print("✅ PASS: FLOW #2 (plan_processing) comes before FLOW #3 (segmentation)")
        print(f"   FLOW #2 position: {flow2_match.start()}")
        print(f"   FLOW #3 position: {flow3_match.start()}")
    else:
        print("❌ FAIL: Flow sequence is incorrect")
        return False
    print()
    
    # Summary
    print("=" * 70)
    print("VALIDATION RESULT: ✅ ALL CHECKS PASSED")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✅ doc_struct is captured from FLOW #2 output")
    print("  ✅ doc_struct is passed to FLOW #3 input")
    print("  ✅ Contracts specify correct types")
    print("  ✅ document_segmenter accepts dict input")
    print("  ✅ Flow sequence is correct")
    print()
    print("Conclusion: FLOW #3 fix is correctly implemented.")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = validate_flow_3_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
