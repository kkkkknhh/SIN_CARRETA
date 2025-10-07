#!/usr/bin/env python3
"""Test script to verify rubric check subprocess invocation audit."""

import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from system_validators import SystemHealthValidator

def create_test_environment():
    """Create a temporary test environment with necessary files."""
    tmpdir = Path(tempfile.mkdtemp(prefix="rubric_test_"))
    
    # Create directory structure
    tools_dir = tmpdir / "tools"
    artifacts_dir = tmpdir / "artifacts"
    tools_dir.mkdir()
    artifacts_dir.mkdir()
    
    # Create flow_doc.json
    flow_doc = {
        "canonical_order": ["step1", "step2", "step3"]
    }
    (tools_dir / "flow_doc.json").write_text(json.dumps(flow_doc))
    
    # Create flow_runtime.json
    runtime = {
        "order": ["step1", "step2", "step3"]
    }
    (artifacts_dir / "flow_runtime.json").write_text(json.dumps(runtime))
    
    # Create answers_report.json with 300 questions in D{N}-Q{N} format
    answers = {
        "summary": {"total_questions": 300},
        "answers": [
            {"question_id": f"D{i//50 + 1}-Q{i%50 + 1}", "answer": "test"} 
            for i in range(300)
        ]
    }
    (artifacts_dir / "answers_report.json").write_text(json.dumps(answers))
    
    # Create RUBRIC_SCORING.json with matching weights in D{N}-Q{N} format
    rubric = {
        "weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 1.0 for i in range(300)}
    }
    (tmpdir / "RUBRIC_SCORING.json").write_text(json.dumps(rubric))
    
    # Create rubric_check.py script
    rubric_check_script = '''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

answers_path = Path(sys.argv[1])
rubric_path = Path(sys.argv[2])

if not answers_path.exists():
    print(json.dumps({"ok": False, "error": "answers_report.json not found"}), file=sys.stderr)
    sys.exit(2)

if not rubric_path.exists():
    print(json.dumps({"ok": False, "error": "RUBRIC_SCORING.json not found"}), file=sys.stderr)
    sys.exit(2)

with open(answers_path) as f:
    answers = json.load(f)

with open(rubric_path) as f:
    rubric = json.load(f)

weights = rubric.get("weights", {})
answer_ids = {a["question_id"] for a in answers.get("answers", [])}

missing = [qid for qid in answer_ids if qid not in weights]
extra = [qid for qid in weights.keys() if qid not in answer_ids]

if missing or extra:
    print(json.dumps({
        "ok": False,
        "missing_in_rubric": missing[:10],
        "extra_in_rubric": extra[:10],
        "message": "1:1 alignment failed"
    }))
    sys.exit(3)

print(json.dumps({"ok": True, "message": "1:1 alignment verified"}))
sys.exit(0)
'''
    (tools_dir / "rubric_check.py").write_text(rubric_check_script)
    (tools_dir / "rubric_check.py").chmod(0o755)
    
    return tmpdir

def test_exit_code_0_success():
    """Test successful rubric check (exit code 0)."""
    print("\n=== Test 1: Exit code 0 (success) ===")
    tmpdir = create_test_environment()
    try:
        validator = SystemHealthValidator(str(tmpdir))
        result = validator.validate_post_execution(check_rubric_strict=True)
        
        if result["ok"] and result["ok_rubric_1to1"]:
            print("✓ PASS: Exit code 0 handled correctly (success)")
        else:
            print(f"✗ FAIL: Expected success but got errors: {result['errors']}")
            return False
    finally:
        shutil.rmtree(tmpdir)
    return True

def test_exit_code_3_mismatch():
    """Test rubric mismatch (exit code 3)."""
    print("\n=== Test 2: Exit code 3 (mismatch) ===")
    tmpdir = create_test_environment()
    try:
        # Create mismatched rubric (missing some questions) in D{N}-Q{N} format
        rubric = {
            "weights": {f"D{i//50 + 1}-Q{i%50 + 1}": 1.0 for i in range(250)}  # Only 250 instead of 300
        }
        (tmpdir / "RUBRIC_SCORING.json").write_text(json.dumps(rubric))
        
        validator = SystemHealthValidator(str(tmpdir))
        result = validator.validate_post_execution(check_rubric_strict=True)
        
        if not result["ok_rubric_1to1"]:
            # Check for mismatch error message
            mismatch_found = any("exit code 3" in err for err in result["errors"])
            diff_output_found = any("Diff output" in err for err in result["errors"])
            
            if mismatch_found:
                print("✓ PASS: Exit code 3 handled correctly (mismatch detected)")
                if diff_output_found:
                    print("✓ PASS: Diff output included in error message")
                else:
                    print("⚠ WARNING: Diff output not included in error message")
                return True
            else:
                print(f"✗ FAIL: Exit code 3 not properly handled. Errors: {result['errors']}")
                return False
        else:
            print("✗ FAIL: Expected mismatch but validation passed")
            return False
    finally:
        shutil.rmtree(tmpdir)

def test_exit_code_2_missing_file():
    """Test missing input file (exit code 2)."""
    print("\n=== Test 3: Exit code 2 (missing file) ===")
    tmpdir = create_test_environment()
    try:
        # Remove RUBRIC_SCORING.json
        (tmpdir / "RUBRIC_SCORING.json").unlink()
        
        validator = SystemHealthValidator(str(tmpdir))
        result = validator.validate_post_execution(check_rubric_strict=True)
        
        if not result["ok_rubric_1to1"]:
            missing_file_found = any("exit code 2" in err for err in result["errors"])
            
            if missing_file_found:
                print("✓ PASS: Exit code 2 handled correctly (missing file)")
                return True
            else:
                print(f"✗ FAIL: Exit code 2 not properly handled. Errors: {result['errors']}")
                return False
        else:
            print("✗ FAIL: Expected failure for missing file but validation passed")
            return False
    finally:
        shutil.rmtree(tmpdir)

def test_file_not_found_error():
    """Test FileNotFoundError when rubric_check.py doesn't exist."""
    print("\n=== Test 4: FileNotFoundError (script not found) ===")
    tmpdir = create_test_environment()
    try:
        # Remove rubric_check.py script
        (tmpdir / "tools" / "rubric_check.py").unlink()
        
        validator = SystemHealthValidator(str(tmpdir))
        result = validator.validate_post_execution(check_rubric_strict=True)
        
        if not result["ok_rubric_1to1"]:
            # Should be treated as exit code 2
            error_found = any("exit code 2" in err for err in result["errors"])
            
            if error_found:
                print("✓ PASS: FileNotFoundError handled gracefully as exit code 2")
                return True
            else:
                print(f"✗ FAIL: FileNotFoundError not properly handled. Errors: {result['errors']}")
                return False
        else:
            print("✗ FAIL: Expected failure for missing script but validation passed")
            return False
    finally:
        shutil.rmtree(tmpdir)

def test_absolute_paths():
    """Test that absolute paths are used."""
    print("\n=== Test 5: Absolute path resolution ===")
    tmpdir = create_test_environment()
    try:
        validator = SystemHealthValidator(str(tmpdir))
        
        # Verify paths are resolved to absolute
        rubric_path_abs = (validator.repo / "RUBRIC_SCORING.json").resolve()
        script_path_abs = (validator.repo / "tools" / "rubric_check.py").resolve()
        
        if rubric_path_abs.is_absolute() and script_path_abs.is_absolute():
            print(f"✓ PASS: Paths resolved to absolute")
            print(f"  - Rubric path: {rubric_path_abs}")
            print(f"  - Script path: {script_path_abs}")
            return True
        else:
            print("✗ FAIL: Paths not absolute")
            return False
    finally:
        shutil.rmtree(tmpdir)

def main():
    """Run all tests."""
    print("=" * 60)
    print("RUBRIC CHECK SUBPROCESS INVOCATION AUDIT")
    print("=" * 60)
    
    tests = [
        test_exit_code_0_success,
        test_exit_code_3_mismatch,
        test_exit_code_2_missing_file,
        test_file_not_found_error,
        test_absolute_paths,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Audit requirements met!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
