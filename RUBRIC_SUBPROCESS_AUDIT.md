# Rubric Check Subprocess Invocation Audit

## Summary

The `validate_post_execution()` function in `system_validators.py` has been audited and modified to invoke `tools/rubric_check.py` as a subprocess with comprehensive error handling.

## Implementation Details

### 1. Absolute Path Resolution

The implementation resolves all paths to absolute paths relative to the project root:

```python
rubric_path_abs = (self.repo / "RUBRIC_SCORING.json").resolve()
rubric_check_script_abs = (self.repo / "tools" / "rubric_check.py").resolve()
answers_path_abs = answers_path.resolve()
```

This ensures the subprocess call works regardless of the current working directory.

### 2. Subprocess Invocation

The function invokes the rubric check script as a subprocess:

```python
result = subprocess.run(
    [sys.executable, str(rubric_check_script_abs), str(answers_path_abs), str(rubric_path_abs)],
    capture_output=True,
    text=True,
    timeout=30
)
exit_code = result.returncode
stdout_output = result.stdout.strip()
stderr_output = result.stderr.strip()
```

### 3. Exit Code Handling

The implementation handles all exit codes from `tools/rubric_check.py`:

#### Exit Code 0 (Success)
- `ok_rubric_1to1` remains `True`
- No validation errors added
- Questions align 1:1 with rubric weights

#### Exit Code 2 (Missing Input Files)
- Sets `ok_rubric_1to1 = False`
- Appends error: `"Rubric check failed (exit code 2): Missing input file(s) - artifacts/answers_report.json or RUBRIC_SCORING.json not found"`
- Includes stdout/stderr details if available

#### Exit Code 3 (Mismatch)
- Sets `ok_rubric_1to1 = False`
- Appends error: `"Rubric mismatch (exit code 3): Questions in answers_report.json do not align with RUBRIC_SCORING.json weights"`
- Includes diff output from stdout: `"\nDiff output: {stdout_output}"`
- Includes error details from stderr: `"\nError details: {stderr_output}"`

#### Other Non-Zero Exit Codes
- Sets `ok_rubric_1to1 = False`
- Appends generic error with exit code and outputs

### 4. FileNotFoundError Handling

The implementation gracefully handles cases where `tools/rubric_check.py` doesn't exist:

```python
except FileNotFoundError as fnf_error:
    # Handle missing rubric_check.py script gracefully - treat as exit code 2
    exit_code = 2
    stderr_output = f"FileNotFoundError: {fnf_error}"
```

This treats missing script as equivalent to exit code 2 (missing input), ensuring consistent error handling.

### 5. Additional Exception Handling

The implementation includes handlers for:
- `subprocess.TimeoutExpired`: 30-second timeout with descriptive error
- Generic `Exception`: Catches unexpected errors with descriptive message

## Exit Code Reference

Based on `tools/rubric_check.py`:

| Exit Code | Meaning | Handling |
|-----------|---------|----------|
| 0 | Success - 1:1 alignment verified | No error, validation passes |
| 1 | Internal error (exception during check) | Generic error message |
| 2 | Missing input files (answers or rubric) | Missing file error with details |
| 3 | Mismatch - questions don't align 1:1 | Mismatch error with diff output |

## Error Message Format

### Exit Code 3 (Mismatch) Example:
```
Rubric mismatch (exit code 3): Questions in answers_report.json do not align with RUBRIC_SCORING.json weights
Diff output: {"ok": false, "missing_in_rubric": ["q250", "q251", ...], "extra_in_rubric": [], "message": "1:1 alignment failed"}
```

### Exit Code 2 (Missing File) Example:
```
Rubric check failed (exit code 2): Missing input file(s) - artifacts/answers_report.json or RUBRIC_SCORING.json not found
Error: {"ok": false, "error": "RUBRIC_SCORING.json not found"}
```

### FileNotFoundError Example:
```
Rubric check failed (exit code 2): Missing input file(s) - artifacts/answers_report.json or RUBRIC_SCORING.json not found
Error: FileNotFoundError: [Errno 2] No such file or directory: '/path/to/tools/rubric_check.py'
```

## Validation Flow

The rubric check is integrated into the post-execution validation flow:

1. **Artifact Verification**: Check for `flow_runtime.json` and `answers_report.json`
2. **Order Validation**: Verify canonical order matches runtime trace
3. **Contract Validation**: Validate against CanonicalFlowValidator
4. **Coverage Validation**: Ensure ≥ 300 questions
5. **Rubric Validation** (if `check_rubric_strict=True`):
   - Resolve absolute paths
   - Invoke `tools/rubric_check.py` subprocess
   - Capture exit code and outputs
   - Append validation errors if exit code is 2 or 3
   - Handle FileNotFoundError gracefully

## Testing

A comprehensive test suite has been created in `test_rubric_subprocess_audit.py` that verifies:

- ✓ Exit code 0 (success) handling
- ✓ Exit code 3 (mismatch) handling with diff output
- ✓ Exit code 2 (missing file) handling
- ✓ FileNotFoundError handling (treated as exit code 2)
- ✓ Absolute path resolution

## Files Modified

- `system_validators.py`: Updated `validate_post_execution()` method

## Files Created

- `test_rubric_subprocess_audit.py`: Comprehensive test suite for audit verification
- `RUBRIC_SUBPROCESS_AUDIT.md`: This documentation file
