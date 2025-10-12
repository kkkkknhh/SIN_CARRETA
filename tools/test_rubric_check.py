#!/usr/bin/env python3
"""Test suite for rubric_check.py"""

import json
import subprocess
import tempfile
import os


def run_rubric_check(answers_path, rubric_path):
    """Run rubric_check.py and return (stdout, stderr, returncode)"""
    result = subprocess.run(
        ['python3', 'tools/rubric_check.py', answers_path, rubric_path],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode


def test_matching_sets():
    """Test with perfectly matching question IDs"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_answers.json', delete=False) as f_ans:
        json.dump({
            "answers": [
                {"question_id": "Q1"},
                {"question_id": "Q2"},
                {"question_id": "Q3"}
            ]
        }, f_ans)
        answers_path = f_ans.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_rubric.json', delete=False) as f_rub:
        json.dump({
            "questions": [
                {"id": "Q1"},
                {"id": "Q2"},
                {"id": "Q3"}
            ]
        }, f_rub)
        rubric_path = f_rub.name
    
    try:
        _stdout, _stderr, returncode = run_rubric_check(answers_path, rubric_path)
        assert returncode == 0, f"Expected exit code 0, got {returncode}"
        result = json.loads(_stdout)
        assert result["match"] is True
        assert result["answers_count"] == 3
        assert result["rubric_count"] == 3
        assert result["missing_weights_count"] == 0
        assert result["extra_weights_count"] == 0
        print("✓ test_matching_sets passed")
    finally:
        os.unlink(answers_path)
        os.unlink(rubric_path)


def test_missing_weights():
    """Test with questions in answers but not in rubric"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_answers.json', delete=False) as f_ans:
        json.dump({
            "answers": [
                {"question_id": "Q1"},
                {"question_id": "Q2"},
                {"question_id": "Q3"}
            ]
        }, f_ans)
        answers_path = f_ans.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_rubric.json', delete=False) as f_rub:
        json.dump({
            "questions": [
                {"id": "Q1"}
            ]
        }, f_rub)
        rubric_path = f_rub.name
    
    try:
        _stdout, _stderr, returncode = run_rubric_check(answers_path, rubric_path)
        assert returncode == 3, f"Expected exit code 3, got {returncode}"
        result = json.loads(_stdout)
        assert result["match"] is False
        assert result["missing_weights_count"] == 2
        assert "Q2" in result["missing_weights"]
        assert "Q3" in result["missing_weights"]
        print("✓ test_missing_weights passed")
    finally:
        os.unlink(answers_path)
        os.unlink(rubric_path)


def test_extra_weights():
    """Test with weights in rubric but not in answers"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_answers.json', delete=False) as f_ans:
        json.dump({
            "answers": [
                {"question_id": "Q1"}
            ]
        }, f_ans)
        answers_path = f_ans.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_rubric.json', delete=False) as f_rub:
        json.dump({
            "questions": [
                {"id": "Q1"},
                {"id": "Q2"},
                {"id": "Q3"}
            ]
        }, f_rub)
        rubric_path = f_rub.name
    
    try:
        _stdout, _stderr, returncode = run_rubric_check(answers_path, rubric_path)
        assert returncode == 3, f"Expected exit code 3, got {returncode}"
        result = json.loads(_stdout)
        assert result["match"] is False
        assert result["extra_weights_count"] == 2
        assert "Q2" in result["extra_weights"]
        assert "Q3" in result["extra_weights"]
        print("✓ test_extra_weights passed")
    finally:
        os.unlink(answers_path)
        os.unlink(rubric_path)


def test_weights_dict_format():
    """Test with weights dictionary format"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_answers.json', delete=False) as f_ans:
        json.dump({
            "answers": [
                {"question_id": "Q1"},
                {"question_id": "Q2"}
            ]
        }, f_ans)
        answers_path = f_ans.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_rubric.json', delete=False) as f_rub:
        json.dump({
            "weights": {
                "Q1": 0.5,
                "Q2": 0.5
            }
        }, f_rub)
        rubric_path = f_rub.name
    
    try:
        _stdout, _stderr, returncode = run_rubric_check(answers_path, rubric_path)
        assert returncode == 0, f"Expected exit code 0, got {returncode}"
        result = json.loads(_stdout)
        assert result["match"] is True
        print("✓ test_weights_dict_format passed")
    finally:
        os.unlink(answers_path)
        os.unlink(rubric_path)


def test_file_not_found():
    """Test with nonexistent file"""
    _stdout, stderr, returncode = run_rubric_check('nonexistent.json', 'rubric_scoring.json')
    assert returncode == 2, f"Expected exit code 2, got {returncode}"
    error = json.loads(stderr)
    assert error["error"] == "file_read_error"
    print("✓ test_file_not_found passed")


def test_invalid_json():
    """Test with invalid JSON"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{invalid json')
        path = f.name
    
    try:
        _stdout, stderr, returncode = run_rubric_check(path, 'rubric_scoring.json')
        assert returncode == 2, f"Expected exit code 2, got {returncode}"
        error = json.loads(stderr)
        assert error["error"] == "invalid_json"
        print("✓ test_invalid_json passed")
    finally:
        os.unlink(path)


def test_missing_answers_key():
    """Test with missing 'answers' key"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_answers.json', delete=False) as f_ans:
        json.dump({"data": []}, f_ans)
        answers_path = f_ans.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_rubric.json', delete=False) as f_rub:
        json.dump({"questions": []}, f_rub)
        rubric_path = f_rub.name
    
    try:
        _stdout, stderr, returncode = run_rubric_check(answers_path, rubric_path)
        assert returncode == 2, f"Expected exit code 2, got {returncode}"
        error = json.loads(stderr)
        assert error["error"] == "missing_answers_key"
        print("✓ test_missing_answers_key passed")
    finally:
        os.unlink(answers_path)
        os.unlink(rubric_path)


def test_invalid_arguments():
    """Test with missing arguments"""
    result = subprocess.run(
        ['python3', 'tools/rubric_check.py'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"
    error = json.loads(result.stderr)
    assert error["error"] == "invalid_arguments"
    print("✓ test_invalid_arguments passed")


if __name__ == "__main__":
    print("Running rubric_check.py tests...\n")
    
    test_matching_sets()
    test_missing_weights()
    test_extra_weights()
    test_weights_dict_format()
    test_file_not_found()
    test_invalid_json()
    test_missing_answers_key()
    test_invalid_arguments()
    
    print("\n✓ All tests passed!")
