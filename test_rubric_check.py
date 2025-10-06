#!/usr/bin/env python3
"""
test_rubric_check.py — Tests for rubric_check.py module
"""
import json
import tempfile
import pytest
from pathlib import Path
from rubric_check import verify_rubric_correspondence, load_json


def test_load_json_success():
    """Test successful JSON loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"test": "data"}, f)
        temp_path = Path(f.name)
    
    try:
        result = load_json(temp_path)
        assert result == {"test": "data"}
    finally:
        temp_path.unlink()


def test_load_json_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_json(Path("/nonexistent/file.json"))


def test_verify_rubric_correspondence_perfect_match():
    """Test perfect 1:1 correspondence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        answers = {
            "answers": [
                {"question_id": "q1"},
                {"question_id": "q2"},
                {"question_id": "q3"}
            ]
        }
        rubric = {
            "weights": {
                "q1": 10,
                "q2": 20,
                "q3": 30
            }
        }
        
        answers_path = tmpdir / "answers.json"
        rubric_path = tmpdir / "rubric.json"
        
        answers_path.write_text(json.dumps(answers))
        rubric_path.write_text(json.dumps(rubric))
        
        result = verify_rubric_correspondence(str(answers_path), str(rubric_path))
        
        assert result["ok"] is True
        assert result["missing_weights"] == []
        assert result["extra_weights"] == []
        assert result["total_questions"] == 3
        assert result["total_weights"] == 3


def test_verify_rubric_correspondence_missing_weights():
    """Test missing rubric weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        answers = {
            "answers": [
                {"question_id": "q1"},
                {"question_id": "q2"},
                {"question_id": "q3"}
            ]
        }
        rubric = {
            "weights": {
                "q1": 10
            }
        }
        
        answers_path = tmpdir / "answers.json"
        rubric_path = tmpdir / "rubric.json"
        
        answers_path.write_text(json.dumps(answers))
        rubric_path.write_text(json.dumps(rubric))
        
        result = verify_rubric_correspondence(str(answers_path), str(rubric_path))
        
        assert result["ok"] is False
        assert set(result["missing_weights"]) == {"q2", "q3"}
        assert result["extra_weights"] == []


def test_verify_rubric_correspondence_extra_weights():
    """Test extra rubric weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        answers = {
            "answers": [
                {"question_id": "q1"}
            ]
        }
        rubric = {
            "weights": {
                "q1": 10,
                "q2": 20,
                "q3": 30
            }
        }
        
        answers_path = tmpdir / "answers.json"
        rubric_path = tmpdir / "rubric.json"
        
        answers_path.write_text(json.dumps(answers))
        rubric_path.write_text(json.dumps(rubric))
        
        result = verify_rubric_correspondence(str(answers_path), str(rubric_path))
        
        assert result["ok"] is False
        assert result["missing_weights"] == []
        assert set(result["extra_weights"]) == {"q2", "q3"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
