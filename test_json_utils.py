#!/usr/bin/env python3
"""
Test suite for JSON utilities with NaN/Inf handling
"""

import json
import os
import tempfile

import numpy as np
import pytest

from json_utils import (
    clean_data_for_json,
    safe_json_dump,
    safe_json_dumps,
    safe_json_write,
)


class TestDataCleaning:
    """Test data cleaning functionality"""

    @staticmethod
    def test_clean_nan_values():
        """Test NaN value replacement"""
        data = {"value": float("nan"), "normal": 42}
        cleaned = clean_data_for_json(data, nan_replacement=None)
        assert cleaned["value"] is None
        assert cleaned["normal"] == 42

        cleaned = clean_data_for_json(data, nan_replacement="NaN")
        assert cleaned["value"] == "NaN"

    @staticmethod
    def test_clean_inf_values():
        """Test infinity value replacement"""
        data = {"pos_inf": float("inf"), "neg_inf": float(
            "-inf"), "normal": 42}
        cleaned = clean_data_for_json(data)
        assert cleaned["pos_inf"] == "Infinity"
        assert cleaned["neg_inf"] == "-Infinity"
        assert cleaned["normal"] == 42

    @staticmethod
    def test_clean_nested_structures():
        """Test cleaning of nested data structures"""
        data = {
            "list": [1, float("nan"), 3, float("inf")],
            "nested_dict": {"inner_list": [float("-inf"), 2], "value": float("nan")},
            "tuple": (1, float("inf"), 3),
        }

        cleaned = clean_data_for_json(data)
        assert cleaned["list"] == [1, None, 3, "Infinity"]
        assert cleaned["nested_dict"]["inner_list"] == ["-Infinity", 2]
        assert cleaned["nested_dict"]["value"] is None
        assert cleaned["tuple"] == (1, "Infinity", 3)

    @staticmethod
    def test_numpy_array_cleaning():
        """Test NumPy array handling"""
        data = {
            "numpy_array": np.array([1, np.nan, np.inf, -np.inf, 5]),
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
        }

        cleaned = clean_data_for_json(data)
        assert cleaned["numpy_array"] == [1, None, "Infinity", "-Infinity", 5]
        assert cleaned["numpy_int"] == 42
        assert cleaned["numpy_float"] == 3.14
        assert isinstance(cleaned["numpy_int"], int)
        assert isinstance(cleaned["numpy_float"], float)

    @staticmethod
    def test_custom_replacements():
        """Test custom replacement values"""
        data = {"nan": float("nan"), "inf": float("inf"),
                "neg_inf": float("-inf")}
        cleaned = clean_data_for_json(
            data,
            nan_replacement="NOT_A_NUMBER",
            inf_replacement="POSITIVE_INFINITY",
            neg_inf_replacement="NEGATIVE_INFINITY",
        )

        assert cleaned["nan"] == "NOT_A_NUMBER"
        assert cleaned["inf"] == "POSITIVE_INFINITY"
        assert cleaned["neg_inf"] == "NEGATIVE_INFINITY"


class TestSafeJsonDump:
    """Test safe JSON dump functionality"""

    @staticmethod
    def test_safe_json_dump_with_special_values():
        """Test JSON dump with special float values"""
        data = {"nan": float("nan"), "inf": float("inf"), "normal": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            safe_json_dump(data, f)
            filename = f.name

        try:
            # Read back and verify
            with open(filename, "r") as f:
                result = json.load(f)

            assert result["nan"] is None
            assert result["inf"] == "Infinity"
            assert result["normal"] == [1, 2, 3]
        finally:
            os.unlink(filename)

    @staticmethod
    def test_default_json_parameters():
        """Test that default JSON parameters are applied"""
        data = {"test": "value", "unicode": "café"}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            safe_json_dump(data, f)
            filename = f.name

        try:
            # Read file content as text to check formatting
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for proper indentation and Unicode handling
            assert "café" in content  # ensure_ascii=False
            assert "\n" in content  # indent=2 should create newlines
        finally:
            os.unlink(filename)

    @staticmethod
    def test_parameter_override():
        """Test that parameters can be overridden"""
        data = {"test": "value"}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            safe_json_dump(data, f, indent=4, separators=(",", ": "))
            filename = f.name

        try:
            with open(filename, "r") as f:
                content = f.read()

            # Check that 4-space indentation was applied
            lines = content.split("\n")
            assert any('    "test": "value"' in line for line in lines)
        finally:
            os.unlink(filename)


class TestSafeJsonDumps:
    """Test safe JSON dumps functionality"""

    @staticmethod
    def test_safe_json_dumps_with_special_values():
        """Test JSON dumps with special float values"""
        data = {"nan": float("nan"), "inf": float("inf"),
                "list": [float("-inf"), 42]}

        result = safe_json_dumps(data)

        # Parse back to verify structure
        parsed = json.loads(result)
        assert parsed["nan"] is None
        assert parsed["inf"] == "Infinity"
        assert parsed["list"] == ["-Infinity", 42]

    @staticmethod
    def test_json_dumps_formatting():
        """Test JSON dumps formatting parameters"""
        data = {"key": "value", "number": 42}

        result = safe_json_dumps(data)

        # Should have proper formatting
        assert "\n" in result  # indentation creates newlines
        assert "value" in result


class TestSafeJsonWrite:
    """Test convenience write function"""

    @staticmethod
    def test_safe_json_write():
        """Test writing JSON to file"""
        data = {"nan": float("nan"), "array": np.array([1, np.inf, 3])}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            filename = f.name

        try:
            safe_json_write(data, filename)

            # Read back and verify
            with open(filename, "r") as f:
                result = json.load(f)

            assert result["nan"] is None
            assert result["array"] == [1, "Infinity", 3]
        finally:
            os.unlink(filename)

    @staticmethod
    def test_encoding_parameter():
        """Test custom encoding parameter"""
        data = {"unicode_test": "café"}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            filename = f.name

        try:
            safe_json_write(data, filename, encoding="utf-8")

            with open(filename, "r", encoding="utf-8") as f:
                result = json.load(f)

            assert result["unicode_test"] == "café"
        finally:
            os.unlink(filename)


class TestIntegration:
    """Integration tests for various scenarios"""

    @staticmethod
    def test_complex_nested_structure():
        """Test complex nested structure with all special value types"""
        data = {
            "metadata": {
                "version": 1.0,
                "stats": {
                    "mean": float("nan"),
                    "max": float("inf"),
                    "min": float("-inf"),
                    "values": [1, 2, float("nan"), 4, float("inf")],
                },
            },
            "results": [
                {"score": 0.95, "confidence": float("inf")},
                {"score": float("nan"), "confidence": 0.8},
            ],
            "numpy_data": np.array([[1, np.nan], [np.inf, 4]]),
            "mixed_tuple": (1, float("nan"), "text", float("inf")),
        }

        # Test round-trip through JSON
        json_str = safe_json_dumps(data)
        parsed = json.loads(json_str)

        # Verify structure is preserved and special values are handled
        assert parsed["metadata"]["stats"]["mean"] is None
        assert parsed["metadata"]["stats"]["max"] == "Infinity"
        assert parsed["metadata"]["stats"]["min"] == "-Infinity"
        assert parsed["results"][0]["confidence"] == "Infinity"
        assert parsed["results"][1]["score"] is None
        assert parsed["numpy_data"] == [[1, None], ["Infinity", 4]]
        assert parsed["mixed_tuple"] == [1, None, "text", "Infinity"]


if __name__ == "__main__":
    pytest.main([__file__])
