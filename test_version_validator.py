"""
Test suite for version validation functionality.
"""

import sys
import pytest
import warnings
from unittest.mock import patch
from version_validator import (
    validate_python_310,
    validate_numpy_compatibility,
    get_python_version_info
)

class TestVersionValidator:
    """Test version validation functionality."""
    
    def test_get_python_version_info(self):
        """Test version info retrieval."""
        version_info = get_python_version_info()
        assert isinstance(version_info, tuple)
        assert len(version_info) == 3
        assert all(isinstance(v, int) for v in version_info)
    
    def test_validate_python_310_success(self):
        """Test successful Python 3.10 validation."""
        # Should not raise if actually running Python 3.10
        if sys.version_info.major == 3 and sys.version_info.minor == 10:
            assert validate_python_310() is True
    
    def test_validate_python_310_failure(self):
        """Test Python version validation failure."""
        with patch('sys.version_info', (3, 9, 0)):
            with pytest.raises(RuntimeError) as exc_info:
                validate_python_310()
            assert "Python 3.10 required" in str(exc_info.value)
            assert "Found Python 3.9.0" in str(exc_info.value)
    
    def test_validate_python_310_wrong_major(self):
        """Test validation with wrong major version."""
        with patch('sys.version_info', (2, 7, 18)):
            with pytest.raises(RuntimeError) as exc_info:
                validate_python_310()
            assert "Python 3.10 required" in str(exc_info.value)
    
    @patch('version_validator.np', None)
    def test_validate_numpy_compatibility_not_installed(self):
        """Test NumPy validation when not installed."""
        with patch.dict('sys.modules', {'numpy': None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = validate_numpy_compatibility()
                assert result is False
                assert len(w) == 1
                assert "NumPy not installed" in str(w[0].message)
    
    def test_validate_numpy_compatibility_old_version(self):
        """Test NumPy validation with old version."""
        with patch('version_validator.np') as mock_np:
            mock_np.__version__ = "1.20.0"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = validate_numpy_compatibility()
                assert result is False
                assert len(w) == 1
                assert "may not fully support Python 3.10" in str(w[0].message)
    
    def test_validate_numpy_compatibility_new_version(self):
        """Test NumPy validation with potentially problematic new version."""
        with patch('version_validator.np') as mock_np:
            mock_np.__version__ = "1.25.0"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = validate_numpy_compatibility()
                assert result is True
                assert len(w) == 1
                assert "may have breaking changes" in str(w[0].message)
    
    def test_validate_numpy_compatibility_good_version(self):
        """Test NumPy validation with good version."""
        with patch('version_validator.np') as mock_np:
            mock_np.__version__ = "1.23.0"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = validate_numpy_compatibility()
                assert result is True
                # Should have no warnings for good version
                numpy_warnings = [warning for warning in w if "NumPy" in str(warning.message)]
                assert len(numpy_warnings) == 0

class TestCompatibilityChecker:
    """Test the comprehensive compatibility checker."""
    
    def test_checker_initialization(self):
        """Test checker can be imported and initialized."""
        from python_310_compatibility_checker import Python310CompatibilityChecker
        checker = Python310CompatibilityChecker()
        assert checker.python_version == sys.version_info
        assert checker.results == []
    
    def test_critical_modules_defined(self):
        """Test that critical modules are properly defined."""
        from python_310_compatibility_checker import Python310CompatibilityChecker
        checker = Python310CompatibilityChecker()
        
        # Check that NumPy is in critical modules (special attention requirement)
        assert 'numpy' in checker.CRITICAL_MODULES
        assert 'scipy' in checker.CRITICAL_MODULES
        assert 'torch' in checker.CRITICAL_MODULES
        
        # Should have reasonable number of critical modules
        assert len(checker.CRITICAL_MODULES) >= 5
    
    def test_import_result_structure(self):
        """Test ImportResult dataclass structure."""
        from python_310_compatibility_checker import ImportResult
        
        result = ImportResult(
            module_name="test_module",
            success=True,
            version="1.0.0"
        )
        
        assert result.module_name == "test_module"
        assert result.success is True
        assert result.version == "1.0.0"
        assert result.error_message is None
        assert result.warnings == []  # Should be initialized to empty list

if __name__ == "__main__":
    pytest.main([__file__])
