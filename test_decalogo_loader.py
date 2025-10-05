"""
Test suite for DECALOGO_INDUSTRIAL template loader.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open
import pytest
from decalogo_loader import (
    load_decalogo_industrial, 
    write_template_atomically,
    get_decalogo_industrial,
    DECALOGO_INDUSTRIAL_TEMPLATE
)


class TestDecalogoLoader(unittest.TestCase):
    """Tests for the DECALOGO_INDUSTRIAL template loader."""
    
    def setUp(self):
        # Reset cache before each test
        import decalogo_loader
        decalogo_loader._cached_template = None
    
    def test_load_existing_file(self):
        """Test loading from an existing file."""
        test_content = "Test DECALOGO content"
        with patch("builtins.open", mock_open(read_data=test_content)) as mock_file:
            with patch("os.path.exists", return_value=True):
                content = load_decalogo_industrial("test_path.txt")
                self.assertEqual(content, test_content)
                mock_file.assert_called_once_with("test_path.txt", "r", encoding="utf-8")
    
    def test_create_new_file(self):
        """Test creating a new file with template content."""
        with patch("os.path.exists", return_value=False):
            with patch("decalogo_loader.write_template_atomically", return_value=True) as mock_write:
                content = load_decalogo_industrial("test_path.txt")
                self.assertEqual(content, DECALOGO_INDUSTRIAL_TEMPLATE)
                mock_write.assert_called_once_with("test_path.txt", DECALOGO_INDUSTRIAL_TEMPLATE)
    
    def test_fallback_on_read_error(self):
        """Test fallback to template when file read fails."""
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", side_effect=IOError("Read error")):
                content = load_decalogo_industrial("test_path.txt")
                self.assertEqual(content, DECALOGO_INDUSTRIAL_TEMPLATE)
    
    def test_fallback_on_write_error(self):
        """Test fallback to template when file write fails."""
        with patch("os.path.exists", return_value=False):
            with patch("decalogo_loader.write_template_atomically", side_effect=OSError("Write error")):
                content = load_decalogo_industrial("test_path.txt")
                self.assertEqual(content, DECALOGO_INDUSTRIAL_TEMPLATE)
    
    def test_caching(self):
        """Test that template content is cached."""
        test_content = "Test DECALOGO content"
        with patch("builtins.open", mock_open(read_data=test_content)) as mock_file:
            with patch("os.path.exists", return_value=True):
                # First call should read the file
                content1 = load_decalogo_industrial("test_path.txt")
                # Second call should use cached content
                content2 = load_decalogo_industrial("test_path.txt")
                
                self.assertEqual(content1, test_content)
                self.assertEqual(content2, test_content)
                mock_file.assert_called_once()  # File should be read only once
    
    def test_write_template_atomically(self):
        """Test atomic writing of template content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_decalogo.txt")
            success = write_template_atomically(test_path, "Test content")
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(test_path))
            
            with open(test_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertEqual(content, "Test content")
    
    def test_write_template_with_directory_creation(self):
        """Test writing template with directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "test.txt")
            
            with patch("os.makedirs") as mock_makedirs:
                write_template_atomically(nested_path, "Test content")
                mock_makedirs.assert_called_once()
    
    def test_write_template_error_handling(self):
        """Test error handling during template writing."""
        with patch("tempfile.NamedTemporaryFile", side_effect=IOError("Temp file error")):
            success = write_template_atomically("test_path.txt", "Test content")
            self.assertFalse(success)
    
    def test_get_decalogo_industrial(self):
        """Test convenience wrapper function."""
        with patch("decalogo_loader.load_decalogo_industrial") as mock_load:
            mock_load.return_value = "Convenience wrapper test"
            content = get_decalogo_industrial()
            self.assertEqual(content, "Convenience wrapper test")
            mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
    def test_get_decalogo_convenience_function(tmp_path):
        """Test convenience function with caching."""
        cache_path = tmp_path / "cached_decalogo.txt"

        result = get_decalogo_industrial(str(cache_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert cache_path.exists()
        assert (
            cache_path.read_text(encoding="utf-8")
            == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        )

    @staticmethod
    def test_get_decalogo_no_cache(caplog):
        """Test convenience function without caching."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        result = get_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert "no target path specified" in caplog.text

    @staticmethod
    def test_temp_file_cleanup_on_error(tmp_path):
        """Test that temporary files are cleaned up when rename fails."""
        target_path = tmp_path / "decalogo.txt"

        with patch("decalogo_loader.os.replace") as mock_replace:
            mock_replace.side_effect = PermissionError("Cannot rename")

            load_decalogo_industrial(str(target_path))

            # Check that no temporary files are left behind
            temp_files = list(tmp_path.glob(".*_tmp_*"))
            assert len(temp_files) == 0

    @staticmethod
    def test_directory_creation(tmp_path):
        """Test that parent directories are created when they don't exist."""
        nested_path = tmp_path / "nested" / "dirs" / "decalogo.txt"

        result = load_decalogo_industrial(str(nested_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert (
            nested_path.read_text(encoding="utf-8")
            == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        )
