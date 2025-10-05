"""Tests for safe_io persistence fallbacks."""

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import safe_io
from safe_io import (
    clear_in_memory_store,
    get_in_memory_store_snapshot,
    safe_write_text,
    set_fallback_directory,
)


class SafeIOFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_fallback = safe_io.get_fallback_directory()
        set_fallback_directory(Path(self.temp_dir.name) / "fallback")
        clear_in_memory_store()
        self.addCleanup(clear_in_memory_store)
        self.addCleanup(lambda: set_fallback_directory(self.original_fallback))

    def test_safe_write_text_permission_fallback(self) -> None:
        target = Path(self.temp_dir.name) / "primary.txt"
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and str(path) == str(target) and "w" in mode:
                raise PermissionError("permission denied")
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", new=fake_open):
            result = safe_write_text(target, "contenido crítico", label="test_permission")

        self.assertEqual(result.status, "fallback")
        self.assertIsNotNone(result.path)
        self.assertTrue(result.path.exists())
        self.assertIn("fallback", str(result.path))
        self.assertEqual(result.path.read_text(encoding="utf-8"), "contenido crítico")

    def test_safe_write_text_disk_full_memory_fallback(self) -> None:
        target = Path(self.temp_dir.name) / "primary_disk_full.txt"
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if "w" in mode:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", new=fake_open):
            result = safe_write_text(target, "datos resilientes", label="test_disk_full")

        self.assertEqual(result.status, "memory")
        self.assertIsNone(result.path)
        self.assertIsNotNone(result.key)
        store = get_in_memory_store_snapshot()
        self.assertIn(result.key, store)
        self.assertEqual(store[result.key].decode("utf-8"), "datos resilientes")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
