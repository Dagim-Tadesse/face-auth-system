import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import RAW_DIR, ensure_project_directories, user_raw_dir
from src.feature_engineering import all_images
from src.preprocessing import assess_face_quality


class TestPipelineBasics(unittest.TestCase):
    def test_user_raw_dir_strips_whitespace(self):
        self.assertEqual(user_raw_dir(" alice "), RAW_DIR / "alice")

    def test_ensure_project_directories_creates_required_paths(self):
        ensure_project_directories()
        self.assertTrue((RAW_DIR).exists())

    def test_assess_face_quality_empty_frame(self):
        is_good, reasons, metrics = assess_face_quality(None)
        self.assertFalse(is_good)
        self.assertIn("Empty frame.", reasons)
        self.assertEqual(metrics, {})

    def test_all_images_returns_empty_when_folder_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "does_not_exist"
            X, y = all_images(missing_path)
            self.assertEqual(X.size, 0)
            self.assertEqual(y.size, 0)

    def test_all_images_skips_non_file_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            user_dir = base / "user1"
            nested_dir = user_dir / "nested"
            nested_dir.mkdir(parents=True, exist_ok=True)

            X, y = all_images(base)
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(X.size, 0)
            self.assertEqual(y.size, 0)


if __name__ == "__main__":
    unittest.main()
