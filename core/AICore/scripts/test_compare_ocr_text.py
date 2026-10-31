#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_ocr_text.py")
SPEC = importlib.util.spec_from_file_location("compare_ocr_text", MODULE_PATH)
assert SPEC and SPEC.loader
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


class CompareOcrTextTests(unittest.TestCase):
    def test_pairs_nearest_centers_once_and_counts_exact_text(self):
        ours = [
            COMPARE.TextRow(0, 10, 10, 20, 10, "hello"),
            COMPARE.TextRow(1, 20, 10, 20, 10, "world"),
            COMPARE.TextRow(2, 200, 10, 20, 10, "extra"),
        ]
        paddle = [
            COMPARE.TextRow(0, 11, 10, 20, 10, "hello"),
            COMPARE.TextRow(1, 19, 10, 20, 10, "other"),
            COMPARE.TextRow(2, 300, 10, 20, 10, "missed"),
        ]
        report = COMPARE.compare(ours, paddle, 5)
        self.assertEqual(report["matched_rows"], 2)
        self.assertEqual(report["identical_text_rows"], 1)
        self.assertEqual(report["aicore_unmatched_rows"], 1)
        self.assertEqual(report["paddle_unmatched_rows"], 1)
        self.assertEqual(report["identical_text_rate_on_matches"], 0.5)

    def test_reads_empty_text_and_rejects_invalid_protocol_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"
            path.write_text("ignore\nTEXT 2 1 2 3x4 \n", encoding="utf-8")
            self.assertEqual(COMPARE.read_text_rows(path)[0].text, "")
            path.write_text("TEXT invalid\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                COMPARE.read_text_rows(path)


if __name__ == "__main__":
    unittest.main()
