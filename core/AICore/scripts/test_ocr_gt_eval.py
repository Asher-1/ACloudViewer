#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("ocr_gt_eval.py")
SPEC = importlib.util.spec_from_file_location("ocr_gt_eval", MODULE_PATH)
assert SPEC and SPEC.loader
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)


class OcrGroundTruthEvalTests(unittest.TestCase):
    def test_detection_and_text_metrics_include_misses_and_extras(self):
        annotations = [
            EVAL.Annotation(((0, 0), (10, 0), (10, 10), (0, 10)), "Hello"),
            EVAL.Annotation(((20, 0), (30, 0), (30, 10), (20, 10)), "world"),
            EVAL.Annotation(((40, 0), (50, 0), (50, 10), (40, 10)), "###", True),
        ]
        predictions = [
            EVAL.Prediction(((0, 0), (10, 0), (10, 10), (0, 10)), "hello"),
            EVAL.Prediction(((20, 0), (30, 0), (30, 10), (20, 10)), "word"),
            EVAL.Prediction(((60, 0), (70, 0), (70, 10), (60, 10)), "extra"),
            EVAL.Prediction(((40, 0), (50, 0), (50, 10), (40, 10)), "ignored"),
        ]
        report = EVAL.evaluate(predictions, annotations, 0.5, case_sensitive=False)
        self.assertEqual(report["gt_boxes"], 2)
        self.assertEqual(report["predicted_boxes"], 3)
        self.assertEqual(report["matched_boxes"], 2)
        self.assertAlmostEqual(report["precision"], 2 / 3)
        self.assertEqual(report["exact_text_matches"], 1)
        self.assertEqual(report["char_distance"], 6)
        self.assertEqual(report["char_reference"], 10)
        self.assertEqual(report["word_distance"], 2)
        self.assertEqual(report["word_reference"], 2)

    def test_paddle_label_format_and_ocrbox_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt.txt"
            pred = root / "pred.txt"
            gt.write_text('a.png\t[{"transcription":"A", "points":[[0,0],[2,0],[2,2],[0,2]]}]\n',
                          encoding="utf-8")
            pred.write_text("OCRBOX 0 0.9 0,0 2,0 2,2 0,2\tA\n", encoding="utf-8")
            records = EVAL.load_ground_truth(gt)
            predictions = EVAL.load_predictions(pred)
            self.assertEqual(records["a.png"][0].text, "A")
            self.assertEqual(predictions["__single__"][0].text, "A")


if __name__ == "__main__":
    unittest.main()
