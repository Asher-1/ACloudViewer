#!/usr/bin/env python3
import importlib.util
import json
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
    def test_receipt_labels_keep_full_lines_commas_and_ignore_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "001.csv").write_text(
                "\ufeff0,0,40,0,40,10,0,10,NO.53, JALAN,\n"
                "0,20,40,20,40,30,0,30,###\n\n", encoding="utf-8")
            labels = EVAL.load_ground_truth(root)
            self.assertEqual(labels["001.jpg"][0].text, "NO.53, JALAN,")
            self.assertEqual(labels["001.jpg"][0].points, ((0, 0), (40, 0), (40, 10), (0, 10)))
            self.assertTrue(labels["001.jpg"][1].ignore)

    def test_receipt_subset_digest_covers_names_contents_and_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "001.csv"
            first.write_text("0,0,2,0,2,2,0,2,A\n")
            pinned = EVAL.ground_truth_digest(root)
            first.rename(root / "002.csv")
            self.assertNotEqual(pinned, EVAL.ground_truth_digest(root))
            (root / "002.csv").rename(first)
            self.assertEqual(pinned, EVAL.ground_truth_digest(root))
            first.write_text("0,0,2,0,2,2,0,2,B\n")
            self.assertNotEqual(pinned, EVAL.ground_truth_digest(root))
            first.write_text("0,0,2,0,2,2,0,2,A\n")
            (root / "003.csv").write_bytes(first.read_bytes())
            self.assertNotEqual(pinned, EVAL.ground_truth_digest(root))
            first.unlink()
            (root / "003.csv").unlink()
            with self.assertRaises(ValueError):
                EVAL.ground_truth_digest(root)

    def test_receipt_labels_reject_malformed_and_degenerate_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for text in ("1,2,A", "nan,0,2,0,2,2,0,2,A", "0,0,0,0,0,0,0,0,A"):
                (root / "001.csv").write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    EVAL.load_ground_truth(root)

    def test_xfund_keeps_semantic_field_regions_and_transcriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ja.val.json"
            path.write_text(json.dumps({"documents": [{"img": {"fname": "ja_val_0.jpg"},
                "document": [{"box": [1, 2, 30, 40], "text": "一行目\n二行目", "label": "question",
                              "words": [{"box": [2, 3, 10, 11], "text": "一"}]}]}]}))
            records = EVAL.load_ground_truth(path)
            self.assertEqual(len(records["ja_val_0.jpg"]), 1)
            field = records["ja_val_0.jpg"][0]
            self.assertEqual(field.text, "一行目\n二行目")
            self.assertEqual(field.points, ((1, 2), (30, 2), (30, 40), (1, 40)))

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

    def test_wer_counts_word_edits_on_lines_including_misses_and_extras(self):
        box = ((0, 0), (10, 0), (10, 10), (0, 10))
        missed = ((20, 0), (30, 0), (30, 10), (20, 10))
        extra = ((40, 0), (50, 0), (50, 10), (40, 10))
        report = EVAL.evaluate([
            EVAL.Prediction(box, "one changed three four"),
            EVAL.Prediction(extra, "two extras")], [
                EVAL.Annotation(box, "one two three"),
                EVAL.Annotation(missed, "two missing")], .5, False)
        self.assertEqual(report["word_distance"], 6)
        self.assertEqual(report["word_reference"], 5)
        self.assertEqual(report["wer"], 6 / 5)
        self.assertEqual(report["word_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
