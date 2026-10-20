#!/usr/bin/env python3
"""Acceptance-boundary tests for class-aware layout page evaluation."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

# These numerical tests belong to the optional OCR reference environment.
# Do not add its Python packages to application packaging dependencies.
# CTest records missing packages as SKIP (77), never a passing accuracy gate.
if __name__ == "__main__":
    missing = [name for name in ("numpy", "PIL", "cv2", "pycocotools")
               if importlib.util.find_spec(name) is None]
    if missing:
        print("SKIP: layout evaluator tests require " + ", ".join(missing))
        sys.exit(77)

import numpy as np
from PIL import Image

import layout_gt_eval as layout
from export_cdla_layout_reference import scored_regions


def fixture():
    gt = dict(info={}, images=[dict(id=0, file_name="page.jpg", width=100,
        height=100, sha256="page-sha")],
        categories=[dict(id=0, name="Text"), dict(id=1, name="Title")],
        annotations=[dict(id=0, image_id=0, category_id=0,
            bbox=[10, 20, 30, 40], area=1200, iscrowd=0)])
    report = dict(images=[dict(image="page.jpg", width=100, height=100,
        sha256="page-sha", regions=[dict(class_id=0, score=.9,
                                      bbox=[10, 20, 40, 60])])])
    return gt, report


class LayoutMetricsTests(unittest.TestCase):
    def test_perfect_prediction_and_wrong_class_have_different_ap(self):
        gt, report = fixture()
        self.assertAlmostEqual(layout.evaluate(gt, report)["map_50_95"], 1)
        report["images"][0]["regions"][0]["class_id"] = 1
        self.assertEqual(layout.evaluate(gt, report)["map_50_95"], 0)

    def test_per_class_metrics_follow_coco_category_ids(self):
        gt, report = fixture()
        gt["categories"].reverse()
        result = layout.evaluate(gt, report)["per_class_map_50_95"]
        self.assertAlmostEqual(result["Text"], 1)
        self.assertIsNone(result["Title"])

    def test_high_confidence_false_positive_reduces_ap(self):
        gt, report = fixture()
        report["images"][0]["regions"].insert(0,
            dict(class_id=0, score=.99, bbox=[65, 65, 85, 85]))
        self.assertAlmostEqual(layout.evaluate(gt, report)["map_50"], .5)

    def test_iou_thresholds_are_not_a_single_threshold_f1(self):
        gt, report = fixture()
        report["images"][0]["regions"][0]["bbox"] = [15, 20, 45, 60]
        metrics = layout.evaluate(gt, report)
        self.assertAlmostEqual(metrics["map_50"], 1)
        self.assertEqual(metrics["map_75"], 0)
        self.assertGreater(metrics["map_50_95"], 0)
        self.assertLess(metrics["map_50_95"], 1)

    def test_explicit_empty_page_is_valid_and_scores_zero(self):
        gt, report = fixture()
        report["images"][0]["regions"] = []
        self.assertEqual(layout.evaluate(gt, report)["map_50_95"], 0)

    def test_missing_duplicate_extra_pages_fail(self):
        gt, report = fixture()
        page = report["images"][0]
        for pages in ([], [page, page], [dict(page, image="other.jpg")]):
            with self.subTest(pages=pages), self.assertRaisesRegex(ValueError, "coverage"):
                layout.evaluate(gt, dict(images=pages))

    def test_image_identity_and_dimensions_must_match(self):
        gt, report = fixture()
        for update in (dict(sha256="other"), dict(width=101), dict(height=101)):
            bad = copy.deepcopy(report)
            bad["images"][0].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "identity"):
                layout.evaluate(gt, bad)

    def test_invalid_class_probability_or_box_cannot_enter_metrics(self):
        gt, report = fixture()
        for update in (dict(class_id=3), dict(class_id=True), dict(score=float("nan")),
                       dict(score=1.01), dict(bbox=[10, 0, 5, 10]),
                       dict(bbox=[0, 0, 0, 10]), dict(bbox=[0, 0, float("inf"), 10])):
            bad = copy.deepcopy(report)
            bad["images"][0]["regions"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                layout.evaluate(gt, bad)

    def test_coco_keeps_valid_boxes_outside_the_image(self):
        gt, report = fixture()
        report["images"][0]["regions"][0]["bbox"] = [-1, 20, 100.00002, 60]
        converted = layout.scored_predictions(gt, report)
        self.assertEqual(converted[0]["bbox"], [-1, 20, 101.00002, 40])


class CdlaConversionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        Image.new("RGB", (20, 20)).save(self.root / "page.jpg")
        self.annotation = dict(imagePath="page.jpg", imageWidth=20,
            imageHeight=20, shapes=[dict(label="Text", group_id=None,
                shape_type="polygon", points=[[1, 2], [5, 2], [5, 5], [1, 5]])])

    def pins(self):
        (self.root / "page.json").write_text(json.dumps(self.annotation))
        return dict(labels=list(layout.CDLA_LABELS), pages=[dict(
            image=dict(path="page.jpg", sha256=layout.digest(self.root / "page.jpg")),
            annotation=dict(path="page.json", sha256=layout.digest(self.root / "page.json")))])

    def test_rasterized_bounds_include_edge_pixels(self):
        gt = layout.cdla_ground_truth(self.root, self.pins())
        self.assertEqual(gt["annotations"][0]["bbox"], [1, 2, 5, 4])
        self.assertEqual(gt["annotations"][0]["area"], 20)
        self.assertEqual(gt["annotations"][0]["category_id"], 0)

    def test_grouped_polygons_use_union_area_not_envelope_area(self):
        first = self.annotation["shapes"][0]
        first["group_id"] = 5
        second = copy.deepcopy(first)
        second["points"] = [[10, 2], [14, 2], [14, 5], [10, 5]]
        self.annotation["shapes"].append(second)
        gt = layout.cdla_ground_truth(self.root, self.pins())
        self.assertEqual(len(gt["annotations"]), 1)
        self.assertEqual(gt["annotations"][0]["area"], 40)
        self.assertEqual(gt["annotations"][0]["bbox"], [1, 2, 14, 4])

    def test_corrupt_annotation_or_image_fails_pins(self):
        for name in ("page.json", "page.jpg"):
            pins = self.pins()
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "SHA-256"):
                layout.cdla_ground_truth(self.root, pins)
            path.write_bytes(original)

    def test_asset_paths_are_confined_to_the_selected_split(self):
        for name in ("../page.jpg", "/tmp/page.jpg", "nested/page.jpg", "..\\page.jpg"):
            pins = self.pins()
            pins["pages"][0]["image"]["path"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                layout.cdla_ground_truth(self.root, pins)

    def test_wrong_label_map_duplicate_pages_or_dimensions_fail(self):
        for mode in ("classes", "pages", "dimensions"):
            pins = self.pins()
            if mode == "classes":
                pins["labels"] = list(reversed(pins["labels"]))
            elif mode == "pages":
                pins["pages"].append(pins["pages"][0])
            else:
                self.annotation["imageWidth"] = 21
                pins = self.pins()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                layout.cdla_ground_truth(self.root, pins)


class OriginalScoreObservationTests(unittest.TestCase):
    def test_observes_original_scores_without_mutating_nms_or_regions(self):
        original = lambda value: value
        module = SimpleNamespace(hard_nms=original)
        selected = np.array([[1, 2, 3, 4, .87], [5, 6, 7, 8, .64]])
        expected = [dict(label="text", bbox=[2, 4, 6, 8]),
                    dict(label="table_caption", bbox=[10, 12, 14, 16])]

        def processor(*args):
            self.assertIs(module.hard_nms(selected), selected)
            return expected

        actual = scored_regions(module, processor, None, None, {})
        self.assertEqual([row["class_id"] for row in actual], [0, 5])
        self.assertEqual([row["score"] for row in actual], [.87, .64])
        self.assertEqual([row["bbox"] for row in actual], [row["bbox"] for row in expected])
        self.assertIs(module.hard_nms, original)

    def test_exception_restores_original_function(self):
        original = lambda value: value
        module = SimpleNamespace(hard_nms=original)

        def processor(*args):
            raise RuntimeError("original source failed")

        with self.assertRaisesRegex(RuntimeError, "original source failed"):
            scored_regions(module, processor, None, None, {})
        self.assertIs(module.hard_nms, original)

    def test_incomplete_score_observation_is_rejected(self):
        module = SimpleNamespace(hard_nms=lambda value: value)
        with self.assertRaisesRegex(ValueError, "coverage"):
            scored_regions(module, lambda *a: [dict(label="text", bbox=[1, 2, 3, 4])],
                           None, None, {})


if __name__ == "__main__":
    unittest.main()
