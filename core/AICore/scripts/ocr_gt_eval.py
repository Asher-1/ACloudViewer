#!/usr/bin/env python3
"""Evaluate OCR boxes against labelled ground truth.

The evaluator deliberately consumes ground truth, not another engine's output.
GT records use the PaddleOCR detection-label shape::

  {"image": "relative/or/absolute.png", "annotations": [
    {"points": [[x,y], ...], "transcription": "text", "ignore": false}
  ]}

One JSON object per line is preferred. A PaddleOCR ``image<TAB>json-list``
label file is accepted as input too. Predictions are the ``OCRBOX`` records
emitted by ``bench_ocr``::

  OCRBOX <index> <score> x0,y0 x1,y1 x2,y2 x3,y3<TAB>utf8 text

Detection matching is one-to-one greedy quadrilateral IoU. End-to-end CER/WER
counts unmatched GT as deletions and unmatched non-ignore predictions as
insertions, so a detector cannot hide recognition misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Annotation:
    points: tuple[tuple[float, float], ...]
    text: str
    ignore: bool = False


@dataclass(frozen=True)
class Prediction:
    points: tuple[tuple[float, float], ...]
    text: str
    score: float = 0.0


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    polygon = list(points)
    if len(polygon) < 3:
        return 0.0
    return float(abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] -
                         polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                         for i in range(len(polygon))) / 2.0))


def signed_area(points: list[tuple[float, float]]) -> float:
    return sum(points[i][0] * points[(i + 1) % len(points)][1] -
               points[(i + 1) % len(points)][0] * points[i][1]
               for i in range(len(points))) / 2.0


def inside(point: tuple[float, float], edge_a: tuple[float, float],
           edge_b: tuple[float, float], clockwise: bool) -> bool:
    cross = ((edge_b[0] - edge_a[0]) * (point[1] - edge_a[1]) -
             (edge_b[1] - edge_a[1]) * (point[0] - edge_a[0]))
    return cross <= 1e-7 if clockwise else cross >= -1e-7


def line_intersection(start: tuple[float, float], end: tuple[float, float],
                      edge_a: tuple[float, float], edge_b: tuple[float, float]
                      ) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    ex, ey = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) < 1e-12:
        return end
    t = ((edge_a[0] - start[0]) * ey - (edge_a[1] - start[1]) * ex) / denominator
    return start[0] + t * dx, start[1] + t * dy


def convex_intersection(subject: Iterable[tuple[float, float]],
                        clip: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clipping; GT boxes and OCR output are convex quads."""
    output = list(subject)
    clip_polygon = list(clip)
    if len(output) < 3 or len(clip_polygon) < 3:
        return []
    clockwise = signed_area(clip_polygon) < 0.0
    for edge_index, edge_a in enumerate(clip_polygon):
        edge_b = clip_polygon[(edge_index + 1) % len(clip_polygon)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        start = input_polygon[-1]
        start_inside = inside(start, edge_a, edge_b, clockwise)
        for end in input_polygon:
            end_inside = inside(end, edge_a, edge_b, clockwise)
            if end_inside != start_inside:
                output.append(line_intersection(start, end, edge_a, edge_b))
            if end_inside:
                output.append(end)
            start, start_inside = end, end_inside
    return output


def polygon_iou(a: Iterable[tuple[float, float]],
                b: Iterable[tuple[float, float]]) -> float:
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    intersection = polygon_area(convex_intersection(a, b))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def normalize_text(text: str, case_sensitive: bool) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    return text if case_sensitive else text.casefold()


def levenshtein(a: list[str], b: list[str]) -> int:
    """O(len(a)*len(b)) dynamic programming; OCR words are short."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def annotation_from_dict(raw: dict) -> Annotation:
    points = raw.get("points", raw.get("quad", []))
    if len(points) < 4:
        raise ValueError("annotation needs at least four points")
    text = str(raw.get("transcription", raw.get("text", "")))
    return Annotation(tuple((float(point[0]), float(point[1])) for point in points),
                      text, bool(raw.get("ignore", False) or text == "###"))


def load_ground_truth(path: Path) -> dict[str, list[Annotation]]:
    if path.is_dir():
        return load_sroie_ground_truth(path)
    records: list[dict] = []
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "documents" in parsed:
            # XFUND annotates semantic fields, sometimes containing multiple
            # lines. These regions are suitable for fixed-region recognition;
            # they are not text-line detection ground truth.
            records = []
            for document in parsed["documents"]:
                annotations = []
                for field in document["document"]:
                    x0, y0, x1, y1 = field["box"]
                    annotations.append({"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                        "transcription": field["text"]})
                records.append({"image": document["img"]["fname"], "annotations": annotations})
        else:
            records = parsed.get("images", parsed) if isinstance(parsed, dict) else parsed
    else:
        for line_number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            if "\t" in line and not line.lstrip().startswith("{"):
                image, annotations = line.split("\t", 1)
                records.append({"image": image, "annotations": json.loads(annotations)})
            else:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid GT record") from error
    result: dict[str, list[Annotation]] = {}
    for raw in records:
        image = str(raw.get("image", raw.get("image_path", "")))
        if not image:
            raise ValueError(f"{path}: GT record has no image")
        annotations = raw.get("annotations", raw.get("labels", []))
        if not isinstance(annotations, list):
            raise ValueError(f"{path}: annotations for {image} must be a list")
        result[image] = [annotation_from_dict(annotation) for annotation in annotations]
    return result


def ground_truth_digest(path: Path) -> str:
    """Pin one label file, or a sorted SROIE directory's names and raw bytes.

    Directory hashes include every CSV basename and its binary SHA-256, so a
    missing, additional, renamed or modified receipt label invalidates the pin.
    Images are pinned separately by the common validation input catalog.
    """
    if not path.is_dir():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = sorted(path.glob("*.csv"))
    if not files:
        raise ValueError("SROIE label directory is empty")
    digest = hashlib.sha256()
    for label in files:
        digest.update(label.name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(label.read_bytes()).digest())
    return digest.hexdigest()


def load_sroie_ground_truth(directory: Path) -> dict[str, list[Annotation]]:
    """Read original receipt localisation/transcription CSV, without reboxing.

    Only the first eight commas separate coordinates. Commas in the original
    unquoted transcript, including trailing commas, remain part of the text.
    A directory contains the predeclared subset supplied by the common runner.
    """
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError("SROIE label directory is empty")
    result = {}
    for label in files:
        annotations = []
        for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split(",", 8)
            try:
                if len(fields) != 9:
                    raise ValueError("expected eight coordinates and a transcript")
                values = [float(value) for value in fields[:8]]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite coordinates")
                points = tuple(zip(values[::2], values[1::2]))
                if polygon_area(points) <= 0:
                    raise ValueError("degenerate text box")
                annotations.append(Annotation(points, fields[8], fields[8] == "###"))
            except ValueError as error:
                raise ValueError(f"{label}:{number}: invalid SROIE annotation") from error
        if not annotations:
            raise ValueError(f"{label}: no SROIE text annotations")
        result[label.with_suffix(".jpg").name] = annotations
    return result


def parse_ocrbox(line: str, source: str) -> Prediction:
    try:
        header, text = line.rstrip("\n").split("\t", 1)
        fields = header.split()
        if len(fields) != 7 or fields[0] != "OCRBOX":
            raise ValueError
        points = tuple(tuple(float(component) for component in field.split(","))
                       for field in fields[3:])
        if any(len(point) != 2 for point in points):
            raise ValueError
        return Prediction(points, text, float(fields[2]))
    except (ValueError, IndexError) as error:
        raise ValueError(f"{source}: malformed OCRBOX record: {line!r}") from error


def load_predictions(path: Path) -> dict[str, list[Prediction]]:
    """Load one bench output or JSONL records with explicit ``image`` fields."""
    content = path.read_text(encoding="utf-8")
    result: dict[str, list[Prediction]] = {}
    current: list[Prediction] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if line.startswith("OCRBOX "):
            current.append(parse_ocrbox(line, f"{path}:{line_number}"))
            continue
        if not line.lstrip().startswith("{"):
            continue
        raw = json.loads(line)
        if "image" in raw and "predictions" in raw:
            result[str(raw["image"])] = [Prediction(
                tuple((float(point[0]), float(point[1])) for point in item["points"]),
                str(item.get("text", "")), float(item.get("score", 0.0)))
                for item in raw["predictions"]]
    if current:
        result.setdefault("__single__", current)
    return result


def match_predictions(predictions: list[Prediction], annotations: list[Annotation],
                       iou_threshold: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    valid_gt = [index for index, annotation in enumerate(annotations)
                if not annotation.ignore]
    ignored_gt = [annotation for annotation in annotations if annotation.ignore]
    eligible_predictions = {
        index for index, prediction in enumerate(predictions)
        if not any(polygon_iou(prediction.points, annotation.points) >= iou_threshold
                   for annotation in ignored_gt)
    }
    candidates = []
    for prediction_index in eligible_predictions:
        for gt_index in valid_gt:
            iou = polygon_iou(predictions[prediction_index].points,
                              annotations[gt_index].points)
            if iou >= iou_threshold:
                candidates.append((iou, prediction_index, gt_index))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    pairs = []
    for iou, prediction_index, gt_index in candidates:
        if prediction_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(prediction_index)
        used_gt.add(gt_index)
        pairs.append((prediction_index, gt_index, iou))
    return pairs, eligible_predictions - used_predictions, set(valid_gt) - used_gt


def evaluate(predictions: list[Prediction], annotations: list[Annotation],
             iou_threshold: float, case_sensitive: bool) -> dict:
    pairs, unmatched_predictions, unmatched_gt = match_predictions(
        predictions, annotations, iou_threshold)
    char_distance = word_distance = char_reference = word_reference = 0
    exact = 0
    errors = []
    for prediction_index, gt_index, iou in pairs:
        predicted = normalize_text(predictions[prediction_index].text, case_sensitive)
        expected = normalize_text(annotations[gt_index].text, case_sensitive)
        char_distance += levenshtein(list(predicted), list(expected))
        char_reference += len(expected)
        # A text line can contain several words. Count word edits, rather
        # than reporting the fraction of mismatching lines as WER.
        word_distance += levenshtein(predicted.split(), expected.split())
        word_reference += len(expected.split())
        exact += int(predicted == expected)
        if predicted != expected:
            errors.append({"kind": "text", "iou": round(iou, 6),
                           "prediction": predictions[prediction_index].text,
                           "ground_truth": annotations[gt_index].text})
    for gt_index in unmatched_gt:
        expected = normalize_text(annotations[gt_index].text, case_sensitive)
        char_distance += len(expected)
        char_reference += len(expected)
        word_distance += len(expected.split())
        word_reference += len(expected.split())
        errors.append({"kind": "missed_gt", "ground_truth": annotations[gt_index].text})
    for prediction_index in unmatched_predictions:
        predicted = normalize_text(predictions[prediction_index].text, case_sensitive)
        char_distance += len(predicted)
        word_distance += len(predicted.split())
        errors.append({"kind": "extra_prediction", "prediction": predictions[prediction_index].text})
    tp = len(pairs)
    fp = len(unmatched_predictions)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    hmean = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gt_boxes": tp + fn, "predicted_boxes": tp + fp, "matched_boxes": tp,
            "precision": precision, "recall": recall, "hmean": hmean,
            "exact_text_matches": exact, "word_accuracy": exact / (tp + fn) if tp + fn else 0.0,
            "cer": char_distance / char_reference if char_reference else 0.0,
            "wer": word_distance / word_reference if word_reference else 0.0,
            "char_distance": char_distance, "char_reference": char_reference,
            "word_distance": word_distance, "word_reference": word_reference,
            "errors": errors}


def aggregate(reports: Iterable[dict]) -> dict:
    """Micro-average counts; never average per-page rates of unequal size."""
    rows = list(reports)
    totals = {key: sum(row[key] for row in rows) for key in (
        "gt_boxes", "predicted_boxes", "matched_boxes", "exact_text_matches",
        "char_distance", "char_reference", "word_distance", "word_reference")}
    precision = totals["matched_boxes"] / totals["predicted_boxes"] if totals["predicted_boxes"] else 0.
    recall = totals["matched_boxes"] / totals["gt_boxes"] if totals["gt_boxes"] else 0.
    return {**totals, "images": len(rows), "precision": precision, "recall": recall,
            "hmean": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "cer": totals["char_distance"] / totals["char_reference"] if totals["char_reference"] else 0.,
            "wer": totals["word_distance"] / totals["word_reference"] if totals["word_reference"] else 0.,
            "word_accuracy": totals["exact_text_matches"] / totals["gt_boxes"] if totals["gt_boxes"] else 0.}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image", help="required when predictions is one bench output")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be in (0, 1]")
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    if "__single__" in predictions:
        if not args.image:
            parser.error("--image is required for a single bench output")
        predictions[args.image] = predictions.pop("__single__")
    aggregate = {"gt_boxes": 0, "predicted_boxes": 0, "matched_boxes": 0,
                 "exact_text_matches": 0, "char_distance": 0, "char_reference": 0,
                 "word_errors": 0, "word_reference": 0, "errors": []}
    per_image = {}
    for image, annotations in ground_truth.items():
        report = evaluate(predictions.get(image, []), annotations, args.iou,
                          args.case_sensitive)
        per_image[image] = report
        for key in ("gt_boxes", "predicted_boxes", "matched_boxes", "exact_text_matches",
                    "char_distance", "char_reference"):
            aggregate[key] += report[key]
        aggregate["word_errors"] += report["word_distance"]
        aggregate["word_reference"] += report["word_reference"]
        aggregate["errors"].extend({"image": image, **error} for error in report["errors"])
    tp = aggregate["matched_boxes"]
    precision = tp / aggregate["predicted_boxes"] if aggregate["predicted_boxes"] else 0.0
    recall = tp / aggregate["gt_boxes"] if aggregate["gt_boxes"] else 0.0
    aggregate.update({
        "precision": precision, "recall": recall,
        "hmean": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "word_accuracy": aggregate["exact_text_matches"] / aggregate["gt_boxes"] if aggregate["gt_boxes"] else 0.0,
        "cer": aggregate["char_distance"] / aggregate["char_reference"] if aggregate["char_reference"] else 0.0,
        "wer": aggregate["word_errors"] / aggregate["word_reference"] if aggregate["word_reference"] else 0.0,
        "iou_threshold": args.iou, "case_sensitive": args.case_sensitive,
        "per_image": per_image,
    })
    text = json.dumps(aggregate, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
