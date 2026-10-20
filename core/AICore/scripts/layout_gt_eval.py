#!/usr/bin/env python3
"""Class-aware CDLA page metrics, separate from OCR transcription metrics.

The caller supplies pinned external annotations/images and scored predictions.
No downloading or inference occurs here. Conversion preserves the raster-mask
bbox/area and grouped-instance semantics of CDLA's labelme2coco.py. COCOeval
owns AP matching; these metrics do not replace raw graph probability gates.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path

CDLA_LABELS = ("Text", "Title", "Figure", "Figure caption", "Table",
               "Table caption", "Header", "Footer", "Reference", "Equation")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verified_path(root, entry):
    name = entry["path"]
    if not name or Path(name).name != name or "\\" in name:
        raise ValueError("CDLA asset must be an immediate file")
    path = root / name
    if path.is_symlink() or digest(path) != entry["sha256"]:
        raise ValueError(f"CDLA SHA-256 mismatch: {name}")
    return path


def cdla_ground_truth(root, pins):
    """Process one page's masks at a time, retaining only COCO metadata.

    Pillow's polygon rasterizer matches labelme 4.5.6 shape_to_mask. Pixel
    extents, inclusive edge pixels and mask area intentionally differ from
    a floating-point polygon envelope. Grouped polygons form one instance.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as mask_api

    if pins["labels"] != list(CDLA_LABELS) or not pins["pages"]:
        raise ValueError("invalid CDLA label mapping or empty split")
    names = [page["image"]["path"] for page in pins["pages"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate CDLA page")
    result = dict(info={}, images=[], annotations=[], categories=[
        dict(id=i, name=name) for i, name in enumerate(CDLA_LABELS)])
    for image_id, page in enumerate(pins["pages"]):
        image = verified_path(root, page["image"])
        annotation = verified_path(root, page["annotation"])
        data = json.loads(annotation.read_text(encoding="utf-8"))
        with Image.open(image) as frame:
            width, height = frame.size
        if (data["imagePath"] != image.name or
                (data["imageWidth"], data["imageHeight"]) != (width, height)):
            raise ValueError("CDLA image identity/dimensions mismatch")
        result["images"].append(dict(id=image_id, file_name=image.name,
            width=width, height=height, sha256=page["image"]["sha256"]))
        masks = {}
        for index, shape in enumerate(data["shapes"]):
            points = shape["points"]
            label = shape["label"]
            if (label not in CDLA_LABELS or shape["shape_type"] != "polygon" or
                    len(points) < 3 or any(len(p) != 2 or
                    not all(math.isfinite(v) for v in p) for p in points)):
                raise ValueError("invalid CDLA polygon/class")
            # O(width*height) per polygon, as in the original converter.
            # Only this page's instance masks are live (not the whole split).
            canvas = Image.new("L", (width, height), 0)
            ImageDraw.Draw(canvas).polygon([tuple(p) for p in points],
                                           outline=1, fill=1)
            mask = np.asarray(canvas, dtype=np.uint8)
            group = shape.get("group_id")
            key = (label, ("shape", index) if group is None else ("group", group))
            masks[key] = mask if key not in masks else masks[key] | mask
        for (label, _), mask in masks.items():
            encoded = mask_api.encode(np.asfortranarray(mask))
            area = float(mask_api.area(encoded))
            if area <= 0:
                raise ValueError("empty CDLA instance mask")
            result["annotations"].append(dict(id=len(result["annotations"]) + 1,
                image_id=image_id, category_id=CDLA_LABELS.index(label),
                bbox=mask_api.toBbox(encoded).tolist(), area=area, iscrowd=0))
    return result


def scored_predictions(ground_truth, report):
    """Require explicit coverage, including pages with zero predictions."""
    images = {row["file_name"]: row for row in ground_truth["images"]}
    pages = report["images"]
    names = [page["image"] for page in pages]
    if len(names) != len(set(names)) or set(names) != set(images):
        raise ValueError("layout prediction page coverage mismatch")
    classes = {row["id"] for row in ground_truth["categories"]}
    predictions = []
    for page in pages:
        image = images[page["image"]]
        if (page["sha256"] != image["sha256"] or
                page["width"] != image["width"] or
                page["height"] != image["height"]):
            raise ValueError("layout prediction image identity mismatch")
        for region in page["regions"]:
            category = region["class_id"]
            box, score = region["bbox"], region["score"]
            if (type(category) is not int or category not in classes or
                    len(box) != 4 or not all(math.isfinite(x) for x in box) or
                    not math.isfinite(score) or not 0 <= score <= 1 or
                    not box[0] < box[2] or not box[1] < box[3]):
                raise ValueError("invalid layout prediction class/score/box")
            # COCO accepts finite boxes outside the frame. Preserve upstream
            # coordinates: its clip/divide can put an image edge a few ULPs
            # beyond the integer image size; clipping here would alter AP.
            x1, y1, x2, y2 = box
            predictions.append(dict(image_id=image["id"], category_id=category,
                bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
    return predictions


def evaluate(ground_truth, report):
    """Official COCO bbox AP at IoU .50:.05:.95, maxDets 1/10/100.

    Report input threshold/NMS separately: product score=.5 metrics must not
    be presented as the published benchmark with another score threshold.
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = scored_predictions(ground_truth, report)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        # COCOeval uses zero as its unmatched sentinel. CDLA's historical
        # converter starts annotation IDs at zero; normalize IDs without
        # changing any geometry, classes or the caller-owned dictionary.
        gt.dataset = dict(ground_truth, annotations=[dict(row, id=index + 1)
            for index, row in enumerate(ground_truth["annotations"])])
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(images=ground_truth["images"],
                categories=ground_truth["categories"], annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    per_class = {}
    categories = {row["id"]: row["name"] for row in ground_truth["categories"]}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        valid = values[values >= 0]
        per_class[categories[category_id]] = float(np.mean(valid)) if valid.size else None
    return dict(pages=len(ground_truth["images"]),
        annotations=len(ground_truth["annotations"]), predictions=len(predictions),
        map_50_95=float(evaluator.stats[0]), map_50=float(evaluator.stats[1]),
        map_75=float(evaluator.stats[2]), mar_100=float(evaluator.stats[8]),
        per_class_map_50_95=per_class,
        protocol=dict(metric="COCOeval bbox", iou=[.5, .95, .05],
                      max_detections=[1, 10, 100], class_aware=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-root", "pins", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ground-truth-output", type=Path)
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text())
    gt = cdla_ground_truth(args.image_root, pins)
    if args.ground_truth_output:
        args.ground_truth_output.write_text(json.dumps(gt) + "\n")
    metrics = evaluate(gt, json.loads(args.predictions.read_text()))
    metrics.update(pins_sha256=digest(args.pins),
        predictions_sha256=digest(args.predictions), evaluator_sha256=digest(__file__),
        scope="Page metrics only; raw probability, GPU speed and release gates are separate")
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
