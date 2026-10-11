#!/usr/bin/env python3
"""Region-level gate for the OCR layout stage (PicoDet CDLA).

The graph probe (``test_ocr_runtime ... layout``) proves the eight heads; this
gate proves what the caller receives: the decoded, NMS-filtered, image-mapped
regions of ``aicore_ocr_run`` against the official PP-Structure output of the
same upstream model on real document pages
(``run_paddle_layout_reference.py``, pinned in ``layout_regions.json``).

Each reference region is matched to at most one AICore region of the same
class, greedily by descending IoU (IoU >= ``--match-iou``). Metrics (pooled
over every page):

    region_precision  matched / AICore regions
    region_recall     matched / reference regions
    region_f1         harmonic mean of the two
    mean_iou          mean IoU of the matched pairs
    min_iou           worst matched pair

One JSON line (``validate_all.py`` generic parser) is printed; the exit code
is 1 when ``region_f1 < --min-f1``, ``region_recall < --min-recall`` or
``mean_iou < --min-mean-iou``, 77 when the reference, images or models are
missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REGION = re.compile(r"^REGION \d+ cls=(\d+) order=-?\d+ score=([0-9.]+) "
                    r"(-?[0-9.]+),(-?[0-9.]+) (-?[0-9.]+),(-?[0-9.]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else 0.0


def match(reference: list[dict], predicted: list[dict],
          threshold: float) -> list[float]:
    """IoUs of the greedy same-class one-to-one matching (O(R*P log RP))."""
    pairs = sorted(((iou(r["bbox"], p["bbox"]), i, j)
                    for i, r in enumerate(reference)
                    for j, p in enumerate(predicted)
                    if r["class_id"] == p["class_id"]), reverse=True)
    used_r, used_p, ious = set(), set(), []
    for value, i, j in pairs:
        if value < threshold:
            break
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        ious.append(value)
    return ious


def run_regions(args, image: Path, environment: dict[str, str]) -> list[dict]:
    command = [str(args.bench), str(image), str(args.det), str(args.rec),
               args.device, str(args.threads), "0", "1", str(args.layout)]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(f"bench_ocr failed ({completed.returncode}) on {image}:\n"
                           f"{completed.stderr[-2000:]}")
    regions = []
    for line in completed.stdout.splitlines():
        hit = REGION.match(line.strip())
        if hit:
            cls, score, x1, y1, x2, y2 = hit.groups()
            regions.append({"class_id": int(cls), "score": float(score),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]})
    return regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True,
                        help="layout_regions.json (run_paddle_layout_reference.py)")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True, help="bench_ocr binary")
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--min-f1", type=float, default=1.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-mean-iou", type=float, default=0.95)
    args = parser.parse_args()

    needed = [args.reference, args.bench, args.det, args.rec, args.layout]
    if any(not path.is_file() for path in needed):
        missing = [str(p) for p in needed if not p.is_file()]
        print(f"check_layout_regions: missing {', '.join(missing)}, skipping",
              file=sys.stderr)
        return 77
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    environment = os.environ.copy()
    build_bin = args.bench.resolve().parent
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(build_bin)] + ([environment["LD_LIBRARY_PATH"]]
                            if environment.get("LD_LIBRARY_PATH") else []))

    total_ref = total_pred = 0
    ious: list[float] = []
    pages = []
    for page in reference["images"]:
        image = args.image_root / page["image"]
        if not image.is_file():
            print(f"check_layout_regions: missing image {image}, skipping",
                  file=sys.stderr)
            return 77
        if sha256_file(image) != page["sha256"]:
            raise SystemExit(f"{image}: SHA-256 differs from the reference")
        predicted = run_regions(args, image, environment)
        page_ious = match(page["regions"], predicted, args.match_iou)
        total_ref += len(page["regions"])
        total_pred += len(predicted)
        ious.extend(page_ious)
        pages.append({"image": page["image"], "reference": len(page["regions"]),
                      "predicted": len(predicted), "matched": len(page_ious)})
    matched = len(ious)
    precision = matched / total_pred if total_pred else 0.0
    recall = matched / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0
    row = {"task": "ocr", "stage": "layout_regions", "device": args.device,
           "layout_model": args.layout.name, "pages": len(pages),
           "reference_regions": total_ref, "predicted_regions": total_pred,
           "matched_regions": matched, "region_precision": precision,
           "region_recall": recall, "region_f1": f1,
           "mean_iou": sum(ious) / matched if matched else 0.0,
           "min_iou": min(ious) if ious else 0.0}
    for page in pages:
        print(json.dumps(page), file=sys.stderr)
    print(json.dumps(row))
    return 0 if (f1 >= args.min_f1 and recall >= args.min_recall
                 and row["mean_iou"] >= args.min_mean_iou) else 1


if __name__ == "__main__":
    raise SystemExit(main())
