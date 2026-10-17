#!/usr/bin/env python3
"""Compose document pages around labelled tables for the layout -> table gate.

The table GT package (PubTabNet-style ``val.txt``, one cropped table image
per line with its ``gt`` HTML) proves table structure + cell text on
pre-cropped tables. The composed pipeline must also *find* the table on a
page. Each page produced here embeds one GT table at its native resolution
(no rescale, lossless PNG) between deterministic text blocks: a title, a
paragraph, a caption and a trailing paragraph, i.e. the CDLA classes the
layout model separates from ``table``. The GT HTML is unchanged; the pasted
pixel box is recorded so the evaluator can select the predicted table that
covers it.

Output: ``<out>/pages/<stem>.png`` and ``<out>/pages.txt`` with one JSON line
per page::

    {"filename": "pages/<stem>.png", "gt": <html>, "table_box": [x1,y1,x2,y2],
     "source": <val filename>, "source_sha256": <hex>}

Deterministic: the same inputs give byte-identical pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
PARAGRAPH = ("Results are reported on the evaluation split under the same "
             "protocol as the baseline. The table below lists every setting "
             "with its measured value; best results are shown in bold.")
TRAILING = ("All numbers were obtained with identical preprocessing. The "
            "differences between rows are discussed in the next section, "
            "together with the limitations of the measurement setup.")


def wrap(text: str, scale: float, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if cv2.getTextSize(candidate, FONT, scale, 1)[0][0] > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def put_lines(page: np.ndarray, lines: list[str], x: int, y: int, scale: float,
              thickness: int, step: int) -> int:
    for line in lines:
        cv2.putText(page, line, (x, y), FONT, scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)
        y += step
    return y


def compose(table: np.ndarray, index: int) -> tuple[np.ndarray, list[int]]:
    th, tw = table.shape[:2]
    margin = 60
    width = max(1000, tw + 2 * margin)
    text_width = width - 2 * margin
    body = wrap(PARAGRAPH, 0.6, text_width)
    tail = wrap(TRAILING, 0.6, text_width)
    height = 70 + 40 + 26 * len(body) + 30 + 30 + th + 40 + 26 * len(tail) + 60
    # CDLA pages are portrait; keep the page at least A4-ish tall so the
    # 608x800 layout resize sees document proportions.
    height = max(height, int(width * 1.3))
    page = np.full((height, width, 3), 255, np.uint8)
    y = put_lines(page, [f"{index + 1}. Experimental Evaluation"], margin, 70,
                  0.9, 2, 40)
    y = put_lines(page, body, margin, y, 0.6, 1, 26) + 10
    y = put_lines(page, [f"Table {index + 1}. Comparison of the evaluated settings."],
                  margin, y, 0.55, 1, 30)
    x0 = (width - tw) // 2
    y0 = y
    page[y0:y0 + th, x0:x0 + tw] = table
    put_lines(page, tail, margin, y0 + th + 50, 0.6, 1, 26)
    return page, [x0, y0, x0 + tw, y0 + th]


def compose_pages(ground_truth: Path, image_root: Path, output: Path,
                  limit: int = 0) -> list[dict]:
    """Write one page per GT table under ``output/pages`` and return the
    page records (also written to ``output/pages.txt``)."""
    (output / "pages").mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in
               ground_truth.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        records = records[:limit]
    pages = []
    for index, record in enumerate(records):
        source = image_root / record["filename"]
        table = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if table is None:
            raise SystemExit(f"cannot decode {source}")
        page, box = compose(table, index)
        name = f"pages/{Path(record['filename']).stem}.png"
        if not cv2.imwrite(str(output / name), page):
            raise SystemExit(f"cannot write {name}")
        pages.append({"filename": name, "gt": record["gt"], "table_box": box,
                      "source": record["filename"],
                      "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    (output / "pages.txt").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages),
        encoding="utf-8")
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N tables (0 = all)")
    args = parser.parse_args()
    pages = compose_pages(args.ground_truth, args.image_root, args.output, args.limit)
    print(f"{len(pages)} pages -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
