#!/usr/bin/env python3
"""Human-checkable view of the SLANet table parity fixtures.

The fixtures written by ``export_table_fixture.py`` are raw tensors
(``table_<k>_probs.npy`` per-step structure softmax, ``table_<k>_loc.npy``
per-step sigmoid cell polygons). This tool turns them back into what they
claim to be, using only Paddle-side artifacts:

* the vocabulary of the SLANet archive itself (``inference.yml``, the same
  TableLabelDecode contract the converter embeds), so the decode does not
  depend on the AICore port;
* the argmax token sequence -> structure HTML, compared with the labelled
  GT structure of the same image (``val.txt``) by exact match and TEDS-S;
* the cell polygons of every ``<td`` step mapped back to source pixels
  (TableLabelDecode._get_bbox_scales: x and y scale by pad / ratio with
  ratio = min(pad / w, pad / h)) and drawn on the original image.

Optionally (``--bench`` + ``--table``) the same images go through AICore
``bench_table`` structure-only so the report shows both sides.

Output: ``<out>/table_<k>_paddle_cells.png`` overlays, ``<out>/report.html``
(image, Paddle grid, GT grid, AICore grid) and one JSON line per fixture.

    inspect_table_fixture.py --fixtures ~/cloudViewer_data/extract/ocr_fixtures \
        --images ~/cloudViewer_data/extract/ppocr_testdata/table_rec/images \
        --model-dir SLANet_plus_infer --gt .../table_rec/val.txt --out /tmp/fx
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_paddle_ocr_gguf import table_metadata  # noqa: E402
import table_teds as TEDS  # noqa: E402


def wrap(structure: str) -> str:
    return f"<html><body><table>{structure}</table></body></html>"


def decode(probs: np.ndarray, loc: np.ndarray, tokens: list[str],
           size: tuple[int, int], pad: int):
    """Argmax tokens up to eos -> (structure HTML body, cell polygons)."""
    ids = probs.argmax(axis=1)
    w, h = size
    scale = pad / min(pad / w, pad / h)
    body, cells = [], []
    for step, i in enumerate(ids):
        token = tokens[int(i)]
        if token == "eos":
            break
        if token == "sos":
            continue
        body.append(token)
        if token in ("<td></td>", "<td"):
            poly = loc[step] * scale
            cells.append(poly.reshape(4, 2))
    return "".join(body), cells


def draw(image_path: Path, cells, out_path: Path) -> None:
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    for n, poly in enumerate(cells):
        pts = poly.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (255, 80, 0), 1, cv2.LINE_AA)
        x, y = pts[0, 0]
        cv2.putText(img, str(n), (int(x) + 2, int(y) + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def aicore_structures(bench: Path, table: Path, images: list[Path],
                      device: str) -> dict[str, str]:
    env = os.environ.copy()
    lib = [str(bench.resolve().parent), str(bench.resolve().parent.parent / "external" / "lib")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib + [env.get("LD_LIBRARY_PATH", "")])
    proc = subprocess.run([str(bench), "--table", str(table), "--device", device,
                           *map(str, images)], capture_output=True, text=True,
                          env=env, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        record = json.loads(line)
        if record.get("summary"):
            continue
        tables = record.get("tables") or []
        out[Path(record["image"]).name] = tables[0]["html"] if tables else ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="directory holding table_fixtures.json + .npy")
    parser.add_argument("--images", type=Path, required=True,
                        help="directory holding the fixture source images")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="SLANet_plus inference dir (inference.yml)")
    parser.add_argument("--gt", type=Path, help="labelled val.txt (JSON lines)")
    parser.add_argument("--bench", type=Path, help="AICore bench_table binary")
    parser.add_argument("--table", type=Path, help="SLANet GGUF for --bench")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = table_metadata(args.model_dir)
    tokens, pad = meta["ocr.table.tokens"], meta["ocr.table.pad_size"]
    manifest = json.loads((args.fixtures / "table_fixtures.json").read_text())
    gt = {}
    if args.gt:
        for line in args.gt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                gt[Path(record["filename"]).name] = record["gt"]
    images = [args.images / row["image"] for row in manifest]
    aicore = (aicore_structures(args.bench, args.table, images, args.device)
              if args.bench and args.table else {})

    args.out.mkdir(parents=True, exist_ok=True)
    sections = []
    for row, image in zip(manifest, images):
        k = row["index"]
        probs = np.load(args.fixtures / row["probs"]["file"])
        loc = np.load(args.fixtures / row["loc"]["file"])
        structure, cells = decode(probs, loc, tokens, tuple(row["size"]), pad)
        overlay = args.out / f"table_{k}_paddle_cells.png"
        draw(image, cells, overlay)
        result = {"index": k, "image": row["image"], "steps": row["steps"],
                  "paddle_cells": len(cells)}
        paddle_html = wrap(structure)
        if row["image"] in gt:
            # TEDS-S == 1 is an identical structure tree (tags + spans).
            teds_s = TEDS.teds(paddle_html, gt[row["image"]], structure_only=True)
            result["paddle_vs_gt_teds_s"] = round(teds_s, 6)
            result["paddle_vs_gt_exact"] = teds_s == 1.0
        if row["image"] in aicore:
            result["aicore_vs_paddle_teds_s"] = round(
                TEDS.teds(aicore[row["image"]], paddle_html, structure_only=True), 6)
        print(json.dumps(result, ensure_ascii=False))

        def grid(title, doc):
            body = doc.replace("<html><body>", "").replace("</body></html>", "")
            return (f"<div class=g><h4>{title}</h4>"
                    f"{body.replace('<table>', '<table border=1>')}</div>")
        parts = [f"<h3>table_{k}: {htmlmod.escape(row['image'])} "
                 f"({row['size'][0]}x{row['size'][1]}, {row['steps']} steps, "
                 f"{len(cells)} cells)</h3>",
                 f"<pre>{htmlmod.escape(json.dumps(result, ensure_ascii=False))}</pre>",
                 f"<img src='{image.resolve().as_uri()}'> "
                 f"<img src='{overlay.name}'>",
                 grid("Paddle fixture (decoded probs)", paddle_html)]
        if row["image"] in gt:
            parts.append(grid("Labelled GT", gt[row["image"]]))
        if row["image"] in aicore:
            parts.append(grid("AICore bench_table", aicore[row["image"]]))
        sections.append("<section>" + "".join(parts) + "</section>")
    (args.out / "report.html").write_text(
        "<html><head><meta charset=utf-8><style>td{min-width:14px;height:12px}"
        ".g{display:inline-block;vertical-align:top;margin:6px}"
        "img{max-width:640px;border:1px solid #ccc}</style></head><body>"
        + "<hr>".join(sections) + "</body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
