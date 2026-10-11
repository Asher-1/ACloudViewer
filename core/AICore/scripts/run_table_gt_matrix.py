#!/usr/bin/env python3
"""Run the AICore `table` task against labelled table ground truth.

Each GT image is one table (PubTabNet-style ``val.txt`` lines with a ``gt``
HTML field, e.g. the PaddleX ``table_rec_dataset_examples`` package). Every
selected backend x SLANet quantization row runs ``bench_table`` (OCR det/rec
+ table structure + cell assignment) over the whole list and is scored with
``table_teds.py`` (PubTabNet TEDS, identical to the upstream APTED
implementation):

    teds              mean TEDS (structure + cell text) against the GT
    teds_s            mean TEDS-S (structure only)
    structure_exact   fraction of tables whose structure tree is identical
                      to the GT (TEDS-S == 1)

Composed mode (``--layout-quant``): the GT lines are full pages from
``make_table_layout_pages.py`` carrying the pasted ``table_box``. The page
runs layout -> table regions -> structure -> det/rec cell text; the scored
table is the predicted one with the highest IoU against ``table_box`` (none
with IoU >= 0.5 scores 0 and counts as a missed detection). The extra metric

    table_recall      fraction of pages whose GT table was found

may not fall below the official composed chain on the same pages
(``run_paddle_table_baseline.py --layout-reference``: the upstream layout
labels some synthetic-border tables "figure", so its recall is < 1), and
the TEDS gate compares against that composed Paddle row.

The official reference is the PaddleX pipeline with the same upstream
models (``run_paddle_table_baseline.py``): either its predictions are
scored here (``--paddle-predictions``), or its pinned means are supplied
(``--paddle-teds``/``--paddle-teds-s``). A row passes when

    teds   >= paddle_teds   - teds_tolerance
    teds_s >= paddle_teds_s - teds_s_tolerance

One JSON summary line per row is printed (``validate_all.py`` generic
parser); the full report with per-image scores goes to ``--output``.
The exit code is non-zero when any row fails its gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("table_teds", SCRIPT_DIR / "table_teds.py")
assert SPEC and SPEC.loader
TEDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TEDS)

QUANTS = ("f32", "f16", "q8_0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth(path: Path) -> tuple[dict[str, str], dict[str, list[float]]]:
    """filename (relative to the image root) -> GT HTML (file order), plus
    the pasted table box of composed pages (empty for cropped tables)."""
    rows: dict[str, str] = {}
    boxes: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "gt" not in record:
            raise ValueError(f"{path}: line without a 'gt' HTML field")
        rows[record["filename"]] = record["gt"]
        if "table_box" in record:
            boxes[record["filename"]] = record["table_box"]
    return rows, boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def runtime_environment(bench: Path, extra_dirs: list[Path]) -> dict[str, str]:
    environment = os.environ.copy()
    build_bin = bench.resolve().parent
    directories = [build_bin, build_bin.parent / "external" / "lib", *extra_dirs]
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(d) for d in directories if d.is_dir()] + ([existing] if existing else []))
    return environment


def score(predictions: dict[str, str], truth: dict[str, str]) -> dict:
    per_image = {}
    for name, gt in truth.items():
        html = predictions.get(name, "")
        per_image[name] = {"teds": TEDS.teds(html, gt),
                           "teds_s": TEDS.teds(html, gt, structure_only=True)}
    teds = [row["teds"] for row in per_image.values()]
    teds_s = [row["teds_s"] for row in per_image.values()]
    return {"images": len(per_image),
            "teds": statistics.fmean(teds),
            "teds_s": statistics.fmean(teds_s),
            "structure_exact": sum(v == 1.0 for v in teds_s) / len(teds_s),
            "per_image": per_image}


def run_bench(args, backend: str, quant: str, truth: dict[str, str],
              boxes: dict[str, list[float]],
              environment: dict[str, str]) -> tuple[dict[str, str], dict]:
    table_model = args.models_dir / f"slanet-plus-table-{quant}.gguf"
    det_model = args.models_dir / f"ppocr-det-ch-{args.ocr_quant}.gguf"
    rec_model = args.models_dir / f"ppocr-rec-ch-{args.ocr_quant}.gguf"
    models = [table_model, det_model, rec_model]
    if args.layout_quant:
        models.append(args.models_dir / f"pplcnet-layout-{args.layout_quant}.gguf")
    for model in models:
        if not model.is_file():
            raise FileNotFoundError(model)
    list_path = args.output / f"images_{backend}_{quant}.txt"
    list_path.write_text("".join(str(args.image_root / name) + "\n" for name in truth),
                         encoding="utf-8")
    command = [str(args.bench), "--table", str(table_model), "--det", str(det_model),
               "--rec", str(rec_model), "--device", backend,
               "--threads", str(args.threads), "--warmups", str(args.warmups),
               # Official table pipeline text detection: max 960 long side.
               "--det-limit-type", "1", "--det-max-side", "960",
               "--list", str(list_path)]
    if args.layout_quant:
        command += ["--layout", str(models[3]), "--layout-regions"]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment, check=False)
    raw = args.output / f"aicore_{backend}_{quant}.jsonl"
    raw.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"bench_table failed ({completed.returncode}): "
                           f"{' '.join(command)}\n{completed.stderr[-4000:]}")
    root = str(args.image_root) + os.sep
    predictions: dict[str, str] = {}
    summary: dict = {}
    found = 0
    for line in completed.stdout.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("summary"):
            summary = record
            continue
        name = record["image"][len(root):] if record["image"].startswith(root) \
            else record["image"]
        tables = record["tables"]
        if name in boxes:
            # Composed page: the predicted table covering the pasted GT box.
            best = max(tables, key=lambda t: box_iou(t["box"], boxes[name]),
                       default=None)
            hit = best is not None and box_iou(best["box"], boxes[name]) >= 0.5
            found += hit
            predictions[name] = best["html"] if hit else ""
        else:
            predictions[name] = tables[0]["html"] if tables else ""
    if len(predictions) != len(truth):
        raise RuntimeError(f"bench_table returned {len(predictions)} of {len(truth)} images")
    if boxes:
        summary["table_recall"] = found / len(truth)
    return predictions, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, required=True,
                        help="PubTabNet-style val.txt with a 'gt' HTML field")
    parser.add_argument("--ground-truth-sha256", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="ocr_models cache (det/rec + slanet-plus-table GGUFs)")
    parser.add_argument("--runtime-lib-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=("cuda", "vulkan"),
                        choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--quants", nargs="+", default=QUANTS, choices=QUANTS)
    parser.add_argument("--ocr-quant", default="f32", choices=QUANTS)
    parser.add_argument("--compose-pages", action="store_true",
                        help="compose one document page per cropped GT table "
                             "(make_table_layout_pages.py) under --output")
    parser.add_argument("--pages-sha256", default="",
                        help="pin the composed pages.txt")
    parser.add_argument("--layout-quant", choices=QUANTS,
                        help="composed pages: run pplcnet-layout-<quant> and "
                             "score the table found over each table_box")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--paddle-predictions", type=Path,
                        help="run_paddle_table_baseline.py output to score")
    parser.add_argument("--paddle-teds", type=float)
    parser.add_argument("--paddle-teds-s", type=float)
    parser.add_argument("--paddle-table-recall", type=float,
                        help="composed pages: pinned Paddle table_recall")
    parser.add_argument("--teds-tolerance", type=float, default=0.005)
    parser.add_argument("--teds-s-tolerance", type=float, default=0.002)
    args = parser.parse_args()

    digest = sha256_file(args.ground_truth)
    if digest.lower() != args.ground_truth_sha256.lower():
        parser.error(f"ground-truth SHA-256 mismatch: expected "
                     f"{args.ground_truth_sha256}, got {digest}")
    args.output.mkdir(parents=True, exist_ok=True)
    ground_truth = args.ground_truth
    pages_digest = ""
    if args.compose_pages:
        # Deterministic page composition from the pinned cropped GT; the
        # composed pages.txt (GT HTML + pasted boxes) is pinned separately.
        if not args.layout_quant:
            parser.error("--compose-pages needs --layout-quant")
        spec = importlib.util.spec_from_file_location(
            "make_table_layout_pages", SCRIPT_DIR / "make_table_layout_pages.py")
        assert spec and spec.loader
        composer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(composer)
        composer.compose_pages(args.ground_truth, args.image_root, args.output)
        ground_truth = args.output / "pages.txt"
        args.image_root = args.output
        pages_digest = sha256_file(ground_truth)
        if args.pages_sha256 and pages_digest != args.pages_sha256.lower():
            parser.error(f"composed pages SHA-256 mismatch: expected "
                         f"{args.pages_sha256}, got {pages_digest}")
    truth, boxes = load_ground_truth(ground_truth)
    if not truth:
        parser.error("ground truth contains no tables")
    if bool(args.layout_quant) != bool(boxes):
        parser.error("--layout-quant needs composed pages with table_box "
                     "(--compose-pages or make_table_layout_pages.py) and vice versa")

    paddle: dict = {}
    if args.paddle_predictions:
        predictions = {}
        found = 0
        for line in args.paddle_predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                name = record["image"]
                if name in boxes:
                    # Composed pages: the same table selection as AICore.
                    best = max(record.get("tables", []),
                               key=lambda t: box_iou(t["box"], boxes[name]),
                               default=None)
                    hit = best is not None and box_iou(best["box"], boxes[name]) >= 0.5
                    found += hit
                    predictions[name] = best["html"] if hit else ""
                else:
                    predictions[name] = record["html"]
        paddle = score(predictions, truth)
        if boxes:
            paddle["table_recall"] = found / len(truth)
        paddle["source"] = str(args.paddle_predictions)
        paddle["sha256"] = sha256_file(args.paddle_predictions)
    elif args.paddle_teds is not None and args.paddle_teds_s is not None:
        paddle = {"teds": args.paddle_teds, "teds_s": args.paddle_teds_s,
                  "source": "pinned"}
        if args.paddle_table_recall is not None:
            paddle["table_recall"] = args.paddle_table_recall
    else:
        parser.error("supply --paddle-predictions or --paddle-teds/--paddle-teds-s")

    environment = runtime_environment(args.bench, args.runtime_lib_dir)
    report = {"ground_truth": str(args.ground_truth), "ground_truth_sha256": digest,
              "dataset_revision": args.dataset_revision, "ocr_quant": args.ocr_quant,
              "layout_quant": args.layout_quant or "",
              "pages_sha256": pages_digest,
              "paddle": {k: v for k, v in paddle.items() if k != "per_image"},
              "tolerances": {"teds": args.teds_tolerance, "teds_s": args.teds_s_tolerance},
              "rows": []}
    failed = False
    for backend in args.backends:
        for quant in args.quants:
            predictions, summary = run_bench(args, backend, quant, truth, boxes,
                                             environment)
            row = score(predictions, truth)
            row.update({"task": "table", "backend": backend, "quant": quant,
                        "device": summary.get("device", ""),
                        "table_p50_ms": summary.get("table_p50_ms", 0.0),
                        "table_p95_ms": summary.get("table_p95_ms", 0.0),
                        "total_p50_ms": summary.get("total_p50_ms", 0.0),
                        "total_p95_ms": summary.get("total_p95_ms", 0.0)})
            if boxes:
                row["layout_quant"] = args.layout_quant
                row["table_recall"] = summary["table_recall"]
            row["delta_teds"] = row["teds"] - paddle["teds"]
            row["delta_teds_s"] = row["teds_s"] - paddle["teds_s"]
            failures = []
            if row["delta_teds"] < -args.teds_tolerance:
                failures.append("TEDS below Paddle tolerance")
            if row["delta_teds_s"] < -args.teds_s_tolerance:
                failures.append("TEDS-S below Paddle tolerance")
            if boxes and row["table_recall"] < paddle.get("table_recall", 1.0):
                failures.append("layout found fewer GT tables than Paddle")
            row["passes_paddle_gate"] = 0 if failures else 1
            row["gate_failures"] = failures
            failed = failed or bool(failures)
            report["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "per_image"},
                             ensure_ascii=False), flush=True)
    report_path = args.output / "table_gt_matrix.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({"report": str(report_path), "rows": len(report["rows"]),
                      "paddle_teds": paddle["teds"], "paddle_teds_s": paddle["teds_s"],
                      "all_pass": 0 if failed else 1}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
