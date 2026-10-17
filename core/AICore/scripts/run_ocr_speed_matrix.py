#!/usr/bin/env python3
"""Run the reproducible OCR latency matrix against official PaddleOCR.

Each image is decoded outside the timed loop by both benchmark programs. Paddle
runs once per image with the exact source det/rec weights recorded in the F32
GGUFs. AICore runs every backend x det quant x rec quant combination. The raw
JSON rows and aggregate ranges are written to one report; this script does not
turn an uncontrolled host into release evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

QUANTS = ("f32", "f16", "q8_0")


def run(command: list[str], env: dict[str, str]) -> dict:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}")
    for line in reversed(completed.stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "e2e_p50_ms" in row:
            return row
    raise RuntimeError(f"no timing JSON from: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--backends", nargs="+", default=("cuda", "vulkan"))
    parser.add_argument("--quants", nargs="+", choices=QUANTS, default=QUANTS)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    paddle_script = script_dir / "bench_paddle_official.py"
    det_f32 = args.models_dir / "ppocr-det-ch-f32.gguf"
    rec_f32 = args.models_dir / "ppocr-rec-ch-f32.gguf"
    required = [args.bench, paddle_script, det_f32, rec_f32, *args.images]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")

    env = dict(os.environ)
    lib_dir = str(args.bench.resolve().parent)
    env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    rows = []
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("protocol") == {"warmups": args.warmups,
                                        "iters": args.iters,
                                        "threads": args.threads}:
            rows = previous.get("rows", [])

    def checkpoint() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = {"protocol": {"warmups": args.warmups,
                                "iters": args.iters,
                                "threads": args.threads},
                   "backends": args.backends, "quants": args.quants,
                   "rows": rows, "complete": False}
        args.output.write_text(json.dumps(partial, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")

    for image in args.images:
        paddle = next((row for row in rows
                       if row["image"] == str(image) and
                       "backend" not in row), None)
        if paddle is None:
            result = run([
                sys.executable, str(paddle_script), str(image), str(args.warmups),
                str(args.iters), "--det-gguf", str(det_f32), "--rec-gguf",
                str(rec_f32)], env)
            paddle = {**result, "image": str(image), "pipeline": "paddle"}
            rows.append(paddle)
            checkpoint()
        for backend in args.backends:
            for det_quant in args.quants:
                for rec_quant in args.quants:
                    existing = next((row for row in rows
                                     if row["image"] == str(image) and
                                     row.get("backend") == backend and
                                     row.get("det_quant") == det_quant and
                                     row.get("rec_quant") == rec_quant), None)
                    if existing is not None:
                        continue
                    det = args.models_dir / f"ppocr-det-ch-{det_quant}.gguf"
                    rec = args.models_dir / f"ppocr-rec-ch-{rec_quant}.gguf"
                    result = run([
                        str(args.bench), str(image), str(det), str(rec), backend,
                        str(args.threads), str(args.warmups), str(args.iters)], env)
                    row = {"image": str(image), "pipeline": "aicore",
                           "backend": backend, "det_quant": det_quant,
                           "rec_quant": rec_quant, **result}
                    row["speedup_vs_paddle_p50"] = (
                        paddle["e2e_p50_ms"] / result["e2e_p50_ms"])
                    rows.append(row)
                    checkpoint()
                    print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {}
    for image in args.images:
        image_rows = [row for row in rows if row["image"] == str(image)]
        paddle = next(row for row in image_rows if "backend" not in row)
        by_backend = {}
        for backend in args.backends:
            values = [row for row in image_rows
                      if row.get("backend") == backend]
            by_backend[backend] = {
                "p50_min_ms": min(row["e2e_p50_ms"] for row in values),
                "p50_max_ms": max(row["e2e_p50_ms"] for row in values),
                "p95_min_ms": min(row["e2e_p95_ms"] for row in values),
                "p95_max_ms": max(row["e2e_p95_ms"] for row in values),
                "boxes": sorted({row["boxes"] for row in values}),
                "median_speedup_vs_paddle": statistics.median(
                    row["speedup_vs_paddle_p50"] for row in values),
            }
        summary[str(image)] = {
            "paddle_p50_ms": paddle["e2e_p50_ms"],
            "paddle_p95_ms": paddle["e2e_p95_ms"],
            "paddle_texts": paddle["texts"],
            "aicore": by_backend,
        }

    report = {"protocol": {"warmups": args.warmups, "iters": args.iters,
                            "threads": args.threads},
              "backends": args.backends, "quants": args.quants,
              "summary": summary, "rows": rows, "complete": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({"report": str(args.output), "summary": summary},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
