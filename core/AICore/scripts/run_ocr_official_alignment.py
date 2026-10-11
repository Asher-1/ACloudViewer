#!/usr/bin/env python3
"""Run same-protocol AICore CUDA/Vulkan OCR comparisons against PaddleOCR.

usage:
    run_ocr_official_alignment.py <bench-ocr> <image> <det.gguf> <rec.gguf>
                                  [--devices cuda vulkan] [--threads 0]
                                  [--warmups 3] [--iters 10] [--details]
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_ocr_text", SCRIPT_DIR / "compare_ocr_text.py")
assert COMPARE_SPEC and COMPARE_SPEC.loader
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = COMPARE
COMPARE_SPEC.loader.exec_module(COMPARE)


def run(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}")
    return completed.stdout


def summary(report: dict, details: bool) -> dict:
    if not details:
        report = {key: value for key, value in report.items() if key != "pairs"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_ocr", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("det_model", type=Path)
    parser.add_argument("rec_model", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cuda", "vulkan"])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-center-distance", type=float, default=30.0)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    paddle_output = run([sys.executable, str(SCRIPT_DIR / "bench_paddle_official.py"),
                         str(args.image), str(args.warmups), str(args.iters),
                         "--det-gguf", str(args.det_model),
                         "--rec-gguf", str(args.rec_model)])
    paddle_rows = COMPARE.read_text_rows_from_output(paddle_output)
    results = {}
    failed = False
    for device in args.devices:
        try:
            aicore_output = run([str(args.bench_ocr), str(args.image),
                                 str(args.det_model), str(args.rec_model), device,
                                 str(args.threads), str(args.warmups), str(args.iters)])
            report = COMPARE.compare(
                    COMPARE.read_text_rows_from_output(aicore_output), paddle_rows,
                    args.max_center_distance)
            results[device] = summary(report, args.details)
        except RuntimeError as error:
            failed = True
            results[device] = {"status": "error", "error": str(error)}
    print(json.dumps({"paddle_rows": len(paddle_rows), "devices": results},
                     ensure_ascii=False, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
