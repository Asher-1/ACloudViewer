#!/usr/bin/env python3
"""Run OCR CUDA/Vulkan model matrices against labelled ground truth.

This runner covers complete det+rec pipelines.  It requires a real GT file in
``ocr_gt_eval.py`` format and executes every det/rec quantization pair for the
selected Chinese and/or English recognition model.  Paddle is evaluated on the
same labelled images first; an AICore row passes only if its Hmean and CER stay
within the supplied relative-to-Paddle tolerances.

A layout GGUF needs a labelled region dataset and class mapping, which is a
separate metric contract; this script intentionally refuses to mis-score layout
regions as text boxes.
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
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ocr_gt_eval", SCRIPT_DIR / "ocr_gt_eval.py")
assert SPEC and SPEC.loader
GT_EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GT_EVAL
SPEC.loader.exec_module(GT_EVAL)

QUANTS = ("f32", "f16", "q8_0")


def host_state() -> dict:
    """Capture, but never mutate, the host state that changes Vulkan p95."""
    governors = {}
    for governor in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            governors[governor.parts[-3]] = governor.read_text().strip()
        except OSError:
            continue
    frequencies = {}
    for frequency in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"):
        try:
            frequencies[frequency.parts[-3]] = int(frequency.read_text().strip())
        except (OSError, ValueError):
            continue
    gpu_processes = []
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False)
        gpu_processes = [line for line in completed.stdout.splitlines() if line.strip()]
    except OSError:
        pass
    return {"cpu_governors": governors, "cpu_frequency_khz": frequencies,
            "gpu_compute_processes": gpu_processes}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_runtime_library_path(environment: dict[str, str], bench: Path,
                             extra_dirs: list[Path]) -> None:
    """Keep benchmark subprocesses on the same AICore/Qt libraries as build."""
    build_bin = bench.resolve().parent
    directories = [build_bin, build_bin.parent / "external" / "lib", *extra_dirs]
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(directory) for directory in directories if directory.is_dir()] +
        ([existing] if existing else []))


def parse_summary(output: str, source: str) -> dict:
    for line in output.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
            if "e2e_p50_ms" in payload:
                return payload
    raise RuntimeError(f"{source}: missing benchmark JSON summary")


def run(command: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stderr}")
    if not completed.stdout.strip():
        raise RuntimeError("benchmark produced no stdout; verify its runtime library path")
    return completed.stdout


def serialise_predictions(rows: list[GT_EVAL.Prediction]) -> list[dict]:
    return [{"points": [list(point) for point in row.points], "text": row.text,
             "score": row.score} for row in rows]


def evaluate_commands(images: dict[str, list[GT_EVAL.Annotation]], image_root: Path,
                      command_for_image, env: dict[str, str], output_dir: Path,
                      name: str, iou: float, case_sensitive: bool) -> dict:
    prediction_records = []
    timings = []
    raw_dir = output_dir / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    for image in sorted(images):
        image_path = image_root / image
        if not image_path.is_file():
            raise FileNotFoundError(f"GT image missing: {image_path}")
        output = run(command_for_image(image_path), env)
        (raw_dir / (Path(image).name + ".txt")).write_text(output, encoding="utf-8")
        parsed = GT_EVAL.load_predictions(raw_dir / (Path(image).name + ".txt"))
        rows = parsed.get("__single__", [])
        prediction_records.append({"image": image, "predictions": serialise_predictions(rows)})
        timings.append(parse_summary(output, str(image_path)))
    prediction_path = output_dir / f"{name}.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                        for row in prediction_records), encoding="utf-8")
    predictions = GT_EVAL.load_predictions(prediction_path)
    reports = {image: GT_EVAL.evaluate(predictions.get(image, []), annotations, iou,
                                        case_sensitive)
               for image, annotations in images.items()}
    gt_boxes = sum(report["gt_boxes"] for report in reports.values())
    pred_boxes = sum(report["predicted_boxes"] for report in reports.values())
    matched = sum(report["matched_boxes"] for report in reports.values())
    exact = sum(report["exact_text_matches"] for report in reports.values())
    char_distance = sum(report["char_distance"] for report in reports.values())
    char_reference = sum(report["char_reference"] for report in reports.values())
    word_distance = sum(report["word_distance"] for report in reports.values())
    word_reference = sum(report["word_reference"] for report in reports.values())
    precision = matched / pred_boxes if pred_boxes else 0.0
    recall = matched / gt_boxes if gt_boxes else 0.0
    return {
        "prediction_file": str(prediction_path), "images": len(images),
        "gt_boxes": gt_boxes, "predicted_boxes": pred_boxes, "matched_boxes": matched,
        "precision": precision, "recall": recall,
        "hmean": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "word_accuracy": exact / word_reference if word_reference else 0.0,
        "cer": char_distance / char_reference if char_reference else 0.0,
        "wer": word_distance / word_reference if word_reference else 0.0,
        "e2e_p50_ms": statistics.median(row["e2e_p50_ms"] for row in timings),
        "e2e_p95_ms": max(row["e2e_p95_ms"] for row in timings),
        "per_image": reports,
    }


def passes_against_paddle(candidate: dict, paddle: dict, hmean_tolerance: float,
                           cer_tolerance: float) -> tuple[bool, list[str]]:
    failures = []
    if candidate["hmean"] < paddle["hmean"] - hmean_tolerance:
        failures.append("hmean below Paddle tolerance")
    if candidate["cer"] > paddle["cer"] + cer_tolerance:
        failures.append("CER above Paddle tolerance")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--ground-truth-sha256", required=True,
                        help="published SHA-256 for the exact GT label file")
    parser.add_argument("--dataset-revision", required=True,
                        help="official dataset/version identifier recorded in the report")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--runtime-lib-dir", type=Path, action="append", default=[],
                        help="extra Qt/AICore runtime library directory; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=("ch", "en"),
                        default=("ch", "en"))
    parser.add_argument("--backends", nargs="+", choices=("cuda", "vulkan"),
                        default=("cuda", "vulkan"))
    parser.add_argument("--quants", nargs="+", choices=QUANTS, default=QUANTS,
                        help="det x rec quantizations; each image costs one process per pair")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--hmean-tolerance", type=float, default=0.005)
    parser.add_argument("--cer-tolerance", type=float, default=0.005)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--recognize-full-image", action="store_true",
                        help="evaluate cropped-word GT through the public recognition-only ABI")
    parser.add_argument("--require-performance-governor", action="store_true",
                        help="refuse an uncontrolled Vulkan latency run; never changes host state")
    parser.add_argument("--skip-paddle", action="store_true")
    parser.add_argument("--paddle-cpu", action="store_true",
                        help="run the Paddle reference on CPU: Paddle GPU det maps carry "
                             "~2.6e-2 kernel noise against Paddle CPU, which moves boxes")
    parser.add_argument("--paddle-bench", type=Path,
                        default=SCRIPT_DIR / "bench_paddle_official.py")
    args = parser.parse_args()
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be in (0, 1]")
    actual_gt_digest = sha256_file(args.ground_truth)
    if actual_gt_digest.lower() != args.ground_truth_sha256.lower():
        parser.error("ground-truth SHA-256 mismatch: "
                     f"expected {args.ground_truth_sha256}, got {actual_gt_digest}")
    images = GT_EVAL.load_ground_truth(args.ground_truth)
    if not images:
        parser.error("ground truth contains no images")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    add_runtime_library_path(environment, args.bench, args.runtime_lib_dir)
    measured_host = host_state()
    if args.require_performance_governor:
        non_performance = {cpu: value for cpu, value in measured_host["cpu_governors"].items()
                           if value != "performance"}
        if non_performance:
            parser.error("controlled latency run requires performance governor; "
                         f"found {non_performance}")
        if measured_host["gpu_compute_processes"]:
            parser.error("controlled latency run requires no foreign GPU process; "
                         f"found {measured_host['gpu_compute_processes']}")
    results: dict[str, object] = {
        "ground_truth": str(args.ground_truth),
        "ground_truth_sha256": actual_gt_digest,
        "dataset_revision": args.dataset_revision,
        "image_root": str(args.image_root),
        "iou": args.iou, "host_state": measured_host, "rows": []}
    paddle_reports = {}
    if not args.skip_paddle:
        for language in args.languages:
            paddle_reports[language] = evaluate_commands(
                images, args.image_root,
                lambda image, language=language: [sys.executable, str(args.paddle_bench),
                    str(image), str(args.warmups), str(args.iters), "--language", language,
                    "--det-gguf", str(args.models_dir / "ppocr-det-ch-f32.gguf"),
                    "--rec-gguf", str(args.models_dir / f"ppocr-rec-{language}-f32.gguf")] +
                    ((["--recognize-full-image"] if args.recognize_full_image else []) +
                     (["--cpu"] if args.paddle_cpu else [])),
                environment, args.output, f"paddle-{language}", args.iou,
                args.case_sensitive)
    for language in args.languages:
        rec_stem = f"ppocr-rec-{language}"
        for backend in args.backends:
            for det_quant in args.quants:
                det_model = args.models_dir / f"ppocr-det-ch-{det_quant}.gguf"
                if not det_model.is_file():
                    raise FileNotFoundError(det_model)
                for rec_quant in args.quants:
                    rec_model = args.models_dir / f"{rec_stem}-{rec_quant}.gguf"
                    if not rec_model.is_file():
                        raise FileNotFoundError(rec_model)
                    name = f"{language}_{backend}_det-{det_quant}_rec-{rec_quant}"
                    report = evaluate_commands(
                        images, args.image_root,
                        lambda image, det_model=det_model, rec_model=rec_model, backend=backend: [
                            str(args.bench), str(image), str(det_model), str(rec_model), backend,
                            str(args.threads), str(args.warmups), str(args.iters)] +
                            (["--recognize-full-image"] if args.recognize_full_image else []),
                        environment, args.output, name, args.iou, args.case_sensitive)
                    row = {"language": language, "backend": backend,
                           "det_quant": det_quant, "rec_quant": rec_quant, **report}
                    if language in paddle_reports:
                        passed, failures = passes_against_paddle(
                            report, paddle_reports[language], args.hmean_tolerance,
                            args.cer_tolerance)
                        row["paddle"] = paddle_reports[language]
                        row["passes_paddle_gate"] = passed
                        row["gate_failures"] = failures
                    results["rows"].append(row)
    results["paddle"] = paddle_reports
    results["paddle_device"] = "cpu" if args.paddle_cpu else "gpu"
    report_path = args.output / "ocr_gt_matrix.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"report": str(report_path), "rows": len(results["rows"]),
                      "paddle_languages": sorted(paddle_reports)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
