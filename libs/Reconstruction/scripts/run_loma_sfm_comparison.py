#!/usr/bin/env python3
"""Run reproducible SIFT-versus-LoMa sparse SfM through the public CLI harness.

The script deliberately never calls Colmap directly.  It exercises the same
``cli-anything-acloudviewer reconstruct`` entry points exposed to automation,
keeps a per-stage transcript, and writes metrics that make the comparison
auditable without treating synthetic matcher output as reconstruction evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOMA_SOURCES = REPOSITORY_ROOT / "core/AICore/tools/loma_sources.json"


def default_loma_model_cache() -> Path:
    data_root_text = os.environ.get("CLOUDVIEWER_DATA_ROOT", "")
    data_root = (Path(data_root_text) if data_root_text
                 else Path.home() / "cloudViewer_data")
    return data_root / "extract" / "loma_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty output directory")
    parser.add_argument("--harness", default="cli-anything-acloudviewer")
    parser.add_argument("--acv-colmap", type=Path,
                        help="ACloudViewer Colmap executable; defaults to harness discovery")
    parser.add_argument("--upstream-colmap", type=Path,
                        help="Current upstream COLMAP executable. Enables ONNX-versus-GGUF A/B")
    parser.add_argument("--loma-variant", choices=("b", "b128", "r", "l", "g"),
                        default="b")
    parser.add_argument("--loma-sources", type=Path, default=DEFAULT_LOMA_SOURCES,
                        help="Pinned upstream ONNX source manifest")
    parser.add_argument("--loma-model-cache", type=Path,
                        default=default_loma_model_cache(),
                        help="Shared AICore LoMa cache; defaults to "
                             "${CLOUDVIEWER_DATA_ROOT:-~/cloudViewer_data}/extract/loma_models")
    parser.add_argument("--loma-device", default="auto")
    parser.add_argument("--matcher-workers", type=int, default=-1)
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--sift-gpu", action="store_true",
                        help="Use SIFT GPU; default keeps both paths on CPU")
    return parser.parse_args()


def run_stage(command: list[str], log_path: Path, colmap_binary: Path | None) -> float:
    started = time.perf_counter()
    environment = os.environ.copy()
    if colmap_binary is not None:
        environment["COLMAP_PATH"] = str(colmap_binary)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False, env=environment)
    elapsed = time.perf_counter() - started
    launcher = ""
    if colmap_binary is not None:
        launcher = f"COLMAP_PATH={colmap_binary} "
    log_path.write_text(
        "$ " + launcher + " ".join(command) + "\n\n" + completed.stdout +
        f"\n[exit_code={completed.returncode} wall_seconds={elapsed:.6f}]\n",
        encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"stage failed ({completed.returncode}): {' '.join(command)}\n"
            f"see {log_path}")
    return elapsed


def choose_model(sparse_root: Path) -> Path:
    candidates = sorted(path for path in sparse_root.iterdir()
                        if path.is_dir() and path.name.isdigit())
    if not candidates:
        raise RuntimeError(f"mapper produced no sparse model below {sparse_root}")
    return candidates[0]


def parse_text_model(model_text_dir: Path, input_images: int) -> dict[str, Any]:
    images_txt = model_text_dir / "images.txt"
    points_txt = model_text_dir / "points3D.txt"
    registered = 0
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if line.startswith("#") or len(fields) < 10:
            continue
        try:
            int(fields[0])
            int(fields[8])
            [float(value) for value in fields[1:8]]
        except ValueError:
            continue
        registered += 1

    point_count = 0
    observations = 0
    squared_error = 0.0
    for line in points_txt.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if line.startswith("#") or len(fields) < 8:
            continue
        try:
            int(fields[0])
            error = float(fields[7])
            if not math.isfinite(error):
                continue
        except ValueError:
            continue
        point_count += 1
        squared_error += error * error
        observations += max(0, (len(fields) - 8) // 2)

    return {
        "input_images": input_images,
        "registered_images": registered,
        "registered_image_fraction": registered / input_images if input_images else 0.0,
        "sparse_points": point_count,
        "track_observations": observations,
        "mean_track_length": observations / point_count if point_count else 0.0,
        # COLMAP stores mean reprojection error per point. This is therefore a
        # point-weighted RMS, not the unavailable per-observation RMS.
        "point_reprojection_rms_px": math.sqrt(squared_error / point_count)
        if point_count else None,
        "sparse_completeness_points_per_registered_image": point_count / registered
        if registered else 0.0,
    }


def database_counts(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        result = {}
        for table in ("keypoints", "descriptors", "float_descriptors", "matches",
                      "two_view_geometries"):
            try:
                result[f"database_{table}_rows"] = connection.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.DatabaseError:
                result[f"database_{table}_rows"] = 0
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loma_source_provenance(manifest: Path, variant: str) -> dict[str, Any]:
    with manifest.open(encoding="utf-8") as source:
        models = json.load(source)["models"]
    descriptor = "descriptor_dedode_b" if variant == "b128" else "descriptor_dedode_g"
    matcher = f"matcher_{variant.upper()}"
    selected = {"detector": models["detector_B"], "descriptor": models[descriptor],
                "matcher": models[matcher]}
    return {
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "models": selected,
    }


def loma_gguf_provenance(cache_dir: Path, variant: str) -> dict[str, Any]:
    """Report the exact F32 triplet loaded from the shared AICore cache."""
    descriptor = "loma_descriptor_dedode_b" if variant == "b128" else "loma_descriptor_dedode_g"
    selected_names = {
        "detector": "loma_detector.f32.gguf",
        "descriptor": f"{descriptor}.f32.gguf",
        "matcher": f"loma_matcher_{variant.upper()}.f32.gguf",
    }
    missing = [name for name in selected_names.values()
               if not (cache_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "LoMa GGUF cache is missing required artifacts: " +
            ", ".join(missing) + ". Reconstruction should have provisioned "
            "the default catalog models before this report is written.")
    return {
        "cache_dir": str(cache_dir),
        "precision": "f32",
        "models": {
            role: {
                "filename": filename,
                "path": str(cache_dir / filename),
                "sha256": sha256_file(cache_dir / filename),
            }
            for role, filename in selected_names.items()
        },
    }


def run_variant(args: argparse.Namespace, name: str, feature_type: str,
                loma_runtime: str | None, colmap_binary: Path | None,
                input_images: int) -> dict[str, Any]:
    workspace = args.output / name
    workspace.mkdir()
    database = workspace / "database.db"
    sparse_root = workspace / "sparse"
    model_text = workspace / "model_txt"
    base = [args.harness, "--mode", "headless", "reconstruct"]

    extract = base + ["extract-features", str(args.images), "--database", str(database),
                      "--camera-model", args.camera_model, "--max-features",
                      str(args.max_features), "--feature-type", feature_type]
    if args.single_camera:
        extract.append("--single-camera")
    if feature_type == "loma":
        extract += ["--loma-device", args.loma_device, "--loma-runtime", loma_runtime,
                    "--loma-variant", args.loma_variant]
        if args.loma_device == "cpu":
            extract.append("--no-gpu")
    elif not args.sift_gpu:
        extract.append("--no-gpu")

    match = base + ["match", str(database), "--method", "exhaustive",
                    "--feature-type", feature_type]
    if feature_type == "loma":
        match += ["--loma-device", args.loma_device, "--loma-runtime", loma_runtime,
                  "--loma-variant", args.loma_variant]
        if args.loma_device == "cpu":
            match.append("--no-gpu")
        if args.matcher_workers > 0:
            match += ["--matcher-workers", str(args.matcher_workers)]
    elif not args.sift_gpu:
        match.append("--no-gpu")

    stages = {
        "feature_extraction_s": run_stage(extract, workspace / "extract.log", colmap_binary),
        "matching_s": run_stage(match, workspace / "match.log", colmap_binary),
        "mapping_s": run_stage(base + ["sparse", "--database", str(database),
                                         "--image-path", str(args.images), "--output",
                                         str(sparse_root)], workspace / "mapper.log", colmap_binary),
    }
    model = choose_model(sparse_root)
    run_stage(base + ["convert-model", str(model), "--output", str(model_text),
                      "--output-type", "TXT"], workspace / "convert_model.log", colmap_binary)
    metrics = parse_text_model(model_text, input_images)
    metrics.update(database_counts(database))
    metrics["stage_seconds"] = stages
    metrics["total_reconstruction_seconds"] = sum(stages.values())
    metrics["model_path"] = str(model)
    metrics["colmap_binary"] = str(colmap_binary) if colmap_binary else "harness discovery"
    if loma_runtime:
        metrics["loma_runtime"] = loma_runtime
        metrics["loma_variant"] = args.loma_variant
    (workspace / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                              encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"images directory does not exist: {args.images}")
    if args.max_features < 1:
        raise SystemExit("--max-features must be positive")
    for binary_name, binary in (("--acv-colmap", args.acv_colmap),
                                ("--upstream-colmap", args.upstream_colmap)):
        if binary is not None and (not binary.is_file() or not os.access(binary, os.X_OK)):
            raise SystemExit(f"{binary_name} must name an executable: {binary}")
    if args.upstream_colmap is not None and not args.loma_sources.is_file():
        raise SystemExit(f"--loma-sources does not exist: {args.loma_sources}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    input_images = sum(path.suffix.lower() in IMAGE_SUFFIXES
                       for path in args.images.iterdir() if path.is_file())
    if input_images < 2:
        raise SystemExit("at least two images are required")

    started = time.perf_counter()
    sift = run_variant(args, "sift", "sift", None, args.acv_colmap, input_images)
    acloudviewer_loma = run_variant(args, "acloudviewer_loma", "loma",
                                    "acloudviewer", args.acv_colmap, input_images)
    comparison = {
        "schema": 2,
        "images": str(args.images),
        "input_images": input_images,
        "configuration": {
            "max_features": args.max_features,
            "camera_model": args.camera_model,
            "single_camera": args.single_camera,
            "loma_device": args.loma_device,
            "loma_variant": args.loma_variant,
            "matcher_workers": args.matcher_workers,
            "sift_gpu": args.sift_gpu,
        },
        "sift": sift,
        "acloudviewer_loma": acloudviewer_loma,
        "metric_notes": {
            "tracks": "track_observations and mean_track_length are parsed from points3D.txt",
            "reprojection_rms": "RMS over COLMAP per-point mean reprojection errors",
            "sparse_completeness": "registered image fraction and points per registered image",
            "timing": "stage and total timings include process startup and model loading; they are end-to-end SfM timings, not a warm-model kernel microbenchmark",
        },
    }
    if args.upstream_colmap is not None:
        comparison["upstream_loma"] = run_variant(
            args, "upstream_loma", "loma", "upstream", args.upstream_colmap,
            input_images)
        comparison["upstream_loma_source"] = loma_source_provenance(
            args.loma_sources, args.loma_variant)
        comparison["metric_notes"]["runtime_comparison"] = (
            "Both LoMa routes use the COLMAP-pinned F32 source graphs; the "
            "ACloudViewer route consumes their GGUF conversion and the upstream "
            "route consumes ONNX through current COLMAP.")
    comparison["acloudviewer_gguf"] = loma_gguf_provenance(
        args.loma_model_cache, args.loma_variant)
    comparison["wall_seconds"] = time.perf_counter() - started
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
