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
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # Report a clear prerequisite for the L3 geometry gate.
    np = None


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
    parser.add_argument("--runtime-library-dir", action="append", type=Path,
                        default=[], metavar="DIR",
                        help="Runtime library directory prepended for child COLMAP processes")
    parser.add_argument("--upstream-colmap", type=Path,
                        help="Pinned upstream COLMAP executable for the ONNX route")
    parser.add_argument("--upstream-revision",
                        help="Required immutable revision of --upstream-colmap in L3 mode")
    parser.add_argument("--upstream-source", type=Path,
                        help="Source checkout used to build --upstream-colmap; verified in L3 mode")
    parser.add_argument("--l3", action="store_true",
                        help="Run the fixed-input candidate-versus-upstream model gate")
    parser.add_argument("--l3-calibrate", action="store_true",
                        help="Record L3 metrics without applying thresholds; requires --l3")
    parser.add_argument("--seed", type=int,
                        help="Fixed mapper seed; required in L3 mode")
    parser.add_argument("--mapper-option", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Raw mapper option forwarded unchanged to both variants")
    parser.add_argument("--l3-thresholds", type=Path,
                        help="Calibrated JSON thresholds; required in L3 mode")
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


def child_environment(runtime_library_dirs: list[Path]) -> dict[str, str]:
    environment = os.environ.copy()
    runtime_dirs = [str(path) for path in runtime_library_dirs if path.is_dir()]
    if runtime_dirs:
        if os.name == "nt":
            library_variable, separator = "PATH", ";"
        elif sys.platform == "darwin":
            library_variable, separator = "DYLD_LIBRARY_PATH", ":"
        else:
            library_variable, separator = "LD_LIBRARY_PATH", ":"
        environment[library_variable] = separator.join(
            runtime_dirs + [environment.get(library_variable, "")]).rstrip(separator)
    return environment


def run_stage(command: list[str], log_path: Path, colmap_binary: Path | None,
              runtime_library_dirs: list[Path]) -> float:
    started = time.perf_counter()
    environment = child_environment(runtime_library_dirs)
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


def parse_text_geometry(model_text_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Read camera centers/rotations and sparse points from a TXT model."""
    if np is None:
        raise RuntimeError("L3 comparison requires numpy for Sim3 alignment")

    cameras: dict[str, Any] = {}
    image_lines = (model_text_dir / "images.txt").read_text(encoding="utf-8").splitlines()
    for line in image_lines:
        fields = line.split()
        if line.startswith("#") or len(fields) < 10:
            continue
        try:
            qvec = np.array([float(value) for value in fields[1:5]], dtype=float)
            tvec = np.array([float(value) for value in fields[5:8]], dtype=float)
            image_id = int(fields[0])
        except ValueError:
            continue
        qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
        rotation = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        cameras[fields[9]] = {"id": image_id, "rotation": rotation,
                               "center": -rotation.T @ tvec}

    points: list[list[float]] = []
    for line in (model_text_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if line.startswith("#") or len(fields) < 8:
            continue
        try:
            points.append([float(value) for value in fields[1:4]])
        except ValueError:
            continue
    return cameras, np.asarray(points, dtype=float).reshape((-1, 3))


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return target_from_source Sim3 for corresponding 3D rows."""
    if len(source) < 3:
        raise RuntimeError("L3 Sim3 alignment requires at least three common registered images")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / len(source)
    left, _, right = np.linalg.svd(covariance)
    determinant = np.linalg.det(left @ right)
    correction = np.diag([1.0, 1.0, 1.0 if determinant >= 0 else -1.0])
    rotation = left @ correction @ right
    source_variance = float(np.sum(source_zero * source_zero) / len(source))
    if source_variance <= np.finfo(float).eps:
        raise RuntimeError("L3 Sim3 source camera centers are degenerate")
    scale = float(np.trace(np.diag(_) @ correction) / source_variance)
    return scale, rotation, target_center - scale * rotation @ source_center


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def compare_l3_models(candidate_model: Path, baseline_model: Path) -> dict[str, Any]:
    """Compare candidate to baseline after camera-center Sim3 alignment.

    Point nearest-neighbour distances are O(candidate_points * baseline_points).
    L3 fixtures are intentionally small, keeping this direct metric auditable.
    """
    candidate_cameras, candidate_points = parse_text_geometry(candidate_model)
    baseline_cameras, baseline_points = parse_text_geometry(baseline_model)
    common_names = sorted(set(candidate_cameras) & set(baseline_cameras))
    source = np.asarray([candidate_cameras[name]["center"] for name in common_names])
    target = np.asarray([baseline_cameras[name]["center"] for name in common_names])
    scale, rotation, translation = umeyama(source, target)
    aligned_centers = (scale * (rotation @ source.T)).T + translation
    center_errors = np.linalg.norm(aligned_centers - target, axis=1)
    rotation_errors = []
    for name in common_names:
        candidate_rotation = candidate_cameras[name]["rotation"]
        baseline_rotation = baseline_cameras[name]["rotation"]
        difference = candidate_rotation @ rotation.T @ baseline_rotation.T
        cosine = min(1.0, max(-1.0, (np.trace(difference) - 1.0) / 2.0))
        rotation_errors.append(math.degrees(math.acos(cosine)))
    aligned_points = ((scale * (rotation @ candidate_points.T)).T + translation
                      if len(candidate_points) else candidate_points)
    if len(aligned_points) and len(baseline_points):
        forward = np.sqrt(((aligned_points[:, None, :] - baseline_points[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        reverse = np.sqrt(((baseline_points[:, None, :] - aligned_points[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        point_distances = np.concatenate((forward, reverse))
    else:
        point_distances = np.empty(0)
    return {
        "common_registered_images": len(common_names),
        "sim3_candidate_to_baseline": {"scale": scale, "rotation": rotation.tolist(),
                                         "translation": translation.tolist()},
        "camera_center_error": {"p50": percentile(center_errors, 50), "p95": percentile(center_errors, 95),
                                "max": float(center_errors.max())},
        "camera_rotation_error_deg": {"p50": percentile(np.asarray(rotation_errors), 50),
                                        "p95": percentile(np.asarray(rotation_errors), 95),
                                        "max": max(rotation_errors)},
        "symmetric_point_nn_distance": {"p50": percentile(point_distances, 50),
                                          "p95": percentile(point_distances, 95),
                                          "max": float(point_distances.max()) if len(point_distances) else None},
    }


def file_manifest(root: Path) -> list[dict[str, str]]:
    return [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def relative_difference(candidate: float, baseline: float) -> float:
    return abs(candidate - baseline) / max(abs(baseline), 1.0)


def evaluate_l3(candidate: dict[str, Any], baseline: dict[str, Any], geometry: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Return all threshold violations; each key is explicit gate policy."""
    observed = {
        "registered_images_relative_difference": relative_difference(candidate["registered_images"], baseline["registered_images"]),
        "sparse_points_relative_difference": relative_difference(candidate["sparse_points"], baseline["sparse_points"]),
        "track_observations_relative_difference": relative_difference(candidate["track_observations"], baseline["track_observations"]),
        "mean_track_length_relative_difference": relative_difference(candidate["mean_track_length"], baseline["mean_track_length"]),
        "point_reprojection_rms_px_relative_difference": relative_difference(candidate["point_reprojection_rms_px"] or 0.0, baseline["point_reprojection_rms_px"] or 0.0),
        "camera_center_p95": geometry["camera_center_error"]["p95"],
        "camera_rotation_p95_deg": geometry["camera_rotation_error_deg"]["p95"],
        "point_nn_p95": geometry["symmetric_point_nn_distance"]["p95"],
    }
    missing = sorted(set(observed) - set(thresholds))
    if missing:
        raise RuntimeError("L3 thresholds omit required keys: " + ", ".join(missing))
    return [f"{key}={value:.12g} exceeds {thresholds[key]:.12g}"
            for key, value in observed.items() if value > thresholds[key]]


def l3_configuration(args: argparse.Namespace,
                     input_manifest: list[dict[str, str]]) -> dict[str, Any]:
    """Every run input that the calibrated L3 thresholds depend on."""
    manifest_digest = hashlib.sha256(
        json.dumps(input_manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "input_manifest_sha256": manifest_digest,
        "max_features": args.max_features,
        "camera_model": args.camera_model,
        "single_camera": args.single_camera,
        "sift_gpu": args.sift_gpu,
        "seed": args.seed,
        "mapper_options": sorted(args.mapper_option),
    }


def configuration_mismatches(expected: dict[str, Any],
                             actual: dict[str, Any]) -> list[str]:
    """Thresholds are calibrated for one fixture; any other run is not gated."""
    return [f"{key}: thresholds expect {expected.get(key)!r}, run uses {actual.get(key)!r}"
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)]


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
                      str(args.max_features)]
    if args.single_camera:
        extract.append("--single-camera")
    if feature_type == "loma":
        extract += ["--loma-device", args.loma_device, "--loma-runtime", loma_runtime,
                    "--loma-variant", args.loma_variant]
        if args.loma_device == "cpu":
            extract.append("--no-gpu")
    elif not args.sift_gpu:
        extract.append("--no-gpu")
    # COLMAP assigns image ids in extraction completion order; one extraction
    # thread pins them to the sorted file order. Pair ids, two-view geometry
    # orientation, and mapper iteration all follow the image ids.
    if args.seed is not None:
        extract += ["--num-threads", "1"]

    match = base + ["match", str(database), "--method", "exhaustive"]
    if feature_type == "loma":
        match += ["--loma-device", args.loma_device, "--loma-runtime", loma_runtime,
                  "--loma-variant", args.loma_variant]
        if args.loma_device == "cpu":
            match.append("--no-gpu")
        if args.matcher_workers > 0:
            match += ["--matcher-workers", str(args.matcher_workers)]
    elif not args.sift_gpu:
        match.append("--no-gpu")
    # COLMAP reseeds the two-view geometry RANSAC per run unless a seed is
    # given; unseeded matching stores run-dependent inliers and relative poses.
    if args.seed is not None:
        match += ["--random-seed", str(args.seed)]

    sparse = base + ["sparse", "--database", str(database),
                     "--image-path", str(args.images), "--output", str(sparse_root)]
    if args.seed is not None:
        sparse += ["--random-seed", str(args.seed)]
    for mapper_option in args.mapper_option:
        sparse += ["--mapper-option", mapper_option]
    stages = {
        "feature_extraction_s": run_stage(
            extract, workspace / "extract.log", colmap_binary, args.runtime_library_dir),
        "matching_s": run_stage(
            match, workspace / "match.log", colmap_binary, args.runtime_library_dir),
        "mapping_s": run_stage(
            sparse, workspace / "mapper.log", colmap_binary, args.runtime_library_dir),
    }
    model = choose_model(sparse_root)
    run_stage(base + ["convert-model", str(model), "--output", str(model_text),
                      "--output-type", "TXT"], workspace / "convert_model.log",
              colmap_binary, args.runtime_library_dir)
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


def git_revision(source: Path) -> str:
    completed = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=False)
    if completed.returncode:
        raise RuntimeError(f"cannot resolve Git revision for {source}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def revisions_match(expected: str, actual: str) -> bool:
    revision_pattern = re.compile(r"[0-9a-fA-F]{7,64}")
    if not revision_pattern.fullmatch(expected) or not revision_pattern.fullmatch(actual):
        return False
    expected = expected.lower()
    actual = actual.lower()
    return expected.startswith(actual) or actual.startswith(expected)


def colmap_binary_revision(binary: Path,
                           runtime_library_dirs: list[Path]) -> str:
    completed = subprocess.run([str(binary), "version"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False,
                               env=child_environment(runtime_library_dirs))
    if completed.returncode:
        raise RuntimeError(
            f"cannot query COLMAP revision from {binary}: exit "
            f"{completed.returncode}: {completed.stdout.strip()}")
    match = re.search(r"\bCommit\s+([0-9a-fA-F]{7,64})\b", completed.stdout)
    if match is None:
        raise RuntimeError(
            f"cannot parse COLMAP revision from {binary} version output: "
            f"{completed.stdout.strip()}")
    return match.group(1).lower()


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
    if args.l3_calibrate and not args.l3:
        raise SystemExit("--l3-calibrate requires --l3")
    if args.l3:
        required = {
            "--acv-colmap": args.acv_colmap,
            "--upstream-colmap": args.upstream_colmap,
            "--upstream-revision": args.upstream_revision,
            "--upstream-source": args.upstream_source,
            "--seed": args.seed,
        }
        if not args.l3_calibrate:
            required["--l3-thresholds"] = args.l3_thresholds
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit("L3 requires " + ", ".join(missing))
        if np is None:
            raise SystemExit("L3 requires numpy for Sim3 alignment")
        if (not args.l3_calibrate and
                not args.l3_thresholds.is_file()):
            raise SystemExit(f"--l3-thresholds does not exist: {args.l3_thresholds}")
        verified_revision = git_revision(args.upstream_source)
        if not revisions_match(args.upstream_revision, verified_revision):
            raise SystemExit(
                f"upstream source revision mismatch: expected {args.upstream_revision}, "
                f"got {verified_revision}")
        try:
            verified_binary_revision = colmap_binary_revision(
                args.upstream_colmap, args.runtime_library_dir)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not revisions_match(args.upstream_revision, verified_binary_revision):
            raise SystemExit(
                f"upstream binary revision mismatch: expected "
                f"{args.upstream_revision}, got {verified_binary_revision} from "
                f"{args.upstream_colmap}")
    args.output.mkdir(parents=True, exist_ok=True)
    input_manifest = file_manifest(args.images)
    input_images = len(input_manifest)
    minimum_images = 3 if args.l3 else 2
    if input_images < minimum_images:
        raise SystemExit("L3 requires at least three images" if args.l3 else "at least two images are required")

    started = time.perf_counter()
    configuration = {
        "max_features": args.max_features,
        "camera_model": args.camera_model,
        "single_camera": args.single_camera,
        "loma_device": args.loma_device,
        "loma_variant": args.loma_variant,
        "matcher_workers": args.matcher_workers,
        "sift_gpu": args.sift_gpu,
    }
    if args.l3:
        thresholds = (None if args.l3_calibrate else
                      json.loads(args.l3_thresholds.read_text(encoding="utf-8")))
        run_configuration = l3_configuration(args, input_manifest)
        if thresholds is not None:
            mismatches = configuration_mismatches(
                thresholds.get("configuration", {}), run_configuration)
            if mismatches:
                raise SystemExit("L3 thresholds do not apply to this run: " +
                                 "; ".join(mismatches))
        # The public harness guarantees the stock SIFT path on every supported
        # platform. L3 compares the two COLMAP implementations, not a private
        # feature backend, so both sides use the same public SIFT argv.
        candidate = run_variant(args, "candidate", "sift", None,
                                args.acv_colmap, input_images)
        baseline = run_variant(args, "baseline_upstream", "sift", None,
                               args.upstream_colmap, input_images)
        geometry = compare_l3_models(args.output / "candidate" / "model_txt",
                                     args.output / "baseline_upstream" / "model_txt")
        violations = ([] if thresholds is None else
                      evaluate_l3(candidate, baseline, geometry, thresholds))
        comparison = {
            "schema": 4,
            "gate": "l3_normalized_model_comparison",
            "verdict": "CALIBRATION" if thresholds is None else
                       "PASS" if not violations else "FAIL",
            "violations": violations,
            "input_manifest": input_manifest,
            "configuration": {**configuration, "seed": args.seed,
                              "mapper_options": args.mapper_option,
                              "feature_type": "sift"},
            "l3_configuration": run_configuration,
            "candidate": candidate,
            "baseline": baseline,
            "baseline_revision": verified_revision,
            "baseline_binary_revision": verified_binary_revision,
            "baseline_source": str(args.upstream_source),
            "binary_sha256": {"candidate": sha256_file(args.acv_colmap),
                              "baseline": sha256_file(args.upstream_colmap)},
            "thresholds": thresholds,
            "normalized_geometry": geometry,
            "metric_notes": {
                "baseline": "The runner verifies that the source checkout and executable embedded commit match the recorded immutable revision.",
                "reprojection_rms": "RMS over COLMAP per-point mean reprojection errors.",
                "point_nn": "Symmetric nearest-neighbour distance after camera-center Sim3 alignment.",
            },
        }
    else:
        sift = run_variant(args, "sift", "sift", None, args.acv_colmap, input_images)
        acloudviewer_loma = run_variant(args, "acloudviewer_loma", "loma",
                                        "acloudviewer", args.acv_colmap, input_images)
        comparison = {
            "schema": 2,
            "images": str(args.images),
            "input_images": input_images,
            "configuration": configuration,
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
        comparison["acloudviewer_gguf"] = loma_gguf_provenance(
            args.loma_model_cache, args.loma_variant)
    comparison["images"] = str(args.images)
    comparison["input_images"] = input_images
    comparison["wall_seconds"] = time.perf_counter() - started
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 1 if args.l3 and comparison["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
