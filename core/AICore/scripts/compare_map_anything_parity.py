#!/usr/bin/env python3
"""Fail-closed numerical, PLY, camera, and screenshot parity gate for qMapAnything.

The candidate RAW_OUTPUT_DIR contains pose.bin, depth.bin, depth_conf.bin, and
camera_to_world.bin.  Reference files may use the same names or the historical
vggt-cli '<prefix>.<name>' layout.  This tool deliberately does not fit a new
rigid/similarity transform: both exports must already use the benchmark frame.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-raw", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--views", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--candidate-ply", type=Path)
    parser.add_argument("--reference-ply", type=Path)
    parser.add_argument("--candidate-result", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--candidate-screenshot", type=Path)
    parser.add_argument("--reference-screenshot", type=Path)
    parser.add_argument("--candidate-stages", type=Path)
    parser.add_argument("--reference-stages", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-atol", type=float, default=1.0e-5)
    parser.add_argument("--pose-center-atol", type=float, default=None,
                        help="Maximum c2w camera-center absolute error; defaults to --raw-atol.")
    parser.add_argument("--pose-angle-deg-max", type=float, default=1.0e-3,
                        help="Maximum right/forward/renderer-up angular error in degrees.")
    parser.add_argument("--ply-rms-max", type=float, default=1.0e-4)
    parser.add_argument("--ply-p95-max", type=float, default=2.0e-4)
    parser.add_argument("--image-mae-max", type=float, default=0.0)
    parser.add_argument("--stage-atol", type=float, default=1.0e-5)
    parser.add_argument("--sample-count", type=int, default=100000)
    return parser.parse_args()


def resolve_raw(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.is_file():
        return direct
    matches = sorted(directory.glob("*." + name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot uniquely locate {name} in {directory}")


def read_f32(path: Path, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != count:
        raise ValueError(f"{path}: expected {count} f32 values, got {values.size}")
    return values


def tensor_stats(candidate: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "count": int(delta.size),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean() if delta.size else 0.0),
        "median_abs": float(np.median(delta) if delta.size else 0.0),
        "p95_abs": float(np.quantile(delta, 0.95) if delta.size else 0.0),
    }


def pose_to_c2w(pose: np.ndarray) -> np.ndarray:
    """Convert upstream VGGT-Omega [T, quat_xyzw, fov_h, fov_w] w2c rows."""
    if pose.size % 9:
        raise ValueError("w2c pose encoding is not a multiple of 9 floats")
    rows = pose.reshape(-1, 9).astype(np.float64)
    quat = rows[:, 3:7]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    x, y, z, w = (quat[:, index] for index in range(4))
    rotation = np.empty((len(rows), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    c2w = np.zeros((len(rows), 4, 4), dtype=np.float32)
    c2w[:, :3, :3] = np.transpose(rotation, (0, 2, 1))
    c2w[:, :3, 3] = -np.einsum("nij,nj->ni", c2w[:, :3, :3], rows[:, :3])
    c2w[:, 3, 3] = 1.0
    return c2w.reshape(-1)


def direction_stats(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Compare unit direction vectors without accepting a frame realignment."""
    candidate_norm = np.linalg.norm(candidate, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    if np.any(candidate_norm == 0.0) or np.any(reference_norm == 0.0):
        raise ValueError("pose contains a zero-length orientation axis")
    candidate_unit = candidate / candidate_norm[:, None]
    reference_unit = reference / reference_norm[:, None]
    cosine = np.clip(np.sum(candidate_unit * reference_unit, axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosine))
    stats = tensor_stats(candidate_unit, reference_unit)
    stats.update({
        "max_angle_deg": float(angles_deg.max(initial=0.0)),
        "mean_angle_deg": float(angles_deg.mean() if angles_deg.size else 0.0),
        "p95_angle_deg": float(np.quantile(angles_deg, 0.95) if angles_deg.size else 0.0),
    })
    return stats


def c2w_pose_stats(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.size % 16 or reference.size % 16:
        raise ValueError("camera_to_world output is not a multiple of 16 floats")
    candidate_matrix = candidate.reshape(-1, 4, 4).astype(np.float64)
    reference_matrix = reference.reshape(-1, 4, 4).astype(np.float64)
    if candidate_matrix.shape != reference_matrix.shape:
        raise ValueError("camera_to_world matrix shape mismatch")
    # VtkColmap keeps OpenCV camera coordinates: +Z is forward and +Y is
    # image-down.  The renderer-facing up direction is therefore -Y.
    return {
        "matrix": tensor_stats(candidate_matrix, reference_matrix),
        "camera_center": tensor_stats(candidate_matrix[:, :3, 3],
                                      reference_matrix[:, :3, 3]),
        "right": direction_stats(candidate_matrix[:, :3, 0],
                                  reference_matrix[:, :3, 0]),
        "forward": direction_stats(candidate_matrix[:, :3, 2],
                                    reference_matrix[:, :3, 2]),
        "renderer_up": direction_stats(-candidate_matrix[:, :3, 1],
                                        -reference_matrix[:, :3, 1]),
    }


def read_binary_ply(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: missing PLY end_header")
            header.extend(line)
        text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in text:
            raise ValueError(f"{path}: only binary_little_endian PLY is supported")
        vertices = next((int(line.split()[2]) for line in text.splitlines()
                         if line.startswith("element vertex ")), None)
        if vertices is None:
            raise ValueError(f"{path}: missing vertex count")
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = np.fromfile(handle, dtype=dtype, count=vertices)
    if data.size != vertices:
        raise ValueError(f"{path}: truncated vertex payload")
    return np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)


def sampled_nearest_distances(source: np.ndarray, target: np.ndarray, sample_count: int) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return np.array([np.inf])
    sample = source[np.linspace(0, len(source) - 1,
                                min(sample_count, len(source)), dtype=np.int64)]
    try:
        from scipy.spatial import cKDTree
        return cKDTree(target).query(sample, k=1, workers=-1)[0]
    except ImportError as exc:
        raise RuntimeError("scipy is required for point-cloud nearest-neighbor parity") from exc


def ply_stats(candidate: np.ndarray, reference: np.ndarray, sample_count: int) -> dict:
    forward = sampled_nearest_distances(candidate, reference, sample_count)
    reverse = sampled_nearest_distances(reference, candidate, sample_count)
    joined = np.concatenate((forward, reverse))
    return {
        "candidate_points": int(len(candidate)),
        "reference_points": int(len(reference)),
        "candidate_bbox": [candidate.min(axis=0).tolist(), candidate.max(axis=0).tolist()] if len(candidate) else [],
        "reference_bbox": [reference.min(axis=0).tolist(), reference.max(axis=0).tolist()] if len(reference) else [],
        "symmetric_nn_rms": float(np.sqrt(np.mean(joined * joined))),
        "symmetric_nn_p95": float(np.quantile(joined, 0.95)),
        "symmetric_nn_max": float(joined.max()),
    }


def stage_stats(candidate_dir: Path, reference_dir: Path, atol: float) -> dict:
    candidate = {path.name: path for path in candidate_dir.glob("*.bin")}
    reference = {path.name: path for path in reference_dir.glob("*.bin")}
    common = sorted(candidate.keys() & reference.keys())
    if not common:
        raise ValueError("stage directories have no common .bin tensors")
    output, failures = {}, []
    for name in common:
        lhs = np.fromfile(candidate[name], dtype="<f4")
        rhs = np.fromfile(reference[name], dtype="<f4")
        if lhs.size != rhs.size:
            failures.append(f"{name}: element count {lhs.size} != {rhs.size}")
            continue
        output[name] = tensor_stats(lhs, rhs)
        if output[name]["max_abs"] > atol:
            failures.append(name)
    # Lexical order is not execution order. The leading backbone observations
    # identify whether divergence begins before any transformer operation.
    stage_order = ["cpp_conv4.bin", "cpp_tokens0.bin", "cpp_blk0.bin",
                   "cpp_blk1.bin", "cpp_blk3.bin", "cpp_blk7.bin",
                   "cpp_blk11.bin", "cpp_blk15.bin", "cpp_blk19.bin",
                   "cpp_blk23.bin", "cpp_patches.bin"]
    ordered_failures = [name for name in stage_order if name in failures]
    ordered_failures.extend(name for name in failures if name not in stage_order)
    return {"common": common, "tensors": output,
            "failing_tensors": ordered_failures,
            "first_diverging_tensor": ordered_failures[0] if ordered_failures else None,
            "candidate_only": sorted(candidate.keys() - reference.keys()),
            "reference_only": sorted(reference.keys() - candidate.keys())}


def screenshot_stats(candidate_path: Path, reference_path: Path) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for screenshot parity") from exc
    candidate = np.asarray(Image.open(candidate_path).convert("RGBA"), dtype=np.int16)
    reference = np.asarray(Image.open(reference_path).convert("RGBA"), dtype=np.int16)
    if candidate.shape != reference.shape:
        raise ValueError(f"screenshot shape mismatch: {candidate.shape} vs {reference.shape}")
    delta = np.abs(candidate - reference)
    return {"shape": list(candidate.shape), "mae": float(delta.mean()), "max_abs": int(delta.max()),
            "changed_pixels": int(np.any(delta != 0, axis=2).sum())}


def json_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return {key: value.get(key) for key in ("architecture", "views", "height", "width", "points", "backend", "model")}


def main() -> int:
    args = parse_args()
    report = {"pass": True, "raw": {}, "failures": []}
    try:
        candidate_bins = {path.name: path for path in args.candidate_raw.glob("*.bin")}
        reference_bins = {}
        for name in candidate_bins:
            try:
                reference_bins[name] = resolve_raw(args.reference_raw, name)
            except FileNotFoundError:
                pass
        common_bins = sorted((candidate_bins.keys() & reference_bins.keys()) -
                             {"camera_to_world.bin"})
        if "depth.bin" not in common_bins:
            raise ValueError("both raw directories must contain depth.bin")
        if not common_bins:
            raise ValueError("raw directories have no common typed output files")
        report["raw_files"] = common_bins
        raw_values = {}
        for name in common_bins:
            candidate = np.fromfile(candidate_bins[name], dtype="<f4")
            reference = np.fromfile(reference_bins[name], dtype="<f4")
            if candidate.size != reference.size:
                report["failures"].append(
                        f"{name} element count {candidate.size} != {reference.size}")
                continue
            raw_values[name] = (candidate, reference)
            stats = tensor_stats(candidate, reference)
            report["raw"][name] = stats
            if stats["max_abs"] > args.raw_atol:
                report["failures"].append(
                        f"{name} max_abs {stats['max_abs']:.9g} > {args.raw_atol:.9g}")
        if "camera_to_world.bin" in candidate_bins and "pose.bin" in raw_values:
            candidate_c2w = np.fromfile(candidate_bins["camera_to_world.bin"], dtype="<f4")
            reference_pose = raw_values["pose.bin"][1]
            reference_c2w = (pose_to_c2w(reference_pose)
                             if reference_pose.size % 9 == 0 else reference_pose)
            if candidate_c2w.size != reference_c2w.size:
                report["failures"].append("camera_to_world.bin element count mismatch")
            else:
                pose_stats = c2w_pose_stats(candidate_c2w, reference_c2w)
                report["raw"]["camera_to_world.bin"] = pose_stats
                center_atol = (args.raw_atol if args.pose_center_atol is None
                               else args.pose_center_atol)
                if pose_stats["matrix"]["max_abs"] > args.raw_atol:
                    report["failures"].append(
                            "camera_to_world.bin matrix differs from reference pose")
                if pose_stats["camera_center"]["max_abs"] > center_atol:
                    report["failures"].append(
                            "camera center differs from reference pose")
                for axis in ("right", "forward", "renderer_up"):
                    if pose_stats[axis]["max_angle_deg"] > args.pose_angle_deg_max:
                        report["failures"].append(
                                f"camera {axis} exceeds angular pose threshold")
        report["candidate_only_raw"] = sorted(candidate_bins.keys() - reference_bins.keys())
        report["reference_only_raw"] = sorted(
                path.name for path in args.reference_raw.glob("*.bin")
                if path not in reference_bins.values())

        if args.candidate_ply or args.reference_ply:
            if not (args.candidate_ply and args.reference_ply):
                raise ValueError("both PLY arguments are required together")
            report["ply"] = ply_stats(read_binary_ply(args.candidate_ply),
                                      read_binary_ply(args.reference_ply), args.sample_count)
            if report["ply"]["symmetric_nn_rms"] > args.ply_rms_max:
                report["failures"].append("PLY symmetric NN RMS exceeds threshold")
            if report["ply"]["symmetric_nn_p95"] > args.ply_p95_max:
                report["failures"].append("PLY symmetric NN p95 exceeds threshold")

        if args.candidate_result and args.reference_result:
            report["result_metadata"] = {"candidate": json_metadata(args.candidate_result),
                                         "reference": json_metadata(args.reference_result)}
        if args.candidate_stages or args.reference_stages:
            if not (args.candidate_stages and args.reference_stages):
                raise ValueError("both stage directory arguments are required together")
            report["stages"] = stage_stats(args.candidate_stages, args.reference_stages,
                                           args.stage_atol)
            if report["stages"]["failing_tensors"]:
                report["failures"].append("stage tensors exceed threshold")
        if args.candidate_screenshot or args.reference_screenshot:
            if not (args.candidate_screenshot and args.reference_screenshot):
                raise ValueError("both screenshot arguments are required together")
            report["screenshot"] = screenshot_stats(args.candidate_screenshot, args.reference_screenshot)
            if report["screenshot"]["mae"] > args.image_mae_max:
                report["failures"].append("screenshot MAE exceeds threshold")
    except Exception as exc:
        report["failures"].append(str(exc))
    report["pass"] = not report["failures"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
