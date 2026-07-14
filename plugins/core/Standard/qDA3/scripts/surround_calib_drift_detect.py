#!/usr/bin/env python3
"""
surround_calib_drift_detect.py — Detect extrinsic calibration drift for surround-view cameras
using DA3 metric depth + ground plane constraint.

Principle:
  For a vehicle with 4 surround cameras (front/rear/left/right), if the extrinsic
  calibration is correct, the ground plane visible in each camera should satisfy:
    1. The ground normal in vehicle frame should be close to [0, 0, 1] (pointing up)
    2. The ground height should be consistent (= -camera_height)
    3. Overlapping ground regions between adjacent cameras should have consistent depth

  When calibration drifts (e.g. due to vibration, collision), the ground plane
  estimated from depth will deviate from expected, enabling drift detection.

Algorithm:
  1. For each camera, run DA3 metric depth estimation
  2. Back-project depth to 3D points using camera intrinsics
  3. Use RANSAC to fit a plane to the lower portion of the image (assumed ground)
  4. Transform the plane normal to vehicle frame using the current extrinsics
  5. Compare against expected ground normal [0, 0, 1]:
     - Angle deviation → pitch/roll drift
     - Height deviation → height/translation drift
  6. Report per-camera drift and overall health

Usage:
    python scripts/surround_calib_drift_detect.py \
        --config calib_config.json \
        --images front.jpg rear.jpg left.jpg right.jpg \
        --cli ./build/examples/cli/da3-cli \
        --model models/depth-anything-metric-large-f32.gguf

Config JSON format:
    {
        "cameras": [
            {
                "name": "front",
                "intrinsics": [fx, fy, cx, cy],
                "extrinsics": [[r00,r01,r02,tx], [r10,r11,r12,ty], [r20,r21,r22,tz]],
                "image_size": [width, height],
                "ground_roi_y_ratio": [0.6, 1.0]
            },
            ...
        ],
        "vehicle_camera_height": 1.5,
        "drift_threshold_angle_deg": 2.0,
        "drift_threshold_height_m": 0.1
    }
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CameraConfig:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    extrinsics: np.ndarray  # 3x4 camera-to-vehicle transform
    width: int
    height: int
    ground_roi_y_start: float = 0.6
    ground_roi_y_end: float = 1.0


@dataclass
class DriftResult:
    camera_name: str
    normal_angle_deg: float  # angle between estimated and expected ground normal
    height_error_m: float    # difference between estimated and expected ground height
    pitch_drift_deg: float   # estimated pitch drift
    roll_drift_deg: float    # estimated roll drift
    plane_inlier_ratio: float
    is_drifted: bool = False
    message: str = ""


def read_pfm(path: Path) -> np.ndarray:
    """Read PFM file → (H, W) float32."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        channels = 1 if header == "Pf" else 3
        dims = f.readline().decode("ascii").strip().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=f"{endian}f4")
        data = data.reshape(h, w, channels) if channels > 1 else data.reshape(h, w)
        return np.flipud(data).copy()


def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      roi_y_start: int, roi_y_end: int) -> np.ndarray:
    """Back-project depth map to 3D points in camera frame.

    Only processes rows [roi_y_start, roi_y_end) for ground region.
    Returns (N, 3) array of valid 3D points.
    """
    h, w = depth.shape
    y_start = max(0, roi_y_start)
    y_end = min(h, roi_y_end)

    roi_depth = depth[y_start:y_end, :]
    valid_mask = (roi_depth > 0.1) & (roi_depth < 50.0) & np.isfinite(roi_depth)

    v, u = np.where(valid_mask)
    v = v + y_start
    d = roi_depth[valid_mask]

    x3d = (u - cx) * d / fx
    y3d = (v - cy) * d / fy
    z3d = d

    return np.column_stack([x3d, y3d, z3d])


def ransac_plane_fit(points: np.ndarray, n_iterations: int = 1000,
                     distance_threshold: float = 0.05,
                     min_inliers_ratio: float = 0.3) -> Optional[tuple]:
    """RANSAC plane fitting.

    Returns (normal, d, inlier_ratio) where plane: normal . x + d = 0.
    Returns None if fitting fails.
    """
    if len(points) < 10:
        return None

    n_points = len(points)
    best_inliers = 0
    best_normal = None
    best_d = None

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        indices = rng.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        distances = np.abs(points @ normal + d)
        inliers = np.sum(distances < distance_threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if best_normal is None or best_inliers / n_points < min_inliers_ratio:
        return None

    # Refine with all inliers
    distances = np.abs(points @ best_normal + best_d)
    inlier_mask = distances < distance_threshold
    inlier_points = points[inlier_mask]

    if len(inlier_points) >= 3:
        centroid = inlier_points.mean(axis=0)
        centered = inlier_points - centroid
        _, _, vh = np.linalg.svd(centered)
        best_normal = vh[2]
        best_d = -np.dot(best_normal, centroid)

    # Ensure normal points "up" in camera frame (y is down in image coords,
    # so ground normal in camera frame should have negative y component)
    if best_normal[1] > 0:
        best_normal = -best_normal
        best_d = -best_d

    return best_normal, best_d, best_inliers / n_points


def transform_normal_to_vehicle(normal_cam: np.ndarray,
                                extrinsics: np.ndarray) -> np.ndarray:
    """Transform a normal vector from camera frame to vehicle frame.

    extrinsics: 3x4 [R|t] camera-to-vehicle transform.
    """
    R = extrinsics[:3, :3]
    return R @ normal_cam


def compute_drift(normal_vehicle: np.ndarray, ground_height: float,
                  expected_height: float) -> tuple:
    """Compute pitch and roll drift from ground plane normal deviation.

    Expected ground normal in vehicle frame: [0, 0, 1] (z-up).
    Returns (pitch_drift_deg, roll_drift_deg, angle_deg, height_error).
    """
    expected_normal = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(normal_vehicle, expected_normal), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(abs(cos_angle)))

    # Decompose into pitch (rotation around y-axis) and roll (rotation around x-axis)
    # normal_vehicle ≈ [sin(roll), -sin(pitch), cos(pitch)*cos(roll)]
    pitch_drift = np.degrees(np.arcsin(np.clip(-normal_vehicle[1], -1, 1)))
    roll_drift = np.degrees(np.arcsin(np.clip(normal_vehicle[0], -1, 1)))

    height_error = abs(ground_height - (-expected_height))

    return pitch_drift, roll_drift, angle_deg, height_error


def run_da3_metric_depth(cli: str, model: str, metric_model: str,
                         image_path: Path, threads: int) -> Optional[np.ndarray]:
    """Run DA3 CLI to get metric depth map."""
    with tempfile.NamedTemporaryFile(suffix=".pfm", delete=False) as tmp:
        pfm_path = Path(tmp.name)

    try:
        cmd = [cli, "depth", "--model", model, "--input", str(image_path),
               "--pfm", str(pfm_path), "--threads", str(threads)]
        if metric_model:
            cmd.extend(["--metric-model", metric_model])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  DA3 error: {result.stderr.strip()}", file=sys.stderr)
            return None

        return read_pfm(pfm_path)
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return None
    finally:
        pfm_path.unlink(missing_ok=True)


def detect_drift_from_depth(cam: CameraConfig, depth: np.ndarray,
                            expected_height: float, angle_threshold: float,
                            height_threshold: float) -> DriftResult:
    """Detect calibration drift from a pre-computed depth map."""
    print(f"\n[{cam.name}] Analyzing ground plane...")

    scale_x = depth.shape[1] / cam.width
    scale_y = depth.shape[0] / cam.height
    fx = cam.fx * scale_x
    fy = cam.fy * scale_y
    cx = cam.cx * scale_x
    cy = cam.cy * scale_y
    h_depth = depth.shape[0]

    roi_y_start = int(h_depth * cam.ground_roi_y_start)
    roi_y_end = int(h_depth * cam.ground_roi_y_end)

    points_cam = backproject_depth(depth, fx, fy, cx, cy, roi_y_start, roi_y_end)
    print(f"  Ground region points: {len(points_cam)}")

    if len(points_cam) < 100:
        return DriftResult(cam.name, 999, 999, 0, 0, 0, True,
                           f"too few ground points ({len(points_cam)})")

    plane_result = ransac_plane_fit(points_cam, distance_threshold=0.1)
    if plane_result is None:
        return DriftResult(cam.name, 999, 999, 0, 0, 0, True, "RANSAC plane fit failed")

    normal_cam, d, inlier_ratio = plane_result
    print(f"  Plane normal (cam): [{normal_cam[0]:.4f}, {normal_cam[1]:.4f}, {normal_cam[2]:.4f}]")
    print(f"  Inlier ratio: {inlier_ratio:.2%}")

    ground_height_cam = abs(d) / np.linalg.norm(normal_cam)

    normal_vehicle = transform_normal_to_vehicle(normal_cam, cam.extrinsics)
    print(f"  Plane normal (vehicle): [{normal_vehicle[0]:.4f}, {normal_vehicle[1]:.4f}, {normal_vehicle[2]:.4f}]")

    pitch_drift, roll_drift, angle_deg, height_error = compute_drift(
        normal_vehicle, ground_height_cam, expected_height)

    is_drifted = angle_deg > angle_threshold or height_error > height_threshold
    msg = "OK" if not is_drifted else f"DRIFT: angle={angle_deg:.2f}°, height_err={height_error:.3f}m"

    print(f"  Pitch drift: {pitch_drift:.3f}°, Roll drift: {roll_drift:.3f}°")
    print(f"  Ground height: {ground_height_cam:.3f}m (expected: {expected_height:.3f}m)")
    print(f"  Status: {msg}")

    return DriftResult(
        camera_name=cam.name,
        normal_angle_deg=angle_deg,
        height_error_m=height_error,
        pitch_drift_deg=pitch_drift,
        roll_drift_deg=roll_drift,
        plane_inlier_ratio=inlier_ratio,
        is_drifted=is_drifted,
        message=msg
    )


def check_overlap_consistency(cameras: list, depth_maps: dict, config: dict) -> list:
    """Check depth consistency in overlapping regions between adjacent cameras.

    For each pair of adjacent cameras, back-project 3D points from camera A into
    camera B's frame and compare depths. Large discrepancies indicate relative
    extrinsic drift between the two cameras.

    Returns a list of overlap check results.
    """
    adjacency = config.get("camera_adjacency", [])
    if not adjacency:
        # Default: ring topology (front-left, left-rear, rear-right, right-front)
        names = [c.name for c in cameras]
        if len(names) >= 4:
            adjacency = [
                [names[0], names[2]],  # front-left
                [names[2], names[1]],  # left-rear
                [names[1], names[3]],  # rear-right
                [names[3], names[0]],  # right-front
            ]
        elif len(names) >= 2:
            adjacency = [[names[i], names[(i+1) % len(names)]] for i in range(len(names))]

    overlap_results = []
    cam_dict = {c.name: c for c in cameras}

    for pair in adjacency:
        name_a, name_b = pair[0], pair[1]
        if name_a not in cam_dict or name_b not in cam_dict:
            continue
        if name_a not in depth_maps or name_b not in depth_maps:
            continue

        cam_a, cam_b = cam_dict[name_a], cam_dict[name_b]
        depth_a, depth_b = depth_maps[name_a], depth_maps[name_b]

        # Back-project all valid points from camera A to vehicle frame
        h_a, w_a = depth_a.shape
        scale_x_a = w_a / cam_a.width
        scale_y_a = h_a / cam_a.height
        fx_a = cam_a.fx * scale_x_a
        fy_a = cam_a.fy * scale_y_a
        cx_a = cam_a.cx * scale_x_a
        cy_a = cam_a.cy * scale_y_a

        valid_a = (depth_a > 0.1) & (depth_a < 30.0) & np.isfinite(depth_a)
        va, ua = np.where(valid_a)
        da = depth_a[valid_a]

        # 3D in camera A frame
        x_a = (ua - cx_a) * da / fx_a
        y_a = (va - cy_a) * da / fy_a
        z_a = da
        pts_cam_a = np.column_stack([x_a, y_a, z_a])

        # Transform to vehicle frame
        R_a = cam_a.extrinsics[:3, :3]
        t_a = cam_a.extrinsics[:3, 3]
        pts_vehicle = (R_a @ pts_cam_a.T).T + t_a

        # Project into camera B frame
        R_b = cam_b.extrinsics[:3, :3]
        t_b = cam_b.extrinsics[:3, 3]
        R_b_inv = R_b.T
        pts_cam_b = (R_b_inv @ (pts_vehicle - t_b).T).T

        # Only keep points in front of camera B (z > 0)
        in_front = pts_cam_b[:, 2] > 0.1
        pts_cam_b = pts_cam_b[in_front]

        if len(pts_cam_b) == 0:
            overlap_results.append({
                "pair": f"{name_a}-{name_b}",
                "overlap_points": 0,
                "mean_depth_error": float("nan"),
                "median_depth_error": float("nan"),
                "consistency_score": 0.0,
                "status": "no_overlap"
            })
            continue

        # Project to camera B image plane
        h_b, w_b = depth_b.shape
        scale_x_b = w_b / cam_b.width
        scale_y_b = h_b / cam_b.height
        fx_b = cam_b.fx * scale_x_b
        fy_b = cam_b.fy * scale_y_b
        cx_b = cam_b.cx * scale_x_b
        cy_b = cam_b.cy * scale_y_b

        u_b = (pts_cam_b[:, 0] * fx_b / pts_cam_b[:, 2] + cx_b).astype(int)
        v_b = (pts_cam_b[:, 1] * fy_b / pts_cam_b[:, 2] + cy_b).astype(int)
        expected_depth_b = pts_cam_b[:, 2]

        # Filter to valid image coordinates
        valid_proj = (u_b >= 0) & (u_b < w_b) & (v_b >= 0) & (v_b < h_b)
        u_b = u_b[valid_proj]
        v_b = v_b[valid_proj]
        expected_depth_b = expected_depth_b[valid_proj]

        if len(u_b) == 0:
            overlap_results.append({
                "pair": f"{name_a}-{name_b}",
                "overlap_points": 0,
                "mean_depth_error": float("nan"),
                "median_depth_error": float("nan"),
                "consistency_score": 0.0,
                "status": "no_overlap"
            })
            continue

        # Compare with actual depth in camera B
        actual_depth_b = depth_b[v_b, u_b]
        valid_both = (actual_depth_b > 0.1) & np.isfinite(actual_depth_b)
        if valid_both.sum() < 10:
            overlap_results.append({
                "pair": f"{name_a}-{name_b}",
                "overlap_points": int(valid_both.sum()),
                "mean_depth_error": float("nan"),
                "median_depth_error": float("nan"),
                "consistency_score": 0.0,
                "status": "insufficient_overlap"
            })
            continue

        expected_d = expected_depth_b[valid_both]
        actual_d = actual_depth_b[valid_both]
        relative_error = np.abs(expected_d - actual_d) / np.maximum(expected_d, 0.1)

        mean_err = float(relative_error.mean())
        median_err = float(np.median(relative_error))
        # Consistency score: fraction of points with <20% relative depth error
        consistent = (relative_error < 0.2).sum() / len(relative_error)

        overlap_threshold = config.get("overlap_consistency_threshold", 0.5)
        status = "OK" if consistent > overlap_threshold else "INCONSISTENT"

        overlap_results.append({
            "pair": f"{name_a}-{name_b}",
            "overlap_points": int(valid_both.sum()),
            "mean_depth_error": mean_err,
            "median_depth_error": median_err,
            "consistency_score": float(consistent),
            "status": status
        })

    return overlap_results


def create_example_config():
    """Print an example config JSON."""
    config = {
        "cameras": [
            {
                "name": "front",
                "intrinsics": [800.0, 800.0, 640.0, 360.0],
                "extrinsics": [
                    [0, -1, 0, 0],
                    [0, 0, -1, 1.5],
                    [1, 0, 0, 2.0]
                ],
                "image_size": [1280, 720],
                "ground_roi_y_ratio": [0.6, 1.0]
            },
            {
                "name": "rear",
                "intrinsics": [800.0, 800.0, 640.0, 360.0],
                "extrinsics": [
                    [0, 1, 0, 0],
                    [0, 0, -1, 1.5],
                    [-1, 0, 0, -1.0]
                ],
                "image_size": [1280, 720],
                "ground_roi_y_ratio": [0.5, 1.0]
            },
            {
                "name": "left",
                "intrinsics": [600.0, 600.0, 640.0, 360.0],
                "extrinsics": [
                    [1, 0, 0, -1.0],
                    [0, 0, -1, 1.5],
                    [0, 1, 0, 0]
                ],
                "image_size": [1280, 720],
                "ground_roi_y_ratio": [0.5, 0.95]
            },
            {
                "name": "right",
                "intrinsics": [600.0, 600.0, 640.0, 360.0],
                "extrinsics": [
                    [-1, 0, 0, 1.0],
                    [0, 0, -1, 1.5],
                    [0, -1, 0, 0]
                ],
                "image_size": [1280, 720],
                "ground_roi_y_ratio": [0.5, 0.95]
            }
        ],
        "vehicle_camera_height": 1.5,
        "drift_threshold_angle_deg": 2.0,
        "drift_threshold_height_m": 0.15,
        "overlap_consistency_threshold": 0.5,
        "camera_adjacency": [
            ["front", "left"],
            ["left", "rear"],
            ["rear", "right"],
            ["right", "front"]
        ]
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Detect surround-view camera extrinsic calibration drift using DA3 depth")
    parser.add_argument("--config", help="Calibration config JSON file")
    parser.add_argument("--images", nargs="+", help="Camera images (same order as config)")
    parser.add_argument("--cli", default="./build/examples/cli/da3-cli")
    parser.add_argument("--model", default="models/depth-anything-metric-large-f32.gguf")
    parser.add_argument("--metric-model", default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--example-config", action="store_true",
                        help="Print example config JSON and exit")
    parser.add_argument("--output", help="Output results JSON file")
    args = parser.parse_args()

    if args.example_config:
        print(json.dumps(create_example_config(), indent=2))
        return

    if not args.config or not args.images:
        parser.error("--config and --images are required (use --example-config to see format)")

    with open(args.config) as f:
        config = json.load(f)

    cameras = []
    for cam_cfg in config["cameras"]:
        intr = cam_cfg["intrinsics"]
        roi = cam_cfg.get("ground_roi_y_ratio", [0.6, 1.0])
        cameras.append(CameraConfig(
            name=cam_cfg["name"],
            fx=intr[0], fy=intr[1], cx=intr[2], cy=intr[3],
            extrinsics=np.array(cam_cfg["extrinsics"], dtype=np.float64),
            width=cam_cfg["image_size"][0],
            height=cam_cfg["image_size"][1],
            ground_roi_y_start=roi[0],
            ground_roi_y_end=roi[1],
        ))

    expected_height = config.get("vehicle_camera_height", 1.5)
    angle_threshold = config.get("drift_threshold_angle_deg", 2.0)
    height_threshold = config.get("drift_threshold_height_m", 0.15)

    if len(args.images) != len(cameras):
        parser.error(f"Expected {len(cameras)} images, got {len(args.images)}")

    print("=" * 60)
    print("Surround Camera Calibration Drift Detection")
    print(f"  Model: {args.model}")
    print(f"  Expected camera height: {expected_height:.2f} m")
    print(f"  Thresholds: angle={angle_threshold}°, height={height_threshold}m")
    print("=" * 60)

    # First pass: run depth inference for all cameras and cache results
    depth_maps = {}  # name -> depth array
    print("\nPhase 1: Depth inference for all cameras...")
    for cam, img_path_str in zip(cameras, args.images):
        img_path = Path(img_path_str)
        if not img_path.exists():
            continue
        depth = run_da3_metric_depth(args.cli, args.model, args.metric_model,
                                     img_path, args.threads)
        if depth is not None:
            depth_maps[cam.name] = depth
            print(f"  [{cam.name}] depth {depth.shape[1]}x{depth.shape[0]} OK")

    # Second pass: analyze each camera's ground plane
    print("\nPhase 2: Ground plane drift analysis...")
    results = []
    for cam, img_path_str in zip(cameras, args.images):
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"\n[{cam.name}] ERROR: Image not found: {img_path}")
            results.append(DriftResult(cam.name, 999, 999, 0, 0, 0, True, "image not found"))
            continue

        if cam.name not in depth_maps:
            results.append(DriftResult(cam.name, 999, 999, 0, 0, 0, True, "depth failed"))
            continue

        depth = depth_maps[cam.name]
        result = detect_drift_from_depth(
            cam, depth, expected_height, angle_threshold, height_threshold)
        results.append(result)

    # Overlap consistency check between adjacent cameras
    overlap_results = []
    if len(depth_maps) >= 2:
        print("\n" + "-" * 60)
        print("OVERLAP CONSISTENCY CHECK")
        print("-" * 60)
        overlap_results = check_overlap_consistency(cameras, depth_maps, config)
        for ov in overlap_results:
            if ov["status"] == "no_overlap" or ov["status"] == "insufficient_overlap":
                print(f"  {ov['pair']}: {ov['status']} ({ov['overlap_points']} pts)")
            else:
                print(f"  {ov['pair']}: {ov['status']} | "
                      f"overlap={ov['overlap_points']} pts, "
                      f"consistency={ov['consistency_score']:.1%}, "
                      f"median_err={ov['median_depth_error']:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Camera':<10} {'Angle(°)':<10} {'Height(m)':<12} {'Pitch(°)':<10} {'Roll(°)':<10} {'Status'}")
    print("-" * 60)
    any_drift = False
    for r in results:
        status = "DRIFT" if r.is_drifted else "OK"
        if r.is_drifted:
            any_drift = True
        print(f"{r.camera_name:<10} {r.normal_angle_deg:<10.3f} {r.height_error_m:<12.4f} "
              f"{r.pitch_drift_deg:<10.3f} {r.roll_drift_deg:<10.3f} {status}")
    print("-" * 60)
    print(f"Overall: {'CALIBRATION DRIFT DETECTED' if any_drift else 'ALL CAMERAS OK'}")

    if args.output:
        any_overlap_issue = any(
            ov["status"] == "INCONSISTENT" for ov in overlap_results)
        output_data = {
            "overall_status": "drift" if (any_drift or any_overlap_issue) else "ok",
            "cameras": [
                {
                    "name": r.camera_name,
                    "normal_angle_deg": r.normal_angle_deg,
                    "height_error_m": r.height_error_m,
                    "pitch_drift_deg": r.pitch_drift_deg,
                    "roll_drift_deg": r.roll_drift_deg,
                    "inlier_ratio": r.plane_inlier_ratio,
                    "is_drifted": r.is_drifted,
                    "message": r.message,
                }
                for r in results
            ],
            "overlap_consistency": overlap_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
