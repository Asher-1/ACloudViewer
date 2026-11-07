#!/usr/bin/env python3
"""Compare the C++ camera projection with the named upstream camera.py contract."""
import argparse
import ast
import importlib.util
import sys
import hashlib
import json
import subprocess
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np


POINTS = np.array([[0, 0, 5], [0, 0, 10], [1, 0, 5], [-1, 0, 5],
                   [0, 1, 5], [100, 100, 1], [0, 0, -1],
                   [np.nan, 0, 5]], dtype=np.float64)


def upstream_functions(root):
    """Execute actual numeric source functions without loading GUI packages."""
    folder = root / "anylabeling/views/labeling/pointcloud"
    spec = importlib.util.spec_from_file_location("qany_reference_cuboid", folder / "cuboid.py")
    cuboid = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cuboid
    spec.loader.exec_module(cuboid)
    path = folder / "camera.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"load_calibration", "_project_camera_points", "project_points",
             "_clip_image_segments", "project_cuboid"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = dict(np=np, json=json, Path=Path, EDGES=cuboid.EDGES)
    panel = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CameraPanel")
    nodes.append(next(node for node in panel.body if isinstance(node, ast.FunctionDef) and node.name == "_update_depth_colors"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, cuboid.Cuboid


def calibration(model):
    return {
        "schema_version": 1,
        "image_size": [100, 80],
        "camera_model": "pinhole",
        "camera_matrix": [[50, 0, 50], [0, 50, 40], [0, 0, 1]],
        "T_pointcloud_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0],
                                    [0, 0, 1, 0], [0, 0, 0, 1]],
        "distortion_model": model,
        "distortion_coefficients": ([0.04, -0.01, 0.001, -0.002, 0.0005]
                                     if model == "opencv5" else []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.upstream / "assets/pointcloud/detection/KITTI_3D_det/labels/000008.cuboids.json"
    real_boxes = json.loads(sidecar.read_text())["cuboids"]
    reference, Cuboid = upstream_functions(args.upstream)
    maximum = 0.0
    rows = []
    with tempfile.TemporaryDirectory(prefix="qany-camera-") as temporary:
        temporary = Path(temporary)
        synthetic_boxes = [dict(id=i+1, class_id=1, center=center, size=size,
                                rotation=rotation, occluded=False, locked=False)
                           for i, (center, size, rotation) in enumerate([
                               ([0, 0, 5], [2, 2, 2], [0, 0, 0]),
                               ([4, 0, 5], [4, 3, 3], [.1, -.2, .7]),
                               ([0, 0, .5], [2, 2, 2], [.1, .2, .3]),
                               ([0, 0, -5], [2, 2, 2], [0, 0, 0]),
                               ([0, 4, 5], [2, 4, 2], [.3, .1, -.5]),
                           ])]
        synthetic_path = temporary / "synthetic.cuboids.json"
        synthetic_path.write_text(json.dumps(dict(version=1, type="pointcloud-cuboids",
                                                  point_cloud="synthetic.bin", cuboids=synthetic_boxes)))
        cases = [(f"clipping-{model}", calibration(model), synthetic_path, synthetic_boxes)
                 for model in ("none", "opencv5")]
        kitti_config = json.loads((sidecar.parent.parent / "calibration.json").read_text())
        cases.append(("kitti-fixture", kitti_config, sidecar, real_boxes))
        for name, config, input_sidecar, boxes in cases:
            model = config["distortion_model"]
            calibration_path = temporary / f"{name}.json"
            calibration_path.write_text(json.dumps(config))
            output = subprocess.check_output([
                str(args.probe.resolve()), "--probe", str(calibration_path), str(input_sidecar)
            ], text=True)
            actual = json.loads(output)
            upstream_calibration = reference["load_calibration"](calibration_path)
            expected_indices, expected_pixels = reference["project_points"](
                POINTS, upstream_calibration, *config["image_size"])
            got_indices = np.asarray(actual["indices"], dtype=np.int64)
            got_pixels = np.asarray(actual["pixels"], dtype=np.int64).reshape(-1, 2)
            np.testing.assert_array_equal(got_indices, expected_indices)
            np.testing.assert_array_equal(got_pixels, expected_pixels)
            overlay = SimpleNamespace(calibration=upstream_calibration, _indices=expected_indices,
                                      tr=lambda text: text,
                                      depth_legend=SimpleNamespace(setText=lambda text: None,
                                                                  setToolTip=lambda text: None))
            reference["_update_depth_colors"](overlay, POINTS)
            got_colors = np.asarray(actual["depth_colors"], dtype=np.uint8).reshape(-1, 3)
            np.testing.assert_array_equal(got_colors, overlay._depth_colors)
            transform = upstream_calibration["T_pointcloud_to_camera"]
            depth = POINTS[expected_indices, :3] @ transform[2, :3] + transform[2, 3]
            depth_range = np.percentile(depth, [2, 98]) if len(depth) else [0, 0]
            np.testing.assert_allclose(actual["depth_range"], depth_range, atol=1e-12, rtol=0)
            segment_count = 0
            for index, box in enumerate(boxes):
                expected = reference["project_cuboid"](
                    Cuboid(**box), upstream_calibration, *config["image_size"])
                got = np.asarray(actual["cuboids"][index]["segments"], dtype=np.float64).reshape(-1, 2, 2)
                np.testing.assert_equal(got.shape, expected.shape)
                delta = float(np.max(np.abs(got - expected))) if len(expected) else 0.0
                maximum = max(maximum, delta)
                np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0)
                segment_count += len(expected)
            assert segment_count > 0, name
            rows.append({"case": name, "distortion_model": model, "cuboids": len(boxes),
                         "segments": segment_count, "max_delta": maximum})
    camera_source = args.upstream / "anylabeling/views/labeling/pointcloud/camera.py"
    report = {
        "verdict": "PASS",
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "reference_execution": "actual upstream camera.py projection/depth-color functions and cuboid.py",
        "depth_color_rgb_max_delta": 0,
        "camera_source_sha256": hashlib.sha256(camera_source.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "comparison_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuboid_source_sha256": hashlib.sha256((camera_source.parent / "cuboid.py").read_bytes()).hexdigest(),
        "rows": rows,
        "point_cases": len(POINTS),
        "coordinate_tolerance": 1e-9,
        "maximum_delta": maximum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
