#!/usr/bin/env python3
"""Headless batch processing example using the cloudViewer Python API.

Demonstrates: load -> subsample -> compute normals -> ICP -> export

Usage:
    python batch_process.py source.ply target.ply output_dir/
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print("Usage: batch_process.py <source.ply> <target.ply> <output_dir/>")
        sys.exit(1)

    source_path, target_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        import cloudViewer as cv
    except ImportError:
        print("cloudViewer not installed. Install with: pip install cloudViewer")
        sys.exit(1)

    print(f"cloudViewer version: {cv.__version__}")

    # Load point clouds
    print(f"Loading source: {source_path}")
    source = cv.io.read_point_cloud(source_path)
    print(f"  Points: {len(source.points)}")

    print(f"Loading target: {target_path}")
    target = cv.io.read_point_cloud(target_path)
    print(f"  Points: {len(target.points)}")

    # Subsample
    voxel_size = 0.05
    print(f"\nSubsampling (voxel_size={voxel_size})...")
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    print(f"  Source: {len(source.points)} -> {len(source_down.points)}")
    print(f"  Target: {len(target.points)} -> {len(target_down.points)}")

    # Compute normals
    print("\nComputing normals...")
    radius = voxel_size * 2
    source_down.estimate_normals(
        cv.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target_down.estimate_normals(
        cv.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # ICP registration
    threshold = voxel_size * 1.5
    print(f"\nRunning ICP (threshold={threshold})...")
    result = cv.pipelines.registration.registration_icp(
        source_down, target_down, threshold,
        cv.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"  Fitness: {result.fitness:.4f}")
    print(f"  RMSE: {result.inlier_rmse:.6f}")
    print(f"  Transform:\n{result.transformation}")

    # Apply transform and export
    source.transform(result.transformation)
    out_path = str(Path(output_dir) / "aligned_source.ply")
    cv.io.write_point_cloud(out_path, source)
    print(f"\nSaved aligned source to: {out_path}")

    # Save subsampled versions
    cv.io.write_point_cloud(
        str(Path(output_dir) / "source_subsampled.ply"), source_down)
    cv.io.write_point_cloud(
        str(Path(output_dir) / "target_subsampled.ply"), target_down)
    print("Done!")


if __name__ == "__main__":
    main()
