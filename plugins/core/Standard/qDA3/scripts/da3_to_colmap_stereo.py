#!/usr/bin/env python3
"""
da3_to_colmap_stereo.py — Replace COLMAP's PatchMatchStereo with DA3 metric depth.

Given a COLMAP dense workspace (output of `colmap image_undistorter`), this script:
  1. Reads the undistorted images
  2. Runs DA3 (via depth-anything.cpp CLI) to produce metric depth maps
  3. Writes depth maps in COLMAP's .geometric.bin format
  4. Generates a minimal normal map (from depth gradients)
  5. Writes patch-match.cfg and fusion.cfg so `colmap stereo_fusion` can proceed

After running this script, you can skip `colmap patch_match_stereo` and go directly to:
    colmap stereo_fusion --workspace_path <dense_ws> --workspace_format COLMAP \
        --input_type geometric --output_path <dense_ws>/fused.ply

Usage:
    python scripts/da3_to_colmap_stereo.py \
        --workspace /path/to/dense \
        --cli ./build/examples/cli/da3-cli \
        --model models/depth-anything-metric-large-f32.gguf \
        [--threads 8] [--min-depth 0.1] [--max-depth 100]

    For nested metric model (best quality):
    python scripts/da3_to_colmap_stereo.py \
        --workspace /path/to/dense \
        --cli ./build/examples/cli/da3-cli \
        --model models/depth-anything-nested-anyview.gguf \
        --metric-model models/depth-anything-nested-metric.gguf
"""

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def write_colmap_depth_map(path: Path, depth: np.ndarray):
    """Write a depth map in COLMAP's mixed text+binary format.

    Format: ASCII header 'width&height&channels&' followed by row-major float32.
    """
    h, w = depth.shape
    header = f"{w}&{h}&1&"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(depth.astype(np.float32).tobytes())


def write_colmap_normal_map(path: Path, normals: np.ndarray):
    """Write a normal map in COLMAP's format (channels=3)."""
    h, w, c = normals.shape
    assert c == 3
    header = f"{w}&{h}&3&"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(normals.astype(np.float32).tobytes())


def depth_to_normals(depth: np.ndarray) -> np.ndarray:
    """Estimate surface normals from depth map using finite differences.

    Returns (H, W, 3) normal map in camera frame (x-right, y-down, z-forward).
    """
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    dz_dx = np.zeros_like(depth)
    dz_dy = np.zeros_like(depth)
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2.0
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2.0

    normals[:, :, 0] = -dz_dx
    normals[:, :, 1] = -dz_dy
    normals[:, :, 2] = 1.0

    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    normals /= norm
    return normals


def read_pfm(path: Path) -> np.ndarray:
    """Read a PFM (Portable FloatMap) file, return (H, W) float32 array."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header == "Pf":
            channels = 1
        elif header == "PF":
            channels = 3
        else:
            raise ValueError(f"Not a PFM file: {path}")
        dims = f.readline().decode("ascii").strip().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=f"{endian}f4")
        data = data.reshape(h, w, channels) if channels > 1 else data.reshape(h, w)
        data = np.flipud(data).copy()
    return data


def run_da3_depth(cli: str, model: str, metric_model: str, image_path: Path,
                  pfm_path: Path, threads: int) -> bool:
    """Run DA3 CLI to produce a depth PFM file."""
    cmd = [cli, "depth", "--model", model, "--input", str(image_path),
           "--pfm", str(pfm_path), "--threads", str(threads)]
    if metric_model:
        cmd.extend(["--metric-model", metric_model])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return False
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Replace COLMAP PatchMatchStereo with DA3 metric depth")
    parser.add_argument("--workspace", required=True,
                        help="COLMAP dense workspace (output of image_undistorter)")
    parser.add_argument("--cli", default="./build/examples/cli/da3-cli",
                        help="Path to da3-cli binary")
    parser.add_argument("--model", required=True,
                        help="Path to DA3 metric GGUF model")
    parser.add_argument("--metric-model", default="",
                        help="Path to nested metric branch GGUF (for nested mode)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for DA3 inference")
    parser.add_argument("--min-depth", type=float, default=0.01,
                        help="Minimum valid depth (meters)")
    parser.add_argument("--max-depth", type=float, default=100.0,
                        help="Maximum valid depth (meters)")
    args = parser.parse_args()

    ws = Path(args.workspace)
    images_dir = ws / "images"
    stereo_dir = ws / "stereo"
    depth_dir = stereo_dir / "depth_maps"
    normal_dir = stereo_dir / "normal_maps"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    )

    if not image_files:
        print(f"ERROR: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_files)} images...")
    print(f"  CLI: {args.cli}")
    print(f"  Model: {args.model}")
    if args.metric_model:
        print(f"  Metric model: {args.metric_model}")
    print(f"  Threads: {args.threads}")
    print()

    success_count = 0
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")

        with tempfile.NamedTemporaryFile(suffix=".pfm", delete=True) as tmp:
            pfm_path = Path(tmp.name)

        if not run_da3_depth(args.cli, args.model, args.metric_model,
                            img_path, pfm_path, args.threads):
            print(f"  SKIP: depth inference failed")
            continue

        try:
            depth = read_pfm(pfm_path)
        except Exception as e:
            print(f"  SKIP: cannot read PFM: {e}")
            continue
        finally:
            pfm_path.unlink(missing_ok=True)

        mask_invalid = (depth <= args.min_depth) | (depth > args.max_depth) | ~np.isfinite(depth)
        depth[mask_invalid] = 0.0

        normals = depth_to_normals(depth)
        normals[mask_invalid] = 0.0

        depth_out = depth_dir / f"{img_path.name}.geometric.bin"
        normal_out = normal_dir / f"{img_path.name}.geometric.bin"
        write_colmap_depth_map(depth_out, depth)
        write_colmap_normal_map(normal_out, normals)

        valid_pct = 100.0 * np.count_nonzero(depth) / depth.size
        print(f"  depth: {depth.shape[1]}x{depth.shape[0]}, "
              f"range [{depth[depth>0].min():.3f}, {depth[depth>0].max():.3f}] m, "
              f"valid: {valid_pct:.1f}%")
        success_count += 1

    cfg_images = "\n".join(p.name for p in image_files)
    patch_match_cfg = stereo_dir / "patch-match.cfg"
    fusion_cfg = stereo_dir / "fusion.cfg"
    patch_match_cfg.write_text(cfg_images + "\n")
    fusion_cfg.write_text(cfg_images + "\n")

    print(f"\nDone: {success_count}/{len(image_files)} depth maps generated.")
    print(f"Output: {depth_dir}")
    print(f"\nNext step — run stereo fusion (skip patch_match_stereo):")
    print(f"  colmap stereo_fusion \\")
    print(f"    --workspace_path {ws} \\")
    print(f"    --workspace_format COLMAP \\")
    print(f"    --input_type geometric \\")
    print(f"    --output_path {ws}/fused.ply")


if __name__ == "__main__":
    main()
