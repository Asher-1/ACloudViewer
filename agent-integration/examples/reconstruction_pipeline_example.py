#!/usr/bin/env python3
"""Complete 3D reconstruction pipeline example.

Demonstrates the full Colmap reconstruction workflow:
- Automatic reconstruction (simple)
- Step-by-step reconstruction (advanced)
- Different quality levels
- Various camera models
- Dense reconstruction and meshing

Works with ACloudViewer binary + Colmap integration in headless mode.

Usage:
    python reconstruction_pipeline_example.py images_dir/ workspace_dir/
"""

import subprocess
import sys
import json
from pathlib import Path


def run_cli(*args, check=True, timeout=600):
    """Run CLI command and return result."""
    cmd = ["cli-anything-acloudviewer", "--json", "--mode", "headless"] + list(args)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"Error (exit {result.returncode}): {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result


def print_step(number, title):
    """Print a formatted step header."""
    print(f"\n{'='*60}")
    print(f"Step {number}: {title}")
    print('='*60)


def main():
    if len(sys.argv) < 3:
        print("Usage: reconstruction_pipeline_example.py <images_directory> <workspace_directory>")
        print("\nExample:")
        print("  python reconstruction_pipeline_example.py ./photos/ ./reconstruction/")
        sys.exit(1)
    
    images_dir = Path(sys.argv[1])
    workspace_dir = Path(sys.argv[2])
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("3D Reconstruction Pipeline Example")
    print("=" * 60)
    print(f"Images: {images_dir}")
    print(f"Workspace: {workspace_dir}\n")
    
    # Option 1: Automatic reconstruction (recommended for most users)
    print("\n" + "=" * 60)
    print("OPTION 1: Automatic Reconstruction (One Command)")
    print("=" * 60)
    print("\nThis is the easiest way - runs the entire pipeline automatically.\n")
    
    auto_workspace = workspace_dir / "automatic"
    print(f"Workspace: {auto_workspace}\n")
    
    print("Example 1a: High quality reconstruction with default camera")
    print("Command:")
    print(f"  cli-anything-acloudviewer reconstruct auto {images_dir} -w {auto_workspace}_high --quality high")
    print()
    
    print("Example 1b: Medium quality with OPENCV camera model")
    print("Command:")
    print(f"  cli-anything-acloudviewer reconstruct auto {images_dir} -w {auto_workspace}_opencv --quality medium --camera-model OPENCV")
    print()
    
    print("Example 1c: Fast sparse-only reconstruction (no dense)")
    print("Command:")
    print(f"  cli-anything-acloudviewer reconstruct auto {images_dir} -w {auto_workspace}_sparse --quality low --no-dense")
    print()
    
    # Option 2: Step-by-step reconstruction (advanced users)
    print("\n" + "=" * 60)
    print("OPTION 2: Step-by-Step Reconstruction (Advanced)")
    print("=" * 60)
    print("\nThis gives you full control over each stage.\n")
    
    step_workspace = workspace_dir / "step_by_step"
    step_workspace.mkdir(parents=True, exist_ok=True)
    
    print_step(1, "Extract Features")
    print("Extract SIFT features from all images...")
    db_file = step_workspace / "database.db"
    result = run_cli("reconstruct", "extract-features", str(images_dir),
                     "-d", str(db_file))
    print(f"✓ Features extracted to: {db_file}")
    
    print_step(2, "Match Features")
    print("Match features between image pairs (exhaustive matching)...")
    result = run_cli("reconstruct", "match", str(db_file), "--method", "exhaustive")
    print("✓ Feature matching complete")
    
    print_step(3, "Sparse Reconstruction (SfM)")
    print("Build sparse 3D model using Structure from Motion...")
    sparse_dir = step_workspace / "sparse"
    result = run_cli("reconstruct", "sparse", 
                     "-d", str(db_file),
                     "--image-path", str(images_dir),
                     "-o", str(sparse_dir))
    print(f"✓ Sparse model saved to: {sparse_dir}")
    
    print_step(4, "Undistort Images")
    print("Undistort images using camera calibration...")
    dense_dir = step_workspace / "dense"
    result = run_cli("reconstruct", "undistort",
                     "--image-path", str(images_dir),
                     "-i", str(sparse_dir / "0"),
                     "-o", str(dense_dir))
    print(f"✓ Undistorted images saved to: {dense_dir}")
    
    print_step(5, "Dense Stereo Reconstruction")
    print("Compute depth and normal maps for all views...")
    result = run_cli("reconstruct", "dense-stereo", str(dense_dir), timeout=1200)
    print("✓ Dense stereo maps computed")
    
    print_step(6, "Stereo Fusion")
    print("Fuse depth maps into dense point cloud...")
    fused_ply = dense_dir / "fused.ply"
    result = run_cli("reconstruct", "fuse", str(dense_dir), "-o", str(fused_ply))
    print(f"✓ Dense point cloud: {fused_ply}")
    
    print_step(7, "Poisson Surface Reconstruction")
    print("Reconstruct smooth mesh surface...")
    poisson_mesh = step_workspace / "poisson_mesh.ply"
    result = run_cli("reconstruct", "poisson", str(fused_ply), "-o", str(poisson_mesh))
    print(f"✓ Poisson mesh: {poisson_mesh}")
    
    print_step(8, "Delaunay Meshing (Alternative)")
    print("Create mesh using Delaunay triangulation...")
    delaunay_mesh = step_workspace / "delaunay_mesh.ply"
    result = run_cli("reconstruct", "delaunay-mesh", str(fused_ply), 
                     "-o", str(delaunay_mesh))
    print(f"✓ Delaunay mesh: {delaunay_mesh}")
    
    print_step(9, "Texture Mapping")
    print("Apply image textures to the mesh...")
    textured_dir = step_workspace / "textured"
    result = run_cli("reconstruct", "texture-mesh", str(dense_dir),
                     "-o", str(textured_dir),
                     "--mesh", str(delaunay_mesh), timeout=1200)
    print(f"✓ Textured mesh: {textured_dir}")
    
    print_step(10, "Model Analysis")
    print("Analyze the reconstruction model...")
    result = run_cli("reconstruct", "analyze-model", str(sparse_dir / "0"))
    print("✓ Model analysis complete")
    
    print_step(11, "Model Conversion")
    print("Convert Colmap model to PLY format...")
    model_ply = step_workspace / "sparse_model.ply"
    result = run_cli("reconstruct", "convert-model", str(sparse_dir / "0"),
                     "-o", str(model_ply), "--output-type", "PLY")
    print(f"✓ Converted model: {model_ply}")
    
    # Summary
    print("\n" + "=" * 60)
    print("3D Reconstruction Complete!")
    print("=" * 60)
    print("\nKey outputs:")
    print(f"  • Sparse model: {sparse_dir / '0'}")
    print(f"  • Dense point cloud: {fused_ply}")
    print(f"  • Poisson mesh: {poisson_mesh}")
    print(f"  • Delaunay mesh: {delaunay_mesh}")
    print(f"  • Textured model: {textured_dir}")
    print(f"  • PLY export: {model_ply}")
    
    print("\nSupported camera models:")
    print("  SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL (default), RADIAL,")
    print("  OPENCV, OPENCV_FISHEYE, FULL_OPENCV, SIMPLE_RADIAL_FISHEYE,")
    print("  RADIAL_FISHEYE, THIN_PRISM_FISHEYE")
    
    print("\nQuality levels: low, medium, high, extreme")
    print("\nFor SIBR viewers, run:")
    print(f"  cli-anything-acloudviewer sibr prepare-colmap {step_workspace}")
    print(f"  cli-anything-acloudviewer sibr viewer gaussian --model-path ./output/ --path {step_workspace}")


if __name__ == "__main__":
    main()
