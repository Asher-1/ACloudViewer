#!/usr/bin/env python3
"""Mesh processing operations example.

Demonstrates mesh manipulation capabilities:
- Creating meshes from point clouds (Delaunay, Poisson via Colmap)
- Sampling points from mesh surface
- Volume computation
- Vertex extraction
- Triangle flipping
- Mesh merging

NOTE: Advanced mesh operations (simplify, smooth, subdivide) require GUI mode.
This example focuses on headless operations.

Works with ACloudViewer binary in headless mode.

Usage:
    python mesh_processing_example.py input.ply output_dir/
"""

import subprocess
import sys
import json
from pathlib import Path


def run_cli(*args, check=True, return_json=False):
    """Run CLI command and return result."""
    cmd = ["cli-anything-acloudviewer", "--json", "--mode", "headless"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    if return_json and result.stdout:
        try:
            return json.loads(result.stdout)
        except:
            pass
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: mesh_processing_example.py <input_cloud.ply> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Mesh Processing Example (Headless Mode)")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Ensure the input has normals (needed for Delaunay)
    print("Step 1: Computing normals for input point cloud...")
    normals_file = output_dir / "00_with_normals.ply"
    result = run_cli("process", "normals", input_file, "-o", str(normals_file),
                     "--radius", "0.05")
    print(f"  ✓ Saved: {normals_file}")
    
    # Step 2: Create mesh using Delaunay triangulation
    print("\nStep 2: Creating mesh with Delaunay triangulation...")
    delaunay_mesh = output_dir / "01_delaunay_mesh.ply"
    result = run_cli("process", "delaunay", str(normals_file), "-o", str(delaunay_mesh),
                     "--max-edge-length", "0.0")
    print(f"  ✓ Saved: {delaunay_mesh}")
    print("  → Fast meshing, good for visualization")
    
    # Step 3: Sample points from mesh surface
    print("\nStep 3: Sampling 100000 points from mesh surface...")
    sampled_cloud = output_dir / "02_sampled_points.ply"
    result = run_cli("process", "sample-mesh", str(delaunay_mesh),
                     "-o", str(sampled_cloud),
                     "--points", "100000")
    print(f"  ✓ Saved: {sampled_cloud}")
    print("  → Uniform surface sampling")
    print("  → Converted mesh back to point cloud")
    
    # Step 4: Extract vertices as point cloud
    print("\nStep 4: Extracting mesh vertices...")
    vertices_cloud = output_dir / "03_vertices.ply"
    result = run_cli("process", "extract-vertices", str(delaunay_mesh),
                     "-o", str(vertices_cloud))
    print(f"  ✓ Saved: {vertices_cloud}")
    print("  → Extracted vertex positions only")
    print("  → No interpolation, just mesh vertices")
    
    # Step 5: Compute mesh volume
    print("\nStep 5: Computing mesh volume...")
    result = run_cli("process", "mesh-volume", str(delaunay_mesh), return_json=True)
    if isinstance(result, dict):
        volume = result.get("volume", "N/A")
        print(f"  ✓ Mesh volume: {volume}")
        print("  → Useful for physical property analysis")
    else:
        print(f"  ✓ Volume computation complete")
    
    # Step 6: Flip triangle winding order (if needed)
    print("\nStep 6: Flipping triangle winding order...")
    flipped_mesh = output_dir / "04_flipped.ply"
    result = run_cli("process", "flip-triangles", str(delaunay_mesh),
                     "-o", str(flipped_mesh))
    print(f"  ✓ Saved: {flipped_mesh}")
    print("  → Reversed all triangle orientations")
    print("  → Useful for fixing inside-out meshes")
    
    # Step 7: Create a second mesh for merging demo
    print("\nStep 7: Creating a second mesh for demonstration...")
    # Subsample to create a different point cloud
    subset_cloud = output_dir / "subset_cloud.ply"
    result = run_cli("process", "subsample", input_file,
                     "-o", str(subset_cloud),
                     "--voxel-size", "0.1")
    
    subset_normals = output_dir / "subset_normals.ply"
    result = run_cli("process", "normals", str(subset_cloud),
                     "-o", str(subset_normals),
                     "--radius", "0.1")
    
    second_mesh = output_dir / "second_mesh.ply"
    result = run_cli("process", "delaunay", str(subset_normals),
                     "-o", str(second_mesh))
    print(f"  ✓ Created second mesh: {second_mesh}")
    
    # Step 8: Merge multiple meshes
    print("\nStep 8: Merging multiple meshes...")
    merged_mesh = output_dir / "05_merged_meshes.ply"
    result = run_cli("process", "merge-meshes", 
                     str(delaunay_mesh), str(second_mesh),
                     "-o", str(merged_mesh))
    print(f"  ✓ Saved: {merged_mesh}")
    print("  → Combined multiple meshes into one")
    
    # Summary
    print("\n" + "=" * 60)
    print("Mesh Processing Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {delaunay_mesh.name} - Delaunay mesh from point cloud")
    print(f"  2. {sampled_cloud.name} - Sampled points (100K)")
    print(f"  3. {vertices_cloud.name} - Mesh vertices only")
    print(f"  4. {flipped_mesh.name} - Flipped triangles")
    print(f"  5. {merged_mesh.name} - Merged meshes")
    
    print("\nHeadless mesh operations available:")
    print("  • delaunay: Point cloud → mesh triangulation")
    print("  • sample-mesh: Mesh → point cloud (uniform sampling)")
    print("  • extract-vertices: Extract mesh vertex positions")
    print("  • mesh-volume: Compute enclosed volume")
    print("  • flip-triangles: Reverse triangle winding")
    print("  • merge-meshes: Combine multiple meshes")
    
    print("\nFor advanced mesh operations (simplify, smooth, subdivide):")
    print("  These require GUI mode with entity_id:")
    print("    1. Start ACloudViewer GUI")
    print("    2. Enable JSON-RPC plugin")
    print("    3. Load mesh to get entity_id")
    print("    4. Run: cli-anything-acloudviewer mesh simplify <entity_id> --target-triangles 10000")
    print("    5. Run: cli-anything-acloudviewer mesh smooth <entity_id> --iterations 5")
    print("    6. Run: cli-anything-acloudviewer mesh subdivide <entity_id> --method loop")
    
    print("\nReconstruction options:")
    print("  • For Poisson reconstruction: use reconstruct commands")
    print("    cli-anything-acloudviewer reconstruct poisson cloud.ply -o poisson.ply")
    print("  • For Colmap 3D reconstruction from images:")
    print("    cli-anything-acloudviewer reconstruct auto images/ -w workspace/ --quality high")


if __name__ == "__main__":
    main()
