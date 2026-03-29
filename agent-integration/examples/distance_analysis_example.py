#!/usr/bin/env python3
"""Distance computation and analysis example.

Demonstrates various distance computation methods:
- Cloud-to-cloud (C2C) distance
- Cloud-to-mesh (C2M) distance
- Distance-based filtering
- Statistical distance analysis
- Closest point set identification

Works with ACloudViewer binary in headless mode.

Usage:
    python distance_analysis_example.py compared.ply reference.ply output_dir/
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
    if len(sys.argv) < 4:
        print("Usage: distance_analysis_example.py <compared.ply> <reference.ply> <output_directory>")
        print("\nComputes distances from 'compared' cloud to 'reference' cloud/mesh.")
        sys.exit(1)
    
    compared_file = sys.argv[1]
    reference_file = sys.argv[2]
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Distance Computation and Analysis Example")
    print("=" * 60)
    print(f"Compared:  {compared_file}")
    print(f"Reference: {reference_file}")
    print(f"Output:    {output_dir}\n")
    
    # Step 1: Cloud-to-cloud distance (basic)
    print("Step 1: Computing cloud-to-cloud distance...")
    c2c_basic = output_dir / "01_c2c_distance.ply"
    result = run_cli("process", "c2c-dist", compared_file, reference_file,
                     "-o", str(c2c_basic), return_json=True)
    print(f"  ✓ Saved: {c2c_basic}")
    if isinstance(result, dict):
        mean_dist = result.get("mean_distance", "N/A")
        max_dist = result.get("max_distance", "N/A")
        print(f"  → Mean distance: {mean_dist}")
        print(f"  → Max distance:  {max_dist}")
    print("  → Points colored by distance to reference\n")
    
    # Step 2: Cloud-to-cloud with max distance threshold
    print("Step 2: Computing C2C with max distance threshold (1.0)...")
    c2c_thresh = output_dir / "02_c2c_max_distance.ply"
    result = run_cli("process", "c2c-dist", compared_file, reference_file,
                     "-o", str(c2c_thresh),
                     "--max-dist", "1.0", return_json=True)
    print(f"  ✓ Saved: {c2c_thresh}")
    print("  → Distances beyond 1.0 are capped")
    print("  → Useful for highlighting close vs far regions\n")
    
    # Step 3: Create a mesh from reference for C2M
    print("Step 3: Creating mesh from reference cloud...")
    ref_with_normals = output_dir / "reference_normals.ply"
    result = run_cli("process", "normals", reference_file,
                     "-o", str(ref_with_normals),
                     "--radius", "0.05")
    print(f"  ✓ Computed normals: {ref_with_normals}")
    
    ref_mesh = output_dir / "reference_mesh.ply"
    result = run_cli("process", "delaunay", str(ref_with_normals),
                     "-o", str(ref_mesh))
    print(f"  ✓ Created mesh: {ref_mesh}")
    print("  → Reference mesh for C2M computation\n")
    
    # Step 4: Cloud-to-mesh distance
    print("Step 4: Computing cloud-to-mesh distance...")
    c2m_dist = output_dir / "04_c2m_distance.ply"
    result = run_cli("process", "c2m-dist", compared_file, str(ref_mesh),
                     "-o", str(c2m_dist), return_json=True)
    print(f"  ✓ Saved: {c2m_dist}")
    if isinstance(result, dict):
        mean_dist = result.get("mean_distance", "N/A")
        print(f"  → Mean distance to surface: {mean_dist}")
    print("  → Distance to continuous surface (more accurate)\n")
    
    # Step 5: Find closest point pairs
    print("Step 5: Finding closest points in reference for each compared point...")
    closest_pts = output_dir / "05_closest_points.ply"
    result = run_cli("process", "closest-point-set", compared_file, reference_file,
                     "-o", str(closest_pts))
    print(f"  ✓ Saved: {closest_pts}")
    print("  → Each point shows its nearest neighbor in reference\n")
    
    # Step 6: Filter by distance for outlier detection
    print("Step 6: Converting distance SF to RGB colors...")
    # Convert the distance scalar field to RGB colors for visualization
    colored_dist = output_dir / "06_colored_distance.ply"
    result = run_cli("sf", "convert-to-rgb", str(c2c_basic),
                     "-o", str(colored_dist))
    print(f"  ✓ Saved: {colored_dist}")
    print("  → Distance SF converted to RGB")
    print("  → Color gradient shows distance variation")
    print("  → Good for visual quality assessment\n")
    
    # Step 7: Statistical analysis between two clouds
    print("Step 7: Running statistical comparison test...")
    stat_result = run_cli("process", "stat-test", compared_file, reference_file,
                          "--test-type", "CHI2", return_json=True)
    print("  ✓ Statistical test completed")
    if isinstance(stat_result, dict):
        print(f"  → Test type: {stat_result.get('test_type', 'N/A')}")
        print(f"  → Result: {stat_result.get('result', 'N/A')}")
    print("  → Chi-squared test for distribution similarity\n")
    
    # Step 8: Create visual comparison using Python API for coloring
    print("Step 8: Creating visual comparison (colored overlay)...")
    print("  Note: Uniform color painting not available in headless mode")
    print("  → Using cloudViewer Python API for coloring...")
    
    try:
        import cloudViewer as cv
        import numpy as np
        
        # Load and paint compared cloud green
        compared_cloud = cv.io.read_point_cloud(compared_file)
        compared_cloud.paint_uniform_color([0, 1, 0])  # Green
        compared_green = output_dir / "compared_green.ply"
        cv.io.write_point_cloud(str(compared_green), compared_cloud)
        
        # Load and paint reference cloud red
        reference_cloud = cv.io.read_point_cloud(reference_file)
        reference_cloud.paint_uniform_color([1, 0, 0])  # Red
        reference_red = output_dir / "reference_red.ply"
        cv.io.write_point_cloud(str(reference_red), reference_cloud)
        
        # Merge both
        visual_compare = output_dir / "08_visual_comparison.ply"
        result = run_cli("process", "merge-clouds", str(compared_green), str(reference_red),
                         "-o", str(visual_compare))
        print(f"  ✓ Saved: {visual_compare}")
        print("  → Compared = Green, Reference = Red")
        print("  → Overlap appears yellow\n")
    except ImportError:
        print("  ! cloudViewer Python module not available")
        print("  ! Skipping color-coded comparison")
        print("  ! Install with: pip install cloudViewer\n")
    except Exception as e:
        print(f"  ! Error during coloring: {e}\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("Distance Analysis Complete!")
    print("=" * 60)
    print("\nKey outputs:")
    print(f"  • Basic C2C:       {c2c_basic}")
    print(f"  • Thresholded C2C: {c2c_thresh}")
    print(f"  • C2M distance:    {c2m_dist}")
    print(f"  • Closest points:  {closest_pts}")
    print(f"  • Colored by dist: {colored_dist}")
    print(f"  • Visual compare:  {visual_compare}")
    
    print("\nDistance computation use cases:")
    print("  • Quality assessment after ICP registration")
    print("  • Change detection between scan epochs")
    print("  • Outlier identification")
    print("  • Surface deviation analysis (CAD vs scan)")
    print("  • Mesh reconstruction quality validation")
    
    print("\nInterpretation:")
    print("  • Mean distance < 0.01: Excellent alignment")
    print("  • Mean distance < 0.1:  Good alignment")
    print("  • Mean distance < 1.0:  Moderate alignment")
    print("  • Mean distance > 1.0:  Poor alignment, re-register")
    
    print("\nScalar field tip:")
    print("  The 'C2C absolute distances' scalar field is created automatically.")
    print("  Use it for further filtering or statistical analysis:")
    print(f"    cli-anything-acloudviewer sf filter {c2c_basic} -o near.ply \\")
    print("      --sf 'C2C absolute distances' --min 0.0 --max 0.5")


if __name__ == "__main__":
    main()
