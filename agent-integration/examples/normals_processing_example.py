#!/usr/bin/env python3
"""Normal vector processing example.

Demonstrates comprehensive normal vector operations:
- Computing normals with different methods
- Orienting normals consistently (MST algorithm)
- Inverting normal directions
- Converting normals to dip/dip direction
- Exporting normals as scalar fields
- Removing normals

Works with ACloudViewer binary in headless mode.

Usage:
    python normals_processing_example.py input.ply output_dir/
"""

import subprocess
import sys
from pathlib import Path


def run_cli(*args, check=True):
    """Run CLI command and return result."""
    cmd = ["cli-anything-acloudviewer", "--json", "--mode", "headless"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: normals_processing_example.py <input.ply> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Normal Vector Processing Example")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Compute normals using octree method
    print("Step 1: Computing normals with octree method...")
    octree_normals = output_dir / "01_octree_normals.ply"
    result = run_cli("normals", "octree", input_file, "-o", str(octree_normals),
                     "--radius", "0.1")
    print(f"  ✓ Saved: {octree_normals}")
    print("  → Normals computed using octree neighborhood")
    
    # Step 2: Compute normals using standard method
    print("\nStep 2: Computing normals with standard method...")
    std_normals = output_dir / "02_standard_normals.ply"
    result = run_cli("process", "normals", input_file, "-o", str(std_normals),
                     "--radius", "0.05")
    print(f"  ✓ Saved: {std_normals}")
    print("  → Standard k-NN based normal estimation")
    
    # Step 3: Orient normals consistently using MST
    print("\nStep 3: Orienting normals consistently (MST algorithm)...")
    oriented_normals = output_dir / "03_oriented_normals.ply"
    result = run_cli("normals", "orient-mst", str(std_normals), 
                     "-o", str(oriented_normals),
                     "--knn", "6")
    print(f"  ✓ Saved: {oriented_normals}")
    print("  → Normals now have consistent orientation")
    print("  → Uses Minimum Spanning Tree algorithm")
    
    # Step 4: Invert normal directions
    print("\nStep 4: Inverting all normal directions...")
    inverted_normals = output_dir / "04_inverted_normals.ply"
    result = run_cli("normals", "invert", str(oriented_normals),
                     "-o", str(inverted_normals))
    print(f"  ✓ Saved: {inverted_normals}")
    print("  → All normals flipped 180°")
    
    # Step 5: Convert normals to dip/dip direction
    print("\nStep 5: Converting normals to dip and dip direction...")
    dip_file = output_dir / "05_dip_direction.ply"
    result = run_cli("normals", "to-dip", str(oriented_normals),
                     "-o", str(dip_file))
    print(f"  ✓ Saved: {dip_file}")
    print("  → Created 'Dip' and 'Dip direction' scalar fields")
    print("  → Useful for geological analysis")
    
    # Step 6: Export normals as scalar fields
    print("\nStep 6: Exporting normal components as scalar fields...")
    normals_as_sf = output_dir / "06_normals_as_sfs.ply"
    result = run_cli("normals", "to-sfs", str(oriented_normals),
                     "-o", str(normals_as_sf))
    print(f"  ✓ Saved: {normals_as_sf}")
    print("  → Created 'Nx', 'Ny', 'Nz' scalar fields")
    print("  → Each component can be analyzed separately")
    
    # Step 7: Clear normals
    print("\nStep 7: Removing all normals from point cloud...")
    no_normals = output_dir / "07_no_normals.ply"
    result = run_cli("normals", "clear", str(oriented_normals),
                     "-o", str(no_normals))
    print(f"  ✓ Saved: {no_normals}")
    print("  → Point cloud now has no normal vectors")
    
    # Summary
    print("\n" + "=" * 60)
    print("Normal Processing Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {octree_normals.name} - Octree-based normals")
    print(f"  2. {std_normals.name} - Standard k-NN normals")
    print(f"  3. {oriented_normals.name} - Consistently oriented (MST)")
    print(f"  4. {inverted_normals.name} - Inverted directions")
    print(f"  5. {dip_file.name} - Dip angle & direction (geology)")
    print(f"  6. {normals_as_sf.name} - Normal components as SFs")
    print(f"  7. {no_normals.name} - Normals removed")
    
    print("\nNormal vector use cases:")
    print("  • Surface reconstruction (Poisson requires normals)")
    print("  • Lighting and shading in visualization")
    print("  • Geological dip/strike analysis")
    print("  • Surface orientation classification")
    print("  • Feature detection and matching")
    
    print("\nImportant:")
    print("  • MST orientation ensures consistent inward/outward facing")
    print("  • Dip/dip direction useful for geological surveys")
    print("  • Normal components (Nx, Ny, Nz) can reveal surface patterns")


if __name__ == "__main__":
    main()
