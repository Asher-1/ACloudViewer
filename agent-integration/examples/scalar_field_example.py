#!/usr/bin/env python3
"""Scalar field operations example.

Demonstrates comprehensive scalar field manipulation:
- Creating scalar fields from coordinates
- Arithmetic operations between scalar fields
- Mathematical operations on scalar fields
- Filtering points by scalar field values
- Color mapping from scalar fields
- Gradient computation

Works with ACloudViewer binary in headless mode.

Usage:
    python scalar_field_example.py input.ply output_dir/
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
        print("Usage: scalar_field_example.py <input.ply> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Scalar Field Operations Example")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Create scalar field from Z coordinate
    print("Step 1: Creating 'Height' scalar field from Z coordinate...")
    height_file = output_dir / "01_height_sf.ply"
    result = run_cli("sf", "coord-to-sf", input_file, "-o", str(height_file),
                     "--dimension", "Z")
    print(f"  ✓ Saved: {height_file}")
    print("  → Created scalar field from Z coordinate")
    
    # Step 2: Compute density
    print("\nStep 2: Computing density scalar field...")
    density_file = output_dir / "02_density.ply"
    result = run_cli("process", "density", str(height_file), "-o", str(density_file),
                     "--radius", "0.05")
    print(f"  ✓ Saved: {density_file}")
    print("  → Now has both 'Height' and 'Density' scalar fields")
    
    # Step 3: Scalar field operation - square root of density (index 1)
    print("\nStep 3: Computing sqrt(Density) using scalar field index...")
    sqrt_file = output_dir / "03_sqrt_density.ply"
    result = run_cli("sf", "arithmetic", str(density_file), "-o", str(sqrt_file),
                     "--sf-index", "1", "--operation", "SQRT")
    print(f"  ✓ Saved: {sqrt_file}")
    print("  → Applied SQRT operation to Density scalar field")
    
    # Step 4: Scale a scalar field by multiplying with constant
    print("\nStep 4: Scaling Height (SF 0) by 2.0...")
    scaled_file = output_dir / "04_scaled.ply"
    result = run_cli("sf", "operation", str(height_file), "-o", str(scaled_file),
                     "--sf-index", "0", "--operation", "MULTIPLY", "--value", "2.0")
    print(f"  ✓ Saved: {scaled_file}")
    print("  → Multiplied Height by 2.0")
    
    # Step 5: Add constant to scalar field
    print("\nStep 5: Adding 10.0 to Height scalar field...")
    added_file = output_dir / "05_height_plus_10.ply"
    result = run_cli("sf", "operation", str(height_file), "-o", str(added_file),
                     "--sf-index", "0", "--operation", "ADD", "--value", "10.0")
    print(f"  ✓ Saved: {added_file}")
    
    # Step 6: Compute gradient of density
    print("\nStep 6: Computing gradient of Density...")
    gradient_file = output_dir / "06_density_gradient.ply"
    # First set density as active SF
    density_active = output_dir / "density_active.ply"
    result = run_cli("sf", "set-active", str(density_file), "-o", str(density_active),
                     "--sf-index", "1")
    result = run_cli("sf", "gradient", str(density_active), "-o", str(gradient_file),
                     "--euclidean")
    print(f"  ✓ Saved: {gradient_file}")
    print("  → Created gradient magnitude scalar field")
    
    # Step 7: Filter points by scalar field value range
    print("\nStep 7: Filtering points by active scalar field value...")
    # First ensure Height is active
    height_active = output_dir / "height_active.ply"
    result = run_cli("sf", "set-active", str(height_file), "-o", str(height_active),
                     "--sf-index", "0")
    filtered_file = output_dir / "07_filtered_by_height.ply"
    result = run_cli("sf", "filter", str(height_active), "-o", str(filtered_file),
                     "--min", "0.3", "--max", "0.7")
    print(f"  ✓ Saved: {filtered_file}")
    print("  → Kept only points within scalar field range")
    
    # Step 8: Convert active scalar field to RGB
    print("\nStep 8: Converting active scalar field to RGB colors...")
    rgb_file = output_dir / "08_sf_as_rgb.ply"
    result = run_cli("sf", "convert-to-rgb", str(height_active), "-o", str(rgb_file))
    print(f"  ✓ Saved: {rgb_file}")
    print("  → Active scalar field converted to point colors")
    
    # Step 9: Rename scalar field
    print("\nStep 9: Renaming scalar field 0 to 'Elevation'...")
    renamed_file = output_dir / "09_renamed_sf.ply"
    result = run_cli("sf", "rename", str(height_file), "-o", str(renamed_file),
                     "--old", "0", "--new", "Elevation")
    print(f"  ✓ Saved: {renamed_file}")
    
    # Step 10: Remove a scalar field
    print("\nStep 10: Removing scalar field at index 1...")
    removed_file = output_dir / "10_removed_sf.ply"
    result = run_cli("sf", "remove", str(density_file), "-o", str(removed_file),
                     "--sf-index", "1")
    print(f"  ✓ Saved: {removed_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Scalar Field Processing Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {height_file.name} - Height from Z coordinate")
    print(f"  2. {density_file.name} - Height + Density")
    print(f"  3. {sqrt_file.name} - sqrt(Density)")
    print(f"  4. {scaled_file.name} - Height × 2.0")
    print(f"  5. {added_file.name} - Height + 10.0")
    print(f"  6. {gradient_file.name} - Gradient of Density")
    print(f"  7. {filtered_file.name} - Filtered by value range")
    print(f"  8. {rgb_file.name} - SF converted to RGB")
    print(f"  9. {renamed_file.name} - Renamed scalar field")
    print(f" 10. {removed_file.name} - One SF removed")
    
    print("\nKey scalar field commands:")
    print("  • coord-to-sf: Create SF from X/Y/Z coordinates")
    print("  • sf-arithmetic: Unary ops (SQRT, ABS, EXP, LOG)")
    print("  • sf-op: Binary ops with constant (ADD, MULTIPLY, etc.)")
    print("  • sf-gradient: Compute gradient magnitude")
    print("  • filter-sf: Remove points outside SF range")
    print("  • sf-to-rgb: Convert active SF to colors")
    print("  • set-active-sf, rename-sf, remove-sf")
    print("\nScalar field operations enable powerful data analysis and visualization!")


if __name__ == "__main__":
    main()
