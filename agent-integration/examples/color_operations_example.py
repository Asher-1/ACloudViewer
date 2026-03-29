#!/usr/bin/env python3
"""Color operations and visualization example.

Demonstrates color manipulation techniques available in headless mode:
- Color banding by coordinate axis
- Scalar field to color mapping
- Color removal
- Creating color-coded clouds for comparison

NOTE: paint-uniform and paint-by-height require GUI mode with entity_id.
This example focuses on headless operations.

Works with ACloudViewer binary in headless mode.

Usage:
    python color_operations_example.py input.ply output_dir/
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
        print("Usage: color_operations_example.py <input.ply> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Color Operations Example (Headless Mode)")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Color banding by Z axis
    print("Step 1: Applying color banding along Z axis...")
    banded_z = output_dir / "01_banded_z.ply"
    result = run_cli("process", "color-banding", input_file,
                     "-o", str(banded_z),
                     "--axis", "Z", "--frequency", "10.0")
    print(f"  ✓ Saved: {banded_z}")
    print("  → Rainbow color bands from min Z to max Z\n")
    
    # Step 2: Color banding by X axis
    print("Step 2: Applying color banding along X axis...")
    banded_x = output_dir / "02_banded_x.ply"
    result = run_cli("process", "color-banding", input_file,
                     "-o", str(banded_x),
                     "--axis", "X", "--frequency", "10.0")
    print(f"  ✓ Saved: {banded_x}")
    print("  → Rainbow color bands from min X to max X\n")
    
    # Step 3: Color banding by Y axis
    print("Step 3: Applying color banding along Y axis...")
    banded_y = output_dir / "03_banded_y.ply"
    result = run_cli("process", "color-banding", input_file,
                     "-o", str(banded_y),
                     "--axis", "Y", "--frequency", "5.0")
    print(f"  ✓ Saved: {banded_y}")
    print("  → Lower frequency = wider color bands\n")
    
    # Step 4: Compute density and color-code it via SF to RGB
    print("Step 4: Computing density and converting to RGB colors...")
    density_cloud = output_dir / "04_density.ply"
    result = run_cli("process", "density", input_file,
                     "-o", str(density_cloud),
                     "--radius", "0.05")
    print(f"  ✓ Computed density: {density_cloud}")
    
    # Set density as active and convert to RGB
    density_active = output_dir / "density_active.ply"
    result = run_cli("sf", "set-active", str(density_cloud),
                     "-o", str(density_active),
                     "--sf-index", "0")
    
    density_rgb = output_dir / "04_density_rgb.ply"
    result = run_cli("sf", "convert-to-rgb", str(density_active),
                     "-o", str(density_rgb))
    print(f"  ✓ Saved: {density_rgb}")
    print("  → Density scalar field → RGB colors\n")
    
    # Step 5: Create height scalar field and color-code it
    print("Step 5: Creating height SF and converting to colors...")
    height_sf = output_dir / "05_height_sf.ply"
    result = run_cli("sf", "coord-to-sf", input_file,
                     "-o", str(height_sf),
                     "--dimension", "Z")
    print(f"  ✓ Created height SF: {height_sf}")
    
    height_rgb = output_dir / "05_height_rgb.ply"
    result = run_cli("sf", "convert-to-rgb", str(height_sf),
                     "-o", str(height_rgb))
    print(f"  ✓ Saved: {height_rgb}")
    print("  → Height gradient visualization\n")
    
    # Step 6: Remove colors from a file
    print("Step 6: Removing all colors...")
    no_color = output_dir / "06_no_color.ply"
    result = run_cli("process", "remove-rgb", str(banded_z),
                     "-o", str(no_color))
    print(f"  ✓ Saved: {no_color}")
    print("  → Colors removed, ready for re-coloring\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("Color Operations Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {banded_z.name} - Z-axis color bands")
    print(f"  2. {banded_x.name} - X-axis color bands")
    print(f"  3. {banded_y.name} - Y-axis color bands (lower frequency)")
    print(f"  4. {density_rgb.name} - Density as RGB")
    print(f"  5. {height_rgb.name} - Height as RGB")
    print(f"  6. {no_color.name} - Colors removed")
    
    print("\nHeadless color operations:")
    print("  • color-banding: Rainbow bands along X/Y/Z axis")
    print("  • sf convert-to-rgb: Convert any scalar field to colors")
    print("  • remove-rgb: Strip all colors from point cloud")
    
    print("\nFor more color control (GUI mode required):")
    print("  • cloud paint-uniform <entity_id> R G B")
    print("  • cloud paint-by-height <entity_id> --axis z")
    print("  • cloud paint-by-scalar-field <entity_id> --field 'Density'")
    
    print("\nWorkflow for custom coloring:")
    print("  1. Compute a scalar field (density, curvature, roughness)")
    print("  2. Set it as active: sf set-active input.ply -o active.ply --sf-index 0")
    print("  3. Convert to RGB: sf convert-to-rgb active.ply -o colored.ply")
    print("  4. Or filter first: sf filter active.ply -o filtered.ply --min X --max Y")
    
    print("\nVisualization:")
    print(f"  ACloudViewer {banded_z}")


if __name__ == "__main__":
    main()
