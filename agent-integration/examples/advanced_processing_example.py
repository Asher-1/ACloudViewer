#!/usr/bin/env python3
"""Advanced point cloud processing example.

Demonstrates advanced geometric feature extraction and analysis:
- Density computation
- Curvature analysis (mean and Gaussian)
- Roughness estimation
- Connected component extraction
- Statistical outlier removal

Works with ACloudViewer binary in headless mode.

Usage:
    python advanced_processing_example.py input.ply output_dir/
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
        print("Usage: advanced_processing_example.py <input.ply> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Advanced Point Cloud Processing Example")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Clean noise with Statistical Outlier Removal
    print("Step 1: Removing statistical outliers...")
    clean_file = output_dir / "01_cleaned.ply"
    result = run_cli("process", "sor", input_file, "-o", str(clean_file), 
                     "--knn", "6", "--sigma", "1.0")
    print(f"  ✓ Saved: {clean_file}")
    
    # Step 2: Compute local density
    print("\nStep 2: Computing local density...")
    density_file = output_dir / "02_density.ply"
    result = run_cli("process", "density", str(clean_file), "-o", str(density_file),
                     "--radius", "0.05")
    print(f"  ✓ Saved: {density_file}")
    print("  → Point cloud now has 'Density' scalar field")
    
    # Step 3: Compute mean curvature
    print("\nStep 3: Computing mean curvature...")
    curvature_file = output_dir / "03_mean_curvature.ply"
    result = run_cli("process", "curvature", str(clean_file), "-o", str(curvature_file),
                     "--type", "MEAN")
    print(f"  ✓ Saved: {curvature_file}")
    print("  → Point cloud now has 'Mean curvature' scalar field")
    
    # Step 4: Compute Gaussian curvature
    print("\nStep 4: Computing Gaussian curvature...")
    gauss_file = output_dir / "04_gauss_curvature.ply"
    result = run_cli("process", "curvature", str(clean_file), "-o", str(gauss_file),
                     "--type", "GAUSS")
    print(f"  ✓ Saved: {gauss_file}")
    
    # Step 5: Compute roughness
    print("\nStep 5: Computing surface roughness...")
    roughness_file = output_dir / "05_roughness.ply"
    result = run_cli("process", "roughness", str(clean_file), "-o", str(roughness_file),
                     "--radius", "0.1")
    print(f"  ✓ Saved: {roughness_file}")
    print("  → Point cloud now has 'Roughness' scalar field")
    
    # Step 6: Compute geometric features
    print("\nStep 6: Computing geometric features...")
    features_file = output_dir / "06_geometric_features.ply"
    result = run_cli("process", "feature", str(clean_file), 
                     "-o", str(features_file), 
                     "--type", "SURFACE_VARIATION", "--kernel-size", "0.1")
    print(f"  ✓ Saved: {features_file}")
    
    # Step 7: Extract connected components
    print("\nStep 7: Extracting connected components (min 100 points)...")
    components_file = output_dir / "07_components.ply"
    result = run_cli("process", "extract-cc", str(clean_file),
                     "-o", str(components_file), "--min-points", "100", "--octree-level", "8")
    print(f"  ✓ Saved components to: {components_file}")
    
    # Step 8: Color by height using color banding
    print("\nStep 8: Coloring by height (Z-axis) with color banding...")
    colored_file = output_dir / "08_colored_by_height.ply"
    result = run_cli("process", "color-banding", str(clean_file), 
                     "-o", str(colored_file), "--axis", "Z", "--frequency", "10.0")
    print(f"  ✓ Saved: {colored_file}")
    print("  → Rainbow color bands along Z-axis")
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {clean_file.name} - Cleaned point cloud (SOR filter)")
    print(f"  2. {density_file.name} - Local density scalar field")
    print(f"  3. {curvature_file.name} - Mean curvature scalar field")
    print(f"  4. {gauss_file.name} - Gaussian curvature scalar field")
    print(f"  5. {roughness_file.name} - Surface roughness scalar field")
    print(f"  6. {features_file.name} - Geometric features")
    print(f"  7. {components_file.name} - Connected components")
    print(f"  8. {colored_file.name} - Height-colored visualization")
    print("\nAll scalar fields can be visualized in ACloudViewer!")


if __name__ == "__main__":
    main()
