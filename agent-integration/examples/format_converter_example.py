#!/usr/bin/env python3
"""Format conversion and batch processing example.

Demonstrates file format conversion capabilities:
- Single file conversion (50+ formats)
- Batch directory conversion
- Format filtering
- Cross-type conversion (cloud ↔ mesh)
- Compressed formats (Draco, LAZ)

Works with ACloudViewer binary in headless mode.

Usage:
    python format_converter_example.py input_dir/ output_dir/
    python format_converter_example.py single_file.ply output.pcd
"""

import subprocess
import sys
import json
from pathlib import Path


def run_cli(*args, check=True, return_json=False):
    """Run CLI command and return result."""
    cmd = ["cli-anything-acloudviewer", "--json", "--mode", "headless"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    if return_json and result.stdout:
        try:
            return json.loads(result.stdout)
        except:
            pass
    return result


def convert_single_file(input_path, output_path):
    """Convert a single file."""
    print("=" * 60)
    print("Single File Conversion")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")
    
    result = run_cli("convert", str(input_path), str(output_path), return_json=True)
    
    if isinstance(result, dict):
        print("Conversion result:")
        print(f"  Input format:  {result.get('input_format', 'N/A')}")
        print(f"  Output format: {result.get('output_format', 'N/A')}")
        print(f"  Status:        {result.get('status', 'N/A')}")
    
    print(f"\n✓ Converted: {output_path}")


def batch_convert_directory(input_dir, output_dir):
    """Batch convert all files in a directory."""
    print("=" * 60)
    print("Batch Directory Conversion")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")
    
    # Example 1: Convert all files to PLY
    print("Example 1: Converting all files to PLY format...")
    ply_dir = output_dir / "ply_output"
    result = run_cli("batch-convert", str(input_dir), str(ply_dir), 
                     "--format", ".ply", return_json=True)
    
    if isinstance(result, dict):
        print(f"  Converted: {result.get('converted', 0)} files")
        print(f"  Errors:    {result.get('errors', 0)} files")
        print(f"  Status:    {result.get('status', 'N/A')}")
    print(f"  ✓ Output: {ply_dir}\n")
    
    # Example 2: Convert only PLY files to PCD
    print("Example 2: Converting only PLY files to PCD format...")
    pcd_dir = output_dir / "pcd_output"
    result = run_cli("batch-convert", str(input_dir), str(pcd_dir),
                     "--format", ".pcd",
                     "--filter-ext", ".ply", return_json=True)
    
    if isinstance(result, dict):
        print(f"  Converted: {result.get('converted', 0)} files")
        print(f"  Errors:    {result.get('errors', 0)} files")
    print(f"  ✓ Output: {pcd_dir}\n")
    
    # Example 3: Convert to Draco compressed format
    print("Example 3: Converting to Draco compressed format...")
    drc_dir = output_dir / "draco_output"
    result = run_cli("batch-convert", str(input_dir), str(drc_dir),
                     "--format", ".drc", return_json=True)
    
    if isinstance(result, dict):
        print(f"  Converted: {result.get('converted', 0)} files")
        print(f"  ✓ Draco provides high compression for point clouds")
    print(f"  ✓ Output: {drc_dir}\n")
    
    # Example 4: Convert meshes to OBJ
    print("Example 4: Converting all mesh files to OBJ...")
    obj_dir = output_dir / "obj_output"
    result = run_cli("batch-convert", str(input_dir), str(obj_dir),
                     "--format", ".obj",
                     "--filter-ext", ".stl", ".ply", ".fbx", return_json=True)
    
    if isinstance(result, dict):
        print(f"  Converted: {result.get('converted', 0)} files")
    print(f"  ✓ Output: {obj_dir}\n")


def demonstrate_formats():
    """Show supported formats."""
    print("=" * 60)
    print("Supported File Formats")
    print("=" * 60)
    print("\nQuerying available formats...\n")
    
    result = run_cli("formats", return_json=True)
    
    if isinstance(result, dict):
        for category, formats in result.items():
            print(f"{category}:")
            if isinstance(formats, list):
                for fmt in formats[:5]:  # Show first 5
                    print(f"  - {fmt}")
                if len(formats) > 5:
                    print(f"  ... and {len(formats) - 5} more")
            print()


def main():
    if len(sys.argv) < 3:
        print("Usage: format_converter_example.py <input> <output>")
        print("\nExamples:")
        print("  # Single file")
        print("  python format_converter_example.py scene.ply output.pcd")
        print()
        print("  # Batch directory")
        print("  python format_converter_example.py ./scans/ ./converted/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)
    
    # Demonstrate supported formats first
    demonstrate_formats()
    
    # Route to appropriate handler
    if input_path.is_file():
        # Single file conversion
        convert_single_file(input_path, output_path)
    elif input_path.is_dir():
        # Batch directory conversion
        output_path.mkdir(parents=True, exist_ok=True)
        batch_convert_directory(input_path, output_path)
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("Format Conversion Tips")
    print("=" * 60)
    print("\nCommon conversions:")
    print("  PLY ↔ PCD   - Point cloud formats")
    print("  PLY → OBJ   - Point cloud to mesh")
    print("  OBJ → PLY   - Mesh to point cloud (auto-samples)")
    print("  LAZ → PLY   - Decompress LAS/LAZ")
    print("  PLY → DRC   - Compress with Draco")
    print("  E57 → PLY   - Industry standard to PLY")
    print("  FBX → OBJ   - 3D model conversion")
    print("\nCross-type conversion:")
    print("  • Point Cloud → Mesh: Automatic Poisson reconstruction")
    print("  • Mesh → Point Cloud: Uniform surface sampling (100K points)")
    print("\nFor full format list, run:")
    print("  cli-anything-acloudviewer formats")


if __name__ == "__main__":
    main()
