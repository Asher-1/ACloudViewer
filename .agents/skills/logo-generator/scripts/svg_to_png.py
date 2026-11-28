#!/usr/bin/env python3
"""
Convert SVG to PNG using cairosvg.
"""

import sys
import argparse
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg not installed.")
    print("Install with: pip install cairosvg")
    sys.exit(1)


def svg_to_png(svg_path: str, png_path: str, width: int = 1024, height: int = 1024) -> bool:
    """
    Convert SVG file to PNG.

    Args:
        svg_path: Path to input SVG file
        png_path: Path to output PNG file
        width: Output width in pixels
        height: Output height in pixels

    Returns:
        True if successful, False otherwise
    """
    try:
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=width,
            output_height=height
        )
        print(f"✓ Converted: {svg_path} -> {png_path}")
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to PNG")
    parser.add_argument("svg_file", help="Path to SVG file")
    parser.add_argument("--output", "-o", help="Output PNG path (default: same name with .png)")
    parser.add_argument("--width", "-w", type=int, default=1024, help="Output width (default: 1024)")
    parser.add_argument("--height", "-H", type=int, default=1024, help="Output height (default: 1024)")

    args = parser.parse_args()

    svg_path = Path(args.svg_file)
    if not svg_path.exists():
        print(f"Error: SVG file not found: {svg_path}")
        sys.exit(1)

    if args.output:
        png_path = Path(args.output)
    else:
        png_path = svg_path.with_suffix('.png')

    success = svg_to_png(str(svg_path), str(png_path), args.width, args.height)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
