#!/usr/bin/env python3
"""Losslessly re-encode generated PNGs and prove decoded pixels are unchanged."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent
output = root / "output"
rows = []
for path in sorted(output.glob("*.png")):
    with Image.open(path) as image:
        before_mode = image.mode
        before_size = image.size
        before_pixels = image.convert("RGBA").tobytes()
    before_pixel_hash = hashlib.sha256(before_pixels).hexdigest()
    temporary = path.with_suffix(".optimized.png")
    with Image.open(path) as image:
        image.save(temporary, format="PNG", optimize=True, compress_level=9)
    with Image.open(temporary) as image:
        after_pixels = image.convert("RGBA").tobytes()
        if image.size != before_size or after_pixels != before_pixels:
            raise RuntimeError(f"Lossless PNG verification failed: {path.name}")
    temporary.replace(path)
    rows.append({
        "name": path.name,
        "width": before_size[0],
        "height": before_size[1],
        "original_mode": before_mode,
        "output_bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_rgba_sha256": before_pixel_hash,
        "decoded_pixels_unchanged": True,
    })
expected = sorted(Path(root / "expected-names.txt").read_text().splitlines())
actual = sorted(row["name"] for row in rows)
if actual != expected:
    raise RuntimeError(f"Output set mismatch: missing={sorted(set(expected)-set(actual))}; extra={sorted(set(actual)-set(expected))}")
(root / "asset-manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"count": len(rows), "total_bytes": sum(row["output_bytes"] for row in rows), "all_decoded_pixels_unchanged": all(row["decoded_pixels_unchanged"] for row in rows)}, indent=2))
