"""Recover the high-resolution masters and measure the parameters the marks need.

Why this exists
---------------
The only high-resolution record of the original designs is a pair of 1024x1024
rasters.  For ACloudViewer that raster is now stored directly as
``masters/acloudviewer_1024_clean.png``; it was recovered from a thin SVG wrapper
(``logo_256.svg``, 1.48 MB of base64 around the PNG) that has since been deleted,
because it was 0 % vector, referenced by nothing, and shipped in the repository.

CloudViewer's master lives in ``masters/cloudviewer_1024.png``, likewise
recovered from a 559 KB base64 wrapper (``cloudviewer_256.svg``) which has since
been replaced by genuine vector art at that path.

Those rasters are the ground truth this whole folder is derived from.  This
module measures the geometry that ``geometry.py`` states as constants and
extracts the 80-point particle field verbatim.  Running it is optional -- its
output is committed as JSON -- but it makes every number in ``geometry.py``
reproducible instead of folklore.

Usage:
    python3 -m scripts.icon_hd.logo.measure            # print measurements
    python3 -m scripts.icon_hd.logo.measure --write    # refresh particles.json
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARTICLES_JSON = HERE / "particles.json"

# The two wrappers that still carry the original 1024 px artwork.
MASTER_SOURCES = {
    "acv": "scripts/icon_hd/logo/masters/acloudviewer_1024_clean.png",
    "cv": "scripts/icon_hd/logo/masters/cloudviewer_1024.png",
}


def extract_master(kind: str) -> Image.Image:
    """Load the 1024x1024 master for one product mark.

    ACloudViewer reads the recovered PNG directly.  CloudViewer still comes out
    of a legacy SVG wrapper, because that is the only place its 1024 px artwork
    survives in git -- the wrapper is base64 around a raster, so the payload has
    to be decoded rather than read as SVG.
    """
    rel = MASTER_SOURCES[kind]
    path = ROOT / rel

    if rel.endswith(".png"):
        return Image.open(path).convert("RGBA")

    try:
        raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
    except subprocess.CalledProcessError:
        # Fall back to the working tree (already restored in this repo).
        raw = path.read_bytes()
    text = raw.decode("utf-8", "replace")
    blobs = re.findall(r"base64,([A-Za-z0-9+/=\s]+)", text)
    if not blobs:
        raise RuntimeError(f"{rel} carries no embedded raster")
    data = base64.b64decode(re.sub(r"\s+", "", blobs[0]))
    return Image.open(io.BytesIO(data)).convert("RGBA")


def ink_bbox(img: Image.Image, thresh: int = 8):
    """Bounding box of the visible (alpha > thresh) ink."""
    alpha = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(alpha > thresh)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def hexagon_vertices(img: Image.Image) -> dict[str, tuple[int, int]]:
    """Silhouette extremes of the isometric cube."""
    a = np.asarray(img)
    al = a[..., 3] > 8

    top_y = 0
    while not al[top_y].any():
        top_y += 1
    bot_y = al.shape[0] - 1
    while not al[bot_y].any():
        bot_y -= 1

    top_x = int(np.nonzero(al[top_y])[0].mean())
    bot_x = int(np.nonzero(al[bot_y])[0].mean())

    right_x = 0
    while not al[:, right_x + 1].any():
        right_x += 1
    right_x = al.shape[1] - 1
    while not al[:, right_x].any():
        right_x -= 1
    left_x = 0
    while not al[:, left_x].any():
        left_x += 1

    ry = np.nonzero(al[:, right_x])[0]
    ly = np.nonzero(al[:, left_x])[0]
    return {
        "top": (top_x, top_y),
        "bot": (bot_x, bot_y),
        "tr": (right_x, int(ry.min())),
        "br": (right_x, int(ry.max())),
        "tl": (left_x, int(ly.min())),
        "bl": (left_x, int(ly.max())),
    }


def crescent_geometry(img: Image.Image) -> dict[str, float]:
    """Inner/outer radius of the C ring from its widest horizontal cut."""
    a = np.asarray(img)
    al = a[..., 3] > 8
    h, w = al.shape
    cy = h // 2
    row = al[cy]
    segs = []
    start = None
    for x in range(w):
        if row[x] and start is None:
            start = x
        elif not row[x] and start is not None:
            segs.append((start, x - 1))
            start = None
    if start is not None:
        segs.append((start, w - 1))
    # the leftmost solid segment of the ribbon gives inner and outer edge
    left_segs = [s for s in segs if s[1] < w // 2]
    if len(left_segs) < 2:
        return {}
    outer_x = left_segs[0][0]
    inner_x = left_segs[-1][1]
    cx = w // 2
    return {
        "center_x": cx,
        "center_y": cy,
        "r_outer": float(cx - outer_x),
        "r_inner": float(cx - inner_x),
    }


def extract_particles(img: Image.Image, min_px: int = 20, max_px: int = 600):
    """The isolated dots floating beside the cube, with their measured colours."""
    from scipy import ndimage

    a = np.asarray(img).astype(int)
    al = a[..., 3]
    lab, n = ndimage.label(al > 60)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))

    out = []
    for i in range(1, n + 1):
        if not (min_px <= sizes[i - 1] <= max_px):
            continue
        ys, xs = np.nonzero(lab == i)
        solid = al[ys, xs] > 200
        if not solid.any():
            continue
        col = a[ys[solid], xs[solid]][:, :3].mean(axis=0).astype(int)
        out.append(
            {
                "x": round(float(xs.mean()), 1),
                "y": round(float(ys.mean()), 1),
                "r": round(float(math.sqrt(sizes[i - 1] / math.pi)), 1),
                "hex": "#%02X%02X%02X" % tuple(col),
                "a": 0.95,
            }
        )
    out.sort(key=lambda p: (p["y"], p["x"]))
    return out


def main(argv=None) -> int:
    write = "--write" in (argv or sys.argv[1:])

    acv = extract_master("acv")
    cv = extract_master("cv")

    print("### ACloudViewer master", acv.size)
    print("  bbox:", ink_bbox(acv))
    for name, xy in hexagon_vertices(acv).items():
        print(f"  {name:<4} {xy}")

    print("\n### CloudViewer master", cv.size)
    print("  bbox:", ink_bbox(cv))
    for k, v in crescent_geometry(cv).items():
        print(f"  {k:<10} {v}")

    parts = extract_particles(acv)
    print(f"\n### Particles: {len(parts)}")
    if parts:
        print("  r range:", min(p["r"] for p in parts), "..", max(p["r"] for p in parts))
        print("  x range:", min(p["x"] for p in parts), "..", max(p["x"] for p in parts))
        print("  y range:", min(p["y"] for p in parts), "..", max(p["y"] for p in parts))

    if write:
        PARTICLES_JSON.write_text(json.dumps(parts, indent=0) + "\n")
        print(f"\nwrote {PARTICLES_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
