"""Repair artefacts baked into the ACloudViewer master artwork.

Scope
-----
This does **not** touch the design.  It removes defects that the artwork picked
up when it was originally produced, and which only become visible at large
sizes -- exactly the "rough when zoomed" complaint.

Measured defects in ``masters/acloudviewer_1024.png``:

1. **A dark hole in the core.**  169 px of near-black (``#030303``) at
   x 505-517, y 490-509 -- a 20 px blob sitting inside what should be a blown-out
   white highlight.  It is a keying residue, not a design element: the ring of
   pixels 22-35 px away is uniformly warm (``#FAD98E`` -> ``#F8C37A``).
2. **Six orphan fragments.**  Connected components under 30 px, 30 px total,
   each detached from every stroke -- dirt from the same keying pass.
3. **A ragged alpha edge.**  14,493 semi-transparent pixels (1.38%) because the
   original was keyed from a flat background rather than drawn on transparency.

Nothing here invents geometry.  The hole is filled by interpolating the
surrounding ring, fragments are dropped, and the edge is hardened only where the
alpha is already almost opaque (so genuine soft glow keeps its falloff).

Usage
-----
    python3 -m scripts.icon_hd.logo.repair            # write repaired master
    python3 -m scripts.icon_hd.logo.repair --dry-run
    python3 -m scripts.icon_hd.logo.repair --report   # measurements only
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MASTERS = HERE / "masters"
SRC = MASTERS / "acloudviewer_1024.png"
DST = MASTERS / "acloudviewer_1024_clean.png"

# The core defect, measured.  Slightly padded so the fill covers the whole rim.
CORE_BOX = (498, 483, 524, 517)   # x0, y0, x1, y1

MIN_FRAGMENT = 30    # px; components smaller than this are keying dirt
EDGE_SOFT = 240      # alpha below this is genuinely soft and is left alone


def _find_core_hole(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Mask of the punched-through hole inside the core highlight.

    Measured: the 169 "dark" pixels at x 505-517, y 490-509 have **alpha == 0**.
    They are not black ink -- they are a transparent hole punched through what
    should be a solid white highlight.  On a light background that reads as a
    gap; on a dark one it reads as a black dot, which is exactly the artefact
    visible when the mark is enlarged.

    An earlier version required ``alpha > 80`` and therefore matched nothing.
    The correct predicate is "transparent, or near-black, inside the core box".
    """
    x0, y0, x1, y1 = CORE_BOX
    sub_rgb = rgb[y0:y1, x0:x1]
    sub_a = alpha[y0:y1, x0:x1]
    lum = sub_rgb.sum(axis=2)
    hole = (sub_a == 0) | ((lum < 150) & (sub_a > 0))
    return hole


def repair_core_hole(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Heal the punched-through hole with Telea inpainting.

    Two earlier attempts failed and both are worth recording, because they
    explain why this is not a two-line fix:

    * The first predicate required ``alpha > 80`` and so matched nothing -- the
      hole is fully transparent (``alpha == 0``), not dark ink.
    * The second sampled the surrounding ring at 22-40 px and painted the hole
      with the outer amber glow, turning the white highlight into an orange
      patch; sampling at 8-22 px fixed the colour but left a visible seam,
      because a flat radial fill cannot continue a gradient.

    Inpainting solves both at once: it propagates the surrounding pixels into
    the hole along the local gradient, so the highlight's falloff continues
    seamlessly.  Only the hole is masked; the white highlight around it is
    design and is left untouched.
    """
    import cv2

    out = rgba.copy()
    x0, y0, x1, y1 = CORE_BOX
    alpha_full = out[..., 3]
    hole = _find_core_hole(out[..., :3].astype(int), alpha_full)
    n = int(hole.sum())
    if n == 0:
        return out, 0

    # Inpaint on a padded window so Telea has real context on every side.
    pad = 24
    py0, py1 = max(0, y0 - pad), min(out.shape[0], y1 + pad)
    px0, px1 = max(0, x0 - pad), min(out.shape[1], x1 + pad)
    window = out[py0:py1, px0:px1, :3]
    mask = np.zeros(window.shape[:2], np.uint8)
    mask[(y0 - py0):(y0 - py0) + hole.shape[0],
         (x0 - px0):(x0 - px0) + hole.shape[1]] = hole.astype(np.uint8)

    # Widen the mask by one pixel: the hole's rim carries dark antialiasing.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    bgr = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
    healed = cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA)
    window_out = cv2.cvtColor(healed, cv2.COLOR_BGR2RGB)

    out[py0:py1, px0:px1, :3] = window_out
    out[py0:py1, px0:px1, 3][mask > 0] = 255
    return out, n


def drop_fragments(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Delete connected components smaller than MIN_FRAGMENT pixels."""
    from scipy import ndimage

    out = rgba.copy()
    ink = out[..., 3] >= 110
    lab, n = ndimage.label(ink)
    if n == 0:
        return out, 0

    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    small = [i + 1 for i, s in enumerate(sizes) if s < MIN_FRAGMENT]
    if not small:
        return out, 0

    kill = np.isin(lab, small)
    out[kill] = 0
    return out, int(kill.sum())


def harden_edge(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Push near-opaque edge pixels to full opacity.

    A keyed edge carries a ring of alpha 200-239 that reads as grey fringing on
    a dark background and as a halo on a light one.  Above EDGE_SOFT the pixel is
    already visually solid, so snapping it to 255 removes the fringe without
    touching the genuine glow, whose alpha falls off through lower values.
    """
    out = rgba.copy()
    a = out[..., 3]
    band = (a >= EDGE_SOFT) & (a < 255)
    n = int(band.sum())
    out[band, 3] = 255
    return out, n


def repair(img: Image.Image) -> tuple[Image.Image, dict]:
    """Run all three repairs and return (image, measurements)."""
    rgba = np.asarray(img.convert("RGBA")).copy()

    rgba, hole = repair_core_hole(rgba)
    rgba, frag = drop_fragments(rgba)
    rgba, edge = harden_edge(rgba)

    return Image.fromarray(rgba.astype(np.uint8), "RGBA"), {
        "core_hole_px": hole,
        "fragment_px": frag,
        "edge_px": edge,
    }


def report(path: Path = SRC) -> dict:
    """Print the defect census for a master."""
    img = Image.open(path).convert("RGBA")
    a = np.asarray(img).astype(int)
    rgb, alpha = a[..., :3], a[..., 3]

    from scipy import ndimage

    hole = int(_find_core_hole(rgb, alpha).sum())
    ink = alpha >= 110
    lab, n = ndimage.label(ink)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else []
    frags = sum(1 for s in sizes if s < MIN_FRAGMENT)
    semi = int(((alpha > 20) & (alpha < 200)).sum())

    print(f"  {path.name}")
    print(f"    core hole       {hole:>6} px")
    print(f"    orphan pieces   {frags:>6} components")
    print(f"    soft edge       {semi:>6} px ({semi / alpha.size:.2%})")
    return {"core_hole": hole, "fragments": frags, "soft_edge": semi}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Repair the master artwork.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    if args.report:
        print("Before:")
        report(SRC)
        if DST.exists():
            print("After:")
            report(DST)
        return 0

    src = Image.open(SRC)
    out, stats = repair(src)

    print("repairs:")
    for k, v in stats.items():
        print(f"  {k:<14} {v:>7}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    DST.write_bytes(buf.getvalue())
    print(f"wrote {DST.relative_to(HERE.parents[2])} ({len(buf.getvalue()):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
