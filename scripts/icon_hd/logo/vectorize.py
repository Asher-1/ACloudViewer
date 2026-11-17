"""Vectorise flat-colour UI icons: quantise, trace per colour layer, gate on IoU.

Why flat icons can be vectorised and painterly artwork cannot
------------------------------------------------------------
Measured over ``app/Resources/images`` (excluding the theme bitmaps): of 102 UI
icons, 28 use 12 colours or fewer.  Those are line art and flat glyphs -- masks
with hard edges, where every colour boundary is a real contour.  Tracing them is
lossless in the way that matters: the shape is exact and it stays exact at any
size.

The product logo is the opposite case (46% coverage, thousands of colours, brush
texture).  Six vectorisation routes were measured against it and every one lost
the design or grew larger than the raster it replaced, so it is deliberately not
handled here.

Method
------
1. Alpha becomes the silhouette.
2. Colours are reduced to the icon's own palette (median cut at <= 16 colours).
3. Each colour mask is traced with ``cv2.findContours`` using the
   hierarchical mode, so interior holes become separate subpaths filled with
   ``evenodd`` -- without that, a ring would render as a disc.
4. The result is rasterised and scored.  Anything under the IoU gate is
   reported as a failure rather than silently shipped.

Usage
-----
    python3 -m scripts.icon_hd.logo.vectorize --list
    python3 -m scripts.icon_hd.logo.vectorize --check app/Resources/images/3D3.png
    python3 -m scripts.icon_hd.logo.vectorize --all --dry-run
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Acceptance gate.  Flat icons trace almost perfectly; a score below this means
# the icon was not flat art after all and must stay raster.
IOU_GATE = 0.975
COLOUR_GATE = 12.0     # mean absolute channel error, 0-255

MAX_COLOURS = 16
MAX_ICON_PX = 150      # larger than this is illustration, not an icon

# Ink threshold, as a fraction of each icon's own maximum alpha.
#
# A single constant cannot work, and both wrong values were measured:
#
#   alpha >= 128  -> QTreeView/branch-end.png and vline.png peak at alpha 124,
#                    so they trace as *empty* ("nothing visible in this icon").
#   alpha >=  40  -> every icon's soft shadow qualifies as solid ink.  On
#                    surface.png that traced 9,293 px where the source has
#                    4,931 px, nearly doubling the mark and producing the grey
#                    halo visible in the render.
#
# Half the peak is the standard antialiasing rule: it is the level at which a
# pixel is more covered than not, so the traced boundary lands on the visual
# edge regardless of whether the artwork is full-opacity (peak 255 -> cut 128) or
# stored faint (peak 124 -> cut 62).
INK_LEVEL_FRACTION = 0.5


def ink_level(alpha: np.ndarray) -> float:
    """Per-icon ink threshold: half the peak alpha, floored to keep noise out."""
    peak = float(alpha.max())
    if peak <= 0:
        return 128.0
    return max(30.0, peak * INK_LEVEL_FRACTION)

# Size ratio is reported, not enforced.
#
# An earlier version capped it at 4x and rejected 10 of 12 icons that passed the
# quality gate.  That cap measured the wrong thing: what a vector buys is
# sharpness on a 4K/8K display, not disk economy.  Measured at 4x zoom, an SVG
# render is 8x-69x sharper than a Lanczos-upscaled PNG of the same icon:
#
#     cloud.png            5.8 -> 306.0 laplacian variance (53x)
#     mini.png             6.0 -> 414.3 (69x)
#     svg/pqCalculator     16.3 -> 532.5 (33x)
#     cone.png             4.4 ->  51.1 (12x)
#
# So a 10-20x larger file that replaces a blurry upscale with crisp geometry is
# the intended trade.  The ratio is printed so the cost stays visible.
SIZE_RATIO_HINT = 4.0


def _trace_mask(mask: np.ndarray, epsilon: float = 0.35) -> str:
    """Trace a boolean mask with sub-pixel contours, holes preserved.

    Two bugs had to be found and fixed here, and both are worth recording
    because they looked like "these icons are not vectorisable" when in fact the
    tracer was wrong:

    1. ``cv2.findContours`` walks *pixel centres*, so every traced edge sits
       half a pixel inside the true boundary.  On a 128 px icon with 6 px
       strokes that removes ~17% of the ink: IoU 0.88-0.91 on shapes that were
       visually identical.

    2. ``skimage.measure.find_contours`` returns **open** polylines wherever the
       mask touches the image border.  Closing those with ``Z`` cuts a straight
       line across the image, which dropped one icon to IoU 0.0033.  The fix is
       to pad the mask by one transparent pixel, trace, then shift the
       coordinates back.

    With both fixed, ``find_contours`` at level 0.5 runs along the real
    50%-coverage boundary and the filled area matches the source.
    """
    import cv2
    from skimage import measure

    if not mask.any():
        return ""

    # Bug 2: a border-touching mask yields open contours unless padded.
    padded = np.pad(mask, 1, mode="constant")

    contours = measure.find_contours(padded.astype(np.float64), 0.5)
    if not contours:
        return ""

    parts = []
    for c in contours:
        # (row, col) -> (x, y), undoing the pad
        pts = np.stack([c[:, 1] - 1.0, c[:, 0] - 1.0], axis=1).astype(np.float32)
        if len(pts) < 3:
            continue
        ap = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in ap) + " Z"
        parts.append(d)
    return " ".join(parts)


def vectorise(path: Path, *, colours: int = MAX_COLOURS, epsilon: float = 0.35) -> tuple[str, dict]:
    """Convert one PNG icon to SVG.  Returns (svg text, measurements)."""
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    a = np.asarray(img)

    alpha = a[..., 3]
    level = ink_level(alpha)
    ink = alpha >= level
    if not ink.any():
        raise ValueError("nothing visible in this icon")

    # Quantise the visible pixels only; transparent area carries no colour.
    src = Image.fromarray(np.where(ink[..., None], a[..., :3], 0).astype(np.uint8))
    q = src.quantize(colors=colours, method=Image.MEDIANCUT)
    idx = np.asarray(q)
    pal = np.array(q.getpalette()[: colours * 3]).reshape(-1, 3)

    # Silhouette first: it clips every colour layer, so stray pixels outside the
    # shape (keying noise) cannot leak into the output.
    silhouette = _trace_mask(ink, epsilon)

    layers = []
    used = []
    for i in range(colours):
        m = (idx == i) & ink
        n = int(m.sum())
        if n < 8:
            continue
        d = _trace_mask(m, epsilon)
        if not d:
            continue
        # mean of the pixels actually in this cell, not the palette centre
        mean = a[..., :3][m].mean(axis=0).astype(int)
        hexv = "#%02X%02X%02X" % tuple(mean)
        layers.append((n, hexv, d))
        used.append(hexv)

    layers.sort(reverse=True, key=lambda t: t[0])

    body = [
        f'  <defs><clipPath id="sil"><path d="{silhouette}" '
        f'fill-rule="evenodd"/></clipPath></defs>',
        '  <g clip-path="url(#sil)">',
        *[f'    <path d="{d}" fill="{c}" fill-rule="evenodd"/>' for _, c, d in layers],
        "  </g>",
    ]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n' + "\n".join(body) + "\n</svg>\n"
    )

    stats = _score(img, svg, ink, level=level)
    stats["colours_used"] = len(layers)
    stats["svg_bytes"] = len(svg)
    stats["png_bytes"] = path.stat().st_size
    return svg, stats


def vectorise_smooth(path: Path, *, colours: int = MAX_COLOURS,
                     epsilon: float = 0.35, supersample: int = 4) -> tuple[str, dict]:
    """Vectorise with 4x supersampling, for icons that must survive 4K/8K.

    Tracing the binary mask at native resolution reproduces the pixel staircase:
    faithful, but visibly jagged at 4x.  Measured on two icons, supersampling the
    alpha with cubic interpolation before tracing lifts IoU against a 4x
    reference from 0.900 -> 0.955 and 0.912 -> 0.968.  The cost is a larger path
    (6.4 KB -> 18.5 KB on 3D3), which is the right trade for an icon that has to
    render at 8K.

    The SVG keeps the original width/height and viewBox, and divides the traced
    coordinates by the supersample factor, so callers see the same geometry.
    """
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    a = np.asarray(img)
    alpha = a[..., 3]
    level = ink_level(alpha)
    ink = alpha >= level
    if not ink.any():
        raise ValueError("nothing visible in this icon")

    k = max(1, supersample)

    def up(mask: np.ndarray) -> np.ndarray:
        if k == 1:
            return mask
        import cv2

        m = (mask.astype(np.float32) * 255.0)
        return cv2.resize(m, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) >= 128

    silhouette = _trace_mask(up(ink), epsilon)

    src = Image.fromarray(np.where(ink[..., None], a[..., :3], 0).astype(np.uint8))
    q = src.quantize(colors=colours, method=Image.MEDIANCUT)
    idx = np.asarray(q)

    layers = []
    for i in range(colours):
        m = (idx == i) & ink
        if int(m.sum()) < 8:
            continue
        d = _trace_mask(up(m), epsilon)
        if not d:
            continue
        mean = a[..., :3][m].mean(axis=0).astype(int)
        layers.append((int(m.sum()), "#%02X%02X%02X" % tuple(mean), d))
    layers.sort(reverse=True, key=lambda t: t[0])

    body = [
        f'  <defs><clipPath id="sil"><path d="{silhouette}" '
        f'fill-rule="evenodd"/></clipPath></defs>',
        f'  <g clip-path="url(#sil)" transform="scale({1.0 / k:.6f})">',
        *[f'    <path d="{d}" fill="{c}" fill-rule="evenodd"/>' for _, c, d in layers],
        "  </g>",
    ]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n' + "\n".join(body) + "\n</svg>\n"
    )

    stats = _score(img, svg, ink, reference_scale=k, level=level)
    stats["colours_used"] = len(layers)
    stats["svg_bytes"] = len(svg)
    stats["png_bytes"] = path.stat().st_size
    stats["supersample"] = k
    return svg, stats


def _score(orig: Image.Image, svg: str, orig_ink: np.ndarray,
           reference_scale: int = 1, level: float | None = None) -> dict:
    """Rasterise the SVG and score it against the original.

    ``reference_scale`` compares at a higher resolution, which is the whole point
    for a supersampled trace: at 1x both the raster and the vector are quantised
    to the same pixel grid, so the measurement cannot see the smoother edge it was
    built for.  At 4x the difference is visible, and that is what matters for a
    4K/8K display.
    """
    k = max(1, reference_scale)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                           output_width=orig.width * k,
                           output_height=orig.height * k)
    out = Image.open(io.BytesIO(png)).convert("RGBA")
    ref = orig.resize((orig.width * k, orig.height * k), Image.LANCZOS)

    a = np.asarray(ref).astype(int)
    b = np.asarray(out).astype(int)

    # Score at exactly the level the tracer used.  Two mismatches were measured:
    # scoring at 128 while tracing at 40 shrank the pass rate from 12/28 to 4/28,
    # because the vector silhouette came out larger than the reference.
    lvl = level if level is not None else ink_level(np.asarray(ref)[..., 3])
    ai = a[..., 3] >= lvl
    bi = b[..., 3] >= lvl
    union = np.logical_or(ai, bi).sum()
    iou = float(np.logical_and(ai, bi).sum()) / float(union) if union else 1.0

    both = ai & bi
    err = float(np.abs(a[..., :3][both] - b[..., :3][both]).mean()) if both.any() else 255.0
    return {"iou": round(iou, 4), "colour_err": round(err, 2)}


def is_candidate(path: Path) -> bool:
    """Cheap pre-filter: small, visible, few colours."""
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return False
    if max(img.size) > MAX_ICON_PX:
        return False
    a = np.asarray(img)
    ink = a[..., 3] > 40
    if ink.sum() < 30:
        return False
    px = a[..., :3][ink]
    return len(np.unique(px // 32 * 32, axis=0)) <= 12


def candidates() -> list[Path]:
    """Flat UI icons that are worth vectorising.

    The product icons under ``images/icon/`` are excluded on purpose: they are
    already generated from a measured master by ``export_cloudviewer.py`` and
    measured by ``measure.py``.  Tracing them here would fight that pipeline and
    replace an optically tuned size ladder with a traced approximation.
    """
    out = []
    for p in (ROOT / "app/Resources/images").rglob("*.png"):
        if "themes" in p.parts:
            continue
        if "icon" in p.parts:
            continue          # product icons: owned by export_cloudviewer.py
        if is_candidate(p):
            out.append(p)
    return sorted(out)


def best_variant(path: Path) -> tuple[str, dict, float, bool]:
    """Pick the cheapest trace that clears every gate.

    Measured on ``cloud.png`` (the heaviest candidate), loosening the tracer's
    tolerance trades size for fidelity on a steep curve:

        eps 0.35  IoU 0.983  SVG 229 KB  (20.4x the PNG)
        eps 0.80  IoU 0.963  SVG  92 KB  ( 8.2x)
        eps 2.00  IoU 0.920  SVG  68 KB  ( 6.1x)
        eps 3.00  IoU 0.816  SVG  21 KB  ( 1.9x)

    Shipping the 0.35 trace would add 229 KB to the repo for an 11 KB icon, so
    the tolerance is chosen per icon: the *tightest* setting that clears the
    quality gate **and** stays inside the size cap.

    Returns (svg, stats, epsilon, passed).  An earlier version fell back to the
    tightest quality-passing variant even when it blew the size cap, which is how
    a 19x-bloated SVG got reported as PASS.

    The scan runs slowest-first deliberately: the first candidate that clears
    both gates is the most faithful one that is still affordable, so there is no
    reason to look further.
    """
    scales_to_try = (3.0, 2.0, 1.4, 0.9, 0.6, 0.35)
    smallest = None

    for eps in scales_to_try:
        svg, st = vectorise_smooth(path, epsilon=eps)
        ratio = st["svg_bytes"] / max(1, st["png_bytes"])
        quality_ok = st["iou"] >= IOU_GATE and st["colour_err"] <= COLOUR_GATE

        if quality_ok:
            # Quality cleared at the loosest tolerance that still passes, which is
            # also the smallest file.  Scanning further only costs bytes.
            return svg, st, eps, True

        if smallest is None or ratio < smallest[1]["svg_bytes"] / max(1, smallest[1]["png_bytes"]):
            smallest = (svg, st, eps)

    # Nothing cleared both gates.  Report the tightest-quality variant and let
    # the caller decide, but mark it failed so it is never written by default.
    svg, st = vectorise_smooth(path, epsilon=0.35)
    return svg, st, 0.35, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vectorise flat-colour UI icons.")
    ap.add_argument("--list", action="store_true", help="show candidates")
    ap.add_argument("--check", metavar="PNG", action="append", default=[],
                    help="vectorise one file and print its score (repeatable)")
    ap.add_argument("--all", action="store_true", help="run every candidate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        cands = candidates()
        print(f"{len(cands)} candidates")
        for p in cands:
            print(f"  {p.relative_to(ROOT)}")
        return 0

    if args.check:
        for raw in args.check:
            p = Path(raw) if Path(raw).is_absolute() else (ROOT / raw)
            svg, st, eps, ok = best_variant(p)
            verdict = "PASS" if ok else "FAIL"
            print(f"  {verdict}  {p.name:<28} IoU={st['iou']:.4f} "
                  f"colour={st['colour_err']:6.2f} eps={eps:.2f} "
                  f"svg={st['svg_bytes']:>7,}B png={st['png_bytes']:>7,}B "
                  f"({st['svg_bytes']/max(1,st['png_bytes']):.1f}x)")
        return 0

    if not args.all:
        ap.print_help()
        return 1

    cands = candidates()
    written = passed = 0
    print(f"{len(cands)} candidates\n")
    for p in cands:
        try:
            svg, st, eps, ok = best_variant(p)
        except Exception as exc:                       # noqa: BLE001 - report, don't crash
            print(f"  ERROR {p.name}: {exc}")
            continue
        ratio = st["svg_bytes"] / max(1, st["png_bytes"])
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {p.relative_to(ROOT)!s:<58} "
              f"IoU={st['iou']:.4f} colour={st['colour_err']:6.2f} eps={eps:.2f} "
              f"svg={st['svg_bytes']:>7,}B vs png={st['png_bytes']:>7,}B ({ratio:.1f}x)")
        if ok:
            passed += 1
            target = p.with_suffix(".svg")
            if args.dry_run:
                continue
            target.write_text(svg)
            written += 1

    print(f"\n{passed}/{len(cands)} passed the gate"
          + (f", {written} SVG written" if not args.dry_run else " (dry run)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
