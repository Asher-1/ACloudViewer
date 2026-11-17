"""Build the two product marks as true vector, faithful to the original artwork.

Strategy per mark, derived from what each mark actually is
----------------------------------------------------------
**CloudViewer** -- organic.  The C is a tapered ribbon (outer edge swings from
r=243 to r=489, throws detached tails, asymmetric opening) wrapped around a warm
disc carrying 92 points.  A horizontal cut through the master returns four
separate spans, so no parameterised arc can reproduce it.  The ribbon is
therefore *traced* from its own master and re-emitted as smooth Beziers; the
disc and its 92 dots are drawn as clean primitives on top.  Gate: IoU >= 0.985.

**ACloudViewer** -- hybrid.  Its top and right faces are exact geometry (a
hexagonal isometric cube, measured vertices), but its left face is dissolving:
25.5% of that region is fully transparent, because the plane is torn into a
lattice with holes and loose points flying off it.  Measured facts:
    left  face: 25.5% alpha==0, 69.5% alpha>230
    right face:  0.1% alpha==0, 99.8% alpha>230
So the mark is built as: traced body (keeps the torn holes, IoU 0.985) filled
with the indigo left-face gradient, then the exact top and right face polygons
drawn over it (crisp edges), then the amber core, then the 80 measured
particles.

Neither mark gets a background plate.  The originals float on transparency and
that is a large part of their impact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

from .geometry import (
    ACV_CORE_CENTER,
    ACV_CORE_HOT_RADIUS,
    ACV_CORE_RADIUS,
    ACV_CUBE,
    CV_DISC_CENTER,
    CV_DISC_RADIUS,
)
from .palette import ACV, CV

HERE = Path(__file__).resolve().parent


def f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Tracing: contour -> smooth cubic Bezier
# ---------------------------------------------------------------------------


def trace_path(alpha: np.ndarray, *, epsilon: float = 1.0, tension: float = 0.45,
               level: float = 128.0, min_len: int = 8) -> tuple[str, int]:
    """Trace every contour of `alpha` into one evenodd SVG path.

    Returns (path data, contour count).  Catmull-Rom tangents are converted to
    cubic Beziers so the traced outline keeps the smooth, drawn quality of the
    original instead of showing the pixel staircase.
    """
    import cv2

    contours = [c for c in measure.find_contours(alpha.astype(np.float64), level)
                if len(c) >= min_len]
    contours.sort(key=len, reverse=True)

    subpaths = []
    for c in contours:
        pts = np.stack([c[:, 1], c[:, 0]], axis=1)  # (row,col) -> (x,y)
        ap = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.float32),
                              epsilon, True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        n = len(ap)
        d = [f"M {ap[0][0]:.2f} {ap[0][1]:.2f}"]
        for i in range(n):
            p0, p1 = ap[(i - 1) % n], ap[i]
            p2, p3 = ap[(i + 1) % n], ap[(i + 2) % n]
            c1 = p1 + (p2 - p0) * (tension / 3.0)
            c2 = p2 - (p3 - p1) * (tension / 3.0)
            d.append(f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} "
                     f"{p2[0]:.2f} {p2[1]:.2f}")
        d.append("Z")
        subpaths.append(" ".join(d))
    return " ".join(subpaths), len(contours)


def iou(alpha_a: np.ndarray, alpha_b: np.ndarray, level: float = 128.0) -> float:
    """Intersection-over-union of two alpha masks: the fidelity gate."""
    a, b = alpha_a >= level, alpha_b >= level
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / float(union)


# ---------------------------------------------------------------------------
# CloudViewer
# ---------------------------------------------------------------------------


def _ribbon_gradient(master: Image.Image) -> str:
    """Sample the C-ring's own vertical colour ramp.

    Only *cool* pixels are sampled.  An earlier version averaged every opaque
    pixel on a scanline and picked up the warm beige disc, so the ramp collapsed
    to two muddy stops in the middle of the mark (measured: #C9D3BD at mid
    height, a blend of cyan and cream).  Filtering on "blue clearly exceeds red"
    keeps the ribbon's own light and nothing else.

    Measured ramp of the artwork: #92E5EE at the top, #75CBE1 through the
    shoulders, #6FBCE8 at the base.
    """
    a = np.asarray(master).astype(int)
    alpha, rgb = a[..., 3], a[..., :3]
    h, w = alpha.shape

    stops = []
    for frac in np.arange(0.03, 1.0, 0.06):
        y = int(frac * (h - 1))
        xs = np.nonzero(alpha[y] > 200)[0]
        if len(xs) == 0:
            continue
        px = rgb[y, xs]
        cool = px[:, 2] > px[:, 0] + 15      # ribbon, never the warm disc
        if cool.sum() < 30:
            continue
        col = px[cool].mean(axis=0).astype(int)
        stops.append((float(frac), "#%02X%02X%02X" % tuple(col)))

    if not stops:                            # degenerate fallback
        stops = [(0.0, CV["ring_light"]), (1.0, CV["ring_deep"])]

    body = "\n".join(
        f'      <stop offset="{off:.2f}" stop-color="{col}"/>' for off, col in stops
    )
    return (f'    <linearGradient id="cvRibbon" gradientUnits="userSpaceOnUse" '
            f'x1="0" y1="0" x2="0" y2="{h}">\n{body}\n    </linearGradient>')


def _load_json(name: str) -> list:
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else []


def cloudviewer_mark(master: Image.Image, *, small: bool = False) -> str:
    """Rebuild the CloudViewer mark from its own master."""
    a = np.asarray(master)
    alpha = a[..., 3].astype(np.float64)
    rgb = a[..., :3].astype(int)

    # Ribbon = cool pixels (blue dominant); disc and dots = warm pixels.
    warm = (rgb[..., 0] > rgb[..., 2] + 12) & (alpha > 60)
    ribbon_d, _ = trace_path(np.where(warm, 0, alpha),
                             epsilon=1.25 if small else 1.0)

    dots = _load_json("cv_dots.json")
    if small:
        dots = [d for d in dots if d["r"] >= 6.0]

    cx, cy = CV_DISC_CENTER
    parts = [
        "  <defs>",
        _ribbon_gradient(master),
        f'    <radialGradient id="cvDisc" gradientUnits="userSpaceOnUse" '
        f'cx="{f(cx)}" cy="{f(cy)}" r="{f(CV_DISC_RADIUS)}">',
        f'      <stop offset="0.55" stop-color="{CV["disc_pale"]}"/>',
        f'      <stop offset="1" stop-color="{CV["disc_deep"]}"/>',
        "    </radialGradient>",
        "  </defs>",
        f'  <path d="{ribbon_d}" fill="url(#cvRibbon)" fill-rule="evenodd"/>',
        f'  <circle cx="{f(cx)}" cy="{f(cy)}" r="{f(CV_DISC_RADIUS)}" fill="url(#cvDisc)"/>',
        '  <g id="cv-dots">',
        *[f'    <circle cx="{f(d["x"])}" cy="{f(d["y"])}" r="{f(d["r"])}" fill="{d["hex"]}"/>'
          for d in dots],
        "  </g>",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ACloudViewer
# ---------------------------------------------------------------------------


def _acv_defs() -> str:
    c = ACV_CUBE
    tl, tr, bl, br, bot = c["tl"], c["tr"], c["bl"], c["br"], c["bot"]
    return f"""  <defs>
    <linearGradient id="acvTop" gradientUnits="userSpaceOnUse"
                    x1="{f(tl[0])}" y1="{f(c['top'][1])}" x2="{f(br[0])}" y2="{f(bot[1])}">
      <stop offset="0" stop-color="{ACV['top_near']}"/>
      <stop offset="1" stop-color="{ACV['top_far']}"/>
    </linearGradient>
    <linearGradient id="acvRight" gradientUnits="userSpaceOnUse"
                    x1="{f(tr[0])}" y1="{f(tr[1])}" x2="{f(br[0])}" y2="{f(br[1])}">
      <stop offset="0" stop-color="{ACV['right_top']}"/>
      <stop offset="1" stop-color="{ACV['right_bottom']}"/>
    </linearGradient>
    <linearGradient id="acvLeft" gradientUnits="userSpaceOnUse"
                    x1="{f(tl[0])}" y1="0" x2="{f(c['center'][0])}" y2="0">
      <stop offset="0" stop-color="{ACV['left_base']}" stop-opacity="0.34"/>
      <stop offset="0.33" stop-color="{ACV['left_base']}" stop-opacity="0.56"/>
      <stop offset="0.70" stop-color="{ACV['left_deep']}" stop-opacity="0.84"/>
      <stop offset="1" stop-color="{ACV['left_deep']}" stop-opacity="0.97"/>
    </linearGradient>
    <radialGradient id="acvCore">
      <stop offset="0" stop-color="{ACV['core_hot']}"/>
      <stop offset="0.14" stop-color="{ACV['core_mid']}" stop-opacity="0.95"/>
      <stop offset="0.44" stop-color="{ACV['core_warm']}" stop-opacity="0.60"/>
      <stop offset="1" stop-color="{ACV['core_out']}" stop-opacity="0"/>
    </radialGradient>
  </defs>"""


def _acv_body(master: Image.Image) -> tuple[str, int]:
    """Trace the cube's largest connected component, keeping thin struts thin.

    ``tension=0`` is deliberate: Catmull-Rom smoothing rounds every corner, and
    the left face is a lattice of hairlines whose whole character lives in its
    straight segments.  Smoothing them inflates each strut into a blob and
    closes the gaps between them, which is exactly the defect the first attempt
    produced.  Polylines keep the structure; epsilon still removes the pixel
    staircase.
    """
    alpha = np.asarray(master)[..., 3].astype(np.float64)
    lab, n = ndimage.label(alpha >= 128)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        body = np.where(lab == (int(np.argmax(sizes)) + 1), alpha, 0)
    else:
        body = alpha
    return trace_path(body, epsilon=0.85, tension=0.0, min_len=10)


def _acv_faces() -> str:
    """The two exact faces drawn over the traced body.

    Only the top and the right face: they are the crisp, fully opaque planes.
    The left face is left to the traced body so its holes and fibrous edge
    survive.
    """
    c = ACV_CUBE
    o = c["center"]
    pts = lambda q: " ".join(f"{f(x)},{f(y)}" for x, y in q)
    top = [c["top"], c["tr"], o, c["tl"]]
    right = [o, c["tr"], c["br"], c["bot"]]
    return "\n".join([
        f'  <polygon points="{pts(top)}" fill="url(#acvTop)"/>',
        f'  <polygon points="{pts(right)}" fill="url(#acvRight)"/>',
    ])


def _acv_lattice(density: int = 7) -> str:
    """Triangular lattice on the two solid faces, as in the artwork."""
    c = ACV_CUBE
    o = c["center"]
    lerp = lambda p, q, t: (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
    faces = ((c["top"], c["tr"], o, c["tl"]), (o, c["tr"], c["br"], c["bot"]))
    out = [f'  <g stroke="{ACV["mesh"]}" stroke-opacity="0.34" '
           f'stroke-width="1.4" fill="none">']
    for a, b, cc, d in faces:
        for i in range(1, density):
            t = i / density
            for p, q in ((lerp(a, b, t), lerp(d, cc, t)),
                         (lerp(a, d, t), lerp(b, cc, t))):
                out.append(f'    <line x1="{f(p[0])}" y1="{f(p[1])}" '
                           f'x2="{f(q[0])}" y2="{f(q[1])}"/>')
        for i in range(density):
            t = (i + 0.5) / density
            p, q = lerp(a, b, t), lerp(d, cc, t)
            out.append(f'    <line x1="{f(p[0])}" y1="{f(p[1])}" '
                       f'x2="{f(q[0])}" y2="{f(q[1])}"/>')
    out.append("  </g>")
    return "\n".join(out)


def _acv_edges() -> str:
    """The four interior edges: they are what make the cube read as a cube."""
    c = ACV_CUBE
    o = c["center"]
    pairs = ((c["tl"], o), (c["tr"], o), (o, c["bot"]))
    return "\n".join(
        f'  <line x1="{f(p[0])}" y1="{f(p[1])}" x2="{f(q[0])}" y2="{f(q[1])}" '
        f'stroke="{ACV["mesh"]}" stroke-opacity="0.55" stroke-width="2.2"/>'
        for p, q in pairs
    )


def _acv_core() -> str:
    cx, cy = ACV_CORE_CENTER
    return "\n".join([
        f'  <circle cx="{f(cx)}" cy="{f(cy)}" r="{f(ACV_CORE_RADIUS)}" fill="url(#acvCore)"/>',
        f'  <circle cx="{f(cx)}" cy="{f(cy)}" r="{f(ACV_CORE_HOT_RADIUS)}" '
        f'fill="{ACV["core_hot"]}" fill-opacity="0.88"/>',
    ])


def _acv_dissolve(particles, *, solid: bool = False) -> str:
    """The left face: a plane dissolving cleanly into free points.

    Measured facts about this face in the original:
        dark indigo base   #3E4E84
        pale strokes       #BDCBC9
        opacity along x    30% at the silhouette -> 84% at the cube centre
        alpha histogram    bimodal (solid strokes, open gaps)

    Three attempts to reproduce the brush-stroke texture geometrically all
    failed in the same way: a mesh dense enough to reach the measured 72%
    coverage turns into either a dark blob (traced then filled) or a white
    scaffold laid over the face.  The texture is painterly, not geometric, and
    faking it is what made the earlier results look worse than the original.

    So the refinement goes the other way: keep the composition and the colour,
    drop the brush noise, and let a clean gradient hand the plane over to the
    measured point fragments.  That is a *refinement* of the same idea -- the
    plane becomes points -- rendered in the crisp vector language the rest of
    the mark already speaks.
    """
    c = ACV_CUBE
    tl, bl, o, bot = c["tl"], c["bl"], c["center"], c["bot"]

    # At 16-32 px the gradient collapses into a pale, muddy wedge: the
    # measured 32 px master paints this face as one solid navy (#2B3A8F), which
    # is what makes the cube read at small sizes.  So the small tier uses the
    # solid colour and drops the ramp entirely.
    fill = ACV["left_solid"] if solid else "url(#acvLeft)"
    out = [
        f'  <polygon points="{f(tl[0])},{f(tl[1])} {f(o[0])},{f(o[1])} '
        f'{f(bot[0])},{f(bot[1])} {f(bl[0])},{f(bl[1])}" fill="{fill}"/>',
    ]

    # Fragments: the measured points carry the dissolve into open space.
    out.append('  <g id="acv-fragments">')
    for p in particles:
        out.append(f'    <circle cx="{f(p["x"])}" cy="{f(p["y"])}" '
                   f'r="{f(p["r"])}" fill="{p["hex"]}" fill-opacity="0.95"/>')
    out.append("  </g>")
    return "\n".join(out)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clip_to_left_face(mx: float, my: float, dx: float, dy: float, half: float):
    """Clip a line through (mx,my) along (dx,dy) to the left-face quadrilateral.

    A clipPath would be shorter, but Qt's SVG renderer is unreliable with them,
    so the intersection is computed geometrically instead.  Returns the two
    endpoints inside the polygon, or None when the line misses it.
    """
    c = ACV_CUBE
    poly = [c["tl"], c["center"], c["bot"], c["bl"]]

    # parametric point on the line: p(t) = m + t*d,  t in [-half, half]
    ts = []
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            continue
        # m + t*d = a + u*e
        t = ((ax - mx) * ey - (ay - my) * ex) / denom
        u = ((ax - mx) * dy - (ay - my) * dx) / denom
        if -1e-6 <= u <= 1 + 1e-6 and -half <= t <= half:
            ts.append(t)

    inner = [t for t in ts if -half < t < half]
    if len(inner) < 2:
        return None
    t0, t1 = min(inner), max(inner)
    return ((mx + dx * t0, my + dy * t0), (mx + dx * t1, my + dy * t1))


def acloudviewer_mark(master: Image.Image, *, small: bool = False) -> str:
    """Rebuild the ACloudViewer mark.

    Layer order, and the reason for it:

    1. the dissolving left face (parametric lattice + measured fragments),
    2. the two exact right/top faces painted over it, so their edges stay
       razor sharp and the dissolve is clipped at the cube's centre edge,
    3. the lattice on the solid faces,
    4. the interior cube edges,
    5. the amber core.

    The left face is deliberately *not* traced: filling its silhouette turns a
    skeleton of thin struts into a dark blob, which is the defect the earlier
    attempts produced.  See ``_acv_dissolve``.
    """
    parts = [
        _acv_defs(),
        _acv_dissolve([] if small else _load_json("particles.json"), solid=small),
        _acv_faces(),
    ]
    if not small:
        parts.append(_acv_lattice())
    parts.append(_acv_edges())
    parts.append(_acv_core())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def wrap(body: str, size: int = 1024) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}">\n{body}\n</svg>\n')
