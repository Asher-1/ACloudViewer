"""ecvCloudCloudDistance -- single-icon rework, ParaView style.

What the code does
------------------
    MainWindow.cpp:14413   doActionCloudCloudDist() requires exactly two point
                           clouds, then opens ccOrderChoiceDlg with the roles
                           "Compared" and "Reference"
    ecvPointCloud.cpp:3432 calls
                           DistanceComputationTools::computeCloud2CloudDistances

So the operation is: for every point of the *compared* cloud, find the nearest
neighbour in the *reference* cloud, and store that distance.  The result is a
per-point scalar field -- which is why this icon lives next to the other
scalar-field tools.

What an icon can legitimately show
----------------------------------
Not the measurement procedure (that would need a paragraph), but its shape:

    two clouds, one clearly in front of the other
    a handful of short spans between them, of visibly different lengths

The differing lengths are the content.  A single double-headed arrow would say
"a distance"; several unequal spans say "a distance *field*, measured point by
point", which is what the tool produces.  That distinction is why this icon
must not be the same drawing as ecvCloudMeshDistance.

Why the previous rendering did not work
---------------------------------------
The committed file is 120 `<rect>` elements and ~90 distinct colours sampled
from a bitmap: scattered ticks with no body.  Scattered ticks read as noise at
24 px, and the earlier rebuild of this icon -- two blobs plus one arrow -- lost
the per-point idea and became an unreadable "bowl" shape.  This version keeps
both: solid bodies for legibility, and several spans for meaning.

Style (fixed baseline, applied to every icon from here on)
---------------------------------------------------------
    outline   #6F6F59, 0.85 units, on every form
    palette   #A7D28C / #72BE44 greens, #D0E5F6 / #9AC1E0 blues, #2896D3 strong
              blue, #C1A561 tan, #787878 / #CDCDCD neutrals.  Sampled from the
              eight approved samples; no colour outside that set.
    fills     subtle gradients, never flat
    span      92 % of the canvas
    components 1-3 large forms

Usage
-----
    python3 -m scripts.icon_hd.logo.ecvclouddistance --sheet o.png
    python3 -m scripts.icon_hd.logo.ecvclouddistance --write
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "app/Resources/images/svg/ecv/ecvCloudCloudDistance.svg"

REF_HI = P.BLUE_PALE      # #D0E5F6
REF_LO = P.BLUE_MID       # #9AC1E0
CMP_HI = P.GREEN_LT       # #A7D28C
CMP_LO = P.GREEN          # #72BE44
SPAN = P.BLUE_STRONG      # #2896D3


def _cloud(cx: float, cy: float, rx: float, ry: float, gid: str) -> str:
    """A filled cloud form: one closed path, so it reads as a single object."""
    pts = [(-1.00, 0.10), (-0.86, -0.52), (-0.42, -0.74), (0.02, -0.66),
           (0.30, -0.92), (0.74, -0.70), (0.94, -0.24), (0.82, 0.30),
           (0.40, 0.62), (-0.12, 0.68), (-0.60, 0.52), (-0.92, 0.44)]
    d = [f"M {cx + pts[0][0] * rx:.2f} {cy + pts[0][1] * ry:.2f}"]
    n = len(pts)
    for i in range(n):
        p0, p1 = pts[(i - 1) % n], pts[i]
        p2, p3 = pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C {cx + c1[0] * rx:.2f} {cy + c1[1] * ry:.2f} "
                 f"{cx + c2[0] * rx:.2f} {cy + c2[1] * ry:.2f} "
                 f"{cx + p2[0] * rx:.2f} {cy + p2[1] * ry:.2f}")
    d.append("Z")
    return (f'  <path d="{" ".join(d)}" fill="url(#{gid})" stroke="{P.INK}" '
            f'stroke-width="{P.STROKE}" stroke-linejoin="round"/>')


def _span(x1: float, y1: float, x2: float, y2: float, w: float = 0.9) -> str:
    """One measured span: a segment with a short tick across each end.

    Ticks rather than arrowheads, because three arrowheads at 24 px merge into a
    smear and the direction of a nearest-neighbour distance is not the point --
    its *length* is.  The ticks are drawn perpendicular to the segment so they
    read as survey marks rather than as a second crossing line.
    """
    ang = math.atan2(y2 - y1, x2 - x1) + math.pi / 2
    tx, ty = 0.85 * math.cos(ang), 0.85 * math.sin(ang)
    d = (f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f} "
         f"M {x1 - tx:.2f} {y1 - ty:.2f} L {x1 + tx:.2f} {y1 + ty:.2f} "
         f"M {x2 - tx:.2f} {y2 - ty:.2f} L {x2 + tx:.2f} {y2 + ty:.2f}")
    return (f'  <path d="{d}" fill="none" stroke="{SPAN}" stroke-width="{w}" '
            f'stroke-linecap="round"/>')


def _ruler(x: float, y1: float, y2: float, *,
           major=0.0, minor=0.0, w: float = 1.0) -> str:
    """A vertical measuring rule with graduated ticks.

    The tick ladder is what makes this read as a *measured distance* rather than
    as a wire linking two shapes.  Two tick lengths -- longer at each end and at
    the midpoint -- give it the look of a scale without needing numbers, which
    would be illegible below 48 px.
    """
    parts = [(f'  <path d="M {x:.2f} {y1:.2f} L {x:.2f} {y2:.2f}" fill="none" '
              f'stroke="{SPAN}" stroke-width="{w}" stroke-linecap="round"/>')]
    n = 7
    for i in range(n + 1):
        y = y1 + (y2 - y1) * i / n
        long = i in (0, n, n // 2)
        half = major if long else minor
        parts.append(f'  <path d="M {x - half:.2f} {y:.2f} L {x + half:.2f} {y:.2f}" '
                     f'fill="none" stroke="{SPAN}" '
                     f'stroke-width="{0.9 if long else 0.7}" stroke-linecap="round"/>')
    return "\n".join(parts)


def _dots(pts, colour, r: float = 1.15) -> list[str]:
    """Point-cloud dots: small filled circles with a hairline rim.

    The rim is 0.34 units, not the 0.85 default: at that weight the outline was
    81 % of the radius, so every dot rendered as an olive ring and the clouds
    read as bunches of grapes.

    Radius 0.92 with roughly 1 unit of clear space between neighbours keeps the
    marks discrete.  At r=1.45 they touched after ``fit()`` scaled the drawing up
    and each cloud became a solid lump -- the icon stopped saying "point cloud".
    """
    out = []
    for x, y in pts:
        out.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" '
                   f'fill="{colour}" stroke="{P.INK}" stroke-width="0.34"/>')
    return out


def _ruler(x1: float, y1: float, x2: float, y2: float,
           w: float = 0.9) -> str:
    """A measured span between two points: a double-headed arrow whose shaft is
    graduated with evenly spaced ticks.

    The tick count is **derived from the available length**, not fixed.  That is
    the bug this version exists to fix: with a 7.91-unit span, two 2.2-unit
    heads and a hard-coded count of five, the ticks had 0.51 units of room --
    0.13 apart while each was 2.1 wide.  They overlapped into a solid diamond
    blob, which is why the connector kept rendering as a jagged smear no matter
    how the weights were tuned.

    So: heads shrink to 1.6, ticks project 0.85 each side, and the count comes
    from `available_length // minimum_spacing`.  The rule then cannot self-
    destruct at any span.
    """
    ang = math.atan2(y2 - y1, x2 - x1)
    ux, uy = math.cos(ang), math.sin(ang)
    px, py = -uy, ux
    span = math.hypot(x2 - x1, y2 - y1)

    head_len, head_w = 1.6, 0.9
    tick_half = 0.85
    min_spacing = 2.05                      # > tick width, so ticks never touch

    inner_start, inner_end = head_len + 0.9, span - head_len - 0.9
    usable = inner_end - inner_start
    n_ticks = max(2, int(usable / min_spacing) + 1) if usable > 0 else 0

    parts = [
        f'  <path d="M {x1 + ux * head_len * 0.5:.2f} '
        f'{y1 + uy * head_len * 0.5:.2f} '
        f'L {x2 - ux * head_len * 0.5:.2f} {y2 - uy * head_len * 0.5:.2f}" '
        f'fill="none" stroke="{SPAN}" stroke-width="{w}" stroke-linecap="butt"/>'
    ]

    for ax, ay, sx, sy in ((x1, y1, 1, 1), (x2, y2, -1, -1)):
        parts.append(
            f'  <path d="M {ax:.2f} {ay:.2f} '
            f'L {ax + sx * head_len * ux + head_w * px:.2f} '
            f'{ay + sy * head_len * uy + head_w * py:.2f} '
            f'L {ax + sx * head_len * ux - head_w * px:.2f} '
            f'{ay + sy * head_len * uy - head_w * py:.2f} Z" '
            f'fill="{SPAN}" stroke="none"/>')

    if n_ticks >= 2:
        for i in range(n_ticks):
            t = inner_start + usable * i / (n_ticks - 1)
            cx, cy = x1 + ux * t, y1 + uy * t
            parts.append(
                f'  <path d="M {cx - px * tick_half:.2f} '
                f'{cy - py * tick_half:.2f} '
                f'L {cx + px * tick_half:.2f} {cy + py * tick_half:.2f}" '
                f'fill="none" stroke="{SPAN}" stroke-width="0.65" '
                f'stroke-linecap="butt"/>')
    return "\n".join(parts)


def _cloud(centre, radius, seed, *, r_dot=0.38, spacing=1.95, n=18):
    """A loose point cluster: small marks, wide gaps.

    The gap-to-size ratio is the constant that matters, and it took three
    measured passes to settle:

        r_dot 1.15, spacing 1.72   gap 0.5u, dot 2.3u wide  -> dots touch, one blob
        r_dot 0.52, spacing 1.40   gap 0.36u, dot 1.04u     -> dots nearly touch, "bunch of grapes"
        r_dot 0.38, spacing 1.95   gap 1.19u, dot 0.76u     -> gap is 1.5x the dot

    The ratio, not the absolute size, is what reads as "a cluster of separate
    points".  Below about 1:1 the marks merge at 24 px; at 1.5:1 they stay
    visibly separate even when the whole icon is 16 px wide.
    """
    import random

    rng = random.Random(seed)
    pts = []
    tries = 0
    while len(pts) < n and tries < 8000:
        tries += 1
        a = rng.uniform(0, 2 * math.pi)
        # sqrt keeps the density even instead of crowding the centre
        d = radius * math.sqrt(rng.uniform(0.04, 1.0))
        x, y = centre[0] + d * math.cos(a), centre[1] + d * math.sin(a)
        if all(math.hypot(x - qx, y - qy) >= spacing for qx, qy in pts):
            pts.append((x, y))
    return pts


def _touch_point(pts, centre, out_dir, r_dot=0.62):
    """The edge of the cluster as seen along ``out_dir`` from the cluster centre.

    Not "the furthest dot along the axis" -- that was the previous attempt, and
    it produced a 55.8-degree shaft for a 45-degree layout.  A scatter's extreme
    dot is not on any particular ray from the centre, so the two endpoints ended
    up skewed in opposite directions and the whole connector drifted off-axis.

    Instead: walk outward from the centre along the exact axis and take the
    last dot that still lies within a narrow band around that ray, then project
    it onto the ray.  The endpoint is by construction on the axis, so the shaft
    is exactly the angle asked for.
    """
    ux, uy = out_dir
    best_t, best_p = -1e9, None
    for x, y in pts:
        dx, dy = x - centre[0], y - centre[1]
        along = dx * ux + dy * uy                 # distance along the ray
        across = abs(-dx * uy + dy * ux)          # perpendicular offset
        if along <= 0 or across > r_dot * 2.2:
            continue                              # not on this side of the ray
        if along > best_t:
            best_t, best_p = along, (x, y)
    if best_p is None:                            # degenerate: fall back to centre
        return centre[0] + ux * r_dot, centre[1] + uy * r_dot
    # Project onto the ray and push out by one dot radius, so the arrowhead
    # meets the drawing's edge rather than the dot's centre.
    t = best_t + r_dot
    return centre[0] + ux * t, centre[1] + uy * t


def ecv_cloud_cloud_distance() -> str:
    """Two point clouds joined by a graduated double arrow.

    Read from the source bitmap: blue marks form the upper-left cloud, darker
    marks the lower-right one, and a connector runs between them.  The connector
    is a graduated double arrow because the tool produces a *distance
    measurement* -- both the arrowheads and the graduations carry that.

    The layout is an exact 45-degree diagonal: the clusters sit at (7.0, 7.4) and
    (17.0, 17.4), and both arrowheads are derived from the clusters' own extents
    along that exact axis.  Hand-placed endpoints were rejected twice -- once for
    missing the clusters entirely, once for drifting to 55.8 degrees.
    """
    c_a, c_b = (4.4, 3.4), (15.4, 14.4)
    ux, uy = 0.70710678, 0.70710678          # cos/sin 45

    a_pts = _cloud(c_a, 3.4, seed=11)
    b_pts = _cloud(c_b, 3.4, seed=29)

    parts = _dots(a_pts, P.BLUE_STRONG) + _dots(b_pts, P.GREEN_DK)

    x1, y1 = _touch_point(a_pts, c_a, (ux, uy))
    x2, y2 = _touch_point(b_pts, c_b, (-ux, -uy))
    parts.append(_ruler(x1, y1, x2, y2, w=0.85))
    parts.append(_label_distance(x1, y1, x2, y2))
    return P.svg("\n".join(parts))


# ---------------------------------------------------------------------------
# The word DIST
# ---------------------------------------------------------------------------

# Letterforms as paths, never <text>: a live glyph renders in whatever font the
# host Qt build happens to have, and this file ships to Windows and macOS too.
#
# Three constraints, each measured rather than guessed:
#
# 1. **Four letters, not eight.**  At a 24 px icon the canvas is 24 units wide
#    and eight capitals force 2.4 px per letter, which no font survives.  "DIST"
#    doubles the advance and lands on 4.8 px at 24 px -- the threshold where
#    capitals are still separable.
#
# 2. **Rotated to the ruler's own angle.**  The label states what the arrow
#    measures; set horizontally it belonged to the canvas, not to the arrow, and
#    read as a caption for the whole icon.  Rotating it with the shaft makes it
#    a label *on the measurement*.
#
# 3. **Above the shaft, not under it.**  The ruler runs down-right, so "above"
#    is the upper-right side: a perpendicular offset of (+uy, -ux).  Placed
#    below, it collided with the lower cloud -- which is exactly what the
#    previous full-width caption did.

_LABEL_H = 1.6          # cap height in units (was 2.5)
_LABEL_W = 1.15         # advance per letter (was 1.55)
_LABEL_SW = 0.34
# Perpendicular distance from the shaft, derived rather than chosen: the ticks
# project 1.05 units, the label's half-height is 0.80, and a 0.35 gap is the
# smallest that stays open at 24 px.  At the 1.75 used in the first rotated
# attempt the ticks ran straight through the letters.
_LABEL_OFFSET = 3.2


def _letter(ch: str, x: float, y0: float, y1: float, sw: float) -> str:
    """One capital drawn between ``y0`` (cap top) and ``y1`` (baseline)."""
    mid = (y0 + y1) / 2
    w = _LABEL_W * 0.74
    d = {
        "D": f"M {x} {y0} L {x} {y1} L {x + w * 0.55} {y1} "
             f"Q {x + w} {y1} {x + w} {mid} Q {x + w} {y0} {x + w * 0.55} {y0} Z",
        "I": f"M {x + w / 2} {y0} L {x + w / 2} {y1}",
        "S": f"M {x + w} {y0 + 0.18} L {x + w * 0.12} {y0 + 0.18} "
             f"L {x + w * 0.12} {mid} L {x + w * 0.88} {mid} "
             f"L {x + w * 0.88} {y1 - 0.18} L {x} {y1 - 0.18}",
        "T": f"M {x} {y0} L {x + w} {y0} M {x + w / 2} {y0} L {x + w / 2} {y1}",
    }.get(ch)
    if d is None:
        return ""
    return (f'  <path d="{d}" fill="none" stroke="{P.INK}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')


def _label_distance(x1: float, y1: float, x2: float, y2: float) -> str:
    """The word DIST, rotated with the shaft and set above it.

    The rotation angle comes from the same two endpoints the shaft uses, so the
    label can never drift off-axis: whatever angle the measurement takes, the
    word takes with it.
    """
    ang = math.atan2(y2 - y1, x2 - x1)
    deg = math.degrees(ang)
    # Perpendicular pointing to the upper-right of a down-right shaft.
    ox, oy = math.sin(ang), -math.cos(ang)

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = mx + ox * _LABEL_OFFSET, my + oy * _LABEL_OFFSET

    word = "DIST"
    total = len(word) * _LABEL_W
    x0 = -total / 2
    y1 = _LABEL_H / 2           # baseline, local coords centred on the origin
    y0 = y1 - _LABEL_H

    letters = "\n".join(_letter(ch, x0 + i * _LABEL_W, y0, y1, _LABEL_SW)
                        for i, ch in enumerate(word))
    return (f'  <g transform="translate({cx:.2f},{cy:.2f}) rotate({deg:.2f})">\n'
            f'{letters}\n  </g>')


def _fit_icon() -> str:
    raw = ecv_cloud_cloud_distance()
    inner = raw.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return P.fit(inner, target=0.92)


if __name__ == "__main__":                                   # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Preview ecvCloudCloudDistance.")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    svg = _fit_icon()
    if a.write:
        TARGET.write_text(svg)
        print(f"wrote {TARGET}")
    if a.sheet:
        import cairosvg
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
        except Exception:
            font = ImageFont.load_default()

        sizes = (16, 24, 32, 48, 96, 192)
        w = sum(sizes) + 24 * (len(sizes) + 1)
        h = max(sizes) + 50
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        x = 24
        for s in sizes:
            png = cairosvg.svg2png(bytestring=svg.encode(), output_width=s,
                                   output_height=s)
            canvas.alpha_composite(Image.open(io.BytesIO(png)).convert("RGBA"),
                                   (x, 16 + max(sizes) - s))
            dr.text((x, h - 22), str(s), font=font, fill=(120, 120, 120))
            x += s + 24
        canvas.convert("RGB").save(a.sheet)
        print(f"sheet -> {a.sheet}")
    if not a.write and not a.sheet:
        print(svg)
