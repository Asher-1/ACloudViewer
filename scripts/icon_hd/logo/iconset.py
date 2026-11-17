"""Tool icon design system: one grid, one stroke, one ink, one accent.

Why a system instead of 286 hand-drawings
-----------------------------------------
The existing tool icons were imported from at least three different sources and
never reconciled.  Measured over ``app/Resources/images/svg/ecv/``:

    viewBox sizes present   13, 16, 18, 21, 24, 25, 32, 48   (eight variants)
    unique fills, median    58 per icon
    unique fills, max      319  (ecvClone.svg)

At the size these are actually drawn -- the toolbar sets ``iconWidth = 18`` in
``app/ecvUIManager.cpp:1908`` -- a 319-colour gradient collapses into a smear.
The colour count is not detail, it is noise inherited from the source artwork.

So the icons are rebuilt as a system rather than traced.  Tracing was measured
first and rejected: 13 of 24 flat icons failed the fidelity gate, because fine
strokes and baked-in drop shadows do not survive contour extraction.

Design constraints, derived from the render size and not from taste
------------------------------------------------------------------
canvas      24 x 24 viewBox, so an 18 px render scales by 3/4 and lands strokes
            on whole device pixels
stroke      2.0 canvas units -> 1.5 px at 18 px, the thinnest width that stays
            legible there
caps/joins  round, because square caps break up at 1.5 px
ink         one dark slate, themed by recolouring rather than by shipping
            light/dark variants
accent      one cyan, used at most twice per icon, to mark "the thing that
            changes" -- which is what makes a toolbar icon readable in a glance
detail floor 3 canvas units -> 2.25 px at 18 px; anything finer is invisible,
            so it is not drawn
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

CANVAS = 24

INK = "#2C3E50"
ACCENT = "#2AA8DC"
ACCENT_DEEP = "#1B7FA8"
PAPER = "#FFFFFF"
MUTED = "#8FA3B0"

STROKE = 2.0
STROKE_THIN = 1.5

# Every icon is emitted with these attributes on the root group, so a theme can
# recolour the whole set by changing one CSS value or by replacing INK.
_ROOT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" '
    'viewBox="0 0 24 24" fill="none" stroke="{ink}" stroke-width="{sw}" '
    'stroke-linecap="round" stroke-linejoin="round">'
)


def svg(body: str, *, view: int = CANVAS, ink: str = INK, sw: float = STROKE) -> str:
    """Wrap an icon body in the system's root element."""
    return (
        _ROOT.format(s=view, ink=ink, sw=sw) + "\n" + body + "\n</svg>\n"
    )


# ---------------------------------------------------------------------------
# Primitives.  Everything is built from these, which is what keeps 35 icons
# looking like one family instead of 35 drawings.
# ---------------------------------------------------------------------------


def line(x1, y1, x2, y2, *, color=None, width=None) -> str:
    attrs = f' d="M {x1} {y1} L {x2} {y2}"'
    if color:
        attrs += f' stroke="{color}"'
    if width:
        attrs += f' stroke-width="{width}"'
    return f"  <path{attrs}/>"


def poly(points, *, color=None, width=None, close=False, fill=None) -> str:
    """Closed or open polyline.

    ``fill`` and ``color`` are mutually exclusive in the emitted attributes: an
    earlier version appended both and produced ``stroke="X" ... stroke="none"``,
    which is invalid XML (duplicate attribute) and made cairosvg refuse the file
    outright.  A filled shape has no stroke unless one is asked for explicitly,
    so the caller's intent is unambiguous.
    """
    d = "M " + " L ".join(f"{x} {y}" for x, y in points) + (" Z" if close else "")
    attrs = f' d="{d}"'
    if fill:
        attrs += f' fill="{fill}"'
        if color:
            attrs += f' stroke="{color}"'
        else:
            attrs += ' stroke="none"'
    elif color:
        attrs += f' stroke="{color}"'
    if width:
        attrs += f' stroke-width="{width}"'
    return f"  <path{attrs}/>"


def rect(x, y, w, h, *, r=0, color=None, fill=None, width=None) -> str:
    attrs = f' x="{x}" y="{y}" width="{w}" height="{h}"'
    if r:
        attrs += f' rx="{r}"'
    if fill:
        attrs += f' fill="{fill}"'
        attrs += f' stroke="{color}"' if color else ' stroke="none"'
    elif color:
        attrs += f' stroke="{color}"'
    if width:
        attrs += f' stroke-width="{width}"'
    return f"  <rect{attrs}/>"


def circle(cx, cy, r, *, color=None, fill=None, width=None) -> str:
    attrs = f' cx="{cx}" cy="{cy}" r="{r}"'
    if fill:
        attrs += f' fill="{fill}"'
        attrs += f' stroke="{color}"' if color else ' stroke="none"'
    elif color:
        attrs += f' stroke="{color}"'
    if width:
        attrs += f' stroke-width="{width}"'
    return f"  <circle{attrs}/>"


def dot(cx, cy, r=1.4, *, color=None) -> str:
    return circle(cx, cy, r, fill=color or ACCENT)


def arrow(x1, y1, x2, y2, *, color=None, head=3.2) -> str:
    """A line with a filled triangular head at (x2, y2)."""
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + math.radians(152), ang - math.radians(152)
    hx1, hy1 = x2 + head * math.cos(a1), y2 + head * math.sin(a1)
    hx2, hy2 = x2 + head * math.cos(a2), y2 + head * math.sin(a2)
    c = color or INK
    return "\n".join([
        line(x1, y1, x2, y2, color=c),
        poly([(x2, y2), (hx1, hy1), (hx2, hy2)], close=True, fill=c),
    ])


def plus(cx, cy, arm=4.5, *, color=None, width=None) -> str:
    c = color or INK
    return "\n".join([
        line(cx - arm, cy, cx + arm, cy, color=c, width=width),
        line(cx, cy - arm, cx, cy + arm, color=c, width=width),
    ])


def minus(cx, cy, arm=4.5, *, color=None, width=None) -> str:
    return line(cx - arm, cy, cx + arm, cy, color=color or INK, width=width)


def cloud(cx, cy, s=1.0, *, color=None, width=None, filled=False) -> str:
    """Cloud silhouette as ONE closed path.

    The first version was three circles plus a baseline, and it rendered as
    three overlapping bubbles -- at 18 px the internal overlaps read as texture
    noise rather than as a cloud.  A single outline with the lobes baked into its
    curve has no interior strokes to collide at small sizes, which is the whole
    reason it works there.
    """
    c = color or INK
    # Normalised outline spanning x -9..8.2, y -5..5.6, then scaled about (cx, cy).
    pts = [
        (-9.0, 3.0),
        (-9.0, 0.4), (-7.2, -1.4), (-4.8, -1.4),
        (-4.4, -3.6), (-2.6, -5.0), (-0.6, -5.0),
        (1.6, -5.0), (3.4, -3.6), (3.8, -1.6),
        (6.2, -1.8), (8.2, 0.0), (8.2, 2.2),
        (8.2, 4.2), (6.6, 5.6), (4.6, 5.6),
    ]
    tail = [(-8.2, 5.6), (-9.0, 4.6), (-9.0, 3.0)]
    d = [f"M {cx + pts[0][0] * s:.2f} {cy + pts[0][1] * s:.2f}"]
    i = 1
    while i + 2 < len(pts):
        a, b, e = pts[i], pts[i + 1], pts[i + 2]
        d.append(
            f"C {cx + a[0] * s:.2f} {cy + a[1] * s:.2f} "
            f"{cx + b[0] * s:.2f} {cy + b[1] * s:.2f} "
            f"{cx + e[0] * s:.2f} {cy + e[1] * s:.2f}"
        )
        i += 3
    # bottom edge, closing back to the start
    d.append(f"L {cx + tail[0][0] * s:.2f} {cy + tail[0][1] * s:.2f}")
    d.append(
        f"C {cx + tail[1][0] * s:.2f} {cy + tail[1][1] * s:.2f} "
        f"{cx + tail[2][0] * s:.2f} {cy + tail[2][1] * s:.2f} "
        f"{cx + pts[0][0] * s:.2f} {cy + pts[0][1] * s:.2f}"
    )
    d.append("Z")
    attrs = f' d="{" ".join(d)}" stroke="{c}"'
    if width:
        attrs += f' stroke-width="{width}"'
    if filled:
        attrs += f' fill="{c}" stroke="none"'
    return f"  <path{attrs}/>"


def cube(cx, cy, r=6.0, *, color=None, width=None) -> str:
    """Isometric cube outline -- the shared mark for mesh/geometry."""
    c = color or INK
    top = (cx, cy - r)
    right = (cx + r * 0.866, cy - r * 0.5)
    br = (cx + r * 0.866, cy + r * 0.5)
    bot = (cx, cy + r)
    bl = (cx - r * 0.866, cy + r * 0.5)
    tl = (cx - r * 0.866, cy - r * 0.5)
    return "\n".join([
        poly([top, right, br, bot, bl, tl], close=True, color=c, width=width),
        poly([tl, (cx, cy), bot], color=c, width=width),
        line(cx, cy, right[0], right[1], color=c, width=width),
    ])


def grid3(cx, cy, r=5.0, *, color=None, width=None) -> str:
    """2x2 field grid: the shared mark for scalar fields and rasters."""
    c = color or INK
    half = r / 2
    parts = [
        rect(cx - r, cy - r, 2 * r, 2 * r, color=c, width=width, r=1),
        line(cx - r, cy - half, cx + r, cy - half, color=c, width=width),
        line(cx - r, cy + half, cx + r, cy + half, color=c, width=width),
        line(cx - half, cy - r, cx - half, cy + r, color=c, width=width),
        line(cx + half, cy - r, cx + half, cy + r, color=c, width=width),
    ]
    return "\n".join(parts)


def funnel(cx, cy, w=9.0, h=9.0, *, color=None) -> str:
    c = color or INK
    return poly(
        [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
         (cx + 1.8, cy + h * 0.12), (cx + 1.8, cy + h / 2),
         (cx - 1.8, cy + h / 2), (cx - 1.8, cy + h * 0.12)],
        close=True, color=c,
    )


def bars(cx, cy, *, color=None, heights=(3, 7, 5, 9)) -> str:
    """Histogram bars: shared mark for statistics and histograms."""
    c = color or INK
    n = len(heights)
    gap = 1.0
    bw = (9.0 - gap * (n - 1)) / n
    x0 = cx - 4.5
    parts = [line(x0 - 1.0, cy + 5.0, x0 + 9.6, cy + 5.0, color=c)]
    for i, h in enumerate(heights):
        x = x0 + i * (bw + gap)
        parts.append(rect(x, cy + 5.0 - h, bw, h, r=0.6, color=c, width=1.4))
    return "\n".join(parts)


def bell(cx, cy, *, color=None) -> str:
    """Gaussian curve: shared mark for any smoothing/blur operation."""
    c = color or INK
    d = (f"M {cx - 9} {cy + 4} C {cx - 5} {cy + 4}, {cx - 4} {cy - 6}, "
         f"{cx} {cy - 6} C {cx + 4} {cy - 6}, {cx + 5} {cy + 4}, {cx + 9} {cy + 4}")
    return f'  <path d="{d}" stroke="{c}"/>'


def magnifier(cx, cy, r=4.6, *, color=None) -> str:
    c = color or INK
    return "\n".join([
        circle(cx, cy, r, color=c),
        line(cx + r * 0.72, cy + r * 0.72, cx + r + 3.2, cy + r + 3.2, color=c),
    ])


def trash(cx, cy, *, color=None) -> str:
    c = color or INK
    return "\n".join([
        line(cx - 5.0, cy - 4.4, cx + 5.0, cy - 4.4, color=c),
        line(cx - 1.8, cy - 4.4, cx - 1.2, cy - 6.2, color=c),
        line(cx + 1.8, cy - 4.4, cx + 1.2, cy - 6.2, color=c),
        line(cx - 1.2, cy - 6.2, cx + 1.2, cy - 6.2, color=c),
        poly([(cx - 4.0, cy - 4.4), (cx - 3.2, cy + 6.0), (cx + 3.2, cy + 6.0),
              (cx + 4.0, cy - 4.4)], color=c),
        line(cx - 1.4, cy - 2.2, cx - 1.4, cy + 3.8, color=c, width=1.4),
        line(cx + 1.4, cy - 2.2, cx + 1.4, cy + 3.8, color=c, width=1.4),
    ])


def sphere_points(cx, cy, r=6.0, n=7, *, color=None) -> str:
    """A dotted sphere: point cloud as a volume, used for cloud operations."""
    import math

    c = color or INK
    golden = math.pi * (3 - math.sqrt(5))
    parts = [circle(cx, cy, r, color=c, width=1.4)]
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = math.sqrt(max(0.0, 1 - z * z)) * r
        th = golden * i
        parts.append(dot(cx + rr * math.cos(th), cy + rr * math.sin(th), 1.1, color=c))
    return "\n".join(parts)


def terminal(cx, cy, *, color=None) -> str:
    c = color or INK
    return "\n".join([
        rect(cx - 8.0, cy - 6.0, 16.0, 12.0, r=2.0, color=c),
        line(cx - 8.0, cy - 2.4, cx + 8.0, cy - 2.4, color=c, width=1.4),
        line(cx - 5.4, cy + 1.0, cx - 3.0, cy + 3.0, color=c, width=1.6),
        line(cx - 3.0, cy + 3.0, cx - 5.4, cy + 5.0, color=c, width=1.6),
        line(cx - 0.6, cy + 5.0, cx + 3.4, cy + 5.0, color=c, width=1.6),
    ])
