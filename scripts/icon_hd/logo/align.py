"""ecvAlign -- single-icon rework, ParaView style.

Why this file exists
--------------------
`ecvAlign.svg` as committed is a 21x21 pixel mosaic: a *hundred* `<rect
width="1" height="1">` elements plus polygons sampled from the original bitmap.
There is no vector original to preserve, so "follow the original's details" was
never achievable here -- the detail *is* the pixel grid.  What can be matched is
the construction method of the eight approved ParaView samples.

What the sample measures
------------------------
Read off pqPreview.png at 6x zoom, then confirmed numerically in
`paraview_style.py`:

    outline      #6F6F59, 2 px at 64 px, on **every** shape including interior
                 ones.  In pqPreview the square, the lens barrel, the glass and
                 the handle all carry the same outline.  This is the strongest
                 single signal in the whole set.
    fill         flat colour plus a subtle gradient -- enough volume to read as
                 an object, not enough to become an airbrush.
    form         two large simple objects with rounded joins.  Nothing small
                 enough to vanish at 16 px.
    scale        the pair fills 92 % of the box.
    palette      sage green #A7D28C, neutral grey #CDCDCD / #787878.

The drawing
-----------
Two entities and one relation, which is what "align two clouds by picking point
pairs" means:

    green cloud   bottom left, large     the cloud being moved
    red cloud     top right, smaller     the cloud it is aligned to
    blue arrow    dot to dot             the correspondence being established

Drawing the arrow **between two marked points** rather than as a free curve is
what fixes the earlier version: a curve with no anchors read as a hook, and its
head disappeared inside the red form.  Two outlined dots give the arrow a start
and an end, and they say "point pair" outright, which is the tool's actual name.

Usage
-----
    python3 -m scripts.icon_hd.logo.icon align --show
    python3 -m scripts.icon_hd.logo.icon align --sheet out.png
    python3 -m scripts.icon_hd.logo.icon align --write
"""

from __future__ import annotations

import math

from . import paraview_style as P


GREEN_HI = P.GREEN_LT      # #A7D28C -- pqPreview's dominant fill
GREEN_LO = P.GREEN         # #72BE44 -- pqIsosurface's saturated green
TAN_HI = "#D8C081"         # pqPaste's tan (#C1A561) lifted one step, because at
TAN_LO = P.TAN             # full value it read as muddy olive beside the green
ARROW = P.BLUE_STRONG      # #2896D3 -- pqFullscreen's action blue


def _grad(gid: str, hi: str, lo: str, angle: float) -> str:
    """Two-stop linear gradient on an arbitrary axis in object-bounding-box
    units.  Neighbouring forms get different angles so the set does not look
    stamped from one template."""
    a = math.radians(angle)
    dx, dy = math.cos(a) * 0.5, math.sin(a) * 0.5
    return (f'<linearGradient id="{gid}" x1="{0.5 - dx:.3f}" y1="{0.5 - dy:.3f}" '
            f'x2="{0.5 + dx:.3f}" y2="{0.5 + dy:.3f}">'
            f'<stop offset="0" stop-color="{hi}"/>'
            f'<stop offset="1" stop-color="{lo}"/></linearGradient>')


def _form(points, gid: str) -> str:
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"
    return f'  <path d="{d}" fill="url(#{gid})" stroke="{P.INK}" ' \
           f'stroke-width="{P.STROKE}" stroke-linejoin="round"/>'


def build() -> str:
    """Green cloud, red cloud, and a blue arrow from one marked point to the
    other: align by picking corresponding point pairs."""
    # Two forms of *equal* size with a clear gap between them.
    #
    # Measured on the original: the green form spans 39.2 x 38.3 % of the canvas
    # and the red form 43.3 x 38.3 % -- they are the same size.  The previous
    # version drew the green one at 75.8 x 69.2 % and the red at 22.5 x 19.2 %,
    # which inverted both the size and the hierarchy.  At 24 units, 40 % of the
    # canvas is 9.6 units, so each form is a ~9.5-unit blob in opposite corners
    # with roughly 2.5 units of clear space between them.
    green = [(1.6, 14.6), (4.6, 12.8), (8.6, 13.2), (10.6, 16.6),
             (9.4, 20.8), (5.0, 22.2), (1.8, 19.8)]
    red = [(13.6, 2.0), (17.8, 1.6), (21.8, 3.4), (22.2, 7.6),
           (19.4, 10.4), (15.2, 10.0), (13.2, 6.4)]

    defs = (_grad("gA", GREEN_HI, GREEN_LO, 35)
            + _grad("gB", TAN_HI, TAN_LO, 215))

    parts = [_form(green, "gA"), _form(red, "gB")]

    # Point pair: one outlined dot on each cloud, joined by the arrow.
    #
    # The arrow is drawn **before** the dots and stops short of the target dot,
    # so the head stays visible.  An earlier version drew the head underneath
    # the dot and it vanished at 24 px, which made the mark read as a plain
    # connector instead of a directed correspondence.
    px, py = 8.6, 15.6       # anchor on the green cloud
    qx, qy = 17.8, 7.2       # anchor on the red cloud
    hx, hy = 16.2, 8.6       # head tip, short of the dot
    parts.append(P.stroke(f"M {px} {py} C 12.8 16.2, 16.2 12.0, {hx} {hy}",
                          colour=ARROW, sw=2.1))
    ang = math.atan2(hy - 12.0, hx - 16.2)
    head = 2.7
    parts.append(P.poly([(hx, hy),
                         (hx - head * math.cos(ang - 0.6),
                          hy - head * math.sin(ang - 0.6)),
                         (hx - head * math.cos(ang + 0.6),
                          hy - head * math.sin(ang + 0.6))],
                        ARROW, outline=None))
    for cx, cy in ((px, py), (qx, qy)):
        parts.append(P.circle(cx, cy, 1.4, ARROW))

    inner = "\n".join(parts)
    return P.svg(inner, defs=defs)
