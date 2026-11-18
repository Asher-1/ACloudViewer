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
    python3 -m scripts.icon_hd.logo.ecvalign                # print the SVG
    python3 -m scripts.icon_hd.logo.ecvalign --sheet out.png
    python3 -m scripts.icon_hd.logo.ecvalign --write        # replace the icon
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "app/Resources/images/svg/ecv/ecvAlign.svg"

GREEN_HI = "#C2DCAE"
GREEN_LO = "#7BAA5E"
RED_HI = "#D89089"
RED_LO = "#A63A32"
ARROW = "#2896D3"


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


def ecv_align() -> str:
    """Green cloud, red cloud, and a blue arrow from one marked point to the
    other: align by picking corresponding point pairs."""
    # Six-sided forms with a deliberate, calm irregularity -- enough to read as
    # scanned data, not enough to look like a bug.
    #
    # Both are large and slightly overlapping.  Measured: with a clear gap
    # between them the icon came out at 38 % ink against the samples' 54-69 %,
    # because the whitespace between two separate objects dominates.  In
    # pqPreview the magnifier overlaps the square, and that overlap is what
    # carries the mass; the same trick applies here, and it also reads better as
    # "these two clouds are being related" rather than "two clouds".
    green = [(1.0, 12.0), (5.6, 9.4), (12.4, 10.4), (15.2, 15.2),
             (12.6, 21.6), (4.4, 22.0), (0.8, 17.6)]
    red = [(14.0, 0.8), (21.8, 1.2), (23.2, 7.6), (19.8, 12.4),
           (13.6, 11.4), (11.8, 5.2)]

    defs = (_grad("gA", GREEN_HI, GREEN_LO, 35)
            + _grad("gB", RED_HI, RED_LO, 215))

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


def _fit_icon() -> str:
    """``P.svg`` above keeps the raw 24-unit frame; the icon itself is fitted so
    its bounds span 92 % of the canvas like the samples."""
    raw = ecv_align()
    body = raw.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return P.fit(body, target=0.92)


if __name__ == "__main__":                                   # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Preview the ecvAlign rework.")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    svg = _fit_icon()
    if a.write:
        TARGET.write_text(svg)
        print(f"wrote {TARGET}")
    if a.sheet:
        import cairosvg
        from PIL import Image, ImageDraw

        sizes = (16, 24, 32, 48, 96, 192)
        gap, pad, lab = 16, 16, 22
        w = sum(sizes) + gap * (len(sizes) + 1)
        h = max(sizes) + lab + pad * 2
        canvas = Image.new("RGBA", (w, h), (250, 250, 251, 255))
        dr = ImageDraw.Draw(canvas)
        x = gap
        for s in sizes:
            png = cairosvg.svg2png(bytestring=svg.encode(),
                                   output_width=s, output_height=s)
            canvas.alpha_composite(Image.open(io.BytesIO(png)).convert("RGBA"),
                                   (x, pad + max(sizes) - s))
            dr.text((x, pad + max(sizes) + 4), f"{s}", fill=(90, 90, 90))
            x += s + gap
        canvas.convert("RGB").save(a.sheet)
        print(f"sheet -> {a.sheet}")
    if not a.write and not a.sheet:
        print(svg)
