"""ecvDeleteSF -- single-icon rework, ParaView style.

What the code does
------------------
    MainWindow.cpp:1303  actionDeleteScalarField triggers
                         clearSelectedEntitiesProperty(CURRENT_SCALAR_FIELD)
    ecvEntityAction.cpp:2307  which calls
                              pointCloud->deleteScalarField(
                                  getCurrentDisplayedScalarFieldIndex())

So it removes the **currently displayed scalar field** from every selected point
cloud -- the one field, not all of them (that is the neighbouring "delete all"
action).  Two facts follow for the drawing:

1. The subject is a *scalar field*, and a scalar field's identity in this
   application is its colour ramp.  The icon must show the ramp, or it is just a
   bin and could mean "delete anything".
2. Only one field goes, so the ramp is a single bar, not a stack.

Read from the source bitmap: a vertical rainbow bar on the left, a grey bin on
the right.  That composition is correct and is kept; what changes is that the
ramp becomes a smooth gradient instead of 1-pixel colour steps, and the bin
becomes a solid outlined form instead of a grey rectangle.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   greys #CDCDCD / #787878 for the bin, the samples' field ramp for
              the bar, #2896D3 for the arrow that says "remove"
    fills     subtle gradients
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.ecvdelsf --sheet o.png
    python3 -m scripts.icon_hd.logo.ecvdelsf --write
"""

from __future__ import annotations

import io
from pathlib import Path

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "app/Resources/images/svg/ecv/ecvDeleteSF.svg"

# The scalar-field ramp.  Ends are the samples' own saturated colours; the
# interior follows the usual blue-green-yellow-red order so it reads as a
# scalar-field legend rather than as a decorative rainbow.
RAMP = [
    (0.00, "#3B4FA8"),   # deep blue
    (0.22, "#2896D3"),   # strong blue  (pqFullscreen's action blue)
    (0.42, "#7FC24F"),   # green
    (0.60, "#E8C21A"),   # yellow
    (0.78, "#D2552A"),   # orange-red
    (1.00, "#A3245C"),   # magenta
]

BIN_HI = "#E2E2E2"
BIN_LO = "#A8A8A8"
BIN_DARK = "#7A7A7A"


def _ramp_bar(x: float, y: float, w: float, h: float) -> str:
    """The scalar field as a gradient bar.

    A real gradient, not stacked strips: at 24 px a six-strip stack shows five
    hard boundaries and reads as a colour *catalogue*, while a gradient reads as
    one continuous quantity -- which is what a scalar field is.  Gradients are
    legitimate here; the samples shade their bodies and this is vector-defined,
    so it cannot band.
    """
    return (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="1.0" '
            f'fill="url(#ramp)" stroke="{P.INK}" stroke-width="{P.STROKE}"/>')


def _bin(x: float, y: float, w: float, h: float) -> str:
    """A bin: lid, body, two ribs.  Drawn as solid outlined shapes.

    The body tapers inward and the lid overhangs it -- that overhang is the one
    feature that makes a bin read as a bin rather than as a cup, so it is kept
    even though it costs a little width at 16 px.
    """
    lid_h = h * 0.16
    body_top = y + lid_h + 0.5
    taper = w * 0.11
    parts = [
        # Body, tapered.
        f'  <path d="M {x + taper * 0.4:.2f} {body_top:.2f} '
        f'L {x + w - taper * 0.4:.2f} {body_top:.2f} '
        f'L {x + w - taper:.2f} {y + h:.2f} '
        f'L {x + taper:.2f} {y + h:.2f} Z" '
        f'fill="url(#binG)" stroke="{P.INK}" stroke-width="{P.STROKE}" '
        f'stroke-linejoin="round"/>',
        # Lid, overhanging on both sides.
        f'  <rect x="{x - 0.5:.2f}" y="{y:.2f}" width="{w + 1:.2f}" '
        f'height="{lid_h:.2f}" rx="0.5" fill="{BIN_HI}" stroke="{P.INK}" '
        f'stroke-width="{P.STROKE}"/>',
        # Two ribs, the standard shorthand for "this is a container".
        f'  <path d="M {x + w * 0.36:.2f} {body_top + (h - lid_h) * 0.22:.2f} '
        f'L {x + w * 0.36:.2f} {y + h - (h - lid_h) * 0.18:.2f}" fill="none" '
        f'stroke="{BIN_DARK}" stroke-width="0.5" stroke-linecap="round"/>',
        f'  <path d="M {x + w * 0.64:.2f} {body_top + (h - lid_h) * 0.22:.2f} '
        f'L {x + w * 0.64:.2f} {y + h - (h - lid_h) * 0.18:.2f}" fill="none" '
        f'stroke="{BIN_DARK}" stroke-width="0.5" stroke-linecap="round"/>',
    ]
    return "\n".join(parts)


def _arrow(x1: float, y: float, x2: float) -> str:
    """A short solid arrow from the ramp toward the bin: "this goes away".

    Without it the icon is a ramp next to a bin with no stated relation -- it
    could be "import ramp into bin" as easily as "delete".  The arrow is the verb
    and it points at the bin, which is the only direction that makes sense.
    """
    head = 1.5
    return "\n".join([
        f'  <path d="M {x1:.2f} {y:.2f} L {x2 - head * 0.6:.2f} {y:.2f}" '
        f'fill="none" stroke="{P.BLUE_STRONG}" stroke-width="0.95" '
        f'stroke-linecap="round"/>',
        f'  <path d="M {x2:.2f} {y:.2f} L {x2 - head:.2f} {y - head * 0.72:.2f} '
        f'L {x2 - head:.2f} {y + head * 0.72:.2f} Z" '
        f'fill="{P.BLUE_STRONG}" stroke="none"/>',
    ])


def ecv_delete_sf() -> str:
    """A scalar-field ramp being dropped into a bin."""
    defs = (
        '<linearGradient id="ramp" x1="0" y1="0" x2="0" y2="1">'
        + "".join(f'<stop offset="{o:.2f}" stop-color="{c}"/>'
                  for o, c in RAMP)
        + "</linearGradient>"
        '<linearGradient id="binG" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{BIN_LO}"/>'
        f'<stop offset="0.45" stop-color="{BIN_HI}"/>'
        f'<stop offset="1" stop-color="{BIN_LO}"/></linearGradient>'
    )

    parts = [
        _ramp_bar(2.2, 3.4, 5.2, 17.4),
        _bin(11.6, 6.0, 8.4, 14.0),
        _arrow(8.2, 12.2, 11.2),
    ]
    return P.svg("\n".join(parts), defs=defs)


def _fit_icon() -> str:
    raw = ecv_delete_sf()
    inner = raw.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return P.fit(inner, target=0.92)


if __name__ == "__main__":                                   # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Preview ecvDeleteSF.")
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
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 13)
        except Exception:
            font = ImageFont.load_default()

        sizes = (16, 24, 32, 48, 96, 192)
        w = sum(sizes) + 26 * (len(sizes) + 1)
        h = max(sizes) + 52
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        x = 26
        for s in sizes:
            png = cairosvg.svg2png(bytestring=svg.encode(), output_width=s,
                                   output_height=s)
            canvas.alpha_composite(Image.open(io.BytesIO(png)).convert("RGBA"),
                                   (x, 16 + max(sizes) - s))
            dr.text((x, h - 22), f"{s}px", font=font, fill=(120, 120, 120))
            x += s + 26
        canvas.convert("RGB").save(a.sheet)
        print(f"sheet -> {a.sheet}")
    if not a.write and not a.sheet:
        print(svg)
