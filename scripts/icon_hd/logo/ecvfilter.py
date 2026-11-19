"""ecvFilterByValue -- single-icon rework, ParaView style.

What the code does
------------------
    MainWindow.cpp:12546  doActionFilterByValue()
      - requires an active scalar field on every selected entity
      - reads the entities' combined display range as (minVald, maxVald)
      - opens ccFilterByValueDlg(min, max) for the user to narrow that range
      - ccFilterByValueDlg::Mode is { EXPORT, SPLIT, CANCEL } -- EXPORT keeps
        the points inside [min, max], SPLIT also produces the complement

So the operation is: **pick a sub-interval of a scalar field's value range**.

Why the drawing is not the original's
-------------------------------------
The source shows the literal words "min" and "max" above and below a rainbow
bar.  That is honest but it fails at the size the icon is used: at 24 px each
word is around 2.5 px tall and reads as a red smudge, which leaves the icon as
"a colourful bar" and says nothing about *filtering*.

What carries the meaning instead is geometric, and geometry survives:

    a value ramp            the scalar field
    two handles             the chosen min and max
    a lit band between      the part that is kept
    dimmed ramp outside     the part that is dropped

The handles are what separate this icon from "show histogram" and "colour map":
those two are about *seeing* the field, this one is about *cutting* it.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   the field ramp plus #2896D3 for the active handles
    fills     subtle gradients
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.ecvfilter --sheet o.png
    python3 -m scripts.icon_hd.logo.ecvfilter --write
"""

from __future__ import annotations

import io
from pathlib import Path

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "app/Resources/images/svg/ecv/ecvFilterByValue.svg"

# Same ramp as ecvDeleteSF, so the two "scalar field" icons are visibly a pair.
RAMP = [
    (0.00, "#3B4FA8"),
    (0.22, "#2896D3"),
    (0.42, "#7FC24F"),
    (0.60, "#E8C21A"),
    (0.78, "#D2552A"),
    (1.00, "#A3245C"),
]

HANDLE = P.BLUE_STRONG
KEEP_EDGE = P.INK

# Split of the bar: the kept window is genuinely a sub-interval, not centred, so
# the icon does not read as a symmetric "range" slider.
BAR_X, BAR_W = 2.4, 19.2
BAR_Y, BAR_H = 9.4, 6.2
KEEP_A, KEEP_B = 6.2, 16.4          # as fractions of the bar's own width


def _ramp(*, dim_from: float | None = None, dim_to: float | None = None) -> str:
    """The value ramp as one gradient with the dropped segments veiled.

    The veil is drawn in two pieces with **asymmetric corner radii** -- rounded
    on the outside, square on the inside.  An earlier version rounded all four
    corners of each veil, and the result was three separate rounded blocks that
    read as three fields rather than as one field with a window cut out of it.
    The outer corners must match the bar; the inner edges must be square,
    because a square edge is what a cut looks like.

    Veil opacity is 0.55: at 0.72 the discarded ends became muddy grey slabs and
    the ramp stopped reading as continuous.
    """
    parts = [
        f'  <rect x="{BAR_X}" y="{BAR_Y}" width="{BAR_W}" height="{BAR_H}" '
        f'rx="1.0" fill="url(#ramp)" stroke="{KEEP_EDGE}" '
        f'stroke-width="{P.STROKE}"/>'
    ]
    if dim_from is not None:
        # Left veil: round the left corners only.
        parts.append(
            f'  <path d="M {BAR_X + 1.0:.2f} {BAR_Y:.2f} '
            f'L {dim_from:.2f} {BAR_Y:.2f} L {dim_from:.2f} {BAR_Y + BAR_H:.2f} '
            f'L {BAR_X + 1.0:.2f} {BAR_Y + BAR_H:.2f} '
            f'Q {BAR_X:.2f} {BAR_Y + BAR_H:.2f} {BAR_X:.2f} {BAR_Y + BAR_H - 1.0:.2f} '
            f'L {BAR_X:.2f} {BAR_Y + 1.0:.2f} '
            f'Q {BAR_X:.2f} {BAR_Y:.2f} {BAR_X + 1.0:.2f} {BAR_Y:.2f} Z" '
            f'fill="#8E8E8E" fill-opacity="0.55" stroke="none"/>')
    if dim_to is not None:
        # Right veil: round the right corners only.
        right = BAR_X + BAR_W
        parts.append(
            f'  <path d="M {dim_to:.2f} {BAR_Y:.2f} '
            f'L {right - 1.0:.2f} {BAR_Y:.2f} '
            f'Q {right:.2f} {BAR_Y:.2f} {right:.2f} {BAR_Y + 1.0:.2f} '
            f'L {right:.2f} {BAR_Y + BAR_H - 1.0:.2f} '
            f'Q {right:.2f} {BAR_Y + BAR_H:.2f} {right - 1.0:.2f} {BAR_Y + BAR_H:.2f} '
            f'L {dim_to:.2f} {BAR_Y + BAR_H:.2f} Z" '
            f'fill="#8E8E8E" fill-opacity="0.55" stroke="none"/>')
    return "\n".join(parts)


def _handle(x: float, *, first: bool = False) -> str:
    """A range boundary: a stem that **touches** the bar's top edge, plus a cap.

    The single constraint that decides this design is contact.  An earlier
    version floated the same bracket 1.6 units above the bar and the result was
    four unrelated glyphs hovering in space -- the eye has no way to know they
    refer to the bar.  The stem must start exactly at ``BAR_Y`` for the mark to
    read as "this position on the scale".

    The cap is horizontal and points inward on the first boundary, so the pair
    reads as one interval rather than as two independent pointers.
    """
    stem_top = BAR_Y - 1.9
    cap_w = 1.1
    inward = 1 if first else -1
    return "\n".join([
        # Stem: from the bar's edge upward, so it is visibly anchored.
        f'  <path d="M {x:.2f} {BAR_Y:.2f} L {x:.2f} {stem_top:.2f}" '
        f'fill="none" stroke="{HANDLE}" stroke-width="0.7" '
        f'stroke-linecap="round"/>',
        # Cap: turns inward, toward the kept interval.
        f'  <path d="M {x:.2f} {stem_top:.2f} L {x + inward * cap_w:.2f} {stem_top:.2f}" '
        f'fill="none" stroke="{HANDLE}" stroke-width="0.8" '
        f'stroke-linecap="round"/>',
    ])


def ecv_filter_by_value() -> str:
    """A value ramp with a chosen sub-interval lit up and the rest dimmed."""
    ka = BAR_X + BAR_W * (KEEP_A - BAR_X) / BAR_W if False else KEEP_A
    kb = KEEP_B

    defs = ('<linearGradient id="ramp" x1="0" y1="0" x2="1" y2="0">'
            + "".join(f'<stop offset="{o:.2f}" stop-color="{c}"/>'
                      for o, c in RAMP)
            + "</linearGradient>")

    return P.svg("\n".join([
        _ramp(dim_from=ka, dim_to=kb),
        _handle(ka, first=True),
        _handle(kb),
    ]), defs=defs)


def _fit_icon() -> str:
    raw = ecv_filter_by_value()
    inner = raw.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return P.fit(inner, target=0.92)


if __name__ == "__main__":                                   # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Preview ecvFilterByValue.")
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
