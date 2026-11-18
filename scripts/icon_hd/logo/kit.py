"""Minimal drawing kit for hand-authoring the pixel-mosaic icons.

Why this file exists at all
---------------------------
The repository ships 109 files with a ``.svg`` extension that are not vector
art: each is a small bitmap re-encoded as several hundred 2-3 px ``<rect>``
elements.  Measured grid sizes are 16, 17, 21, 24, 25, 32 and 48 px, while a
toolbar icon at 18 logical px needs 54 device px on a 3x display.  Every one of
them is therefore blurry on a HiDPI screen, and no amount of retracing fixes it:
automated tracing of all 109 was run, produced visible artefacts (a green cast on
``ecvExit``, a distorted ``B`` on ``pqBold24``, a severed ``typeXCoordinate``),
and was fully reverted.

What actually works is drawing them again, one at a time, against the function
the code says they perform.  This module holds only the primitives that drawing
needs -- nothing here regenerates an existing icon on its own.

Canvas 24x24, stroke 2.0, one ink, one accent: the numbers come from the render
size.  At 18 px a 2.0 stroke lands on 1.5 device px, the thinnest width that
survives antialiasing; anything thinner disappears.
"""

from __future__ import annotations

import math

CANVAS = 24
INK = "#2C3E50"
ACCENT = "#2AA8DC"
MUTED = "#8FA3B0"
PAPER = "#FFFFFF"

_ROOT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" '
    'viewBox="0 0 24 24" fill="none" stroke="{ink}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
)


def svg(body: str, *, ink: str = INK) -> str:
    return _ROOT.format(s=CANVAS, ink=ink) + "\n" + body + "\n</svg>\n"


def line(x1, y1, x2, y2, *, color=None, width=None) -> str:
    a = f' d="M {x1} {y1} L {x2} {y2}"'
    if color:
        a += f' stroke="{color}"'
    if width:
        a += f' stroke-width="{width}"'
    return f'  <path{a}/>'


def poly(points, *, color=None, width=None, close=False, fill=None) -> str:
    d = "M " + " L ".join(f"{x} {y}" for x, y in points) + (" Z" if close else "")
    a = f' d="{d}"'
    if fill:
        a += f' fill="{fill}"'
        a += f' stroke="{color}"' if color else ' stroke="none"'
    elif color:
        a += f' stroke="{color}"'
    if width:
        a += f' stroke-width="{width}"'
    return f"  <path{a}/>"


def rect(x, y, w, h, *, r=0, color=None, fill=None, width=None) -> str:
    a = f' x="{x}" y="{y}" width="{w}" height="{h}"'
    if r:
        a += f' rx="{r}"'
    if fill:
        a += f' fill="{fill}"'
        a += f' stroke="{color}"' if color else ' stroke="none"'
    elif color:
        a += f' stroke="{color}"'
    if width:
        a += f' stroke-width="{width}"'
    return f"  <rect{a}/>"


def circle(cx, cy, r, *, color=None, fill=None, width=None) -> str:
    a = f' cx="{cx}" cy="{cy}" r="{r}"'
    if fill:
        a += f' fill="{fill}"'
        a += f' stroke="{color}"' if color else ' stroke="none"'
    elif color:
        a += f' stroke="{color}"'
    if width:
        a += f' stroke-width="{width}"'
    return f"  <circle{a}/>"


def dot(cx, cy, r=1.5, *, color=None) -> str:
    return circle(cx, cy, r, fill=color or ACCENT)


def arrow(x1, y1, x2, y2, *, color=None, head=3.4) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    c = color or INK
    return "\n".join([
        line(x1, y1, x2, y2, color=c),
        poly([(x2, y2),
              (x2 + head * math.cos(a1), y2 + head * math.sin(a1)),
              (x2 + head * math.cos(a2), y2 + head * math.sin(a2))],
             close=True, fill=c),
    ])


def plus(cx, cy, arm=5.0, *, color=None, width=None) -> str:
    c = color or INK
    return "\n".join([
        line(cx - arm, cy, cx + arm, cy, color=c, width=width),
        line(cx, cy - arm, cx, cy + arm, color=c, width=width),
    ])


def minus(cx, cy, arm=5.0, *, color=None, width=None) -> str:
    return line(cx - arm, cy, cx + arm, cy, color=color or INK, width=width)


def trash(cx, cy, *, color=None) -> str:
    c = color or INK
    return "\n".join([
        line(cx - 5.4, cy - 4.6, cx + 5.4, cy - 4.6, color=c),
        line(cx - 1.9, cy - 4.6, cx - 1.3, cy - 6.6, color=c),
        line(cx + 1.9, cy - 4.6, cx + 1.3, cy - 6.6, color=c),
        line(cx - 1.3, cy - 6.6, cx + 1.3, cy - 6.6, color=c),
        poly([(cx - 4.2, cy - 4.6), (cx - 3.4, cy + 6.4),
              (cx + 3.4, cy + 6.4), (cx + 4.2, cy - 4.6)], color=c),
        line(cx - 1.5, cy - 2.2, cx - 1.5, cy + 3.8, color=c, width=1.5),
        line(cx + 1.5, cy - 2.2, cx + 1.5, cy + 3.8, color=c, width=1.5),
    ])


def cube(cx, cy, r=7.0, *, color=None, width=None) -> str:
    c = color or INK
    top, right = (cx, cy - r), (cx + r * 0.866, cy - r * 0.5)
    br, bot = (cx + r * 0.866, cy + r * 0.5), (cx, cy + r)
    bl, tl = (cx - r * 0.866, cy + r * 0.5), (cx - r * 0.866, cy - r * 0.5)
    return "\n".join([
        poly([top, right, br, bot, bl, tl], close=True, color=c, width=width),
        poly([tl, (cx, cy), bot], color=c, width=width),
        line(cx, cy, right[0], right[1], color=c, width=width),
    ])


def check(cx, cy, s=1.0, *, color=None, width=None) -> str:
    return poly([(cx - 5.0 * s, cy + 0.4 * s), (cx - 1.4 * s, cy + 4.2 * s),
                 (cx + 5.4 * s, cy - 4.0 * s)], color=color or INK, width=width)
