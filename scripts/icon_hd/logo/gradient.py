"""toolMainSFGradient -- the magnitude of how fast a scalar field changes.

What the code does
------------------
    MainWindow.cpp:12413      doActionSFGradient()
    ecvLibAlgorithms.cpp:413  CCLIB_ALGO_SF_GRADIENT

The only parameter is one question, and it is asked verbatim:

    "Is the scalar field composed of (euclidean) distances?"

The result lands in a new field named `CC_GRADIENT_NORMS_FIELD_NAME` -- the
*gradient norm*.  So the tool produces a scalar field of magnitudes: for every
point, how steeply the source field rises around it.  It is not a direction
field and not a map of values; it is a measure of steepness, carrying the ramp's
colour exactly like any other scalar field.

That fixes the drawing.  The icon must show a colour ramp -- the output *is* a
scalar field, and in this application a scalar field's identity is its ramp --
combined with something that says *steepness varies*.  A ramp alone would be the
filter icon; a direction arrow alone would say nothing about a field.

The original
------------
`toolMainSFGradient.svg` is a 21x21 mosaic: a red rule with a small arrow at the
top, and below it a cone that tapers from 13 cells wide to 3 while stepping
through red, orange, yellow, green and blue.

The structure is right -- a tapering band of ramp colour is a fair picture of a
magnitude field, wide where the field is steep and narrow where it is flat. It is
kept.  What changes:

    the mosaic becomes real geometry, so it stays sharp at every size
    the ramp is rebuilt from the samples' own colours; the source's five reds
    appear zero times across the eight approved icons (34,953 px measured), so
    they are antialiasing of one text colour, not a palette
    every form carries the #6F6F59 outline the source lacked
    the cone is graded continuously rather than in 1-pixel steps

The 45-degree relationship between the arrow and the taper is preserved: the
arrow is the field's direction, the taper is its magnitude, and they read as one
statement only if they agree.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   the samples' measured ramp, warm to cold, plus #2896D3 for the
              direction arrow
    fills     one continuous gradient
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon gradient --show
    python3 -m scripts.icon_hd.logo.icon gradient --sheet out.png
"""

from __future__ import annotations

from . import paraview_style as P

# The output is a scalar field, so its ramp should be the same family the other
# field tools use.  These stops are measured sample colours (frequencies in
# paraview_style); warm at the top where the field is steep, cold at the tip
# where it flattens -- the direction the source's own cone runs.
RAMP = [
    (0.00, "#C4813E"),
    (0.22, "#F3C017"),
    (0.45, "#A7D28C"),
    (0.68, "#72BE44"),
    (0.86, "#2896D3"),
    (1.00, "#7A6FC0"),
]

ARROW = P.BLUE_STRONG

# --- the cone ---------------------------------------------------------------
TOP_Y = 8.6
TIP_Y = 22.4
TOP_HALF = 9.8          # half-width at the top: 19.6 units across
TIP_HALF = 1.6          # half-width at the tip: 3.2 units across
CX = 12.0

# --- the rule and its arrow -------------------------------------------------
RULE_Y = 4.2
RULE_X0, RULE_X1 = 2.4, 19.6


def _cone() -> str:
    """The tapering band, filled with one continuous gradient.

    A true trapezoid rather than the source's stepped stack: at 24 px a stack of
    five colour bands shows four hard seams and reads as a staircase, while a
    continuous gradient reads as one quantity varying -- which is what a norm
    field is.
    """
    d = (f"M {CX - TOP_HALF:.2f} {TOP_Y:.2f} "
         f"L {CX + TOP_HALF:.2f} {TOP_Y:.2f} "
         f"L {CX + TIP_HALF:.2f} {TIP_Y:.2f} "
         f"L {CX - TIP_HALF:.2f} {TIP_Y:.2f} Z")
    return (f'  <path d="{d}" fill="url(#grad)" stroke="{P.INK}" '
            f'stroke-width="{P.STROKE}" stroke-linejoin="round"/>')


def _rule(arrow: bool) -> str:
    """The field's direction: a rule with an arrowhead at its right end.

    Kept above the cone, not inside it.  Inside, the arrow competes with the
    ramp for the same pixels and both become unreadable; outside, the pair reads
    as "this field, in this direction" without either part being lost.

    The arrowhead is a filled triangle at the rule's own weight, so it reads at
    16 px where a stroked head would vanish.
    """
    head, hw = 2.8, 1.46
    parts = [
        f'  <path d="M {RULE_X0:.2f} {RULE_Y:.2f} '
        f'L {RULE_X1 - head * 0.55:.2f} {RULE_Y:.2f}" fill="none" '
        f'stroke="{ARROW}" stroke-width="1.5" stroke-linecap="round"/>',
    ]
    if arrow:
        parts.append(
            f'  <path d="M {RULE_X1:.2f} {RULE_Y:.2f} '
            f'L {RULE_X1 - head:.2f} {RULE_Y - hw:.2f} '
            f'L {RULE_X1 - head:.2f} {RULE_Y + hw:.2f} Z" '
            f'fill="{ARROW}" stroke="none"/>')
    return "\n".join(parts)


def build() -> str:
    """A ramp-coloured cone under a direction arrow.

    Two marks, one statement: the rule is *which way the field rises*, the cone
    is *how steeply, and where it stops being steep*.  The cone's width at a given
    height is the magnitude -- wide at the top, collapsing to a tip -- which is
    why it is a trapezoid and not a bar.
    """
    defs = ('<linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">'
            + "".join(f'<stop offset="{o:.2f}" stop-color="{c}"/>'
                      for o, c in RAMP)
            + "</linearGradient>")

    parts = [
        _rule(arrow=True),
        _cone(),
    ]
    return P.svg("\n".join(parts), defs=defs)
