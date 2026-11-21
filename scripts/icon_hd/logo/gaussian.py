"""toolMainGaussianFilter -- the filter that has no edge term.

What the code does, and why it decides the drawing
--------------------------------------------------
    MainWindow.cpp:12478   doActionSFGaussianFilter()
    MainWindow.cpp:12489   doActionSFBilateralFilter()

Both call the same `ccEntityAction::sfGaussianFilter`, with one difference.
At ecvEntityAction.cpp:865 the scalar-field BELL_SIGMA is only computed

    if (filterParams.filterType == ccPointCloud::RGB_FILTER_TYPES::BILATERAL)

A Gaussian filter therefore has a *spatial* weight and no range term: every
neighbour is weighted purely by how far away it is.  The consequence is exact and
is the only thing the icon can honestly show -- **a step edge does not survive.**
It is smoothed like everything else.

That single fact is what separates this button from the bilateral button beside
it. The approved bilateral icon carries a hard crease that is deliberately
preserved. This one must show the same kind of feature being softened, or the two
toolbar buttons say the same thing twice.

The original
------------
`ecvGaussianFilter.svg` is a grey disc, brightest at the centre, on a 21x21 pixel
grid: the radial profile of a Gaussian kernel seen head-on. That is a true and
useful description of the *kernel*, but it says nothing about what the kernel
does to the data, and a grey disc is visually closest to the neighbouring
bilateral icon's grey-era ancestor -- the two would look like one button drawn
twice.

Two concepts are built here.

A -- crease, before and after (chosen)
    A step edge in grey, and the same edge after filtering in blue, filled. The
    two coincide at both ends and separate at the step, so the drawing states
    the operation: *this sharp thing becomes this round thing.* Reading it needs
    no vocabulary.

B -- the kernel itself
    The source's own idea, rebuilt: a radial falloff as concentric tinted bands.
    Faithful, and it keeps the original's meaning, but it describes the tool's
    mechanism rather than its effect, and at 16 px it is a blue-grey blob.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   blues for the filtered result (#D0E5F6 / #9AC1E0 / #2896D3), the
              samples' greys for the original edge (#CDCDCD / #A4A6A6)
    fills     subtle gradients
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon gaussian --show
    python3 -m scripts.icon_hd.logo.icon gaussian --sheet out.png
"""

from __future__ import annotations

import math

from . import paraview_style as P

# Blues for the result, greys for the input.  The approved bilateral icon owns
# the greens; keeping the two filters on different hues means they stay
# distinguishable even before the shapes are read.
RESULT_HI = P.BLUE_PALE
RESULT_LO = P.BLUE_MID
RESULT_EDGE = P.BLUE_STRONG
INPUT = P.GREY_MID
INPUT_EDGE = P.GREY_DK

# The horizontal extent of the bell and of the distance axis it stands on.
X0, X1 = 1.2, 22.8

# The bell.  `SIGMA` sets its width, `BASE`/`PEAK` its height, and the curve is
# w(s) = exp(-(s-BELL_MU)^2 / 2*BELL_SIGMA^2) -- the same expression as the code's weight
# at ecvPointCloud.cpp:2317.
BELL_MU = 12.0
BELL_SIGMA = 4.8
BASE = 21.2
PEAK = 3.2
# Outline weight.  A Gaussian's filled area is only ~42 % of its own bounding
# box, so a hairline curve cannot reach the samples' 41-69 % density band; the
# stroke has to carry part of the ink.
CURVE_SW = 1.15
AXIS_SW = 1.5


# ---------------------------------------------------------------------------
# The Gaussian bell
# ---------------------------------------------------------------------------


def concept_a() -> str:
    """The kernel itself: a Gaussian bell with its one-BELL_SIGMA marks.

    This is the drawing the code licenses, exactly.  At ecvPointCloud.cpp:2317
    the weight is

        weight = exp(-d^2 / (2*BELL_SIGMA^2))

    which is a Gaussian probability density in distance.  So the icon draws a
    Gaussian bell, filled, standing on the distance axis -- not a *consequence*
    of smoothing (a smoothed step, which is what an earlier version showed and
    which reads as a generic S-curve at any size), but the kernel's own shape.

    Two dashed verticals at BELL_MU +/- BELL_SIGMA put the parameter on the drawing.  They
    are borrowed from the dialog's own note -- "3*BELL_SIGMA = 98% attenuation" -- so
    the marks are the control the user actually turns, not decoration.

    Against the neighbouring bilateral icon: this drawing has ONE bell.  The
    bilateral weight is this bell multiplied by a second bell over the value
    difference (ecvPointCloud.cpp:2322), which is exactly why its icon carries
    the preserved crease and this one carries a single profile.
    """

    def bell(samples: int = 96):
        pts = []
        for i in range(samples + 1):
            x = X0 + (X1 - X0) * i / samples
            y = BASE - (BASE - PEAK) * math.exp(
                -((x - BELL_MU) ** 2) / (2 * BELL_SIGMA * BELL_SIGMA))
            pts.append((x, y))
        return pts

    pts = bell()
    body_d = [f"M {pts[0][0]:.2f} {BASE:.2f}",
              f"L {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    body_d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
    body_d.append(f"L {pts[-1][0]:.2f} {BASE:.2f} Z")

    top_d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    top_d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]

    defs = ('<linearGradient id="gBell" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{RESULT_HI}"/>'
            f'<stop offset="1" stop-color="{RESULT_LO}"/></linearGradient>')

    parts = [
        # The baseline: the distance axis the kernel lives on.
        f'  <path d="M {X0} {BASE:.2f} L {X1} {BASE:.2f}" fill="none" '
        f'stroke="{P.INK}" stroke-width="0.8" stroke-linecap="round"/>',

        # The distribution, filled.
        f'  <path d="{" ".join(body_d)}" fill="url(#gBell)" stroke="none"/>',

        # BELL_MU +/- BELL_SIGMA: dashed, so they read as measurements rather than as
        # further outlines of the curve.
        f'  <path d="M {BELL_MU - BELL_SIGMA:.2f} {BASE - (BASE - PEAK) * 0.607:.2f} '
        f'L {BELL_MU - BELL_SIGMA:.2f} {BASE:.2f}" fill="none" stroke="{RESULT_EDGE}" '
        f'stroke-width="0.55" stroke-dasharray="1.0 0.8"/>',
        f'  <path d="M {BELL_MU + BELL_SIGMA:.2f} {BASE - (BASE - PEAK) * 0.607:.2f} '
        f'L {BELL_MU + BELL_SIGMA:.2f} {BASE:.2f}" fill="none" stroke="{RESULT_EDGE}" '
        f'stroke-width="0.55" stroke-dasharray="1.0 0.8"/>',

        # The span between them, marked once.  One measured interval, not two
        # arrows -- the pair of dashed lines already reads as the boundary.
        f'  <path d="M {BELL_MU - BELL_SIGMA:.2f} {BASE - 3.0:.2f} '
        f'L {BELL_MU + BELL_SIGMA:.2f} {BASE - 3.0:.2f}" fill="none" '
        f'stroke="{RESULT_EDGE}" stroke-width="1.1" stroke-linecap="round"/>',
        f'  <path d="M {BELL_MU - BELL_SIGMA:.2f} {BASE - 3.6:.2f} '
        f'L {BELL_MU - BELL_SIGMA:.2f} {BASE - 2.4:.2f}" fill="none" '
        f'stroke="{RESULT_EDGE}" stroke-width="0.8"/>',
        f'  <path d="M {BELL_MU + BELL_SIGMA:.2f} {BASE - 3.6:.2f} '
        f'L {BELL_MU + BELL_SIGMA:.2f} {BASE - 2.4:.2f}" fill="none" '
        f'stroke="{RESULT_EDGE}" stroke-width="0.8"/>',

        # The curve's own outline, on top of the fill.
        f'  <path d="{" ".join(top_d)}" fill="none" stroke="{P.INK}" '
        f'stroke-width="{P.STROKE}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>',
    ]
    return P.svg("\n".join(parts), defs=defs)


CONCEPTS = {"A": concept_a}

# A is the kernel's own shape.  An earlier version drew a *smoothed step* instead
# -- the consequence of filtering rather than the filter -- and it failed review
# for a good reason: at any size a step with a rounded corner reads as a generic
# S-curve, not as a Gaussian.
CONCEPT = "A"


def build() -> str:
    """The approved concept (A), raw and unfitted."""
    return CONCEPTS[CONCEPT]()
