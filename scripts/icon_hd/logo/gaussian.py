"""toolMainGaussianFilter -- the filter that has no edge term.

What the code does, and why it decides the drawing
--------------------------------------------------
    MainWindow.cpp:12478   doActionSFGaussianFilter()
    MainWindow.cpp:12489   doActionSFBilateralFilter()

Both call the same `ccEntityAction::sfGaussianFilter`, with one difference.
At ecvEntityAction.cpp:865 the scalar-field sigma is only computed

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

# Shared profile geometry.  The step sits slightly left of centre so the two
# terraces are unequal, which reads as a real cross-section rather than as a
# symmetrical glyph.
X0, X1 = 1.5, 22.5
BASE = 21.8
LOW_Y = 18.2
HIGH_Y = 7.6
STEP_X = 11.2


# ---------------------------------------------------------------------------
# A -- the same edge, before and after
# ---------------------------------------------------------------------------


def _smooth_profile(samples: int = 64) -> list[tuple[float, float]]:
    """A smooth step at ``STEP_X`` between the two terraces.

    A tanh is the right curve here rather than a Bezier: it is the exact
    continuous limit of convolving a step with a Gaussian, which is literally
    what the filter computes.  The width is set so the transition spans about a
    third of the canvas -- wide enough to be unmistakable at 24 px, narrow enough
    that both terraces are still flat over a readable run.
    """
    out = []
    width = 2.6
    for i in range(samples + 1):
        x = X0 + (X1 - X0) * i / samples
        t = math.tanh((x - STEP_X) / width)      # -1 .. +1
        y = (LOW_Y + HIGH_Y) / 2 - (LOW_Y - HIGH_Y) / 2 * t
        out.append((x, y))
    return out


def _step_profile() -> list[tuple[float, float]]:
    """The original edge: two flat terraces and one vertical jump."""
    return [(X0, LOW_Y), (STEP_X, LOW_Y), (STEP_X, HIGH_Y), (X1, HIGH_Y)]


def concept_a() -> str:
    """Grey step edge behind, blue smoothed edge in front, both filled.

    The fill is the point.  A pair of bare lines at this scale reads as a graph
    decoration; a filled body with a contrasting edge reads as a *surface that
    changed shape*, which is the operation.  The grey step is drawn first and is
    visible only above the smoothed curve -- exactly where the sharp corner was
    and no longer is.
    """
    smooth = _smooth_profile()
    step = _step_profile()

    def poly(pts, close_at_base=True):
        d = [f"M {pts[0][0]:.2f} {BASE:.2f}", f"L {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
        if close_at_base:
            d.append(f"L {pts[-1][0]:.2f} {BASE:.2f}")
        d.append("Z")
        return " ".join(d)

    def line(pts):
        d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
        return " ".join(d)

    def closed_step(pts):
        """The step's *whole* profile: down both ends and along the baseline.

        Drawing only the top edge left two loose vertical stubs floating over the
        curve.  The original shape has to close against the same baseline the
        filtered body uses, so the ghost reads as "the shape that used to be
        here" rather than as debris.
        """
        d = [f"M {pts[0][0]:.2f} {BASE:.2f}",
             f"L {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
        d.append(f"L {pts[-1][0]:.2f} {BASE:.2f}")
        d.append("Z")
        return " ".join(d)

    defs = (
        '<linearGradient id="gOut" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{RESULT_HI}"/>'
        f'<stop offset="1" stop-color="{RESULT_LO}"/></linearGradient>'
        '<linearGradient id="gIn" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{P.GREY_LT}"/>'
        f'<stop offset="1" stop-color="{INPUT}"/></linearGradient>'
    )

    body = "\n".join([
        # The filtered result, filled and outlined.
        f'  <path d="{poly(smooth)}" fill="url(#gOut)" stroke="{P.INK}" '
        f'stroke-width="{P.STROKE}" stroke-linejoin="round"/>',

        # The original step, as a thin marker: two verticals at the jump and a
        # dash across the top terrace.  Drawing its whole outline (v3) meant the
        # part below the curve was hidden and only loose fragments showed above
        # it, which read as debris.  Three intentional segments cannot fragment.
        f'  <path d="M {STEP_X:.2f} {LOW_Y + 1.2:.2f} L {STEP_X:.2f} '
        f'{HIGH_Y - 1.4:.2f}" fill="none" stroke="{INPUT_EDGE}" '
        f'stroke-width="1.1" stroke-linecap="round"/>',

        # The smoothed edge itself, the loudest line in the drawing.
        f'  <path d="{line(smooth)}" fill="none" stroke="{RESULT_EDGE}" '
        f'stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"/>',
    ])
    return P.svg(body, defs=defs)


# ---------------------------------------------------------------------------
# B -- the kernel, as the source drew it
# ---------------------------------------------------------------------------


def concept_b() -> str:
    """Concentric bands with a radial falloff: the Gaussian kernel head-on.

    Faithful to the original's idea.  Each band is one standard deviation wide,
    echoing the dialog's own note that *3 sigma = 98 % attenuation* -- so the
    bands are not decoration, they mark where the weight has dropped to ~61 %,
    ~14 %, ~1 %.
    """
    cx, cy, r = 12.0, 12.0, 10.2
    stops = [
        (0.00, P.BLUE_PALE, 1.00),
        (0.34, P.BLUE_MID, 0.72),
        (0.68, RESULT_EDGE, 0.42),
        (1.00, P.GREY_DK, 0.20),
    ]
    defs = ('<radialGradient id="kern" cx="0.5" cy="0.5" r="0.5">'
            + "".join(f'<stop offset="{o:.2f}" stop-color="{c}" '
                      f'stop-opacity="{a:.2f}"/>' for o, c, a in stops)
            + "</radialGradient>")

    rings = []
    for frac, sw in ((1.0, P.STROKE), (0.66, 0.5), (0.33, 0.5)):
        rings.append(f'  <circle cx="{cx}" cy="{cy}" r="{r * frac:.2f}" '
                     f'fill="none" stroke="{P.INK}" stroke-width="{sw}" '
                     f'stroke-opacity="{"0.85" if frac == 1.0 else "0.35"}"/>')

    body = "\n".join([
        f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#kern)" '
        f'stroke="none"/>',
        *rings,
    ])
    return P.svg(body, defs=defs)


CONCEPTS = {"A": concept_a, "B": concept_b}

# A is chosen: it shows what the filter does, not what its kernel looks like,
# and it cannot be confused with the bilateral button beside it.
CONCEPT = "A"


def build() -> str:
    """The approved concept (A), raw and unfitted."""
    return CONCEPTS[CONCEPT]()
