"""ecvComputeStat -- single-icon rework, ParaView style.

What the code does
------------------
    MainWindow.cpp:12409  doActionComputeStatParams()
    ecvEntityAction.cpp:2519  computeStatParams() opens a "Distribution Fitting"
                              dialog offering exactly two distributions,
                              **Gauss** and **Weibull**, then fits the chosen one
                              to the currently displayed scalar field.

So the tool answers one question: *what distribution describes this field, and
what are its parameters?*  The parameters are the mean and the standard
deviation -- which is what the original icon's red "mu" and "sigma" were already
saying.

What the icon must show
-----------------------
A fitted distribution and its two parameters.  Three marks carry that, and all
three survive down to 16 px because they are geometric rather than typographic:

    the curve        a Gaussian bell, filled under the outline
    mu               the vertical line through the peak
    sigma            the horizontal double arrow spanning one standard
                     deviation either side of the peak

Letterforms for "mu" and "sigma" are included as well, but they are detail for
32 px and above -- at toolbar size the line and the arrow are what read.

Read from the source bitmap: a black L-shaped axis pair, a blue curve decaying
from the left, and red "mu sigma" at the top.  The axis pair and the curve are
kept; the letters move from floating in the corner onto the marks they name.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   #D0E5F6 / #2896D3 for the distribution, #6F6F59 ink
    fills     subtle gradients
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon stat --show
    python3 -m scripts.icon_hd.logo.icon stat --sheet out.png
    python3 -m scripts.icon_hd.logo.icon stat --write
"""

from __future__ import annotations

import math

from . import paraview_style as P


CURVE_HI = P.BLUE_PALE     # #D0E5F6
CURVE_LO = P.BLUE_MID      # #9AC1E0
CURVE_EDGE = P.BLUE_STRONG  # #2896D3

# Geometry.  The axes sit on a 3-unit inset so the stroke has room, and the bell
# is centred in the plot area.
X0, Y0 = 3.0, 20.6         # axis origin
X1, Y1 = 21.6, 3.4         # far ends of the two axes
MU = 12.0                  # curve centre
SIGMA = 3.15               # standard deviation in canvas units
PEAK = 13.0                # peak height above the baseline


def _bell_path(mu: float, sigma: float, peak: float,
               x0: float, x1: float, base: float, samples: int = 48) -> str:
    """The Gaussian, sampled finely enough to stay smooth at 192 px.

    A cubic-Bezier fit would be shorter, but the exact shape matters here -- the
    curve *is* the icon -- and a 48-point polyline is under a kilobyte.  At 24
    units, 48 samples put a vertex every 0.4 units, which is below one pixel at
    every size this icon is used at.
    """
    pts = []
    for i in range(samples + 1):
        x = x0 + (x1 - x0) * i / samples
        y = base - peak * math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))
        pts.append((x, y))
    d = [f"M {pts[0][0]:.2f} {base:.2f}", f"L {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
    d.append(f"L {pts[-1][0]:.2f} {base:.2f} Z")
    return " ".join(d)


def _bell_line(mu: float, sigma: float, peak: float,
               x0: float, x1: float, base: float, samples: int = 48) -> str:
    """Just the top edge, for the separate outline stroke."""
    pts = []
    for i in range(samples + 1):
        x = x0 + (x1 - x0) * i / samples
        y = base - peak * math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))
        pts.append((x, y))
    d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
    return " ".join(d)


def _greek_mu(x: float, y: float, size: float = 2.0) -> str:
    """Greek mu as a path: stem, bowl, descender."""
    s = size / 3.0
    d = (f"M {x:.2f} {y:.2f} "
         f"L {x:.2f} {y + 2.0 * s:.2f} "
         f"Q {x:.2f} {y + 2.9 * s:.2f} {x + 1.05 * s:.2f} {y + 2.9 * s:.2f} "
         f"Q {x + 2.1 * s:.2f} {y + 2.9 * s:.2f} {x + 2.1 * s:.2f} {y + 2.0 * s:.2f} "
         f"L {x + 2.1 * s:.2f} {y:.2f} "
         f"M {x + 2.1 * s:.2f} {y + 0.2 * s:.2f} "
         f"L {x + 2.45 * s:.2f} {y + 3.6 * s:.2f}")
    return f'  <path d="{d}" fill="none" stroke="{P.INK}" stroke-width="0.4" ' \
           f'stroke-linecap="round" stroke-linejoin="round"/>'


def _greek_sigma(x: float, y: float, size: float = 2.0) -> str:
    """Greek sigma as a path: top bar, diagonal in, flat foot."""
    s = size / 3.0
    d = (f"M {x + 2.4 * s:.2f} {y:.2f} L {x:.2f} {y:.2f} "
         f"L {x + 1.45 * s:.2f} {y + 1.5 * s:.2f} "
         f"L {x:.2f} {y + 3.0 * s:.2f} L {x + 2.4 * s:.2f} {y + 3.0 * s:.2f}")
    return f'  <path d="{d}" fill="none" stroke="{P.INK}" stroke-width="0.4" ' \
           f'stroke-linecap="round" stroke-linejoin="round"/>'


def build() -> str:
    """A fitted Gaussian over axes, with mu and sigma marked on it."""
    defs = (
        '<linearGradient id="bell" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{CURVE_HI}"/>'
        f'<stop offset="1" stop-color="{CURVE_LO}"/></linearGradient>'
    )

    bell = _bell_path(MU, SIGMA, PEAK, X0 + 0.2, X1 - 0.2, Y0, 48)
    top = _bell_line(MU, SIGMA, PEAK, X0 + 0.2, X1 - 0.2, Y0, 48)

    parts = []

    # Axes first, so the fill sits on top of the baseline.
    parts.append(f'  <path d="M {X0} {Y1} L {X0} {Y0} L {X1} {Y0}" fill="none" '
                 f'stroke="{P.INK}" stroke-width="0.85" stroke-linecap="round" '
                 f'stroke-linejoin="round"/>')

    # The distribution.
    parts.append(f'  <path d="{bell}" fill="url(#bell)" stroke="none"/>')

    # mu: the vertical line through the peak, drawn at the samples' weight so it
    # reads as the primary mark even when the letters are illegible.
    parts.append(f'  <path d="M {MU} {Y0} L {MU} {Y0 - PEAK:.2f}" fill="none" '
                 f'stroke="{CURVE_EDGE}" stroke-width="0.9" '
                 f'stroke-linecap="round"/>')

    # sigma: a double arrow spanning one standard deviation either side.
    sy = Y0 - PEAK * 0.42
    sa, sb = MU - SIGMA, MU + SIGMA
    parts.append(f'  <path d="M {sa:.2f} {sy:.2f} L {sb:.2f} {sy:.2f}" fill="none" '
                 f'stroke="{CURVE_EDGE}" stroke-width="0.7"/>')
    for ax, dx in ((sa, 1), (sb, -1)):
        parts.append(f'  <path d="M {ax:.2f} {sy:.2f} '
                     f'L {ax + dx * 1.15:.2f} {sy - 0.62:.2f} '
                     f'L {ax + dx * 1.15:.2f} {sy + 0.62:.2f} Z" '
                     f'fill="{CURVE_EDGE}" stroke="none"/>')
    # End ticks, so the arrow reads as a measured interval rather than a bar.
    for ax in (sa, sb):
        parts.append(f'  <path d="M {ax:.2f} {sy - 0.85:.2f} L {ax:.2f} {sy + 0.85:.2f}" '
                     f'fill="none" stroke="{CURVE_EDGE}" stroke-width="0.6"/>')

    # The curve's own outline, on top of the fill.
    parts.append(f'  <path d="{top}" fill="none" stroke="{P.INK}" '
                 f'stroke-width="0.85" stroke-linecap="round" '
                 f'stroke-linejoin="round"/>')

    # mu and sigma naming their own marks.  Detail at 32 px and above; at 24 px
    # the line and the arrow already carry the meaning.
    parts.append(_greek_mu(MU + 0.9, Y0 - PEAK - 2.6, 2.0))
    parts.append(_greek_sigma(MU + SIGMA + 1.4, sy - 1.4, 2.0))

    return P.svg("\n".join(parts), defs=defs)
