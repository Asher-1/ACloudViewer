"""toolMainSORFilter -- statistical outlier removal.

What the code does
------------------
    MainWindow.cpp:10523          doActionSORFilter()
    Core/src/CloudSamplingTools.cpp:464   sorFilter(cloud, knn, nSigma)

Two passes, and the icon should describe both because together they *are* the
method:

    1.  for every point, the mean distance to its `knn` nearest neighbours
        (default 6)
    2.  across the whole cloud, the mean and standard deviation of those
        per-point means; any point whose mean exceeds

            avgDist + nSigma * stdDev          (nSigma defaults to 1.0)

        is dropped.

So it is not a smoothing filter and not a radius cut.  It is *a statistical
threshold on local density*: points sitting far from their neighbours relative
to the cloud's own distribution go away, and everything normal stays exactly
where it was.  Nothing is moved or averaged -- the surviving cloud is a subset.

Why the original is not a reference
-----------------------------------
`toolMainSORFilter.svg` is the three letters **S O R** as a 21x21 pixel mosaic,
with four small green marks scattered above them.  There is no drawing of the
operation at all: the marks are decoration and the letters are an abbreviation.

At 16 px -- the size a toolbar actually uses -- three black letterforms blur into
a single dark smudge, so it does not even work as a label.  The tool's own
behaviour has a clear, drawable idea, so the letters are replaced with it.

The drawing
-----------
A field of points with a few of them marked as *outside the band*, plus the
statistical band itself:

    the cluster      evenly scattered dots, the cloud at its normal density
    the outliers     3 dots set well clear of the cluster, drawn in red
    the band         a dashed closed contour around the cluster, labelled by its
                     own shape -- the threshold the algorithm computes

The dotted contour is the load-bearing mark: it says *"anything beyond here is
measured to be abnormal"*, which is exactly what avgDist + nSigma*stdDev is.
Without it the icon is "dots, some of which are red" and could mean anything.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   #2896D3 for the normal points (the approved cloud colour),
              #C0392B for the removed ones, #A4A6A6 dashed for the band
    dot ratio gap:size = 1.5:1, the same constant as the approved cloud icons,
              so this reads as the same family
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon sor --show
    python3 -m scripts.icon_hd.logo.icon sor --sheet out.png
"""

from __future__ import annotations

import math
import random

from . import paraview_style as P

KEEP = P.BLUE_MID         # normal points: light fill, dark outline
DROP = P.RED              # points the statistical test removes
BAND = P.GREY_MID         # the threshold contour
BAND_SW = 0.85

# Same dot geometry as the approved cloud icons: a 1.5:1 gap-to-dot ratio is what
# keeps a scatter reading as separate points rather than a blob.
R_DOT = 0.72
SPACING = 2.55

CLUSTER_C = (10.4, 13.4)
CLUSTER_R = 5.6

# The threshold contour: an ellipse around the cluster, slightly irregular so it
# reads as *computed from the data* rather than as a geometric decoration.
BAND_C = (10.6, 13.2)
BAND_RX, BAND_RY = 8.5, 7.4


def _cluster() -> list[tuple[float, float]]:
    """The normal points: evenly spread inside the band, wide gaps."""
    rng = random.Random(11)
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < 9 and tries < 9000:
        tries += 1
        a = rng.uniform(0, 2 * math.pi)
        d = CLUSTER_R * math.sqrt(rng.uniform(0.05, 1.0))
        x = CLUSTER_C[0] + d * math.cos(a)
        y = CLUSTER_C[1] + d * 0.86 * math.sin(a)
        if all(math.hypot(x - qx, y - qy) >= SPACING for qx, qy in pts):
            pts.append((x, y))
    return pts


def _outliers() -> list[tuple[float, float]]:
    """Three points well clear of the band, at different bearing and distance.

    Placed by hand rather than by the same rejection loop: a random draw kept
    landing one of them close enough to the band that the icon read as a
    *density* gradient instead of as a binary in/out decision.
    """
    return [(20.6, 6.2), (4.0, 5.6), (20.2, 19.4)]


def _dots(pts, colour, r=R_DOT) -> list[str]:
    """Kept points: solid, outlined the samples' way.

    No inner stroke on the fill -- at 0.4 units an outline darkens the mark into
    a ring and the whole cluster reads as a field of small eyes. The outline is
    on the fill colour itself instead, at the icon's own weight.
    """
    return [f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{colour}" '
            f'stroke="{P.INK}" stroke-width="0.34"/>'
            for x, y in pts]


def _removed(pts, r=R_DOT * 1.5) -> list[str]:
    """Removed points: an X, not a red dot.

    Colour alone could not carry the meaning -- a red dot beside blue dots reads
    as "a differently coloured point", which is what outliers are, not what the
    tool does to them. An X is the only mark here that states an *action*, and it
    survives at 16 px where a subtle hue shift does not.
    """
    out = []
    for x, y in pts:
        d = (f"M {x - r:.2f} {y - r:.2f} L {x + r:.2f} {y + r:.2f} "
             f"M {x + r:.2f} {y - r:.2f} L {x - r:.2f} {y + r:.2f}")
        out.append(f'  <path d="{d}" fill="none" stroke="{DROP}" '
                   f'stroke-width="1.0" stroke-linecap="butt"/>')
    return out


def _band() -> str:
    """The threshold contour, dashed.

    Dashed on purpose: a solid ellipse would read as a container or a boundary of
    the data, but the band is *computed from* the data -- it moves when knn or
    nSigma changes.  A dashed line is the conventional mark for a derived
    threshold, and it also lets the cluster show through it.
    """
    cx, cy = BAND_C
    # Four-point cubic approximation of an ellipse; no reason to sample finer at
    # this scale, and the four control points keep the shape perfectly regular.
    k = 0.5523
    d = (f"M {cx - BAND_RX:.2f} {cy:.2f} "
         f"C {cx - BAND_RX:.2f} {cy - BAND_RY * k:.2f} "
         f"{cx - BAND_RX * k:.2f} {cy - BAND_RY:.2f} {cx:.2f} {cy - BAND_RY:.2f} "
         f"C {cx + BAND_RX * k:.2f} {cy - BAND_RY:.2f} "
         f"{cx + BAND_RX:.2f} {cy - BAND_RY * k:.2f} {cx + BAND_RX:.2f} {cy:.2f} "
         f"C {cx + BAND_RX:.2f} {cy + BAND_RY * k:.2f} "
         f"{cx + BAND_RX * k:.2f} {cy + BAND_RY:.2f} {cx:.2f} {cy + BAND_RY:.2f} "
         f"C {cx - BAND_RX * k:.2f} {cy + BAND_RY:.2f} "
         f"{cx - BAND_RX:.2f} {cy + BAND_RY * k:.2f} {cx - BAND_RX:.2f} {cy:.2f} Z")
    return (f'  <path d="{d}" fill="none" stroke="{BAND}" '
            f'stroke-width="{BAND_SW}" stroke-dasharray="1.5 1.25"/>')


def build() -> str:
    """A normal cluster inside a computed threshold, three points outside it.

    The three states are what the algorithm distinguishes: inside the band and
    kept (blue), outside and removed (red), and the band itself (dashed, derived).
    Nothing else is drawn, because nothing else happens -- the surviving points
    are the same points, in the same places.
    """
    parts = [
        _band(),
        *_dots(_cluster(), KEEP),
        # The removed points are drawn slightly larger.  At 16 px a 0.4-unit dot
        # is barely two pixels, and the whole icon's message is carried by the
        # contrast between kept and removed.
        *_removed(_outliers()),
    ]
    return P.svg("\n".join(parts))
