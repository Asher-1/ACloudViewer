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
R_DOT = 0.66
SPACING = 2.3

# The layout is split: the point cloud above, the label below.  The original
# gives the letters the bottom 54 % of the canvas; here they get the bottom third
# because the cloud has to stay recognisable at the same time.
CLUSTER_C = (12.0, 8.4)
CLUSTER_R = 5.0

BAND_C = (12.0, 9.4)
BAND_RX, BAND_RY = 7.6, 8.4

# --- the label --------------------------------------------------------------
#
# The original icon's letters are its content, not decoration: "SOR" is how the
# user tells this filter from the five other filter buttons in the same toolbar.
# They are therefore reproduced, as paths (never <text>, which depends on the
# host having the font).
#
# Measured from the source: the three letters span 91 % of the canvas width and
# 54 % of its height, i.e. roughly 6 units tall on a 21-unit canvas.  Here they
# are 5.2 on 24 -- large enough to read at 24 px, small enough to leave the cloud
# most of the space.
LETTER_H = 4.6
LETTER_W = 4.0
LETTER_GAP = 1.5
LETTER_SW = 1.2
# 27.8 % of the cloud's height, measured.  The first attempt left a 3.40u gap
# between the ring's bottom and the letters' top -- almost exactly a quarter of
# the cloud -- which read as loose rather than composed.  The label now rides up
# into the ring's lower arc instead of sitting under it.
LABEL_Y = 18.5                    # baseline
LABEL_W = 3 * LETTER_W + 2 * LETTER_GAP


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
    return [(19.8, 4.6), (4.2, 5.2), (20.2, 11.6)]


# ---------------------------------------------------------------------------
# letterforms
# ---------------------------------------------------------------------------


def _letter(ch: str, x: float, yb: float, *, h: float = LETTER_H,
            w: float = LETTER_W, sw: float = LETTER_SW) -> str:
    """One capital, drawn as a stroked path.

    Capitals because the source's label is capitalised, and at 5 units tall a
    lowercase letter's counters close at toolbar size.  Each is built from arcs
    and straight runs rather than a rounded rectangle, so S keeps its double
    curve and R keeps its leg -- the two features that separate them from O at a
    glance.
    """
    yt = yb - h
    r = w * 0.5
    d = {
        # S: upper bowl opens right, lower bowl opens left.
        "S": (f"M {x + w:.2f} {yt + h * 0.16:.2f} "
              f"Q {x + w * 0.55:.2f} {yt - h * 0.02:.2f} {x + w * 0.30:.2f} {yt + h * 0.10:.2f} "
              f"Q {x - w * 0.06:.2f} {yt + h * 0.28:.2f} {x + w * 0.18:.2f} {yt + h * 0.46:.2f} "
              f"L {x + w * 0.82:.2f} {yt + h * 0.54:.2f} "
              f"Q {x + w * 1.06:.2f} {yt + h * 0.72:.2f} {x + w * 0.70:.2f} {yt + h * 0.90:.2f} "
              f"Q {x + w * 0.40:.2f} {yt + h * 1.02:.2f} {x:.2f} {yt + h * 0.84:.2f}"),
        # O: one closed oval.
        "O": (f"M {x + r:.2f} {yt:.2f} "
              f"C {x + w * 1.12:.2f} {yt:.2f} {x + w * 1.12:.2f} {yb:.2f} {x + r:.2f} {yb:.2f} "
              f"C {x - w * 0.12:.2f} {yb:.2f} {x - w * 0.12:.2f} {yt:.2f} {x + r:.2f} {yt:.2f} Z"),
        # R: bowl plus a diagonal leg that starts at the bowl's foot.
        "R": (f"M {x:.2f} {yb:.2f} L {x:.2f} {yt:.2f} "
              f"L {x + w * 0.60:.2f} {yt:.2f} "
              f"Q {x + w * 1.05:.2f} {yt:.2f} {x + w * 1.05:.2f} {yt + h * 0.27:.2f} "
              f"Q {x + w * 1.05:.2f} {yt + h * 0.54:.2f} {x + w * 0.60:.2f} {yt + h * 0.54:.2f} "
              f"L {x:.2f} {yt + h * 0.54:.2f} "
              f"M {x + w * 0.55:.2f} {yt + h * 0.54:.2f} L {x + w * 1.02:.2f} {yb:.2f}"),
    }[ch]
    return (f'  <path d="{d}" fill="none" stroke="{P.INK}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')


def _knockout(ch: str, x: float, yb: float) -> str:
    """The same letter, drawn wide in the canvas colour, as a halo.

    The label overlaps the ring's lower quarter by design, and without this the
    dashed arc runs straight through the middle of the letters: on a 16 px
    render that is three characters and eight dash segments fighting for the same
    pixels.  The halo removes the dashes under the text so the label reads as one
    object sitting in front of the ring, which is the correct depth order too.
    """
    return _letter(ch, x, yb, sw=LETTER_SW + 3.1).replace(
        f'stroke="{P.INK}"', 'stroke="#FFFFFF"')


def _label() -> str:
    """SOR, riding up into the ring's lower quarter.

    Measured: the letters' top sits 27.8 % of the cloud's height above the ring's
    bottom.  Sitting fully under the ring left a 3.40u gap -- almost exactly a
    quarter of the cloud's height -- and the icon read as two stacked pieces
    rather than one composed mark.
    """
    x0 = (24.0 - LABEL_W) / 2
    step = LETTER_W + LETTER_GAP
    halos = "\n".join(_knockout(ch, x0 + i * step, LABEL_Y)
                       for i, ch in enumerate("SOR"))
    ink = "\n".join(_letter(ch, x0 + i * step, LABEL_Y)
                     for i, ch in enumerate("SOR"))
    return halos + "\n" + ink


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
        _label(),
    ]
    return P.svg("\n".join(parts))
