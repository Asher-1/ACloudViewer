"""toolMainSubsample -- keep fewer points, preserve the shape of the cloud.

What the code does
------------------
    MainWindow.cpp:13050         doActionSubsample()
    ecvSubsamplingDlg.cpp:65     three strategies on one switch

        RANDOM   keep exactly N points, chosen at random
        SPACE    keep a point only if no already-kept point is nearer than
                 `minDist` -- a spatially even thinning
        (plus optional SF modulation, which varies that spacing with the
        scalar field's value)

All three share one outcome: **far fewer points, same overall form**. The cloud
keeps its extent, its density gradient and its holes; it just gets sparser. That
is the whole idea, and unlike its neighbour SOR it is not about *which* points are
abnormal -- every point is equally valid, there are simply too many.

Why it is not the SOR icon again
--------------------------------
The two buttons sit together and both act on point clouds, so the distinction has
to be visible at a glance:

    SOR          keeps almost everything, removes a few marked as outliers
                 -> a dense cloud, a threshold band, three ✗
    subsample    keeps a regular subset of everything
                 -> two densities side by side, and nothing removed

So this icon is drawn as a *before / after pair*: a dense field on the left, its
thinned version on the right, the same footprint in both. The eye compares dot
spacing, which is exactly what the tool changes.

The original
------------
`toolMainSubsample.svg` is a scatter of red and blue 2x2 blocks -- no text, no
arrows, no frame. Reading the colours as meaning is the only sensible
interpretation available (the blue ones are the ones that survive), and that is
close to the before/after idea but cannot show *spacing*, because the source is a
single field rather than a pair.

The composition was therefore rebuilt as an explicit pair. The colour split is
kept -- blue for the kept points -- so the palette still says the same thing, and
every form carries the samples' #6F6F59 outline.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   #2896D3 kept points, the samples' greys for the source density,
              so the pair reads as "input, output" rather than as two colours
    dot ratio gap:size = 1.5:1, matching the approved cloud icons
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon subsample --show
    python3 -m scripts.icon_hd.logo.icon subsample --sheet out.png
"""

from __future__ import annotations

import math
import random

from . import paraview_style as P

DENSE = P.GREY_MID         # the cloud before
KEPT = P.BLUE_STRONG       # the points that survive

R_DOT = 0.85
# Dot size is set by 16 px, not by 192 px.  At 0.44 units a dot is 0.29 px at
# toolbar size and both panels collapse into two grey smudges -- the density
# contrast, which is the entire message, disappears.  0.85 units with 12 / 5
# dots keeps the left panel visibly denser than the right at 16 px.
MIN_R = 2.30

# Two panels of equal footprint.  The split is vertical because dot spacing is a
# horizontal read: the eye compares the left half's rhythm with the right half's
# across the same width.
PANEL_W = 8.6
LEFT_CX, RIGHT_CX = 5.8, 18.2
CY = 12.0
PANEL_H = 9.4


def _scatter(cx: float, spacing: float, seed: int, target: int):
    """Points inside the panel, at least ``spacing`` apart."""
    rng = random.Random(seed)
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < target and tries < 20000:
        tries += 1
        x = cx + rng.uniform(-PANEL_W / 2, PANEL_W / 2)
        y = CY + rng.uniform(-PANEL_H / 2, PANEL_H / 2)
        if all(math.hypot(x - qx, y - qy) >= spacing for qx, qy in pts):
            pts.append((x, y))
    return pts


def _dots(pts, colour, r=R_DOT) -> list[str]:
    return [f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{colour}" '
            f'stroke="{P.INK}" stroke-width="0.3"/>'
            for x, y in pts]


def _arrow() -> str:
    """A short arrow between the panels: input -> output.

    Without it the two fields read as "two clouds", which is a different
    statement. The arrow makes the pair a before/after and is the only mark that
    says a transformation happened.
    """
    x1, x2, y = LEFT_CX + PANEL_W / 2 + 0.55, RIGHT_CX - PANEL_W / 2 - 0.55, CY
    head, hw = 1.5, 0.95
    return "\n".join([
        f'  <path d="M {x1:.2f} {y:.2f} L {x2 - head * 0.5:.2f} {y:.2f}" '
        f'fill="none" stroke="{P.INK}" stroke-width="0.95" '
        f'stroke-linecap="round"/>',
        f'  <path d="M {x2:.2f} {y:.2f} L {x2 - head:.2f} {y - hw:.2f} '
        f'L {x2 - head:.2f} {y + hw:.2f} Z" fill="{P.INK}" stroke="none"/>',
    ])


def build() -> str:
    """A dense cloud, an arrow, and the same cloud thinned.

    The two fields use the *same* panel size and the same dot size, so the only
    visible difference is spacing -- which is precisely the parameter the tool
    exposes.  Any other difference (different extent, different dot size) would
    let the eye read a change that did not happen.
    """
    dense = _scatter(LEFT_CX, MIN_R, seed=5, target=17)
    thin = _scatter(RIGHT_CX, MIN_R * 2.15, seed=5, target=7)

    parts = [
        *_dots(dense, DENSE),
        _arrow(),
        *_dots(thin, KEPT),
    ]
    return P.svg("\n".join(parts))
