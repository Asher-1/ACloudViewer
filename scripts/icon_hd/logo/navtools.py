"""Navigation and picking tool icons -- five at a time.

Written as one batch because they share a domain (how you look at or select in the
3D view) and because drawing them together keeps their visual grammar consistent:
the two projections are a matched pair, and the two picking modes differ only in
how many points they take.

What each one means
-------------------
    GlobalZoom              fit the whole scene in view -- a frame with arrows
                            pulling outward to its corners
    OrthoProjection         a box seen with parallel edges
    PerspectiveProjection   the same box, with edges converging to a vanishing
                            point
    PointPicking            pick one point: a cursor over a single mark
    PointListPicking        pick many points: a numbered list of picked points

The projection pair is the one that has to be *read against itself*. A box drawn
in the same place, same colour, same size, differing only in whether its far edges
converge -- that is the entire difference between the two projections, and a pair
that does anything else makes the toolbar buttons guesswork.

The picking pair works the same way: one mark versus a list of marks.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   the samples' blues and greys, #2896D3 for the action marks
    fills     subtle gradients only where volume is needed
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon zoom --show
    python3 -m scripts.icon_hd.logo.icon --all --score
"""

from __future__ import annotations

from . import paraview_style as P

BLUE = P.BLUE_STRONG
PALE = P.BLUE_PALE
MID = P.BLUE_MID
GREY = P.GREY
GREY_DK = P.GREY_DK
INK = P.INK
SW = P.STROKE


# ---------------------------------------------------------------------------
# GlobalZoom -- expand the frame to hold everything
# ---------------------------------------------------------------------------


def global_zoom() -> str:
    """A frame with four corner arrows pushing outward.

    Read from the original, which is two heavy vertical bars with magenta corner
    blocks -- unreadable as anything. The action is *fit everything in view*, and
    the standard mark for that is a frame growing to its corners. Redrawn as four
    L-brackets because a full rectangle plus arrows reads as "export" or "resize",
    while broken corners read as "expand to these bounds".
    """
    def bracket(cx: float, cy: float, sx: int, sy: int) -> str:
        ln = 4.6
        return (f'  <path d="M {cx + sx * ln:.2f} {cy:.2f} '
                f'L {cx:.2f} {cy:.2f} L {cx:.2f} {cy + sy * ln:.2f}" '
                f'fill="none" stroke="{BLUE}" stroke-width="2.1" '
                f'stroke-linecap="square"/>')

    def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        h, w = 2.0, 1.15
        ax, ay = x2 - math.cos(a) * h, y2 - math.sin(a) * h
        px, py = -math.sin(a) * w, math.cos(a) * w
        return "\n".join([
            f'  <path d="M {x1:.2f} {y1:.2f} L {ax:.2f} {ay:.2f}" fill="none" '
            f'stroke="{BLUE}" stroke-width="1.5" stroke-linecap="round"/>',
            f'  <path d="M {x2:.2f} {y2:.2f} L {ax + px:.2f} {ay + py:.2f} '
            f'L {ax - px:.2f} {ay - py:.2f} Z" fill="{BLUE}" stroke="none"/>',
        ])

    return "\n".join([
        bracket(6.2, 6.2, 1, 1),
        bracket(17.8, 6.2, -1, 1),
        bracket(6.2, 17.8, 1, -1),
        bracket(17.8, 17.8, -1, -1),
        arrow(9.4, 9.4, 4.2, 4.2),
        arrow(14.6, 14.6, 19.8, 19.8),
    ])


# ---------------------------------------------------------------------------
# The projection pair
# ---------------------------------------------------------------------------


def _box(converge: bool) -> str:
    """A cube in three-quarter view.

    `converge` is the whole parameter: true draws the far face smaller and the
    receding edges aimed at one point (perspective); false keeps the far face
    identical in size and every edge parallel (orthographic). Nothing else
    differs, which is what makes the pair legible as a pair.
    """
    # near face
    nx0, ny0, nx1, ny1 = 3.2, 6.4, 13.4, 19.4
    # far face: inset toward the vanishing point when perspective
    if converge:
        k = 0.74                      # far face is 74 % the near face's size
        fx0 = nx0 + (15.6 - nx0) * 0.30
        fy0 = ny0 - (ny0 - 4.4) * 0.62
        fx1 = fx0 + (nx1 - nx0) * k
        fy1 = fy0 + (ny1 - ny0) * k
    else:
        fx0, fy0 = 8.6, 1.9
        fx1, fy1 = fx0 + (nx1 - nx0), fy0 + (ny1 - ny0)

    parts = [
        # far face
        f'  <path d="M {fx0:.2f} {fy0:.2f} L {fx1:.2f} {fy0:.2f} '
        f'L {fx1:.2f} {fy1:.2f} L {fx0:.2f} {fy1:.2f} Z" fill="{MID}" '
        f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"/>',
        # connecting edges
        f'  <path d="M {nx0:.2f} {ny0:.2f} L {fx0:.2f} {fy0:.2f} '
        f'M {nx1:.2f} {ny0:.2f} L {fx1:.2f} {fy0:.2f} '
        f'M {nx1:.2f} {ny1:.2f} L {fx1:.2f} {fy1:.2f} '
        f'M {nx0:.2f} {ny1:.2f} L {fx0:.2f} {fy1:.2f}" fill="none" '
        f'stroke="{INK}" stroke-width="{SW}"/>',
        # near face
        f'  <path d="M {nx0:.2f} {ny0:.2f} L {nx1:.2f} {ny0:.2f} '
        f'L {nx1:.2f} {ny1:.2f} L {nx0:.2f} {ny1:.2f} Z" fill="{PALE}" '
        f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"/>',
    ]
    return "\n".join(parts)


def ortho_projection() -> str:
    """The box with parallel edges."""
    return _box(converge=False)


def perspective_projection() -> str:
    """The same box with converging edges."""
    return _box(converge=True)


# ---------------------------------------------------------------------------
# The picking pair
# ---------------------------------------------------------------------------


def _cursor(x: float, y: float, s: float = 1.0) -> str:
    """An arrow cursor with its tip at (x, y)."""
    d = (f"M {x:.2f} {y:.2f} L {x:.2f} {y + 9.4 * s:.2f} "
         f"L {x + 2.4 * s:.2f} {y + 7.2 * s:.2f} "
         f"L {x + 4.0 * s:.2f} {y + 10.6 * s:.2f} "
         f"L {x + 5.6 * s:.2f} {y + 9.9 * s:.2f} "
         f"L {x + 3.9 * s:.2f} {y + 6.6 * s:.2f} "
         f"L {x + 7.0 * s:.2f} {y + 6.4 * s:.2f} Z")
    return "\n".join([
        f'  <path d="{d}" fill="#FFFFFF" stroke="{INK}" stroke-width="{SW}" '
        f'stroke-linejoin="round"/>',
    ])


def point_picking() -> str:
    """One point picked: a cursor over a single marked point."""
    return "\n".join([
        # the point being picked
        f'  <circle cx="6.6" cy="6.8" r="2.2" fill="{BLUE}" stroke="{INK}" '
        f'stroke-width="0.6"/>',
        f'  <circle cx="6.6" cy="6.8" r="4.1" fill="none" stroke="{BLUE}" '
        f'stroke-width="0.8" stroke-dasharray="1.3 1.1"/>',
        _cursor(9.4, 8.2, 1.02),
    ])


def point_list_picking() -> str:
    """Many points picked: a list, each line preceded by a picked marker.

    The list form is the distinction from `point_picking`: same cursor idea is not
    enough once several points are involved, because what the tool produces is not
    a selection but a *table* of picked points.
    """
    rows = []
    for i in range(4):
        y = 6.4 + i * 4.2
        rows.append(f'  <circle cx="6.2" cy="{y:.2f}" r="1.5" fill="{BLUE}" '
                    f'stroke="{INK}" stroke-width="0.5"/>')
        rows.append(f'  <path d="M 9.8 {y:.2f} L 18.4 {y:.2f}" fill="none" '
                    f'stroke="{INK}" stroke-width="1.15" stroke-linecap="round"/>')
    return "\n".join([
        f'  <rect x="2.8" y="3.4" width="18.4" height="17.2" rx="1.4" '
        f'fill="#FFFFFF" stroke="{BLUE}" stroke-width="1.7"/>',
        *rows,
    ])


ICONS = {
    "zoom": ("toolMainGlobalZoom", global_zoom),
    "ortho": ("toolMainOrthoProjection", ortho_projection),
    "perspective": ("toolMainPerspectiveProjection", perspective_projection),
    "pointpicking": ("toolMainPointPicking", point_picking),
    "pointlist": ("toolMainPointListPicking", point_list_picking),
}
