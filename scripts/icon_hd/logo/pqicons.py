"""Redesigned panel-toolbar icons (the pq* family).

Same system as the toolbar set: 24x24, 2.0 stroke, one ink, one accent, detail
floor 3 units.  Measured state of the family before this file existed:

    viewBox values in use   29 distinct numbers (13.5, 16, 18.75, 24, 26, 28.6,
                            32, 38.5, 44.8, 46.6, 48.7, 50, 52, 52.6, 52.9, 53.1,
                            53.4, 53.6, 54.2, 54.9, 56.5, 56.8, 57.6, 58.9, 59.1,
                            59.6, 59.9, 64 ...)
    unique fills per icon   median 0, max 224

Twenty-nine canvas sizes for one toolbar is not a style, it is a merge accident.
Only icons whose meaning can be stated in one sentence are rebuilt here; the
rest keep their artwork rather than getting an invented mark.
"""

from __future__ import annotations

import math

from . import iconset as K


# ---------------------------------------------------------------------------
# Generic actions
# ---------------------------------------------------------------------------


def pq_plus() -> str:
    return K.svg(K.plus(12, 12, arm=6.0))


def pq_minus() -> str:
    return K.svg(K.minus(12, 12, arm=6.0))


def pq_delete() -> str:
    return K.svg(K.trash(12, 12))


def pq_cancel() -> str:
    return K.svg("\n".join([
        K.line(6.4, 6.4, 17.6, 17.6, color=K.INK, width=2.2),
        K.line(17.6, 6.4, 6.4, 17.6, color=K.INK, width=2.2),
    ]))


def pq_apply() -> str:
    """Apply: a check inside a frame -- committed, not just validated."""
    return K.svg("\n".join([
        K.rect(3.4, 3.4, 17.2, 17.2, r=2.0, color=K.MUTED, width=1.4),
        K.poly([(7.4, 12.2), (10.8, 15.6), (17.0, 8.4)], color=K.ACCENT, width=2.6),
    ]))


def pq_auto_apply() -> str:
    """Auto-apply: the check with a live indicator, meaning 'happens by itself'."""
    return K.svg("\n".join([
        K.poly([(4.2, 12.6), (8.4, 17.0), (15.6, 7.6)], color=K.ACCENT, width=2.6),
        K.dot(19.4, 6.0, 2.2),
    ]))


def pq_settings() -> str:
    """Settings: a gear reduced to eight teeth, the minimum that reads as one."""
    cx, cy, r_out, r_in = 12.0, 12.0, 8.0, 5.4
    pts = []
    for i in range(16):
        a = math.radians(i * 22.5)
        rr = r_out if i % 2 == 0 else r_in
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return K.svg("\n".join([
        K.poly(pts, close=True, color=K.INK),
        K.circle(cx, cy, 2.4, fill=K.ACCENT),
    ]))


def _arc_arrow(cx, cy, r, a0, a1, *, color=None, width=None, head=3.0) -> str:
    """Arc from angle a0 to a1 (degrees, y down) with a filled head at the a1 end.

    Two measured failures are fixed here:

    1. The head was drawn as a fixed upright triangle regardless of the arc's
       arrival direction, so on ``pqReset`` it pointed up while the arc came in
       from the side -- a stray blob at 18 px.

    2. Once the head was derived from the end tangent, the arc still ran all the
       way to the tip, so the 2.0-wide butt cap sat on top of the triangle and
       ate it: the arrowhead measured as a 1 px notch.  The arc now stops short
       by the head length, which leaves the triangle sitting in a gap where its
       full width is visible.
    """
    span = a1 - a0
    sign = 1.0 if span > 0 else -1.0
    # stop the stroke short of the tip, expressed as an angle along the arc
    gap_deg = math.degrees(head * 0.92 / r)
    a_end = a1 - sign * gap_deg

    p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
    pe = (cx + r * math.cos(math.radians(a_end)), cy + r * math.sin(math.radians(a_end)))
    tip = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))

    sweep = 1 if span > 0 else 0
    large = 1 if abs(span) > 180 else 0
    tx = -math.sin(math.radians(a1)) * sign
    ty = math.cos(math.radians(a1)) * sign
    nx, ny = -ty, tx
    c = color or K.INK
    return "\n".join([
        f'  <path d="M {p0[0]:.2f} {p0[1]:.2f} A {r} {r} 0 {large} {sweep} '
        f'{pe[0]:.2f} {pe[1]:.2f}" stroke="{c}"'
        + (f' stroke-width="{width}"' if width else '') + '/>',
        K.poly([tip,
                (tip[0] - tx * head + nx * head * 0.72, tip[1] - ty * head + ny * head * 0.72),
                (tip[0] - tx * head - nx * head * 0.72, tip[1] - ty * head - ny * head * 0.72)],
               close=True, fill=c),
    ])


def pq_reset() -> str:
    """Reset: three-quarter circular arrow, head at the top-left.

    Numbers matter here -- a 270 degree sweep from 160 to -110 puts the gap on
    the left and the head where a reader looks for it.  An earlier call with
    a1 > a0 took the long way round and rendered the head as a scallop.
    """
    return K.svg(_arc_arrow(12, 12.4, 6.8, 145, -125, head=3.6))


def pq_reset_settings() -> str:
    """Restore defaults: a gear with a return badge in the opposite corner.

    Nested (gear inside arc) merged into a grey ring at 18 px, so the two ideas
    are separated in space: gear upper-right, badge lower-left, no shared edge.
    """
    cx, cy, r_out, r_in = 14.6, 9.4, 5.4, 3.8
    pts = []
    for i in range(16):
        a = math.radians(i * 22.5)
        rr = r_out if i % 2 == 0 else r_in
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return K.svg("\n".join([
        K.poly(pts, close=True, color=K.INK),
        K.circle(cx, cy, 1.7, fill=K.INK),
        _arc_arrow(5.6, 17.4, 3.1, 200, -10, color=K.ACCENT, width=2.0, head=2.6),
    ]))


def pq_reset_ui() -> str:
    """Reset UI: a layout grid with a returning arrow."""
    return K.svg("\n".join([
        K.rect(3.4, 4.0, 8.0, 16.0, r=1.2, color=K.INK, width=1.6),
        K.rect(13.4, 4.0, 7.2, 7.0, r=1.2, color=K.MUTED, width=1.4),
        K.rect(13.4, 13.0, 7.2, 7.0, r=1.2, color=K.ACCENT, width=1.6),
    ]))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def pq_select() -> str:
    """Marquee: dashed rectangle with handles."""
    return K.svg("\n".join([
        K.rect(3.6, 5.0, 16.8, 14.0, r=1.2, color=K.ACCENT, width=1.6),
        K.circle(3.6, 5.0, 1.7, fill=K.INK),
        K.circle(20.4, 5.0, 1.7, fill=K.INK),
        K.circle(3.6, 19.0, 1.7, fill=K.INK),
        K.circle(20.4, 19.0, 1.7, fill=K.INK),
    ]))


def pq_select_plus() -> str:
    return K.svg("\n".join([
        K.rect(2.8, 5.0, 13.0, 12.0, r=1.2, color=K.ACCENT, width=1.6),
        K.plus(18.6, 17.2, arm=3.0, color=K.INK, width=2.0),
    ]))


def pq_select_minus() -> str:
    return K.svg("\n".join([
        K.rect(2.8, 5.0, 13.0, 12.0, r=1.2, color=K.ACCENT, width=1.6),
        K.minus(18.6, 17.2, arm=3.0, color=K.INK, width=2.0),
    ]))


def pq_select_toggle() -> str:
    """Toggle: a marquee with a switch, the universal 'flip membership' mark.

    Plus and minus already own add/remove, so toggle needs a third shape.  A
    switch reads as binary state at 18 px, where a circular arrow pair collapses
    into a smudge.
    """
    return K.svg("\n".join([
        K.rect(2.8, 3.4, 13.0, 10.4, r=1.2, color=K.ACCENT, width=1.6),
        K.rect(12.0, 14.4, 9.6, 5.0, r=2.5, color=K.INK, width=1.6),
        K.circle(19.0, 16.9, 1.7, fill=K.INK),
    ]))


def pq_select_block() -> str:
    """Block select: a marquee with one cell of the dataset filled inside it."""
    return K.svg("\n".join([
        K.rect(2.8, 4.0, 18.2, 16.0, r=1.4, color=K.ACCENT, width=1.6),
        K.rect(8.4, 9.6, 7.0, 7.0, r=1.2, fill=K.INK),
    ]))


# ---------------------------------------------------------------------------
# Data kinds: point vs cell is the pair that must differ by silhouette
# ---------------------------------------------------------------------------


def pq_point_data() -> str:
    """Point data: dots on a lattice, one accented."""
    parts = [K.dot(x, y, 1.7, color=K.INK)
             for x, y in ((5.0, 6.0), (12.0, 6.0), (19.0, 6.0),
                          (5.0, 12.0), (19.0, 12.0),
                          (5.0, 18.0), (12.0, 18.0), (19.0, 18.0))]
    parts.append(K.dot(12.0, 12.0, 3.0))
    return K.svg("\n".join(parts))


def pq_cell_data() -> str:
    """Cell data: filled cells, one accented -- squares, not dots."""
    parts = [K.rect(x, y, 4.2, 4.2, r=0.8, color=K.INK, width=1.4)
             for x, y in ((4.4, 5.0), (11.0, 5.0), (17.6, 5.0),
                          (4.4, 11.6), (17.6, 11.6),
                          (4.4, 18.2), (11.0, 18.2), (17.6, 18.2))]
    parts.append(K.rect(11.0, 11.6, 4.2, 4.2, r=0.8, fill=K.ACCENT,
                        color=K.ACCENT, width=0))
    return K.svg("\n".join(parts))


def pq_global_data() -> str:
    """Global data: the lattice with nothing singled out."""
    return K.svg("\n".join(
        K.rect(x, y, 4.2, 4.2, r=0.8, color=K.INK, width=1.4)
        for x, y in ((4.4, 5.0), (11.0, 5.0), (17.6, 5.0),
                     (4.4, 11.6), (11.0, 11.6), (17.6, 11.6),
                     (4.4, 18.2), (11.0, 18.2), (17.6, 18.2))
    ))


# ---------------------------------------------------------------------------
# Camera / view
# ---------------------------------------------------------------------------


def pq_edit_camera() -> str:
    """Camera body, lens, and the viewfinder wedge: three shapes, no overlap.

    An extra handle outline above the body was measured at 120 px: it read as a
    notch in the silhouette rather than as a cine-camera handle, so it is gone.
    """
    return K.svg("\n".join([
        K.rect(2.6, 8.8, 11.6, 8.8, r=1.8, color=K.INK),
        K.circle(8.4, 13.2, 2.5, color=K.INK, width=1.6),
        K.poly([(14.6, 12.4), (20.6, 8.6), (20.6, 17.8), (14.6, 14.0)],
               close=True, color=K.ACCENT, width=1.8),
    ]))


def pq_invert() -> str:
    """Flip triangle winding: two triangles whose normals swap sides.

    Two wrong readings preceded this one.  The file is called ``pqInvert`` and
    the first drawing was a half-filled disc ("invert values"); the second was a
    half-square.  Both are wrong.  The only loader is

        <action name="actionFlipMeshTriangles">
          ... icon .../pqInvert.svg ...
          text  "Flip triangles"

    so the command reverses each triangle's vertex order, which flips its
    normal.  The mark is therefore two triangles plus opposed normal arrows.
    """
    # Triangles on the outer edges, normals in the gap between them, so no
    # arrow crosses a body.  The two normals point opposite ways, which is the
    # whole content of "flip the winding".
    left = [(2.8, 20.6), (9.6, 20.6), (6.2, 14.2)]
    right = [(14.4, 3.4), (21.2, 3.4), (17.8, 9.8)]
    return K.svg("\n".join([
        K.poly(left, close=True, color=K.INK, width=1.7),
        K.poly(right, close=True, color=K.ACCENT, width=1.7),
        K.arrow(11.9, 6.2, 11.9, 17.8, color=K.ACCENT, head=2.2),
        K.arrow(12.1, 17.8, 12.1, 6.2, color=K.INK, head=2.2),
    ]))


def pq_find_data() -> str:
    """Magnifier beside a table, not on top of it: find a row.

    The first version centred the magnifier over the grid, so the two outlines
    crossed and the result was a grey tangle.  Side by side, each mark stays
    readable on its own.
    """
    return K.svg("\n".join([
        K.rect(2.6, 5.0, 11.6, 14.0, r=1.4, color=K.INK, width=1.6),
        K.line(2.6, 9.4, 14.2, 9.4, color=K.INK, width=1.4),
        K.line(2.6, 14.2, 14.2, 14.2, color=K.MUTED, width=1.2),
        K.circle(16.2, 13.0, 4.0, color=K.ACCENT),
        K.line(19.0, 15.8, 21.6, 18.4, color=K.ACCENT, width=2.2),
    ]))


def pq_contours() -> str:
    """Iso-lines climbing one peak: nested open curves, not concentric rings.

    Closed rings read as a target or an eye at any size -- the eye-shaped
    version was rejected on exactly that reading.  Open curves that tighten
    toward a summit say "elevation" without borrowing either symbol.
    """
    return K.svg("\n".join([
        f'  <path d="M 2.4 20.4 C 5.2 20.4, 5.6 9.6, 12.0 9.6 '
        f'C 18.4 9.6, 18.8 20.4, 21.6 20.4" stroke="{K.MUTED}" stroke-width="1.5"/>',
        f'  <path d="M 4.0 20.4 C 6.4 20.4, 6.8 6.4, 12.0 6.4 '
        f'C 17.2 6.4, 17.6 20.4, 20.0 20.4" stroke="{K.ACCENT}" stroke-width="1.6"/>',
        f'  <path d="M 6.4 20.4 C 8.0 20.4, 8.4 3.2, 12.0 3.2 '
        f'C 15.6 3.2, 16.0 20.4, 17.6 20.4" stroke="{K.INK}" stroke-width="1.8"/>',
    ]))


def pq_undo_camera() -> str:
    return K.svg("\n".join([
        f'  <path d="M 5.4 12 a 6.6 6.6 0 1 0 6.6 -6.6" stroke="{K.INK}"/>',
        K.poly([(12.0, 2.6), (8.2, 5.4), (12.0, 8.2)], close=True, fill=K.INK),
    ]))


def pq_redo_camera() -> str:
    return K.svg("\n".join([
        f'  <path d="M 18.6 12 a 6.6 6.6 0 1 1 -6.6 -6.6" stroke="{K.INK}"/>',
        K.poly([(12.0, 2.6), (15.8, 5.4), (12.0, 8.2)], close=True, fill=K.INK),
    ]))


def pq_axes() -> str:
    """Axes: three arrows from one origin."""
    return K.svg("\n".join([
        K.arrow(5.0, 19.0, 19.0, 19.0, head=2.6),
        K.arrow(5.0, 19.0, 5.0, 5.0, head=2.6),
        K.arrow(5.0, 19.0, 14.0, 10.0, color=K.ACCENT, head=2.6),
    ]))


def pq_show_orientation_axes() -> str:
    """The corner axis triad: three perpendicular arrows from one origin.

    The widget's job is to say which way X/Y/Z point, so the triad *is* the
    mark.  A cube was carrying no information and stole the space the third
    arrow needed; dropping it is what lets all three read at 18 px.
    """
    return K.svg("\n".join([
        K.line(8.4, 15.6, 8.4, 4.0, color=K.MUTED, width=1.2),
        K.line(8.4, 15.6, 20.0, 15.6, color=K.MUTED, width=1.2),
        K.arrow(8.4, 15.6, 8.4, 4.6, head=2.8),
        K.arrow(8.4, 15.6, 19.4, 15.6, color=K.ACCENT, head=2.8),
        K.arrow(8.4, 15.6, 16.4, 7.6, head=2.8),
        K.dot(8.4, 15.6, 1.8),
    ]))


def pq_fixed_repr() -> str:
    """Fixed-point formatting: a numeral slot with a swap arrow beneath it.

    The button toggles the spreadsheet's *number formatting*, scientific <->
    fixed-point.  It was first drawn as a cube with a padlock, which reads as
    "locked mesh" -- a different control entirely.  A hash is the standard stand-
    in for "a digit goes here"; the two-headed arrow says it flips notation.
    """
    return K.svg("\n".join([
        K.line(9.6, 3.4, 7.8, 13.4, color=K.INK, width=1.8),
        K.line(15.6, 3.4, 13.8, 13.4, color=K.INK, width=1.8),
        K.line(5.2, 6.6, 17.6, 6.6, color=K.INK, width=1.8),
        K.line(4.4, 10.6, 16.8, 10.6, color=K.INK, width=1.8),
        K.line(6.4, 18.8, 17.6, 18.8, color=K.ACCENT, width=1.8),
        K.poly([(6.2, 18.8), (9.0, 16.9), (9.0, 20.7)], close=True, fill=K.ACCENT),
        K.poly([(17.8, 18.8), (15.0, 16.9), (15.0, 20.7)], close=True, fill=K.ACCENT),
    ]))


# ---------------------------------------------------------------------------
# Split / layout
# ---------------------------------------------------------------------------


def pq_split_horizontal() -> str:
    return K.svg("\n".join([
        K.rect(3.2, 4.0, 17.6, 16.0, r=1.4, color=K.INK),
        K.line(3.2, 12.0, 20.8, 12.0, color=K.ACCENT, width=2.0),
    ]))


def pq_split_vertical() -> str:
    return K.svg("\n".join([
        K.rect(3.2, 4.0, 17.6, 16.0, r=1.4, color=K.INK),
        K.line(12.0, 4.0, 12.0, 20.0, color=K.ACCENT, width=2.0),
    ]))


def pq_close_view() -> str:
    return K.svg("\n".join([
        K.rect(3.2, 4.0, 12.4, 11.0, r=1.2, color=K.INK, width=1.6),
        K.line(15.0, 15.6, 20.8, 20.6, color=K.ACCENT, width=2.2),
        K.line(20.8, 15.6, 15.0, 20.6, color=K.ACCENT, width=2.2),
    ]))


# ---------------------------------------------------------------------------
# Misc tools
# ---------------------------------------------------------------------------


def pq_lock() -> str:
    return K.svg("\n".join([
        f'  <path d="M 8.4 11.0 v -3.4 a 3.6 3.6 0 0 1 7.2 0 v 3.4" stroke="{K.INK}"/>',
        K.rect(6.0, 11.0, 12.0, 9.4, r=1.6, fill=K.ACCENT),
    ]))





def pq_ruler() -> str:
    """Ruler: a bar with ticks, the measure mark."""
    return K.svg("\n".join([
        K.rect(2.8, 8.4, 18.4, 7.2, r=1.2, color=K.INK),
        *[K.line(x, 8.4, x, 12.0, color=K.INK, width=1.4) for x in (6.0, 9.2, 12.4, 15.6, 18.8)],
        K.line(9.2, 12.0, 9.2, 15.6, color=K.ACCENT, width=1.4),
    ]))


def pq_protractor() -> str:
    """Protractor: a half-disc with a protractor arc and a needle."""
    return K.svg("\n".join([
        f'  <path d="M 3.6 17.4 a 8.4 8.4 0 0 1 16.8 0 Z" stroke="{K.INK}"/>',
        K.line(12.0, 17.4, 12.0, 9.0, color=K.ACCENT, width=1.6),
        K.line(8.4, 17.4, 15.6, 17.4, color=K.MUTED, width=1.3),
    ]))


def pq_capture_screenshot() -> str:
    """Screenshot: camera corners over a frame."""
    return K.svg("\n".join([
        K.rect(3.4, 3.4, 17.2, 17.2, r=1.2, color=K.MUTED, width=1.2),
        K.poly([(6.0, 9.4), (6.0, 6.0), (9.4, 6.0)], color=K.ACCENT, width=2.2),
        K.poly([(14.6, 6.0), (18.0, 6.0), (18.0, 9.4)], color=K.ACCENT, width=2.2),
        K.poly([(6.0, 14.6), (6.0, 18.0), (9.4, 18.0)], color=K.ACCENT, width=2.2),
        K.poly([(14.6, 18.0), (18.0, 18.0), (18.0, 14.6)], color=K.ACCENT, width=2.2),
    ]))


def pq_zoom_global() -> str:
    return K.svg("\n".join([
        K.circle(10.6, 10.6, 6.4, color=K.INK),
        K.line(15.2, 15.2, 20.4, 20.4, color=K.INK, width=2.2),
        K.line(7.4, 10.6, 13.8, 10.6, color=K.ACCENT, width=1.4),
        K.line(10.6, 7.4, 10.6, 13.8, color=K.ACCENT, width=1.4),
    ]))


def pq_zoom_selected() -> str:
    return K.svg("\n".join([
        K.circle(10.6, 10.6, 6.4, color=K.INK),
        K.line(15.2, 15.2, 20.4, 20.4, color=K.INK, width=2.2),
        K.rect(7.8, 7.8, 5.6, 5.6, r=0.8, fill=K.ACCENT, color=K.ACCENT, width=0),
    ]))



def pq_recent_file() -> str:
    """A document with a clock badge: recently opened, not newly created.

    The 64 px original was a flat two-colour bitmap; a faithful trace of it
    measured IoU 0.966 but weighed 58 KB for a menu glyph.  Rebuilt on the
    system canvas it is ~700 bytes and stays sharp at any size.
    """
    return K.svg("\n".join([
        K.poly([(3.4, 2.8), (12.2, 2.8), (15.6, 6.2), (15.6, 14.4),
                (3.4, 14.4)], close=True, color=K.INK, width=1.6),
        K.poly([(12.2, 2.8), (12.2, 6.2), (15.6, 6.2)], color=K.INK, width=1.3),
        K.line(6.0, 8.4, 12.0, 8.4, color=K.MUTED, width=1.2),
        K.line(6.0, 11.4, 10.4, 11.4, color=K.MUTED, width=1.2),
        K.circle(16.8, 16.4, 4.6, color=K.ACCENT, width=1.8),
        K.line(16.8, 16.4, 16.8, 13.8, color=K.ACCENT, width=1.6),
        K.line(16.8, 16.4, 18.8, 17.6, color=K.ACCENT, width=1.6),
    ]))


REGISTRY = {
    "pqPlus": pq_plus,
    "pqMinus": pq_minus,
    "pqDelete": pq_delete,
    "pqCancel": pq_cancel,
    "pqApply": pq_apply,
    "pqAutoApply": pq_auto_apply,
    "pqSettings": pq_settings,
    "pqReset": pq_reset,
    "pqResetSettings": pq_reset_settings,
    "pqResetUI": pq_reset_ui,
    "pqSelect16": pq_select,
    "pqSelectPlus": pq_select_plus,
    "pqSelectMinus": pq_select_minus,
    "pqSelectToggle": pq_select_toggle,
    "pqSelectBlock": pq_select_block,
    "pqPointData": pq_point_data,
    "pqCellData": pq_cell_data,
    "pqGlobalData": pq_global_data,
    "pqEditCamera": pq_edit_camera,
    "pqUndoCamera": pq_undo_camera,
    "pqRedoCamera": pq_redo_camera,
    "pqAxes": pq_axes,
    "pqShowOrientationAxes": pq_show_orientation_axes,
    "pqFixedRepr32": pq_fixed_repr,
    "pqSplitHorizontal": pq_split_horizontal,
    "pqSplitVertical": pq_split_vertical,
    "pqCloseView": pq_close_view,
    "pqLock": pq_lock,
    "pqInvert": pq_invert,
    "pqFindData": pq_find_data,
    "pqContours": pq_contours,
    "pqRuler": pq_ruler,
    "pqProtractor": pq_protractor,
    "pqCaptureScreenshot": pq_capture_screenshot,
    "pqZoomToGlobalData": pq_zoom_global,
    "pqZoomToSelectedData": pq_zoom_selected,
    "pqRecentFile": pq_recent_file,
}
