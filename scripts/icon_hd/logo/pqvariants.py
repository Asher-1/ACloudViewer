"""The remaining panel icons: variants built from a shared surface/frustum base.

Twenty-six of the thirty-two icons left in this family are *pairs and triples*:
surface selection in {cell, point} x {plain, interactive, data-interactive}, a
frustum selection in {block, cell, point}, a polygon selection in {cell, point},
and six single-axis nudge buttons.

Drawing 32 separate pictures would be wrong twice over.  It would not look like
one family, and it would put 32 chances of an inconsistent badge in front of the
eye.  The variant axis is made explicit instead: one base mark per family (a
surface patch, a view cone, a polygon), plus one badge that says which member of
the pair it is.  The user reads the base once and then only has to tell the badge
apart.

Measured at 18 px, two first attempts failed and are replaced here:

* The surface patch carried a 1.0-wide internal grid.  Those lines merged with
  the 1.5 outline into a single grey lozenge, and the accent cell sat inside the
  mud.  The patch is now an outline only; the selection is the only filled area.
* The chart marquee was drawn *across* the bars, so box and bars crossed at four
  places.  Selection is now shown by accenting the bars themselves, which needs
  no second outline at all.
"""

from __future__ import annotations

from . import iconset as K


# ---------------------------------------------------------------------------
# Shared bases
# ---------------------------------------------------------------------------

# The patch lives in the lower-left so the badge corner is always free:
#   x 2.2 .. 13.4,  y 9.0 .. 16.6
_P_O = (2.2, 11.6)
_P_U = (7.4, -2.6)
_P_V = (3.8, 5.0)


def _P(a, b):
    return (_P_O[0] + a * _P_U[0] + b * _P_V[0],
            _P_O[1] + a * _P_U[1] + b * _P_V[1])


def _plane() -> str:
    return K.poly([_P(0, 0), _P(1, 0), _P(1, 1), _P(0, 1)],
                  close=True, color=K.MUTED, width=1.5)


def _surface_mark(kind: str) -> str:
    """The selected element inside the patch: a cell or a point."""
    if kind == "cell":
        return K.poly([_P(0.18, 0.55), _P(0.96, 0.30), _P(0.96, 0.94), _P(0.18, 1.0)],
                      close=True, fill=K.ACCENT)
    return K.dot(*_P(1.0, 1.0), 2.2, color=K.ACCENT)


def _cursor(x, y, s=0.55):
    """Pointer arrow: the badge for "you are clicking this"."""
    shape = [(0, 0), (0, 9.4), (2.5, 7.0), (4.3, 11.2), (6.3, 10.2),
             (4.4, 6.1), (7.6, 5.8)]
    return K.poly([(x + px * s, y + py * s) for px, py in shape],
                  close=True, fill=K.INK)


def _probe(x, y):
    """Ring with a centre dot: the badge for "under the pointer"."""
    return "\n".join([
        K.circle(x, y, 2.5, color=K.INK, width=1.5),
        K.dot(x, y, 1.0, color=K.INK),
    ])


def _cursor_data(x, y):
    """Pointer with two attached bars: clicking to read the values.

    One composite badge rather than two neighbouring ones -- at 18 px a separate
    cursor and a separate mini-chart crowd each other and read as noise.
    """
    return "\n".join([
        _cursor(x, y, s=0.50),
        K.line(x + 6.6, y + 5.2, x + 6.6, y + 1.4, color=K.ACCENT, width=1.3),
        K.line(x + 8.8, y + 5.2, x + 8.8, y - 0.6, color=K.ACCENT, width=1.3),
        K.line(x + 5.6, y + 5.4, x + 9.8, y + 5.4, color=K.INK, width=1.1),
    ])


def _frustum() -> str:
    """View cone seen from the side, apex at the left."""
    return K.poly([(2.6, 12.0), (18.6, 5.6), (18.6, 18.4)], close=True,
                  color=K.MUTED, width=1.5)


# ---------------------------------------------------------------------------
# Surface selection: {cell, point} x {plain, interactive, data-interactive}
# ---------------------------------------------------------------------------


def _surface(kind, badge=None):
    parts = [_plane(), _surface_mark(kind)]
    if badge == "interactive":
        parts.append(_cursor(15.0, 1.6))
    elif badge == "data":
        parts.append(_cursor_data(13.6, 1.4))
    return K.svg("\n".join(parts))


def pq_surface_selection_cell() -> str:
    return _surface("cell")


def pq_surface_selection_point() -> str:
    return _surface("point")


def pq_surface_selection_cell_interactive() -> str:
    return _surface("cell", "interactive")


def pq_surface_selection_point_interactive() -> str:
    return _surface("point", "interactive")


def pq_surface_selection_cell_data() -> str:
    return _surface("cell", "data")


def pq_surface_selection_point_data() -> str:
    return _surface("point", "data")


def _hovering(kind):
    """Same base with a probe ring instead of a pointer: under, not clicked."""
    return K.svg("\n".join([_plane(), _surface_mark(kind), _probe(16.4, 4.4)]))


def pq_surface_hovering_cell() -> str:
    return _hovering("cell")


def pq_surface_hovering_point() -> str:
    return _hovering("point")


# ---------------------------------------------------------------------------
# Frustum selection: a view cone with the picked primitive inside it
# ---------------------------------------------------------------------------


def pq_frustum_selection_block() -> str:
    return K.svg("\n".join([
        _frustum(),
        K.rect(13.0, 8.4, 5.6, 7.2, r=0.9, color=K.ACCENT, width=1.8),
        K.line(13.0, 11.2, 18.6, 11.2, color=K.ACCENT, width=1.1),
    ]))


def pq_frustum_selection_cell() -> str:
    return K.svg("\n".join([
        _frustum(),
        K.poly([(12.8, 12.4), (17.8, 10.4), (18.4, 13.8), (13.4, 15.8)],
               close=True, fill=K.ACCENT),
    ]))


def pq_frustum_selection_point() -> str:
    return K.svg("\n".join([
        _frustum(),
        K.dot(15.6, 11.4, 2.4),
    ]))


# ---------------------------------------------------------------------------
# Polygon selection over the surface
# ---------------------------------------------------------------------------


def _polygon_badge(kind):
    poly = K.poly([(12.8, 3.0), (19.0, 4.8), (20.6, 10.4),
                   (15.6, 12.4), (11.4, 7.6)],
                  close=True, color=K.INK, width=1.5)
    return K.svg("\n".join([_plane(), _surface_mark(kind), poly]))


def pq_polygon_select_cell() -> str:
    return _polygon_badge("cell")


def pq_polygon_select_point() -> str:
    return _polygon_badge("point")


# ---------------------------------------------------------------------------
# Single-axis nudges: one arrow, one signed badge
# ---------------------------------------------------------------------------


def _axis_sign(direction, sign):
    """A single axis arrow with a signed disc at its tip.

    A bare plus or minus beside the arrow was measured as ambiguous: at 18 px a
    minus reads as a continuation of the shaft, not as a symbol.  Putting the
    sign on a filled disc separates it from the arrow regardless of direction,
    and keeps the two operations distinguishable by shape rather than by a 1 px
    difference in a dash.
    """
    if direction == "x":
        body = K.arrow(3.4, 12.0, 12.4, 12.0, color=K.INK, head=2.4)
        cx, cy = 17.0, 12.0
    elif direction == "y":
        body = K.arrow(12.0, 20.6, 12.0, 11.6, color=K.INK, head=2.4)
        cx, cy = 12.0, 6.6
    else:
        body = K.arrow(4.6, 19.4, 12.6, 11.4, color=K.INK, head=2.4)
        cx, cy = 17.2, 6.6

    if sign == "plus":
        glyph = "\n".join([
            K.line(cx - 1.9, cy, cx + 1.9, cy, color=K.PAPER, width=1.9),
            K.line(cx, cy - 1.9, cx, cy + 1.9, color=K.PAPER, width=1.9),
        ])
    else:
        glyph = K.line(cx - 1.9, cy, cx + 1.9, cy, color=K.PAPER, width=1.9)
    return K.svg("\n".join([body, K.circle(cx, cy, 3.2, fill=K.ACCENT), glyph]))


def pq_x_plus() -> str:
    return _axis_sign("x", "plus")


def pq_x_minus() -> str:
    return _axis_sign("x", "minus")


def pq_y_plus() -> str:
    return _axis_sign("y", "plus")


def pq_y_minus() -> str:
    return _axis_sign("y", "minus")


def pq_z_plus() -> str:
    return _axis_sign("z", "plus")


def pq_z_minus() -> str:
    return _axis_sign("z", "minus")


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def _chevron(cx, cy, up=True, arm=4.6, color=None):
    d = -1 if up else 1
    return K.poly([(cx - arm, cy - arm * 0.52 * d), (cx, cy + arm * 0.52 * d),
                   (cx + arm, cy - arm * 0.52 * d)], color=color or K.INK, width=2.1)


def pq_arrow_up() -> str:
    return K.svg("\n".join([_chevron(12.0, 12.0, up=True, arm=6.4)]))


def pq_arrow_down() -> str:
    return K.svg("\n".join([_chevron(12.0, 12.0, up=False, arm=6.4)]))


def pq_clear_selection() -> str:
    """A cleared marquee: the selection is emptied, not re-sorted.

    The file is named ``pqClearSort`` and the previous drawing was two chevrons
    with a strike -- a "remove sorting" mark.  Checked against the only place
    that loads it, ``cvSelectionToolController.cpp``:

        make("actionClearSelection", ".../pqClearSort.svg",
             tr("Clear Selection"), tr("Clear selection (Esc)"), false);

    So the icon belongs to *Clear Selection*.  Drawing what the name suggests
    would have shipped a mark for a command that does not exist in this app.
    """
    return K.svg("\n".join([
        K.rect(3.4, 5.0, 17.2, 14.0, r=1.4, color=K.MUTED, width=1.5),
        K.poly([(3.4, 5.0), (20.6, 19.0)], color=K.MUTED, width=1.0),
        K.line(5.6, 17.4, 18.4, 6.6, color=K.ACCENT, width=2.4),
    ]))


# ---------------------------------------------------------------------------
# Panels and dialogs
# ---------------------------------------------------------------------------


def pq_advanced() -> str:
    """Three sliders: the advanced-properties mark."""
    parts = []
    for y, knob, color in ((7.0, 8.6, K.INK), (12.0, 15.4, K.ACCENT), (17.0, 11.0, K.INK)):
        parts.append(K.line(3.6, y, 20.4, y, color=K.MUTED, width=1.5))
        parts.append(K.circle(knob, y, 2.1, fill=color))
    return K.svg("\n".join(parts))


def pq_extract_selection() -> str:
    """A framed selection leaving the dataset: extract, not copy."""
    return K.svg("\n".join([
        K.rect(2.8, 5.0, 10.8, 14.0, r=1.2, color=K.MUTED, width=1.3),
        K.rect(5.2, 7.6, 6.0, 8.8, r=0.9, fill=K.ACCENT),
        K.arrow(14.6, 12.0, 20.8, 12.0, color=K.INK, head=2.6),
    ]))


def pq_programmable_filter() -> str:
    """Angle brackets and a slash: code that runs on the data."""
    return K.svg("\n".join([
        K.poly([(9.8, 6.0), (4.6, 12.0), (9.8, 18.0)], color=K.INK, width=2.1),
        K.poly([(14.2, 6.0), (19.4, 12.0), (14.2, 18.0)], color=K.INK, width=2.1),
        K.line(13.6, 4.6, 10.4, 19.4, color=K.ACCENT, width=1.9),
    ]))


def pq_save_table() -> str:
    """Table plus a tray with a down arrow: write the sheet out to disk."""
    return K.svg("\n".join([
        K.rect(2.8, 3.6, 12.4, 16.8, r=1.3, color=K.INK, width=1.6),
        K.line(2.8, 8.4, 15.2, 8.4, color=K.INK, width=1.3),
        K.line(2.8, 13.0, 15.2, 13.0, color=K.MUTED, width=1.1),
        K.line(2.8, 17.0, 15.2, 17.0, color=K.MUTED, width=1.1),
        K.arrow(19.8, 4.6, 19.8, 15.0, color=K.ACCENT, head=2.4),
        K.poly([(17.0, 15.4), (19.8, 20.4), (22.6, 15.4)], color=K.ACCENT, width=1.6),
    ]))


def pq_interaction_mode_3d() -> str:
    """Cube with an orbit arrow: rotate the scene, not pan it."""
    return K.svg("\n".join([
        K.cube(10.0, 12.6, r=6.2),
        f'  <path d="M 17.8 16.6 A 5.4 5.4 0 0 0 20.4 11.0" '
        f'stroke="{K.ACCENT}" stroke-width="1.9"/>',
        K.poly([(18.6, 8.6), (21.2, 11.2), (17.8, 11.6)], close=True, fill=K.ACCENT),
    ]))


def pq_plot_over_time() -> str:
    """Axes, a polyline and its sample dots: values over time."""
    return K.svg("\n".join([
        K.line(3.4, 3.6, 3.4, 20.2, color=K.MUTED, width=1.5),
        K.line(3.4, 20.2, 20.8, 20.2, color=K.MUTED, width=1.5),
        K.poly([(5.8, 15.6), (9.4, 10.0), (13.2, 13.0), (17.0, 6.4), (20.0, 8.6)],
               color=K.INK, width=2.0),
        K.dot(9.4, 10.0, 1.5),
        K.dot(17.0, 6.4, 1.5),
    ]))


# ---------------------------------------------------------------------------
# Colour ramp and grid
# ---------------------------------------------------------------------------


def pq_ramp() -> str:
    """A four-step ramp: a lookup table, not a gradient smear.

    The system has no gradients on purpose -- they collapse to mud at 18 px, and
    a ramp's job here is to say "many values map to colour", which four discrete
    steps say better than a smooth blend.
    """
    steps = [(3.0, "#D7E4EC"), (7.6, "#91BFD6"), (12.2, K.ACCENT), (16.8, K.INK)]
    return K.svg("\n".join(
        K.rect(x, 7.4, 4.6, 9.2, r=0.4, fill=c) for x, c in steps
    ))


def pq_rectilinear_grid() -> str:
    """A grid whose spacing is deliberately uneven: rectilinear, not regular.

    Measured at 18 px: three interior lines per axis covered 46% of the canvas
    and read as a solid hatch.  Two per axis keeps the uneven rhythm while
    leaving the cells open.
    """
    parts = [K.rect(3.4, 4.2, 17.2, 15.6, r=1.2, color=K.INK, width=1.7)]
    for x in (8.6, 15.0):
        parts.append(K.line(x, 4.2, x, 19.8, color=K.MUTED, width=1.0))
    for y in (9.4, 14.6):
        parts.append(K.line(3.4, y, 20.6, y, color=K.MUTED, width=1.0))
    parts.append(K.rect(8.6, 9.4, 6.4, 5.2, fill=K.ACCENT))
    return K.svg("\n".join(parts))


# ---------------------------------------------------------------------------
# Chart selection
# ---------------------------------------------------------------------------


def _chart(selected=()):
    """Three bars; ``selected`` indexes which are accented."""
    bars = [(6.0, 12.2, 8.2), (10.6, 7.8, 12.6), (15.2, 10.4, 10.0)]
    parts = [
        K.line(3.4, 5.0, 3.4, 20.4, color=K.MUTED, width=1.4),
        K.line(3.4, 20.4, 21.0, 20.4, color=K.MUTED, width=1.4),
    ]
    for i, (x, y, h) in enumerate(bars):
        if i in selected:
            parts.append(K.rect(x, y, 3.6, h, r=0.5, fill=K.ACCENT))
        else:
            parts.append(K.rect(x, y, 3.6, h, r=0.5, color=K.INK, width=1.5))
    return "\n".join(parts)


def pq_select_chart() -> str:
    """Bars selected by accent, not by a box drawn across them.

    The first version wrapped a marquee around the bars; outline and bars
    crossed at four points and merged into a grid at 18 px.  Highlighting is a
    stronger selection cue and needs no second outline.
    """
    return K.svg(_chart(selected=(0, 2)))


def pq_select_chart_polygon() -> str:
    """A lasso across the bars: freeform selection over the plot."""
    return K.svg("\n".join([
        _chart(selected=(1,)),
        K.poly([(8.6, 3.4), (19.8, 5.2), (21.4, 12.6), (14.0, 17.4), (7.4, 11.0)],
               close=True, color=K.ACCENT, width=1.8),
    ]))


REGISTRY = {
    "pqSurfaceSelectionCell": pq_surface_selection_cell,
    "pqSurfaceSelectionPoint": pq_surface_selection_point,
    "pqSurfaceSelectionCellInteractive": pq_surface_selection_cell_interactive,
    "pqSurfaceSelectionPointInteractive": pq_surface_selection_point_interactive,
    "pqSurfaceSelectionCellDataInteractive": pq_surface_selection_cell_data,
    "pqSurfaceSelectionPointDataInteractive": pq_surface_selection_point_data,
    "pqSurfaceHoveringCell": pq_surface_hovering_cell,
    "pqSurfaceHoveringPoint": pq_surface_hovering_point,
    "pqFrustumSelectionBlock": pq_frustum_selection_block,
    "pqFrustumSelectionCell": pq_frustum_selection_cell,
    "pqFrustumSelectionPoint": pq_frustum_selection_point,
    "pqPolygonSelectSurfaceCell": pq_polygon_select_cell,
    "pqPolygonSelectSurfacePoint": pq_polygon_select_point,
    "pqXPlus": pq_x_plus,
    "pqXMinus": pq_x_minus,
    "pqYPlus": pq_y_plus,
    "pqYMinus": pq_y_minus,
    "pqZPlus": pq_z_plus,
    "pqZMinus": pq_z_minus,
    "pqArrowUp16": pq_arrow_up,
    "pqArrowDown16": pq_arrow_down,
    "pqClearSort": pq_clear_selection,
    "pqAdvanced": pq_advanced,
    "pqExtractSelection": pq_extract_selection,
    "pqProgrammableFilter": pq_programmable_filter,
    "pqSaveTable32": pq_save_table,
    "pqInteractionMode3D": pq_interaction_mode_3d,
    "pqPlotSelectionOverTime": pq_plot_over_time,
    "pqRamp24": pq_ramp,
    "pqRectilinearGrid16": pq_rectilinear_grid,
    "pqSelectChart": pq_select_chart,
    "pqSelectChartPolygon": pq_select_chart_polygon,
}
