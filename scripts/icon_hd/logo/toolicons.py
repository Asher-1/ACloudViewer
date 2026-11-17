"""Redesigned toolbar icons: one drawing per tool, composed from the shared kit.

Every icon here answers "what does this tool do to the data?", not "what object
is this?".  That distinction is what separates a toolbar from clip art:
``ecvFilterByValue`` is a funnel with one survivor kept, ``ecvSor`` is a cloud
where the outliers have been struck out, ``ecvMerge`` is two clouds becoming
one.  The accent colour marks the thing that changes, which is how the eye finds
the verb in 18 px.

Canvas 24x24, stroke 2.0, detail floor 3 units -- see iconset.py for why.
"""

from __future__ import annotations

from . import iconset as K


# ---------------------------------------------------------------------------
# Edit / basic operations
# ---------------------------------------------------------------------------


def ecv_plus() -> str:
    return K.svg(K.plus(12, 12, arm=6.0))


def ecv_minus() -> str:
    return K.svg(K.minus(12, 12, arm=6.0))


def ecv_delete() -> str:
    return K.svg(K.trash(12, 12))


def ecv_exit() -> str:
    """Door with an arrow leaving it: 'leave the application', not 'close'."""
    return K.svg("\n".join([
        K.poly([(6, 4.5), (13, 4.5), (13, 19.5), (6, 19.5)], color=K.INK, close=True),
        K.arrow(14.5, 12, 21, 12, color=K.ACCENT),
        K.line(6.5, 12, 14.0, 12, color=K.ACCENT, width=1.0),
    ]))


def ecv_pencil() -> str:
    """Pencil at 45 degrees: the universal edit mark."""
    return K.svg("\n".join([
        K.poly([(4.5, 19.5), (6.2, 15.0), (16.5, 4.7), (19.3, 7.5), (9.0, 17.8)],
               color=K.INK, close=True),
        K.line(15.2, 6.0, 18.0, 8.8, color=K.ACCENT, width=1.6),
    ]))


def ecv_clone() -> str:
    """One cloud and a plus: duplicate, not overlay.

    Measured at 18 px: two overlapping clouds merged into a single mass.  One
    cloud plus a plus sign separates "the object" from "the action".
    """
    return K.svg("\n".join([
        K.cloud(11.0, 13.0, s=0.86),
        K.plus(19.0, 6.5, arm=2.6, color=K.ACCENT, width=2.2),
    ]))


def ecv_console() -> str:
    return K.svg(K.terminal(12, 12))


# ---------------------------------------------------------------------------
# Selection / picking
# ---------------------------------------------------------------------------


def ecv_point_picking() -> str:
    """Sparse dots with one enlarged and accented: pick a vertex.

    Measured at 18 px: a visible lattice plus a centre dot covered 41% of the
    canvas because the grid lines merged with the dots.  Dropping the grid and
    keeping five dots -- one of them bigger and accented -- says the same thing
    with less than half the ink.
    """
    parts = [K.dot(x, y, 1.5, color=K.INK)
             for x, y in ((5.0, 6.0), (12.0, 5.0), (19.0, 6.5),
                          (6.5, 18.0), (18.0, 18.5))]
    parts.append(K.circle(12.0, 12.5, 3.4, color=K.ACCENT, fill=K.ACCENT, width=0))
    return K.svg("\n".join(parts))


def ecv_point_list_picking() -> str:
    """Three rows, one highlighted: pick by list, not by clicking the view."""
    rows = []
    for i, y in enumerate((7.0, 12.0, 17.0)):
        rows.append(K.line(4.5, y, 8.0, y, color=K.INK))
        rows.append(K.line(9.5, y, 19.5, y, color=K.INK, width=1.4))
    rows.append(K.rect(3.2, 10.0, 17.6, 4.0, r=1.0, color=K.ACCENT, width=1.6))
    return K.svg("\n".join(rows))


def ecv_clipping_box() -> str:
    """Corner brackets around a solid shape: the region that clips.

    Measured at 18 px: the first version (cube outline inside a rectangle) covered
    48% of the canvas and its two nested outlines merged into a blob.  Nested
    outlines are the specific thing that fails at this size, so the box is drawn
    as four corner brackets -- fewer pixels, and it still reads as "a box".
    """
    b = []
    for x, y, dx, dy in ((3.4, 3.4, 1, 1), (20.6, 3.4, -1, 1),
                         (3.4, 20.6, 1, -1), (20.6, 20.6, -1, -1)):
        b.append(K.poly([(x + dx * 5.0, y), (x, y), (x, y + dy * 5.0)],
                        color=K.ACCENT, width=2.0))
    b.append(K.cube(12, 12, r=4.2))
    return K.svg("\n".join(b))


def ecv_clipping_box_single_export() -> str:
    """Corner brackets, one filled block, one arrow out: box out a region, export it."""
    b = []
    for x, y, dx, dy in ((3.4, 4.0, 1, 1), (14.6, 4.0, -1, 1),
                         (3.4, 20.0, 1, -1), (14.6, 20.0, -1, -1)):
        b.append(K.poly([(x + dx * 4.4, y), (x, y), (x, y + dy * 4.4)],
                        color=K.ACCENT, width=2.0))
    b.append(K.rect(6.4, 8.0, 5.2, 8.0, r=1.0, fill=K.INK))
    b.append(K.arrow(16.0, 12, 21.5, 12, color=K.INK, head=2.8))
    return K.svg("\n".join(b))


def ecv_clipping_box_mult_export() -> str:
    return K.svg("\n".join([
        K.cube(9.0, 12, r=4.2),
        K.rect(2.2, 2.6, 14.0, 18.8, r=1.2, color=K.ACCENT, width=1.3),
        K.arrow(15.5, 8.5, 21.5, 8.5, color=K.INK, head=2.4),
        K.arrow(15.5, 15.5, 21.5, 15.5, color=K.INK, head=2.4),
    ]))


# ---------------------------------------------------------------------------
# Filtering / statistics
# ---------------------------------------------------------------------------


def ecv_filter_by_value() -> str:
    """Funnel with the kept value below it: 'keep what passes'."""
    return K.svg("\n".join([
        K.funnel(12, 8.0, w=14.0, h=9.0),
        K.line(9.0, 16.5, 15.0, 16.5, color=K.MUTED, width=1.4),
        K.dot(12, 19.5, 2.0, color=K.ACCENT),
    ]))


def ecv_histogram() -> str:
    return K.svg(K.bars(12, 11))


def ecv_compute_stat() -> str:
    """Histogram plus a sigma: the numbers, not just the shape."""
    return K.svg("\n".join([
        K.bars(9.5, 10.0, heights=(3, 6, 4.5, 7.5)),
        K.poly([(17.0, 8.0), (21.0, 8.0), (17.6, 12.4), (21.2, 12.4)],
               color=K.ACCENT, width=1.8),
    ]))


def ecv_sor() -> str:
    """Cloud with outliers struck out: statistical outlier removal."""
    return K.svg("\n".join([
        K.cloud(12, 11.0, s=0.95),
        K.circle(7.0, 6.0, 1.5, color=K.MUTED, width=1.2),
        K.circle(19.0, 5.5, 1.5, color=K.MUTED, width=1.2),
        K.line(17.4, 15.2, 21.0, 18.8, color=K.ACCENT, width=1.8),
        K.line(21.0, 15.2, 17.4, 18.8, color=K.ACCENT, width=1.8),
    ]))


# ---------------------------------------------------------------------------
# Filtering kernels
# ---------------------------------------------------------------------------


def ecv_gaussian_filter() -> str:
    return K.svg("\n".join([K.bell(12, 12)]))


def ecv_bilateral_filter() -> str:
    """Bell inside a grid: a kernel applied over the field, edges preserved."""
    return K.svg("\n".join([
        K.grid3(12, 12, r=7.2, color=K.MUTED, width=1.2),
        K.bell(12, 12),
    ]))


def ecv_gradient() -> str:
    """Ramp with direction arrow: rate of change across the field."""
    return K.svg("\n".join([
        K.rect(3.0, 7.5, 18.0, 9.0, r=1.4, color=K.MUTED, width=1.2),
        K.poly([(4.2, 15.2), (4.2, 8.8), (9.0, 8.8)], fill="#D6E6F0"),
        K.poly([(4.2, 15.2), (9.0, 15.2), (9.0, 8.8)], fill="#9DC6DF"),
        K.arrow(5.5, 15.6, 18.5, 8.4, color=K.ACCENT, head=2.8),
    ]))


# ---------------------------------------------------------------------------
# Scalar field arithmetic
# ---------------------------------------------------------------------------


def ecv_add_const_sf() -> str:
    """Grid plus a constant: 'add a number to a field'."""
    return K.svg("\n".join([
        K.grid3(10.0, 12.0, r=6.2),
        K.plus(18.5, 7.0, arm=2.6, color=K.ACCENT, width=2.2),
    ]))


def ecv_delete_sf() -> str:
    return K.svg("\n".join([
        K.grid3(10.0, 12.0, r=6.2),
        K.minus(18.5, 7.0, arm=2.6, color=K.ACCENT, width=2.2),
    ]))


def ecv_sf_arithmetic() -> str:
    """Two grids and an operator: field op field."""
    return K.svg("\n".join([
        K.grid3(7.0, 12.0, r=4.6),
        K.grid3(17.0, 12.0, r=4.6, color=K.ACCENT),
        K.line(11.0, 6.0, 11.0, 18.0, color=K.MUTED, width=1.2),
    ]))


# ---------------------------------------------------------------------------
# Merge / boolean
# ---------------------------------------------------------------------------


def ecv_merge() -> str:
    """Two clouds converging into one: merge keeps both, joined."""
    return K.svg("\n".join([
        K.cloud(7.5, 8.0, s=0.60),
        K.cloud(16.5, 8.0, s=0.60),
        K.poly([(7.5, 12.2), (12.0, 15.4), (16.5, 12.2)], color=K.ACCENT, width=1.8),
        K.cloud(12.0, 18.0, s=0.72, color=K.ACCENT),
    ]))


def ecv_union() -> str:
    """Merge several selections and export the result as one new cloud.

    "Union" was read as set union and drawn as two overlapping discs; the actual
    binding is

        <widget name="unionModeButton">
          toolTip  "Export selection as a new cloud"
          statusTip "union selections"

    so the verb is *combine, then export*.  Two source regions joined into one
    accented region with an arrow leaving it states both halves.
    """
    return K.svg("\n".join([
        K.rect(2.6, 6.4, 7.4, 11.2, r=1.1, color=K.MUTED, width=1.5),
        K.rect(2.6, 12.2, 14.0, 5.4, r=1.1, fill=K.ACCENT),
        K.arrow(15.0, 9.4, 21.0, 9.4, color=K.INK, head=2.4),
        K.line(4.6, 20.4, 18.0, 20.4, color=K.MUTED, width=1.4),
    ]))


def ecv_unstack() -> str:
    """A stack separating into layers: split, not delete."""
    return K.svg("\n".join([
        K.poly([(4.0, 8.0), (12.0, 4.0), (20.0, 8.0), (12.0, 12.0)],
               close=True, color=K.INK),
        K.poly([(4.0, 13.0), (12.0, 17.0), (20.0, 13.0)], color=K.ACCENT, width=1.8),
        K.poly([(4.0, 17.5), (12.0, 21.5), (20.0, 17.5)], color=K.MUTED, width=1.4),
    ]))


# ---------------------------------------------------------------------------
# Alignment / registration / transformation
# ---------------------------------------------------------------------------


def ecv_align() -> str:
    """Two shapes snapping to a common axis."""
    return K.svg("\n".join([
        K.rect(3.5, 4.0, 7.0, 7.0, r=1.2, color=K.INK),
        K.rect(13.5, 13.0, 7.0, 7.0, r=1.2, color=K.ACCENT),
        K.line(7.0, 12.0, 17.0, 12.0, color=K.MUTED, width=1.2),
        K.line(12.0, 7.0, 12.0, 17.0, color=K.MUTED, width=1.2),
    ]))


def ecv_register() -> str:
    """Four correspondences pinned: registration, a stronger align."""
    parts = [
        K.rect(2.8, 2.8, 9.0, 9.0, r=1.2, color=K.INK),
        K.rect(12.2, 12.2, 9.0, 9.0, r=1.2, color=K.ACCENT),
    ]
    for x, y in ((5.5, 5.5), (9.3, 5.5), (5.5, 9.3), (9.3, 9.3)):
        parts.append(K.dot(x, y, 1.2))
    parts.append(K.arrow(10.5, 10.5, 13.5, 13.5, color=K.MUTED, head=2.2))
    return K.svg("\n".join(parts))


def ecv_interactive_transformation() -> str:
    """A shape with a rotate handle: direct manipulation."""
    return K.svg("\n".join([
        K.rect(6.0, 6.0, 11.0, 11.0, r=1.4, color=K.INK),
        K.circle(19.0, 19.0, 2.2, color=K.ACCENT, width=1.8),
        K.line(17.0, 17.0, 14.0, 14.0, color=K.ACCENT, width=1.4),
    ]))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def ecv_sample_cloud() -> str:
    """A cloud with a subset highlighted: sample the whole body."""
    return K.svg("\n".join([
        K.cloud(12, 12.0, s=0.95),
        K.dot(9.0, 11.0, 1.6),
        K.dot(13.5, 9.5, 1.6),
        K.dot(15.5, 13.0, 1.6),
    ]))


def ecv_sample_points() -> str:
    """Three filled nodes and one hollow: a regular lattice thinned out.

    Measured at 18 px: nine dots on a visible grid covered 53% of the canvas and
    read as noise; an earlier "remove the centre" attempt used a wrong index and
    covered 55%.  Four nodes, no grid, one hollow to show the removal -- two
    elements of information instead of forty pixels of texture.
    """
    parts = [
        K.dot(6.5, 6.5, 2.6),
        K.dot(17.5, 6.5, 2.6),
        K.dot(6.5, 17.5, 2.6),
    ]
    # the removed node: hollow, muted, so it reads as absence not as a mark
    parts.append(K.circle(17.5, 17.5, 2.6, color=K.MUTED, width=1.6))
    return K.svg("\n".join(parts))


def ecv_cloud_cloud_distance() -> str:
    return K.svg("\n".join([
        K.cloud(7.5, 8.5, s=0.60),
        K.cloud(16.5, 15.5, s=0.60, color=K.ACCENT),
        K.arrow(8.8, 11.0, 15.2, 13.0, color=K.INK, head=2.4),
    ]))


def ecv_cloud_mesh_distance() -> str:
    return K.svg("\n".join([
        K.cloud(7.0, 8.0, s=0.58),
        K.cube(16.5, 15.0, r=5.0, color=K.ACCENT),
        K.arrow(8.6, 10.6, 14.4, 13.0, color=K.INK, head=2.4),
    ]))


# ---------------------------------------------------------------------------
# Camera / zoom
# ---------------------------------------------------------------------------


def ecv_global_zoom() -> str:
    """Magnifier over a globe: fit everything in view."""
    return K.svg("\n".join([
        K.circle(11.0, 11.0, 6.0, color=K.INK, width=1.8),
        K.line(15.4, 15.4, 20.0, 20.0, color=K.INK, width=2.2),
        K.line(5.4, 11.0, 16.6, 11.0, color=K.ACCENT, width=1.2),
        K.line(11.0, 5.4, 11.0, 16.6, color=K.ACCENT, width=1.2),
    ]))


def ecv_ortho_projection() -> str:
    """Cube with parallel rays: no convergence."""
    return K.svg("\n".join([
        K.cube(14.5, 12.0, r=5.2),
        K.line(2.5, 7.0, 9.0, 7.0, color=K.ACCENT, width=1.4),
        K.line(2.5, 12.0, 9.0, 12.0, color=K.ACCENT, width=1.4),
        K.line(2.5, 17.0, 9.0, 17.0, color=K.ACCENT, width=1.4),
    ]))


def ecv_perspective_projection() -> str:
    """Cube with converging rays: a vanishing point."""
    return K.svg("\n".join([
        K.cube(15.0, 12.5, r=5.0),
        K.line(2.5, 5.0, 10.4, 10.2, color=K.ACCENT, width=1.4),
        K.line(2.5, 12.5, 10.4, 12.5, color=K.ACCENT, width=1.4),
        K.line(2.5, 20.0, 10.4, 14.8, color=K.ACCENT, width=1.4),
    ]))


# ---------------------------------------------------------------------------
# Registry: name -> builder.  Only icons whose meaning is unambiguous are drawn
# here; anything whose function cannot be stated in one sentence keeps its
# original artwork rather than getting an invented mark.
# ---------------------------------------------------------------------------

REGISTRY = {
    "ecvPlus": ecv_plus,
    "ecvMinus": ecv_minus,
    "ecvdelete": ecv_delete,
    "ecvExit": ecv_exit,
    "ecvPencil": ecv_pencil,
    "ecvClone": ecv_clone,
    "ecvConsole": ecv_console,
    "ecvPointPicking": ecv_point_picking,
    "ecvPointListPicking": ecv_point_list_picking,
    "ecvClippingBox": ecv_clipping_box,
    "ecvClippingBoxSingleExport": ecv_clipping_box_single_export,
    "ecvClippingBoxMultExport": ecv_clipping_box_mult_export,
    "ecvFilterByValue": ecv_filter_by_value,
    "ecvHistogram": ecv_histogram,
    "ecvComputeStat": ecv_compute_stat,
    "ecvSor": ecv_sor,
    "ecvGaussianFilter": ecv_gaussian_filter,
    "ecvBilateralFilter": ecv_bilateral_filter,
    "ecvGradient": ecv_gradient,
    "ecvAddConstSF": ecv_add_const_sf,
    "ecvDeleteSF": ecv_delete_sf,
    "ecvSfArithmetic": ecv_sf_arithmetic,
    "ecvMerge": ecv_merge,
    "ecvUnion": ecv_union,
    "ecvUnstack": ecv_unstack,
    "ecvAlign": ecv_align,
    "ecvRegister": ecv_register,
    "ecvInteractiveTransformation": ecv_interactive_transformation,
    "ecvSampleCloud": ecv_sample_cloud,
    "ecvSamplePoints": ecv_sample_points,
    "ecvCloudCloudDistance": ecv_cloud_cloud_distance,
    "ecvCloudMeshDistance": ecv_cloud_mesh_distance,
    "ecvGlobalZoom": ecv_global_zoom,
    "ecvOrthoProjection": ecv_ortho_projection,
    "ecvPerspectiveProjection": ecv_perspective_projection,
}
