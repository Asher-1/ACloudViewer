"""Redesigned tree, type and view icons: same canvas, same stroke, same ink.

These three families are read at 16-20 px inside the DB tree, the scalar-field
picker and the view toolbar, so they follow the same hard limits as the toolbar
set: 24x24 canvas, 2.0 stroke, detail floor of 3 units, at most four visual
elements per icon.

The "locked" variants are the base icon with a lock badge in the lower right.
Drawing them as separate artwork would let the two versions drift apart, so they
are composed: same silhouette, one extra mark.
"""

from __future__ import annotations

from . import iconset as K


# ---------------------------------------------------------------------------
# Lock badge, shared by every *Locked variant
# ---------------------------------------------------------------------------


def _lock(cx=18.6, cy=18.4, s=1.0, *, color=None) -> str:
    """Small padlock: a body and a shackle, sized to stay legible at 16 px."""
    c = color or K.ACCENT_DEEP
    w, h = 4.6 * s, 3.6 * s
    return "\n".join([
        # shackle
        f'  <path d="M {cx - 1.5 * s} {cy - h / 2} v {-1.5 * s} '
        f'a {1.5 * s} {1.5 * s} 0 0 1 {3.0 * s} 0 v {1.5 * s}" stroke="{c}"/>',
        K.rect(cx - w / 2, cy - h / 2, w, h, r=0.8 * s, fill=c),
    ])


# ---------------------------------------------------------------------------
# DB tree symbols
# ---------------------------------------------------------------------------


def db_cloud() -> str:
    """Point cloud: loose dots in a cloud-shaped grouping."""
    return K.svg("\n".join([
        K.cloud(12, 12, s=0.94, color=K.INK),
        K.dot(9.0, 11.4, 1.5),
        K.dot(13.4, 9.6, 1.5),
        K.dot(15.4, 13.2, 1.5),
    ]))


def _mesh_plane(cx=12.0, cy=12.0, w=8.8, h=6.6, *, highlight=0) -> list[str]:
    """A tessellated quad: four rows of triangles sharing a vertex grid.

    Iteration history, kept because each failure was a metaphor error rather
    than a tuning error:

    1. Upright triangle  -> at 18 px this is the universal warning sign, so four
       DB icons read as "warning".
    2. Quad with two crossing diagonals -> at 96 px the diagonals form an X and
       the mark reads as a kite, not as a surface.

    What works is the structure a real mesh has: a vertex grid, with triangles
    drawn between alternate grid nodes.  The grid is 3x2 so the triangle count
    stays visible at 18 px without turning into noise.
    """
    top = (cx, cy - h)
    right = (cx + w, cy - h * 0.36)
    bottom = (cx, cy + h)
    left = (cx - w, cy + h * 0.36)

    def at(u, v):
        """Bilinear point inside the quad: u across, v down."""
        x = (1 - u) * (1 - v) * top[0] + u * (1 - v) * right[0] \
            + u * v * bottom[0] + (1 - u) * v * left[0]
        y = (1 - u) * (1 - v) * top[1] + u * (1 - v) * right[1] \
            + u * v * bottom[1] + (1 - u) * v * left[1]
        return (x, y)

    # 2x1 cells = 4 triangles.  A 3x2 grid (12 triangles) measured as noise at
    # 18 px: the internal edges fall below one device pixel and merge into grey.
    nu, nv = 2, 1
    nodes = {(i, j): at(i / nu, j / nv) for i in range(nu + 1) for j in range(nv + 1)}

    parts = [K.poly([top, right, bottom, left], close=True, color=K.INK)]

    # triangles: each cell split into two, the accent marks only `highlight`
    k = 0
    for i in range(nu):
        for j in range(nv):
            a, b = nodes[(i, j)], nodes[(i + 1, j)]
            c, d = nodes[(i + 1, j + 1)], nodes[(i, j + 1)]
            for tri in ((a, b, c), (a, c, d)):
                if k == highlight:
                    parts.append(K.poly(list(tri), close=True, fill=K.ACCENT,
                                        color=K.ACCENT, width=0))
                else:
                    parts.append(K.poly(list(tri), close=True,
                                        color=K.MUTED, width=1.0))
                k += 1
    return parts


def db_mesh() -> str:
    return K.svg("\n".join(_mesh_plane()))


def db_mesh_locked() -> str:
    return K.svg("\n".join(_mesh_plane(cx=10.4, cy=10.6, w=7.4, h=5.4)
                            + [_lock(18.8, 19.0, 0.92)]))


def db_sub_mesh() -> str:
    """Mesh with one triangle highlighted: a sub-mesh is a selected part."""
    return K.svg("\n".join(_mesh_plane(highlight=2)))


def db_sub_mesh_locked() -> str:
    return K.svg("\n".join(_mesh_plane(cx=10.4, cy=10.6, w=7.4, h=5.4, highlight=2)
                            + [_lock(18.8, 19.0, 0.92)]))


def db_octree() -> str:
    """Cube split into eight: the octree's defining structure."""
    return K.svg("\n".join([
        K.cube(12, 12, r=8.4, color=K.INK),
        K.line(12, 3.6, 12, 20.4, color=K.INK, width=1.4),
        K.line(4.7, 7.8, 19.3, 7.8, color=K.INK, width=1.4),
        K.line(4.7, 16.2, 19.3, 16.2, color=K.INK, width=1.4),
    ]))


def db_octree_locked() -> str:
    return K.svg("\n".join([
        K.cube(10.6, 11.4, r=7.2, color=K.INK),
        K.line(10.6, 4.2, 10.6, 18.6, color=K.INK, width=1.3),
        K.line(4.4, 7.8, 16.8, 7.8, color=K.INK, width=1.3),
        _lock(19.2, 18.8, 0.92),
    ]))


def db_image() -> str:
    """Picture: frame, horizon and one sun -- three elements, no more."""
    return K.svg("\n".join([
        K.rect(3.2, 5.0, 17.6, 14.0, r=1.6, color=K.INK),
        K.poly([(4.6, 17.4), (10.0, 11.2), (14.2, 17.4)], close=True,
               fill=K.ACCENT, color=K.ACCENT, width=0),
        K.dot(16.4, 9.4, 1.8),
    ]))


def db_material() -> str:
    """Material: a shaded sphere, the universal 'surface appearance' mark."""
    return K.svg("\n".join([
        K.circle(12, 12, 8.2, color=K.INK),
        f'  <path d="M 5.2 15.4 a 8.2 8.2 0 0 0 13.6 0" '
        f'stroke="{K.ACCENT}" fill="none"/>',
        K.circle(9.0, 9.0, 2.2, fill=K.ACCENT),
    ]))


def db_cam_sensor() -> str:
    """Camera: body plus a frustum, because this entity carries a pose."""
    return K.svg("\n".join([
        K.rect(2.8, 8.0, 8.0, 7.4, r=1.2, color=K.INK),
        K.poly([(10.8, 9.4), (20.8, 4.6), (20.8, 18.8), (10.8, 14.0)],
               close=True, color=K.ACCENT, width=1.8),
    ]))


def db_gbl_sensor() -> str:
    """GBL sensor: a scan head radiating ranges."""
    return K.svg("\n".join([
        K.rect(9.0, 3.4, 6.0, 5.0, r=1.0, fill=K.INK),
        K.poly([(12, 9.0), (12, 12.0)], color=K.INK, width=0),
        K.line(12, 9.2, 12, 11.6, color=K.INK),
        K.line(12, 12.0, 4.4, 20.2, color=K.ACCENT, width=1.6),
        K.line(12, 12.0, 12.0, 21.0, color=K.ACCENT, width=1.6),
        K.line(12, 12.0, 19.6, 20.2, color=K.ACCENT, width=1.6),
    ]))


def db_file() -> str:
    """Document with a folded corner: the standard file mark."""
    return K.svg("\n".join([
        K.poly([(5.2, 2.8), (14.0, 2.8), (19.2, 8.0), (19.2, 21.2), (5.2, 21.2)],
               close=True, color=K.INK),
        K.poly([(14.0, 2.8), (14.0, 8.0), (19.2, 8.0)], color=K.INK, width=1.6),
        K.line(8.4, 13.0, 16.0, 13.0, color=K.ACCENT, width=1.6),
        K.line(8.4, 16.6, 13.4, 16.6, color=K.ACCENT, width=1.6),
    ]))


def db_circle() -> str:
    """A 2D circle primitive: outline plus centre, nothing else."""
    return K.svg("\n".join([
        K.circle(12, 12, 8.4, color=K.INK),
        K.dot(12, 12, 2.0),
    ]))


def db_misc_geom_locked() -> str:
    """Misc geometry: a polygon with a vertex handle."""
    return K.svg("\n".join([
        K.poly([(6.0, 4.6), (16.2, 6.4), (15.0, 16.0), (5.0, 14.0)],
               close=True, color=K.INK),
        K.circle(16.2, 6.4, 2.0, fill=K.ACCENT),
        _lock(19.0, 18.8, 0.92),
    ]))


# ---------------------------------------------------------------------------
# Scalar-field type icons: these appear in a picker, so they must differ from
# each other at 16 px by SILHOUETTE, not by colour.
# ---------------------------------------------------------------------------


def type_normal() -> str:
    """No scalar field: a hollow dot."""
    return K.svg(K.circle(12, 12, 6.4, color=K.INK, width=2.0))


def type_sf() -> str:
    """Scalar field: a frame with a diagonal ramp, two elements total.

    Measured across three attempts: a filled block hit 56% ink, a three-step
    strip 51%, both because a bordered frame plus interior bands is five
    elements at 18 px.  Frame + one diagonal wedge is the minimum that still
    reads as "a gradient lives inside this field".
    """
    return K.svg("\n".join([
        K.rect(3.6, 4.8, 16.8, 14.4, r=1.4, color=K.INK),
        K.poly([(5.0, 17.8), (5.0, 14.6), (19.0, 8.0), (19.0, 17.8)],
               close=True, fill=K.ACCENT, color=K.ACCENT, width=0),
    ]))


def type_positive_sf() -> str:
    """Positive-only field: the same strip, but nothing below the baseline."""
    return K.svg("\n".join([
        K.rect(3.4, 4.6, 17.2, 14.8, r=1.4, color=K.INK),
        K.rect(6.2, 6.6, 3.2, 5.0, r=0.6, fill=K.ACCENT, color=K.ACCENT, width=0),
        K.rect(10.0, 5.4, 4.2, 6.2, r=0.6, fill=K.ACCENT_DEEP, color=K.ACCENT_DEEP, width=0),
        K.line(3.4, 15.4, 20.6, 15.4, color=K.MUTED, width=1.3),
    ]))


def type_rgb_color() -> str:
    """RGB: three overlapping discs, the additive-primaries mark."""
    return K.svg("\n".join([
        K.circle(9.2, 9.6, 5.0, color=K.INK, width=1.8),
        K.circle(14.8, 9.6, 5.0, color=K.INK, width=1.8),
        K.circle(12.0, 14.6, 5.0, fill=K.ACCENT, color=K.ACCENT, width=0),
    ]))


def type_gray_color() -> str:
    """Greyscale: a stepped ramp, not a smooth one, so it differs from a
    scalar-field gradient at small size."""
    return K.svg("\n".join([
        K.rect(3.4, 8.4, 4.4, 7.2, r=0.8, color=K.INK, width=1.4),
        K.rect(8.6, 8.4, 4.4, 7.2, r=0.8, fill=K.MUTED, color=K.MUTED, width=0),
        K.rect(13.8, 8.4, 4.4, 7.2, r=0.8, fill=K.INK, color=K.INK, width=0),
    ]))


def _axis(letter_hint: str) -> str:
    """An axis arrow with its triple marker: X, Y and Z differ by rotation."""
    return K.svg(letter_hint)


def type_x() -> str:
    return K.svg("\n".join([
        K.arrow(4.6, 12.0, 19.4, 12.0, color=K.ACCENT),
        K.line(6.0, 18.6, 18.0, 18.6, color=K.MUTED, width=1.2),
    ]))


def type_y() -> str:
    return K.svg("\n".join([
        K.arrow(4.6, 19.4, 19.4, 4.6, color=K.ACCENT),
        K.line(6.0, 18.6, 18.0, 18.6, color=K.MUTED, width=1.2),
    ]))


def type_z() -> str:
    return K.svg("\n".join([
        K.arrow(12.0, 19.4, 12.0, 4.6, color=K.ACCENT),
        K.line(6.0, 18.6, 18.0, 18.6, color=K.MUTED, width=1.2),
    ]))


# ---------------------------------------------------------------------------
# View icons: a cube with the visible face marked, plus the axis arrow.
# The face highlight is what tells Xpos from Xneg at 18 px, not a label.
# ---------------------------------------------------------------------------


def _view(face: str, positive: bool) -> str:
    """Cube with one face filled; `face` is 'x', 'y' or 'z'."""
    r = 7.4
    cx, cy = 11.4, 12.0
    top = (cx, cy - r)
    right = (cx + r * 0.866, cy - r * 0.5)
    br = (cx + r * 0.866, cy + r * 0.5)
    bot = (cx, cy + r)
    bl = (cx - r * 0.866, cy + r * 0.5)
    tl = (cx - r * 0.866, cy - r * 0.5)
    o = (cx, cy)

    faces = {
        "x": [o, right, br, bot],        # right plane
        "y": [tl, o, bot, bl],           # left plane
        "z": [top, right, o, tl],        # top plane
    }
    body = [K.poly([top, right, br, bot, bl, tl], close=True, color=K.INK)]
    body.append(K.poly(faces[face], close=True, fill=K.ACCENT,
                       color=K.ACCENT, width=1.0))
    # the axis arrow shows direction; positive points away from the origin
    if face == "x":
        pts = ((20.0, 6.2), (13.6, 6.2)) if positive else ((13.6, 6.2), (20.0, 6.2))
    elif face == "y":
        pts = ((20.0, 6.2), (13.6, 6.2)) if positive else ((13.6, 6.2), (20.0, 6.2))
    else:
        pts = ((20.0, 4.4), (20.0, 10.4)) if positive else ((20.0, 10.4), (20.0, 4.4))
    body.append(K.arrow(pts[0][0], pts[0][1], pts[1][0], pts[1][1], head=2.4))
    return K.svg("\n".join(body))


def view_x_pos() -> str:
    return _view("x", True)


def view_x_neg() -> str:
    return _view("x", False)


def view_y_pos() -> str:
    return _view("y", True)


def view_y_neg() -> str:
    return _view("y", False)


def view_z_pos() -> str:
    return _view("z", True)


def view_z_neg() -> str:
    return _view("z", False)


def view_iso_1() -> str:
    """Isometric 1: all three faces, top weighted."""
    return K.svg(K.cube(12, 12, r=8.6))


def view_iso_2() -> str:
    """Isometric 2: the mirror orientation."""
    return K.svg("\n".join([
        K.cube(12, 12, r=7.6),
        K.poly([(12, 4.4), (18.6, 8.2), (12, 12.0), (5.4, 8.2)], close=True,
               fill=K.ACCENT, color=K.ACCENT, width=0),
    ]))


# ---------------------------------------------------------------------------
# Small action icons (16 px in the property panels)
# ---------------------------------------------------------------------------


def small_validate() -> str:
    return K.svg(K.poly([(5.0, 12.6), (10.0, 17.4), (19.2, 6.8)],
                        color=K.ACCENT, width=2.6))


def small_cancel() -> str:
    return K.svg("\n".join([
        K.line(6.0, 6.0, 18.0, 18.0, color=K.INK, width=2.4),
        K.line(18.0, 6.0, 6.0, 18.0, color=K.INK, width=2.4),
    ]))


def small_trash() -> str:
    return K.svg(K.trash(12, 12))


def small_reset() -> str:
    """Circular arrow: reset returns to the previous state."""
    return K.svg("\n".join([
        f'  <path d="M 19.0 12 a 7.0 7.0 0 1 1 -7.0 -7.0" stroke="{K.INK}"/>',
        K.arrow(12.0, 3.6, 12.6, 6.4, color=K.INK, head=2.6),
        K.dot(12, 12, 2.0),
    ]))


def small_revert() -> str:
    """Mirror of reset: return to saved state, arrow pointing back."""
    return K.svg("\n".join([
        f'  <path d="M 5.0 12 a 7.0 7.0 0 1 0 7.0 -7.0" stroke="{K.INK}"/>',
        K.arrow(12.0, 3.6, 11.4, 6.4, color=K.INK, head=2.6),
        K.rect(9.6, 9.6, 4.8, 4.8, r=1.0, fill=K.ACCENT),
    ]))


def small_pause() -> str:
    return K.svg("\n".join([
        K.rect(7.6, 5.6, 3.6, 12.8, r=1.0, fill=K.INK),
        K.rect(12.8, 5.6, 3.6, 12.8, r=1.0, fill=K.INK),
    ]))


def small_basket() -> str:
    return K.svg("\n".join([
        K.poly([(3.6, 9.4), (20.4, 9.4), (17.6, 19.6), (6.4, 19.6)],
               close=True, color=K.INK),
        K.poly([(8.6, 9.4), (12.0, 4.4), (15.4, 9.4)], color=K.ACCENT, width=1.8),
    ]))


def small_csv_file() -> str:
    return K.svg("\n".join([
        K.poly([(5.6, 3.0), (14.0, 3.0), (18.8, 7.8), (18.8, 21.0), (5.6, 21.0)],
               close=True, color=K.INK),
        K.line(8.6, 12.6, 15.8, 12.6, color=K.ACCENT, width=1.6),
        K.line(8.6, 16.2, 15.8, 16.2, color=K.ACCENT, width=1.6),
        K.line(11.2, 12.6, 11.2, 19.0, color=K.ACCENT, width=1.6),
        K.line(13.6, 12.6, 13.6, 19.0, color=K.ACCENT, width=1.6),
    ]))


def small_point_distance() -> str:
    return K.svg("\n".join([
        K.dot(6.2, 17.0, 2.4, color=K.INK),
        K.dot(17.8, 7.0, 2.4, color=K.INK),
        K.arrow(8.4, 15.6, 15.6, 8.4, color=K.ACCENT, head=2.4),
    ]))


def small_points_angle() -> str:
    return K.svg("\n".join([
        K.line(5.0, 19.0, 19.0, 19.0, color=K.INK),
        K.line(5.0, 19.0, 17.6, 6.4, color=K.INK),
        f'  <path d="M 13.0 19.0 a 8.0 8.0 0 0 0 -2.4 -5.6" stroke="{K.ACCENT}"/>',
    ]))


def small_point_properties() -> str:
    return K.svg("\n".join([
        K.circle(8.4, 12.0, 3.0, color=K.INK),
        K.line(13.0, 12.0, 20.0, 12.0, color=K.INK, width=1.8),
        K.line(13.0, 7.6, 17.6, 7.6, color=K.MUTED, width=1.4),
        K.line(13.0, 16.4, 17.6, 16.4, color=K.MUTED, width=1.4),
    ]))


def small_rectangle_select() -> str:
    return K.svg("\n".join([
        K.rect(4.0, 6.4, 16.0, 11.2, r=1.4, color=K.ACCENT, width=1.8),
        K.poly([(4.0, 6.4), (4.0, 3.4), (7.0, 3.4)], color=K.INK, width=2.2),
        K.poly([(20.0, 17.6), (20.0, 20.6), (17.0, 20.6)], color=K.INK, width=2.2),
    ]))


def small_polygon_select() -> str:
    return K.svg("\n".join([
        K.poly([(6.0, 4.0), (19.0, 7.0), (17.0, 19.0), (5.0, 16.0)],
               close=True, color=K.ACCENT, width=1.8),
        K.dot(6.0, 4.0, 1.8, color=K.INK),
        K.dot(19.0, 7.0, 1.8, color=K.INK),
    ]))


def small_sphere() -> str:
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.2, color=K.INK),
        f'  <ellipse cx="12" cy="12" rx="8.2" ry="3.4" stroke="{K.ACCENT}"/>',
    ]))


def small_split() -> str:
    return K.svg("\n".join([
        K.line(12.0, 3.6, 12.0, 9.0, color=K.INK, width=2.0),
        K.line(12.0, 15.0, 12.0, 20.4, color=K.INK, width=2.0),
        K.poly([(8.4, 9.0), (15.6, 9.0), (12.0, 13.0)], close=True, fill=K.ACCENT),
    ]))


def small_segment_in() -> str:
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 7.6, color=K.INK),
        K.arrow(3.0, 12.0, 10.0, 12.0, color=K.ACCENT, head=2.6),
    ]))


def small_segment_out() -> str:
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 7.6, color=K.INK),
        K.arrow(14.0, 12.0, 21.0, 12.0, color=K.ACCENT, head=2.6),
    ]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------



def db_export_polyline() -> str:
    """Send the traced polyline back to the main database.

    ``dbContainerSymbol3`` is *not* a tree icon despite the family name it was
    filed under.  It has exactly one consumer:

        <widget name="saveToolButton">
          toolTip "Export current polyline to the main DB"

    The real group icon is ``dbContainerSymbol`` (loaded by ``ecvDBRoot.cpp`` at
    line 225).  Drawing a folder here would have given the trace dialog a
    "group" glyph for an export command -- the same class of error as a wrong
    tooltip, only harder to notice because it looks plausible.
    """
    return K.svg("\n".join([
        K.poly([(2.6, 16.0), (6.4, 12.0), (10.6, 14.6), (15.0, 8.4)],
               color=K.ACCENT, width=2.0),
        K.dot(2.6, 16.0, 1.5),
        K.dot(10.6, 14.6, 1.5),
        K.dot(15.0, 8.4, 1.5),
        K.arrow(17.0, 11.4, 21.4, 7.0, color=K.INK, head=2.4),
        K.rect(3.0, 18.6, 18.0, 3.0, r=0.8, fill=K.MUTED),
    ]))


REGISTRY = {
    # db tree
    "dbCloudSymbol": db_cloud,
    "dbCircle": db_circle,
    "dbFileSymbol": db_file,
    "dbCamSensorSymbol": db_cam_sensor,
    "dbGBLSensorSymbol": db_gbl_sensor,
    "dbImageSymbol": db_image,
    "dbMaterialSymbol": db_material,
    "dbMeshSymbol": db_mesh,
    "dbMeshSymbolLocked": db_mesh_locked,
    "dbMiscGeomSymbolLocked": db_misc_geom_locked,
    "dbOctreeSymbol": db_octree,
    "dbOctreeSymbolLocked": db_octree_locked,
    "dbSubMeshSymbol": db_sub_mesh,
    "dbSubMeshSymbolLocked": db_sub_mesh_locked,
    # scalar-field types
    "typeNormal": type_normal,
    "typeSF": type_sf,
    "typePositiveSF": type_positive_sf,
    "typeRgbCcolor": type_rgb_color,
    "typeGrayColor": type_gray_color,
    "typeXCoordinate": type_x,
    "typeYCoordinate": type_y,
    "typeZCoordinate": type_z,
    # views
    "ecvViewXpos": view_x_pos,
    "ecvViewXneg": view_x_neg,
    "ecvViewYpos": view_y_pos,
    "ecvViewYneg": view_y_neg,
    "ecvViewZpos": view_z_pos,
    "ecvViewZneg": view_z_neg,
    "ecvViewIso1": view_iso_1,
    "ecvViewIso2": view_iso_2,
    # small actions
    "smallValidate": small_validate,
    "smallCancel": small_cancel,
    "smallTrash": small_trash,
    "smallReset": small_reset,
    "smallRevert": small_revert,
    "smallPause": small_pause,
    "smallBasket": small_basket,
    "smallCSVFile": small_csv_file,
    "smallPointDistance": small_point_distance,
    "smallPointsAngle": small_points_angle,
    "smallPointProperties": small_point_properties,
    "smallRectangleSelect": small_rectangle_select,
    "smallPolygonSelect": small_polygon_select,
    "smallSphere": small_sphere,
    "smallSplit": small_split,
    "smallSegmentIn": small_segment_in,
    "smallSegmentOut": small_segment_out,
}


# ---------------------------------------------------------------------------
# Root-level db symbols: the DB tree's entity marks (PNG in the repo).
# Same grid, same ink; the lock badge is shared with the *Locked variants above.
# ---------------------------------------------------------------------------


def db_hobject() -> str:
    """Generic DB entity: a stacked layer, the 'any object' mark."""
    return K.svg("\n".join([
        K.poly([(12.0, 3.6), (20.4, 8.4), (12.0, 13.2), (3.6, 8.4)],
               close=True, color=K.INK),
        K.poly([(3.6, 12.4), (12.0, 17.2), (20.4, 12.4)], color=K.ACCENT, width=1.8),
        K.poly([(3.6, 16.4), (12.0, 21.2), (20.4, 16.4)], color=K.MUTED, width=1.4),
    ]))


def db_hobject_locked() -> str:
    return K.svg("\n".join([
        K.poly([(10.4, 3.6), (17.6, 7.8), (10.4, 12.0), (3.2, 7.8)],
               close=True, color=K.INK),
        K.poly([(3.2, 11.2), (10.4, 15.4), (17.6, 11.2)], color=K.ACCENT, width=1.6),
        _lock(19.0, 19.0, 0.92),
    ]))


def db_container() -> str:
    """Group folder: a box that holds other entities."""
    return K.svg("\n".join([
        K.poly([(3.2, 7.4), (9.6, 7.4), (11.2, 9.6), (20.8, 9.6), (20.8, 19.6), (3.2, 19.6)],
               close=True, color=K.INK),
        K.line(3.2, 12.4, 20.8, 12.4, color=K.ACCENT, width=1.6),
    ]))


def db_container_locked() -> str:
    return K.svg("\n".join([
        K.poly([(2.6, 6.6), (8.4, 6.6), (9.8, 8.6), (18.0, 8.6), (18.0, 17.4), (2.6, 17.4)],
               close=True, color=K.INK),
        K.line(2.6, 11.0, 18.0, 11.0, color=K.ACCENT, width=1.5),
        _lock(19.0, 19.0, 0.92),
    ]))


def db_cloud_locked() -> str:
    return K.svg("\n".join([
        K.cloud(10.4, 11.0, s=0.80),
        _lock(19.0, 19.0, 0.92),
    ]))


def db_label() -> str:
    """Label: a tag with a punched hole, the standard 'annotate' mark."""
    return K.svg("\n".join([
        K.poly([(3.4, 9.0), (11.4, 9.0), (20.6, 12.0), (11.4, 15.0), (3.4, 15.0)],
               close=True, color=K.INK),
        K.circle(6.4, 12.0, 1.5, fill=K.ACCENT),
    ]))


def db_area_label() -> str:
    """Area label: a tag pinned inside a bounded region.

    The first version drew a corner bracket over a box, and at 18 px the two
    marks read as a stray arrow head rather than as "a label in an area".
    A tag shape whose point touches the region's edge says both things with one
    silhouette.
    """
    return K.svg("\n".join([
        K.rect(3.2, 3.6, 17.6, 12.4, r=1.2, color=K.MUTED, width=1.3),
        K.poly([(7.0, 19.6), (12.4, 14.6), (17.8, 19.6), (17.8, 19.6)],
               color=K.ACCENT, width=0),
        K.poly([(7.0, 19.6), (17.8, 19.6), (17.8, 15.0), (12.4, 10.6), (7.0, 15.0)],
               close=True, color=K.ACCENT, fill=K.ACCENT, width=0),
    ]))


def db_calibrated_image() -> str:
    """Calibrated image: picture frame whose sensor mark is the centre cross.

    The first attempt filled a diamond in the middle, which read as a decorative
    lozenge.  A crosshair with a ring is the convention for a calibrated optical
    sensor and survives 18 px because it is one mark, not two.
    """
    return K.svg("\n".join([
        K.rect(3.2, 4.6, 17.6, 14.8, r=1.4, color=K.INK),
        K.circle(12.0, 12.0, 4.0, color=K.ACCENT, width=1.6),
        K.line(12.0, 6.2, 12.0, 17.8, color=K.ACCENT, width=1.4),
        K.line(6.0, 12.0, 18.0, 12.0, color=K.ACCENT, width=1.4),
    ]))


def db_polyline() -> str:
    """Polyline: a broken line through vertices."""
    return K.svg("\n".join([
        K.poly([(3.4, 17.4), (8.6, 6.6), (14.6, 14.0), (20.6, 4.8)],
               color=K.INK),
        K.dot(8.6, 6.6, 1.8),
        K.dot(14.6, 14.0, 1.8),
    ]))


def db_viewport() -> str:
    """Viewport: a frame with a camera frustum inside."""
    return K.svg("\n".join([
        K.rect(2.8, 4.8, 18.4, 14.4, r=1.4, color=K.INK),
        K.poly([(9.4, 16.4), (17.6, 9.2), (17.6, 16.4)], close=True,
               fill=K.ACCENT, color=K.ACCENT, width=0),
    ]))


def db_lock_symbol() -> str:
    """The lock on its own, at full size."""
    return K.svg("\n".join([
        f'  <path d="M 8.4 11.0 v -3.4 a 3.6 3.6 0 0 1 7.2 0 v 3.4" stroke="{K.INK}"/>',
        K.rect(6.0, 11.0, 12.0, 9.4, r=1.6, fill=K.ACCENT),
    ]))


def db_misc_geom() -> str:
    """Misc geometry: a polygon with a vertex handle."""
    return K.svg("\n".join([
        K.poly([(6.0, 4.6), (17.6, 6.4), (16.0, 18.4), (5.0, 15.0)],
               close=True, color=K.INK),
        K.circle(17.6, 6.4, 2.2, fill=K.ACCENT),
    ]))


REGISTRY.update({
    "dbCloudSymbol": db_cloud,
    "dbCloudSymbolLocked": db_cloud_locked,
    "dbHObjectSymbol": db_hobject,
    "dbHObjectSymbolLocked": db_hobject_locked,
    "dbContainerSymbol3": db_export_polyline,
    "dbContainerSymbolLocked": db_container_locked,
    "dbLabelSymbol": db_label,
    "dbAreaLabelSymbol": db_area_label,
    "dbCalibratedImageSymbol": db_calibrated_image,
    "dbPolylineSymbol": db_polyline,
    "dbViewportSymbol": db_viewport,
    "dbLockSymbol": db_lock_symbol,
    "dbMiscGeomSymbol": db_misc_geom,
})
