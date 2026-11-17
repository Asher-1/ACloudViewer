"""Root-level application icons: menus, primitives, and window controls.

These are the icons under ``app/Resources/images/*.png`` that the ``.ui`` files
reference directly.  Unlike the toolbar family they are a grab bag -- primitives,
menu glyphs, window controls, legacy PixVue imports -- so the shared kit is used
only for what it is good at (a 24x24 grid, one ink, one accent) and each drawing
is chosen from what the referenced ``QAction`` says the item does.

Names that carry no meaning on their own (``3D3``, ``clean_72px_27328``,
``Wavefrom``, ``ecvicp``) were resolved from the enclosing ``QAction`` text or
tooltip rather than guessed.
"""

from __future__ import annotations

import math

from . import iconset as K


# ---------------------------------------------------------------------------
# Window / application
# ---------------------------------------------------------------------------


def new_3d_view() -> str:
    """Two offset view panels: a second independent 3D view."""
    return K.svg("\n".join([
        K.rect(2.8, 5.6, 12.0, 10.0, r=1.2, color=K.MUTED, width=1.4),
        K.rect(9.2, 8.4, 12.0, 10.0, r=1.2, color=K.ACCENT, width=1.8),
        K.dot(12.2, 11.4, 1.4),
    ]))


def full_screen() -> str:
    """Four corners pushed outward: exclusive full screen."""
    parts = []
    for x, y, dx, dy in ((3.0, 3.0, 1, 1), (21.0, 3.0, -1, 1),
                         (3.0, 21.0, 1, -1), (21.0, 21.0, -1, -1)):
        parts.append(K.poly([(x + dx * 6.0, y), (x, y), (x, y + dy * 6.0)],
                            color=K.INK, width=2.1))
    parts.append(K.rect(8.6, 8.6, 6.8, 6.8, r=1.0, color=K.ACCENT, width=1.6))
    return K.svg("\n".join(parts))


def maximize() -> str:
    return K.svg(K.rect(4.0, 4.0, 16.0, 16.0, r=1.4, color=K.INK, width=2.0))


def close_window() -> str:
    return K.svg(K.poly([(5.6, 5.6), (18.4, 18.4)], color=K.INK, width=2.2)
                 + "\n" + K.poly([(18.4, 5.6), (5.6, 18.4)], color=K.INK, width=2.2))


def about() -> str:
    """Information: a ring with a stem and a dot."""
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.4, color=K.INK, width=1.9),
        K.line(12.0, 11.0, 12.0, 16.4, color=K.ACCENT, width=2.2),
        K.dot(12.0, 7.8, 1.35),
    ]))


def help() -> str:
    """Question mark in a ring: the help entry point."""
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.4, color=K.INK, width=1.9),
        f'  <path d="M 9.2 9.4 a 2.9 2.9 0 1 1 3.6 2.8 '
        f'c -0.8 0.4 -0.8 1.1 -0.8 2.0" stroke="{K.ACCENT}" stroke-width="2.0"/>',
        K.dot(12.0, 16.6, 1.25),
    ]))


def language() -> str:
    """Globe with meridians: language selection."""
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.2, color=K.INK, width=1.8),
        f'  <path d="M 3.8 12 h 16.4" stroke="{K.ACCENT}" stroke-width="1.5"/>',
        f'  <path d="M 12 3.8 C 7.6 7.0, 7.6 17.0, 12 20.2 '
        f'C 16.4 17.0, 16.4 7.0, 12 3.8 Z" stroke="{K.INK}" stroke-width="1.4"/>',
    ]))


def settings_gear() -> str:
    """Global settings: an eight-tooth gear with an open centre."""
    cx, cy, r_out, r_in = 12.0, 12.0, 8.2, 5.6
    pts = []
    for i in range(16):
        a = math.radians(i * 22.5)
        rr = r_out if i % 2 == 0 else r_in
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return K.svg("\n".join([
        K.poly(pts, close=True, color=K.INK),
        K.circle(cx, cy, 2.6, color=K.ACCENT, width=1.8),
    ]))


def theme_palette() -> str:
    """Four swatches: the theme entry."""
    return K.svg("\n".join([
        K.rect(3.2, 3.2, 7.6, 7.6, r=1.2, fill=K.INK),
        K.rect(13.2, 3.2, 7.6, 7.6, r=1.2, fill=K.ACCENT),
        K.rect(3.2, 13.2, 7.6, 7.6, r=1.2, fill=K.ACCENT),
        K.rect(13.2, 13.2, 7.6, 7.6, r=1.2, color=K.INK, width=1.6),
    ]))


def qt_warnings() -> str:
    """A terminal with a warning triangle: show Qt diagnostics in the console."""
    return K.svg("\n".join([
        K.rect(2.6, 5.4, 18.8, 13.2, r=1.8, color=K.INK, width=1.7),
        K.line(2.6, 9.2, 21.4, 9.2, color=K.INK, width=1.3),
        K.poly([(12.0, 11.0), (16.2, 17.6), (7.8, 17.6)], close=True,
               color=K.ACCENT, width=1.5),
    ]))


def donate() -> str:
    return K.svg("\n".join([
        f'  <path d="M 12 20.4 C 5.2 16.0, 3.0 12.6, 3.0 9.6 '
        f'A 4.6 4.6 0 0 1 12 7.4 A 4.6 4.6 0 0 1 21 9.6 '
        f'C 21 12.6, 18.8 16.0, 12 20.4 Z" '
        f'stroke="{K.ACCENT}" stroke-width="2.2" fill="none"/>',
    ]))


def components() -> str:
    """A plug: optional pieces you can attach."""
    return K.svg("\n".join([
        K.line(9.4, 2.8, 9.4, 6.6, color=K.INK, width=1.8),
        K.line(14.6, 2.8, 14.6, 6.6, color=K.INK, width=1.8),
        K.rect(6.2, 6.6, 11.6, 8.2, r=1.8, color=K.INK, width=1.7),
        K.line(12.0, 14.8, 12.0, 21.2, color=K.ACCENT, width=2.2),
    ]))


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def open_folder() -> str:
    return K.svg("\n".join([
        K.poly([(2.4, 19.4), (2.4, 5.0), (9.2, 5.0), (11.2, 7.6), (19.2, 7.6),
                (19.2, 10.0)], color=K.INK, width=1.7),
        K.poly([(2.4, 19.4), (6.0, 10.6), (22.0, 10.6), (18.4, 19.4)],
               close=True, color=K.ACCENT, width=1.7),
    ]))


def save() -> str:
    """Floppy outline: the universal save."""
    return K.svg("\n".join([
        K.poly([(3.4, 3.4), (17.2, 3.4), (20.6, 6.8), (20.6, 20.6), (3.4, 20.6)],
               close=True, color=K.INK, width=1.7),
        K.rect(7.0, 3.4, 8.4, 6.0, r=0.6, color=K.MUTED, width=1.3),
        K.rect(6.6, 12.4, 10.8, 8.2, r=1.0, fill=K.ACCENT),
    ]))


def save_project() -> str:
    """Floppy with a stack line: save *all* entities to one file."""
    return K.svg("\n".join([
        K.poly([(2.6, 6.4), (2.6, 3.2), (14.0, 3.2), (17.0, 6.2)],
               color=K.MUTED, width=1.4),
        K.poly([(5.0, 8.0), (17.2, 8.0), (20.4, 11.2), (20.4, 20.8), (5.0, 20.8)],
               close=True, color=K.INK, width=1.7),
        K.rect(8.0, 8.0, 7.6, 5.0, r=0.6, color=K.MUTED, width=1.2),
        K.rect(7.8, 14.4, 9.6, 6.4, r=1.0, fill=K.ACCENT),
    ]))


def export_out() -> str:
    """Arrow leaving a tray: write the data out of the application.

    Measured at 220 px: a 3.0 head on a 13-unit shaft was swamped by the tray's
    rounded ends and the direction was unreadable.  The tray is now a plain U
    with square-ish corners and the head is 4.2, which is the size where the
    triangle still reads as a triangle at 18 px.
    """
    return K.svg("\n".join([
        K.poly([(2.8, 13.6), (2.8, 21.0), (21.2, 21.0), (21.2, 13.6)],
               color=K.INK, width=2.0),
        K.arrow(12.0, 17.4, 12.0, 4.2, color=K.ACCENT, head=4.2),
    ]))


def import_in() -> str:
    """Arrow entering a tray: bring data in."""
    return K.svg("\n".join([
        K.poly([(2.8, 13.6), (2.8, 21.0), (21.2, 21.0), (21.2, 13.6)],
               color=K.INK, width=2.0),
        K.arrow(12.0, 4.2, 12.0, 17.4, color=K.ACCENT, head=4.2),
    ]))


def add_item() -> str:
    """Import a scalar field from a file: table plus an inward arrow.

    ``add.svg`` is loaded by

        <action name="actionImportSFFromFile">
          ... icon .../add.svg ...
          text "Import SF from file"

    so a bare plus is wrong -- it says "create", and this command *imports*.  A
    table with a field column and an arrow entering it states the operation.
    The annotation dialog reuses the same file for its class-set import, which
    reads correctly under the same "bring data in" meaning.
    """
    return K.svg("\n".join([
        K.rect(2.8, 4.0, 11.4, 13.6, r=1.2, color=K.INK, width=1.6),
        K.line(2.8, 8.4, 14.2, 8.4, color=K.INK, width=1.3),
        K.line(6.4, 4.0, 6.4, 17.6, color=K.MUTED, width=1.1),
        K.arrow(18.2, 3.4, 18.2, 16.4, color=K.ACCENT, head=2.8),
        K.line(2.8, 20.4, 14.2, 20.4, color=K.MUTED, width=1.5),
    ]))


def clear_all() -> str:
    """A broom: sweep everything away, not a delete of one item."""
    return K.svg("\n".join([
        K.line(19.6, 3.6, 11.4, 11.8, color=K.INK, width=2.1),
        K.poly([(12.8, 10.4), (13.6, 20.6), (4.4, 20.6), (3.6, 11.6)],
               close=True, color=K.ACCENT, width=1.7),
        K.line(5.4, 14.6, 12.6, 14.6, color=K.ACCENT, width=1.3),
    ]))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def prim_cube() -> str:
    return K.svg(K.cube(12, 12, r=8.0))


def prim_sphere() -> str:
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.2, color=K.INK, width=1.8),
        f'  <path d="M 4.4 9.0 C 8.0 11.0, 16.0 11.0, 19.6 9.0" '
        f'stroke="{K.ACCENT}" stroke-width="1.4"/>',
        f'  <path d="M 3.8 12.0 C 8.0 14.2, 16.0 14.2, 20.2 12.0" '
        f'stroke="{K.ACCENT}" stroke-width="1.4"/>',
    ]))


def prim_cylinder() -> str:
    return K.svg("\n".join([
        f'  <path d="M 5.0 6.4 a 7 2.6 0 1 0 14 0 a 7 2.6 0 1 0 -14 0" '
        f'stroke="{K.INK}" stroke-width="1.7"/>',
        K.line(5.0, 6.4, 5.0, 17.6, color=K.INK, width=1.7),
        K.line(19.0, 6.4, 19.0, 17.6, color=K.INK, width=1.7),
        f'  <path d="M 5.0 17.6 a 7 2.6 0 0 0 14 0" stroke="{K.ACCENT}" stroke-width="1.7"/>',
    ]))


def prim_dish() -> str:
    """A dish seen from above-front, with its focal point marked.

    The first drawing put a long stem under the rim, which read as a table lamp
    at every size.  A dish is an opening plus a focus, so those are the two
    things drawn: an elliptical rim, a shallower inner curve, and the focus.
    """
    return K.svg("\n".join([
        f'  <ellipse cx="12" cy="9.4" rx="9.0" ry="4.6" stroke="{K.INK}" stroke-width="1.8"/>',
        f'  <path d="M 4.2 10.8 C 5.6 16.2, 18.4 16.2, 19.8 10.8" '
        f'stroke="{K.INK}" stroke-width="1.8"/>',
        f'  <path d="M 5.6 10.4 C 7.0 14.0, 17.0 14.0, 18.4 10.4" '
        f'stroke="{K.ACCENT}" stroke-width="1.4"/>',
        K.dot(12.0, 6.6, 1.9),
    ]))


def prim_torus() -> str:
    return K.svg("\n".join([
        f'  <ellipse cx="12" cy="12" rx="9" ry="5.4" stroke="{K.INK}" stroke-width="1.7"/>',
        f'  <ellipse cx="12" cy="10.2" rx="4.4" ry="2.0" stroke="{K.ACCENT}" stroke-width="1.5"/>',
    ]))


def prim_plane() -> str:
    """A tilted plane with its normal: a plane has a facing, which matters."""
    return K.svg("\n".join([
        K.poly([(3.0, 16.0), (12.6, 10.4), (21.0, 15.0), (11.4, 20.6)],
               close=True, color=K.INK, width=1.7),
        K.arrow(12.0, 13.4, 12.0, 4.2, color=K.ACCENT, head=3.4),
        K.dot(12.0, 13.4, 1.5),
    ]))


# ---------------------------------------------------------------------------
# Mesh operations
# ---------------------------------------------------------------------------


def mesh_grid() -> str:
    return K.svg("\n".join([
        K.poly([(2.6, 15.0), (12.0, 9.8), (21.4, 15.0), (12.0, 20.2)],
               close=True, color=K.INK, width=1.7),
        K.line(12.0, 9.8, 12.0, 20.2, color=K.INK, width=1.3),
        K.line(7.3, 12.4, 16.7, 17.6, color=K.INK, width=1.3),
        K.line(16.7, 12.4, 7.3, 17.6, color=K.ACCENT, width=1.6),
    ]))


def mesh_smooth() -> str:
    """Mesh with a wavy surface above it: soften the facets."""
    return K.svg("\n".join([
        K.poly([(3.0, 16.6), (12.0, 11.6), (21.0, 16.6), (12.0, 21.6)],
               close=True, color=K.MUTED, width=1.4),
        f'  <path d="M 3.4 9.2 C 6.4 5.2, 9.4 5.2, 12.0 9.2 '
        f'C 14.6 13.2, 17.6 13.2, 20.6 9.2" stroke="{K.ACCENT}" stroke-width="1.9"/>',
    ]))


def subdivide() -> str:
    """One triangle split into four: subdivide, not decimate."""
    tri = [(12.0, 3.4), (21.0, 19.4), (3.0, 19.4)]
    mids = [((tri[0][0] + tri[1][0]) / 2, (tri[0][1] + tri[1][1]) / 2),
            ((tri[1][0] + tri[2][0]) / 2, (tri[1][1] + tri[2][1]) / 2),
            ((tri[2][0] + tri[0][0]) / 2, (tri[2][1] + tri[0][1]) / 2)]
    return K.svg("\n".join([
        K.poly(tri, close=True, color=K.INK, width=1.8),
        K.poly(mids, close=True, color=K.ACCENT, width=1.6),
        K.line(*mids[0], *mids[2], color=K.ACCENT, width=1.4),
        K.line(*mids[1], *mids[2], color=K.ACCENT, width=1.4),
    ]))


def decimate() -> str:
    """The same outline with fewer vertices, plus a minus: reduce the mesh."""
    return K.svg("\n".join([
        K.poly([(12.0, 3.8), (20.4, 18.8), (3.6, 18.8)], close=True,
               color=K.MUTED, width=1.5),
        K.poly([(12.0, 7.6), (16.4, 15.6), (7.6, 15.6)], close=True,
               color=K.INK, width=1.7),
        K.minus(12.0, 21.2, arm=3.0, color=K.ACCENT, width=2.2),
    ]))


def convex_hull() -> str:
    """Scattered points enclosed by their hull."""
    pts = [(8.0, 5.4), (15.6, 4.6), (20.4, 10.0), (18.6, 17.4),
           (11.0, 20.0), (4.6, 16.0), (3.4, 9.4), (12.4, 12.6)]
    parts = [K.poly([(8.0, 5.4), (15.6, 4.6), (20.4, 10.0), (18.6, 17.4),
                     (11.0, 20.0), (4.6, 16.0), (3.4, 9.4)], close=True,
                    color=K.ACCENT, width=1.7)]
    parts += [K.dot(x, y, 1.2) for x, y in pts[:-1]]
    parts.append(K.circle(12.4, 12.6, 1.3, color=K.MUTED, width=1.1))
    return K.svg("\n".join(parts))


def surface_render() -> str:
    """A shaded patch: surface rendering."""
    return K.svg("\n".join([
        K.poly([(2.8, 16.2), (12.0, 10.4), (21.2, 16.2), (12.0, 21.0)],
               close=True, color=K.INK, width=1.7),
        K.poly([(2.8, 16.2), (12.0, 10.4), (12.0, 6.0), (2.8, 11.8)],
               close=True, fill=K.ACCENT),
    ]))


def poisson_reconstruction() -> str:
    """Cloud turning into a closed surface: reconstruction, not merging."""
    parts = [K.cloud(9.0, 8.6, s=0.72, color=K.MUTED)]
    parts.append(K.poly([(3.2, 20.6), (12.0, 13.6), (20.8, 20.6)],
                        close=False, color=K.ACCENT, width=2.0))
    parts.append(K.line(3.2, 20.6, 20.8, 20.6, color=K.ACCENT, width=2.0))
    return K.svg("\n".join(parts))


def smooth_filter() -> str:
    """Three wavy strands becoming one: smoothing the field."""
    return K.svg("\n".join([
        f'  <path d="M 2.8 8.0 C 6.4 4.6, 9.6 11.4, 13.2 8.0 '
        f'C 16.8 4.6, 19.6 10.0, 21.6 8.0" stroke="{K.MUTED}" stroke-width="1.4"/>',
        f'  <path d="M 2.8 13.2 C 6.4 10.4, 9.6 15.8, 13.2 13.2 '
        f'C 16.8 10.6, 19.6 14.6, 21.6 13.2" stroke="{K.MUTED}" stroke-width="1.4"/>',
        f'  <path d="M 2.8 18.4 C 7.4 15.6, 16.6 21.0, 21.6 18.4" '
        f'stroke="{K.ACCENT}" stroke-width="2.2"/>',
    ]))


def unroll() -> str:
    """A cylinder opening into a flat sheet: unroll."""
    return K.svg("\n".join([
        f'  <path d="M 3.0 12 a 5.2 2.4 0 1 0 10.4 0 a 5.2 2.4 0 1 0 -10.4 0" '
        f'stroke="{K.MUTED}" stroke-width="1.4"/>',
        K.line(3.0, 12.0, 3.0, 18.4, color=K.MUTED, width=1.4),
        K.line(13.4, 12.0, 13.4, 18.4, color=K.MUTED, width=1.4),
        K.poly([(13.4, 6.2), (21.6, 4.4), (21.6, 17.6), (13.4, 19.4)],
               close=True, color=K.ACCENT, width=1.7),
    ]))


def scale_coordinates() -> str:
    """A box with outward arrows: multiply coordinates."""
    return K.svg("\n".join([
        K.rect(7.4, 7.4, 9.2, 9.2, r=1.2, color=K.INK, width=1.7),
        K.arrow(7.4, 7.4, 3.2, 3.2, color=K.ACCENT, head=2.4),
        K.arrow(16.6, 16.6, 20.8, 20.8, color=K.ACCENT, head=2.4),
    ]))


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def colorize() -> str:
    """A sun over a surface: recolour by lightness, values unchanged."""
    return K.svg("\n".join([
        K.circle(9.6, 9.4, 3.4, color=K.ACCENT, width=1.7),
        K.line(9.6, 3.0, 9.6, 1.8, color=K.ACCENT, width=1.4),
        K.line(9.6, 17.0, 9.6, 15.8, color=K.ACCENT, width=1.4),
        K.poly([(3.4, 20.4), (9.4, 14.0), (15.4, 20.4)], close=True,
               color=K.INK, width=1.6),
        K.rect(15.0, 14.0, 6.2, 6.2, r=1.0, fill=K.INK),
    ]))


def texture_to_rgb() -> str:
    """A swatch dissolving into pixel cells: convert texture to colours."""
    parts = [K.rect(2.8, 4.0, 8.0, 8.0, r=1.0, fill=K.ACCENT)]
    for i in range(2):
        for j in range(2):
            parts.append(K.rect(13.0 + i * 4.4, 4.0 + j * 4.4, 3.6, 3.6,
                                r=0.5, color=K.INK, width=1.2))
    parts.append(K.arrow(9.0, 17.8, 16.6, 17.8, color=K.MUTED, head=2.4))
    return K.svg("\n".join(parts))


def clear_colors() -> str:
    """A colour droplet struck through: remove the colours, keep the geometry."""
    return K.svg("\n".join([
        f'  <path d="M 12 3.4 C 8.0 8.0, 5.4 11.4, 5.4 14.4 '
        f'A 6.6 6.6 0 0 0 18.6 14.4 C 18.6 11.4, 16.0 8.0, 12 3.4 Z" '
        f'stroke="{K.INK}" stroke-width="1.7" fill="{K.ACCENT}"/>',
        K.line(4.0, 20.0, 20.0, 4.0, color=K.INK, width=2.2),
    ]))


def background_color() -> str:
    """A viewport frame with a colour band across it: set the backdrop.

    A solid inner fill measured 53% ink at 18 px and read as a black box; one
    accent band inside the frame says "the field behind the scene" and keeps the
    mark open.
    """
    return K.svg("\n".join([
        K.rect(2.8, 4.4, 18.4, 15.2, r=1.6, color=K.INK, width=1.8),
        K.rect(5.4, 13.2, 13.2, 3.8, r=0.7, fill=K.ACCENT),
        K.line(5.4, 9.0, 18.6, 9.0, color=K.MUTED, width=1.3),
        K.line(5.4, 11.2, 13.4, 11.2, color=K.MUTED, width=1.3),
    ]))


def object_color() -> str:
    """One cube among outlines, filled: colour this object alone."""
    return K.svg("\n".join([
        K.cube(8.0, 15.0, r=4.2, color=K.MUTED, width=1.4),
        K.rect(12.6, 3.0, 8.4, 8.4, r=1.2, fill=K.ACCENT),
        K.arrow(12.8, 12.6, 10.2, 13.6, color=K.INK, head=2.2),
    ]))


def color_menu() -> str:
    """A palette with a drop: the colour-edit popup."""
    return K.svg("\n".join([
        f'  <path d="M 12 20.4 A 8.6 8.6 0 1 1 17.4 4.2 '
        f'C 20.4 6.0, 20.8 9.4, 18.6 10.4 '
        f'C 16.8 11.2, 15.2 10.0, 14.6 11.6 '
        f'C 14.0 13.2, 15.4 14.2, 15.0 16.0 '
        f'C 14.6 18.4, 13.6 20.4, 12 20.4 Z" stroke="{K.INK}" stroke-width="1.7"/>',
        K.circle(8.6, 8.6, 1.5, fill=K.ACCENT),
        K.circle(14.0, 6.4, 1.5, fill=K.ACCENT),
        K.circle(7.0, 14.4, 1.5, fill=K.ACCENT),
    ]))


# ---------------------------------------------------------------------------
# Display and measurement
# ---------------------------------------------------------------------------


def display_settings() -> str:
    """A monitor with three sliders beneath: how the scene is drawn."""
    return K.svg("\n".join([
        K.rect(2.8, 3.4, 18.4, 11.0, r=1.6, color=K.INK, width=1.7),
        K.line(9.0, 3.4, 9.0, 14.4, color=K.MUTED, width=1.1),
        K.line(2.8, 8.6, 21.2, 8.6, color=K.MUTED, width=1.1),
        K.rect(8.8, 15.6, 9.0, 1.8, r=0.9, color=K.ACCENT, width=1.3),
        K.rect(8.8, 19.0, 5.4, 1.8, r=0.9, color=K.ACCENT, width=1.3),
    ]))


def cloud_primitive_distance() -> str:
    """A cloud and a shape with a measured span between them."""
    return K.svg("\n".join([
        K.cloud(6.6, 7.6, s=0.60),
        K.rect(14.6, 14.4, 7.0, 6.4, r=1.0, color=K.ACCENT, width=1.6),
        K.arrow(7.6, 10.6, 14.0, 13.6, color=K.INK, head=2.2),
        K.line(9.4, 12.6, 12.6, 12.6, color=K.MUTED, width=1.1),
    ]))


def icp_register() -> str:
    """Two clouds closing on each other: fine registration."""
    return K.svg("\n".join([
        K.cloud(6.4, 8.0, s=0.58),
        K.cloud(17.6, 16.0, s=0.58, color=K.ACCENT),
        K.arrow(8.8, 10.8, 15.2, 13.6, color=K.INK, head=2.4),
        K.arrow(15.4, 6.4, 18.2, 4.6, color=K.MUTED, head=2.0),
    ]))


def segment() -> str:
    """Scissors over a cloud: cut it into two parts."""
    return K.svg("\n".join([
        K.cloud(8.4, 16.0, s=0.62),
        K.circle(12.0, 5.6, 2.2, color=K.INK, width=1.5),
        K.circle(19.0, 8.2, 2.2, color=K.INK, width=1.5),
        K.line(13.6, 7.2, 20.6, 17.4, color=K.ACCENT, width=1.8),
        K.line(17.6, 6.6, 11.4, 16.2, color=K.ACCENT, width=1.8),
    ]))


def trace_polyline() -> str:
    """A polyline with its vertices: build a line by picking points."""
    return K.svg("\n".join([
        K.poly([(3.2, 18.4), (8.0, 9.6), (13.4, 13.2), (20.8, 4.8)],
               color=K.ACCENT, width=2.0),
        K.circle(3.2, 18.4, 1.7, fill=K.INK),
        K.circle(8.0, 9.6, 1.7, fill=K.INK),
        K.circle(13.4, 13.2, 1.7, fill=K.INK),
        K.circle(20.8, 4.8, 1.7, fill=K.INK),
    ]))


def ruler_tool() -> str:
    return K.svg("\n".join([
        K.rect(2.6, 8.4, 18.8, 7.2, r=1.2, color=K.INK, width=1.7),
        *[K.line(x, 8.4, x, 11.6, color=K.INK, width=1.3) for x in (6.0, 9.4, 12.8, 16.2)],
        K.line(9.4, 11.6, 9.4, 15.6, color=K.ACCENT, width=1.4),
    ]))


def zoom_tool() -> str:
    return K.svg("\n".join([
        K.circle(10.6, 10.6, 6.5, color=K.INK, width=1.9),
        K.line(15.4, 15.4, 20.6, 20.6, color=K.INK, width=2.3),
        K.line(7.4, 10.6, 13.8, 10.6, color=K.ACCENT, width=1.4),
        K.line(10.6, 7.4, 10.6, 13.8, color=K.ACCENT, width=1.4),
    ]))


# ---------------------------------------------------------------------------
# Rotation pivot (the @2x centre tools)
# ---------------------------------------------------------------------------


def pivot_target() -> str:
    """Crosshair with an accent centre: the rotation centre."""
    return K.svg("\n".join([
        K.circle(11.0, 11.0, 6.4, color=K.INK, width=1.7),
        K.line(11.0, 2.0, 11.0, 6.4, color=K.INK, width=1.5),
        K.line(11.0, 15.6, 11.0, 20.0, color=K.INK, width=1.5),
        K.line(2.0, 11.0, 6.4, 11.0, color=K.INK, width=1.5),
        K.line(15.6, 11.0, 20.0, 11.0, color=K.INK, width=1.5),
        K.dot(11.0, 11.0, 2.0),
    ]))


def pivot_pick() -> str:
    """The target with a pointer: click a point to become the pivot."""
    return K.svg("\n".join([
        K.circle(9.4, 9.4, 5.4, color=K.INK, width=1.6),
        K.dot(9.4, 9.4, 1.8),
        K.poly([(13.2, 12.6), (13.2, 20.4), (15.0, 18.6), (16.4, 21.4),
                (17.8, 20.7), (16.4, 18.0), (19.0, 17.7)],
               close=True, fill=K.ACCENT),
    ]))


def pivot_reset() -> str:
    """The target with a return arc: back to the geometric centre."""
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 6.0, color=K.INK, width=1.6),
        K.dot(12.0, 12.0, 1.8),
        f'  <path d="M 20.4 15.2 A 9.4 9.4 0 0 0 12.6 21.0" '
        f'stroke="{K.ACCENT}" stroke-width="1.9"/>',
        K.poly([(11.0, 21.4), (14.0, 20.4), (13.2, 23.0)], close=True, fill=K.ACCENT),
    ]))


def pivot_axes() -> str:
    """The target with three short axis arrows: show the pivot triad."""
    return K.svg("\n".join([
        K.circle(9.2, 12.6, 5.2, color=K.INK, width=1.6),
        K.dot(9.2, 12.6, 1.6),
        K.arrow(9.2, 12.6, 9.2, 4.4, color=K.ACCENT, head=2.0),
        K.arrow(9.2, 12.6, 17.4, 12.6, color=K.ACCENT, head=2.0),
        K.arrow(9.2, 12.6, 14.6, 18.0, color=K.ACCENT, head=2.0),
    ]))


def zoom_to_box() -> str:
    """A marquee with a magnifier: zoom into a rubber-banded region."""
    return K.svg("\n".join([
        K.rect(2.8, 4.4, 11.6, 9.6, r=1.0, color=K.ACCENT, width=1.7),
        K.circle(14.4, 14.4, 5.0, color=K.INK, width=1.9),
        K.line(18.0, 18.0, 21.4, 21.4, color=K.INK, width=2.2),
    ]))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def waveform() -> str:
    """An oscilloscope trace: signal values over sample index."""
    return K.svg("\n".join([
        K.line(2.6, 12.0, 21.4, 12.0, color=K.MUTED, width=1.2),
        f'  <path d="M 2.8 12.0 L 5.6 12.0 L 7.2 5.0 L 9.4 19.0 '
        f'L 11.4 8.6 L 13.2 15.4 L 15.0 3.6 L 17.2 20.0 L 19.0 12.0 L 21.2 12.0" '
        f'stroke="{K.ACCENT}" stroke-width="1.9"/>',
    ]))


def annotations() -> str:
    """A speech bubble with text lines: annotations."""
    return K.svg("\n".join([
        K.poly([(3.0, 4.4), (21.0, 4.4), (21.0, 16.2), (12.4, 16.2),
                (8.0, 20.4), (8.0, 16.2), (3.0, 16.2)], close=True,
               color=K.INK, width=1.7),
        K.line(6.4, 8.2, 17.6, 8.2, color=K.ACCENT, width=1.5),
        K.line(6.4, 12.2, 13.6, 12.2, color=K.MUTED, width=1.5),
    ]))


def octree() -> str:
    """A cube split into eight octants: the octree."""
    parts = [K.poly([(12.0, 2.6), (20.6, 7.4), (20.6, 16.6), (12.0, 21.4),
                     (3.4, 16.6), (3.4, 7.4)], close=True, color=K.INK, width=1.7)]
    parts.append(K.poly([(3.4, 7.4), (12.0, 12.2), (20.6, 7.4)],
                        color=K.INK, width=1.2))
    parts.append(K.line(12.0, 12.2, 12.0, 21.4, color=K.INK, width=1.2))
    parts.append(K.line(7.7, 9.8, 7.7, 19.0, color=K.MUTED, width=1.0))
    parts.append(K.line(16.3, 9.8, 16.3, 19.0, color=K.MUTED, width=1.0))
    parts.append(K.line(12.0, 16.8, 20.6, 12.0, color=K.MUTED, width=1.0))
    parts.append(K.line(12.0, 16.8, 3.4, 12.0, color=K.MUTED, width=1.0))
    parts.append(K.poly([(12.0, 12.2), (16.3, 9.8), (16.3, 14.6), (12.0, 16.8)],
                        close=True, fill=K.ACCENT))
    return K.svg("\n".join(parts))


def noise_filter() -> str:
    """A cloud with stray specks being dropped: the noise filter."""
    return K.svg("\n".join([
        K.cloud(12.0, 13.6, s=0.94),
        K.circle(5.0, 4.6, 1.2, color=K.MUTED, width=1.1),
        K.circle(19.4, 5.0, 1.2, color=K.MUTED, width=1.1),
        K.line(18.0, 20.0, 21.2, 16.8, color=K.ACCENT, width=1.8),
    ]))


def sensor() -> str:
    """A tripod scanner: a terrestrial laser sensor."""
    return K.svg("\n".join([
        K.rect(7.0, 3.4, 10.0, 7.6, r=1.0, color=K.INK, width=1.7),
        K.circle(12.0, 7.2, 2.0, color=K.ACCENT, width=1.5),
        K.line(12.0, 11.0, 12.0, 14.4, color=K.INK, width=1.6),
        K.line(12.0, 14.4, 5.0, 21.0, color=K.INK, width=1.6),
        K.line(12.0, 14.4, 19.0, 21.0, color=K.INK, width=1.6),
        K.line(12.0, 14.4, 12.0, 21.0, color=K.MUTED, width=1.3),
    ]))



def clear_eraser() -> str:
    """An eraser wiping a block away: 'Clear All' removes content, not the view.

    A broom was used for the ``clean`` menu; an eraser is the right mark for
    emptying a dataset, and it keeps the two clear-ish entries distinct.
    """
    return K.svg("\n".join([
        K.poly([(3.0, 15.4), (9.0, 9.4), (15.4, 15.8), (9.4, 21.8)],
               close=False, color=K.MUTED, width=1.5),
        f'  <path d="M 9.0 9.4 L 14.6 3.8 A 1.8 1.8 0 0 1 21.0 3.8 '
        f'L 21.0 9.4 L 15.4 15.8 Z" fill="{K.ACCENT}"/>',
        K.circle(15.4, 15.8, 2.6, fill=K.PAPER),
        K.line(2.6, 21.8, 21.4, 21.8, color=K.INK, width=1.8),
    ]))


def ecv_gear() -> str:
    """A small gear for the property tree's per-item settings action."""
    cx, cy, r_out, r_in = 12.0, 12.0, 7.6, 5.2
    pts = []
    for i in range(16):
        a = math.radians(i * 22.5)
        rr = r_out if i % 2 == 0 else r_in
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return K.svg("\n".join([
        K.poly(pts, close=True, color=K.INK),
        K.circle(cx, cy, 2.4, color=K.ACCENT, width=1.7),
    ]))


REGISTRY = {
    "3D3": new_3d_view,
    "FullScreen": full_screen,
    "max": maximize,
    "close": close_window,
    "about": about,
    "help": help,
    "language": language,
    "settings": settings_gear,
    "theme": theme_palette,
    "qt": qt_warnings,
    "donate": donate,
    "Components_72px": components,
    "open": open_folder,
    "save": save,
    "small_save": save,
    "saveBinary": save_project,
    "exportCloud": export_out,
    "exportIcon": export_out,
    "add": add_item,
    "clean_72px_27328": clear_all,
    "clear": clear_eraser,
    "ecvGear": ecv_gear,
    "cube": prim_cube,
    "sphere": prim_sphere,
    "cylinder": prim_cylinder,
    "dish": prim_dish,
    "torus": prim_torus,
    "Plane": prim_plane,
    "mesh4": mesh_grid,
    "meshSmooth": mesh_smooth,
    "subdivide": subdivide,
    "decimate": decimate,
    "convex_concave_hull": convex_hull,
    "surface": surface_render,
    "poisson": poisson_reconstruction,
    "vtkSmooth": smooth_filter,
    "unroll": unroll,
    "transformScale": scale_coordinates,
    "bgcolor": colorize,
    "color": texture_to_rgb,
    "ecvClearColor": clear_colors,
    "ecvBackgroundColor": background_color,
    "ecvObjectColor": object_color,
    "ecvColorMenu": color_menu,
    "ecvDisplaySettings": display_settings,
    "ecvCloudPrimitiveDistance": cloud_primitive_distance,
    "ecvicp": icp_register,
    "ecvSegment": segment,
    "ecvTracePolyline": trace_polyline,
    "zoom": zoom_tool,
    "pqPickCenter@2x": pivot_pick,
    "pqResetCenter@2x": pivot_reset,
    "pqShowCenterAxes@2x": pivot_axes,
    "pqZoomToBox@2x": zoom_to_box,
    "Wavefrom": waveform,
    "annotations": annotations,
    "octree": octree,
    "filter": noise_filter,
    "integrated_sensor": sensor,
}
