"""The last mixed set: sensor/annotation/filter marks under ``svg/`` root.

Fourteen leftovers that never belonged to a family.  Names again had to be read
off the ``QAction`` rather than trusted: ``pqWrench`` is the *Edit* button of a
sensor (a wrench is right, the name happened to be too), ``pqSideSetData`` is
"Box Annotation", ``pqEyeball`` is "Save viewport as object", and
``insert_from_file`` in the neighbouring set is really the animation toggle.
"""

from __future__ import annotations

from . import iconset as K


# ---------------------------------------------------------------------------
# Application and view
# ---------------------------------------------------------------------------


def exit_app() -> str:
    """A door with an arrow leaving it: quit the application."""
    return K.svg("\n".join([
        K.poly([(5.6, 3.6), (13.0, 3.6), (13.0, 20.4), (5.6, 20.4)], close=True,
               color=K.INK, width=1.7),
        K.arrow(13.6, 12.0, 21.2, 12.0, color=K.ACCENT, head=2.8),
    ]))


def fullscreen() -> str:
    """Four corners pushed outward: fill the screen with the 3D view."""
    parts = []
    for x, y, dx, dy in ((3.0, 3.0, 1, 1), (21.0, 3.0, -1, 1),
                         (3.0, 21.0, 1, -1), (21.0, 21.0, -1, -1)):
        parts.append(K.poly([(x + dx * 6.0, y), (x, y), (x, y + dy * 6.0)],
                            color=K.INK, width=2.1))
    parts.append(K.rect(8.6, 8.6, 6.8, 6.8, r=1.0, color=K.ACCENT, width=1.6))
    return K.svg("\n".join(parts))


def save_viewport() -> str:
    """An eye over a frame: keep what the viewport currently shows."""
    return K.svg("\n".join([
        f'  <path d="M 2.6 12.4 C 5.4 7.0, 18.6 7.0, 21.4 12.4 '
        f'C 18.6 17.8, 5.4 17.8, 2.6 12.4 Z" stroke="{K.INK}" stroke-width="1.7"/>',
        K.dot(12.0, 12.4, 3.0),
        K.rect(12.4, 16.6, 8.2, 5.4, r=0.9, color=K.ACCENT, width=1.5),
        K.line(14.2, 18.4, 18.8, 18.4, color=K.ACCENT, width=1.2),
    ]))


def camera_sensor() -> str:
    """A survey camera: the sensor menu."""
    return K.svg("\n".join([
        K.rect(2.8, 7.4, 13.0, 9.6, r=1.8, color=K.INK, width=1.7),
        K.circle(9.3, 12.2, 2.9, color=K.INK, width=1.5),
        K.poly([(16.4, 11.2), (21.2, 7.6), (21.2, 16.8), (16.4, 13.2)],
               close=True, color=K.ACCENT, width=1.7),
        K.line(7.0, 5.8, 11.6, 5.8, color=K.MUTED, width=1.6),
    ]))


def wrench() -> str:
    """An open-end wrench: edit the sensor's parameters."""
    return K.svg("\n".join([
        f'  <path d="M 20.4 3.6 L 15.0 9.0" stroke="{K.INK}" stroke-width="2.4"/>',
        f'  <path d="M 12.6 11.4 L 4.4 19.6 A 1.9 1.9 0 0 0 7.1 22.3 L 15.3 14.1" '
        f'stroke="{K.INK}" stroke-width="2.4" stroke-linecap="round"/>',
        f'  <path d="M 13.8 7.0 A 5.0 5.0 0 0 1 21.0 3.0 L 17.6 6.4 '
        f'L 18.6 9.4 L 21.6 10.4 L 18.2 13.8 A 5.0 5.0 0 0 1 13.8 7.0 Z" '
        f'stroke="{K.INK}" stroke-width="1.9"/>',
        K.dot(5.9, 20.4, 1.5),
    ]))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def clip_section() -> str:
    """A box cut by a plane, the removed half ghosted."""
    return K.svg("\n".join([
        K.cube(9.0, 12.0, r=6.6, color=K.MUTED, width=1.4),
        K.poly([(13.6, 4.0), (21.6, 8.0), (21.6, 16.4), (13.6, 20.0)],
               close=True, color=K.ACCENT, width=1.8),
        K.line(13.6, 4.0, 13.6, 20.0, color=K.INK, width=1.6),
    ]))


def slice_section() -> str:
    """A cloud crossed by a plane, with the cut contour highlighted.

    Two earlier attempts failed the same way: a cloud plus a wide open curve
    read as a mountain at every size, because both halves were the same weight
    and the "plane" had no straight edge.  The plane is now a straight accent
    line that visibly passes *through* the body, and the cut is a second line.
    """
    return K.svg("\n".join([
        K.cloud(12.0, 10.0, s=0.94),
        K.line(2.4, 15.6, 21.6, 15.6, color=K.ACCENT, width=2.2),
        K.line(6.4, 18.8, 17.6, 18.8, color=K.MUTED, width=1.5),
        K.dot(12.0, 15.6, 1.8),
    ]))


def threshold() -> str:
    """A histogram with the kept band filled: threshold filter.

    The first version was a bar with two ticks and a box on top -- at 18 px the
    ticks vanished and it read as a progress bar.  A histogram with one filled
    band is unmistakable and matches what the filter actually operates on.
    """
    bars = [(3.0, 12.4, 4.6), (6.8, 8.0, 9.0), (10.6, 4.4, 12.6),
            (14.4, 7.2, 9.8), (18.2, 11.0, 6.0)]
    parts = [K.line(2.4, 17.2, 22.0, 17.2, color=K.MUTED, width=1.5)]
    for x, y, h in bars:
        parts.append(K.rect(x, y, 3.0, h, r=0.4, color=K.INK, width=1.3))
    parts.append(K.rect(10.6, 4.4, 3.0, 12.6, r=0.4, fill=K.ACCENT))
    parts.append(K.line(9.4, 20.4, 14.8, 20.4, color=K.ACCENT, width=1.8))
    return K.svg("\n".join(parts))


def glyph_filter() -> str:
    """Arrows rooted at sample points: glyph filter."""
    parts = [
        K.dot(6.0, 17.4, 1.5),
        K.dot(12.0, 19.0, 1.5),
        K.dot(18.0, 16.0, 1.5),
    ]
    parts.append(K.arrow(6.0, 17.4, 8.4, 9.4, color=K.ACCENT, head=2.2))
    parts.append(K.arrow(12.0, 19.0, 12.0, 8.0, color=K.ACCENT, head=2.2))
    parts.append(K.arrow(18.0, 16.0, 20.4, 6.6, color=K.ACCENT, head=2.2))
    return K.svg("\n".join(parts))


def streamline() -> str:
    """Curves bending around a solid body: streamlines, not glyphs.

    ``pqGlyph`` is also a set of lines, so the two must differ in silhouette.
    Glyph is straight arrows anchored on a lattice; streamline is curved flow
    deflected by an obstacle, which no straight-arrow mark can be confused with.
    """
    return K.svg("\n".join([
        K.circle(12.6, 12.0, 3.0, color=K.INK, width=1.6),
        f'  <path d="M 2.6 6.0 C 7.6 6.0, 7.6 9.4, 12.6 9.4 '
        f'C 17.6 9.4, 17.6 6.0, 21.4 6.0" stroke="{K.ACCENT}" stroke-width="1.8"/>',
        f'  <path d="M 2.6 18.0 C 7.6 18.0, 7.6 14.6, 12.6 14.6 '
        f'C 17.6 14.6, 17.6 18.0, 21.4 18.0" stroke="{K.ACCENT}" stroke-width="1.8"/>',
        K.line(2.6, 12.0, 9.6, 12.0, color=K.MUTED, width=1.4),
        K.line(15.6, 12.0, 21.4, 12.0, color=K.MUTED, width=1.4),
    ]))


def iso_surface() -> str:
    """A closed iso-contour lifted out of a field: iso-surface filter."""
    return K.svg("\n".join([
        K.line(2.6, 19.4, 21.4, 19.4, color=K.MUTED, width=1.4),
        f'  <path d="M 5.6 19.4 C 5.6 11.0, 18.4 11.0, 18.4 19.4" '
        f'stroke="{K.INK}" stroke-width="1.8"/>',
        f'  <path d="M 5.6 19.4 A 6.4 2.6 0 0 0 18.4 19.4" '
        f'stroke="{K.ACCENT}" stroke-width="1.6"/>',
        K.line(12.0, 5.2, 12.0, 11.4, color=K.ACCENT, width=1.5),
    ]))


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


def annotation_box() -> str:
    """A box grouping points, with its label tag."""
    return K.svg("\n".join([
        K.rect(2.8, 6.4, 13.2, 11.2, r=1.2, color=K.ACCENT, width=1.8),
        K.dot(6.6, 13.4, 1.4),
        K.dot(9.4, 11.0, 1.4),
        K.dot(11.8, 14.4, 1.4),
        K.poly([(15.4, 6.0), (21.4, 6.0), (21.4, 10.2), (18.4, 10.2),
                (16.6, 12.0), (16.6, 9.8), (15.4, 9.8)], close=True, fill=K.INK),
    ]))


def paste_from_clipboard() -> str:
    """A clipboard with an incoming sheet: paste."""
    return K.svg("\n".join([
        K.poly([(5.0, 4.4), (19.0, 4.4), (19.0, 20.6), (5.0, 20.6)], close=True,
               color=K.INK, width=1.7),
        K.rect(8.6, 2.6, 6.8, 3.6, r=1.0, fill=K.INK),
        K.rect(7.6, 9.4, 8.8, 8.4, r=0.9, color=K.ACCENT, width=1.5),
        K.arrow(12.0, 7.8, 12.0, 9.2, color=K.ACCENT, head=0.9),
    ]))


def preview() -> str:
    """An eye with a small play mark: show the result before committing."""
    return K.svg("\n".join([
        f'  <path d="M 2.6 12.6 C 5.6 6.8, 18.4 6.8, 21.4 12.6 '
        f'C 18.4 18.4, 5.6 18.4, 2.6 12.6 Z" stroke="{K.INK}" stroke-width="1.7"/>',
        K.circle(12.0, 12.6, 3.0, color=K.ACCENT, width=1.7),
        K.dot(12.0, 12.6, 1.2),
    ]))


REGISTRY = {
    "pqExitApp": exit_app,
    "pqFullscreen": fullscreen,
    "pqEyeball": save_viewport,
    "pqCamera": camera_sensor,
    "pqWrench": wrench,
    "pqClip": clip_section,
    "slice": slice_section,
    "pqThreshold": threshold,
    "pqGlyph": glyph_filter,
    "pqStreamTracer": streamline,
    "pqIsosurface": iso_surface,
    "pqSideSetData": annotation_box,
    "pqPaste": paste_from_clipboard,
    "pqPreview": preview,
}
