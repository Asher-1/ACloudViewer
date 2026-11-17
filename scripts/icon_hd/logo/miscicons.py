"""The ``svg/misc`` and ``svg/pcl`` icon sets.

Two leftovers of the old resource tree.  They are not one family by design:
``misc`` mixes dialog buttons, camera-lock toggles and PCL import marks that came
from different sources, which is exactly why their viewBoxes disagree (13.5 to
64) and why several of them are 20-70 KB.

Semantics were taken from the enclosing ``QAction`` / widget in the ``.ui`` file
rather than from the file name -- several names are misleading: ``store.svg`` is
the "show bounding box" toggle in the annotation dialog, ``scale.svg`` is
"Match scales" for point-pair alignment, ``hashtag.svg`` is the class-id field of
the segmentation export dialog, and ``insert_from_file.svg`` is the *animation*
action.
"""

from __future__ import annotations

import math

from . import iconset as K


# ---------------------------------------------------------------------------
# Dialog buttons
# ---------------------------------------------------------------------------


def confirm() -> str:
    """A check inside a ring: commit the dialog.

    A filled disc measured 0.457 ink at 18 px -- over the ceiling, and heavier
    than any neighbouring dialog button.  The ring keeps the "commit" reading and
    sits in the same weight class as the rest of the set.
    """
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.8, color=K.ACCENT, width=2.2),
        K.poly([(7.8, 12.4), (10.8, 15.4), (16.4, 8.8)], color=K.ACCENT, width=2.4),
    ]))


def edit_mode() -> str:
    """Pencil over a text line: edit existing content, not add new."""
    return K.svg("\n".join([
        K.line(3.4, 19.4, 13.4, 19.4, color=K.MUTED, width=1.6),
        K.poly([(4.6, 14.2), (6.2, 9.8), (15.4, 0.6), (19.0, 4.2), (9.8, 13.4)],
               close=True, color=K.INK, width=1.7),
        K.line(14.0, 2.0, 17.6, 5.6, color=K.ACCENT, width=1.6),
        K.poly([(4.6, 14.2), (6.2, 9.8), (8.2, 11.8)], close=True, fill=K.ACCENT),
    ]))


def hashtag() -> str:
    """A hash sign: the class-id field of an annotation export."""
    return K.svg("\n".join([
        K.line(10.0, 3.4, 8.2, 20.6, color=K.INK, width=2.0),
        K.line(16.4, 3.4, 14.6, 20.6, color=K.INK, width=2.0),
        K.line(4.6, 8.6, 19.6, 8.6, color=K.ACCENT, width=2.0),
        K.line(3.6, 15.0, 18.6, 15.0, color=K.ACCENT, width=2.0),
    ]))


def info() -> str:
    return K.svg("\n".join([
        K.circle(12.0, 12.0, 8.6, color=K.INK, width=1.9),
        K.line(12.0, 11.0, 12.0, 16.6, color=K.ACCENT, width=2.2),
        K.dot(12.0, 7.6, 1.4),
    ]))


def plus_sign() -> str:
    return K.svg(K.plus(12.0, 12.0, arm=7.0, width=2.4))


def restore_state() -> str:
    """A box returning to a previous position: restore the last clipping box."""
    return K.svg("\n".join([
        K.rect(6.4, 9.4, 14.2, 11.0, r=1.4, color=K.ACCENT, width=1.7),
        f'  <path d="M 4.4 8.2 A 8.2 8.2 0 0 1 15.4 2.6" '
        f'stroke="{K.INK}" stroke-width="1.8"/>',
        K.poly([(13.4, 0.6), (16.6, 2.8), (14.0, 5.2)], close=True, fill=K.INK),
    ]))


def bounding_box() -> str:
    """A wire box: show/hide the bounding box."""
    return K.svg(K.cube(12.0, 12.0, r=8.0))


def store_block() -> str:
    """A box on a base plate: keep this extent, the annotation 'show bbox'."""
    return K.svg("\n".join([
        K.cube(12.0, 9.6, r=6.0),
        K.line(3.4, 20.2, 20.6, 20.2, color=K.ACCENT, width=2.0),
        K.line(6.4, 17.6, 6.4, 20.2, color=K.ACCENT, width=1.5),
        K.line(17.6, 17.6, 17.6, 20.2, color=K.ACCENT, width=1.5),
    ]))


# ---------------------------------------------------------------------------
# Camera and view locks
# ---------------------------------------------------------------------------


def locked_axis() -> str:
    """A rotation ring with a padlock: turntable mode, one axis locked."""
    return K.svg("\n".join([
        f'  <ellipse cx="11.0" cy="10.4" rx="8.2" ry="4.0" '
        f'stroke="{K.INK}" stroke-width="1.7"/>',
        f'  <path d="M 11.0 3.0 A 8.2 8.2 0 0 1 11.0 19.0" '
        f'stroke="{K.MUTED}" stroke-width="1.4"/>',
        K.rect(14.0, 14.0, 7.6, 6.4, r=1.2, fill=K.ACCENT),
        f'  <path d="M 15.8 14.0 v -2.2 a 2.0 2.0 0 0 1 4.0 0 v 2.2" '
        f'stroke="{K.ACCENT_DEEP}" stroke-width="1.6"/>',
    ]))


def log_manager() -> str:
    """A document with lines and a magnifier: browse the log files."""
    return K.svg("\n".join([
        K.rect(3.0, 3.4, 12.0, 16.4, r=1.3, color=K.INK, width=1.6),
        K.line(3.0, 7.6, 15.0, 7.6, color=K.INK, width=1.3),
        *[K.line(5.4, y, 12.6, y, color=K.MUTED, width=1.2) for y in (11.0, 14.0, 17.0)],
        K.circle(16.4, 15.2, 3.6, color=K.ACCENT),
        K.line(19.0, 17.8, 21.4, 20.2, color=K.ACCENT, width=2.0),
    ]))


def ortho_sections() -> str:
    """A structural trace through a surface: lines of orientation.

    The name says "orthosections", a geology term, and the first drawing was a
    box cut into slabs -- plausible from the word alone and wrong.  The only
    binding is

        <widget name="traceModeButton">
          toolTip "Trace Tool: Measure orientation from structure trace"

    It traces a line *on* a surface to read its orientation, so the mark is a
    curved surface with one traced line and its measured bearing.
    """
    return K.svg("\n".join([
        f'  <path d="M 2.8 17.6 C 6.4 9.2, 17.6 9.2, 21.2 17.6" '
        f'stroke="{K.MUTED}" stroke-width="1.5"/>',
        K.poly([(5.0, 15.2), (12.0, 6.4), (19.0, 15.2)], color=K.ACCENT, width=1.9),
        K.dot(12.0, 6.4, 1.9),
        K.line(12.0, 6.4, 12.0, 2.6, color=K.INK, width=1.5),
    ]))


def interactors() -> str:
    """A handle at each corner of a box: the interactive gizmos."""
    parts = [K.rect(7.0, 7.0, 10.0, 10.0, r=1.2, color=K.MUTED, width=1.4)]
    for x, y in ((7.0, 7.0), (17.0, 7.0), (7.0, 17.0), (17.0, 17.0)):
        parts.append(K.circle(x, y, 2.3, fill=K.ACCENT))
    parts.append(K.dot(12.0, 12.0, 1.8, color=K.INK))
    return K.svg("\n".join(parts))


def match_scales() -> str:
    """Two boxes of different size brought to one: match scales."""
    return K.svg("\n".join([
        K.rect(2.8, 9.6, 7.6, 7.6, r=1.0, color=K.MUTED, width=1.4),
        K.rect(12.0, 6.4, 9.6, 9.6, r=1.2, color=K.ACCENT, width=1.7),
        K.arrow(9.6, 13.4, 12.4, 11.4, color=K.INK, head=2.2),
    ]))


# ---------------------------------------------------------------------------
# Primitives shared with the root set
# ---------------------------------------------------------------------------


def plane_primitive() -> str:
    """A tilted plane with its normal: a plane has a facing, which matters."""
    return K.svg("\n".join([
        K.poly([(3.0, 16.0), (12.6, 10.4), (21.0, 15.0), (11.4, 20.6)],
               close=True, color=K.INK, width=1.7),
        K.arrow(12.0, 13.4, 12.0, 4.2, color=K.ACCENT, head=3.4),
        K.dot(12.0, 13.4, 1.5),
    ]))


def torus_primitive() -> str:
    return K.svg("\n".join([
        f'  <ellipse cx="12" cy="12" rx="9.2" ry="5.6" stroke="{K.INK}" stroke-width="1.8"/>',
        f'  <ellipse cx="12" cy="10.2" rx="4.2" ry="1.9" stroke="{K.ACCENT}" stroke-width="1.6"/>',
    ]))


# ---------------------------------------------------------------------------
# Zoom / selection
# ---------------------------------------------------------------------------


def _zoom_magnifier(sign: str) -> str:
    parts = [
        K.circle(10.4, 10.4, 6.6, color=K.INK, width=1.9),
        K.line(15.2, 15.2, 20.6, 20.6, color=K.INK, width=2.3),
        K.line(7.2, 10.4, 13.6, 10.4, color=K.ACCENT, width=1.5),
    ]
    if sign == "plus":
        parts.append(K.line(10.4, 7.2, 10.4, 13.6, color=K.ACCENT, width=1.5))
    return K.svg("\n".join(parts))


def zoom_in() -> str:
    return _zoom_magnifier("plus")


def zoom_out() -> str:
    return _zoom_magnifier("minus")


def zoom_to_selection() -> str:
    """A rubber-band box with a magnifier: Zoom to Box.

    The name says "zoom to selection" and the first drawing was a marquee plus a
    magnifier, which happens to be right -- but for the wrong reason.  The
    loader is

        make("actionZoomToBox", ".../misc/zoom_to_selection.svg",
             tr("Zoom to Box"), tr("Zoom to selection"), false);

    The icon is the *box* gesture: corners drawn as a drag in progress, with the
    lens confirming that the drag zooms.  Keeping the box corners open (not a
    closed rectangle) is what distinguishes it from ``pqZoomToBox@2x``, which is
    a closed frame.
    """
    return K.svg("\n".join([
        K.poly([(2.8, 8.4), (2.8, 3.2), (8.0, 3.2)], color=K.ACCENT, width=2.0),
        K.poly([(12.0, 3.2), (17.2, 3.2), (17.2, 8.4)], color=K.ACCENT, width=2.0),
        K.poly([(2.8, 11.6), (2.8, 16.8), (8.0, 16.8)], color=K.ACCENT, width=2.0),
        K.circle(15.0, 15.0, 4.6, color=K.INK, width=1.9),
        K.line(18.4, 18.4, 21.4, 21.4, color=K.INK, width=2.2),
    ]))


def frustum_selection() -> str:
    """A view cone with the picked points: select everything inside the frustum."""
    return K.svg("\n".join([
        K.poly([(2.6, 12.0), (18.4, 5.4), (18.4, 18.6)], close=True,
               color=K.INK, width=1.7),
        K.dot(11.6, 9.4, 1.6),
        K.dot(14.4, 12.6, 1.6),
        K.dot(11.2, 15.0, 1.6),
        K.dot(15.8, 16.4, 1.6, color=K.MUTED),
    ]))


# ---------------------------------------------------------------------------
# Point-cloud import / extraction
# ---------------------------------------------------------------------------


def insert_from_file() -> str:
    """A document feeding a cloud.

    The name says "insert from file", but the action it is wired to in
    ``MainWindow.ui`` is the automatic-rotation *animation* -- so the mark is a
    rotation arc around a cloud, and the file corner is deliberately absent.
    """
    return K.svg("\n".join([
        K.cloud(12.0, 12.6, s=0.92),
        f'  <path d="M 3.4 12.6 A 8.6 8.6 0 0 1 8.6 4.6" '
        f'stroke="{K.ACCENT}" stroke-width="1.9"/>',
        K.poly([(6.0, 3.4), (9.4, 4.6), (7.6, 7.4)], close=True, fill=K.ACCENT),
    ]))


def insert_from_device() -> str:
    """A sensor feeding a cloud: import from a live device."""
    return K.svg("\n".join([
        K.rect(2.8, 3.4, 9.4, 7.0, r=1.2, color=K.INK, width=1.6),
        K.circle(7.5, 6.9, 1.8, color=K.ACCENT, width=1.4),
        K.arrow(7.5, 11.0, 7.5, 16.0, color=K.MUTED, head=2.2),
        K.cloud(13.0, 17.4, s=0.72, color=K.ACCENT),
    ]))


def new_cloud_from_selection() -> str:
    """A frame lifting away from a cloud: extract the selection as a new cloud."""
    return K.svg("\n".join([
        K.cloud(10.6, 15.4, s=0.86),
        K.rect(11.4, 2.6, 10.0, 8.4, r=1.0, color=K.ACCENT, width=1.7),
        K.arrow(16.4, 11.4, 16.4, 14.4, color=K.MUTED, head=2.2),
    ]))



def normal_estimation() -> str:
    """A surface patch with outward normal arrows: estimate the normals.

    The source bitmap was noise-like at 64 px (2,007 partially transparent pixels
    of 4,096, with no flat regions); the mark is rebuilt from what the menu does
    rather than from those pixels.
    """
    return K.svg("\n".join([
        K.poly([(2.8, 16.8), (12.0, 11.6), (21.2, 16.8), (12.0, 22.0)],
               close=True, color=K.INK, width=1.7),
        K.arrow(8.6, 15.0, 8.6, 8.2, color=K.ACCENT, head=2.6),
        K.arrow(12.0, 16.4, 12.0, 5.0, color=K.ACCENT, head=2.6),
        K.arrow(15.4, 15.0, 15.4, 8.2, color=K.ACCENT, head=2.6),
    ]))


REGISTRY = {
    "ccGear": confirm,
    "edit_48px": edit_mode,
    "hashtag": hashtag,
    "info": info,
    "plus": plus_sign,
    "restore": restore_state,
    "square": bounding_box,
    "store": store_block,
    "lockedAxis": locked_axis,
    "logManager": log_manager,
    "orthoSections": ortho_sections,
    "interactors": interactors,
    "scale": match_scales,
    "Plane": plane_primitive,
    "torus": torus_primitive,
    "zoomIn_48px": zoom_in,
    "zoomOut_48px": zoom_out,
    "zoom_to_selection": zoom_to_selection,
    "frustum_selection": frustum_selection,
    "insert_from_file": insert_from_file,
    "insert_from_OpenNi": insert_from_device,
    "new_cloud_from_selection": new_cloud_from_selection,
    "normal_estimation": normal_estimation,
}
