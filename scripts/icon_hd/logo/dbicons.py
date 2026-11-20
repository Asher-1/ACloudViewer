"""The database-tree icons, redrawn as hand-authored vectors.

Why these were rebuilt rather than traced
-----------------------------------------
The originals are 72-128 px painterly bitmaps (500-6000 distinct colours each).
Tracing them produced 60-146 KB SVGs against 2-28 KB PNGs -- a 3-16x *increase*,
which is backwards: a drawn vector should be far smaller than a raster of the same
mark. The size was not a tuning problem, it was the method:

    tracing a gradient means emitting thousands of small filled paths, one per
    colour step; a gradient is a paint description, not a shape

Two older conventions already in `svg/db` have the same root: `dbCircle.svg` is
149 `<rect>` elements transcribing a 24x24 raster, and the authored-polygon files
carry the comment "sampled from the original raster". All of them are limited by
the bitmap they came from.

These icons are therefore drawn from meaning, in the ParaView language the main
toolbar now uses: flat fills, one #6F6F59 outline on every form, no gradients
except where a form genuinely needs volume (a cylinder's curved top), and short
path data. Typical output is 0.6-2 KB.

What each icon means
--------------------
    AreaLabel          two crossing arrows -- the handles that resize an area
    CalibratedImage    a camera: body, lens, viewfinder
    Cloud              a point cloud, as a cloud
    CloudLocked        that cloud, with a padlock
    Container3         a stacked store: three discs
    ContainerLocked    that store, locked
    HObject            a folder -- the generic container of objects
    HObjectLocked      that folder, locked
    Label              a label sheet with a pencil
    Lock               a padlock
    MiscGeom           the three primitives: sphere, cone, cube
    Polyline           a polyline and its vertices
    Viewport           an eye -- "look through this"

The locked variants reuse their unlocked twin's geometry and add one padlock, so
the pair reads as one object and its state rather than as two unrelated icons.

Usage
-----
    python3 -m scripts.icon_hd.logo.dbicons                 # list + sizes
    python3 -m scripts.icon_hd.logo.dbicons --sheet out.png
    python3 -m scripts.icon_hd.logo.dbicons --write
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
TARGET_DIR = ROOT / "app/Resources/images/svg/db"

BLUE = P.BLUE_STRONG
BLUE_LT = P.BLUE_PALE
BLUE_MID = P.BLUE_MID
GREEN = P.GREEN
GREEN_LT = P.GREEN_LT
TAN = P.TAN
TAN_LT = "#D8C081"
GREY = P.GREY
GREY_DK = P.GREY_DK
LGREY = P.GREY_LT
GREY_LT = P.GREY_LT
TAN_DK = P.TAN_DK
BLUE_PALE = P.BLUE_PALE
GREY_MID = P.GREY_MID
YELLOW = P.YELLOW
RED = P.RED
VIOLET = P.VIOLET
INK = P.INK
SW = P.STROKE


# ---------------------------------------------------------------------------
# shared forms
# ---------------------------------------------------------------------------


def _lock(x: float, y: float, s: float = 1.0, *, body: str = YELLOW) -> str:
    """A padlock, `s` scale units tall, bottom-left at (x, y).

    The single repeated mark across the four locked variants.  Drawn once here so
    the four stay identical, and sized to stay legible when it is the *only* thing
    distinguishing a locked icon from its twin at 16 px.
    """
    w, h = 7.0 * s, 6.4 * s
    sh = 3.4 * s                            # shackle above the body
    bw = 4.2 * s
    bx = x + (w - bw) / 2
    return "\n".join([
        # shackle
        f'  <path d="M {bx + 0.55 * s:.2f} {y + sh:.2f} '
        f'L {bx + 0.55 * s:.2f} {y + sh * 0.55:.2f} '
        f'Q {bx + 0.55 * s:.2f} {y:.2f} {bx + bw / 2:.2f} {y:.2f} '
        f'Q {bx + bw - 0.55 * s:.2f} {y:.2f} {bx + bw - 0.55 * s:.2f} {y + sh * 0.55:.2f} '
        f'L {bx + bw - 0.55 * s:.2f} {y + sh:.2f}" fill="none" '
        f'stroke="{INK}" stroke-width="{1.5 * s:.2f}" stroke-linecap="round"/>',
        # body
        f'  <rect x="{x:.2f}" y="{y + sh:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{0.9 * s:.2f}" fill="{body}" stroke="{INK}" stroke-width="{SW}"/>',
        # keyhole
        f'  <circle cx="{x + w / 2:.2f}" cy="{y + sh + h * 0.42:.2f}" '
        f'r="{0.62 * s:.2f}" fill="{INK}"/>',
        f'  <path d="M {x + w / 2:.2f} {y + sh + h * 0.5:.2f} '
        f'L {x + w / 2:.2f} {y + sh + h * 0.74:.2f}" fill="none" '
        f'stroke="{INK}" stroke-width="{1.5 * s:.2f}" stroke-linecap="round"/>',
    ])


def _cylinder(cx: float, y0: float, y1: float, rx: float, *, discs: int = 0) -> str:
    """A vertical cylinder: elliptical top, curved sides, elliptical foot.

    The top ellipse is the only place volume is required, so it is the only fill
    that is not flat.  `discs` adds horizontal rules for the stacked-store variant.
    """
    ry = rx * 0.30
    parts = [
        f'  <path d="M {cx - rx:.2f} {y0:.2f} L {cx - rx:.2f} {y1:.2f} '
        f'A {rx:.2f} {ry:.2f} 0 0 0 {cx + rx:.2f} {y1:.2f} '
        f'L {cx + rx:.2f} {y0:.2f} Z" fill="{BLUE_MID}" stroke="{INK}" '
        f'stroke-width="{SW}" stroke-linejoin="round"/>',
        f'  <ellipse cx="{cx:.2f}" cy="{y0:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
        f'fill="{BLUE_LT}" stroke="{INK}" stroke-width="{SW}"/>',
    ]
    for i in range(discs):
        t = (i + 1) / (discs + 1)
        yy = y0 + (y1 - y0) * t
        parts.append(
            f'  <path d="M {cx - rx:.2f} {yy:.2f} '
            f'A {rx:.2f} {ry:.2f} 0 0 0 {cx + rx:.2f} {yy:.2f}" fill="none" '
            f'stroke="{INK}" stroke-width="0.6"/>')
    return "\n".join(parts)


def _cloud(cx: float, cy: float, w: float) -> str:
    """A cloud as one closed path: three lobes over a flat base.

    Not a mesh approximation and not a traced outline -- the three-lobe silhouette
    is the whole idea of "cloud", and it costs about 300 bytes.
    """
    s = w / 10.0
    d = (f"M {cx - 5 * s:.2f} {cy + 1.6 * s:.2f} "
         f"Q {cx - 5.6 * s:.2f} {cy - 0.4 * s:.2f} {cx - 3.6 * s:.2f} {cy - 0.9 * s:.2f} "
         f"Q {cx - 3.4 * s:.2f} {cy - 3.0 * s:.2f} {cx - 1.0 * s:.2f} {cy - 3.1 * s:.2f} "
         f"Q {cx + 0.4 * s:.2f} {cy - 4.4 * s:.2f} {cx + 2.3 * s:.2f} {cy - 2.9 * s:.2f} "
         f"Q {cx + 4.6 * s:.2f} {cy - 2.9 * s:.2f} {cx + 4.6 * s:.2f} {cy - 0.7 * s:.2f} "
         f"Q {cx + 5.6 * s:.2f} {cy - 0.3 * s:.2f} {cx + 5.0 * s:.2f} {cy + 1.6 * s:.2f} Z")
    return (f'  <path d="{d}" fill="{BLUE_LT}" stroke="{INK}" '
            f'stroke-width="{SW}" stroke-linejoin="round"/>')


def _folder(cx: float, cy: float, w: float) -> str:
    """A folder: tab on the left, body to its right."""
    s = w / 10.0
    return "\n".join([
        f'  <path d="M {cx - 5 * s:.2f} {cy + 3.2 * s:.2f} '
        f'L {cx - 5 * s:.2f} {cy - 3.4 * s:.2f} L {cx - 1.6 * s:.2f} {cy - 3.4 * s:.2f} '
        f'L {cx - 0.4 * s:.2f} {cy - 2.0 * s:.2f} L {cx + 5 * s:.2f} {cy - 2.0 * s:.2f} '
        f'L {cx + 5 * s:.2f} {cy + 3.2 * s:.2f} Z" fill="{BLUE_MID}" '
        f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"/>',
    ])


def _gear(cx: float, cy: float, r: float, teeth: int = 8) -> str:
    """A gear: alternating radii, drawn from trigonometry.

    Compact because the outline is one path regardless of tooth count -- the
    original's 890-colour chrome gear became a single flat silhouette.
    """
    ri = r * 0.72
    pts = []
    n = teeth * 4
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r if (i % 4) in (0, 1) else ri
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    return "\n".join([
        f'  <path d="{d}" fill="{LGREY}" stroke="{INK}" stroke-width="{SW}" '
        f'stroke-linejoin="round"/>',
        f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r * 0.30:.2f}" fill="{GREY}" '
        f'stroke="{INK}" stroke-width="0.6"/>',
    ])


def _eye(cx: float, cy: float, w: float) -> str:
    """An almond eye with an iris."""
    s = w / 10.0
    d = (f"M {cx - 5 * s:.2f} {cy:.2f} "
         f"Q {cx:.2f} {cy - 4.0 * s:.2f} {cx + 5 * s:.2f} {cy:.2f} "
         f"Q {cx:.2f} {cy + 4.0 * s:.2f} {cx - 5 * s:.2f} {cy:.2f} Z")
    return "\n".join([
        f'  <path d="{d}" fill="{LGREY}" stroke="{INK}" stroke-width="{SW}" '
        f'stroke-linejoin="round"/>',
        f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{2.0 * s:.2f}" fill="{BLUE}" '
        f'stroke="{INK}" stroke-width="0.6"/>',
        f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{0.8 * s:.2f}" fill="{INK}"/>',
    ])


def _pencil(x: float, y: float, s: float = 1.0) -> str:
    """A pencil at 45 degrees, tip at (x, y)."""
    ln, wd = 7.4 * s, 1.9 * s
    d = (f"M {x:.2f} {y:.2f} "
         f"L {x + 1.5 * s:.2f} {y - 0.5 * s:.2f} "
         f"L {x + ln * 0.72:.2f} {y - ln * 0.72 + 1.0 * s:.2f} "
         f"L {x + ln * 0.72 - 1.0 * s:.2f} {y - ln * 0.72:.2f} Z")
    return "\n".join([
        f'  <path d="{d}" fill="{TAN_LT}" stroke="{INK}" stroke-width="0.6" '
        f'stroke-linejoin="round"/>',
        f'  <path d="M {x:.2f} {y:.2f} L {x + 0.75 * s:.2f} {y - 0.25 * s:.2f} '
        f'L {x + 0.9 * s:.2f} {y - 0.55 * s:.2f} Z" fill="{INK}"/>',
    ])


# ---------------------------------------------------------------------------
# the icons
# ---------------------------------------------------------------------------


def area_label() -> str:
    """Two interlocking L-arrows enclosing a rectangle.

    Read from the original: a green L whose vertical arm points up and whose foot
    runs right along the bottom, and a teal L whose top arm runs right and whose
    vertical arm points down. Together they trace an area and point at two
    opposite corners -- the resize gesture, not two loose diagonal arrows.
    """
    green = ("M 5.4 10.6 L 5.4 4.4 "           # vertical, arrowhead up
             "M 3.5 6.3 L 5.4 4.2 L 7.3 6.3 "
             "M 5.4 10.6 L 16.2 10.6 L 16.2 20.2")   # foot, then down
    teal = ("M 9.0 5.6 L 19.6 5.6 L 19.6 14.4 "      # top arm, then down
            "M 17.7 12.3 L 19.6 14.6 L 21.5 12.3")   # arrowhead down
    return "\n".join([
        f'  <path d="{green}" fill="none" stroke="{GREEN}" stroke-width="2.0" '
        f'stroke-linecap="square" stroke-linejoin="miter"/>',
        f'  <path d="{teal}" fill="none" stroke="{BLUE}" stroke-width="1.75" '
        f'stroke-linecap="square" stroke-linejoin="miter"/>',
    ])


def calibrated_image() -> str:
    """A camera: body, lens, viewfinder and shutter."""
    return "\n".join([
        # viewfinder
        f'  <path d="M 8.6 5.4 L 10.4 3.0 L 14.0 3.0 L 15.4 5.4 Z" fill="{GREY}" '
        f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"/>',
        # body
        f'  <rect x="2.6" y="5.4" width="18.8" height="12.6" rx="1.6" '
        f'fill="{GREY_DK}" stroke="{INK}" stroke-width="{SW}"/>',
        # lens
        f'  <circle cx="12.0" cy="11.7" r="4.3" fill="{LGREY}" stroke="{INK}" '
        f'stroke-width="{SW}"/>',
        f'  <circle cx="12.0" cy="11.7" r="2.4" fill="{BLUE_MID}" stroke="{INK}" '
        f'stroke-width="0.6"/>',
        f'  <circle cx="12.0" cy="11.7" r="1.0" fill="{INK}"/>',
        # shutter
        f'  <circle cx="18.4" cy="8.0" r="1.1" fill="{RED}" stroke="{INK}" '
        f'stroke-width="0.5"/>',
    ])


def cloud() -> str:
    """A point cloud, drawn as a cloud."""
    return _cloud(12.0, 13.2, 17.0)


def cloud_locked() -> str:
    """The cloud, with a padlock over its lower right."""
    return "\n".join([
        _cloud(11.0, 11.6, 14.4),
        _lock(12.6, 12.4, 1.02),
    ])


def container() -> str:
    """Four stored bands in one cylinder, using the measured ParaView palette."""
    rx, y0, y1 = 7.6, 4.8, 20.2
    ry = rx * 0.30
    bands = 4
    shades = [P.BLUE_PALE, P.BLUE_MID, P.BLUE_STRONG, P.GREY_MID]
    parts = []
    for i in range(bands):
        ya = y0 + (y1-y0)*i/bands
        yb = y0 + (y1-y0)*(i+1)/bands
        parts.append(f'  <path d="M 12.00 {ya:.2f} L {12-rx:.2f} {ya:.2f} L {12-rx:.2f} {yb:.2f} L {12+rx:.2f} {yb:.2f} L {12+rx:.2f} {ya:.2f} Z" fill="{shades[i]}" stroke="none"/>')
    for i in range(1,bands):
        yy=y0+(y1-y0)*i/bands
        parts.append(f'  <path d="M {12-rx:.2f} {yy:.2f} A {rx:.2f} {ry:.2f} 0 0 0 {12+rx:.2f} {yy:.2f}" fill="none" stroke="{INK}" stroke-width="0.6"/>')
    parts += [f'  <path d="M {12-rx:.2f} {y0:.2f} L {12-rx:.2f} {y1:.2f} A {rx:.2f} {ry:.2f} 0 0 0 {12+rx:.2f} {y1:.2f} L {12+rx:.2f} {y0:.2f} Z" fill="none" stroke="{INK}" stroke-width="{SW}"/>',
              f'  <ellipse cx="12.00" cy="{y0:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" fill="{BLUE_LT}" stroke="{INK}" stroke-width="{SW}"/>']
    return "\n".join(parts)


def container3() -> str:
    """A stacked store: three discs on one cylinder."""
    return _cylinder(12.0, 5.4, 20.0, 7.4, discs=2)


def container_locked() -> str:
    """The stacked store, locked."""
    return "\n".join([
        _cylinder(10.4, 4.6, 19.4, 6.4, discs=2),
        _lock(13.0, 12.2, 0.98),
    ])


def hobject() -> str:
    """The generic container of database objects: a folder with a gear."""
    return "\n".join([
        _folder(11.2, 12.2, 17.6),
        _gear(16.6, 15.4, 4.3),
    ])


def hobject_locked() -> str:
    """The generic object, locked."""
    return "\n".join([
        _folder(10.6, 11.4, 15.6),
        _lock(12.4, 12.6, 0.98),
    ])


def label() -> str:
    """A label sheet with a pencil: annotate this."""
    return "\n".join([
        f'  <rect x="3.0" y="3.4" width="13.0" height="17.2" rx="1.2" '
        f'fill="{TAN_LT}" stroke="{INK}" stroke-width="{SW}"/>',
        f'  <path d="M 5.6 8.0 L 13.4 8.0 M 5.6 11.6 L 13.4 11.6 '
        f'M 5.6 15.2 L 10.0 15.2" fill="none" stroke="{INK}" '
        f'stroke-width="0.8" stroke-linecap="round"/>',
        _pencil(14.6, 19.4, 0.92),
    ])


def lock() -> str:
    """A padlock, alone and large."""
    return _lock(5.4, 3.6, 1.85, body=YELLOW)


def misc_geom() -> str:
    """The three primitives: sphere, cone, cube."""
    return "\n".join([
        # cube behind
        f'  <path d="M 8.6 8.6 L 13.8 8.6 L 13.8 13.6 L 8.6 13.6 Z" '
        f'fill="{GREY}" stroke="{INK}" stroke-width="0.7"/>',
        f'  <path d="M 8.6 8.6 L 11.0 6.4 L 16.2 6.4 L 13.8 8.6" fill="{LGREY}" '
        f'stroke="{INK}" stroke-width="0.7" stroke-linejoin="round"/>',
        f'  <path d="M 13.8 8.6 L 16.2 6.4 L 16.2 11.4 L 13.8 13.6" fill="{GREY_DK}" '
        f'stroke="{INK}" stroke-width="0.7" stroke-linejoin="round"/>',
        # cone right
        f'  <path d="M 17.2 19.6 L 19.4 11.4 L 21.6 19.6 Z" fill="{BLUE_MID}" '
        f'stroke="{INK}" stroke-width="0.7" stroke-linejoin="round"/>',
        # sphere front
        f'  <circle cx="8.6" cy="17.2" r="4.0" fill="{GREEN}" stroke="{INK}" '
        f'stroke-width="0.7"/>',
        f'  <circle cx="7.4" cy="16.0" r="1.3" fill="{GREEN_LT}"/>',
    ])


def polyline() -> str:
    """A polyline and its vertices: the points that define a curve."""
    pts = [(4.2, 16.8), (9.0, 8.4), (13.4, 15.0), (19.6, 6.6)]
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    dots = "\n".join(
        f'  <circle cx="{x}" cy="{y}" r="1.5" fill="{INK}"/>' for x, y in pts)
    return "\n".join([
        f'  <path d="{d}" fill="none" stroke="{INK}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>',
        dots,
    ])


def viewport() -> str:
    """An eye: the visible view."""
    return _eye(12.0, 12.0, 18.0)



# ---------------------------------------------------------------------------
# the second batch: the files that were still pixel-transcriptions
# ---------------------------------------------------------------------------


def cam_sensor() -> str:
    """A camera sensor: the camera body without a lens barrel.

    Distinguished from `calibrated_image` on purpose -- that one is the *calibrated*
    image (a full camera with a shutter button), this one is the sensor itself, so
    it carries the body outline and a single lens ring, no button.
    """
    return "\n".join([
        f'  <path d="M 8.8 5.2 L 10.6 3.0 L 13.6 3.0 L 15.0 5.2 Z" fill="{GREY_LT}" '
        f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"/>',
        f'  <rect x="2.8" y="5.2" width="18.4" height="12.4" rx="1.5" '
        f'fill="{GREY_LT}" stroke="{INK}" stroke-width="{SW}"/>',
        f'  <circle cx="12.0" cy="11.4" r="4.1" fill="{LGREY}" stroke="{INK}" '
        f'stroke-width="{SW}"/>',
        f'  <circle cx="12.0" cy="11.4" r="2.1" fill="{GREY}" stroke="{INK}" '
        f'stroke-width="0.6"/>',
    ])


def circle() -> str:
    """A circle primitive: one ring, no fill."""
    return (f'  <circle cx="12.0" cy="12.0" r="9.2" fill="none" '
            f'stroke="{BLUE}" stroke-width="2.0"/>')


def file() -> str:
    """A file: a folder with a turned corner, in the tree's file colour."""
    return "\n".join([
        f'  <path d="M 3.4 19.6 L 3.4 5.4 L 10.2 5.4 L 11.8 7.6 L 20.6 7.6 '
        f'L 20.6 19.6 Z" fill="{TAN}" stroke="{INK}" stroke-width="{SW}" '
        f'stroke-linejoin="round"/>',
        f'  <path d="M 3.4 9.2 L 20.6 9.2" fill="none" stroke="{INK}" '
        f'stroke-width="0.7"/>',
        f'  <path d="M 3.4 12.4 L 20.6 12.4" fill="none" stroke="{TAN_DK}" '
        f'stroke-width="0.5" stroke-opacity="0.6"/>',
    ])


def gbl_sensor() -> str:
    """A GBL sensor: the odometry reference frame, as a crosshair ring.

    The original is a red ring with four arms and a cross inside -- the global
    reference target.  Red is kept because that is the sensor's own colour in the
    tree, and it is the only icon in the set using it.
    """
    return "\n".join([
        f'  <circle cx="12.0" cy="12.0" r="7.2" fill="none" stroke="{RED}" '
        f'stroke-width="1.7"/>',
        f'  <path d="M 12.0 2.6 L 12.0 21.4 M 2.6 12.0 L 21.4 12.0" fill="none" '
        f'stroke="{RED}" stroke-width="1.7" stroke-linecap="round"/>',
        f'  <circle cx="12.0" cy="12.0" r="2.0" fill="none" stroke="{RED}" '
        f'stroke-width="1.4"/>',
    ])


def image() -> str:
    """An image: a framed picture with a brush -- this is image data."""
    return "\n".join([
        f'  <rect x="2.8" y="3.6" width="15.2" height="17.0" rx="1.0" '
        f'fill="{BLUE_PALE}" stroke="{INK}" stroke-width="{SW}"/>',
        # horizon + sun: the two cheapest marks that say "picture"
        f'  <path d="M 2.8 16.0 L 7.6 11.0 L 11.4 15.0 L 14.0 12.4 L 18.0 16.0" '
        f'fill="none" stroke="{GREEN}" stroke-width="1.3" stroke-linejoin="round"/>',
        f'  <circle cx="13.4" cy="7.4" r="1.6" fill="{YELLOW}" stroke="{INK}" '
        f'stroke-width="0.5"/>',
        _pencil(15.0, 20.6, 0.86),
    ])


def material() -> str:
    """A material: a sphere carrying a checker -- what a material is."""
    import math as _m
    cx, cy, r = 12.0, 12.0, 8.6
    parts = [f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{LGREY}" '
             f'stroke="{INK}" stroke-width="{SW}"/>']
    # a 4x4 checker clipped to the sphere; generated so the count is trivial
    cell = 2 * r / 4
    clip = f'  <clipPath id="matclip"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath>'
    parts.append(clip)
    parts.append('  <g clip-path="url(#matclip)">')
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                continue
            x = cx - r + i * cell
            y = cy - r + j * cell
            parts.append(f'    <rect x="{x:.2f}" y="{y:.2f}" '
                         f'width="{cell:.2f}" height="{cell:.2f}" fill="{GREY_DK}"/>')
    parts.append("  </g>")
    parts.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                 f'stroke="{INK}" stroke-width="{SW}"/>')
    return "\n".join(parts)


def _mesh_cell(cx: float, cy: float, r: float) -> str:
    """One triangulated cell, seen as a diamond with a cross.

    The shared mark for the three mesh icons. Four triangles meeting at the
    centre is the smallest drawing that reads as "triangulated surface" rather
    than as a plain diamond, and it costs four short paths.
    """
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    return "\n".join([
        f'  <path d="{d}" fill="{TAN_LT}" stroke="{INK}" stroke-width="0.7" '
        f'stroke-linejoin="round"/>',
        f'  <path d="M {cx:.2f} {cy - r:.2f} L {cx:.2f} {cy + r:.2f} '
        f'M {cx - r:.2f} {cy:.2f} L {cx + r:.2f} {cy:.2f}" fill="none" '
        f'stroke="{INK}" stroke-width="0.6"/>',
    ])


def mesh() -> str:
    """One mesh cell."""
    return _mesh_cell(12.0, 12.0, 9.0)


def mesh_locked() -> str:
    """One mesh cell, locked."""
    return "\n".join([_mesh_cell(10.0, 10.4, 7.6), _lock(12.4, 12.4, 0.96)])


def misc_geom_locked() -> str:
    """The primitives, locked."""
    return "\n".join([misc_geom(), _lock(12.8, 12.6, 0.94)])


def octree() -> str:
    """An octree cell: a cube seen corner-on, with its internal subdivision.

    The hexagon silhouette plus three spokes is the conventional octree mark --
    it shows the eight-way split without drawing eight separate boxes.
    """
    cx, cy, r = 12.0, 12.0, 8.8
    pts = [(cx + r * 0.86, cy - r * 0.5), (cx + r * 0.86, cy + r * 0.5),
           (cx, cy + r), (cx - r * 0.86, cy + r * 0.5),
           (cx - r * 0.86, cy - r * 0.5), (cx, cy - r)]
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    return "\n".join([
        f'  <path d="{d}" fill="{TAN_LT}" stroke="{INK}" stroke-width="{SW}" '
        f'stroke-linejoin="round"/>',
        f'  <path d="M {cx:.2f} {cy - r:.2f} L {cx:.2f} {cy + r:.2f} '
        f'M {cx - r * 0.86:.2f} {cy - r * 0.5:.2f} '
        f'L {cx + r * 0.86:.2f} {cy + r * 0.5:.2f} '
        f'M {cx + r * 0.86:.2f} {cy - r * 0.5:.2f} '
        f'L {cx - r * 0.86:.2f} {cy + r * 0.5:.2f}" fill="none" '
        f'stroke="{INK}" stroke-width="0.7"/>',
    ])


def octree_locked() -> str:
    """An octree cell, locked."""
    return "\n".join([octree(), _lock(12.6, 12.8, 0.94)])


def sub_mesh() -> str:
    """Three mesh cells: a mesh subdivided into parts."""
    return "\n".join([
        _mesh_cell(7.4, 7.6, 5.2),
        _mesh_cell(16.6, 7.6, 5.2),
        _mesh_cell(12.0, 17.0, 5.2),
    ])


def sub_mesh_locked() -> str:
    """Three mesh cells, locked."""
    return "\n".join([sub_mesh(), _lock(13.4, 13.6, 0.92)])


ICONS = {
    "dbAreaLabelSymbol": area_label,
    "dbCalibratedImageSymbol": calibrated_image,
    "dbCloudSymbol": cloud,
    "dbCloudSymbolLocked": cloud_locked,
    "dbContainerSymbol": container,
    "dbContainerSymbol3": container3,
    "dbContainerSymbolLocked": container_locked,
    "dbHObjectSymbol": hobject,
    "dbHObjectSymbolLocked": hobject_locked,
    "dbLabelSymbol": label,
    "dbLockSymbol": lock,
    "dbMiscGeomSymbol": misc_geom,
    "dbPolylineSymbol": polyline,
    "dbViewportSymbol": viewport,
    # second batch
    "dbCamSensorSymbol": cam_sensor,
    "dbCircle": circle,
    "dbFileSymbol": file,
    "dbGBLSensorSymbol": gbl_sensor,
    "dbImageSymbol": image,
    "dbMaterialSymbol": material,
    "dbMeshSymbol": mesh,
    "dbMeshSymbolLocked": mesh_locked,
    "dbMiscGeomSymbolLocked": misc_geom_locked,
    "dbOctreeSymbol": octree,
    "dbOctreeSymbolLocked": octree_locked,
    "dbSubMeshSymbol": sub_mesh,
    "dbSubMeshSymbolLocked": sub_mesh_locked,
}


def svg_of(name: str) -> str:
    """The finished SVG for one icon, on the 24-unit canvas."""
    body = f'<g stroke-linecap="round" stroke-linejoin="round">\n{ICONS[name]()}\n</g>'
    return P.svg(body)


if __name__ == "__main__":                                   # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild the db-tree icons as vectors.")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="report size against the PNG being replaced")
    a = ap.parse_args()

    png_dir = ROOT / "app/Resources/images"
    print(f"{'icon':<30} {'svg':>7} {'png':>8}  ratio")
    total_s = total_p = 0
    count_p = 0
    for name in ICONS:
        s = svg_of(name)
        n = len(s.encode())
        png_file = png_dir / f"{name}.png"
        pb = png_file.stat().st_size if png_file.exists() else 0
        total_s += n
        total_p += pb
        count_p += bool(pb)
        r = f"{n / pb:.2f}x" if pb else "(png removed)"
        flag = "" if (not pb or n < pb) else "  OVER"
        print(f"{name:<30} {n:>7,} {pb:>8,}  {r}{flag}")
    if total_p:
        print(f"{'TOTAL':<30} {total_s:>7,} {total_p:>8,}  "
              f"{total_s / total_p:.2f}x  (over {count_p} still-present PNGs)")
    else:
        print(f"{'TOTAL':<30} {total_s:>7,}  (no PNGs left to compare)")

    if a.write:
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        for name in ICONS:
            (TARGET_DIR / f"{name}.svg").write_text(svg_of(name))
        print(f"\nwrote {len(ICONS)} files to {TARGET_DIR}")

    if a.sheet:
        import cairosvg
        from PIL import Image, ImageDraw, ImageFont

        sizes = (16, 24, 32, 48, 96, 192)
        big = max(sizes)
        lab, gap, pad = 250, 20, 14
        names = list(ICONS)
        w = lab + sum(sizes) + gap * len(sizes) + 30
        h = 40 + (big + pad) * len(names)
        canvas = Image.new("RGBA", (w, h), (252, 252, 253, 255))
        dr = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 13)
        x = lab
        for s in sizes:
            dr.text((x, 10), f"{s}px", font=font, fill=(130, 130, 130))
            x += s + gap
        for i, name in enumerate(names):
            y = 40 + i * (big + pad)
            dr.text((16, y + big // 2 - 8), name.replace("Symbol", ""),
                    font=font, fill=(30, 30, 30))
            data = svg_of(name)
            x = lab
            for s in sizes:
                png = cairosvg.svg2png(bytestring=data.encode(),
                                       output_width=s, output_height=s)
                canvas.alpha_composite(
                    Image.open(io.BytesIO(png)).convert("RGBA"),
                    (x, y + big - s))
                x += s + gap
            dr.line([(16, y + big + 4), (w - 16, y + big + 4)],
                    fill=(226, 226, 228))
        canvas.convert("RGB").save(a.sheet)
        print(f"sheet -> {a.sheet}")
