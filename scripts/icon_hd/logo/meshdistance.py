"""ecvCloudMeshDistance -- single-icon rework, ParaView style.

What the code does
------------------
    MainWindow.cpp:14446  doActionCloudMeshDist() accepts either
                            one mesh + one cloud, or two meshes
                          and refuses anything without at least one mesh.
    ecvPointCloud.cpp     the computation is
                            DistanceComputationTools::computeCloud2MeshDistances

So the operation is: for each point of the cloud, find the distance to the
nearest surface of the mesh.  The result is a per-point scalar field, like the
cloud/cloud case -- but the reference is a *surface*, not a point set.

What distinguishes it from ecvCloudCloudDistance
------------------------------------------------
The two toolbar buttons sit side by side, so their icons have to be tellable
apart in a glance at 16 px.  The difference is the reference object:

    cloud/cloud   two dot clusters, both the same kind of thing
    cloud/mesh    one dot cluster and one *triangulated surface*

The mesh is drawn as an explicitly triangulated sheet -- not a solid blob, and
not a smooth plane.  If it were smooth it would read as "distance to something",
and the pairing with the cloud/cloud icon would collapse.  The triangulation is
the only feature that carries the distinction, so it is drawn large enough to be
legible at 32 px.

Read from the source bitmap: the upper-right mass is yellow triangulated, the
left and lower masses are square marks.  That composition is kept -- one surface
above right, one cloud below left, measurement between.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    palette   #A7D28C / #72BE44 greens, #D0E5F6 / #9AC1E0 blues, #2896D3 strong
              blue, #C1A561 / #A3915E tans, #787878 / #CDCDCD neutrals
    fills     subtle gradients
    span      92 % of the canvas
    label     rotated with the shaft, set above it, four letters

Usage
-----
    python3 -m scripts.icon_hd.logo.icon meshdistance --show
    python3 -m scripts.icon_hd.logo.icon meshdistance --sheet out.png
    python3 -m scripts.icon_hd.logo.icon meshdistance --write
"""

from __future__ import annotations

import math

from . import paraview_style as P


MESH_HI = "#E8C87A"       # tan family, lifted toward the samples' #C1A561
MESH_LO = "#B98F1E"
TRI_LINE = "#7A5C0C"
CLOUD_C = P.BLUE_STRONG   # #2896D3
RULER = P.BLUE_STRONG


# ---------------------------------------------------------------------------
# The mesh: a triangulated sheet, drawn as explicitly as the small size allows
# ---------------------------------------------------------------------------


def _mesh(cx: float, cy: float, width: float, *, cell_n=3) -> tuple[str, float, float]:
    """A triangulated quad grid in isometric projection.

    Three features make it read as a *mesh* and not as a plane:

    1. Every cell is split into two triangles by a visible diagonal.  Without
       the diagonals it is a checkerboard, which reads as a table.
    2. The sheet is skewed in isometric projection, because a front-on grid
       looks like a spreadsheet and this application is 3D.
    3. The rim is a single closed outline at the samples' weight, so the whole
       sheet reads as one object even though it carries interior structure.

    **The basis vectors were wrong in the first attempt** and it is worth
    recording: A was (+,+) in x/y and B was (-,-), so their y contributions
    cancelled and the sheet collapsed into a flat sliver.  True isometric has
    both axes gaining y as they go out from the front corner:

        A = (+0.866, +0.5) * cell      runs right and down the screen
        B = (-0.866, +0.5) * cell      runs left and down the screen

    With those signs the extent is 5.2*cell wide by 3*cell tall -- the standard
    1.73:1 isometric footprint.

    Cell count is 3x3: at 32 px that is a cell every ~7 px, the smallest grid
    that still shows diagonals after antialiasing.  A 4x4 grid collapsed into a
    grey haze at toolbar size.

    Returns (svg, lower-left corner, rim radius) so the caller can attach the
    measuring arrow to the actual edge of the drawing instead of guessing.
    """
    cell = width / (4 * 0.866)
    ax, ay = cell * 0.866, cell * 0.5
    bx, by = -cell * 0.866, cell * 0.5

    def pt(i: int, j: int):
        return (cx + (i - cell_n / 2) * ax + (j - cell_n / 2) * bx,
                cy + (i - cell_n / 2) * ay + (j - cell_n / 2) * by)

    parts = []
    for j in range(cell_n):
        for i in range(cell_n):
            for tri, shade in (((pt(i, j), pt(i + 1, j), pt(i + 1, j + 1)), j),
                               ((pt(i, j), pt(i + 1, j + 1), pt(i, j + 1)), j)):
                col = MESH_HI if shade < cell_n / 2 else MESH_LO
                d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in tri) + " Z"
                parts.append(f'  <path d="{d}" fill="{col}" stroke="{TRI_LINE}" '
                             f'stroke-width="0.34" stroke-linejoin="round"/>')

    corners = [pt(0, 0), pt(cell_n, 0), pt(cell_n, cell_n), pt(0, cell_n)]
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in corners) + " Z"
    parts.append(f'  <path d="{d}" fill="none" stroke="{P.INK}" '
                 f'stroke-width="0.75" stroke-linejoin="round"/>')

    # The lowest point of the sheet is the front corner, pt(cell_n, cell_n).
    front = pt(cell_n, cell_n)
    return "\n".join(parts), front


# ---------------------------------------------------------------------------
# The cloud: the same loose clusters as the cloud/cloud icon
# ---------------------------------------------------------------------------


def _cloud(centre, radius, seed, *, r_dot=0.38, spacing=1.95, n=14):
    """A loose cluster of small dots.  Same constants as ecvCloudCloudDistance,
    so the two icons read as one family: the gap is 1.5x the dot width, which
    is what keeps the marks visibly separate."""
    import random

    rng = random.Random(seed)
    pts, tries = [], 0
    while len(pts) < n and tries < 8000:
        tries += 1
        a = rng.uniform(0, 2 * math.pi)
        d = radius * math.sqrt(rng.uniform(0.05, 1.0))
        x, y = centre[0] + d * math.cos(a), centre[1] + d * math.sin(a)
        if all(math.hypot(x - qx, y - qy) >= spacing for qx, qy in pts):
            pts.append((x, y))
    return pts


def _dots(pts, colour, r_dot=0.38) -> list[str]:
    return [f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{r_dot}" '
            f'fill="{colour}" stroke="{P.INK}" stroke-width="0.26"/>'
            for x, y in pts]


# ---------------------------------------------------------------------------
# Ruler and label -- shared construction with ecvCloudCloudDistance
# ---------------------------------------------------------------------------


def _ruler(x1, y1, x2, y2, w=0.85) -> str:
    """Double-headed arrow with evenly spaced ticks; tick count derived from the
    available length so the graduations can never overlap (see
    ecvCloudCloudDistance._ruler for the measurement that forced this)."""
    ang = math.atan2(y2 - y1, x2 - x1)
    ux, uy = math.cos(ang), math.sin(ang)
    px, py = -uy, ux
    span = math.hypot(x2 - x1, y2 - y1)

    head_len, head_w = 1.4, 0.8
    tick_half, min_spacing = 0.78, 1.9
    inner_a, inner_b = head_len + 0.8, span - head_len - 0.8
    usable = inner_b - inner_a
    n_ticks = max(2, int(usable / min_spacing) + 1) if usable > 0 else 0

    parts = [f'  <path d="M {x1 + ux * head_len * 0.5:.2f} '
             f'{y1 + uy * head_len * 0.5:.2f} '
             f'L {x2 - ux * head_len * 0.5:.2f} '
             f'{y2 - uy * head_len * 0.5:.2f}" fill="none" '
             f'stroke="{RULER}" stroke-width="{w}" stroke-linecap="butt"/>']
    for ax, ay, sx, sy in ((x1, y1, 1, 1), (x2, y2, -1, -1)):
        parts.append(f'  <path d="M {ax:.2f} {ay:.2f} '
                     f'L {ax + sx * head_len * ux + head_w * px:.2f} '
                     f'{ay + sy * head_len * uy + head_w * py:.2f} '
                     f'L {ax + sx * head_len * ux - head_w * px:.2f} '
                     f'{ay + sy * head_len * uy - head_w * py:.2f} Z" '
                     f'fill="{RULER}" stroke="none"/>')
    if n_ticks >= 2:
        for i in range(n_ticks):
            t = inner_a + usable * i / (n_ticks - 1)
            cx, cy = x1 + ux * t, y1 + uy * t
            parts.append(f'  <path d="M {cx - px * tick_half:.2f} '
                         f'{cy - py * tick_half:.2f} '
                         f'L {cx + px * tick_half:.2f} '
                         f'{cy + py * tick_half:.2f}" fill="none" '
                         f'stroke="{RULER}" stroke-width="0.6" '
                         f'stroke-linecap="butt"/>')
    return "\n".join(parts)


_LABEL_H = 1.55
_LABEL_W = 1.12
_LABEL_SW = 0.32
_LABEL_OFFSET = 3.1

# POINTS sits on the opposite side of the shaft.  It is shorter than MESH but
# uses a *larger* offset, because the space below the shaft is occupied by the
# cloud itself: at the MESH offset of 3.1 the word landed on the dots and the
# two became one smear.  3.9 clears the cloud's own extent.
_LABEL_OFFSET_PTS = 3.9


def _letter(ch: str, x: float, y0: float, y1: float, sw: float) -> str:
    mid = (y0 + y1) / 2
    w = _LABEL_W * 0.74
    d = {
        "M": f"M {x} {y1} L {x} {y0} L {x + w / 2} {y0 + (y1 - y0) * 0.45} "
             f"L {x + w} {y0} L {x + w} {y1}",
        "E": f"M {x + w} {y0} L {x} {y0} L {x} {y1} L {x + w} {y1} "
             f"M {x} {mid} L {x + w * 0.72} {mid}",
        "S": f"M {x + w} {y0 + 0.16} L {x + w * 0.12} {y0 + 0.16} "
             f"L {x + w * 0.12} {mid} L {x + w * 0.88} {mid} "
             f"L {x + w * 0.88} {y1 - 0.16} L {x} {y1 - 0.16}",
        "H": f"M {x} {y0} L {x} {y1} M {x + w} {y0} L {x + w} {y1} "
             f"M {x} {mid} L {x + w} {mid}",
        "P": f"M {x} {y1} L {x} {y0} L {x + w * 0.62} {y0} "
             f"Q {x + w} {y0} {x + w} {y0 + (y1 - y0) * 0.26} "
             f"Q {x + w} {mid} {x + w * 0.62} {mid} L {x} {mid}",
        "O": f"M {x + w / 2} {y0} Q {x + w} {y0} {x + w} {mid} "
             f"Q {x + w} {y1} {x + w / 2} {y1} Q {x} {y1} {x} {mid} "
             f"Q {x} {y0} {x + w / 2} {y0} Z",
        "I": f"M {x + w / 2} {y0} L {x + w / 2} {y1}",
        "N": f"M {x} {y1} L {x} {y0} L {x + w} {y1} L {x + w} {y0}",
        "T": f"M {x} {y0} L {x + w} {y0} M {x + w / 2} {y0} L {x + w / 2} {y1}",
    }.get(ch)
    if d is None:
        return ""
    return (f'  <path d="{d}" fill="none" stroke="{P.INK}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')


def _rotated_word(word: str, x1, y1, x2, y2, offset: float, *,
                  flip: bool = False) -> str:
    """A word set along the shaft, rotated with it.

    ``offset`` is measured along the perpendicular; ``flip`` puts it on the other
    side.  Both words therefore keep the same angle as the ruler no matter which
    side they sit on, and neither can drift off-axis: the rotation is computed
    from the same two endpoints the shaft uses.
    """
    ang = math.atan2(y2 - y1, x2 - x1)
    ox, oy = math.sin(ang), -math.cos(ang)
    if flip:
        ox, oy = -ox, -oy
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = mx + ox * offset, my + oy * offset

    total = len(word) * _LABEL_W
    x0 = -total / 2
    y1l = _LABEL_H / 2
    y0l = y1l - _LABEL_H
    letters = "\n".join(_letter(ch, x0 + i * _LABEL_W, y0l, y1l, _LABEL_SW)
                        for i, ch in enumerate(word))
    return (f'  <g transform="translate({cx:.2f},{cy:.2f}) '
            f'rotate({math.degrees(ang):.2f})">\n{letters}\n  </g>')


def _labels(x1, y1, x2, y2) -> str:
    """MESH above the shaft, POINTS below it.

    The pair states the operation in the icon's own words: the arrow measures
    something *between* a surface and a set of samples, and naming both ends
    removes the last ambiguity -- without the second word the icon could still be
    read as "a mesh with an arrow", which is not what the tool computes.
    """
    return "\n".join([
        _rotated_word("MESH", x1, y1, x2, y2, _LABEL_OFFSET),
        _rotated_word("POINTS", x1, y1, x2, y2, _LABEL_OFFSET_PTS, flip=True),
    ])


# ---------------------------------------------------------------------------


def build() -> str:
    """A triangulated surface above right, a point cloud below left, and a
    graduated arrow between them at an exact 45 degrees.

    Both endpoints are computed from the geometry, never typed in.  The first
    attempt hard-coded the mesh end at (11.9, 11.2) while the cloud end came
    from a projection -- the two did not lie on a common 45-degree line, and the
    arrow came out at -72 degrees pointing the wrong way.
    """
    # The mesh spans 1.73x its `width` argument, not `width`: the parameter sets
    # one isometric axis, and the sheet's footprint is that axis doubled on both
    # diagonals.  At width 9.4 the corners ran from x 8.3 to 22.5 and y 2.1 to
    # 10.3 -- 14.1 units wide on a 24-unit canvas, so after `fit()` the whole
    # icon was squeezed and the arrow collapsed to 3.8 units.
    mesh_c = (16.6, 6.4)
    cloud_c = (8.46, 17.6)

    # Axis points from the cloud UP-RIGHT to the mesh: +x, -y.
    ux, uy = 0.70710678, -0.70710678

    mesh_svg, mesh_front = _mesh(mesh_c[0], mesh_c[1], width=6.6, cell_n=3)
    pts = _cloud(cloud_c, 3.0, seed=17)
    parts = [mesh_svg] + _dots(pts, CLOUD_C)

    # Both endpoints reconstructed **on the exact ray**.  Taking the dot with the
    # largest projection and using its raw coordinates was a bug: that dot is
    # furthest along the axis but not necessarily on it, so the shaft came out at
    # -43.66 degrees instead of -45.
    along = max((p[0] - cloud_c[0]) * ux + (p[1] - cloud_c[1]) * uy for p in pts)
    t_cloud = along + 0.55
    x1 = cloud_c[0] + ux * t_cloud
    y1 = cloud_c[1] + uy * t_cloud

    t_mesh = (mesh_front[0] - cloud_c[0]) * ux + (mesh_front[1] - cloud_c[1]) * uy
    t_mesh -= 0.55
    x2 = cloud_c[0] + ux * t_mesh
    y2 = cloud_c[1] + uy * t_mesh

    parts.append(_ruler(x1, y1, x2, y2))
    parts.append(_labels(x1, y1, x2, y2))
    return P.svg("\n".join(parts))
