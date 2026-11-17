"""Style bench: the same eight tools drawn in five visual languages.

Why a bench instead of a proposal
---------------------------------
Four attempts at restyling this icon set were rejected, each for the same
underlying reason: style was decided while drawing, so there was no style to
follow -- only whatever each individual drawing happened to become.  A bench
separates those concerns.  Five languages are defined once, as tokens; every
candidate icon is then a consequence of a token set, and a reviewer picks a
column rather than arguing about individual pictures.

The five languages
------------------
A  Line          2.0 stroke, slate ink, one cyan accent.  Matches dev-tool
                 conventions; the most legible family at 18 px because nothing
                 relies on fills meeting exactly.
B  Solid         filled silhouette, white knockouts, no stroke.  Highest contrast
                 at small sizes, but adjacent shapes can merge.
C  Duotone       two flat fills, no stroke: a mid slate body plus an accent part.
                 Softer, reads as one family at medium sizes.
D  Isometric     2.5D boxes and plates.  Speaks the product's own language (it
                 manipulates 3D data) at the cost of pixel budget at 18 px.
E  Line + block  outline for structure, one filled accent block for the verb.
                 Gives each icon a focal point without a second colour.

Every icon is drawn at 24 units on its longest edge so the columns are directly
comparable, and each is rendered at 18 / 36 / 96 px because 18 is the real
toolbar size and 96 is where craft becomes visible.

Nothing here writes to the repository.  It exists to be looked at and chosen
from; `--write` is deliberately not implemented.
"""

from __future__ import annotations

import math

CANVAS = 24

# --- palette: one neutral ramp, one accent, shared by every language ---------
INK = "#2E3B47"        # body
INK_DEEP = "#1B2530"   # deepest plane
INK_MID = "#4A5A68"    # second plane
INK_LIGHT = "#8B9AA6"  # third plane / muted
PAPER = "#FFFFFF"
ACCENT = "#1E9BD7"
ACCENT_DEEP = "#1577A8"


# ---------------------------------------------------------------------------
# primitives -- deliberately few; a language is a constraint, not a toolkit
# ---------------------------------------------------------------------------


def path(d: str, *, fill=None, stroke=None, w=None, extra="") -> str:
    a = f' d="{d}"'
    if fill:
        a += f' fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}"'
    if w:
        a += f' stroke-width="{w}"'
    return f'  <path{a}{extra}/>'


def line(x1, y1, x2, y2, *, c=None, w=None) -> str:
    return path(f"M {x1} {y1} L {x2} {y2}", stroke=c or INK, w=w)


def poly(pts, *, fill=None, stroke=None, w=None, close=True) -> str:
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts) + (" Z" if close else "")
    return path(d, fill=fill, stroke=stroke, w=w)


def rect(x, y, w_, h, *, r=0, fill=None, stroke=None, w=None) -> str:
    a = f' x="{x}" y="{y}" width="{w_}" height="{h}"'
    if r:
        a += f' rx="{r}"'
    if fill:
        a += f' fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}"'
    if w:
        a += f' stroke-width="{w}"'
    return f"  <rect{a}/>"


def circ(cx, cy, r, *, fill=None, stroke=None, w=None) -> str:
    a = f' cx="{cx}" cy="{cy}" r="{r}"'
    if fill:
        a += f' fill="{fill}"'
    if stroke:
        a += f' stroke="{stroke}"'
    if w:
        a += f' stroke-width="{w}"'
    return f"  <circle{a}/>"


def arrow(x1, y1, x2, y2, *, c=None, head=3.0, w=None) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    col = c or INK
    return "\n".join([
        line(x1, y1, x2, y2, c=col, w=w),
        poly([(x2, y2), (x2 + head * math.cos(a1), y2 + head * math.sin(a1)),
              (x2 + head * math.cos(a2), y2 + head * math.sin(a2))], fill=col),
    ])


def iso(cx, cy, r, *, faces=(INK_LIGHT, INK_MID, INK), gap=False) -> str:
    """Isometric box: top, left, right faces filled with the neutral ramp."""
    top = [(cx, cy - r), (cx + r * 0.866, cy - r * 0.5),
           (cx, cy), (cx - r * 0.866, cy - r * 0.5)]
    left = [(cx - r * 0.866, cy - r * 0.5), (cx, cy), (cx, cy + r),
            (cx - r * 0.866, cy + r * 0.5)]
    right = [(cx + r * 0.866, cy - r * 0.5), (cx, cy), (cx, cy + r),
             (cx + r * 0.866, cy + r * 0.5)]
    parts = [poly(top, fill=faces[0]), poly(left, fill=faces[1]), poly(right, fill=faces[2])]
    if gap:
        parts = [poly(top, fill=faces[0], stroke=PAPER, w=0.9),
                 poly(left, fill=faces[1], stroke=PAPER, w=0.9),
                 poly(right, fill=faces[2], stroke=PAPER, w=0.9)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# the eight tools, five languages each
# ---------------------------------------------------------------------------
# Chosen to span the semantic range of the toolbar: arithmetic, duplication,
# statistics, alignment, picking, smoothing, clipping, merging.  A language that
# only works for "plus" and fails for "smooth" is not a language.


def svg(body: str, *, ink: str = INK) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
            f'viewBox="0 0 24 24" fill="none" stroke="{ink}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">\n{body}\n</svg>\n')


# --- A: line -----------------------------------------------------------------


def A_plus():
    return svg(line(5.2, 12, 18.8, 12, w=2.4) + "\n" + line(12, 5.2, 12, 18.8, w=2.4))


def A_clone():
    return svg(rect(3.4, 6.8, 11.2, 11.2, r=1.6, stroke=INK, w=2.0) + "\n"
               + rect(9.4, 3.4, 11.2, 11.2, r=1.6, stroke=ACCENT, w=2.0))


def A_histogram():
    b = [rect(3.6, 13.0, 3.4, 8.0, r=0.5, stroke=INK, w=1.8),
         rect(10.3, 8.4, 3.4, 12.6, r=0.5, stroke=ACCENT, w=1.8),
         rect(17.0, 11.2, 3.4, 9.8, r=0.5, stroke=INK, w=1.8)]
    return svg("\n".join(b))


def A_align():
    return svg(line(4.0, 12, 20.0, 12, c=ACCENT, w=2.2) + "\n"
               + rect(3.4, 4.2, 7.4, 5.4, r=1.0, stroke=INK, w=1.9) + "\n"
               + rect(13.2, 14.4, 7.4, 5.4, r=1.0, stroke=INK, w=1.9))


def A_picking():
    pts = [(6.0, 6.4), (17.4, 5.2), (5.4, 17.6), (18.2, 18.0)]
    d = "".join(circ(x, y, 1.5, fill=INK) for x, y in pts)
    return svg(d + "\n" + circ(12, 12, 3.6, stroke=ACCENT, w=2.0) + "\n"
               + circ(12, 12, 1.5, fill=ACCENT))


def A_smooth():
    return svg(path("M 3 15.2 C 6.4 15.2, 6.4 8.8, 12 8.8 C 17.6 8.8, 17.6 15.2, 21 15.2",
                    stroke=INK, w=2.1) + "\n"
               + line(3, 19.4, 21, 19.4, c=ACCENT, w=2.0))


def A_clip():
    b = []
    for x, y, dx, dy in ((3.6, 3.6, 1, 1), (20.4, 3.6, -1, 1),
                         (3.6, 20.4, 1, -1), (20.4, 20.4, -1, -1)):
        b.append(path(f"M {x+dx*5.4} {y} L {x} {y} L {x} {y+dy*5.4}", stroke=ACCENT, w=2.2))
    b.append(rect(8.4, 8.4, 7.2, 7.2, r=1.0, stroke=INK, w=2.0))
    return svg("\n".join(b))


def A_merge():
    return svg(circ(8.4, 8.4, 5.0, stroke=INK, w=2.0) + "\n"
               + circ(15.6, 8.4, 5.0, stroke=ACCENT, w=2.0) + "\n"
               + path("M 8.4 13.4 L 12 17.6 L 15.6 13.4", stroke=INK, w=1.8) + "\n"
               + path("M 8.6 19.0 L 12 22.0 L 15.4 19.0", stroke=ACCENT, w=2.0))


# --- B: solid -----------------------------------------------------------------


def B_plus():
    return svg(poly([(9.4, 4.4), (14.6, 4.4), (14.6, 9.4), (19.6, 9.4), (19.6, 14.6),
                     (14.6, 14.6), (14.6, 19.6), (9.4, 19.6), (9.4, 14.6),
                     (4.4, 14.6), (4.4, 9.4), (9.4, 9.4)], fill=INK))


def B_clone():
    return svg(rect(3.4, 6.8, 11.2, 11.2, r=1.8, fill=INK_LIGHT) + "\n"
               + rect(9.4, 3.4, 11.2, 11.2, r=1.8, fill=ACCENT))


def B_histogram():
    return svg(rect(3.6, 13.0, 3.4, 8.0, r=0.6, fill=INK_LIGHT) + "\n"
               + rect(10.3, 8.4, 3.4, 12.6, r=0.6, fill=ACCENT) + "\n"
               + rect(17.0, 11.2, 3.4, 9.8, r=0.6, fill=INK))


def B_align():
    return svg(rect(4.0, 5.4, 16.0, 2.6, r=1.3, fill=ACCENT) + "\n"
               + rect(3.4, 8.8, 8.0, 5.6, r=1.4, fill=INK) + "\n"
               + rect(12.6, 14.6, 8.0, 5.6, r=1.4, fill=INK_MID))


def B_picking():
    b = [circ(x, y, 1.9, fill=INK) for x, y in
         ((6.0, 6.4), (17.4, 5.2), (5.4, 17.6), (18.2, 18.0))]
    b.append(circ(12, 12, 4.6, fill=ACCENT))
    return svg("\n".join(b))


def B_smooth():
    return svg(path("M 2.6 15.0 C 6.2 15.0, 6.2 8.4, 12 8.4 C 17.8 8.4, 17.8 15.0, 21.4 15.0 "
                    "L 21.4 19.6 L 2.6 19.6 Z", fill=INK)+"\n"
               + path("M 3.2 12.0 C 6.4 12.0, 6.6 6.4, 12 6.4 C 17.4 6.4, 17.6 12.0, 20.8 12.0 "
                      "L 20.8 14.0 L 3.2 14.0 Z", fill=ACCENT))


def B_clip():
    b = [rect(6.6, 6.6, 10.8, 10.8, r=1.0, fill=INK_LIGHT)]
    for x, y, dx, dy in ((3.2, 3.2, 1, 1), (20.8, 3.2, -1, 1),
                         (3.2, 20.8, 1, -1), (20.8, 20.8, -1, -1)):
        b.append(poly([(x + dx * 6.2, y), (x + dx * 6.2, y + dy * 1.8),
                       (x, y + dy * 1.8), (x, y + dy * 6.2),
                       (x + dx * 1.8, y + dy * 6.2), (x + dx * 1.8, y)], fill=ACCENT))
    return svg("\n".join(b))


def B_merge():
    return svg(circ(8.2, 8.2, 5.4, fill=INK_MID) + "\n"
               + circ(15.8, 8.2, 5.4, fill=ACCENT) + "\n"
               + poly([(8.2, 16.4), (12, 21.6), (15.8, 16.4)], fill=INK))


# --- C: duotone ---------------------------------------------------------------


def C_plus():
    return svg(rect(4.4, 9.6, 15.2, 4.8, r=2.4, fill=INK_MID) + "\n"
               + rect(9.6, 4.4, 4.8, 15.2, r=2.4, fill=ACCENT))


def C_clone():
    return svg(rect(3.4, 6.8, 11.2, 11.2, r=2.6, fill=INK_MID) + "\n"
               + rect(9.4, 3.4, 11.2, 11.2, r=2.6, fill=ACCENT))


def C_histogram():
    return svg(rect(3.6, 13.0, 3.4, 8.0, r=1.7, fill=INK_MID) + "\n"
               + rect(10.3, 8.4, 3.4, 12.6, r=1.7, fill=ACCENT) + "\n"
               + rect(17.0, 11.2, 3.4, 9.8, r=1.7, fill=INK_MID))


def C_align():
    return svg(poly([(3.4, 5.2), (20.6, 5.2), (20.6, 7.6), (3.4, 7.6)], fill=ACCENT) + "\n"
               + rect(3.4, 9.4, 8.2, 9.8, r=2.4, fill=INK_MID) + "\n"
               + rect(12.4, 9.4, 8.2, 9.8, r=2.4, fill=INK))


def C_picking():
    b = [circ(x, y, 2.1, fill=INK_MID) for x, y in
         ((6.0, 6.4), (17.4, 5.2), (5.4, 17.6), (18.2, 18.0))]
    b.append(circ(12, 12, 4.8, fill=ACCENT))
    return svg("\n".join(b))


def C_smooth():
    return svg(path("M 2.6 13.4 C 6.2 13.4, 6.2 7.4, 12 7.4 C 17.8 7.4, 17.8 13.4, 21.4 13.4 "
                    "C 17.8 17.0, 6.2 17.0, 2.6 13.4 Z", fill=INK_MID) + "\n"
               + path("M 3.6 18.6 C 7.0 18.6, 7.0 15.2, 12 15.2 C 17.0 15.2, 17.0 18.6, 20.4 18.6 "
                      "C 17.0 21.0, 7.0 21.0, 3.6 18.6 Z", fill=ACCENT))


def C_clip():
    b = []
    for x, y, dx, dy in ((3.2, 3.2, 1, 1), (20.8, 3.2, -1, 1),
                         (3.2, 20.8, 1, -1), (20.8, 20.8, -1, -1)):
        b.append(poly([(x + dx * 6.4, y), (x + dx * 6.4, y + dy * 2.0),
                       (x, y + dy * 2.0), (x, y + dy * 6.4),
                       (x + dx * 2.0, y + dy * 6.4), (x + dx * 2.0, y)], fill=INK_MID))
    b.append(rect(8.6, 8.6, 6.8, 6.8, r=1.8, fill=ACCENT))
    return svg("\n".join(b))


def C_merge():
    return svg(circ(8.4, 9.0, 5.6, fill=INK_MID) + "\n"
               + circ(15.6, 9.0, 5.6, fill=ACCENT) + "\n"
               + rect(3.4, 16.2, 17.2, 4.4, r=2.2, fill=INK))


# --- D: isometric -------------------------------------------------------------


def D_plus():
    return svg(iso(12, 11.4, 7.6, gap=True) + "\n"
               + path("M 12 15.4 L 12 21.2", stroke=ACCENT, w=2.4))


def D_clone():
    return svg(iso(8.6, 10.6, 5.4, faces=(INK_LIGHT, INK_MID, INK_MID), gap=True) + "\n"
               + iso(15.2, 14.0, 5.4, faces=("#7FD0EE", ACCENT, ACCENT_DEEP), gap=True))


def D_histogram():
    b = []
    for i, (x, h, col) in enumerate(((5.0, 5.0, INK_MID), (12.0, 10.0, ACCENT),
                                     (19.0, 7.0, INK_MID))):
        b.append(rect(x - 2.0, 19.0 - h, 4.0, h, r=0.8, fill=col))
        b.append(poly([(x - 2.0, 19.0 - h), (x, 19.0 - h - 1.6),
                       (x + 2.0, 19.0 - h)], fill=INK_LIGHT if col == INK_MID else "#7FD0EE"))
    return svg("\n".join(b))


def D_align():
    return svg(iso(8.4, 9.4, 5.0, faces=(INK_LIGHT, INK_MID, INK), gap=True) + "\n"
               + iso(15.6, 14.6, 5.0, faces=("#7FD0EE", ACCENT, ACCENT_DEEP), gap=True) + "\n"
               + line(3.0, 20.6, 21.0, 20.6, c=INK_MID, w=1.8))


def D_picking():
    d = [circ(x, y, 1.3, fill=INK_LIGHT) for x, y in
         ((5.4, 9.0), (9.4, 5.4), (17.6, 6.6), (19.0, 13.0), (6.0, 18.0), (17.0, 19.0))]
    d.append(circ(12, 12, 3.2, fill=ACCENT))
    return svg("\n".join(d))


def D_smooth():
    return svg(iso(12, 10.0, 6.6, faces=(INK_LIGHT, INK_MID, INK), gap=True) + "\n"
               + path("M 3.0 19.4 C 6.4 19.4, 6.4 16.0, 12 16.0 C 17.6 16.0, 17.6 19.4, 21.0 19.4",
                      stroke=ACCENT, w=2.2))


def D_clip():
    b = [iso(12, 11.6, 6.0, faces=(INK_LIGHT, INK_MID, INK), gap=True)]
    for x, y, dx, dy in ((2.8, 2.8, 1, 1), (21.2, 2.8, -1, 1),
                         (2.8, 21.2, 1, -1), (21.2, 21.2, -1, -1)):
        b.append(poly([(x + dx * 5.0, y), (x + dx * 5.0, y + dy * 1.6),
                       (x, y + dy * 1.6), (x, y + dy * 5.0),
                       (x + dx * 1.6, y + dy * 5.0), (x + dx * 1.6, y)], fill=ACCENT))
    return svg("\n".join(b))


def D_merge():
    return svg(iso(8.0, 9.4, 5.4, faces=(INK_LIGHT, INK_MID, INK_MID), gap=True) + "\n"
               + iso(16.0, 9.4, 5.4, faces=("#7FD0EE", ACCENT, ACCENT_DEEP), gap=True) + "\n"
               + path("M 8.0 16.2 L 12 20.6 L 16.0 16.2", stroke=INK, w=2.0))


# --- E: line + accent block ---------------------------------------------------


def E_plus():
    return svg(line(5.0, 12, 19.0, 12, c=INK, w=2.4) + "\n"
               + line(12, 5.0, 12, 19.0, c=INK, w=2.4) + "\n"
               + circ(12, 12, 3.0, fill=ACCENT))


def E_clone():
    return svg(rect(3.4, 6.8, 11.2, 11.2, r=1.6, stroke=INK, w=2.0) + "\n"
               + rect(9.8, 3.4, 10.8, 10.8, r=1.6, fill=ACCENT) + "\n"
               + line(12.4, 6.4, 18.0, 11.2, c=PAPER, w=1.8))


def E_histogram():
    return svg(rect(3.6, 13.0, 3.4, 8.0, r=0.5, stroke=INK, w=1.8) + "\n"
               + rect(10.3, 8.4, 3.4, 12.6, r=0.5, fill=ACCENT) + "\n"
               + rect(17.0, 11.2, 3.4, 9.8, r=0.5, stroke=INK, w=1.8))


def E_align():
    return svg(line(4.0, 12, 20.0, 12, c=INK_MID, w=1.6) + "\n"
               + rect(3.4, 4.2, 7.4, 5.4, r=1.0, stroke=INK, w=1.9) + "\n"
               + rect(13.2, 14.4, 7.4, 5.4, r=1.0, fill=ACCENT))


def E_picking():
    d = [circ(x, y, 1.5, fill=INK) for x, y in
         ((6.0, 6.4), (17.4, 5.2), (5.4, 17.6), (18.2, 18.0))]
    d.append(circ(12, 12, 4.0, fill=ACCENT))
    return svg("\n".join(d))


def E_smooth():
    return svg(path("M 3 15.2 C 6.4 15.2, 6.4 8.8, 12 8.8 C 17.6 8.8, 17.6 15.2, 21 15.2",
                    stroke=INK, w=2.1) + "\n"
               + rect(7.4, 17.8, 9.2, 3.4, r=1.2, fill=ACCENT))


def E_clip():
    b = []
    for x, y, dx, dy in ((3.6, 3.6, 1, 1), (20.4, 3.6, -1, 1),
                         (3.6, 20.4, 1, -1), (20.4, 20.4, -1, -1)):
        b.append(path(f"M {x+dx*5.4} {y} L {x} {y} L {x} {y+dy*5.4}", stroke=INK, w=2.2))
    b.append(rect(8.0, 8.0, 8.0, 8.0, r=1.2, fill=ACCENT))
    return svg("\n".join(b))


def E_merge():
    return svg(circ(8.4, 8.8, 5.0, stroke=INK, w=2.0) + "\n"
               + circ(15.6, 8.8, 5.0, fill=ACCENT) + "\n"
               + path("M 8.4 13.8 L 12 17.4 L 15.6 13.8", stroke=INK, w=1.8))


STYLES = {
    "A-line":      [A_plus, A_clone, A_histogram, A_align, A_picking, A_smooth, A_clip, A_merge],
    "B-solid":     [B_plus, B_clone, B_histogram, B_align, B_picking, B_smooth, B_clip, B_merge],
    "C-duotone":   [C_plus, C_clone, C_histogram, C_align, C_picking, C_smooth, C_clip, C_merge],
    "D-isometric": [D_plus, D_clone, D_histogram, D_align, D_picking, D_smooth, D_clip, D_merge],
    "E-line-block":[E_plus, E_clone, E_histogram, E_align, E_picking, E_smooth, E_clip, E_merge],
}

CONCEPTS = ["plus", "clone", "histogram", "align", "picking", "smooth", "clip", "merge"]
