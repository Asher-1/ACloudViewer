"""ParaView icon language, measured from the eight samples the user chose.

Three numbers in the first version of this file were wrong, and all three were
visible in the batch-1 render as "the new icons look heavy and cramped beside the
originals".  They came from sampling the wrong thing -- counting dark pixels
instead of measuring the marks.

What the samples actually measure
--------------------------------
    outline colour   #6F6F59   (not #606040: that was a mode over an averaged
                                downscale, which shifts a 2 px stroke toward the
                                fill it sits on)
    outline width    2 px at 64 px  =  0.75 units at 24.  Measured as the modal
                     horizontal ink run across four samples: 2 px in 124/185,
                     120/159, 131/254 and 138/249 runs.  The first version used
                     1.25 -- two thirds too heavy -- which is why every shape in
                     batch 1 read as a fat outline.

    coverage         40-69 %, median ~55 %, over the eight samples.  The first
                     version's `fit()` targeted 93 %, so it inflated every
                     drawing by ~1.7x.  That target came from measuring the ecv
                     MOSAIC sources, whose scattered pixels do fill their box --
                     a bitmap artefact, not a design rule.

Palette, sampled per icon
-------------------------
The samples are soft and slightly warm, and colour is used per icon to name a
role rather than as a single accent:

    green   #A7D28C light / #72BE44 saturated   surfaces, "the surface type"
    blue    #D0E5F6 pale  / #9AC1E0 mid / #2896D3 strong   glass, water, action
    tan     #C1A561 / #A3915E                  paper, clipboard, wood
    grey    #CDCDCD / #787878                  structure, neutral bodies
    yellow  #F3C017                            direction arrows
    brown   #C4813E                            solid colour swatch

Reading the low saturation median as "neutral body plus one accent" was the
second error: pqFullscreen is yellow *and* blue, pqPaste is tan *and* pale grey.
The samples are low-saturation, not single-hue.

Gradients are part of the language.  pqIsosurface's sphere and pqSolidColor's
disc are shaded, not flat.  An earlier rule banned gradients here after traced
raster gradients produced banding; that was a fault in the tracer, not in
gradients.  Written as real SVG gradients they are resolution-independent and
cost nothing at any zoom.

Canvas
------
24 units, longest edge.  The samples are 64 px rasters; reproducing their pixel
grid is what made the originals blurry on a HiDPI screen.
"""

from __future__ import annotations

import math

CANVAS = 24

# --- measured palette ------------------------------------------------------
INK = "#6F6F59"          # outline: 2 px at 64 px in four of four samples

GREEN_LT = "#A7D28C"
GREEN = "#72BE44"
GREEN_DK = "#4E8C2E"

BLUE_PALE = "#D0E5F6"
BLUE_MID = "#9AC1E0"
BLUE_STRONG = "#2896D3"

TAN = "#C1A561"
TAN_DK = "#A3915E"

GREY_LT = "#E0E0E0"
GREY = "#CDCDCD"
GREY_MID = "#A4A6A6"
GREY_DK = "#787878"

YELLOW = "#F3C017"
BROWN = "#C4813E"

RED = "#C0392B"
ROSE = "#C2607A"
VIOLET = "#7A6FC0"
ORANGE = "#D97A32"

# A scalar-field ramp, in the order the ecv sources use it.
FIELD = ["#C0392B", "#D97A32", "#E8B22B", "#72BE44", "#2896D3", "#7A6FC0"]

# Outline weight: 2 px at 64 px is 0.75 units at 24.  Raised slightly because
# 0.75 * (18/24) = 0.56 px, which Qt renders as a grey hairline rather than a
# line; 0.85 keeps the sample's proportion while surviving the toolbar size.
STROKE = 0.85


def svg(body: str, *, w: float = CANVAS, h: float = CANVAS,
        ink: str = INK, sw: float = STROKE, defs: str = "") -> str:
    units = max(w, h)
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{units:g}" '
            f'height="{units:g}" viewBox="0 0 {w:g} {h:g}" stroke="{ink}" '
            f'stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round">\n')
    return head + (f"<defs>{defs}</defs>\n" if defs else "") + body + "\n</svg>\n"


# ---------------------------------------------------------------------------
# gradients -- the samples shade their bodies, so this is a first-class part of
# the language, not a special case
# ---------------------------------------------------------------------------


def radial(gid: str, stops, *, cx=0.5, cy=0.5, r=0.5) -> str:
    s = "".join(f'<stop offset="{o:.3f}" stop-color="{c}"/>' for o, c in stops)
    return (f'<radialGradient id="{gid}" cx="{cx}" cy="{cy}" r="{r}">'
            f"{s}</radialGradient>")


def linear(gid: str, stops, *, x1=0.0, y1=0.0, x2=0.0, y2=1.0) -> str:
    s = "".join(f'<stop offset="{o:.3f}" stop-color="{c}"/>' for o, c in stops)
    return (f'<linearGradient id="{gid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
            f"{s}</linearGradient>")


# ---------------------------------------------------------------------------
# primitives -- every shape carries the outline, because the outline is the
# language; a shape without it reads as belonging to a different set
# ---------------------------------------------------------------------------


def fill(d: str, colour: str, *, outline: str | None = INK, sw: float | None = None) -> str:
    a = f' d="{d}" fill="{colour}"'
    a += f' stroke="{outline}"' if outline else ' stroke="none"'
    if sw:
        a += f' stroke-width="{sw}"'
    return f"  <path{a}/>"


def stroke(d: str, *, colour: str | None = None, sw: float | None = None) -> str:
    a = f' d="{d}" fill="none"'
    if colour:
        a += f' stroke="{colour}"'
    if sw:
        a += f' stroke-width="{sw}"'
    return f"  <path{a}/>"


def poly(pts, colour, *, outline=INK, close=True) -> str:
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts) + (" Z" if close else "")
    return fill(d, colour, outline=outline)


def rect(x, y, w, h, colour, *, r=0.8, outline=INK) -> str:
    a = f' x="{x}" y="{y}" width="{w}" height="{h}"'
    if r:
        a += f' rx="{r}"'
    a += f' fill="{colour}"'
    a += f' stroke="{outline}"' if outline else ' stroke="none"'
    return f"  <rect{a}/>"


def circle(cx, cy, r, colour, *, outline=INK) -> str:
    a = f' cx="{cx}" cy="{cy}" r="{r}" fill="{colour}"'
    a += f' stroke="{outline}"' if outline else ' stroke="none"'
    return f"  <circle{a}/>"


def line(x1, y1, x2, y2, colour=INK, sw=None) -> str:
    w = f' stroke-width="{sw}"' if sw else ""
    return f'  <path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{colour}"{w}/>'


def arrow(x1, y1, x2, y2, colour=INK, head=2.4, sw=None, both=False) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    parts = [line(x1, y1, x2, y2, colour, sw)]
    a1, a2 = ang + math.radians(152), ang - math.radians(152)
    parts.append(poly([(x2, y2),
                       (x2 + head * math.cos(a1), y2 + head * math.sin(a1)),
                       (x2 + head * math.cos(a2), y2 + head * math.sin(a2))],
                      colour, outline=None))
    if both:
        b1, b2 = ang + math.radians(28), ang - math.radians(28)
        parts.append(poly([(x1, y1),
                           (x1 + head * math.cos(b1), y1 + head * math.sin(b1)),
                           (x1 + head * math.cos(b2), y1 + head * math.sin(b2))],
                          colour, outline=None))
    return "\n".join(parts)


def cube(cx, cy, r, *, top=BLUE_PALE, left=BLUE_MID, right=GREY_LT) -> str:
    """Isometric box, the mark pqProbeLocation uses for a data volume."""
    t = [(cx, cy - r), (cx + r * .866, cy - r * .5), (cx, cy), (cx - r * .866, cy - r * .5)]
    l = [(cx - r * .866, cy - r * .5), (cx, cy), (cx, cy + r), (cx - r * .866, cy + r * .5)]
    rr = [(cx + r * .866, cy - r * .5), (cx, cy), (cx, cy + r), (cx + r * .866, cy + r * .5)]
    return "\n".join([poly(t, top), poly(l, left), poly(rr, right)])


def blob(cx, cy, rx, ry, colour, *, seed=3, jitter=0.16, outline=INK) -> str:
    """Irregular closed form built from eight arcs.

    ecvAlign's two shapes are not rectangles or pentagons -- they are
    hand-placed organic lumps, and drawing them as polygons was the most
    obvious tell that batch 1 was a different artwork.  Eight control points
    with a deterministic per-shape jitter reproduce that irregularity without
    the result wobbling between builds.
    """
    import random

    rng = random.Random(seed)
    pts = []
    for i in range(8):
        a = 2 * math.pi * i / 8 - math.pi / 2
        k = 1.0 + rng.uniform(-jitter, jitter)
        pts.append((cx + math.cos(a) * rx * k, cy + math.sin(a) * ry * k))

    # Catmull-Rom through the points, as cubic beziers
    n = len(pts)
    d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} "
                 f"{p2[0]:.2f} {p2[1]:.2f}")
    d.append("Z")
    return fill(" ".join(d), colour, outline=outline)


# ---------------------------------------------------------------------------
# Canvas fit
# ---------------------------------------------------------------------------

def fit(body: str, target: float = 0.92, *, canvas: float = CANVAS) -> str:
    """Scale a drawing so its visible bounds span ``target`` of the canvas.

    ``target`` is **bounding-box span**, because that is what this function
    controls.  Two different numbers were measured from the eight samples and
    they must not be mixed up -- which is exactly what the earlier versions did:

        bbox span        72-97 %, median **92 %**   <-- this is the target
        ink area         41-69 %, median  56 %      <-- a consequence of shape
                                                       density, not a setting

    The first version targeted 0.93 (right number, by accident), the second
    targeted 0.62 after measuring ink area instead, and the result was icons
    visibly smaller than every original they sat beside.  At 0.92 the drawings
    occupy their box the way the samples do; how much ink is inside that box
    then follows from the drawing, as it should.

    Two passes: render the body, measure the alpha bounds, then wrap it in a
    transform that maps those bounds onto the target box, so the ratio holds by
    construction rather than by eye.  The shorter edge follows the drawing's own
    aspect ratio, as in the samples -- pqPaste is tall, pqFullscreen is wide.
    """
    import io

    import cairosvg
    import numpy as np
    from PIL import Image

    probe = svg(body, w=canvas, h=canvas)
    png = cairosvg.svg2png(bytestring=probe.encode(), output_width=240,
                           output_height=240)
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    ys, xs = np.nonzero(a[..., 3] > 60)
    if not len(xs):
        return probe

    k = 240 / canvas                       # probe pixels per canvas unit
    bx, by = xs.min() / k, ys.min() / k
    bw, bh = (xs.max() + 1 - xs.min()) / k, (ys.max() + 1 - ys.min()) / k

    # An earlier version carried `* (target / 0.93)` left over from the old
    # default.  That expressed the same ratio twice, so the number it produced
    # was not the number in the parameter.  One expression: the longer edge
    # spans `target` of the canvas.
    scale = target * canvas / max(bw, bh)
    cx, cy = bx + bw / 2, by + bh / 2
    tx, ty = canvas / 2 - cx * scale, canvas / 2 - cy * scale

    wrapped = (f'<g transform="translate({tx:.4f},{ty:.4f}) scale({scale:.5f})">\n'
               f'{body}\n</g>')
    return svg(wrapped, w=canvas, h=canvas)


# ---------------------------------------------------------------------------
# Self-check: the numbers this file claims must match the samples.
# ---------------------------------------------------------------------------

if __name__ == "__main__":                                  # pragma: no cover
    import io
    from pathlib import Path

    import cairosvg
    import numpy as np
    from PIL import Image

    root = Path(__file__).resolve().parents[3]
    src = root / "app/Resources/images/svg"
    names = ["pqPreview.png", "pqPaste.png", "pqIsosurface.png", "pqGlyph.png",
             "pqFullscreen.png", "pqRecentFile.svg", "pqProbeLocation.svg",
             "pqSolidColor.svg"]

    inks, covs = [], []
    for n in names:
        p = src / n
        if p.suffix == ".png":
            a = np.asarray(Image.open(p).convert("RGBA"))
        else:
            png = cairosvg.svg2png(url=str(p), output_width=64, output_height=64)
            a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        if a.shape[0] != 64:                     # measure in the samples' own box
            im = Image.fromarray(a).convert("RGBA").resize((64, 64), Image.LANCZOS)
            a = np.asarray(im)
        covs.append(100 * (a[..., 3] > 100).mean())
        m = a[..., 3] > 200
        px = a[..., :3][m]
        # Dark pixels, i.e. the outline family.  The cut is 340 because INK
        # itself sums to 311 and a 300 cut excludes the very colour being
        # measured -- which is what happened on the first run of this check.
        dark = px[px.sum(axis=1) < 340]
        if len(dark):
            c = np.median(dark, axis=0).astype(int)
            inks.append((n, "#%02X%02X%02X" % tuple(c)))

    print("outline samples (median of dark pixels):")
    for n, c in inks:
        print(f"  {n:<24} {c}")
    print(f"\nINK in this file       {INK}")
    print(f"coverage  min {min(covs):.0f}%  median {sorted(covs)[len(covs)//2]:.0f}%  "
          f"max {max(covs):.0f}%")
