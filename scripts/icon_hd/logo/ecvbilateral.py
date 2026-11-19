"""ecvBilateralFilter -- three concepts, derived from what the code does.

What the code does
------------------
    ecvEntityAction.cpp:1032  sfBilateralFilter() -> filterType = BILATERAL
    ecvEntityAction.cpp:874   sigmaSF defaults to (max-min)/4 of the field
    ecvPointCloud.h:868       "the weights are computed considering the
                              difference of the neighbors SF values ... (also
                              following a Gaussian distribution)"

Two Gaussian weights multiplied: one on distance, one on value difference.
The consequence -- and the only thing an icon can show -- is that **values are
smoothed inside a region and the boundary between regions survives**, because a
neighbour across the boundary has a large value difference and gets ~zero
weight. A Gaussian filter would blur the boundary too; that is the difference
between the two toolbar buttons.

Why the old icon is not a reference
-----------------------------------
It is 184 `<rect>` elements and ~90 distinct greys sampled from a bitmap: a soft
grey sphere with an indistinct pale patch. A soft grey blur is the opposite of
the tool's defining property, so copying it would preserve the confusion.

The three concepts
------------------
A  Cross-section       a height profile: two gentle terraces joined by a crisp
                        cliff. Smoothed everywhere except at the jump.
B  Crease             two surface patches meeting at a hard fold. Soft shading
                        on both faces, one sharp crease between them.
C  Kernel stops       a soft kernel window sitting on a surface, cut off by a
                        bold edge. The glow reaches the edge and does not cross.

All three are drawn with the samples' construction: #6F6F59 outline on every
form, subtle gradients, large shapes, 92 % bound span.  Each is scored against
the eight approved samples by `stylecheck`-compatible metrics.

Usage
-----
    python3 -m scripts.icon_hd.logo.ecvbilateral --sheet out.png
    python3 -m scripts.icon_hd.logo.ecvbilateral --concept A
"""

from __future__ import annotations

import io
from pathlib import Path

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "app/Resources/images/svg/ecv/ecvBilateralFilter.svg"

# Colours taken from the eight samples' measured palette, not invented.
#
# Sampled frequency across all 17,114 opaque sample pixels:
#     #6F6F59 10.2 %  (outline)      #787878 6.9 %
#     #A7D28C  6.5 %  (pqPreview)    #72BE44 3.1 %  (pqIsosurface)
#     #D0E5F6  5.6 %  (pqGlyph)      #9AC1E0 1.7 %
#     #C1A561  4.3 %  (pqPaste)      #2896D3 1.3 %  (pqFullscreen)
#
# The two faces of the fold use **one hue at two values** (GREEN_LT / GREEN),
# because that is exactly what the filter works on: two regions of the same
# property at different levels.  Red is absent from the whole sample set, so the
# earlier brick-red faces were outside the language entirely.
FACE_HI = P.GREEN_LT      # #A7D28C -- the raised face
FACE_LO = P.GREEN         # #72BE44 -- the lower face
EDGE = P.GREEN_DK         # #4E8C2E -- used only inside the crease shading


# ---------------------------------------------------------------------------
# A -- cross-section with a cliff
# ---------------------------------------------------------------------------


def concept_a() -> str:
    """A height profile: a low terrace, a crisp cliff, a high terrace.

    The terraces carry a very slight undulation and the cliff does not, so the
    single hard line is the thing the eye lands on.  That contrast is the filter:
    smoothed inside, preserved at the step.
    """
    outline = ("M 1.8 21.4 L 1.8 13.0 "
               "C 3.6 12.4, 5.6 13.2, 7.4 12.7 C 9.0 12.3, 10.2 12.9, 11.2 12.6 "
               "L 13.0 7.6 "
               "C 14.8 7.0, 16.4 7.8, 18.0 7.3 C 19.6 6.9, 20.9 7.4, 22.2 7.1 "
               "L 22.2 21.4 Z")
    defs = (
        '<clipPath id="ac"><path d="' + outline + '"/></clipPath>'
        '<linearGradient id="aL" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#DCDCDC"/>'
        f'<stop offset="1" stop-color="{P.GREY_DK}"/></linearGradient>'
        '<linearGradient id="aR" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{P.GREY_LT}"/>'
        f'<stop offset="1" stop-color="{P.GREY}"/></linearGradient>'
    )
    body = "\n".join([
        '  <g clip-path="url(#ac)">'
        '<rect x="0.6" y="6" width="11.6" height="16" fill="url(#aL)"/>'
        '<rect x="12.2" y="6" width="12.4" height="16" fill="url(#aR)"/>'
        "</g>",
        f'  <path d="{outline}" fill="none" stroke="{P.INK}" '
        f'stroke-width="{P.STROKE}" stroke-linejoin="round"/>',
        f'  <path d="M 11.2 12.6 L 13.0 7.6" fill="none" stroke="{P.INK}" '
        f'stroke-width="1.3" stroke-linecap="round"/>',
    ])
    return P.svg(body, defs=defs)


# ---------------------------------------------------------------------------
# B -- two surface patches meeting at a crease
# ---------------------------------------------------------------------------


def concept_b() -> str:
    """Two folded surface patches: soft shading on each face, one hard crease.

    A fold is the clearest 3D reading of "the boundary is intact while the
    surface either side is smooth", and it matches the application domain --
    these filters run on point clouds and meshes, and a crease is what survives.

    Colours are the samples' own greens: the raised face is GREEN_LT (#A7D28C,
    pqPreview's dominant fill) and the lower face is GREEN (#72BE44,
    pqIsosurface's saturated green).  One hue at two values says what the filter
    does -- the same property at two levels -- where grey said nothing and red
    said something no sample uses.

    The patches were also widened: at the earlier size the icon measured 36 %
    ink against the samples' 41-69 %, because the fold left too much empty
    canvas at the top and bottom.
    """
    left = [(1.6, 8.4), (11.2, 12.2), (11.2, 21.6), (1.6, 18.0)]
    right = [(11.2, 12.2), (22.4, 5.6), (22.4, 15.2), (11.2, 21.6)]
    defs = (
        '<linearGradient id="bL" x1="0" y1="0" x2="1" y2="0.6">'
        f'<stop offset="0" stop-color="{FACE_LO}"/>'
        f'<stop offset="1" stop-color="{EDGE}"/></linearGradient>'
        '<linearGradient id="bR" x1="1" y1="0" x2="0" y2="0.7">'
        f'<stop offset="0" stop-color="{FACE_HI}"/>'
        f'<stop offset="1" stop-color="{FACE_LO}"/></linearGradient>'
    )
    body = "\n".join([
        '  <path d="M ' + " L ".join(f"{x} {y}" for x, y in left) +
        f'" fill="url(#bL)" stroke="{P.INK}" stroke-width="{P.STROKE}" '
        'stroke-linejoin="round"/>',
        '  <path d="M ' + " L ".join(f"{x} {y}" for x, y in right) +
        f'" fill="url(#bR)" stroke="{P.INK}" stroke-width="{P.STROKE}" '
        'stroke-linejoin="round"/>',
        # The crease, emphasised: it is the preserved edge.
        f'  <path d="M 11.2 12.5 L 11.2 21.3" fill="none" stroke="{P.INK}" '
        f'stroke-width="1.3" stroke-linecap="round"/>',
    ])
    return P.svg(body, defs=defs)


# ---------------------------------------------------------------------------
# C -- the kernel stops at the edge
# ---------------------------------------------------------------------------


def concept_c() -> str:
    """A soft kernel window over a surface, cut off by a bold edge.

    The kernel is the smoothing; the reach of its glow is clipped at the line,
    which is what the range weight does numerically.
    """
    defs = (
        '<linearGradient id="cSurf" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{P.GREY}"/>'
        f'<stop offset="1" stop-color="{P.GREY_DK}"/></linearGradient>'
        '<radialGradient id="cKern" cx="0.35" cy="0.4" r="0.75">'
        '<stop offset="0" stop-color="#FBFBFB" stop-opacity="0.95"/>'
        f'<stop offset="1" stop-color="{P.GREY_LT}" stop-opacity="0.35"/>'
        "</radialGradient>"
        '<clipPath id="cLeft"><rect x="2" y="3" width="11.6" height="18"/></clipPath>'
    )
    body = "\n".join([
        f'  <rect x="1.6" y="3.4" width="20.8" height="17.2" rx="1.2" '
        f'fill="url(#cSurf)" stroke="{P.INK}" stroke-width="{P.STROKE}"/>',
        # The kernel window, clipped so its glow cannot cross the edge.
        '  <g clip-path="url(#cLeft)">'
        '<rect x="4.2" y="6.2" width="9.6" height="11.6" rx="1.6" '
        f'fill="url(#cKern)" stroke="{P.INK}" stroke-width="0.7"/></g>',
        f'  <path d="M 13.2 3.4 L 13.2 20.6" fill="none" stroke="{P.INK}" '
        f'stroke-width="1.4" stroke-linecap="round"/>',
    ])
    return P.svg(body, defs=defs)


CONCEPTS = {"A": concept_a, "B": concept_b, "C": concept_c}


def fitted(name: str) -> str:
    raw = CONCEPTS[name]()
    inner = raw.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return P.fit(inner, target=0.92)


if __name__ == "__main__":                                   # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="ecvBilateralFilter concepts.")
    ap.add_argument("--concept", default=None, choices=sorted(CONCEPTS))
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--write", default=None, choices=sorted(CONCEPTS))
    a = ap.parse_args()

    if a.write:
        TARGET.write_text(fitted(a.write))
        print(f"wrote {TARGET}  (concept {a.write})")
    if a.concept:
        print(fitted(a.concept))
    if a.sheet:
        import cairosvg
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
        except Exception:
            font = ImageFont.load_default()

        sizes = (16, 24, 32, 48, 96, 192)
        names = sorted(CONCEPTS)
        lab_w, gap, top = 300, 24, 40
        w = lab_w + sum(sizes) + gap * len(sizes) + 20
        h = top + len(names) * (max(sizes) + 34) + 16
        canvas = Image.new("RGBA", (w, h), (252, 252, 253, 255))
        dr = ImageDraw.Draw(canvas)
        x = lab_w
        for s in sizes:
            dr.text((x, 12), f"{s}px", font=font, fill=(120, 120, 120))
            x += s + gap
        for i, nm in enumerate(names):
            y = top + i * (max(sizes) + 34)
            dr.text((14, y + max(sizes) // 2 - 10), f"{nm}  concept_{nm.lower()}",
                    font=font, fill=(30, 30, 30))
            svg = fitted(nm).encode()
            x = lab_w
            for s in sizes:
                png = cairosvg.svg2png(bytestring=svg, output_width=s,
                                       output_height=s)
                canvas.alpha_composite(
                    Image.open(io.BytesIO(png)).convert("RGBA"),
                    (x, y + max(sizes) - s))
                x += s + gap
        canvas.convert("RGB").save(a.sheet)
        print(f"sheet -> {a.sheet}")
