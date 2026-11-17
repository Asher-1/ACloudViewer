"""Build the horizontal lockup: icon + wordmark, at any resolution.

Two things this file fixes, both measured:

1. ``docs/images/ACloudViewer_logo_horizontal.png`` is 1332x268 px.  At 4K
   (3840 wide) that is a 2.9x upscale and at 8K a 5.8x upscale, so the mark goes
   soft exactly where the README and the docs header show it largest.

2. The wordmark must not depend on a font being installed.  Text is converted to
   outlines, so the lockup renders identically on a build machine, in CI and in
   a browser that has never seen Lato.

The icon half is taken from the 1024 px master, so the lockup inherits the real
artwork rather than a re-drawn simplification.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MASTERS = HERE / "masters"

# Font used only to obtain outlines; the shipped asset carries no font
# dependency.  Lato Black matches the wordmark's existing weight.
WORDMARK_FONTS = (
    "/usr/share/fonts/truetype/lato/Lato-Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

INK = "#12324A"      # measured from the existing lockup's wordmark
CAP_RATIO = 0.30     # cap height / icon height, as in the current lockup
GAP_RATIO = 0.26     # gap / icon height
PAD_RATIO = 0.02


def _pick_font() -> str:
    for candidate in WORDMARK_FONTS:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("No wordmark font available; extend WORDMARK_FONTS.")


def wordmark_path(text: str) -> tuple[str, float, float]:
    """Outline `text`; returns (svg path data, advance width, cap height)."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(_pick_font())
    upm = font["head"].unitsPerEm
    try:
        cap = font["OS/2"].sCapHeight or int(upm * 0.72)
    except (KeyError, AttributeError):
        cap = int(upm * 0.72)
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    hmtx = font["hmtx"]

    x = 0.0
    chunks: list[str] = []
    for ch in text:
        name = cmap.get(ord(ch))
        if name is None:
            continue
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))   # flip Y
        d = pen.getCommands()
        if d:
            chunks.append(d)
        x += hmtx[name][0] - 0.012 * upm     # measured optical tracking
    return " ".join(chunks), x, cap


def icon_png(height: int) -> bytes:
    """The icon at the lockup's icon height, taken from the master."""
    master = Image.open(MASTERS / "acloudviewer_1024.png").convert("RGBA")
    img = master.resize((height, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build(label: str = "ACloudViewer", icon_height: int = 1024) -> str:
    """Return the lockup as an SVG, sized for whatever resolution is needed.

    ``icon_height`` is the icon's pixel size in the output; the lockup width
    follows from it, so a 4K header and a README thumbnail use the same file.
    """
    d, adv, cap = wordmark_path(label)
    scale = (icon_height * CAP_RATIO) / cap
    text_w = adv * scale
    gap = icon_height * GAP_RATIO
    pad = icon_height * PAD_RATIO
    total_w = pad * 2 + icon_height + gap + text_w
    total_h = icon_height + pad * 2
    baseline = pad + icon_height * 0.5 + icon_height * CAP_RATIO * 0.5

    icon_b64 = base64.b64encode(icon_png(icon_height)).decode("ascii")

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" '
        f'height="{total_h:.0f}" viewBox="0 0 {total_w:.2f} {total_h:.2f}">\n'
        f'  <image x="{pad:.2f}" y="{pad:.2f}" width="{icon_height}" '
        f'height="{icon_height}" href="data:image/png;base64,{icon_b64}"/>\n'
        f'  <g transform="translate({pad + icon_height + gap:.2f},{baseline:.2f}) '
        f'scale({scale:.6f})">\n'
        f'    <path d="{d}" fill="{INK}"/>\n'
        f'  </g>\n'
        f'</svg>\n'
    )
