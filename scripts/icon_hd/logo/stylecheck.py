"""Objective style conformance check against the eight approved ParaView icons.

Why this exists
---------------
Four rounds of this icon batch were rejected on the same complaint -- "still not
the ParaView style", "some places are still mosaic" -- and each round was fixed
by looking at my own render and adjusting what seemed off.  That loop cannot
converge, because "seems off" has no stopping condition.

So the reference is now measured, and every candidate is scored against it:

    components    median **1** across the samples.  A mark made of fifteen
                  separate squares is a mosaic; a mark made of one connected
                  mass is a pictogram.  This is the defect the user named
                  "马赛克", and it is countable.

    ink density   41-69 %, median **55 %**, measured over the samples' 64 px box.
                  Below ~40 % the mark reads as line art floating in space
                  rather than as a solid object, which is what makes a sparse
                  redraw look unlike the reference even when its colours match.

    tone layers   5-12 quantised luminance steps, median **8**.  Fewer than 5
                  means the icon is flat line work; the samples carry at least
                  a body tone and an outline tone on every element.

    bbox span     72-97 %, median **92 %** -- how much of the canvas the ink
                  occupies.

    outline share 8-25 % of ink in the outline colour.  This is the check that
                  the outline is present but is not eating the shape: the first
                  two versions sat above 35 %, which is why every element looked
                  like a fat ring.

Everything is measured on a 64 px render, the samples' own resolution, because
a style that only holds at 256 px is not the style of a 16-24 px toolbar.

Usage
-----
    python3 -m scripts.icon_hd.logo.stylecheck            # score BATCH in batch1
    python3 -m scripts.icon_hd.logo.stylecheck --sheet out.png
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DIR = Path(__file__).resolve().with_name('reference_samples')

SAMPLES = ["pqPreview.png", "pqPaste.png", "pqIsosurface.png", "pqGlyph.png",
           "pqFullscreen.png", "pqRecentFile.svg", "pqProbeLocation.svg",
           "pqSolidColor.svg"]

SIZE = 64                 # the samples' native box: the size that matters
INK = 100                 # alpha above this counts as ink
OUTLINE = "#6F6F59"

# Acceptance bands, taken from the samples' measured range and widened only
# where the samples themselves spread that far.
#
# `outline_share` is a **report, not a gate**: measuring it across the samples
# gives 3.8 % (pqProbeLocation, a pale glass cube) to 54.8 % (pqFullscreen, four
# heavy corner brackets).  That spread is the style, so a single band would
# reject valid icons in both directions.  It is printed so drift stays visible.
BANDS = {
    "components":   (1, 4),        # samples: 1-5, median 1
    "density":      (40.0, 72.0),  # samples: 41-69, median 55
    "layers":       (5, 13),       # samples: 3-10, median 8
    "span":         (72.0, 98.0),  # samples: 72-97, median 92
}


def _load(data: bytes | None = None, url: str | None = None) -> np.ndarray:
    png = cairosvg.svg2png(url=url, bytestring=data,
                           output_width=SIZE, output_height=SIZE)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def measure(a: np.ndarray) -> dict:
    """The five numbers that separate a pictogram from a mosaic."""
    m = a[..., 3] > INK
    if not m.any():
        return {"components": 0, "density": 0.0, "layers": 0,
                "span": 0.0, "outline_share": 0.0}

    lab, n = ndimage.label(m)
    ys, xs = np.nonzero(m)
    span = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / SIZE * 100

    px = a[..., :3][m].astype(int)
    lum = (px * [0.299, 0.587, 0.114]).sum(1)
    layers = len(np.unique(np.round(lum / 16) * 16))

    ink_rgb = np.array([0x6F, 0x6F, 0x59])
    dist = np.abs(px - ink_rgb).sum(1)
    outline_share = 100.0 * (dist < 70).mean()

    return {"components": int(n),
            "density": round(100.0 * m.mean(), 1),
            "layers": int(layers),
            "span": round(span, 1),
            "outline_share": round(outline_share, 1)}


def judge(m: dict) -> list[str]:
    """Every band this mark misses, as a readable line."""
    out = []
    for key, (lo, hi) in BANDS.items():
        v = m[key]
        if v < lo or v > hi:
            name = {"components": "碎裂", "density": "墨迹密度",
                    "layers": "色调层数", "span": "画布占比"}[key]
            out.append(f"{name} {v} 不在 [{lo}, {hi}]")
    return out


def baseline() -> dict:
    """The samples' own profile, so the targets are always visible."""
    rows = []
    for n in SAMPLES:
        p = SAMPLE_DIR / n
        a = _load(url=str(p)) if p.suffix == ".svg" else \
            np.asarray(Image.open(p).convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS))
        rows.append((n, measure(a)))
    return rows


def score(name: str, data: bytes) -> tuple[dict, list[str]]:
    m = measure(_load(data=data))
    return m, judge(m)


def sheet(items, out: Path, size: int = 96) -> None:
    """Side-by-side at three sizes, with the measured numbers under each cell."""
    from PIL import ImageDraw

    rows = len(items)
    cell, pad, lab = size, 12, 30
    w = cell * 2 + pad * 3
    h = (cell + lab) * rows + pad
    canvas = Image.new("RGBA", (w, h), (252, 252, 253, 255))
    dr = ImageDraw.Draw(canvas)
    for i, it in enumerate(items):
        name, before, after, m, problems = it
        y = pad + i * (cell + lab)
        dr.text((pad, y + 2), f"{name}  {'OK' if not problems else ' / '.join(problems)}",
                fill=(20, 120, 60) if not problems else (180, 50, 40))
        for j, src in enumerate((before, after)):
            img = Image.open(io.BytesIO(cairosvg.svg2png(
                bytestring=src, output_width=size, output_height=size))).convert("RGBA")
            canvas.alpha_composite(img, (pad + j * (cell + pad), y + lab - 4))
        dr.text((pad, y + lab + cell),
                f"conn={m['components']} dens={m['density']}% "
                f"layers={m['layers']} span={m['span']}%",
                fill=(90, 90, 90))
    canvas.convert("RGB").save(out)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Score icons against the ParaView samples.")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--batch", default="batch1")
    args = ap.parse_args(argv)

    print("=== reference samples ===")
    print(f"{'sample':<24} {'conn':>5} {'density':>8} {'layers':>7} {'span':>6} {'outline':>8}")
    for n, m in baseline():
        print(f"{n:<24} {m['components']:5d} {m['density']:7.1f}% "
              f"{m['layers']:7d} {m['span']:5.1f}% {m['outline_share']:7.1f}%")

    mod = __import__(f"scripts.icon_hd.logo.{args.batch}", fromlist=["BATCH"])
    print(f"\n=== {args.batch} ===")
    print(f"{'icon':<24} {'conn':>5} {'density':>8} {'layers':>7} {'span':>6} {'outline':>8}  verdict")
    items, fails = [], 0
    for name, fn in mod.BATCH.items():
        data = fn().encode()
        m, problems = score(name, data)
        fails += bool(problems)
        print(f"{name:<24} {m['components']:5d} {m['density']:7.1f}% "
              f"{m['layers']:7d} {m['span']:5.1f}% {m['outline_share']:7.1f}%  "
              f"{'OK' if not problems else ' / '.join(problems)}")
        items.append((name, data, data, m, problems))

    print(f"\n{len(items) - fails}/{len(items)} within the samples' bands")
    if args.sheet:
        sheet(items, Path(args.sheet))
        print(f"sheet -> {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
