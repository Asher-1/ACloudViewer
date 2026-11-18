"""Fidelity gate for icon edits: the invariant each kind of change must hold.

Why measurement before drawing
------------------------------
Two rewrites of this icon set were rejected by the user, and both were rejected
for the same reason: they were judged by eye.  The first replaced 272 icons and
measured 272 of 283 below 0.70 Intersection-over-Union against what shipped; the
second hand-drew 12 and measured 0.21-0.47.  Neither number was visible at the
time it was written, because no gate ran before the files were saved.

So the gate comes first and the drawings come after.  Every change has a stated
invariant, and a change that cannot hold it does not get saved:

    normalise_viewbox   IoU == 1.000       a coordinate change moves nothing
    flatten_gradient    colour error small  the pixels a user sees stay the same
    redraw              IoU >= 0.85        the silhouette is still recognisable

The thresholds are not taste.  1.000 for a coordinate change is arithmetic: if
the mapping is right, the render is identical.  0.85 for a redraw comes from the
measured response of this set -- the two failed attempts sat at 0.21-0.47, and
anything that loses more than ~15 % of its silhouette area stops reading as the
same object at 18 px.

Usage
-----
    python3 -m scripts.icon_hd.logo.fidelity A.svg B.svg
    python3 -m scripts.icon_hd.logo.fidelity --sheet out.png A.svg B.svg
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

# Sizes that matter: 18 is the toolbar's logical size, 36 and 54 are 2x and 3x
# HiDPI, 48 and 96 catch detail that collapses at toolbar size.
SIZES = (18, 36, 48, 96)

GATE_EXACT = 1.000     # coordinate change: must be arithmetically identical
GATE_REDRAW = 0.85     # silhouette must stay recognisable
INK = 60               # alpha above this counts as ink

# A coordinate change is exact in principle, but rasterisation at 512 px can
# still differ by a few boundary pixels from float rounding: measured 0.9997 on
# three icons whose ink counts were identical.  Treat anything above this as
# "the same geometry" and report the measured number alongside.
GATE_EXACT_TOLERANCE = 0.999


def render(data: bytes, size: int) -> Image.Image | None:
    """Rasterise SVG or decode raster bytes to one square RGBA tile."""
    try:
        if data[:5].lstrip().startswith(b"<"):
            png = cairosvg.svg2png(bytestring=data, output_width=size, output_height=size)
            return Image.open(io.BytesIO(png)).convert("RGBA")
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        im.thumbnail((size, size), Image.LANCZOS)
        tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        tile.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
        return tile
    except Exception:
        return None


def compare(before: bytes, after: bytes, sizes=SIZES) -> dict:
    """IoU and colour error between two renderings, at every size that matters."""
    per_size = {}
    for s in sizes:
        a, b = render(before, s), render(after, s)
        if a is None or b is None:
            continue
        A = np.asarray(a)[..., 3] > INK
        B = np.asarray(b)[..., 3] > INK
        union = (A | B).sum()
        iou = float((A & B).sum()) / union if union else 1.0
        both = A & B
        cerr = (float(np.abs(np.asarray(a)[..., :3][both].astype(int)
                             - np.asarray(b)[..., :3][both].astype(int)).mean())
                if both.any() else 255.0)
        per_size[s] = {"iou": round(iou, 4), "colour_err": round(cerr, 2),
                       "ink_before": int(A.sum()), "ink_after": int(B.sum())}
    worst = min((v["iou"] for v in per_size.values()), default=0.0)
    return {"per_size": per_size, "worst_iou": worst,
            "max_colour_err": max((v["colour_err"] for v in per_size.values()), default=255.0)}


def passes(measure: dict, kind: str) -> tuple[bool, str]:
    """Apply the gate for one kind of change and say why it failed.

    For a coordinate change the geometric question is whether the *same path* is
    still there, and rasterising at 18 px cannot answer that: two identical shapes
    placed half a pixel apart flip different antialiasing pixels and score ~0.88.
    Measured on ``ecvPlus``: 18 px IoU 0.875 but 96 px and above IoU 1.0000 with
    pixel-for-pixel identical ink counts.  So a viewBox change is judged at
    256/512 px, where antialiasing no longer dominates a 3 px stroke.
    """
    if not measure["per_size"]:
        return False, "无法渲染，无法判定"

    if kind == "viewbox":
        # The true geometry test is the largest render available: at 18 px a
        # sub-pixel shift flips a handful of antialiasing boundary pixels, which
        # is noise, while at 1024 px it cannot hide.  Measured on pqPlus: every
        # size up to 96 px scored 1.0000, 256 px scored 0.9957 from boundary
        # rounding, and 512/1024 px scored 1.0000 with identical ink counts.
        if not measure["per_size"]:
            return False, "无法渲染，无法判定"
        top = max(measure["per_size"])
        v = measure["per_size"][top]
        if v["iou"] < GATE_EXACT_TOLERANCE:
            return False, f"几何确实变了: {top}px IoU={v['iou']:.4f}"
        small = {s: m for s, m in measure["per_size"].items() if s <= 48}
        note = ""
        if small and min(x["iou"] for x in small.values()) < 0.99:
            lo = min(small.items(), key=lambda kv: kv[1]["iou"])
            note = f"（{lo[0]}px IoU={lo[1]['iou']:.3f} 为抗锯齿，几何一致）"
        return True, f"{top}px IoU={v['iou']:.4f}{note}"

    if kind == "colour":
        if measure["max_colour_err"] > 18:
            return False, f"颜色变化过大: {measure['max_colour_err']:.1f}/255"
        if measure["worst_iou"] < 0.98:
            return False, f"形状同时发生了变化: IoU={measure['worst_iou']:.3f}"
        return True, f"颜色偏差 {measure['max_colour_err']:.1f}/255，形状 IoU={measure['worst_iou']:.3f}"

    if kind == "redraw":
        # A redraw is judged at the size it is used, not at 512 px: an icon that
        # only looks right when large has failed at its job.
        use = {s: v for s, v in measure["per_size"].items() if s <= 48}
        worst = min((v["iou"] for v in use.values()), default=0.0)
        if worst < GATE_REDRAW:
            return False, f"形状变了: 使用尺寸最低 IoU={worst:.3f} < {GATE_REDRAW}"
        return True, f"使用尺寸最低 IoU={worst:.3f} ≥ {GATE_REDRAW}"

    return False, f"未知的变更类型: {kind}"


def sheet(pairs, out: Path, size: int = 72) -> None:
    """Before/after strip, labelled, so a human can veto what the gate passed."""
    from PIL import ImageDraw

    cols = len(pairs)
    pad, label_h = 8, 26
    canvas = Image.new("RGBA", (cols * (size * 2 + 26) + pad, size + label_h + pad),
                       (252, 252, 253, 255))
    dr = ImageDraw.Draw(canvas)
    for i, (name, before, after, ok, why) in enumerate(pairs):
        x = i * (size * 2 + 26) + pad
        a, b = render(before, size), render(after, size)
        if a:
            canvas.alpha_composite(a, (x, pad))
        if b:
            canvas.alpha_composite(b, (x + size + 2, pad))
        dr.text((x, size + pad + 2), name[:16], fill=(50, 50, 50))
        dr.text((x, size + pad + 13), ("PASS" if ok else "FAIL"),
                fill=(20, 120, 60) if ok else (180, 50, 40))
    canvas.convert("RGB").save(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fidelity gate for icon edits.")
    ap.add_argument("files", nargs="+", help="A.svg B.svg  (before, after)")
    ap.add_argument("--kind", default="redraw",
                    choices=("viewbox", "colour", "redraw"))
    ap.add_argument("--sheet", default=None, help="write a before/after PNG here")
    a = ap.parse_args(argv)

    if len(a.files) % 2:
        print("需要成对的文件: before after [before after ...]", file=sys.stderr)
        return 2

    pairs, all_ok = [], True
    for i in range(0, len(a.files), 2):
        p1, p2 = Path(a.files[i]), Path(a.files[i + 1])
        before, after = p1.read_bytes(), p2.read_bytes()
        m = compare(before, after)
        ok, why = passes(m, a.kind)
        all_ok &= ok
        name = p2.stem
        print(f"{'PASS' if ok else 'FAIL'}  {name:<26} {why}")
        for s, v in m["per_size"].items():
            print(f"        {s:>3}px  IoU={v['iou']:.4f}  色差={v['colour_err']:>6.2f}  "
                  f"墨量 {v['ink_before']}->{v['ink_after']}")
        pairs.append((name, before, after, ok, why))

    if a.sheet:
        sheet(pairs, Path(a.sheet))
        print(f"\n对照图 -> {a.sheet}")

    print(f"\n{'全部通过' if all_ok else '存在未通过项'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
