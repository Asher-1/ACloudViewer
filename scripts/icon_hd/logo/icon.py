"""The one thing every tool icon script needs, written once.

Why this module exists
----------------------
Six of the icon scripts carried a byte-identical 39-line `__main__` block: parse
`--sheet` / `--write`, render a size ladder with PIL, optionally write the SVG.
The only difference between the copies was the description string in
`ArgumentParser`.  Roughly a third of every new icon was that block, and every
new icon meant copying it again.

The duplicated block was not the real problem.  The real problem was that each
script had *four* duties -- what the icon means, how it is drawn, where it goes,
and how it is previewed -- with no boundary between them.  Fixing the preview
meant editing seven files; a seventh icon could diverge without anyone noticing.

So the split is now:

    <icon module>   what it means (docstring) + how it is drawn (build())
    icon.py         where it goes, how to preview it, how to score it

An icon module is now just a spec.  Adding the next one is a file of drawing
code plus one line in `REGISTRY` -- no CLI, no rendering, no hashing.

What "open the box and it works" means here
-------------------------------------------
    python3 -m scripts.icon_hd.logo.icon                  # list every icon
    python3 -m scripts.icon_hd.logo.icon filter --show    # print one SVG
    python3 -m scripts.icon_hd.logo.icon filter --write   # install it
    python3 -m scripts.icon_hd.logo.icon filter --sheet x.png
    python3 -m scripts.icon_hd.logo.icon --all --score    # objective check
    python3 -m scripts.icon_hd.logo.icon --all --sheet x.png

Names are the short one, not the sixteen-character class name: `filter`, not
`ecvFilterByValue`.  The module filename matches: `filter.py`.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import paraview_style as P

ROOT = Path(__file__).resolve().parents[3]
ICON_DIR = ROOT / "app/Resources/images/svg/main"

# The size ladder every preview uses.  16 is the toolbar size that actually
# constrains the drawing; 192 is where craft is visible.  Rendering the whole
# ladder every time is the point -- an icon that works at 192 and fails at 16 is
# not finished.
SIZES = (16, 24, 32, 48, 96, 192)


@dataclass(frozen=True)
class Icon:
    """One tool icon: what it is called, what it draws, where it lands.

    `build` returns the *unfitted* SVG from ``paraview_style.svg``.  Fitting is
    applied by `render` so every icon gets the samples' 92 % span without each
    author remembering to call it.  `build` returning an already-fitted SVG is
    also fine -- `fit` is idempotent in practice because it normalises around the
    ink bbox rather than accumulating.
    """

    key: str                 # short CLI name, e.g. "filter"
    filename: str            # e.g. "toolMainFilterByValue.svg"
    build: Callable[[], str]  # returns raw SVG
    tag: str = ""            # one line, shown by --list


def fit(raw: str) -> str:
    """Normalise a raw icon SVG to the samples' 92 % canvas span.

    `raw` may be either a full document (`<svg ...>...</svg>`) or a bare body; both
    are accepted.

    The bare-body case is handled explicitly because the first version was not, and
    it failed silently rather than loudly: `raw.split(">", 1)[1]` assumes the first
    `>` closes an `<svg>` tag, but on a body it closes the *first element*. An icon
    whose first element was a `<circle>` lost that circle entirely and still
    rendered as a plausible-looking icon -- the defect only showed up as "why is
    the blue dot missing".
    """
    stripped = raw.strip()
    if stripped.startswith("<svg"):
        body = stripped.split(">", 1)[1].rsplit("</svg>", 1)[0]
    else:
        body = stripped
    return P.fit(body, target=0.92)


def svg_of(icon: Icon) -> str:
    """The finished, canvas-normalised SVG for `icon`."""
    return fit(icon.build())


def path_of(icon: Icon) -> Path:
    return ICON_DIR / icon.filename


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _png(data: str, size: int) -> "object":
    import cairosvg
    from PIL import Image

    png = cairosvg.svg2png(bytestring=data.encode(),
                           output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _font(size: int):
    from PIL import ImageFont

    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def sheet(icons: list[Icon], out: Path) -> Path:
    """A size ladder for each icon, stacked vertically.

    Kept deliberately plain: this is a working sheet for judging scale and
    weight, not a presentation board.  The comparison sheets that show the
    original and the reference samples beside the new drawing live in
    `review.make`, because they need git access and that does not belong in a
    library module.
    """
    from PIL import Image, ImageDraw

    big = max(SIZES)
    lab_w, gap, pad = 230, 22, 16
    row_h = big + pad
    w = lab_w + sum(SIZES) + gap * len(SIZES) + 30
    h = row_h * len(icons) + 20

    canvas = Image.new("RGBA", (w, h), (252, 252, 253, 255))
    dr = ImageDraw.Draw(canvas)
    font, small = _font(15), _font(12)

    x = lab_w
    for s in SIZES:
        dr.text((x, 4), f"{s}px", font=small, fill=(130, 130, 130))
        x += s + gap

    for i, ic in enumerate(icons):
        y = 20 + i * row_h
        dr.text((16, y + big // 2 - 8), ic.key, font=font, fill=(30, 30, 30))
        img = svg_of(ic)
        x = lab_w
        for s in SIZES:
            canvas.alpha_composite(_png(img, s), (x, y + big - s))
            x += s + gap
        dr.line([(16, y + row_h - 6), (w - 16, y + row_h - 6)],
                fill=(228, 228, 230), width=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)
    return out


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def registry() -> dict[str, Icon]:
    """Every icon that has been rebuilt, keyed by short name.

    Imported lazily so `--list` still works when one icon module has a syntax
    error, and so a broken module does not take the whole CLI down.
    """
    from . import (align, bilateral, clouddistance, delselsf, entitytools,
                   filter as filt, gaussian, gradient, meshdistance, sor,
                   subsample, stat)

    built = [
        Icon("align", "toolMainPointPairsAlign.svg", align.build,
             "align two clouds from picked point pairs"),
        Icon("bilateral", "toolMainBilateralFilter.svg", bilateral.build,
             "bilateral filter: smoothed inside, crease preserved"),
        Icon("gaussian", "toolMainGaussianFilter.svg", gaussian.build,
             "gaussian filter: smooths everything, edge included"),
        Icon("gradient", "toolMainSFGradient.svg", gradient.build,
             "scalar-field gradient magnitude, as a tapering ramp"),
        Icon("sor", "toolMainSORFilter.svg", sor.build,
             "statistical outlier removal: points outside the band are dropped"),
        Icon("subsample", "toolMainSubsample.svg", subsample.build,
             "keep fewer points, preserve the cloud's extent"),
        Icon("clouddistance", "toolMainCloudCloudDistance.svg",
             clouddistance.build, "point-to-point distance between two clouds"),
        Icon("meshdistance", "toolMainCloudMeshDistance.svg",
             meshdistance.build, "point-to-mesh distance, graduated arrow"),
        Icon("stat", "toolMainComputeStat.svg", stat.build,
             "fitted distribution with mean and sigma"),
        Icon("delsf", "toolMainDeleteCurrentSF.svg", delselsf.build,
             "remove the displayed scalar field"),
        Icon("filter", "toolMainFilterByValue.svg", filt.build,
             "keep a value range on a scalar-field ramp"),
        Icon("addconstsf", "toolMainAddConstSF.svg",
             entitytools.add_constant_sf, "add a constant scalar field"),
        Icon("clippingbox", "toolMainClippingBox.svg",
             entitytools.clipping_box, "clip a volume with a horizontal section"),
        Icon("clipmulti", "toolMainClippingBoxMultiExport.svg",
             entitytools.clipping_multi_export, "create multiple repeated slices"),
        Icon("clipsingle", "toolMainClippingBoxSingleExport.svg",
             entitytools.clipping_single_export, "create a cloud from one section"),
        Icon("clone", "toolMainCloneEntity.svg",
             entitytools.clone_entity, "duplicate the selected entities"),
    ]
    return {ic.key: ic for ic in built}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve(name: str, reg: dict[str, Icon]) -> Icon:
    if name in reg:
        return reg[name]
    stem = name.removesuffix(".py").removesuffix(".svg").lower()
    if stem in reg:
        return reg[stem]
    close = [k for k in reg if stem in k or k in stem]
    hint = f"  did you mean: {', '.join(close)}" if close else ""
    raise SystemExit(f"unknown icon {name!r}.{hint}\n"
                     f"known: {', '.join(sorted(reg))}")


def main(argv=None) -> int:
    reg = registry()

    ap = argparse.ArgumentParser(
        prog="python3 -m scripts.icon_hd.logo.icon",
        description="Preview, score and install the tool icons.")
    ap.add_argument("names", nargs="*", help="icon names; omit with --all")
    ap.add_argument("--all", action="store_true", help="every registered icon")
    ap.add_argument("--list", action="store_true", help="show them and exit")
    ap.add_argument("--show", action="store_true", help="print the SVG")
    ap.add_argument("--write", action="store_true", help="install into the repo")
    ap.add_argument("--sheet", default=None, help="write a size-ladder PNG here")
    ap.add_argument("--score", action="store_true",
                    help="check against the samples' measured bands")
    a = ap.parse_args(argv)

    if a.list or (not a.names and not a.all and
                  not (a.show or a.write or a.sheet or a.score)):
        for k in sorted(reg):
            print(f"  {k:<14} {reg[k].filename:<34} {reg[k].tag}")
        return 0

    chosen = list(reg.values()) if a.all else [_resolve(n, reg) for n in a.names]

    if a.score:
        from .stylecheck import judge, measure, _load

        print(f"{'icon':<14} {'conn':>5} {'dens':>7} {'layers':>7} "
              f"{'span':>6}  verdict")
        bad = 0
        for ic in chosen:
            m = measure(_load(data=svg_of(ic).encode()))
            probs = judge(m)
            bad += bool(probs)
            print(f"{ic.key:<14} {m['components']:5d} {m['density']:6.1f}% "
                  f"{m['layers']:7d} {m['span']:5.1f}%  "
                  f"{'OK' if not probs else ' / '.join(probs)}")
        print(f"\n{len(chosen) - bad}/{len(chosen)} within the samples' bands")

    if a.write:
        for ic in chosen:
            p = path_of(ic)
            p.write_text(svg_of(ic))
            print(f"wrote {p}")

    if a.show:
        for ic in chosen:
            if len(chosen) > 1:
                print(f"<!-- {ic.key} -->")
            print(svg_of(ic))

    if a.sheet:
        print(f"sheet -> {sheet(chosen, Path(a.sheet))}")

    return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
