# Product logos

Two product marks, two different natures, two different treatments. This folder
holds the measurement code, the vector reconstruction and the export pipeline
that back that statement.

## What was measured, not assumed

Both high-resolution masters were recovered from legacy `*.svg` wrappers in git
(ACloudViewer's was 1.48 MB of base64 around a 1024 px PNG; CloudViewer's was
559 KB of the same). They now live in `masters/` as the source of truth, and the
wrappers are gone: ACloudViewer's was 0 % vector and referenced by nothing, and
CloudViewer's has been replaced at its own path by genuine vector art.

`measure.py` reads the masters and prints the geometry that `geometry.py` states
as constants, so every number below is reproducible:

| Mark | Property | Measured |
|---|---|---|
| ACloudViewer | cube vertices | (512,20) (924,256) (924,762) (512,1002) (99,762) (99,252) |
| ACloudViewer | core centre | (505, 500), hot radius 42, bloom 210 |
| ACloudViewer | loose points | 80 blobs, r 4.5–11.2, x 109–466, y 58–931 |
| ACloudViewer | left face | 25.9% fully transparent, 72.2% fully opaque, 2% between |
| ACloudViewer | left face ink | `#6C7F9A` at 1024 px, `#2B3A8F` at 32 px |
| CloudViewer | C ring radii | inner 242, outer 355, tails out to 489 |
| CloudViewer | disc | centre (511, 510), radius 155, 92 dots |
| CloudViewer | ribbon ramp | `#92E5EE` → `#75CBE1` → `#6FBCE8` |

## ACloudViewer: kept as painted artwork, on purpose

Six vectorisation routes were built and measured against the master. Every one
lost the design or cost more than the file it would replace:

| Route | Size | IoU | Why it fails |
|---|---:|---:|---|
| Parametric redraw | 10 KB | 0.81 | drops the brush-stroke texture and the star-burst outline |
| Silhouette fill | 97 KB | 0.99 | interior collapses into flat colour |
| Posterised layers | 850–2600 KB | 0.87–0.92 | gradients band into visible patches |
| Stroke-skeleton trace | 97 KB | 0.89 | ink inflates into capsule blobs |
| Single-outline evenodd | 128 KB | 0.90 | texture gone, files larger than the source |
| Full contour trace | 135 KB | 0.985 | silhouette exact, but one flat fill inside |

The mark is a **painted texture, not geometry**. Its impact is the density of
those strokes and the star-burst silhouette they form. Replacing it with clean
vector shapes is a redesign, not a refinement — and it was explicitly out of
scope.

Its existing raster ladder is also not a defect. Measured against a Lanczos
downscale of the master, the checked-in 32 px icon is *sharper* than the
downscale would be (solid navy left face, clean teal top, one small amber core),
which is what per-size optical tuning looks like. Nothing to fix there.

## CloudViewer: genuinely geometric, so it is now vector

The C is a tapered ribbon around a disc — real geometry. A vector
reconstruction beats the checked-in PNGs at every size up to 128 px:

| Size | repo PNG error | vector error |
|---|---:|---:|
| 16 px | 60.2 | **32.6** |
| 24 px | 42.5 | **23.0** |
| 32 px | 32.9 | **21.6** |
| 48 px | 25.9 | **19.4** |
| 64 px | 21.1 | **16.5** |
| 128 px | 10.4 | 10.5 |
| 256 px | **4.9** | 7.4 |

(error = mean absolute difference vs a master downscale, 0–255)

So 16–128 px are rendered from vector, and the 256 px raster is kept because it
wins there. `cloudviewer_256.svg` is now true vector (15,970 B) instead of a
559 KB base64 wrapper with zero `<path>` elements in it.

## Regenerate

```bash
python3 -m scripts.icon_hd.logo.export_cloudviewer          # write CloudViewer icons
python3 -m scripts.icon_hd.logo.export_cloudviewer --dry-run
python3 -m scripts.icon_hd.logo.measure                     # print measurements
python3 -m scripts.icon_hd.logo.measure --write             # refresh particles.json
```

Dependencies: `cairosvg`, `Pillow`, `numpy`, `scipy`, `scikit-image`, `opencv-python`.
Check them with `python3 scripts/icon_hd/icons.py doctor`.

## The toolchain entry point

Everything in this folder is driven by one command:

```bash
python3 scripts/icon_hd/icons.py doctor   # what is installed, what is missing
python3 scripts/icon_hd/icons.py build    # list current tool icons and file sizes
python3 scripts/icon_hd/icons.py audit    # before/after review page
python3 scripts/icon_hd/icons.py verify   # qrc + XML + raster-in-svg + platforms
python3 scripts/icon_hd/icons.py logo measure   # product-mark geometry
```

Tool icons are documented in [`../TOOLICONS.md`](../TOOLICONS.md); this file
covers only the product marks, which are a different problem (painted artwork,
refined rather than redrawn).

## What was removed and why

A previous pipeline (`lossless.py`, `redesign.py`, `fit.py`, `gallery.py`,
`compare4k.py`, `icon_hd.py`) transcribed raster icons into SVG fixtures and
needed `PySide6` and `potrace`. It was deleted for three independent reasons:

* `icons.txt` listed 238 repo PNGs; **0 of them still exist**, because the icon
  set is now SVG. The pipeline would have written new PNGs to dead paths.
* Its dependencies were never installed here, so it could not run at all.
* The tool icons are now drawn from a design system (`logo/iconset.py`), which
  makes transcription the wrong method regardless of whether it ran.

`orig/` held the 243 source fixtures for that pipeline. Every one of them is
byte-identical to a blob that still exists in git history, so they were removed
from tracking (kept on disk, listed in `.gitignore`) rather than kept in the
repository. To restore any of them:

```bash
git log --all --format=%H -- app/Resources/images/Plane.png | head -1
git show <that-commit>:app/Resources/images/Plane.png > /tmp/Plane.png
```

## Files

| File | Purpose |
|---|---|
| `masters/` | the recovered 1024 px originals — the source of truth |
| `palette.py` | colours sampled from the artwork |
| `geometry.py` | measured geometry as named constants |
| `measure.py` | reproduces those measurements from the masters |
| `marks.py` | vector construction of both marks |
| `export_cloudviewer.py` | renders the CloudViewer size ladder |
| `particles.json`, `cv_dots.json` | measured point fields, verbatim |
