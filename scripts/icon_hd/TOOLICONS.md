# Tool icon system

The product logos live in [`logo/`](logo/README.md) and are a separate problem:
those are painted artwork and were refined, not redrawn. This file covers the
**tool icons** -- everything the UI draws as a glyph.

## Why a system instead of 272 drawings

The original icons were imported from at least three sources and never
reconciled. Measured before the rewrite:

| Family | Canvas sizes in use | Colours per icon |
|---|---|---|
| `svg/ecv` (35 toolbar) | 8 distinct (13, 16, 18, 21, 24, 25, 32, 48) | median 58, max 319 |
| `svg/pq` (68 panel) | **29 distinct** (13.5 … 64) | median 0, max 224 |
| `svg/misc` (18) | 13.5 to 64 | varies |
| root `*.png` (59) | 16 to 512, all raster | — |

Twenty-nine canvas sizes for one toolbar is not a style, it is a merge accident.
At the size these are actually drawn -- the toolbar sets `iconWidth = 18` in
[`app/ecvUIManager.cpp`](../../../app/ecvUIManager.cpp) -- a 319-colour gradient
collapses into a smear. The colour count was never detail; it was noise inherited
from the source artwork.

Tracing was measured first and rejected: 13 of 24 flat icons failed the fidelity
gate, because fine strokes and baked-in drop shadows do not survive contour
extraction. The icons are therefore **rebuilt from a shared kit**, not traced.

## The kit

[`logo/iconset.py`](logo/iconset.py) holds the tokens and the primitives. Every
icon in every family is composed from them, which is what makes 272 drawings look
like one family.

| Token | Value | Derived from |
|---|---|---|
| canvas | 24 × 24 viewBox | 18 px render scales by 3/4, landing strokes on whole device pixels |
| stroke | 2.0 units | 1.5 px at 18 px — the thinnest width that stays legible |
| caps / joins | round | square caps break up below 2 px |
| ink | `#2C3E50` | one dark slate, recoloured by theme rather than shipping light/dark pairs |
| accent | `#2AA8DC` | used at most twice, to mark "the thing that changes" |
| detail floor | 3 units | 2.25 px at 18 px; anything finer is invisible, so it is not drawn |

Primitives: `line`, `poly`, `rect`, `circle`, `dot`, `arrow`, `plus`, `minus`,
`cloud`, `cube`, `grid3`, `funnel`, `bars`, `bell`, `magnifier`, `trash`,
`sphere_points`, `terminal`.

## Modules

Each module is one family, and each function is one icon with a docstring that
states what the tool does and what failed before.

| Module | Icons | Family |
|---|---:|---|
| [`logo/toolicons.py`](logo/toolicons.py) | 35 | main toolbar (`svg/ecv`) |
| [`logo/treeicons.py`](logo/treeicons.py) | 59 | DB tree, type, view, small actions |
| [`logo/pqicons.py`](logo/pqicons.py) | 37 | panel toolbar (`svg/pq`), core set |
| [`logo/pqvariants.py`](logo/pqvariants.py) | 32 | panel toolbar, variant families |
| [`logo/rooticons.py`](logo/rooticons.py) | 59 | application menus, primitives, window controls |
| [`logo/miscicons.py`](logo/miscicons.py) | 23 | `svg/misc` + `svg/pcl` |
| [`logo/slicons.py`](logo/slicons.py) | 14 | filters and dialogs |
| [`logo/fonticons.py`](logo/fonticons.py) | 13 | font properties, field operators |
| **total** | **272** | |

## The three rules that actually decide the drawings

**1. Say what the tool does to the data, not what the object is.**
`ecvFilterByValue` is a funnel with one survivor kept. `ecvSor` is a cloud where
the outliers have been struck out. `ecvMerge` is two clouds becoming one. The
accent marks the thing that changes, which is how the eye finds the verb at 18 px.

**2. Variants share a base and differ by one badge.**
`pqSurfaceSelection*` is six icons: {cell, point} × {plain, interactive,
data-interactive}. They are built from one surface patch plus one badge (pointer,
pointer-with-bars, probe ring). The user reads the base once and only has to tell
the badge apart. Drawing six separate pictures would look like six families and
would give the eye six chances to disagree.

**3. Legibility is measured, not judged.**
Every icon is rendered at 18 px and its ink coverage checked. Anything over ~45%
reads as a blob. Measured failures that drove redraws:

| Icon | Before | Cause | Fix |
|---|---:|---|---|
| `ecvClippingBox` | 48.1% | nested outlines merged | four corner brackets |
| `ecvSamplePoints` | 53.4% | 9 dots on a visible grid | 3 filled + 1 hollow, no grid |
| `pqInvert` | 48.1% | filled half-square | half-filled disc |
| `pqRectilinearGrid` | 46.3% | 3 interior lines per axis | 2 per axis |
| `pqCalculator` | 56.8% | six keys, each outline + fill | four dots + one rule |

## Reproducing the checks

```python
import sys, io
sys.path.insert(0, "scripts/icon_hd")
from logo import pqicons
import cairosvg, numpy as np
from PIL import Image

for name, build in pqicons.REGISTRY.items():
    png = cairosvg.svg2png(bytestring=build().encode(),
                           output_width=18, output_height=18)
    ink = (np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3] > 80).mean()
    if ink > 0.45:
        print(f"DENSE {name} {ink:.3f}")
```

`rcc` is the other gate. After switching a family to SVG:

```bash
cd app && rcc --list MainWindow.qrc     # must exit 0 with no warnings
```

## Constraints that are not negotiable

- **No `<text>` in a finished icon.** Letterforms are paths. A live glyph renders
  in whatever font the host Qt build happens to have, which is not a font we ship.
- **No gradients.** They collapse to mud at 18 px; see `pqRamp24`, which is four
  flat steps for exactly this reason.
- **No framing plate around a product logo.** Marks float on transparency.
- **The product logo is not redrawn.** See [`logo/README.md`](logo/README.md) for
  the six vectorisation routes that were measured and rejected.
- **`QTreeView` branch connectors keep their original geometry.** Qt draws them
  through `border-image` at whatever row height it needs, so `branch-end.svg` and
  `vline.svg` are coordinate-for-coordinate copies of the 81×58 rasters, not
  redesigned.

## What is intentionally still raster

`app/Resources/images/icon/*.png` — the product logos, at their tuned size ladder.
`app/Resources/images/frame.png` — test fixture, not a UI icon.

Everything else under `app/Resources/images` is vector.
