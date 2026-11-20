# Tool icon status

## What this file used to say, and why it was wrong

An earlier revision claimed a "design system" built 272 tool icons from a
shared kit. That is what happened, and it was a mistake worth recording so it is
not repeated.

The measured facts:

| | Count |
|---|---:|
| Icons that already existed as **valid vector art** | **197** |
| Icons that existed only as raster | 86 |
| Icons that were **redrawn from scratch** | **272** |

Every one of those 197 SVG icons was already correct, already sharp, and already
familiar to the people who use this application. Replacing them with newly drawn
marks meant:

* Measured IoU against the shipped artwork: **272 of 283 fell below 0.70**. The
  new drawings were not refinements; they were different pictures.
* Familiarity was destroyed. An icon's whole job is that a returning user
  recognises it without reading the tooltip. A prettier drawing nobody has seen
  before performs that job worse than a plain one they already know.

The lesson generalises: **"looks better" is not the objective function for an
icon.** Recognition is. A redraw has to clear a much higher bar than "I can draw
this more cleanly", and usually it should not be attempted at all.

## Current state

All artwork is restored to what shipped. Measured against `HEAD`, at each icon's
native size:

| Fidelity | Icons |
|---|---:|
| identical (IoU ≥ 0.99) | 200 |
| close (0.95–0.99) | 3 |
| acceptable (0.85–0.95) | 0 |
| below 0.85 | **0** |

203 vector icons, average fidelity **0.9996**.

## The only conversion that is justified

Raster icons that are already flat art can be *traced faithfully* — same shapes,
same proportions, no interpretation. `logo/vectorize.py` does this and reports
its own fidelity score; anything under the gate is refused rather than shipped.

Raster icons that are **not** flat art must stay raster. Measured on this
repository, tracing them produced both artefacts and extreme bloat:

| Icon | Source PNG | Traced SVG | Ratio | Why it fails |
|---|---:|---:|---:|---|
| `decimate.png` | 231 KB | 7,257 KB | 31× | soft shading, many tones |
| `meshSmooth.png` | 253 KB | 5,604 KB | 22× | same |
| `poisson.png` | 261 KB | 4,450 KB | 17× | same |
| `donate.png` | 13 KB | 107 KB | 8× | gradient button; traced version shows banding |

The rule that came out of this: **trace only when the source has few flat
colours**. Above roughly 100 distinct colours the result is a banded, oversized
imitation of the original, which is worse than the PNG in every respect that
matters.

## What is vector and what is raster

| Kind | Form | Reason |
|---|---|---|
| UI glyphs (toolbar, tree, panel, menus) | SVG | scale to any display density |
| Flat primitives (cube, sphere, cylinder, icons with few colours) | SVG | trace is lossless |
| Painted illustrations (decimate, poisson, meshSmooth, sensors, donate) | **PNG** | not flat art; tracing bloats and bands |
| Product logos (`images/icon/*.png`) | PNG ladder | optically tuned per size |

## Editing icons

There is no batch regeneration, on purpose. Edit the SVG in place, or add one
through `logo/vectorize.py` when the source is genuinely flat.

```bash
python3 scripts/icon_hd/icons.py verify   # qrc + XML + raster-in-svg + platforms
python3 scripts/icon_hd/icons.py build    # inventory: vector vs raster per folder
python3 scripts/icon_hd/icons.py audit    # before/after review page with code evidence
python3 scripts/icon_hd/icons.py inventory --out /tmp/icon-audit  # resource and reference report
python3 scripts/icon_hd/icons.py verify --qt  # offscreen app and qPCL resource loading
```

Use Python 3.10 or newer. `inventory` uses the standard library and Git.
`verify --qt` requires a C++ compiler, Qt 5 Widgets and SVG development packages,
`pkg-config` and a matching `rcc`. Use `--rcc /path/to/rcc` for a custom Qt installation.
Completed migration, restoration and batch-review scripts are retired.
The committed resources and provenance manifest are the sources for current verification.

`verify` is the gate that matters. It checks that every reference in every
`.qrc`, Windows `.rc`, CMake file and `.desktop` entry resolves, so a deletion on
one platform cannot silently break packaging on another.

## Determining what an icon should depict

Never from its filename. Three names in this repository are actively misleading
and were mapped by reading the code that loads them:

| File | Name suggests | Actually bound to |
|---|---|---|
| `pqInvert.svg` | invert values | `actionFlipMeshTriangles` — flip triangle winding |
| `pqClearSort.svg` | clear sorting | `actionClearSelection` — clear the selection |
| `dbContainerSymbol3.svg` | a group/container | "Export current polyline to the main DB" |
| `add.svg` | create something | `actionImportSFFromFile` — import a scalar field |
| `orthoSections.svg` | orthographic sections | `traceModeButton` — trace structural orientation |

`audit.py` extracts this evidence automatically from `.ui` action blocks, C++
`mIConList` registrations, and the selection controller's `make()` helper, and
shows it beside each icon in the review page.

## Constraints

* No `<text>` in a finished icon — letterforms are paths, or the glyph renders in
  whatever font the host Qt build happens to have.
* No gradients in UI glyphs — they collapse to mud at 18 px.
* Product logos are not redrawn; see [`logo/README.md`](logo/README.md) for the
  six vectorisation routes measured against the ACloudViewer mark.
* `QTreeView` branch connectors keep exact pixel coordinates; Qt stretches them
  through `border-image` at whatever row height it needs.

## FilterToolBar migration into `svg/filter/`

The nine `FilterToolBar` actions (`MainWindow.ui`) are all filter operations, so
their artwork now lives in `app/Resources/images/svg/filter/` under the folder's
`filter*` naming convention. Ledger and comparison sheets:
`~/ACloudViewer-icon-reviews/filter-toolbar-migration/`.

| Action | Old path | New path | Operation |
|---|---|---|---|
| `actionClipFilter` | `pq/pqClip.svg` | `filter/filterClip.svg` | restore ParaView original |
| `actionSliceFilter` | `pq/pqSlice.svg` | `filter/filterSlice.svg` | restore ParaView original |
| `actionThresholdFilter` | `pq/pqThreshold.svg` | `filter/filterThreshold.svg` | restore ParaView original |
| `actionIsoSurfaceFilter` | `pq/pqIsosurface.svg` | `filter/filterIsoSurface.svg` | restore ParaView original |
| `actionDecimateFilter` | `mesh/meshDecimate.svg` | `filter/filterDecimate.svg` | move (bytes preserved) |
| `actionSmoothFilter` | `filter/filterSmoothVTK.svg` | unchanged | keep |
| `actionProbeFilter` | `pq/pqProbeLocation.svg` | `filter/filterProbeLocation.svg` | move (bytes preserved) |
| `actionGlyphFilter` | `pq/pqGlyph.svg` | `filter/filterGlyph.svg` | move (bytes preserved) |
| `actionStreamlineFilter` | `pq/pqStreamTracer.svg` | `filter/filterStreamTracer.svg` | move (bytes preserved) |

The first four custom redraws were rejected because they neither matched the
operation nor referenced the original ParaView icons (a floating plane read as
slice, a 2-D colour bar read as a scalar ramp, a green ball read as a sphere
primitive). Applying the recognition lesson above, they were restored byte-exact
from the upstream ParaView SVGs rather than redrawn again; the restored bytes are
pinned in `pq_original_sources.json` (`dir: filter`, ids 21–24, plus the relocated
probe original id 17). The resource tests check the manifest and canonical QRC
paths. Edit current artwork at its registered path.

The five restored originals (`filterClip`, `filterSlice`, `filterThreshold`,
`filterIsoSurface`, `filterProbeLocation`) additionally carry an approved
intrinsic preview size `width="128" height="128"` on the root, with
`viewBox="0 0 64 64"` and geometry unchanged. A controlled Qt A/B showed the
default renderer/image/pixmap/label move 64→128 while requested toolbar
16/24/32 px normal/disabled pixmaps stay pixel-identical. The manifest records
this as `default_size: [128, 128]` with `sha256` = installed bytes and
`source_sha256` = upstream geometry pin; `test_pq_smooth_vectors.py` verifies the
installed pin, the declared root size, and byte-exact geometry after normalising
the added root attributes (rows whose upstream already carries width/height keep
the raw pin).
