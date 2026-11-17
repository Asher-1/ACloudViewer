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
```

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
