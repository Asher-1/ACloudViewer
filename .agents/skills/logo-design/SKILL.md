---
name: logo-design
description: It designs, redesigns, critiques and compares logos, wordmarks, monograms, app icons and favicons, and restyles or audits functional icons when users ask for logo concepts, branding, ParaView-style toolbar icons, before-and-after comparisons, Qt QRC migration, SVG zoom repair, or default SVG size and loading checks.
---

# Logo Design

A brand logo identifies an organisation in a distinctive, relevant form that works at 16 px, at signage scale and in one colour. A functional icon communicates an operation and distinguishes it from neighbouring actions. Functional diagrams, familiar conventions and meaningful colour are welcome in that mode. Reply in the user's language with real files and concise rationale.

## Steps

1. **Route the request.** Use the mode table below. For functional work, read [Functional icons and resource closure](<references/functional-toolbar-icons.md>) and [Toolbar icon restyling](<references/toolbar-icon-restyling.md>), then follow the applicable workflow in [Functional icon verification and ACloudViewer cases](<references/functional-icon-verification.md>). Use the brand phases only for brand design and redesign.
2. **Establish scope and evidence.** For brands, write the brief. For functional icons, trace each UI action to implementation, capture original/previous art and original lettering, inventory physical files and effective QRC URLs, trace static and dynamic consumers, and hash approved and out-of-scope assets. Assign stable IDs before drawing or moving anything. For camera views, trace action → enum → camera transform and record the camera side, visible and highlighted faces, and old/new URL per row.
3. **Build within the agreed scope.** Preserve verified functional lettering as paths. Add abbreviations only when useful. Keep plugin assets and QRC local, resolve app behaviour with the plugin absent, and close all changed consumers and generators. For ParaView restore or migration, verify upstream semantics and keep qualified SVGs byte-identical. Reconstruct bitmap-only designs as true vectors. For zoom defects, replace only confirmed targets in place and make earlier migrations preserve the canonical repaired bytes on rerun.
4. **Render and verify.** Resolve bundled scripts against the skill directory. Run `python3 "$skill_dir/scripts/render_png.py" --which` to discover renderers and render functional comparisons directly at 16/24/32/48 px on light and dark backgrounds. Use the [raster checks](<references/functional-icon-verification.md#comparison-and-raster-checks>) when image input is unavailable. For Qt work, use the configured build's `rcc`, `uic` and runtime, probe actual resource loading and compile affected UI. For intrinsic dimension changes, test direct decode and requested-size widget paths before installation. Measure size-only and artwork changes against separate baselines when requested. See [Qt default dimensions and installation](<references/functional-icon-verification.md#qt-default-dimensions-and-installation>).
5. **Deliver the relevant checkpoint.** Brand design/redesign stops at concept presentation before the kit. Functional work delivers fixed-ID overview, per-item/group comparisons, actual-size previews and a ledger. Rework shows original/previous/new. Mark pending visual review explicitly when image inspection is unavailable. If the user has already approved a specific functional draft and requested installation, verify its identity and install that artwork within scope.

## Modes

| User asks for | Mode | Start with |
|---|---|---|
| A new logo | Design | Phase 1 below |
| Feedback on a logo | Critique | [Critique](<references/critique.md>) |
| Modernisation or replacement | Redesign | [Redesign](<references/redesign.md>), then brand phases |
| Guidelines, sub-brands, patterns or motion | System | [Identity system](<references/identity-system.md>) |
| Favicon, app icon or variants from an existing mark | Assets | `scripts/export_variants.py` |
| Toolbar/button graphics, ParaView styling, plugin relocation, QRC audits, zoom defects or Qt SVG sizing | Functional icons | [Functional procedure](<references/functional-toolbar-icons.md>) and [Verification recipes](<references/functional-icon-verification.md>) |

**Fast track.** For a brand request with little information or a tight deadline, ask up to five questions in one message using [Discovery brief](<references/discovery-brief.md>) §2, or state assumptions and proceed to three concepts.

**Brand checkpoint.** Show the tested concepts and offer the full kit after Phase 6. Wait for the direction and kit approval before Phase 7 unless the user explicitly asks for the full delivery without a check-in. If the user cannot reply, stop at the checkpoint and describe the possible kit.

## Bundled tools

Run these Python 3 scripts from the skill directory or use `python3 "$skill_dir/scripts/<script>.py"` from the project. Rendering backends may need separate installation. On Windows use `python` or `py -3` if `python3` is the Store stub. Repository scripts in the functional references run from the repository root.

| Script | Use |
|---|---|
| `concept_sheet.py` | Concept overview with large mark, lockup, actual 64/32/16 px previews, name, idea and recommendation |
| `search_library.py` | Search by `--type`, `--technique`, `--geometry`, `--subject`, `--industry`, `--color`, `--mood` or `--query`. Use `--summary` for conventions and `--format json` for records. Fetch artwork needed for study from official sites or public icon CDNs |
| `svg_audit.py` | Check live text, rasters, filters, colour counts, gradients, strokes, near-miss angles, tiny details, centring and complexity |
| `preview_sheet.py` | HTML size ladder, 16/32 px pixel tests, backgrounds, one-colour, squint blur, mirror/rotate, favicon/app-icon/header/card contexts and competitor shelf tests |
| `presentation_board.py` | Use the [presentation spec](<templates/presentation-spec.example.json>) for brief, rationale, six industry-specific mockups, comparison and recommendation. `--png-dir` exports slides and `--list-mockups` lists contexts |
| `render_png.py` | Exact-size transparent SVG renders, Chrome HTML screenshots and favicon.ico. Use `--which` to list backends |
| `export_variants.py` | Black/white/brand-mono/square/favicon/app-icon variants. `--png` requests sizes. `--web-icons` makes favicon.ico, PNGs, webmanifest and head snippet. `--favicon-source` chooses the simplified art |
| `build_catalog.py` | Maintainers rebuild the library catalog |

```bash
python3 "$skill_dir/scripts/render_png.py" concept-a.svg concept-b.svg --out-dir renders --size 512
```

Inspect the PNGs with an image-capable tool or inspect HTML in a browser. The renderer can select cairosvg, rsvg-convert, Inkscape, Chrome/Chromium or macOS Quick Look. Direct `qlmanage` can crop non-square art, shrink based on `width`/`height` and lose transparency. If rendering is unavailable, keep geometry simple and disclose the missing check. Check every bash exit-code marker and investigate failures.

## Brand design workflow

### Phase 1 — Discovery and brief

Learn exact spelling, activity, audience, 3–5 adjectives, competitors, retained equity, colour/use constraints and decision-maker. Translate adjectives into visual cues and write the brief with assumptions. [Discovery brief](<references/discovery-brief.md>) §5.

### Phase 2 — Research and strategy

```bash
python3 "$skill_dir/scripts/search_library.py" --industry <closest> --summary
python3 "$skill_dir/scripts/search_library.py" --industry <closest> --format json
```

The library skews toward tech. In other sectors also search by subject/query and use category knowledge. List clichés to avoid or transform. Map name, offering, adjectives and promise to nouns, metaphors and opposites, then circle intersections. Explore at least two mark types. [Discovery brief](<references/discovery-brief.md>) §6 · [Mark types](<references/mark-types.md>) §12 · [Library guide](<references/library-guide.md>).

### Phase 3 — Concepts

Write 8–12 one-sentence concepts with an ownable twist across multiple mark types. Score idea clarity, distinction, simplicity, relevance and small-size strength. Build the three strongest distinct options. Show the longlist only if it helps steering. Search same-subject or same-technique marks and study construction in 3–5 exemplary records with `--format json`. Use fetched artwork for study.

```bash
python3 "$skill_dir/scripts/search_library.py" --subject <thing>
```

### Phase 4 — SVG construction

Describe primitives, grid, radii and angles before drawing. Start solid black on white. Use `viewBox="0 0 256 256"` for symbols and height 256 for lockups, few anchors, arcs for circles, exact 0/15/30/45/60/90° angles, consistent radii and strokes, and real holes with `fill-rule="evenodd"`. Outline final lettering and flag any exploration with live text. Preserve meaningful revisions such as `concept-a-v1.svg` and `concept-a-v2.svg`. [SVG construction](<references/svg-construction.md>).

### Phase 5 — Test and refine at least twice

```bash
python3 "$skill_dir/scripts/svg_audit.py" concept-a.svg concept-b.svg concept-c.svg
python3 "$skill_dir/scripts/preview_sheet.py" concept-a.svg concept-b.svg concept-c.svg --refs-industry <industry> -o preview.html
python3 "$skill_dir/scripts/preview_sheet.py" yours.svg --refs <fetched-reference-SVGs> -o shelf.html
```

Inspect and iterate. Open gaps or thicken strokes at 16–24 px, or draw a small cut. Check optical overshoot (~1–3%), rounded-shape bone effects, horizontal stroke weight, optical centring, reversal weight, footprint and balance. Mirror, rotate 180° and squint to expose alternate readings. Check modified letters and junctions for misreads, notches, slivers, lumps and accidental ink traps. Compare at equal size with 3–4 exemplary marks. Push a literal product drawing toward a distinctive brand idea. [Visual techniques](<references/visual-techniques.md>) · [Testing checklist](<references/testing-checklist.md>).

### Phase 6 — Present and pause

```bash
python3 "$skill_dir/scripts/concept_sheet.py" a-symbol.svg b-symbol.svg c-symbol.svg --lockups a-lockup.svg b-lockup.svg c-lockup.svg \
    --names "Name A" "Name B" "Name C" --notes "One-line idea A" "…" "…" --recommend 1 --greyscale -o concepts.png
```

Inspect and show the overview. Lead in greyscale with an optional small colour hint for the recommendation. Give each concept its name, mark type, one-sentence idea and up to three brief reasons grounded in the brief and competition. Recommend one direction with honest risks. Ask whether the user wants the full kit with colour and one-colour/reversed versions, lockups, the small-size cut, favicon/app/web icons, industry board and one-page usage guide. For revisions, return to Phases 4–5 and reshow the overview.

### Phase 7 — Approved kit

1. Refine geometry, optical corrections, small-size and thinned reverse cuts.
2. Choose ideally 1–2 distinctive and reproducible colours. Specify HEX/RGB/CMYK/Pantone and check accessibility, greyscale and monochrome. [Colour](<references/color.md>).
3. Customise lettering and optical spacing, use at most two type families, and lock proportions for horizontal, stacked, symbol-only and wordmark-only versions. [Typography](<references/typography.md>).
4. Build the industry-specific presentation with `presentation_board.py` and the [presentation spec](<templates/presentation-spec.example.json>). Set `industry` or explicit `mockups`, such as cups/bags for a café and README/terminal for developer tools. Use `symbol_on_tile`, `lockup_on_dark` and optional `tile_color` to preserve multicolour art on dark contexts, and `--png-dir slides` for individual images. [Presentation and delivery](<references/presentation-delivery.md>).
5. Export master-derived variants.
   ```bash
   python3 "$skill_dir/scripts/export_variants.py" final-symbol.svg --title "Brand logo" --mono "#HEX" --icon-bg "#HEX" --web-icons --favicon-source final-symbol-small.svg
   python3 "$skill_dir/scripts/export_variants.py" final-horizontal.svg --title "Brand logo" --only black white mono --mono "#HEX" --png 1200
   ```
6. Deliver SVG masters and PDF/AI/EPS when available, rationale, tests and open items. Specify clear space relative to a logo element, minimum sizes, colour codes, backgrounds and misuse. Expand to an identity system for larger brands. [Guidelines template](<templates/brand-guidelines-template.md>) · [Process](<references/process.md>) §7 · [Identity system](<references/identity-system.md>).

## Pitfalls

- Generic stock initials, category clichés, literal clip art, near-miss geometry, lumpy curves, inconsistent weights, hairlines, sub-1/48 details, closing gaps and weak forms propped up by effects undermine brand marks. Final brand masters should avoid live text, embedded rasters, filters and masks. Use library logos for study and check resemblance, font licences and trademark clearance separately. A professional database and reverse-image search can help. [Principles](<references/principles.md>).
- Similar solids failed to distinguish mesh tools, Poisson reconstruction, convex hull, voxel downsampling and octree. Compare actual operations as neighbours at toolbar scale. For views, the `pos`/`neg` suffix is a legacy identifier. Resolve semantics from the action, enum and camera implementation.
- QRC registration and an empty literal grep are weak evidence for use or disuse. Check alias expansion, real and generated consumers, metadata, optional paths and protected source artwork before retirement. Keep qPCL local and exercise app-only loading.
- PNG files with SVG suffixes, embedded rasters and pixel-tile traces can look plausible at toolbar size. Inspect original geometry and high zoom, and run the configured Qt loader. Cairo success and successful SVG parsing cover only those respective checks.
- A faint alpha fringe at a canvas edge needs separate reporting from opaque contact. Keep the generated boards available if the model cannot open them and mark visual review pending.

## Verification

1. Confirm the chosen mode, scope and each expected deliverable. Inspect scripts' exit codes, fix issues and rerun. For brand testing use [Testing checklist](<references/testing-checklist.md>) and [Process](<references/process.md>) §7.
2. Cross-check fixed IDs, original lettering evidence, archive paths, hashes, move mappings, canonical bytes, effective URLs and changed consumers. Keep moved-only, redrawn, unstyled, uncertain/dynamic/source and proven-unused entries distinguishable. Use the [verification matrix](<references/functional-toolbar-icons.md>#verification-matrix) and the [ACloudViewer cases](<references/functional-icon-verification.md>).
3. Compare actual resource bytes with canonical files. Record independent app/plugin and combined probes as applicable, plugin-absent behaviour, obsolete URL negative checks and affected UI compilation. For an in-place repair, the original URL must load the new bytes after replaying migration. For size changes record actual direct-decode and requested-size behaviour and the measured scope of performance and package comparisons.
4. Report automated geometry/raster and Qt results, visual inspection and user aesthetic approval as separate outcomes. Publish the overview and relevant detailed files with explicit TODOs for missing evidence or unperformed tests.
