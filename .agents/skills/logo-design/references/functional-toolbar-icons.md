# Functional toolbar icons and Qt resource closure

Use for operation/button icons, ParaView-style restyling, plugin icon ownership, and resource migration or cleanup. Pair this procedure with [Toolbar icon restyling](<toolbar-icon-restyling.md>) for visual measurements, approved-sample metric bands, text preservation and overlap composition. This guide adds semantic and engineering gates; it does not replace that visual guidance or the project's plugin-development contract.

## 1. Establish scope and inventory before edits

1. Determine the working directory with `pwd`, inspect `git status --short`, and read applicable project/plugin rules. Do not infer the working directory from a harness installation path.
2. Separate authorised redraws, authorised moves, untouched approved assets, and audit-only work. Preserve original and latest rejected versions before writing anything. Store review artifacts at a user-approved writable location; historical home-directory report paths are not mandatory or automatically writable.
3. Inventory every physical file in the requested SVG tree, images-root leftovers and relevant plugin resource directories, not just files already in an optimisation catalog. Count formats by actual content; keep PNGs as PNGs until genuinely redrawn as vectors.
4. Parse each relevant QRC: owning bundle, prefix, alias (or default relative name), resolved source path and resulting runtime URL. QRC source paths resolve relative to the QRC, not the repository root. Detect missing sources, conflicting aliases and duplicate registrations separately from identical content.
5. Trace consumers in C++/headers, UI XML, stylesheets, metadata, build files, scripts, generators and manifests. Follow variable/concatenated paths, resource helper functions and optional/plugin-only paths. QRC registration alone does not prove use, and lack of literal matches does not prove non-use.
6. Link each action to its constructor/action wiring, tooltip/menu text and implementation. Record code/UI path and line evidence in the ledger. If semantics remain ambiguous, mark TODO rather than invent an operation.

Initial non-mutating commands for the evidenced ACloudViewer layout:

```bash
pwd
git status --short
find app/Resources/images -type f -print
find plugins/core/Standard/qPCL -type f -print
git grep -n -E 'Resources/images/svg/pcl|PclAlgorithms/icons|dbContainerSymbol|frame[.]png' -- .
git grep -n -E 'QIcon|QPixmap|QResource|setIcon|iconset' -- app plugins
```

These are starting searches, not a complete audit. `git grep` excludes untracked files; inspect newly generated/untracked files too. Exit code 1 from grep means no literal match, not proof of safe deletion. Inspect every bash exit marker and resolve unexpected failures. Never replace a failed command with an unverified success claim.

## 2. Build a semantic contract per icon

Before drawing, write: **action → input → transformation/output → visual cue → confusing neighbour → code evidence**. A filename or algorithm name alone is insufficient.

For the five rejected mesh/spatial icons, use the following candidate distinctions only after checking current code/UI meanings:

| Action | Functional visual candidate | Avoid confusion with |
|---|---|---|
| Mesh tools/menu | Vertices connected into triangular faces; mesh connectivity rather than a solid polyhedron | Generic cube/solid or a single reconstruction operation |
| Poisson reconstruction | Oriented/discrete points, with normals where legible, becoming a closed surface | Smoothing an already existing mesh |
| Convex hull | A taut outer boundary enclosing irregular points, without internal radial triangulation | Mesh connectivity/triangulation |
| Voxel downsampling | Equal spatial cells containing several input points and one retained representative per cell | Spatial hierarchy or simple volume rendering |
| Octree | A parent space subdividing into eight children, with a clear hierarchy cue | Uniform voxel sampling |

These are useful design hypotheses, not aesthetic acceptance records. Simplify at 16 px without removing the discriminating cue; an exact structural count does not ensure users recognise the action. Show neighbouring operations together so collisions are visible. Functional icons may use familiar diagrams and meaningful colour; neither a shared palette nor decorative geometry replaces semantics.

For ParaView-style requests, derive colour roles, warm-grey strokes, shading, contrast and metric bands from current approved samples, not guessed universal constants. Preserve previously approved artwork unless explicitly authorised to change it.

## 3. Decide resource ownership and loading behaviour

Plugin-owned functionality should be maintained in the plugin's local asset directory and QRC. In the evidenced case, qPCL owns PCL artwork: inspect its existing local conventions and [qPCL resource file](<../../../../plugins/core/Standard/qPCL/qPCL.qrc>) in the project context rather than adopting the rejected app-side consolidation as canonical. The precise destination subdirectory must be chosen from current project conventions; do not assume a proposed directory already exists.

For every application consumer of a plugin icon, choose and document a lifecycle-safe arrangement:

- If the action is plugin-provided, let the plugin provide/register the action and icon after plugin resources are loaded. The app must hide/omit or appropriately disable that feature when the plugin is absent, without requesting a missing URL.
- If the application independently owns the operation, use a legitimate app-owned generic icon or an explicitly designed shared-resource contract permitted by project rules. Do not quietly turn a plugin-local resource into an app build dependency.
- If artwork is genuinely shared, review ownership/build/loading explicitly rather than masking it with `../../../app/...` or an app QRC path into a plugin directory. Report the agreed exception and its lifecycle tests.

Moving files into a plugin while continuing to package them via app's QRC is not sufficient. Nor is leaving always-created app UI pointing at a URL only available after plugin loading. Verify application-only, plugin-only, combined, and plugin-not-loaded behaviour. A full GUI test, when required, must use the project's documented agent toolchain, not guessed binary options.

## 4. Preserve content and perform planned edits

Create a before-state manifest containing source path, SHA-256, format, ownership, aliases and intended disposition. Compute content hashes, e.g. with Python's `hashlib.sha256(path.read_bytes()).hexdigest()`.

Use explicit mapping records for moves:

```json
{
  "old_path": "<observed source>",
  "new_path": "<approved destination>",
  "old_urls": ["<observed runtime URL>"],
  "new_urls": ["<planned runtime URL>"],
  "operation": "move-only",
  "before_sha256": "<digest>",
  "after_sha256": "<digest>"
}
```

- Protected unmoved files must retain their original hashes.
- Authorised move-only files must exist at the mapped destination with identical hashes. Retire the original only after destination and consumer checks pass.
- Redrawn files must have separate archived original/previous art and a redraw ledger event. If a redraw also moves, record both rather than exempting it from protection checks.
- Treat collisions by name separately from duplicates by bytes. Do not discard visually or functionally distinct variants just because names resemble each other.
- Update all known consumers, UI iconsets/resource includes, QRC registrations, build rules and generators in the same planned change. Update any migration/protection scripts so regeneration cannot recreate the obsolete layout.

Evidence included a protection-check repair adding old→new byte-preserving relocation mappings. Preserve that safeguard: do not weaken the checker to ignore missing files or every path under a moved directory.

## 5. Audit the full requested tree conservatively

Keep separate columns for **style status**, **usage status**, **ownership** and **operation**; they are not mutually exclusive. At minimum report:

- Actual redraw/restyling, with design source/provenance and comparison.
- Move-only, byte-preserved (not “optimised”).
- Existing approved/styled, untouched.
- Still unstyled or not yet visually reviewed.
- Source artwork, generated assets or dynamic references needing verification.
- Broken/missing source, invalid alias or stale consumer URL.
- Confirmed unused and safely retired, with evidence.

Audit root leftovers as well as SVG subdirectories. In the evidence, `frame.png` and `dbContainerSymbol.svg` were specifically flagged; check current purpose and ownership, not historical assumptions or blanket catalog exclusions. Unrelated files named `frame.png` in tests do not establish a UI dependency. QRC registration of a DB symbol, on the other hand, must be traced to its actual DB-tree consumer.

For every directory containing one icon, select a semantically appropriate destination among existing categories; record the reason and update all aliases/consumers/generators. Respect plugin boundaries: flattening an app category does not justify moving plugin assets back into app. Do not arbitrarily move an icon merely to meet a count rule if semantics remain unresolved; mark it as an explicit outstanding decision.

Before deleting an “unused” asset, require all applicable checks: registrations and direct consumers, dynamic naming schemes, generated/build manifests, optional plugins/features, metadata, and whether it is an intentionally retained source/master. Preserve uncertain items and explain why. Quarantine/archive outside runtime packaging where appropriate and authorised. Never delete based on one grep with no matches.

## 6. Render and deliver actual comparisons

### Intrinsic preview size and genuine vector quality

- For ACloudViewer toolbar SVGs whose requested preview is **128×128**, set only the root `width="128" height="128"`; keep the existing `viewBox` (for example `0 0 24 24`), paths, transforms, stroke widths and resource URLs unchanged. Root dimensions control the default preview; the viewBox defines drawing coordinates. Do not scale path data just to enlarge a preview.
- Generate the 128×128 PNG directly from the final SVG, never by enlarging a 24px PNG. Also compare fixed-size 16/24/32/48/128 renders before and after: a size-only change must have identical requested-size pixels. For Qt, check `QSvgRenderer::defaultSize()`/default image loading separately from explicit-size `QIcon` rendering.
- Audit consumers before applying this to more files. `QImage`, `QPixmap`, rich-text images and Qt tree decorations may use intrinsic dimensions; do not mass-enlarge them or overwrite byte-exact upstream/approved assets without explicit scope. Update the owning generator and canonical hash ledger so regeneration does not restore the old size.
- A `.svg` extension, a successful XML/Qt load, or a high native-pixel similarity score does **not** prove smooth scaling. Pixel-by-pixel rects, stair-step contour traces, and upsample-then-trace pipelines can preserve the original mosaic at every zoom. Changing 24 to 128 cannot repair bad geometry.
- For ParaView-derived icons, search the provided ParaView checkout for the matching original SVG first and preserve its actual design. Do not substitute a similarly named but visually different asset. If only PNG exists and tracing still produces mosaics, measure the source silhouette, color regions, endpoints and overlaps, then reconstruct the same design using a few polygons, Bézier curves, circles, dash patterns and genuine gradients. Do not invent a different metaphor. Retain original textual content as paths and protect approved icons by hash.
- Keep fixed review IDs in overview, every comparison group and the ledger. Rework comparisons must show **original → rejected trace → reconstructed SVG**, plus true 128px preview, actual toolbar sizes and direct 512/1024px SVG renders. Separate automated rendering checks, actual visual inspection and user approval; do not claim visual inspection if the active model cannot accept images.

1. Produce one itemised image row per actual redraw. Default columns are **original → new**; rejected-version rework requires **original → previous → new**. Archive the actual prior artwork, not a reconstruction. If a version is unavailable, label the gap and request/recover it rather than fabricating a comparison.
2. Include true-size 16/24/32/48 px renders and light/dark backgrounds, plus larger review views. Enlarged thumbnails must not be labelled actual size. Include normal/disabled Qt icon states when affected.
3. Render the final installed SVGs, not merely generator output that may differ from shipped assets. Inspect sheets yourself; revise confusing cues, clipping and small-size contrast.
4. Use the bundled renderer with explicit sizes; resolve the script path against the skill directory:

```bash
python3 <skill-dir>/scripts/render_png.py <final.svg> --out-dir <review-dir>/16 --size 16
python3 <skill-dir>/scripts/render_png.py <final.svg> --out-dir <review-dir>/24 --size 24
python3 <skill-dir>/scripts/render_png.py <final.svg> --out-dir <review-dir>/32 --size 32
python3 <skill-dir>/scripts/render_png.py <final.svg> --out-dir <review-dir>/48 --size 48
```

Compose the before/after and dark-background sheets with an available inspected project review utility or a purpose-written image composition script. The generic brand concept sheet is not a substitute for the required three-version/multi-size layout. TODO if no suitable renderer/compositor is available; do not claim that text descriptions satisfy visual delivery.

Deliver PCL comparisons for PCL redraws themselves. A five-mesh-icon comparison does not cover the plugin set. Show an overview inline and link detailed pages, ledger and evidence. State moved-only counts separately from redraw counts. Await user aesthetic approval.

## 7. Run independent Qt/resource checks

Static source scans and SVG parse tests are necessary but insufficient. Build/run isolated probes with the same Qt major and SVG/image support as the application:

1. Compile app and plugin QRC bundles separately using the configured project's Qt resource tooling. Compile affected UI files with its UI compiler. Do not guess tool paths, compiler flags, Qt package names or build target names.
2. Each isolated probe must read every expected URL via `QFile`/`QResource` and compare the bytes or SHA-256 to the canonical physical file. Confirm effective prefix/alias URLs, not filenames alone.
3. Load SVGs with `QSvgRenderer`, PNGs with `QImage`, and icon paths through `QIcon`; verify valid/nonempty rendering at target sizes and normal/disabled states. A non-null `QIcon` alone may not prove decodability; request actual pixmaps and inspect output.
4. Check retired URLs are absent, except explicitly documented compatibility aliases with intentional expiry. Search production sources/manifests for stale references; historical review reports may legitimately contain old paths and should not be treated as live consumers.
5. Run app-only and plugin-only probes in separate processes so the other bundle cannot mask missing resources. Also test the combined application/plugin registration case for collisions and plugin-absent application behaviour for unintended runtime dependency.
6. Record exact commands, Qt version, assertions, totals and exit results. If tooling is unavailable, mark the affected rows “static only / not run”; leave the audit incomplete.

**TODO — environment-specific commands:** discover and record the current project's Qt resource compiler, UI compiler, include/link flags, generated bundle targets and probe commands. The evidence supports independent probes but does not supply a portable exact build command. Do not invent one.

The historical probe successfully checked aliases against canonical bytes, old URLs and affected UI compilation for one migration. It did not establish that the app-side ownership decision was correct or that the later full-tree cleanup had completed.

## Verification matrix

| Gate | Required evidence | Not a substitute |
|---|---|---|
| Meaning | Current code/UI references plus distinct operation cue per item | Filename, palette or geometric similarity |
| Protection | Original manifest, protected hashes, explicit authorised move mapping | Ignoring missing paths |
| Ownership | Plugin-local files/QRC and documented app/plugin-absent behaviour | Cross-directory packaging that hides dependency |
| Static closure | QRC XML resolved paths, aliases, all known consumers and generators updated | One successful grep |
| Runtime closure | Isolated app/plugin probes, canonical byte equality, real image/icon renders, old URLs absent, affected UI compilation | Combined bundle test only |
| Review | Actual old/new or old/previous/new images, 16/24/32/48 px and dark previews | Text descriptions or structural unit tests |
| Full-tree audit | Every requested asset classified, counts reconciled, unknowns listed | Sampling or a completed sub-batch |
| Acceptance | Explicit user feedback/approval | Passing tests or assistant preference |

## Ledger and final report format

Maintain machine-readable rows plus a readable summary. Suggested fields:

```text
id | action | code_ui_evidence | original_path | previous_snapshot | final_path
owner | qrc_bundle | old_urls | new_urls | static_consumers | dynamic_consumers
before_sha256 | after_sha256 | operation(move/redraw/keep/retire)
style_status | usage_status | deletion_or_merge_reason | comparison_image
static_check | qt_isolated_check | ui_compile | render_review | user_acceptance | TODO
```

Reconcile input files with retained/moved/redrawn/retired outcomes without double-counting move+redraw events. Include explicit lists of unstyled assets, broken references, dynamic/source uncertainty and confirmed-unused deletions. Link review artifacts and code evidence with precise locations when known. End with separate verdicts for resource/structure, visual inspection and user approval, plus remaining limitations. “Complete” is only appropriate when the requested scope has actually been covered; otherwise identify exactly which checks and assets remain.
