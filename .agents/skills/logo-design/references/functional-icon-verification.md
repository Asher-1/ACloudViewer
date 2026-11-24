# Functional icon verification and ACloudViewer cases

Use these procedures alongside [Functional toolbar icons and Qt resource closure](<functional-toolbar-icons.md>) and [Toolbar icon restyling](<toolbar-icon-restyling.md>). Run project entry points from the repository root and skill scripts from the resolved skill directory. Treat the historical paths, modes and counts below as scope checks for those cases and use the current checkout as evidence. Read the existing scripts before running state-changing commands. Mark missing entry points, arguments or checks TODO when they cannot be resolved on the current branch.

## 1. Original lettering and fixed IDs

1. Assign permanent IDs before drawing. Record ID, action and source evidence, original/previous/candidate/installed paths and hashes, original text and confidence, added text and reason, old/new effective URLs, owner, disposition, and separate automated/Qt/visual/approval results. Keep explicitly approved IDs and out-of-scope files protected by hashes. Archive actual art before editing. For a rename or move, give every review version an explicit file mapping to the same ID.
2. For every original, look for live text, glyph outlines, compound paths, stored glyph references, and text inside embedded bitmaps. Render native and high zoom, including both backgrounds where useful. OCR is supporting evidence. Compare its output with the original art and real action. Record uncertain text separately with the source and reason. Absence of `<text>` only rules out live text in that search.
3. For the 22-item PCL redraw, check the actual originals for `MLS`, `SIFT`, `SOR` and template registration `(R,t)`. Preserve verified spelling and punctuation as path glyphs and check letterforms and counters at actual size. Resolve runtime font dependencies in new work. Where the original has no text, add an abbreviation only when useful to distinguish the confirmed operation and record why.
4. Use `01–22` consistently in the PCL overview, each group sheet and the itemised ledger. Group sheets contain actual original, prior and new artwork, plus native 16/24/32/48 px and dark-background previews. If a version cannot be recovered, label the gap. For batch 9 items `06–10`, resolve relocated/renamed voxel and octree source images via the ledger, label each action, and generate repeatable comparisons under an authorised repository-external review directory. Hash or snapshot the tracked source assets before and after the review generator to show no new source changes. Do not modify a source icon to improve a comparison layout.

## 2. Ownership, consumer closure and idempotence

1. Inventory physical assets across the requested tree, root leftovers and relevant plugin directories. Parse QRC as XML. Resolve source relative to each QRC and compute the effective `:/` URL using prefix and decoded alias or default source name. Record missing sources, duplicates and collisions independently. List SVG/PNG content types, singleton directories and backed-up source or review-only art.
2. Trace runtime consumers across C++/headers, `.ui`, stylesheets, metadata and optional/plugin paths. Parse XML entities in UI and resource entries. Resolve adjacent C++ string literals split across lines, concatenation, format interpolation and lookup helpers. Inspect build lists, scripts, generators and manifests, including untracked files where relevant. Classify registrations, actual runtime consumers, generation dependencies, tool/doc-only hits and unresolved dynamic references separately. Verify intentional protected exceptions by path/hash and reason. Archive only individual authorised assets with affirmative coverage of the relevant consumers and source-retention checks. Keep unresolved and pending-redraw items separate from confirmed unused ones.
3. Keep qPCL algorithms and their QRC within the plugin and verify the QRC uses plugin-local physical sources. App-owned actions use valid app URLs in app-only loading. Plugin-provided actions get their icon after plugin resources register and are omitted or handled appropriately without the plugin. Check both registrations and actual app UI behaviour. A shared-resource exception requires explicit ownership and lifecycle verification.
4. Use the evidenced repository entries in this order for the qPCL workflow. Confirm their actual launcher paths/imports on this branch before invoking them. The names below are entry points, and the arguments shown are the evidenced arguments. Supply the real installed tessdata path.
   ```text
   check_pcl_original_text --data <tessdata-directory>
   audit_svg_inventory
   migrate_qpcl_icons --install --review
   migrate_qpcl_icons --install
   finalize_icon_ownership
   ```
   Capture the first install and the replay inventories, consumer rewrites, protection hashes, moved-only equality and archived originals. The replay must add no moves, rewrites or retirements and must preserve review provenance. Confirm closure after finalisation. Report unresolved dynamic references and assets awaiting redraw explicitly.

## 3. ParaView migration, restoration and zoom repair

1. **Preserve the evidence.** Keep a separate readable ParaView colour-sample set outside the runtime asset tree. Measure the approved samples, with roles for warm grey strokes, shades and accents, spacing and optical weights. Keep source-path and version provenance. In ACloudViewer inspect `scripts/icon_hd/logo/paraview_style.py` if present. Verify that sample paths remain readable after moves. Hash protected/approved assets before work. A concurrently changed hash requires investigation and explicit reporting. Preserve the concurrent work and avoid claiming the original comparison passed.
2. **Migrate mixed sources using an explicit manifest.** In the historical 19-item app SVG-root move to `pq`, the requested split is 14 bitmap sources reconstructed as real vector art without embedded bitmaps and 5 qualified SVGs copied byte-for-byte. Record each old/new path and URL, source archive and transformation. Update QRC, UI, code and generators. Run the current branch's old ParaView root migration entry point twice and verify that a second run does not re-add references, restore legacy artwork or discard archived provenance. TODO resolve the exact old migration launcher from the checkout. Report app/plugin independent Qt checks and affected UI compilation.
3. **Restore upstream design when requested.** Check the selected ParaView checkout with `git -C "$paraview_dir" ls-files` and inspect each candidate's path, actual format, version and semantic correspondence. For axes and every other action, reuse a verified SVG byte-for-byte when it represents the same operation and is suitable. Treat similarly named/drawn SVGs as candidates until equivalence is confirmed. For PNG-only items reconstruct vector geometry from the actual PNG, keeping composition, colours, detail and verified lettering. Distinguish exact SVG reuse, PNG-source vectors, missing or mismatched assets and protected approvals in the ledger. In the historical restore case PQ-15/16/19 were approved and protected. Report differences from the source, including unresolved contour, gradients or lettering fit. Cairo/Pillow compositing on both backgrounds and optional colour/alpha differences help quantify differences.
4. **Audit zoom defects across the whole `pq` scope.** Flag dense small pixel rectangles or fragments, axis-aligned staircase contours even with no `<image>`, embedded rasters, `crispEdges` and filter dependencies. Inspect paths and directly render original high zoom. Use the actual UI operation/consumer and target Qt renderer to distinguish damage, Qt filter risks and legitimate geometry. Complexity thresholds are triage. Keep an itemised status with path or render evidence and leave sound vectors unchanged. Archive the true defective originals separately and replace confirmed defects at the same physical path and URL. Reconstruct smooth, meaningful geometry and preserve functional details. Check real 16/24/32/48 px and directly rendered 384/1024 px originals and installed versions under the same IDs, plus actual Qt high-zoom normal/disabled output at the resource URLs.
5. **Prove repair survives the legacy path.** Save SHA-256 for canonical repairs, archived originals, protected assets and the URL map. Inspect the existing legacy migration and keep its installed bytes sourced from the canonical repairs wherever the mapping overlaps. Run from repository root:
   ```bash
   python3 -m scripts.icon_hd.repair_pq_pixel_icons
   python3 -m scripts.icon_hd.repair_pq_pixel_icons
   ```
   Then replay the actual old ParaView migration entry point. After each invocation verify the installed hashes against the canonical repairs, archive hashes against true originals, non-target hashes and the unchanged URL map. Use the actual Qt resource probe after replay and compare URL bytes to the installed canonical file. Detect stale generator output even if XML and URL registration still succeed.
6. **Handle Cairo-pass / Qt-fail on converted SVGs.** Reproduce with the configured Qt on the failing URL and an isolated canonical source, checking bytes to localise the defect. For a conversion serialiser that emitted prefixed SVG namespace tags, the evidenced minimal fix is `ET.register_namespace('', 'http://www.w3.org/2000/svg')` before serialising conversion output. Compare path data and visual attributes before/after and scope changes to the generated format. Preserve original upstream SVG bytes and approval hashes. For the recorded restoration workflow, run the following once the project entry points have been confirmed. Verify images, resource results, high zoom, affected UI and any remaining fit differences.
   ```bash
   python3 -m scripts.icon_hd.restore_pq_paraview_originals
   python3 -m scripts.icon_hd.review_pq_original_restore
   python3 -m scripts.icon_hd.verify_qt_resources
   python3 -m scripts.icon_hd.verify_pq_zoom
   python3 -m unittest discover -s scripts/icon_hd/tests -q
   ```
   If a full application or UI operation check is required, use the configured project's documented agent workflow. Record any unperformed test alongside the offscreen probe results.

## 4. View-orientation rename example

1. In ACloudViewer, start at the current `.ui` widget/action names, tooltips and effective resource URLs. Trace Qt signals and slot/lambda connections in the main window and dialog implementations, then `CC_VIEW_ORIENTATION` in `core/include/CVConst.h`, forwarding to the active view and camera implementation in `libs/VtkEngine/Visualization/vtkGLView.cpp` and `libs/CV_db/src/ecvDisplayTools.cpp`. Start with the following searches where `rg` is available. If a presumed implementation path fails, find files by UI object/slot name and use `rg --files` rather than guessing a second path.
   ```bash
   rg -n 'ecvView(Iso[12]|[XYZ](pos|neg))|actionSetView|CC_(ISO|TOP|BOTTOM|FRONT|BACK|LEFT|RIGHT)' app core libs
   rg -n 'setView\(|SetView\(|set(Top|Bottom|Front|Back|Left|Right)View' app libs
   ```
2. Build eight evidence rows before drawing. Record action/control and tooltip, signal and enum, camera side/vector in the implemented coordinate convention, resulting visible faces and highlighted face, source lettering, and old/new effective URL with path/line evidence. Distinguish camera **position relative to the target** from **look direction toward the target**. Record both where ambiguity matters. Use a shared reference cube for the six orthographic actions and highlight the face the camera looks at. Draw isometric projections from their actual camera sides and preserve the verified `FRONT`/`BACK` as path glyphs for the original redraw. Any later approved lettering revision belongs to that review's ledger.
3. This table gives the eight-row rename scaffold and the directions available from the recorded example. Fill TODOs from the active code and observed operation before drawing, and recompute the full URLs from the current QRC. The example URLs share `:/Resources/images/svg/view/` under the observed `/` prefix. Rows below show the suffix with `.svg`; the actual ledger contains the full old and new URI and all affected action/consumer IDs.

   | ID | Historic action | Camera side and visible faces | Old → new URL suffix |
   |---|---|---|---|
   | 01 | Front isometric | −X, −Y, +Z. Front, right and top per enum comment | `ecvViewIso1.svg` → `mainViewIso1.svg` |
   | 02 | Back isometric | +X, +Y, +Z. Back, left and top per enum comment | `ecvViewIso2.svg` → `mainViewIso2.svg` |
   | 03 | Right view | +X. Record the actual face from the implementation and check it in the UI | `ecvViewXneg.svg` → `mainViewXneg.svg` |
   | 04 | Left view | −X. Record the actual face from the implementation and check it in the UI | `ecvViewXpos.svg` → `mainViewXpos.svg` |
   | 05 | TODO trace action | TODO trace camera and faces | `ecvViewYneg.svg` → `mainViewYneg.svg` |
   | 06 | TODO trace action | TODO trace camera and faces | `ecvViewYpos.svg` → `mainViewYpos.svg` |
   | 07 | TODO trace action | TODO trace camera and faces | `ecvViewZneg.svg` → `mainViewZneg.svg` |
   | 08 | TODO trace action | TODO trace camera and faces | `ecvViewZpos.svg` → `mainViewZpos.svg` |

4. Close the resource rename in `app/MainWindow.qrc`, `app/ui_templates/MainWindow.ui`, `annotationsDlg.ui`, `filterToolDlg.ui`, and `clippingBoxDlg.ui` as applicable, plus any additional static or generated consumers. Inspect iconset fields, fallback URLs, and dialog button wiring individually. Search for stale stems and dynamic patterns, not just full strings. Preserve other assets by hash and archive originals outside runtime packaging. Compile the affected `.ui` files with the selected `uic`. Verify eight new URLs, normal/disabled pixmaps, canonical bytes and negative old URLs from loaded bundles. Publish the eight-row ledger and native-size pair comparisons.

## Comparison and raster checks

1. Keep original/previous/new versions side by side on light and dark boards with stable IDs and action labels. Use separate input/output directories per phase so renamed files and same basenames cannot collide. Render directly at each requested pixel size. Use larger source renders for high-zoom boards and label them with the actual size. Keep comparison generation read-only with respect to source assets, and save the command or generator with the ledger for repeatability.
   ```bash
   python3 "$skill_dir/scripts/render_png.py" --which
   for size in 16 24 32 48; do
     python3 "$skill_dir/scripts/render_png.py" "$phase_dir"/*.svg --size "$size" --out-dir "$review_dir/$phase-$size"
   done
   ```
   Add 64 px when useful and 384/1024 px for the zoom repair. Repeat the loop with each phase. Compose the two backgrounds from the native-size transparent renders. Use archived raster originals as actual inputs for bitmap cases, and label amplified original pixels accurately.
2. For a set expected to be distinct, the following Pillow check consumes one output directory for one phase and size. Supply the expected row count and explicitly inspect any duplicate pairs or expected shared glyphs. Also parse the SVGs, validate root dimensions and `viewBox`, and inspect geometry and letter paths against the ledger.
   ```bash
   python3 - "$render_dir" "$size" "$expected_count" <<'PY'
   from pathlib import Path
   import sys
   from PIL import Image

   directory, expected_size, expected_count = sys.argv[1:]
   size = int(expected_size)
   files = sorted(Path(directory).glob('*.png'))
   assert len(files) == int(expected_count), (directory, len(files))
   seen = {}
   duplicate_pairs = []
   for path in files:
       with Image.open(path) as source:
           rgba = source.convert('RGBA')
       assert rgba.size == (size, size), (path.name, rgba.size)
       alpha = rgba.getchannel('A')
       bbox = alpha.getbbox()
       assert bbox, (path.name, 'empty render')
       edge = [alpha.getpixel((x, y)) for y in range(size) for x in range(size)
               if x in (0, size - 1) or y in (0, size - 1)]
       partial = sum(0 < value < 255 for value in edge)
       opaque = edge.count(255)
       bbox_touch = bbox[0] == 0 or bbox[1] == 0 or bbox[2] == size or bbox[3] == size
       key = bytes(channel for r, g, b, a in rgba.getdata()
                   for channel in ((r, g, b, a) if a else (0, 0, 0, 0)))
       if key in seen:
           duplicate_pairs.append((seen[key], path.name))
       else:
           seen[key] = path.name
       print(path.name, bbox, 'bbox touches:', bbox_touch,
             'partial-alpha edge:', partial, 'opaque edge:', opaque)
   assert not duplicate_pairs, duplicate_pairs
   PY
   ```
   Treat partial alpha at the edge as a screening result. Assess possible clipping against path/stroke extents, SVG viewport, enlarged/padded comparison or renderer differences. Empty opaque contact alone cannot establish complete geometry. Check PNG/board decode, expected dimensions and row completeness. An optional alpha character map can help compare silhouettes and locate anomalies.
3. If the image tool says the runtime lacks image input, stop repeating that call. Keep the generated comparison and any standalone preview available for the reviewer. State which XML, Pillow and Qt checks ran and explicitly mark visual review pending. Successful rendering, file decode, alpha checks and Qt loading are measurable results. Record direct visual inspection and user aesthetic approval independently.

## Qt resource probe

1. Use the active build directory's `CMakeCache.txt` and generated build settings. For ACloudViewer inspect `USE_QT6`, `Qt5_DIR`/`Qt6_DIR`, and the appropriate Core/tools package entries, then resolve the same configured distribution for `rcc`, `uic`, headers, Qt Widgets/Svg and runtime. Example cache query:
   ```bash
   rg -n '^(USE_QT6|Qt[56]_DIR|Qt[56]Core_DIR):' "$build_dir/CMakeCache.txt"
   "$rcc" --version
   "$uic" --version
   ```
   Record tool paths, versions, platform and actual runtime/library selection. One historical build pointed to `/opt/qt515` while PATH tools reported Qt 5.12.8. That path is an example and the active cache governs. On Linux use the selected runtime search configuration, including `LD_LIBRARY_PATH` if needed, and `QT_QPA_PLATFORM=offscreen` for runtime probes.
2. Read and reuse [the project Qt probe](<../../../../scripts/icon_hd/verify_qt_resources.py>). If its default toolchain resolution differs from the active build, use its inspected probe implementation with the selected compiler, Qt linkage and runtime, or configure the entry point in the manner supported on this branch. Build app and plugin resources separately via the selected `rcc`. For the observed app layout, the offscreen fixture uses `"$rcc" --binary app/MainWindow.qrc -o "$review_dir/app.rcc"`. Use `"$uic" "$ui_path" -o "$generated_header"` on every affected UI with outputs outside source assets. Record missing tooling or unverified flags as TODOs.
3. Load each bundle independently and then in combination when relevant. Verify QFile/QResource bytes (or SHA-256) for each expected effective URL against the corresponding physical file. Render non-null icons at at least two actual requested sizes in `QIcon::Normal` and `QIcon::Disabled` and record size/state counts. For the view rename probe, 16/24/32 px × both states × eight URLs is 48 checks. Exercise real Qt high zoom for zoom defects. Test known obsolete URLs absent when they should be retired, and ensure app-only loading succeeds without qPCL. Report missing links and failed loads by URL. Offscreen resource coverage and actual GUI interaction coverage are distinct fields.

## Third-party Qt framework integration gate

This gate applies when UI/icon work brings in or ports a library; a successful
standalone gallery does not establish a correct repository dependency boundary.

1. **Read and reuse the existing owners first.** Read the repository's
   [reuse/ownership contract](<../../acloudviewer-aicore-plugin/references/reuse-contract.md>)
   for the owner-first decision process, and inspect the active
   [dependency registration](<../../../../3rdparty/find_dependencies.cmake>),
   [libpng recipe](<../../../../3rdparty/libpng/libpng.cmake>) and
   [OpenImageIO recipe](<../../../../3rdparty/openimageio/openimageio.cmake>) before
   writing an integration. Apply the ownership principle to the actual UI/build
   owner; this does not make a pure UI dependency an AICore task.
2. **Use the repository dependency flow.** Add or extend
   `3rdparty/<dep>/<dep>.cmake` with `ExternalProject_Add`: upstream URL, explicit
   version, SHA-256, `DOWNLOAD_DIR` under
   `CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR`, `UPDATE_COMMAND ""`, declared
   dependencies/byproducts, install location and the existing target/export
   convention. Keep a fixed default; an explicit upgrade changes the pin, digest
   and reviewed patch/test evidence together. Do not follow floating HEAD, commit
   source tarballs into the repository, add a second private download cache, or
   replace this flow with an app-local unpack/add_subdirectory mechanism.
3. **Patch at the dependency owner.** Store Qt5/Qt6 source compatibility as an
   ordered, reviewable Git patch set invoked by CMake `PATCH_COMMAND` after
   extraction. Use `git apply --no-index --check`, then `git apply --no-index`
   with `WORKING_DIRECTORY` set to the exact extracted source root, so enclosing
   repository discovery cannot redirect or silently skip the patch. Inspect the
   [existing patch runner](<../../../../3rdparty/openimageio/patches/apply_openimageio_patch.cmake>)
   for that failure mode, not as an unqualified template: **do not copy its
   completed-build-stamp early return**. After apply, verify every required
   change against the current patch identity and source content. A reused tree
   must prove the complete expected patched state (reverse-check where valid,
   or complete content/hash/semantic checks when ordered patches overlap).
   Neither a single marker, patcher exit 0, nor "already built" proves this.
   If neither cleanly applicable nor fully verified as already applied, fail
   loudly. Missing patch files, partial application and source drift are errors,
   not "already applied or not applicable" success. Do not use shell `patch`,
   hand-edit production extracted trees, or promise automatic conflict-free
   compatibility with arbitrary upstream versions.
4. **Close platform/build propagation.** Consume the Qt major/targets already
   chosen by `cmake/CMakeExternalLibs.cmake` (`QT_PREFIX`, `QT_VERSION_MAJOR`,
   `Qt::` aliases); do not add another `if(USE_QT6)`/outer `find_package` in a
   dependency recipe. A separate ExternalProject CMake process still discovers
   Qt using the exact parent-selected package paths; that is not permission to
   independently select a different major. Reuse shared ExternalProject CMake
   arguments and verify compiler/launcher/toolchain, C++ standard, Qt major and
   required Qt component package paths. Preserve parent macOS architecture,
   sysroot and deployment target; escape list-valued arguments across sub-builds.
   Declare actual platform static library names/prefixes/suffixes, Debug postfix
   and MSVC CRT/configuration compatibility, PIC when embedding archives into
   shared libraries, generated export headers, install/export targets and final
   dependency closure. Do not let a dependency override parent policy or multiply
   nested build/autogen concurrency beyond the authorized total job budget.
5. **Prove replay and runtime separately.** Test fresh extraction, verified
   already-patched reuse, changed patch/pin and deliberate conflict rejection,
   then the actual consumer build/link and installed resource/license closure.
   For a Qt style, test real dense widgets, popup/input/focus/disabled/local
   palette, animations and teardown, Native/Custom round trips and configured
   DPI; protect approved DB Tree/artwork and plugin-local resources. Preserve
   explicitly requested semantic action emphasis (such as the shared sample/test-
   data button) in both Native and Custom; consolidate it in the existing helper
   instead of deleting it as generic chrome. Keep all Qt
   diagnostics. Record Qt version, OS, compiler, DPR and whether the test was
   offscreen or a real desktop. A Qt5 run does not prove Qt6, Windows/macOS or
   multi-monitor behavior; same-source adaptation is not cross-major ABI
   compatibility. Pure UI changes need their relevant UI/build gates, not
   unrelated AICore numerical inference tests.
   For translucent menus/popups, also compare native windows created before and
   after style installation. Qt5 `QMenu::popup` can create a depth24, alpha0
   surface before polish; a late transparent attribute does not recreate that
   native visual. Widget-grab PNG alpha is not proof of native transparency.
   Check actual surface/depth and compositor separately; opaque fallback must
   agree in paint, metrics, shadow margins and placement. A background-only
   desktop capture is failed capture evidence, not successful visual acceptance.

## Qt default dimensions and installation

1. **Inventory before changing defaults.** Identify direct `QImage`, `QPixmap`, `QSvgRenderer` consumers, `QLabel` or other size-driven widgets, rich text, stylesheet tree branches and dynamic consumers in addition to toolbar `QIcon` requests. Protect accepted or byte-identical upstream artwork unless the user has scoped it in. Inventory proposed, already-at-size, size-sensitive and protected files with paths, consumers, reasons and hashes. Keep numeric results from prior branches as evidence only for those branches.
2. **Prepare a controlled pair.** In a review location copy the same SVG and change only root `width` and `height` (e.g. 24 to 150 or 128). Keep viewBox, geometry and encoding otherwise exact. Verify an attribute-level diff and use equivalent QRC URL registrations with isolated fixture bundles or processes so one variant cannot shadow the other. Record the exact sources, build toolchain and probe invocation with reproducible inputs.
3. **Run the project Qt A/B.** Use a widget-capable offscreen Qt probe with the [Qt resource setup](<#qt-resource-probe>). For both copies record validity, `QSvgRenderer::defaultSize()`, default `QImage` size and `sizeInBytes()`, default `QPixmap` size, and `QLabel::sizeHint()` after `setPixmap()`. Use the same loading path as each consumer. Request `QIcon` pixmaps at 16/24/32 px (and other real sizes) in normal/disabled and other affected states. Compare dimensions, device pixel ratio, pixel bytes in the same image format, and relevant button `sizeHint()` at 1× and 2×. Report intrinsic-size-sensitive call paths separately. For example, one 24→150 probe produced 24×24 versus 150×150 for default renderer/image/pixmap and label hint, 2,304 versus 90,000 image bytes, with pixel-identical requested 16 px normal/disabled icons. Re-run for the current artwork, Qt version and platform before claiming equality.
4. **Install approved artwork.** Match the approved candidate hashes to the review ledger for all requested icons. If the user requests 128×128 installed defaults, set just those root attributes on the approved artwork as needed, retaining drawing geometry, `viewBox` and the agreed URLs. Write each row with approved hash, installed hash, root dimensions, path, URL, and test result. Recompile all affected `.ui` files and retest isolated app/plugin and combined resource loading, canonical bytes and applicable old-URL negatives from the installed state.
5. **Measure three isolated states with a common scope.** A is the pre-size snapshot, B is the same artwork with intrinsic dimensions changed only, and C is B plus the approved artwork replacement. Exclude unrelated changes from each snapshot. Use equivalent QRCs, scope, ordering, output format, compression flags and the same `rcc` version and invocation for every state. For example, run the app `rcc --binary` fixture from each preserved state's root with the agreed identical flags and different output paths. Include affected plugin bundles if applicable. Record full commands, per-state totals of actual source SVG bytes (`len(path.read_bytes())`) and `.rcc` bytes, then report B−A and C−B for both measures. Verify the expected differences via manifests and hashes, and retain the three states for review. Mark any unavailable baseline TODO and avoid estimating its measured delta.
6. **Bound the results to the measurement.** Report successful pixel and widget comparisons with their sizes, Qt version, DPRs and call paths. Report startup timing only from a controlled application A/B using the documented runtime workflow. Report final installer/package size only from actual comparable packages made with the same release inputs, settings and platform. For missing measurements state startup and/or installer size unmeasured, even when raw SVG and RCC deltas are known. Publish the ledger, resource/UI logs and measurement table.
