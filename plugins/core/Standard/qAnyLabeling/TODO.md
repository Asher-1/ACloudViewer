# qAnyLabeling parity TODO

Upstream synchronization (2026-10-17): ACloudViewer
`origin/users/dahailu/v3.9.5` at `aff48cd6a` is merged into the current branch.
The earlier local OCR experiments were withdrawn at synchronization. The
subsequent user-requested direction/v5/v6 extension now builds on that merged
OCR runtime through the public task API; qOCR has no local workflow changes.
See the final "OCR model and point-cloud camera implementation" section for
the current scope, evidence and unpublished-model limits. Older OCR notes
remain historical and must be checked against the current implementation.

Synchronization validation: qAnyLabeling, qOCR and JSON-RPC build successfully;
all 28 registered AnyLabeling tests pass with no skips. The LoMa and LightGlue
bridges also pass on Vulkan after the final rebuild. The complete 40-patch
ggml chain replays from the pinned archive and passes the idempotence check.
Selected OCR recognition/detection/layout and RAM scenarios pass on CPU,
CUDA and Vulkan (four scenarios per backend), as recorded in
`build_app/Testing/upstream-sync-final-model-gate-{cpu,cuda,vulkan}.json`.
The 18 RelateAnything F32/Q8 public-interface and stability probes pass, but
independent upstream accuracy/performance validation remains incomplete:
the existing modified `RelateAnything-GGML/cpp_ggml/src/full_model.cpp` fails
the reference snapshot guard. Those external edits are preserved. Source
identity checks and the final 28-test run are recorded in
`build_app/Testing/upstream-sync-source-audit.json` and
`build_app/Testing/upstream-sync-final-qany-tests.log`.

## Current upstream follow-up (2026-10-17)

The local upstream is now v4.1.3, commit
`745f6c355f94cddb06f6965b3a4118f82033e935`. The older revision below remains
the provenance of the copied icons and visual references, not a claim that the
latest upstream is fully covered.

- [x] Image cuboid editing: front corners and four edge centers, visible rear
  endpoints and depth center, left/right/back face drags, minimum-size
  constraints, out-of-image control and whole-object movement, locked shapes,
  undo/redo, and `cuboid3d` metadata. Front and back faces are drawn separately,
  with dashed rear edges and an orientation edge in both canvas and export.
  The focused test covers both depth directions, zoom and stale image changes;
  the plugin test exercises real-window drag/save/undo/redo. The upstream
  comparison script generates transient references from the actual Python
  `Canvas`/`Shape` methods and records revision and source hashes.
- [x] Point-cloud workspace entry now uses the upstream `Ctrl+6` shortcut.
- [x] New upstream point-cloud **3D cuboid task mode** core: cuboid sidecars,
  orthogonal editing viewports, detection-mode integration, semantic/instance
  assignment, undo/redo and real-window interaction are implemented. The
  generated probe matches the upstream geometry over 167 cases / 1,920
  sequential edits (maximum point delta `1.78e-14`), and the offscreen window
  test covers task switching, top/front/side views, selection and save/reload.
  The camera source now indexes images by the upstream basename rule, loads an
  optional `calibration.json`, matches the current point-cloud frame, and
  displays projected points/cuboid edges in a camera dock. The calibration,
  depth ordering, near-plane/image clipping and `none`/`opencv5` distortion
  output match actual upstream `camera.py` methods over synthetic clipping and
  KITTI fixtures: 1,018 cuboid segments, maximum coordinate delta `1.14e-13`,
  and identical depth-palette RGB values. The core `PointCloudDatasetExporter` now writes the
  upstream Datumaro, KITTI Raw and Sly Point Cloud ZIP entry contracts,
  including PCD payloads, cuboid metadata and Sly key maps, with cancellation
  and ZIP-integrity regression coverage. The core `PointCloudDatasetImporter`
  now safely parses the same three archive contracts, validates
  frame/class/cuboid records and supports cancellation. The 2026-10-17 follow-up
  adds `preparePointCloudDatasetImport` / `applyPointCloudDatasetImport`: unique
  frame matching, reserved class-ID/name merging, sidecar/config previews,
  grouped detection/segmentation configuration, staged/fsynced writes and
  rollback after cancellation or an external file change. The exporter now
  commits through `QSaveFile`, preserving the old archive on failure/cancellation,
  and PCD preserves both intensity and RGB in little-endian records. The live
  window now exposes all three formats through import/export dialogs, confirms
  the preview, applies annotations/configuration transactionally, rejects stale
  or reentrant workspace edits, and preserves unsaved segmentation labels.
  Detection and segmentation use separate class tables with legacy migration.
  Camera configuration supports multiple sources, previous/next/selection,
  point/cuboid visibility, depth/RGB color, point size and opacity. Main-view
  Z-up basis, fit, pan/orbit/zoom, projection, cuboid focus and oriented ray-box
  selection now follow the actual upstream numeric methods. The live upstream
  window uses CuboidViewport (perspective, bounded scale); the plugin follows
  that default and also offers orthographic projection. Dataset import applies
  annotations to the open local sequence, as upstream does; decoding archived
  PCD payloads is not part of that import contract. Full-window screenshot
  parity remains open; the source uses whole-frame background loading, not
  renderer-tile streaming (see the resource/service follow-up below).
- [x] Verify the v4.1.3 label-file path resolution changes, including opening
  annotation JSON outside the image directory, relative `imagePath`, basename
  fallback, explicit image overrides and embedded-image precedence. JSON files
  can be opened/dropped directly and saved to the same document; saved image
  paths are relative to the destination JSON. Failed decoding leaves the active
  document intact. Eight cases match the actual upstream `LabelFile` methods.
- [x] Shape Manager: four operations, inclusive ranges, output-directory
  routing, one-pixel identity tolerance, locked-shape removal protection,
  bounds/deduplication, cancellation and reload. Eight filesystem-result cases
  match the actual upstream dialog methods. A pre-existing `_delete_` collision
  is reported and preserved instead of overwriting an earlier deleted image.
- [x] `file_list_checkbox_editable` now defaults to false, as upstream does;
  explicit persisted preferences still apply.
- [x] Resolve the previously skipped LoMa test: use the shared pinned
  Sacre-Coeur image pair by default and published F16 detector/DeDoDe-G/B
  matcher models. Missing/corrupt inputs fail instead of silently skipping.
  CPU/CUDA/Vulkan produce 578/581/578 reviewed matches. Replacing an image
  clears accepted matches and invalidates an in-flight result; the CPU stale
  result test also passes. LoMa is an AICore extension, not an upstream model
  port or a substitute for the separate LightGlue matcher.
- [x] Point-cloud camera configuration preview: browse first readable images,
  switch/step through multiple camera sources, debounce directory edits and
  bound thumbnail cache to eight images. Previewing invalid calibration does
  not mutate the active workspace; cancellation preserves its camera sources.
- [x] GPU surface selection now reads point ownership from an RGBA8 framebuffer
  with a depth attachment, using the displayed pixel footprints and physical
  viewport size. Occluded points are excluded, equal-depth coincident points
  are retained, and Through selects hidden depth layers. Fast brush strokes
  include the segment between events. A real OpenGL window tests these paths.
- [x] Main-view cuboid creation: choose the nearest visible point within the
  upstream one-metre ray cylinder, preview a one-metre cube, double-click to
  commit and Esc to cancel all viewports. Click selection waits until release
  so orbit/pan drags do not select objects. The upstream numeric comparison
  now covers 94 view scenarios, 5,208 ray-box queries and 18,216 creation-ray
  queries; maximum creation-center delta is `7.11e-15`.
- [ ] Screenshot-level parity remains open; numeric cuboid parity does not
  establish complete styling or full-workspace interaction parity.

Validation for this follow-up: the plugin, all registered plugin test
targets and the standalone inference runner build successfully. The current
plugin ctest run is **27 PASS, zero failures or skips**, including the
point-cloud camera and camera-dock UI regression, recorded in
`build_app/Testing/qanylabeling-round28-ctest.log` (27 plugin tests).
The real CPU RAM/RelateAnything Run-button workflow and point-cloud window
are included. The actual upstream Python cuboid methods
match the C++ probe over 167 scenarios and 1,920 sequential edits (maximum
point delta `1.78e-14`). The strict repository-wide AICore boundary check
passes. Local evidence: `build_app/Testing/qanylabeling-cuboid-upstream.json`,
`qanylabeling-document-upstream.json`, `qanylabeling-shape-batch-upstream.json`,
`qanylabeling-loma-real-{cpu,cuda,vulkan}.log` and
`qanylabeling-loma-cpu-stale.log` in the same directory.
The point-cloud camera probe is independently compared with the upstream
synthetic clipping and KITTI calibration/cuboid fixtures in
`build_app/Testing/qanylabeling-pointcloud-camera-upstream.json`: three cases,
1,018 cuboid segments, maximum coordinate delta `1.14e-13`, and exact depth
palette RGB values. `test_qanylabeling_pointcloud_dataset` additionally creates and checks
the three upstream ZIP formats and cancellation cleanup.
`test_qanylabeling_pointcloud_dataset_importer` round-trips those archives
through the safe core parser and checks CRC corruption rejection, frame/class
merging, preview immutability, reserved IDs, conflicting destinations, existing
class reuse, external edits and cancellation rollback for both existing/new
sidecar/config files. `compare_pointcloud_dataset_upstream.py` calls the actual
upstream archive readers and import-plan/apply functions; five archive cases,
two JSON changes and class/reserved-ID merging match in
`build_app/Testing/qanylabeling-pointcloud-dataset-upstream.json`. The cases
include per-frame Datumaro ID reassignment, numeric frame names, unsupported
annotation rejection, KITTI optional pose fields and inherited occlusion.
The actual-window test also exercises all three dataset dialogs, confirmation,
cancellation/rollback, stale previews, reentrant workspace edits, separate task
class tables, reserved IDs and preservation of segmentation/unmatched frames.
It checks camera source switching/reordering, invalid configuration and missing
frames, and samples overlay pixels for visibility, RGB/depth, size and opacity.
`qanylabeling-pointcloud-view-round26-upstream.json` compares the actual viewport
fit/basis/matrix, pan/orbit/zoom, cuboid focus and ray-box selection methods,
including portrait windows, scale clamps, poles and large coordinates. Matrix
comparison allows one float32 ULP; ray-box IDs must match exactly.
The real KITTI window capture is `qanylabeling-pointcloud-kitti.png`; this
records the current rendering and does not establish full screenshot parity.
The controlled GL geometry gate is additionally recorded in
`build_app/Testing/qanylabeling-pointcloud-render-upstream-round28.json`: six
Top/Front/Side/perspective framebuffer cases have zero differing pixels.

## Open parity gaps at a glance

Every remaining upstream misalignment, in one table. Status / owner notes:

| # | Area | Gap | Owner / landing | Status |
|---|------|-----|-----------------|--------|
| 1 | OCR interaction | `skip_detection` selected-shape recognition | Public `aicore_ocr_recognize_regions`; update only descriptions, preserving geometry and metadata; explicit Run-button reset, autosave, reviewed state and undo/redo | Implemented for rectangles and polygons with at least three points; source normalization and CPU/CUDA/Vulkan worker/window checks below |
| 2 | OCR interaction | Direction classifier `cls` (`use_angle_cls`, 0/180 flip before recognition) | Public AICore optional classifier/options/typed orientations; task-owned recipes and local converted F32 weights | Implemented for run/recognize and enabled by default; publication and complete dataset/release gates remain open |
| 3 | OCR geometry | Full source det/rec polygon and transcription equality on a shared corpus | Consume the merged upstream OCR implementation; retain source integer corners and exact rectangle classification in the annotation adapter | Six exact source page and selected-polygon comparisons supplement graph gates; corpus/dataset accuracy remains open |
| 4a | OCR evidence | Current model/backend acceptance | Merged public OCR catalog: det-ch, rec-ch, rec-en and CDLA layout, F32/F16/Q8_0. Current common-runner results are 15/15 on each of CPU/CUDA/Vulkan, plus real-window checks on all three | Current evidence below supersedes historical local OCR experiment results; no controlled speedup claim |
| 4 | Region/KIE | Optional layout, PP-OCR KIE annotation format and source v5/v6 recipes | Layout is off by default; preserve real classes/reading order. Entity/link format round trip is implemented; source registers no SER/RE auto model. All six active source OCR presets have local F32 recipes through public OCR | Local workflows implemented; new weights unpublished, dataset/quantization/release gates open |
| 6 | Visual parity | Screenshot-level layout comparison against the in-tree upstream references | Manual fixture pass per panel using `docs/images/*` baselines; controlled GL geometry comparison is separate | TODO — point-cloud GL geometry had exact 6-case framebuffer parity, but full Qt window theme, typography, controls and animation still lack a screenshot gate |
| 7 | Point cloud | Remaining camera panels, shortcut/gesture and full-window rendering details | Cuboid geometry/sidecars/orthogonal editing, task classes, dataset transactions, multi-camera configuration and preview are implemented. GPU RGBA/depth ownership picking now covers occlusion, equal-depth ties, point footprints and continuous brush strokes. Selection projection now caches 16-pixel tiles and reuses them across brush events; 18,216 creation-ray queries match | Partial — background frame loading, bounded prefetch, selection tiles and focus animation are implemented; complete camera-panel and visual/interaction parity remain |
| 9 | External services | Remaining `open_vision` and multi-model source pipelines | Remote Server transport already exists; ChatBot/VQA now share the asynchronous service client described below. Service inference stays remote; local inference stays in AICore | Partial; ChatBot/VQA workflow details and exact remaining model pipelines are still open |
| 10 | Config module | Upstream `configs/auto_labeling/*.yaml` carries per-model interaction defaults (display_name, conf/iou thresholds, input size) that the picker should surface per model | AICore catalog owns model facts (ownership map): extend catalog entries with the interaction-default fields, plugin reads them from the catalog — never a private yaml table in the plugin | facedetect LANDED 2026-09-30 (`aicore_facedetect_model_entry` now carries conf/iou/input_size — buffalo 0.5/0.4/640, yunet 0.6/0.3/160 — and the picker applies the catalog conf when a detector is selected); yolo/ocr entries are the same-shape follow-up |
| 11 | Packaging | Run/.exe/.app installers must carry the plugin resources (translations qm, docs) and rely on catalog remote-download for models; default model per task must match upstream | Existing docker/build scripts + qrc resources; verify installer contents per platform (Linux/Windows/macOS) | Linux generated plugin install rules, stripping and Unicode-path relocation are verified below; complete Linux installer and native Windows/macOS package verification remain open |

The detailed ledger below also retains older open workflow and validation
items. New upstream additions are tracked separately above.

---

Reference: X-AnyLabeling-GGML commit `926017d09b22564a78c55eb2f368c7b20faa0466`.
The dated research/rollout notes below preserve historical plans; the current
public catalogs, implementation and follow-up evidence above take precedence.
The source tree has 205 `anylabeling/configs/auto_labeling/*.yaml` model
configurations with 92 distinct `type` values. A model is available here only when AICore publishes a matching
task contract and a catalog GGUF. A similar output shape alone does not mean
the source model has been ported. Re-check the live AICore catalogs before
closing a row.

Catalog-backed source `type` values whose corresponding workflows are fully
or partly wired are `ram`, `rfdetr`, `rfdetr_seg`, `rmbg`, `scrfd`,
`segment_anything_2`, `segment_anything_3`, `yolo26`, `yolo26_obb`,
`yolo26_pose`, `yolo26_seg`, `yoloe`, `yolov8`, `yolov8_seg`, and `yolow`.
This is a task-family mapping, not a claim that every source checkpoint or
configuration is in the AICore catalog. The remaining 77 source types are
listed below, including composite types that can reuse existing AICore models.

## Source models with no matching AICore catalog

The source families below have missing or partially implemented model coverage.
Existing catalog-backed subsets remain available in the model picker. For
missing models, implement the inference task, GGUF catalog and validation first;
then add the plugin worker and matching interaction. The names below are source
config `type` values or exact source model families.

Current OCR scope (2026-10-17): qOCR keeps its merged workflow; the public OCR
task now also supports the optional direction classifier, all six active source
OCR presets as local F32 recipes, and selected arbitrary polygons as described
in the final section. PP-OCR KIE annotation format is implemented. New-model
publication, quantization and dataset/release gates remain open. The following dated OCR
port/accuracy narrative records historical experiments and is not current
acceptance evidence.

- [ ] OCR and document understanding: `ppocr_v4`, `ppocr_v5`, `ppocr_v6`,
  `doclayout_yolo`, PPOCR KIE. Research verdict (2026-09): upstream's
  PaddleOCR choice is confirmed best-in-class for this use case — 2026
  benchmarks rank the PaddleOCR family first among efficient open-source
  OCR engines (PaddleOCR-VL 0.9B beats 235B VLMs on OmniDocBench;
  PP-OCR leads handwriting/multilingual accuracy per-parameter), and the
  source adapters consume exactly the PP-OCR det/rec/KIE pipeline that
  this plugin's sidecar stream already imports/exports. No GGUF or ggml
  port of PP-OCR exists (community deployments use ONNX Runtime; ONNX
  conversions are not an AICore backend). The port is therefore a new
  `ocr` AICore task plus a first-party Paddle → GGUF converter
  (DBNet-style det CNN + SVTR/CRNN rec + FCE layout/KIE heads; all map
  onto existing ggml conv/transformer ops plus the RMBG custom-op
  pattern for fusion), NOT a substitution with a different engine.
  Acceptance gate: typed results — per-box `aicore_ocr_box`
  (quadrilateral corners + UTF-8 transcription + float score + script),
  layout regions (region class + reading-order index), KIE relation pairs
  (typed index pairs + role labels) — with CER/edit-distance accuracy
  gates against the PaddlePaddle reference on pinned fixtures, and the
  full f32/f16/q8 x CUDA/Vulkan/CPU ladder proving end-to-end parity
  before the plugin picker gains any OCR entry (the PaddleOCR toolbar
  entry stays disabled with its official name until then).
  Rollout plan (2026-09): P0 first-party converter
  `core/AICore/tools/convert_paddle_ocr_gguf.py` plus pinned fixtures
  with PaddlePaddle reference JSON (G0: op-level reference match);
  P1 task `core/AICore/src/tasks/ocr/` + `AICORE_MODEL_OCR` + public
  `aicore/ocr_capi.h` (opaque context, extensible options, three typed
  result arrays, release, last_error, common timings, runtime_shutdown
  delegation) with f32 CPU end-to-end CER equal to the Paddle
  reference (G1); P2 f16/q8 via the shared `aicore_gguf_quantize`
  tool and the full 3x3 backend/quant matrix through validate_all
  (G2: matrix green and `test_catalog_dump_urls --json` exposes every
  row); P3 plugin worker/model-bar enable flipping
  `AnnotationTODO_open_paddleocr` atomically with the picker (G3:
  plugin regression plus headless real-inference evidence).
  Progress (2026-09-28): P0 data plane COMPLETE — the converter
  (`core/AICore/tools/convert_paddle_ocr_gguf.py` + the dependency-free
  `paddle_ocr_extract.py` binary parser, since the paddle runtime
  bindings crash on the conversion host) emitted nine GGUFs (det-ch /
  rec-ch / rec-en x F32/F16/Q8_0) from the real PP-OCRv4 checkpoints,
  with the 6623/95-entry recognition dictionaries embedded as
  `ocr.dictionary` and conv kernels folded to block-aligned 2D for
  Q8_0 (`ocr.quant.reshaped`). All nine are published at the new PP-OCR
  release, pinned in `asset_digests.h`, and exposed through the
  catalog-only `aicore/ocr_capi.h` with `test_catalog_dump_urls` and
  `test_catalog_remote_assets` passing. Progress continued
  (2026-09-28): the Paddle reference channel was repaired on the
  conversion host (paddle.inference Config with IR optimization and
  ONEDNN disabled — the static-graph loader crashes there), and
  `core/AICore/tools/export_ocr_graph.py` now emits the authoritative op
  topology (rec: 323 ops, det: 312 ops — one shared op vocabulary:
  conv2d/depthwise/batch_norm/elementwise/hard_swish/hard_sigmoid/
  pool2d/matmul_v2/transpose2/slice/layer_norm/softmax) plus parity
  fixtures (fixed-seed input -> paddle output) at
  `core/AICore/tests/fixtures/ocr/{rec,det}_ch_ref.npz`. The topology is
  embedded in the released GGUFs (`ocr.graph.ops`), the digests were
  re-pinned, and the inference contract is declared in
  `aicore/ocr_capi.h` (18 exported functions: options/context/run/typed
  results/timings/error/shutdown). A dedicated Standard plugin `qOCR`
  (`PLUGIN_STANDARD_QOCR`) now carries a full workspace dialog
  (`OCRDialog`): bilingual Chinese/English recognition model
  selection over the shared script-agnostic detector, F32/F16/Q8_0
  tier choice, digest-verified downloads through the unified
  ecvModelDownloader pipeline into the task-owned `ocr_models`
  cache (exposed as `aicore_ocr_model_cache_dir`), a one-click
  "Use test data" button that stages the official PaddleOCR demo
  image for the selected language, and a results table. The Run
  entry inside the dialog stays disabled until the gates pass, so
  the workspace is usable for staging without faking inference.
  Loading-chain fixes (2026-09-29): the qrc now embeds info.json
  (the missing entry made the plugin load as a blank name), the
  icon was redesigned (document + scan frame + detection box +
  OCR lettering), the Use-test-data demo URLs were replaced with
  verified-live official assets (docs/images/00006737.jpg for
  Chinese, tests/test_files/textline.png for English), and the
  full official PaddleOCR test-image set (55 images, MANIFEST
  with per-file SHA-256) is staged at
  ~/cloudViewer_data/download/sources/ppocr_testdata/ with a
  ready-to-upload ppocr_testdata.zip for the shared test-data
  release (60 official images: 9 pytest test_files, 21 street scenes,
  20 release/2.7 imgs, 5 ch words, 5 en words; MANIFEST carries
  per-file SHA-256); ecvTestDataRepository registration follows once
  the upload lands.
  P1 interpreter status (2026-09-29): `ocr_runtime.cpp/hpp` implements
  the topology-driven interpreter — 20 of the 21 op types are wired
  (conv2d, depthwise via per-channel decomposition, batch-norm folding,
  elementwise with Paddle auto-alignment broadcast, hard_swish /
  hard_sigmoid / swish / relu / sigmoid, pool2d avg/max with adaptive,
  matmul_v2, transpose2, slice (tensor views + metadata-constant
  slicing), reshape2/squeeze2/flatten, concat, softmax, layer_norm,
  scale, dropout, shape, fill_constant) and the CTC greedy decoder plus
  the rec preprocessing are in place. P1 G1 CORRECTNESS ANCHOR REACHED
  (2026-09-29): test_ocr_runtime runs the full 323-op rec graph and the
  full 312-op det graph on CPU and matches the Paddle reference fixtures
  bit-near-exactly (rec f32 argmax-match + maxDiff 2e-6; det f32 maxDiff
  0.000000). Root causes fixed on the way: flatten_contiguous_range must
  merge [start_axis, stop_axis] instead of fully flattening (the embedded
  graph JSON was also stale - strides/paddings were None - so all three
  topologies were re-exported with per-tensor ranks + static shapes and
  the GGUFs re-converted; release digests pending re-upload); the
  numpy->ggml permute must be the INVERSE mapping (ggml scatters, numpy
  gathers); matmul_v2 needs transposed B plus a contract-layout transpose
  on the result; elementwise with axis=1 must place a 1-D bias on the
  channel axis (can_repeat silently accepted a W-axis broadcast when 16
  divides the width); squeeze2 collects in numpy order; nearest_interp_v2
  resolves FPN targets from the concat/add alignment scan (non-integer
  8->15 cases) via ggml_interpolate; conv2d_transpose uses the
  (Cin,Cout,kh,kw) kernel spelling; mul/add/conv/softmax/norm materialize
  non-contiguous views. C API RUNTIME LIVE (2026-09-29): ocr_capi.cpp
  implements the full 18-function contract — context (det/rec weights),
  run(image_view) with det preprocess (bilinear limit-side resize + 32
  align + ImageNet NCHW), hand-written DB post-process (threshold 0.3 +
  4-connected BFS + unclip offset = area*ratio/perimeter), per-box
  bilinear crop -> 48xH rec -> CTC, typed box{quad,text,score,language},
  truthful pipeline timings, thread-local last_error. qOCR Run is
  enabled (QtConcurrent worker, lazy context); the qAnyLabeling
  PaddleOCR toolbar entry is live and hands off to the qOCR workspace
  (plugin-to-plugin code coupling is against the AICore boundary).
  GPU MATRIX UNBLOCKED (2026-09-29): the CUDA/Vulkan segfault root cause
  (host-resident weight tensors mixed into a device graph with no
  scheduling) is fixed with the facedetect ModelLoader pattern: the ocr
  weights wrap both host pools (GGUF meta + folded constants) as zero-copy
  CPU backend buffers (OcrWeights::realizeHostBuffers/tagHostLeafs),
  upload the named weights to a device buffer once per context
  (realizeToDevice), and run through ggml_backend_sched across {gpu, cpu}
  with ggml_set_input on the image leaf; CPU keeps the verified gallocr
  fast path. Backend matrix (f32 probe p50, RTX 3060): rec CPU 220.6ms /
  CUDA 205.8ms / Vulkan 217.3ms; det CPU 381.8ms / CUDA 358.5ms / Vulkan
  356.6ms; accuracy is identical across all three backends (rec maxDiff
  2e-6 + argmax-match, det maxDiff 0.000000), and f16/q8_0 on CUDA keep
  argmax parity at their storage-precision diffs. GPU wall time is
  dominated by scheduler split/copy overhead and per-run host constant
  copies (folded constants stay host-side), so a shape-stable graph cache
  plus device-side constant folding is the measured performance follow-up,
  not a correctness gap. The C API context now owns its backend pair for
  the whole session (the previous code re-initialized a backend per run).
  Image toolkit promoted to common (2026-09-29): fd::Image + the view
  wrappers + cv_resize_linear_u8 now live in src/common/image_utils.{hpp,
cpp} (namespace aicore::img); facedetect keeps a using-shim header (its
  13 include sites unchanged), the two duplicated internal resize kernels
  were deleted, and the ocr task no longer borrows a facedetect symbol —
  that extern declaration had left an unresolved cv_resize_linear_u8 in
  libAICore's dynamic table (a live load-time defect the CPU probe never
  hit because it does not link ocr_capi.cpp); nm -D confirms it is gone.
  Still queued: polygon-accurate unclip for the official det-eval round
  and region/KIE graph loading (typed arrays and catalog already reserve
  them).
  IN-WINDOW OCR + FACE DETECTION LIVE (2026-09-30): the PaddleOCR toolbar
  entry (Ctrl+4) now runs PP-OCR inside this window through the AICore ocr
  C API (AnnotationWorker ModelKind::Ocr; the picker lists PP-OCRv4
  (ch/en) rec families with the det partner resolved per quantization
  through the dependency channel, exactly like the face landmark model);
  each recognized region becomes a text annotation whose description
  carries the transcription with score/language in extra, so
  the existing PP-OCR dataset export round-trips. Detection through the
  facedetect task (buffalo packs incl. the SCRFD det_500m/2.5g/10g
  detectors) was already fully wired (ModelKind::FaceDetect + detector/
  landmark catalog enumeration); the source scrfd entry above is closed
  as functionally aligned with only the standalone bnkps checkpoints
  remaining exact-weights-unsupported. The old qOCR-workspace hand-off
  dialog is gone (test_plugin asserts the entry triggers in-window
  inference and that the picker carries the PP-OCR families).
  UPSTREAM OCR INTERACTION PARITY (2026-09-30): the det post-process now
  mirrors the official DB pipeline — component boundary pixels feed a
  convex-hull minimum-area rectangle, the Vatti unclip expands it
  (area × ratio / perimeter), and the ordered corners are emitted as a
  clockwise quad; rotated text yields genuine quads, and the rec crop
  uses the official get_rotate_crop_image perspective warp. The plugin
  adopts the upstream ppocr_v4 semantics verbatim: label is always
  "text" (description carries the transcription), an axis-aligned quad
  becomes a rectangle (rebuilt from corner 0/2 as upstream) while any
  rotated quad stays a quadrilateral, and the rec gate is the model
  drop_score 0.5 (upstream wires no confidence slider for OCR). Verified
  end-to-end on the dense ppocrv4 scene image; upstream's
  skip-detection (re-recognize selected shapes only) and the direction
  classifier (cls GGUF) remain the two open interaction gaps.
  IMPLEMENTATION CHECKLIST for the two gaps (design frozen):
  (a) skip_detection —
      1. DONE: `aicore_ocr_recognize(ctx, view, quads, count)` lands in
         `ocr_capi.h`/`ocr_capi.cpp` (perspective warp -> height-48 ->
         CTC over caller quads, full-image pixel coords, same result
         ownership/timings).
      2. DONE: `AnnotationWorker` Ocr branch now routes non-empty
         `prompt.rois` through recognize() and emits shapes with
         `sourceIndices` (the existing GKD/SAM3 roi channel — no new
         infer signature).
      3. REMAINING: `AnnotationWindow` — add the `Skip Detection`
         button next to Run (visible for the OCR family), collect the
         selected text shapes' rects into `prompt.rois` +
         `roiSourceIndices`, and in the finished handler update those
         shapes' description/score in place instead of appending.
      4. test_inference: selected-rects path with the real rec GGUF.
  (b) cls direction classifier —
      1. Download `ch_ppocr_mobile_v2.0_cls_infer.tar` (paddle format,
         paddleocr dygraph_v2.0 release); convert with
         `convert_paddle_ocr_gguf.py --role cls` (extractor is generic)
         + `export_ocr_graph.py` for the embedded topology (ops: conv/
         bn/relu/pool/linear/softmax — all already in the interpreter).
      2. Catalog: add role "cls" entries (f16/q8_0/f32) + digests +
         manifest scenario (cls probe gate: argmax label 0/180 on two
         rotated crops).
      3. `ocr_capi.h`: extend `aicore_ocr_options` with
         `cls_model_path` (extensible struct: struct_size already gates
         ABI); `run()` passes each crop through cls when loaded, flips
         180-degree crops before rec, mirroring upstream
         `use_angle_cls` + `cls_thresh=0.9`.
  DET F16/Q8 CONTAINER FIX (2026-09-30): the det head BN's frozen
  variance reaches 1e7, beyond the F16 range (65504) — the previous
  det F16/Q8_0 containers stored Inf there, which zeroed the head and
  made every det quantized run output an all-zero probability map (the
  old det fixture reference was itself a near-zero map from a bad
  preprocessing distribution, so the old "bit-exact" det anchor was
  vacuous). Fixes: the converter keeps any tensor whose magnitude
  exceeds the F16 range in F32 inside quantized containers (GGUF types
  are per tensor); OcrWeights::load now fails loudly on non-finite F16
  values; the det fixture pair was regenerated through the
  paddle.inference reference channel with the OFFICIAL PP-OCR det
  preprocessing (release + manifest anchors updated), and the manifest
  ocr scenarios were split per quantization with honest gates (det F32
  0.01 / F16 0.1 / Q8_0 0.5 sigmoid-domain max|diff|, rec argmax==1
  plus per-quant max|diff|). Verified on the new anchors: det F32
  maxDiff 1.9e-5, F16 0.058, Q8_0 0.348; rec unchanged (F32 2e-6,
  F16/Q8_0 argmax-identical); CUDA matches CPU on all six rows.
  Probe harness (2026-09-29): the pinned rec/det
  fixture npy pairs moved OUT of the source tree into the shared asset
  cache (uploaded to the PP-OCR release, downloaded by
  FetchAICoreTestAssets into <asset-root>/ocr_fixtures); the probe emits
  a single-line JSON verdict (task/backend/max_diff/argmax/p50/p95) for
  the manifest's generic parser, and ocr-rec/ocr-det scenarios with
  structured accuracy gates (argmax==1, max_diff<1e-3) are registered in
  validation_manifest.json. Quantization: q8_0 containers keep matmul-Y
  weights in F32 (CPU mul_mat src1 must be F32; quantization lives on A)
  and the loader dequantizes F16/Q8_0 parameters; remaining q8 CPU edge:
  1x1 conv weights stored as native 2-D need a reshape-registry entry
  (GPU matrix round will unify this). Re-audited 2026-10-07:
  DONE — the Vulkan backend rows for ocr (`validate_all --backend vulkan
  --tasks ocr` runs det/rec/layout x f32/f16/q8_0 to verdict PASS after
  the fixture pin chain was re-anchored to the regenerated cache files),
  and the qAnyLabeling auto-label entry flip (the PaddleOCR toolbar entry
  is live as AnnotationOpenPaddleOCR). Remaining: the q8 CPU 1x1-conv
  reshape-registry edge, the det DBPostProcess parity-fixture gate in
  validate_all (the postprocess itself was rewritten to match paddlex
  semantics), and the rec semantic-divergence follow-up (paddle-anchored
  gate exposes a systematic divergence; 79% exact on 98 valid rows).
  Also DONE (2026-10-08): the fixture pin chain is fully closed — the 15
  regenerated ocr_fixtures/*.npy were uploaded to the
  Asher-1/cloudViewer_downloads `ocr-fixtures` release tag (replacing the
  stale first-generation assets the user had partially uploaded) via the
  GitHub API using the machine's own git credential store, and all 15
  manifest URLs were rewired from the dead `PP-OCR` tag to
  `ocr-fixtures`. A fresh-clone simulation (cache directory removed,
  validate_all re-run) auto-downloaded all fixtures from the release and
  reached verdict PASS 9/9 — every machine now self-provisions the
  fixtures for testing.
- [ ] rec semantic-divergence follow-up (re-baselined 2026-10-08): the
  official ground-truth capture is unusable — the only saved official
  dump (.qoder/tmp/real_official_pp.txt) is a corrupted capture (empty/
  garbage texts) and the 550x345 medal-table source image is gone from
  disk, so the remaining divergence cannot be quantified without
  rebuilding the official comparison baseline (pick a dense-text image,
  run paddleocr, save the per-box gold, re-run the matching analysis).
  Current verified state: our output decodes 98/98 boxes with sensible
  table text, two independent runs agree 100%, and the Vulkan build
  agrees 90.8% with the CUDA build (cross-backend numeric noise on a
  handful of boxes).
- [ ] Vision-language captioning, VQA and document grounding: `florence2`,
  the video description/segmentation providers and chatbot models. Research
  verdict (2026-09): the ggml tensor library itself does not carry a VLM
  layer — captioning/VQA lives in llama.cpp's `libmtmd` (multi-modal
  library, April 2025, mmproj projectors; supports Qwen2-VL/2.5-VL and
  other vision LMs). The viable route is a new `vlm` AICore task that
  vendors llama.cpp core+mtmd pinned to a release whose bundled ggml
  matches (or is patched to) our pinned ggml v0.21.0, wired through the
  task-private cmake pattern; the plugin must not call llama.cpp directly.
  Scope note: libmtmd covers local captioning, VQA and chatbot (remote
  toolbar workflows are implemented by the later service follow-up) but does NOT replicate Florence2's grounding head —
  phrase-to-box labeling stays on the catalog-backed YOLO-World (CLIP
  text) / YOLOE (MobileCLIP/SAVPE) paths, which are open-vocabulary
  detectors, not VLMs, and are already wired. Integrating mtmd is
  therefore an additive caption/VQA capability, not a weight-parity port
  of `florence2` (the same distinction as YOLO26 vs YOLOv8 exact
  checkpoints). The contract before any UI work: session from a catalog
  entry, text prompt plus borrowed image view in, typed text result with
  release, cancellation and stale-generation discard, common timing.
  `grounding_dino`, `grounding_dino_api`, `grounding_sam`,
  and `grounding_sam2` do not have their exact grounding model in AICore.
  VQA is question-answer annotation for image/video content (answers,
  rationales and question metadata). Grounding resolves a phrase to a box
  or mask for referring-expression labels; it needs the text result
  contracts and a review UI rather than ordinary detector boxes.
- [x] RAM / RAM++ / Tag2Text image tagging uses the public tagging catalog,
  per-class thresholds, shared downloader, resident worker and typed results.
  The RAM Options popup exposes `tag_mode`, `delete_tags` and `filter_tags`.
  English filtering precedes translation; delete takes precedence over keep.
  All 4585 paired source labels and 16 option combinations match actual source
  methods. Raw Chinese whitespace is preserved in the vocabulary; saved tags
  are trimmed and deduplicated by the source UI contract. Repeating identical
  normalized tags preserves the reviewed flag; changed or empty results replace
  the tag array. Object geometry remains intact in single-image and batch runs.
- [x] RAM/RAM++ explicitly selects X-AnyLabeling's OpenCV uint8 INTER_LINEAR
  preprocessing in AICore. The original Pillow bilinear/antialias mode remains
  the default for qRAM and legacy callers. The original by-value request ABI is
  frozen; `aicore_tagging_request_v2` and the two v2 entry points carry the new
  mode without changing old callers' storage. The common validation manifest
  has independent same-GGUF PyTorch truth with the actual pinned source input.
- [x] Tag2Text captions reach the document description, editor and sidecar;
  a real Q8 Run-button test verifies saving and reloading the caption. Tag2Text
  is an additional local catalog workflow without an exact X-AnyLabeling model.
- [ ] Exact source ONNX checkpoint equality is still unproven. Same-GGUF
  PyTorch accuracy, backend agreement and source preprocessing do not establish
  identical threshold decisions for independently exported source checkpoints.
  Historical loaded-host latency numbers and earlier Pillow/OpenCV tag counts
  are not current performance or full-precision parity evidence.
- [ ] RelateAnything has no counterpart in X-AnyLabeling-GGML (no adapter,
  config or widget there); its reference is the official RelateAnything-GGML
  (`deploy/demo_webcam.py`, `deploy/render_video.py`) and the standalone
  qRelateAnything plugin. In the workspace it supports selected rectangles or
  the detector stage (objects added as real rectangles; detector chosen from
  the standalone plugin's catalog: YOLOE, YOLO-World, YOLO segment, RF-DETR,
  YOLO detect), the two official
  operating points with every threshold and edge cap in an Options popup,
  custom predicates through the model's text student, the shared edge
  selection, a display-only overlay, and a filter table. Covered by
  `test_qanylabeling_run_path` (real relsgg-vits16plus Q8_0 through the Run
  button: detector stage adds rectangles and an edge selects one of them,
  selected-rectangle flow, custom vocabulary restricts the edges, sidecar and
  checked state, Auto Run refusal, overlay cleared on deletion),
  `test_qanylabeling_relate_options` (operating points equal the
  qRelateAnything presets; popup behaviour), `test_qanylabeling_relation_filter`
  and `test_qanylabeling_canvas`. Real runs cover YOLOE, RF-DETR nano and
  YOLOv8s-worldv2 as the detector. Gaps: video smoothing lives only in the
  standalone plugin; the detector list is not filtered by what is cached;
  Auto Run is refused because the official demos have no batch mode; and
  there is no screenshot-level UI comparison.
- [ ] Image tagging and attributes still open: `pulc_attribute`,
  `internimage_cls`; generic YOLO classification is not these models.
  Research verdict (2026-09): the image-level classification RESULT
  contract already exists — `aicore_yolo_classify_result` is an image-
  level softmax with class names and no geometry, so no new C ABI is
  needed; what is missing is the models. yolo-cls checkpoints cannot
  substitute RAM (Recognize Anything is a ViT-L multi-label tagger over
  a fixed 4,574-tag vocabulary; a closed-set yolo-cls head covers only
  its training classes), nor PULC (per-attribute human-attribute
  classifiers). The minimal route per model: first-party GGUF conversion
  (RAM's ViT image encoder maps onto existing ggml ops; PULC heads are
  small CNNs), catalog rows + digests, and probes that gate top-k
  accuracy against the reference on pinned images across f32/f16/q8 and
  CUDA/Vulkan/CPU — consumed through the existing classify ABI, with the
  plugin storing tags/attributes as sidecar-level fields (never object
  boxes). The manual Classifier toolbar workflow is now implemented (2026-10-17);
  its optional YOLO classification assistant uses trained categories and does
  not supply the missing PULC/InternImage or local VLM graphs. The later
  service follow-up adds arbitrary-category remote VLM classification.
  Rollout plan (2026-09): R0 converter
  `core/AICore/tools/convert_ram_gguf.py` (ViT encoder + tag decoder
  onto existing ggml ops, tag vocabulary carried as GGUF metadata) with
  reference top-k JSON (G0); R1 a `ram` architecture branch inside the
  existing `yolo` task (role CLASSIFY, sharing that task's classify
  postprocess — RAM is the second classify consumer, so no new helper
  and no new ABI) + catalog f32/f16/q8 rows + digests + `ram-topk`
  probe + manifest rows (G1: top-k gate, f16/q8 parity, matrix green);
  R2 plugin consumption over the unchanged classify ABI with
  tags/attributes persisted as sidecar fields only (G2: regression
  green plus a test asserting no object-box representation).
  Progress (2026-09-28): R0 data plane COMPLETE — the converter
  (`core/AICore/tools/convert_ram_gguf.py`) converted the real
  `ram_swin_large_14m.pth` checkpoint (Swin-L encoder + tag decoder) into
  F32/F16/Q8_0 GGUFs with the tag vocabulary embedded as `ram.tag_list`,
  all three published at the new RAM release. Remaining: R1 yolo-task ram
  architecture branch + catalog rows/digests/probe, R2 plugin sidecar
  fields. The ram digests join `asset_digests.h` together with their
  catalog rows so no digest is published without a consumer.
- [ ] Other detection/pose/lane architectures: `damo_yolo`, `gold_yolo`,
  `yolo_nas`, `yolov6`, `yolov6_face`, `yolov7`, `yolov9`, `yolov10`,
  `yolo12`, `yolox`, `rtdetr`, `rtdetrv2`, `u_rtdetr`, `dfine`,
  `dfine_seg`, `deimv2`, `rio_detr`, `rtmdet_pose`, `upn`, `clrnet`,
  `yolox_dwpose`, and the source YOLOv5/YOLO11 checkpoint families. This
  includes `yolov5`, `yolov5_cls`, `yolov5_det_track`, `yolov5_obb`,
  `yolov5_ram`, `yolov5_resnet`, `yolov5_sahi`, `yolov5_sam`, `yolov5_seg`,
  `yolo11`, `yolo11_cls`, `yolo11_det_track`, `yolo11_obb`,
  `yolo11_obb_track`, `yolo11_pose`, `yolo11_pose_track`, `yolo11_sahi`,
  `yolo11_seg`, `yolo11_seg_track`. Exact weights unsupported: none of
  these checkpoint families exist in the AICore catalog and none may be
  substituted silently. The task-level semantics (HBB, OBB, pose,
  segmentation, SAHI slicing, tracking) already have catalog-backed
  equivalents through YOLO26/YOLOv8/RF-DETR workflows, so a port of any
  given checkpoint is a catalog addition (new GGUF + digest + manifest
  row + probe) inside the existing `yolo`/`rfdetr` tasks — not a new
  plugin surface. Lane detection (`clrnet`) additionally needs a lane
  point-sequence result type that no AICore contract expresses yet.
  Rollout mapping (2026-09): DETR-family (`rtdetr`, `rtdetrv2`,
  `u_rtdetr`, `dfine`, `dfine_seg`, `deimv2`, `rio_detr`, `upn`) ->
  `rfdetr`/YOLO26 detection roles; CNN detectors (`damo_yolo`,
  `gold_yolo`, `yolo_nas`, `yolov6/7/9/10/12`, `yolox`, YOLOv5/YOLO11
  families) -> YOLOv8/YOLO26 detect/segment/pose/OBB roles;
  `rtmdet_pose`/`yolox_dwpose` -> POSE role (COCO-17 skeletons only —
  whole-body 133-point skeletons stay exact-weights-unsupported);
  `yolov8_obb` -> OBB role; `grounding_*` -> WORLD/YOLOE
  open-vocabulary roles; `yolov8_cls`/`pulc_attribute`/
  `internimage_cls` -> CLASSIFY role (RAM covers the multi-label
  tagging use case). Every enabled mapping must cite its catalog row;
  unmatched official entries keep their names disabled with TODOs,
  and `clrnet` stays exact-weights-unsupported until a polyline typed
  result exists.
- [ ] Source YOLOv8 classification, pose, and OBB checkpoints
  (`yolov8_cls`, `yolov8_pose`, `yolov8_obb`) lack matching AICore catalog
  weights. Verified (2026-09-29): the contract side is fully reusable
  already — the `yolo` task implements the pose/obb/classify semantics
  (loader head checks + typed `aicore_yolo_pose/obb/classify_result`),
  and the plugin enumerates those tasks from the live catalog — so this
  TODO is purely a weights-publication round (convert the checkpoints
  with the upstream ultralytics-ggml converter, publish GGUFs, add
  digests + manifest rows + probes); no plugin or task code is missing.
  The YOLO26 equivalents (`yolo26_cls`, `yolo26_pose`,
  `yolo26_obb`) are separate checkpoint families already wired in the
  plugin; exact-weight parity for the YOLOv8 heads remains a catalog
  addition gated on published GGUFs.
- [ ] Source `yolow_ram` exposes one composite YOLO-World + RAM run. The
  underlying chain is now available in the worker: YOLO-World detections can
  be followed by per-box RAM tagging, while standalone RAM and YOLO-World
  runs retain their source-compatible behavior. The real combined-run test
  now covers integer crops, description persistence, undo/redo, stale-result
  rejection and shared `tag_mode`/`delete_tags`/`filter_tags` semantics.
  A dedicated source composite picker/configuration, its default detector/RAM
  pair and exact source checkpoint equality remain open.
- [x] Interactive SAM labeling is functionally covered by AICore SAM3 point,
  box, and text prompts. Exact upstream weights remain future work:
  `segment_anything` (SAM 1), `sam_hq`,
  `sam_med2d`, `mobile_sam`, `edge_sam`, `efficientvit_sam`, `geco`.
  AICore SAM3 does not load these exact weights or preserve their specialized
  preprocessing/output behavior.
- [x] Monocular depth labeling is functionally covered by AICore DA3. It
  replaces the upstream Depth Anything v1/v2 labeling role when a DA3 GGUF is
  selected. Exact `depth_anything`/`depth_anything_v2` checkpoints and their
  source preprocessing remain unsupported until matching AICore GGUF contracts
  exist, so exact-weight parity remains a TODO.
- [x] Source RMBG v1.4 (`rmbg_v14.yaml`) has no matching catalog weight;
  the plugin uses catalog-backed RMBG-2.0 for the corresponding workflow.
  Confirmed status (2026-09): the AICore `rmbg` task owns a BiRefNet-based
  RMBG-2.0 graph with pinned f32/f16/q8 catalog GGUFs
  (`rmbg_f32/f16/q8.gguf` in `asset_digests.h`) and a proven custom-op
  fusion path on every backend, so RMBG-2.0 is the deliberate
  catalog-backed substitute for the v1.4 *workflow* (foreground matte →
  editable polygon), not a weight-parity port. RMBG-2.0 is a different
  architecture (distinct input normalization and output head), so its
  foreground polygons are not pixel-comparable with a v1.4 run; the
  model picker exposes only the catalog-backed RMBG-2.0 entries and a
  v1.4 GGUF stays "exact weights unsupported" until AICore publishes
  one.
- [x] Source `scrfd` configurations are functionally aligned (2026-09-30):
  the facedetect task already publishes every buffalo pack — buffalo_sc
  carries the SCRFD det_500m detector, buffalo_m det_2.5g,
  buffalo_l/antelopev2 det_10g — and the plugin's ModelKind::FaceDetect
  worker path enumerates those packs from the catalog
  (`aicore_facedetect_detector_model_*`), runs
  `aicore_facedetect_detect_image`, and emits face rectangles + 5
  landmarks. Porting the standalone `scrfd_10g_bnkps`-style checkpoints
  (different landmark counts) stays exact-weights-unsupported until
  AICore publishes them.
  bnkps PUBLISHED (2026-09-30): the standalone `scrfd_10g_bnkps` pack
  is now part of the facedetect catalog. `convert_scrfd_onnx_gguf.py`
  (first-party, graph-driven) emits the ONNX topology as the facedetect
  KV graph — node specs keyed by ONNX output names, sanitized to flat
  identifiers, Transpose perms as `pm=`, weights verbatim-F32 — so the
  runtime replays it metadata-driven with ZERO C++ changes (same
  interpreter as buffalo_m/sc). Verified on a real image: 6/6 faces
  match the buffalo_m baseline, 5-point landmarks finite, run-to-run
  stable (test_facedetect_detect_only, ctest `capi;model;validation`).
  Catalog rows + digests + release upload + `facedetect-bnkps` manifest
  scenario landed; the plugin picker lists them automatically through the
  detector catalog enumeration. Renamed to per-tier assets
  `scrfd_10g_bnkps-f32.gguf` / `-f16.gguf` (F32 verified: 6/6 faces, sum
  parity with f16 to 2.5e-5; the F16 container keeps 1-D biases/scales at
  F32 because the CPU binary-op kernels reject f32+f16 operands). A Q8_0
  tier is NOT published: the Q8_0 block quantizer requires 32-aligned
  axes that the SCRFD 1x1 conv weights (e.g. 56x88) do not satisfy, and
  the facedetect conv path has never consumed quantized weights — a
  quantized-conv verification round would be needed first. Landing notes: spec node names must be
  the ONNX OUTPUT names (not node names), gguf-py reverses shapes on
  write (pass ONNX-shaped arrays), and ints arrays land as INT32.
- [ ] `open_vision` and source-specific multi-model combinations are
  external dependencies by design: `open_vision` composes provider-specific
  model runs that no AICore task contract expresses, so it stays disabled
  with its official name; mapping it onto AICore would require defining an
  external-transport contract that violates the ownership map (the plugin
  must not implement a private downloader, backend, or scheduler).
  (remote_server was previously listed here but is fully landed — see the
  remote-server parity entry below.)
- [x] OCR fixture pin-chain repair (DONE 2026-10-08): the 15 regenerated
  ocr_fixtures/*.npy were uploaded to the `ocr-fixtures` release tag via
  the GitHub API (git credential store credentials), replacing the stale
  first-generation assets, and all manifest URLs were rewired from the
  dead `PP-OCR` tag to `ocr-fixtures`. Fresh-clone simulation (cache
  removed, validate_all re-run) auto-downloaded every fixture and reached
  verdict PASS 9/9 — every machine now self-provisions the fixtures.

## AICore-supported work still required in this plugin

These are not exempted by the missing-model rule.

- [x] LoMa has a paired-image correspondence
  review bridge with detector/descriptor/matcher quantization and backend
  selectors, accepted-match point annotations, and a real two-image test
  (`test_qanylabeling_loma_bridge`, current F16 counts CPU/CUDA/Vulkan
  578/581/578 on the pinned Sacre-Coeur pair). Gaussian, LingBot, TRELLIS
  and SAM3D remain owned by `qFreeSplatter`, `qLingbotMap`, `qTrellis` and
  `qSAM3D`; their model matrices are documented in [MODEL_MATRIX.md](MODEL_MATRIX.md)
  but do not belong in this image-labeling picker.
- [x] The separate LightGlue matcher has a paired-image execution path in this
  plugin. ALIKED proposals feed the public LightGlue worker, with real bridge
  coverage and stale-input invalidation; SIFT remains intentionally unexposed.

- [x] Catalog-backed YOLOv8/YOLO26 detection and instance segmentation,
  YOLO26 pose/OBB/classification, RF-DETR detection/segmentation, SAM 2.1/3
  point/box/text interaction, RMBG-2.0 foreground polygon conversion,
  GKD text-keypoint proposals with an optional box ROI, FaceDetect boxes and
  five landmarks, YOLO-World text detection, and YOLOE text/prompt-free
  segmentation with text-encoder quantization and backend selection.
- [x] DA3 single-file depth and nested AnyView Q8_0/Q4_K plus metric-branch
  depth inference through the AICore depth catalog, including dependency
  verification, backend selection and batch depth-image output.
- [x] DeepLSD wireframe/MD line proposals and ALIKED N16Rot local keypoint
  proposals use all published catalog quantizations, the shared verified
  downloader, backend selector and reusable AICore worker context. Line
  coordinates are restored to source-image space. LightGlue is a separate
  two-image feature correspondence worker alongside the LoMa review bridge;
  SIFT remains intentionally outside the picker.
- [x] YOLOE image-prompt (SAVPE) visual prompting from selected visible
  rectangles, with original-image coordinates, checkpoint capability checks,
  prompt-count graph shaping, and backend/quantization selection. AICore
  supports this only when the GGUF includes SAVPE weights; non-SAVPE files
  report an actionable error instead of silently changing prompt modes.
- [x] YOLO26 semantic segmentation converts the source-resolution class map
  into editable exact masks with class IDs, including class 0, holes, and
  isolated pixels. Semantic masks identify categories, not object instances;
  instance overlap and track IDs remain in the instance/video workflows.
- [x] GKD top-down multi-ROI batches from selected rectangles, with grouped
  keypoints. The optional prompt box and whole-image fallback remain available.
- [x] GKD 1-shot visual and paired text/visual prompting using selected point
  annotations as a reusable support image, including batch and multi-ROI runs.
- [x] FaceDetect dense 106/68 landmarks with its paired landmark catalog.
  The detector and landmark GGUFs are resolved through the AICore catalog, the
  dependency is verified/downloaded with the shared asset service, and typed
  2D/3D points are grouped with each face for editing/export.
- [x] AICore-backed SAM3 video segmentation/tracking using the shared frame
  reader and typed tracker results. The qAnyLabeling video action preserves
  frame and instance IDs, supports text/box/point prompts, propagation,
  refinement, timeline review and DB mask export. Exact source SAM2 video
  checkpoints remain unsupported. Other detector/tracker source types still
  need their AICore contracts and catalogs: `yolo26_det_track`,
  `yolo26_obb_track`, `yolo26_pose_track`, `yolo26_seg_track`,
  `yolov8_det_track`, `yolov8_obb_track`, `yolov8_pose_track`,
  `yolov8_seg_track`; YOLOv8 pose/OBB still need their exact GGUFs.
- [x] SAM3 tracking archives can be imported back into the image annotation
  workspace. The importer validates the manifest, frame/image dimensions and
  mask paths, preserves exact PNG masks and holes, and maps stable instance IDs
  to `group_id` for subsequent brush edits and MOTS export.
- [x] Open a tracked video archive directly into an editable frame workspace.
  The shared OpenCV reader extracts every source frame, validates count and
  dimensions, and publishes the folder atomically with matching sidecars.
  The alternate import into an existing folder still requires a complete,
  ordered frame list.
- [x] Bind SAM3 tracking archives to the source-video SHA-256. New exports
  compare source digests before and after tracking; source changes or read
  failures mark the archive incomplete. Imports reject malformed or mismatched
  digests. Legacy v1 manifests without `source_sha256` remain readable.
- [x] AICore-backed sliced detection (`yolo26_sahi`, `yolov8_sahi`) now uses
  source-compatible 512px tiles, 20% overlap, full-image fallback and
  class-specific IOS GREEDYNMM merging. The slice controls are persisted in
  the prompt and expose size, overlap and merge threshold in the Qt UI.
- [x] Detection-to-SAM chaining is available for the catalog-backed workflow:
  select visible detector rectangles, choose a SAM3 model, and run it without
  a manual point/box prompt. Each box is inferred independently and its mask
  retains the source class/group through `source_roi`. The exact upstream
  `yolov8_sam2` checkpoint and its detector/tracker ownership are still absent;
  `yolo26_*_track` ReID encoders remain embeddings and must never appear as
  classification models.
- [ ] Validate each wired task on every available backend/quantization with
  real pinned models and task-specific accuracy gates. The existing repository
  report covers 122 real catalog rows on CUDA and passes all qAnyLabeling task
  rows. A 161-row CPU/Vulkan expansion was run far enough to execute the
  non-SAM3 rows and retain per-row probes, then interrupted during the slow
  SAM3/YOLO tail before either backend emitted a complete summary. The
  expanded CPU/CUDA/Vulkan cross product remains the release gate. Benchmark
  against the source application before claiming a labeling speedup.
  Re-run status: the single non-qAnyLabeling failure row in the CUDA report
  (`sam3d/sam3d-e2e-q4k`, return code -6) was re-verified with a narrowed
  `--models "sam3d-e2e-q4k*"` run (2026-09-30): BOTH candidates now finish
  rc=0 with stable fingerprints at a consistent ~278 s wall time
  (verdict PASS), so the earlier abort no longer reproduces and the row is
  no longer a blocker for the cross-backend matrix. The expanded
  CPU/Vulkan/CUDA cross product remains the release gate. Benchmark
  against the source application before claiming a labeling speedup.
  OCR matrix (2026-09-30): the manifest ocr scenarios were split per
  quantization (F32/F16/Q8_0 x rec/det) with per-quant gates, the probe
  exports a machine-readable JSON verdict with a per-scenario tolerance,
  and the narrowed CPU matrix over all six rows PASSES (F32 bit-near-exact;
  F16/Q8_0 within their sigmoid-domain/argmax gates).

## Manual UI, data and export parity

- [x] Translation catalogs are fully populated (2026-09-30): the Qt
  `.ts`/`.qm` pipeline now carries complete zh_CN, ja_JP and ko_KR
  catalogs (856/856/856 finished entries). The upstream X-AnyLabeling
  translation tables (1635 fully-finished entries per locale) supplied
  the exact-match base plus the canonical vocabulary for shared UI
  concepts, and the adapter-specific strings were translated to match
  that vocabulary. All three locales can now be enabled in the Language
  menu; untranslated strings still fall back to English for strings
  added later until their entries are translated.

These features must be implemented regardless of model coverage.

- [x] Image JSON sidecars, nine geometry tools, brush Paint/Erase/Invert modes
  (including Ctrl erase, `R` switching, target-class editing and hole-safe
  stroke undo/redo), magic wand, object
  labels/groups/flags, undo/redo, object copy/paste/duplicate and visibility
  shortcuts, checked-only navigation,
  YOLO HBB/Seg/OBB import and export, VOC det/seg, DOTA, and COCO
  detection/instance segmentation/keypoints import,
  YOLO Pose import/export using the official
  `has_visible`/`classes` list layout, VOC det/seg, DOTA and
  COCO detection/instance segmentation/keypoints export, and RGB/grayscale
  mask import/export with source color-map priorities and checked/empty options.
- [x] Label Change Manager range operations include category rename/removal,
  visibility, and color editing. Recoloring preserves exact mask alpha and
  holes, and unchanged sidecars are skipped. The `acloudviewer_color` shape
  field is an ACloudViewer extension for reopening the chosen color.
- [x] Image-level `tags` panel reads the source ordered array, normalizes and
  deduplicates tags, and supports official default names (`tag`, `tag1`, ...),
  add, rename, Escape cancel, focus-out commit, confirmed multi-delete,
  batch-delete mode with selected-tag strike-through, copy-all, drag ordering,
  auto display for tagged sidecars and `Ctrl+Shift+T` visibility. Persistence
  is covered when switching images with an unfinished editor. Exact source
  chip layout, hover close button, keyboard drag semantics, and screenshot
  parity still need visual verification.
- [x] Image Flags now follow the source workflow: `Upload > Image Flags` reads
  a UTF-8 one-name-per-line template, keeps source order, shows the Flags dock
  only after a template is loaded, and merges each image's checked values into
  its JSON sidecar. The per-image add/remove controls were intentionally
  removed because they are not part of the source interaction model.
- [x] Label Flags, Attributes and Label Classes template editors are
  implemented against the source contracts: Label Flags reads the YAML
  label-to-flag-list template through the shared safe_load-subset parser,
  merges labels into the list and renders per-label checkbox fields in
  the label dialog (persisted in shape `flags`); Attributes validates the
  JSON `__widget_types__` schema (combobox, radiobutton, group_id,
  lineedit) and edits the selected shape through a generated dialog
  (persisted in shape `attributes`, group_id widget edits the shape
  group); Label Classes replaces the label list after the source loss
  warning. The three Upload actions keep their official names and icons
  and are covered by the plugin regression test.
- [x] Source digit-key shape shortcuts: editable 0-9 drawing mode and label
  mapping, persistent Qt settings, clear-all control, and synchronized tool
  selection while the canvas has focus.
- [x] MOT detection-box tracking import/export with frame IDs, group IDs,
  `gt.txt`, `det.txt`, and `seqinfo.ini`.
- [x] MOTS polygon tracking export with `custom_gt.txt` and `seqinfo.ini`.
- [x] ODVG detection import/export with JSONL image records and `label_map.json`.
- [x] VLM-R1 OVD JSONL conversation import/export for valid JSON box answers,
  optional classes and image path prefix.
- [x] Canvas Ctrl-click and object-list multi-selection, group/ungroup ID
  merging, rectangle/polygon union, source shortcuts `G`/`U`/`Ctrl+Shift+M`,
  undo and sidecar persistence. File actions include source-compatible
  current-image close (`Ctrl+W`), label-only Save As (`Ctrl+Shift+S`), and
  confirmed label-file deletion (`Ctrl+Delete`).
- [x] File menu placement now follows the source order for open/navigation,
  compare, recent files, save, close and delete actions. `Save Automatically`
  is persisted and writes after edits; `Save With Image Data` embeds the
  current image as PNG base64 in the sidecar.
- [x] Change Output Dir is implemented as a persistent session contract:
  per-image sidecar resolution (load, save, save-as suggestion, checked
  state discovery, delete with image-dir fallback) routes through one
  helper that consults only the active output directory, batch export
  dialogs default to it, Auto Run saves through it, and the File menu
  action re-scans the folder while retaining the current selection —
  matching the source runtime state semantics (not a saved setting).
- [x] Video-classification sidecar schema, playback, labels/colors, segment
  timeline creation, time/description editing, atomic save and collision guard.
- [x] Video-classifier batch dataset export groups all labeled segments by
  class, writes MP4 clips and/or FPS-selectable JPEG sequences, emits the
  sorted label map and metadata, and optionally publishes a validated ZIP.
  It reads the current editor state, including unsaved segment edits, and
  rejects output path collisions before encoding.
- [x] Source video classifier's stream-copy/re-encode choice with fallback,
  metadata, and a single settings dialog. Stream copy uses FFmpeg when
  available; explicit re-encoding matches the source `libx264`, `veryfast`,
  CRF 23 command, and either path falls back to the existing OpenCV encoder
  when FFmpeg is unavailable.
- [x] Source video classifier's keyboard shortcut parity, including digit label
  selection and an in-dialog shortcut list. Label settings/import,
  frame-snapped
  segment boundary handles, thumbnail timeline, selected-segment clip/raw-frame
  export, image recent-file menu, and image/video drag/drop are implemented in
  the Qt workspace.
- [x] Point-cloud `.bin`/`.ply` and `.label` workflow, Qt/OpenGL rendering,
  top/front/side/orbit views, pan/zoom, brush and polygon selection,
  through-selection, semantic/instance assignment, frame navigation, and
  undo/redo. Labels use the source little-endian 32-bit packing and atomic writes.
- [x] Point-cloud class and instance visibility lists, source version-1 class
  JSON load/save with semantic names and colors, and new/edit/remove class
  actions with color validation. Hidden filters persist while changing frames.
- [x] Point-cloud class-configuration dirty-state prompts now cover close,
  frame changes and replacing a class file. Save/Discard/Cancel preserves
  point labels while keeping class IDs stable.
- [x] Point-cloud parity remaining, resolved 2026-10-07 (the four
  "subtasks" split apart after upstream verification — two had no upstream
  counterpart at all):
  (1) Surface-depth picking — LANDED. The view already projected every
  visible point and kept the closest per screen pixel; it now also matches
  upstream selection.py's exact semantics: each point owns a
  point_size²-pixel footprint (closest depth tracked per footprint
  pixel) and the non-through filter keeps a point when any of its
  footprint pixels lists it at the exact closest depth (float equality,
  so equal-depth ties stay selectable together — the old 0.0001-tolerance
  variant silently dropped them).
  (2) Renderer tile acceleration — DROPPED, not a parity item: the
  upstream viewport.py has no tile/chunk rendering (the only "tile" is
  numpy.tile), and this view's pipeline is a single glDrawArrays over one
  VBO that re-uploads only on visibility/color/frame changes — no
  measured bottleneck exists to justify chunked draw ranges. Reopen only
  with a profiling trace showing draw/upload time dominates.
  (3) Camera image calibration — MATH KERNEL LANDED: the upstream
  load_calibration strict schema (schema_version 1, pinhole, rigid
  transform with det=+1 orthonormal rotation, opencv5 or none distortion),
  project_points (far-to-near ordering, in-bounds filtering) and
  project_cuboid (12 edges, near-plane clip with interpolation, 33-sample
  subdivision under distortion, image-rect segment clipping) are ported
  to PointCloudCamera.{h,cpp} and pinned by test_qanylabeling_pointcloud_camera
  (valid/invalid schema, principal-point projection, depth sorting,
  radial distortion direction, behind-camera rejection, rect clipping).
  Still open: the 3D-viewport cuboid wireframe pass and interactive
  rotation/resizing (the data model, .cuboids.json sidecar IO, the
  Create-cuboid-from-selection operation and the camera-dock cuboid
  overlay are all landed and tested — see PointCloudCuboid.{h,cpp} and
  the pointcloud_camera fixture).
  (4) Streaming frame loader — LANDED: selectFrame now serves neighboring
  frames from a bounded background prefetch cache (QtConcurrent decode of
  index±1/±2, four-entry cache, in-flight de-duplication), so stepping
  through a folder skips the file decode entirely.
  Also fixed a stash-merge conflict remnant in plugins/core/Standard/
  CMakeLists.txt that had silently removed the qAnyLabeling
  add_subdirectory registration.

- [x] Point-cloud follow-up: live Datumaro/KITTI Raw/Sly import/export dialogs,
  transactional application with cancel/stale-state protection, separate
  detection/segmentation classes, multi-camera configuration and switching,
  overlay settings and point/cuboid visibility. Z-up camera transforms,
  perspective/default limits, focus and oriented ray-box selection now use
  the upstream numeric contracts. UI, archive and numeric probes cover these.
- [ ] Point-cloud parity remaining: the upstream perspective camera-control
  panels, full keyboard/gesture coverage and screenshot-level theme/details.
  Whole-frame background loading with cancellation and bounded neighboring-frame
  prefetch is implemented. Focus now follows the upstream 250 ms orthographic
  easing and distance-dependent perspective decay. Chunk/tile rendering is an
  optional optimization absent from the inspected upstream loading contract. GPU
  surface-depth picking, renderer tile caching, camera configuration preview,
  cuboid workflow and dataset transactions are covered by current tests.
  These remain UI/rendering tasks independent of model support.
- [x] Image Navigator dock with viewport thumbnails and F9 toggle, plus a
  native Compare View with recursive basename matching, same-size validation,
  shared canvas zoom/pan, split slider, close/reset, and `Ctrl+Alt+C`. The
  source-compatible `Ctrl+Shift+Delete` action moves the current image and
  sidecar into `_delete_` after confirmation.
- [x] Serial source-compatible `Auto Run` (`Ctrl+B`) for supported catalog
  models. It supports an inclusive image range and cancellation, reuses the
  AICore worker context,
  replaces each image's generated shapes, saves each sidecar atomically, and
  stops on a download, input, or inference error.
- [x] The host plugin contributes one clearly named X-AnyLabeling action using
  the upstream icon. Its image, video, and point-cloud entry points stay inside
  the annotation workspace. The left toolbar now follows the source action
  order, uses the copied source icons, and has one all-shapes visibility toggle;
  the plugin test checks the host action count, source toolbar order, and icon
  presence. Remaining ACloudViewer-only commands follow the source sequence.
- [x] SAM3 video tracking can run from the current frame through the clip end
  and export per-instance PNG masks plus an atomic `manifest.json`; cancelled
  and failed runs retain a marked partial manifest. Starting later prompts for
  a remaining-frame export or a reset to frame 0; resetting clears manual
  instances, which must be prompted again before full-clip export. Source-specific video
  classifier batch export is listed above.
- [x] View menu order, model panel placement and command-state lifecycle now
  match the source workbench: the View menu follows the official
  label_widget.py sequence (Navigator, Image Tags, Fill Drawing Polygon,
  Loop Through/Select Labels, Zoom In/Out/Original Size, Keep Previous
  Scale/Brightness/Contrast, mutually exclusive Fit Window/Fit Width, and the
  show/hide visibility group), `Keep Previous Scale` reuses the previous zoom
  across images, Compare View lives only in the File menu, Auto Labeling only
  in the left toolbar, and the model panel is a plain section between the
  instruction bar and the canvas with no dock chrome or title bar.
  Image-dependent commands (save, navigation, Delete File, Mark as Checked,
  Auto Run, zoom/fit, loop and visibility toggles) start disabled and follow
  the image, file-list, sidecar and model-panel lifecycle, verified by the
  plugin regression test.
- [x] Interaction-logic parity round two: `Loop Through Labels` now matches
  the source review loop (snapshot count, per-shape zoom-to-fit at one
  quarter viewport width, progress in the status bar, selection cleared and
  view restored at the end, reset on image change/close);
  `Loop Select Labels` clears the selection when the walk passes the last
  object; the Edit menu follows the source order (create modes, edit modes,
  object commands, undo, coordinates/merge, mode toggles) with the Tool-menu
  managers removed from their duplicate Edit placement; `Copy Coordinates`
  copies the source `[x1, y1, x2, y2]` integer rectangle or flat polygon
  payload; `Use System Clipboard` is exposed as the source Edit toggle and
  stays in sync with the settings dialog; and the selection-driven enable
  states (delete, duplicate, copy, paste, same-type edit, single-selection
  coordinates, rectangle/polygon merge, single unlocked polygon brush)
  follow the source canvas_selection_changed model.
- [x] Interaction-logic parity round three: the canvas context menu now
  leads with the source `Filter by Label` / `Filter by Group ID` submenus
  (exclusive options derived from the current shapes, driving the object
  list visibility with the source matching rules) plus Mark as Checked and
  the full source actions.menu sequence including `Lock/Unlock`; the
  `Remove selected point` action now follows the source vertex-level hover
  signal instead of shape selection; the review loop reports progress
  through the focus-free click-through `AnnotationPopup` toast (source
  Popup flags, 1800 ms auto-close) instead of the status bar; the settings
  dialog gains the upstream `Display Label Popup`, `File List Checkbox
  Editable` and `Crosshair Color` fields with real behavior, plus a visible
  TODO note for unimplemented upstream field groups; and the shortcut
  table was fully diffed against `xanylabeling_config.yaml` — the only
  missing binding was `quit` (Ctrl+Q), now registered through the
  confirm-on-close path (the file-list checkbox default intentionally
  stays `true` because the checked-image workflow renders through it).
- [x] Review-loop toast occlusion fix (root cause only): two earlier
  positioning-only repairs were reverted back to the committed Popup
  mechanism after the backtrace-probed real root cause proved unrelated to
  coordinates — Qt children are visible-by-inheritance, so the
  constructor-created toast was mapped at (0,0) over the menu bar by the
  workspace's show(). The popup now starts hidden and showMessage() is the
  only show path; the workspace became a top-level OS window (Qt::Window,
  matching the source standalone window) instead of a frameless child
  embedded in the main window, with the plugin's stale m_window pointer
  reset on destroy. The offscreen end-to-end probe
  (test_qanylabeling_popup_probe) pins open-time hidden, toast placement
  and auto-close; on-screen screenshots confirm the titled standalone
  window and an unobstructed menu bar.
- [x] Interaction-logic parity round four: the source shape.* color group is
  fully wired into the canvas render path (color strategy auto/manual,
  default shape color, auto-color hue shift, line/fill/vertex/select-line/
  select-fill colors, plus a hover-vertex highlight using
  hvertex_fill_color) with nine persisted settings rows; the label dialog
  was replaced with a source-compatible AnnotationLabelDialog honoring
  sort_labels, show_label_text_field, label_completion (startswith/contains
  filtering) and fit_to_content. Corrected 2026-10-17: move_mode controls
  label-popup placement (auto at the cursor, center on the screen);
  left-drag canvas panning over empty space is independent of this setting;
  font_family applies an application font to the workspace; and two
  offscreen end-to-end fixtures now drive the exec()-blocking surfaces from
  inside their nested loops: the canvas context menu (filter submenu
  structure and population, plus SAM right-click suppression) and the
  modal label dialog (sort, completion filter, accept).
- [ ] Remaining interaction parity (re-audited 2026-10-07, split into
  closable subtasks): (a) `validate_label` — DONE (2026-10-07): the
  General settings page exposes the source (none)/"exact" choices, the
  field persists, `AnnotationWindow::validateLabel` encodes the rule, and
  the check is wired into the batch-edit dialog (editLabels) apply path —
  under "exact" every replacement label is validated against the unique
  label list before any file is written. (b) `logger_level`
  — DONE (2026-10-08): the General page exposes the level field and all
  14 plugin log sites route through the level-gated qany_log helper
  (AnnotationLogging.h) — "warning" drops info lines, "error" keeps
  only errors — before falling through to the repository's single-point
  aicore_inference_log [Error]/[Warning] routing; the settings dialog
  refreshes the cached level immediately on apply.
  (c) `file_list_checkbox_editable` now defaults to false, matching upstream;
  the historical true-default exception was removed. `model_hub` remains
  owned by the AICore catalog; (d)
  `qt_image_allocation_limit` stays blocked on a Qt 6 build.
  Full-surface audit (2026-10-08): a feature-level diff of the upstream
  label_widget action inventory (312 tr entries) against the plugin
  found no missing interaction surface — the apparent gaps (Create
  Cuboid/Circle/Rotation, Copy Object, Batch Edit, Copy Coordinates) are
  wording differences over implemented features (canvas supports all
  upstream shape types incl. cuboid/circle/rotation; copyShapes uses the
  upstream clipboard MIME format; the batch edit dialog is the Label
  change manager). The genuinely remaining items are visual-fixture
  work and the large model-zoo/AICore-contract matrix tracked above.
- [ ] Remaining image workflow: exact edit/settings action ordering beyond
  the now source-ordered File, View, Edit, Upload/Export menus and verified
  left toolbar sequence, selection visuals,
  image-level annotation settings, complete shortcut and theme parity, and
  source-specific compare reset details. The implemented view
  subset covers Navigator, Compare View, original/width zoom, independent
  shape/mask/label/score/description/attribute/group/degree/linking visibility,
  group-boundary selection, and the bottom-left Canvas Display
  panel (opacity/brightness/contrast), per-file brightness/contrast state and
  source-style keep-previous toggles. A four-category Settings dialog now has
  the upstream `Ctrl+0` entry, a live QAction shortcut inventory, editable and
  persisted upstream mappings, duplicate/conflict rejection, host-action
  suspension while qAny is active, and persisted brush radius, brush operation,
  mask opacity, shape/mask/label visibility and keep-previous/last-label
  behavior. Exact overlay styling, remaining source settings fields and
  screenshot-level layout comparison still need parity fixtures and controls.
  Upstream visual references are now in-tree (2026-09-30):
  `docs/images/` carries the upstream `assets/resources` screenshots and
  interaction GIFs (overview, navigator, label/shape managers, settings,
  compare view, brush/magic-wand loops) copied verbatim from the
  X-AnyLabeling sources as the baseline fixtures for those comparisons.
  The mask opacity control now matches the source compositing — masks are
  drawn as a binary silhouette filled with the shape color at the 0-255
  opacity (cached per shape/mask), verified by an image-level fixture —
  but exact overlay styling and remaining screenshot-level comparisons
  stay open above. Settings-field diff (2026-10-07, CONVERGED): the
  upstream schema is anylabeling/views/labeling/settings/schema.py
  (SETTINGS_GENERAL_KEYS :353, SETTINGS_SHAPE_KEYS :365, SETTING_FIELD_MAP
  through :1499, ~90 keys); the two key groups diff to 18 concrete
  settings and all of them are now either implemented in the four-category
  AnnotationSettings pages (General: auto-highlight, auto-switch-to-edit,
  exif scan, file-list checkbox, system clipboard, font family,
  logger_level, validate_label; Shape: default color, auto color shift,
  line/fill/vertex/select-line/select-fill/hover-vertex colors, point
  size, line width) or consciously owned (model_hub by the AICore
  catalog; qt_image_allocation_limit
  pending a Qt 6 build). No settings-field work remains; what is left for
  full parity is visual only — exact overlay styling and screenshot-level
  layout comparison against docs/images/ fixtures.
- [x] PPOCR Rec/KIE annotation stream import/export, including transcription,
  quadrilateral/polygon geometry, difficulty, labels, group IDs, and linking.
- [x] PPOCR Rec directory export generates perspective-corrected `crop_img`
  JPEGs with `rec_gt.txt` and `Label.txt`, clears prior Rec artifacts, and
  preserves empty-file behavior. Pixel interpolation and polygon-to-quad
  ordering still require a source OpenCV image-level parity fixture.
- [x] VLM-R1 import accepts valid, fenced and `<answer>`-wrapped JSON plus
  common `json_repair` forms (single quotes, unquoted keys and trailing
  commas); unrepairable answers follow the source and become empty results.
  Broader `json_repair` grammar and exact prompt fixtures remain to be tested.
- [x] COCO uncompressed and compressed RLE segmentation import decodes the
  official column-major runs, preserves exact alpha masks (including holes and
  disconnected components), and exports native exact masks back as COCO RLE.
- [x] Compare mask edge pixels against OpenCV `fillPoly` on the same source
  fixture. OpenCV is used for the production mask rasterizer when available;
  the Qt painter remains a documented fallback for builds without OpenCV.
- [x] Pose config parsing accepts the full safe_load construct set the
  source reads: any consistent block indentation, sequence items at the
  parent key's indent, single- and multi-line flow style, quoted scalars,
  comments, extra root keys, YAML 1.1 booleans for `has_visible`
  (default true), anchors/aliases, duplicate mapping keys (last value
  wins, first position kept) and preserved duplicate keypoints; tabs,
  multiple documents and non-mapping roots stay rejected. Mixed-class
  and flow/indent/anchor fixtures verify identical YOLO pose text output
  against the canonical schema.
- [x] Fixed the `test_qanylabeling_plugin` harness (2026-10-07). The
  earlier "fails in Qt's metadata validation layer" diagnosis was a
  measurement blind spot (strace was filtered to openat only). Two real
  causes, both fixed: (1) QPluginLoader rejects a non-normalized relative
  path with "The shared library was not found" — the test now resolves
  argv[1] through QFileInfo::absoluteFilePath; (2) the ctest environment
  lacked the dlopened plugin's dependency dirs — LD_LIBRARY_PATH (PATH on
  Windows) is now injected with generator expressions, the same pattern
  as test_qanylabeling_popup_probe. The probe remains in place pinning
  the review-toast open-time-hidden contract end to end.
- [ ] Remote-server auto-labeling parity (investigated 2026-10-07):
  upstream `remote_server` (services/auto_labeling/remote_server.py,
  879 lines) is an inference CLIENT — it POSTs base64 data-URI frames to
  `{server_url}/v1/predict` (plus `/v1/models`, SAM3 and video-session
  endpoints), parses the returned shapes, and is configured through
  `remote_server_settings` {server_url, api_key, timeout} with an
  XANYLABELING_SERVER_URL env override. Verdict: genuinely useful
  (offloads heavy models to a remote inference service) and cheap in
  dependencies — every transport piece maps onto Qt-native
  QNetworkAccessManager/QImage/QJsonDocument, no new third-party library.
  The earlier ownership-map objection (no private
  downloader/backend/scheduler) does not apply: the remote service is the
  inference backend by explicit user configuration, not a plugin-side
  reimplementation of one.
  P0 LANDED (2026-10-07): `RemoteServerClient` (transport + data-URI
  framing + shape JSON parsing, upstream protocol parity), the
  `RemoteServerDialog` connection dialog (URL/api-key/timeout with live
  /v1/models fetch), the `Remote Server` pseudo-entry in the model picker,
  and the `startRemoteInference` dispatch through onInference with
  stale-generation discard. Settings persist under qAnyLabeling/
  remoteServer{Url,ApiKey,Timeout,Model} with the XANYLABELING_SERVER_URL
  env override. Verified by `test_qanylabeling_remote` (in-process
  QTcpServer mock, models+predict round trips).
  P1+P2 LANDED (2026-10-07): video-session endpoints
  `/v1/video/init` (frame upload, timeout x2), `/v1/video/prompt` (text
  prompts with upstream trailing-period strip; point prompts with
  relative coords, 0/1 labels and the obj_id 10000 sentinel),
  `/v1/video/propagate/stream` SSE parsing (started/progress/completed/
  error events, no timeout as upstream, results keyed by frame index),
  and `/v1/video/cleanup/{id}?model=` on session reset (local state
  clears even if the POST fails). P2: `setResetTracker()` one-shot flag
  riding the next predict params (upstream
  set_auto_labeling_reset_tracker parity); upstream `unload()` is a
  no-op (`pass`) with no server protocol, so no client API is needed.
  All verified in `test_qanylabeling_remote` (keep-alive QTcpServer
  mock: init/prompt/point-prompt/SSE propagation/cleanup round trips,
  reset-tracker one-shot assertion).
  VIDEO WIRING LANDED (2026-10-07): `AnnotationWindow::startRemoteInference`
  now probes /v1/models first and routes by the server-declared
  batch_processing_mode (upstream get_batch_processing_mode parity; a
  single-task available_tasks entry may override the model-level mode).
  default keeps the plain predict; video runs the full upstream session
  flow in-window: frameDataUris packs the file-list frames (capped at
  2000 before the base64 payload can exhaust the process), /v1/video/init
  uploads them, the current frame is prompted with the text box or the
  canvas points converted to relative coordinates (obj_id 10000), prompt
  shapes land through the regular stale-checked inference path, and the
  SSE propagation writes every frame's masks into its sidecar JSON
  (saveRemoteShapesToSidecar preserves existing flags/tags/checked and
  fills image dimensions via QImageReader headers). Progress reports
  through the status line; the session is released on completion or
  error. Remaining: the qAnyLabeling toolbar
  `AnnotationTODO_open_paddleocr`-style automation mapping.
- [x] Test-build layout refactor (2026-10-07): the 450-line monolithic
  CMakeLists.txt is now split into the repository-standard form used by
  every other tested plugin — the main file (138 lines) keeps only the
  plugin target, and all 17 test executables moved to tests/CMakeLists.txt
  built with the new repository-wide cloudviewer_add_plugin_test helper
  (cmake/CloudViewerOutputDirectories.cmake) plus
  cloudviewer_set_plugin_test_env_offscreen, replacing ~180 lines of
  hand-copied PIC/-no-pie/output-directory/env boilerplate. The refactor
  also fixed three latent defects: a stray ">" in the hand-written
  LD_LIBRARY_PATH/PATH generator expressions, the canvas offscreen
  property registered inside the sliced-detection block, and seven tests
  that ran without any offscreen/library-path environment injection.
  Note for the other tested plugins (qYOLO/qGKD/qRMBG/qTrellis/
  qMapAnything/qRFDetr/qFaceDetect): their tests/CMakeLists.txt still
  hand-write the same boilerplate and can adopt
  cloudviewer_add_plugin_test incrementally.
- [ ] Training UI, annotation quality/consistency tools, and workflows that
  depend on external services. Implement only after their data contracts and
  ownership are identified.
- [x] Unimplemented training retains its official name/icon and disabled TODO
  tooltip. ChatBot and VQA are enabled by the resource/service follow-up below.
  Image Classifier now provides manual classification, category editing,
  navigation, export, and an existing-model assistant. Save Visualization
  Video is implemented in the 2026-10-17 follow-up below. PaddleOCR, Shape Manager, Change Output Dir, and the
  Label Flags / Attributes / Label Classes uploads are implemented and must
  not be counted as disabled placeholders.

## 2026-10-17 defaults, canvas and image-classifier follow-up

Reference: X-AnyLabeling-GGML `745f6c355f94cddb06f6965b3a4118f82033e935`
(v4.1.3). This closes the following measured surfaces; the complete TODO,
model-zoo contracts and full-window visual parity remain open.

- [x] Central source-default snapshot with YAML provenance/SHA-256 and shared
  accessors for the workspace/canvas. Cold-start comparison executes the real
  plugin: all 80 shortcut groups, 13 checkable actions, five docks and 35 model
  confidence/IoU defaults agree with the source. Existing saved preferences
  still override defaults. Japanese/Korean translations are selectable.
- [x] Fix `move_mode`: auto positions the label popup near the cursor; center
  positions it at screen center. It does not configure canvas panning. Preserve
  shortcut arrays and restore source defaults when an override is cleared.
- [x] Source palette, encounter-order label colors, selected/custom colors,
  label/group/score placement, description/attribute styling, circle
  center/radius geometry, shape-mask visibility, vertex markers, difficult
  dashed lines and difficult-point triangles. The real upstream rendering
  comparison passes 36 cases and 768 palette indices. All 18 geometry-only
  images are pixel-exact; text checks explicitly allow one-pixel Qt 5/6 font
  differences with separate error gates. This is export/basic canvas evidence,
  not a full-window or overlapping/selected-state screenshot claim.
- [x] Public YOLO/RF-DETR catalog entries own annotation confidence/IoU defaults;
  the plugin has no duplicate model-default table. The editable IoU value is
  passed through the current worker and remote parameters. Fix the public YOLO
  setters to accept both endpoints of [0,1], retaining the last valid value on
  out-of-range/NaN/infinity input. The public C API contract probe passes.
  With `yolov8n-q8_0.gguf` and the shared `bus.jpg`, two forwards on the same
  worker/context at IoU 0 then 1 produce 3/40 boxes on CPU and 4/44 boxes on
  CUDA/Vulkan. These are threshold-propagation checks, not framework accuracy
  or performance evidence. CUDA requires the matching 11.8 runtime on this
  machine; the first attempt picked system CUDA 11.1 and did not load.
- [x] Image Classifier is enabled at Ctrl+3. Manual MultiClass/MultiLabel,
  AND search, category add/delete/rename, statistics, navigation, numeric
  shortcuts, flags/metadata preservation and class-directory export are
  implemented. The real plugin regression verifies mode switching, filtering,
  navigation, saving and export. It uses the main document/worker/batch flow;
  there is no second downloader, model cache or inference session owner.
- [x] Guard the classifier's busy controls/shortcuts and invalidate its own
  in-flight result on close. Reuse main-window batch cancellation. Image-level
  batch classification preserves existing object shapes instead of replacing
  them with the empty classification geometry.
- [x] Final real-model classifier/batch acceptance and full current qAnyLabeling
  regression: **30/30 PASS, no skips**, 103.92 seconds on the current tree.
  The YOLO26n q8 classifier probe verifies all 1000 flags, exactly one selected
  class, two-image batch preservation of shapes/metadata and stale-result
  rejection after close. Strict plugin/AICore ownership check also passes.
- [ ] Still open: local VLM graphs and full ChatBot/VQA interaction
  parity, PULC and
  InternImage, full-window themes/layouts and all hover/selection/overlap states,
  face-model live IoU configuration (no public setter), the remaining exact
  model families, complete point-cloud UI/rendering parity and the earlier release gates.
  The Local model classifier uses trained YOLO categories. The later service
  follow-up adds the source arbitrary-category remote VLM workflow.

Evidence (build-tree files): `qanylabeling-config-upstream.json`,
`qanylabeling-render-upstream/report.json`, `qanylabeling-iou-{cpu,cuda,vulkan}.log`,
`qanylabeling-yolo-capi-contract.log`, `qanylabeling-parity-boundaries.log`, and
`qanylabeling-full-final.log` under `build_app/Testing`. Reproduction commands
are in README. OCR implementation/accuracy work is excluded from this round.

## 2026-10-17 RAM/OCR consumer and visualization-video follow-up

Reference remains X-AnyLabeling-GGML
`745f6c355f94cddb06f6965b3a4118f82033e935`. **The entire TODO is not complete.**
This round changes the AnyLabeling OCR consumer and reuses the merged OCR
runtime/qOCR unchanged; the earlier withdrawn OCR experiments stay withdrawn.

- [x] RAM/RAM++ language selection, English delete/keep filters with delete
  precedence, exact paired 4,585-label vocabulary and raw Chinese whitespace,
  source-compatible saved-tag normalization and reviewed-state handling.
  The real source methods match over 16 presentation/filter/normalization
  profiles. Existing shapes survive single-image and Auto Run image tagging.
- [x] Source OpenCV uint8 INTER_LINEAR preprocessing for RAM/RAM++ through
  public `aicore_tagging_request_v2` and `aicore_tagging_infer_image_v2`.
  The original request and by-value return layout remain frozen; its inference
  path and qRAM retain official Pillow bilinear/antialias behavior. Both modes
  pass real image stride/channel, cancellation, owner/result lifetime and
  live-context shutdown checks. Invalid v2 headers/modes clear outputs.
- [x] Tag2Text caption updates the image description, persists through autosave
  and reload, and keeps tags separate from shapes. An empty caption clears
  the old description. This is an AICore extension, not a source ONNX claim.
- [x] YOLO-World per-box RAM descriptions reuse the same language/filter
  options and resident catalog-backed tagging context. Source integer,
  half-open crops, replacement on rerun, successful empty-result clearing,
  undo/redo, reload and stale-result rejection pass in actual-window tests on
  CPU, CUDA and Vulkan. A dedicated source composite preset and exact original
  detector/tagger checkpoint pair are still tracked above.
- [x] OCR uses the selected backend and source PP-OCRv4 defaults: 960-pixel
  max side, binary threshold 0.3, box threshold 0.6, unclip 1.5, recognition
  batch 6 and drop score 0.5. Cached layout files never enable layout; its
  explicit picker defaults off and recreates the resident session on change.
  Optional CDLA regions retain their real classes and reading order.
- [x] OCR detection replaces annotations, including clearing after an empty
  successful result. Skip Detection recognizes selected visible, unlocked
  rectangles/four-corner regions through the public quadrilateral API and
  changes descriptions only. Geometry, custom labels, confidence and unknown
  metadata survive autosave/undo/redo. The one-shot mode resets before the next
  normal Run. Each backend's real PP-OCRv4 English fixture yields 114 text
  shapes; explicit layout yields one region. Direction classification,
  arbitrary polygons, PP-OCRv5/v6 and KIE are still absent capabilities.
- [x] Save Visualization Video is enabled with OpenCV video support. Defaults
  match the source: 25 FPS (1..120), initially unchecked shape types, five
  overlay defaults from the canvas, Skip Empty Files off and the sibling
  visualizations-folder destination. Frame rendering uses a fresh export
  visibility state, source drawable-shape thresholds and bounded frame memory.
  Tests decode RGB/BGR colors, dimensions, FPS and frame count, drive the real
  options dialog and skip-empty behavior, and verify cancellation/failure
  preserves an existing destination and removes temporary files. Uniform,
  even frame dimensions are required by the native MP4 workflow. The full
  dialog theme/layout comparison remains open.

Model evidence: all 12 current OCR catalog weights across 15 scenarios and all
nine tagging catalog weights across 15 selected scenarios pass on **each** of
CPU/CUDA/Vulkan: **90/90 PASS**. Tagging includes nine official-preprocess rows
and six RAM/RAM++ OpenCV-preprocess rows. OpenCV truth is generated from the
actual pinned source `RecognizeAnything.preprocess` method and matching-GGUF
PyTorch/CPU arithmetic references. These are developer-profile accuracy and
repeatability checks (two processes, one warmup, two forwards), not controlled
speedup/release evidence or equality to the source ONNX checkpoints. The
separate RAM dataset/caption scenarios and the all-task release matrix remain
open. The tagging/validation Python suite passes 110 tests plus two subtests;
the strict repository-wide plugin/AICore boundary and C-API coverage gates pass.
Final current-tree qAnyLabeling regression: **32/32 PASS, zero failures or
skips**, 120.62 seconds, including the corrected composite replacement and
visualization-video workflow.

Evidence under `build_app/Testing`: `qanylabeling-tagging-upstream.json`,
`qanylabeling-{ocr-core,tagging}-full-{cpu,cuda,vulkan}.json`,
`qanylabeling-tagging-contracts-{official,opencv}.log`,
`qanylabeling-ocr-ui-{cpu,cuda,vulkan}.log`,
`qanylabeling-tagged-ui-{cuda,vulkan}.log`,
`qanylabeling-ram-runner-tests-final.log`,
`qanylabeling-ram-ocr-boundaries.log` and
`qanylabeling-render-ram-ocr-upstream/report.json`. The latter retains 36
passing render cases, 768 palette checks and 18 pixel-exact geometry images.
`qanylabeling-ram-ocr-acceptance.json` indexes the current evidence and explicitly
records `complete_upstream_parity: false`. The final plugin log is
`qanylabeling-ram-ocr-full-final.log`. Reproduction commands are in README.

Still open: the exact source model families and composite presets listed
above, OCR direction classification/PP-OCRv5/v6/KIE/multilingual checkpoints,
local VLM graphs and remaining ChatBot/VQA interactions,
training and quality tools, complete point-cloud camera/gesture behavior,
full-window visual consistency
and complete release/independent-source accuracy gates. Functional substitutes
and partial test matrices do not close these requirements.


## 2026-10-17 resources, remote Chat/VQA/VLM and window follow-up

Reference: X-AnyLabeling-GGML
`745f6c355f94cddb06f6965b3a4118f82033e935`. **The entire TODO remains open.**
No OCR runtime or qOCR changes are made by this follow-up.

- [x] Verify real RAM/RAM++ GGUF metadata and converter: `ram.tag_list` is
  English; `ram.tag_thresholds` owns inference thresholds. No Chinese list
  is embedded. Keep the paired 4,585-line resources solely for upstream
  English-name-to-Chinese presentation, including raw source whitespace.
  Reordered/custom/Tag2Text vocabularies use name lookup and English fallback.
- [x] Verify resource distribution: vocabularies, light/dark theme and panel
  styles, icons and three compiled translation catalogs live in Qt RCC,
  inside the centrally installed plugin. The actual generated Linux install
  block removes build RPATH and strips the plugin. Loading that installed
  binary, copying it to a Unicode path and changing the working directory
  preserves vocabulary hashes, translations and all live panel styles.
  No separate external resource-copy rule or developer path is required.
- [x] Enable real Chatbot and VQA windows with source Ctrl+1/Ctrl+2 actions,
  provider defaults and shared asynchronous OpenAI-compatible transport.
  Model listing, Bearer authentication, optional tokens, nonstream responses,
  split-UTF-8 SSE, HTTP/JSON errors, timeout and cancellation are covered.
  Settings survive provider changes and closing without a request. Local
  inference remains owned by AICore; service model IDs are remote configuration.
- [x] Chat image references and multi-turn streaming persist per-image
  `chat_history` role/content/image records. History reloads after navigation;
  base64 stays out of sidecars. Stop/import/export/clear and stale-source
  cancellation are implemented. Other annotation fields remain intact.
- [x] VQA implements QLineEdit, QRadioButton, QComboBox and QCheckBox schema
  types, source string/null/list values, component editing, manual autosave,
  text/widget/label/image references, AI preview/apply and current-image
  JSONL export. Source identity has its own generation so a model confidence
  change does not invalidate manual VQA edits. Navigation cancels old work.
- [x] Image Classifier AI Assistant and AutoRun now use the same remote
  VLM service for user-defined categories. Six MultiClass/MultiLabel prompts
  (including Unicode/quoted and empty categories) match the actual source
  generator exactly. Editable prompts, fenced/raw JSON boolean flags,
  single-image and two-image batch, cancellation, source changes, progress
  and malformed sidecar preservation pass actual-window mock-service tests.
  Unknown model categories are ignored, missing configured flags are false,
  and non-boolean responses fail without annotation changes. The Local model
  menu retains the tested AICore YOLO classifier and its shared batch/cache.
- [x] Dark window QSS and 48 palette roles match the source (with three
  explicitly recorded Qt 5 placeholder-role differences). Eight light/dark
  dock/panel/description/checkbox styles match actual source generators.
  Fix the canvas hardcoded light background: pixel samples now match dark
  `#1c1c1e` and light `#ffffff`. Remove the cold-start empty Relation Filter
  header and use the source Shapes sidebar title. Changing plugin theme is
  scoped to this workspace and its dialogs.
- [x] Point-cloud loading matches the source's background whole-frame
  contract, with bounded neighbor prefetch, cancellation and transactional
  commit. The UI cancellation test verifies event-loop responsiveness and
  preservation of the active frame/cuboids. Orthographic focus uses the source
  250 ms InOutCubic curve; perspective focus uses exponential decay to its
  tolerance and stops on camera input. Chunk/tile rendering is absent from
  the inspected upstream loader and is an optional optimization, not an
  unimplemented upstream streaming requirement.

Final current-tree regression: **33/33 PASS, zero failures or skips**,
150.67 seconds, including remote Chat/VQA/VLM, the real local YOLO classifier,
RAM/OCR consumers, paired-image matching and point-cloud UI cancellation/focus.
The strict repository plugin/AICore ownership check passes. All three language
catalogs compile 1,086 finished translations with no unfinished entries.

Current evidence under `build_app/Testing`: `qanylabeling-service-full-final.log`,
`qanylabeling-service-theme-upstream.{json,log}`, `qanylabeling-vlm-chat.log`,
`qanylabeling-resource-install.{json,log}` and `qanylabeling-service-boundaries.log`.
`qanylabeling-service-acceptance.json` indexes these results and records
`complete_upstream_parity: false`. Runtime tests use an in-process HTTP mock;
no remote-provider accuracy, credentials or availability verdict is implied.
The full-window snapshots are `qanylabeling-window-dark.png` and
`qanylabeling-upstream-window-dark.png`. Their font/DPI metadata are retained
in the probe output/log; native Qt 5 and source Qt 6 defaults differ.

Remaining acceptance work:

- [ ] Full window and all dialog screenshots, typography/DPI, layout,
  hover/selection/overlap states and complete point-cloud camera panels and
  gestures; palette/QSS/numeric fixtures alone do not close this.
- [x] Chat message edit/save/delete/copy/regenerate/resend and provider/model
  JSON import/export, eight provider defaults, search/favorites/selected flags.
  Native plugin and actual upstream method comparisons pass; delayed VQA close
  cannot overwrite newer Chat provider parameters.
- [ ] Complete Chat Markdown/code/math rendering, exact conversation/dialog
  layout and remaining visual states. Current
  history export remains a JSON array; ShareGPT dataset export is separate.
- [x] Chat directory-wide ShareGPT multimodal ZIP export. Multi-turn messages,
  portable image paths and original image bytes match the pinned source
  export method; invalid annotations abort without replacing an archive.
  The current source has no registered Alpaca dataset export.
- [x] Chat folder AutoRun with bounded concurrency and source CPU-count
  defaults, independent requests, failed-row counts, cancellation/source
  generation guards and sidecar field preservation. Native plugin tests cover
  two-image completion, API failures, cancellation and navigation.
- [x] ShareGPT multimodal dataset import with first-available-image semantics,
  source-version annotation templates, original-image copying and message-role
  preservation. Native output matches the actual upstream importer. Invalid
  later records and path escapes fail before writes; disk errors trigger an
  attempted rollback of this operation's files, not a crash-safe transaction.
- [x] New sidecars across editor/import/export, video masks, remote results,
  Chat and classification share the pinned source `4.1.3` default version.
  Existing normal-editor sidecar versions remain preserved.
- [x] VQA folder-wide selected-field/renamed-column JSONL, component JSON
  import/export, ordering and title/option/removal migration. Actual plugin
  modal-dialog tests cover two images, defaults, unknown fields and malformed
  sidecars. Export and migration fixtures match execution of the pinned
  upstream methods; this does not close the complete source workflow.
- [x] VQA source-default prompt templates, user-template add/edit/delete and
  automatic `xanylabeling_data/vqa/{components,prompts}.json` interoperability.
  Native tests verify restart loading, single selection, immutable defaults,
  malformed-file preservation and explicit component-config replacement.
  All 11 default name/content pairs match the pinned source configuration.
- [x] VQA ordered disk-sidecar references, Python-compatible nested scalar/
  list/dict formatting and sequential replacement match the actual pinned
  source method. The dedicated prompt, cancellable loading and result preview
  dialogs are implemented; tests cover malformed sidecars, image whitespace,
  overlapping references, cancelled requests and Escape closing.
- [x] Share provider/model JSON configuration between Chat and VQA, including
  source defaults, nullable parameters, unknown fields, malformed-file
  preservation and selected/favorite flags.
- [ ] Complete VQA window visual states and
  native Qt6/platform verification.
  Migration preflights parsing and saves each file atomically; a late disk
  write failure is reported and does not provide a whole-folder transaction.
- [ ] Remaining exact source model families/checkpoints and composites in the
  model ledger above, including local Florence2/VLM/Grounding and
  PULC/InternImage; remote user-category classification is not a local model
  graph. Training/quality tools and independent source-framework accuracy
  and complete release matrices remain open.
- [ ] Complete Linux installer inspection and native Windows/macOS builds,
  installs and relocations. Embedded-resource design and Linux plugin install
  validation do not replace these platform package gates.

## 2026-10-17 OCR model and point-cloud camera implementation

Reference remains X-AnyLabeling-GGML
`745f6c355f94cddb06f6965b3a4118f82033e935`. This section supersedes the earlier
blanket direction-classifier/v5/v6/KIE pending statements only within the
scope below. **Complete upstream parity is still open.**

- [x] Add an optional text-line direction classifier inside the existing
  monolithic AICore OCR task. Extend ABI v1 through `struct_size`, validate
  model roles/options and expose typed angle/probability/rotation per box.
  Both detection/recognition and selected-quad recognition classify after
  perspective crop, with source BGR normalization, aspect-sorted batches of
  six and strict `score > 0.9` rotation. Old callers retain their no-classifier
  behavior and task/runtime ownership.
- [x] Convert the exact source ONNX direction classifier, Chinese PP-OCRv4,
  Japanese PP-OCRv3, PP-OCRv5 and PP-OCRv6 Tiny/Small/Medium det/rec graphs to
  twelve F32 GGUFs. ONNX is an
  offline conversion/reference dependency; production inference stays in
  the shared GGML OCR graph interpreter. Preserve source/dictionary SHA-256,
  logical ranks, shape constants and exact topology; unknown operators and
  unsupported attributes fail conversion. Rec dictionaries are embedded in
  GGUF and need no installed resource files.
- [x] Expose the original task-owned v4 Chinese/English and six new pipeline recipes,
  source model names and source DB thresholds/candidate counts/batches.
  qAnyLabeling consumes the public recipes, keeps layout off, and defaults
  Text orientation on. New recipe selectors expose F32 only. Missing local
  unpublished weights show the shared cache path; no guessed download URL
  or duplicate plugin catalog is introduced.
- [x] Add independent ONNX Runtime graph checks, ten real upright/inverted
  word crops per new recognizer (including mixed width/order, 0.9/1.0
  thresholds, padded BGR stride, repeats and context lifecycle), and direct
  public-C-ABI versus unmodified source TextSystem checks on a real page.
  The page contains 74 retained texts for source v4, 77 for Japanese v3,
  81 for v5 and 73 for each v6 model;
  compare exact strings and quads, not just output counts. Source commits,
  ONNX assets and consumed image archive members have pinned digests.
  Thirty new local-GGUF scenarios live in the common validation manifest:
  twelve raw graphs, six ten-word orientation workflows, six source pages
  and six ten-polygon selected-region workflows.
  Every recognizer uses its own pinned ONNX/dictionary to generate the word
  expectations; Japanese recognition is not compared with English literals.
- [x] Preserve the exact source direct-quad crop for ONNX detection output;
  keep the original PaddleX integer min-area refit for published models.
  Support the source v4 dynamic attention shapes. Retain F32 columns in the
  ONNX CPU bounded depthwise fallback: on the real BGR page, v6 Medium's
  maximum raw-map error falls from 0.001048 to 0.000004 without changing the
  0.001 gate. Detection probes reject nonfinite probabilities and size mismatch.
- [x] Add public `aicore_ocr_recognize_regions` and route selected annotation
  recognition through it. Match source integer truncation, minimum-area
  polygon normalization, vertex ordering, reading-order batches and unclipped
  border-replicated crops. Preserve original result indices and UI geometry;
  retain low-confidence/empty texts. Cover triangles, pentagons, skew,
  fractional/negative and out-of-image points, malformed input and valid
  degenerate polygons against the actual source workflow.
- [x] Match PP-OCR KIE annotation import/export for entity labels and directed
  links, integer truncation of fractional/negative points, explicit null IDs
  and absent-ID defaults. A four-entity Unicode round trip preserves exact
  exported bytes. The inspected source has no active SER/RE model registration;
  its annotation workflow does not imply automatic KIE inference. AICore
  continues to reject the reserved nonempty KIE model option.
- [x] Move the camera image/projection into image-space QGraphicsView items.
  Match cursor-anchored 1.2 wheel zoom, absolute 0.02..32 limits, hand drag,
  double-click/explicit fit and manual-transform preservation on resize.
  Status text remains a viewport overlay and does not scale with the image.
  Add source-size hover navigation arrows, visibility for multiple cameras,
  cyclic switching and the source camera-arrow geometry/style.
- [x] Replace the bottom camera dock with the source's upper-right floating
  panel, responsive minimum/clamped geometry, lower-left resize handle and
  user-size request retained during the panel's lifetime, as upstream does.
  Add color radio buttons, slider values,
  independent reset buttons, frame depth legend and persisted overlay settings.
  Closing the panel closes its settings menu. Retain four decoded images with
  LRU eviction, and preserve manual zoom/pan when labels or cuboids change.
  UI tests cover these transitions; the numeric source comparison also checks
  the depth percentiles used by the legend.
- [x] Feed semantic/intensity/RGB/instance display colors and live creation
  previews from the main view to the camera. Keep instance zero visible when
  hiding numbered instances, clear previews on cancel/outside movement, retain
  shared point storage during overlay reads, and preserve zoom/pan on edits.
  Use source FastTransformation for photographs/point splats and one painter
  path per cuboid to avoid repeated blending at edge intersections. Fifteen
  executed-source camera scene comparisons match every pixel across scales,
  colors, sizes, opacity, selected/occluded boxes and the creation preview.
  This is a scene comparison; full-window fonts/chrome remain open.
- [x] Match orthographic edit-event boundaries in Top/Side/Front: consume the
  final release position, retain the initiating mouse button, ignore competing
  presses/releases, and finish at the last valid preview on Leave, outside
  motion/release or missing-button motion. Shift rotation snaps to 15 degrees
  with Python's ties-to-even rounding. Preserve source handle order and hit
  priority, resize/rotation cursors and active white 4-pixel handles; empty
  frames and locked boxes cannot begin edits. The source-executing gesture
  comparison covers 87 cases at `1e-10` tolerance, including rotated local axes
  and edit counts; the normal UI ctest also covers release-only movement,
  foreign release, Leave and snapped rotation in all three views. This does
  not establish full visual parity.
- [x] Broadcast an orthographic edit preview to the main GL view, both other
  side views and the camera overlay. Replace the existing box by ID instead
  of appending a second box. Other side views follow pure translations, while
  rotation follows the source camera-offset rule. Preview updates leave the
  document, undo history and point arrays unchanged. Cancel restores the
  original box, and release commits once. Real-window checks verify the shared
  preview value, changed GL geometry and exact GL restoration after Escape;
  four additional executed-source camera scenes cover move/resize/rotate/cancel.

Current local evidence is under `build_app/Testing`:
`qanylabeling-ocr-pointcloud-native-final.log` (33/33 tests, zero
failures/skips, formal build, no preload),
`ocr-formal-{cpu,cuda,vulkan}/qanylabeling-ocr-formal-{cpu,cuda,vulkan}.json`
(45/45 PASS per backend), `ocr-formal-snapshot.json` (32 source and ten binary
fingerprints, no preload), `qanylabeling-ocr-formal-regions-plugin.json` (18/18
worker runs), `qanylabeling-ocr-formal-polygon-window.json` (3/3 real-window runs),
`qanylabeling-camera-render-upstream.json` (15/15 scene comparisons),
`qanylabeling-cuboid-gestures-upstream.json` (87/87 event/handle comparisons),
`qanylabeling-ocr-converter-real-final.json` (all twelve source graphs accepted)
and `qanylabeling-ocr-extension-quant2.json` (retained failed F16/Q8 candidate
gates). The complete current-tree evidence is indexed in
`qanylabeling-ocr-pointcloud-acceptance.json`; inspect each verdict and count.
These are developer checks, not release performance or dataset accuracy claims.
The earlier `ocr-regions-*` and `qanylabeling-ocr-regions-plugin.json` results
used an isolated candidate runtime and remain historical evidence only; the
formal reports above replace them for the current build.

Remaining OCR/point-cloud acceptance:

- [ ] For each F32/F16/Q8_0 file, complete original-source probability/argmax,
  labelled-page Hmean/CER/WER, CPU/CUDA/Vulkan consistency, controlled GPU
  throughput/memory and same-host baseline before uploading to the existing
  `PP-OCR` release. Upload is already authorized. Then register the real
  URL/SHA/catalog/manifest and verify cold-cache automatic download. The local
  `ocr-extension-release-candidate/` ZIP is not a release; the public OCR catalog
  still contains 12 entries. Existing published OCR files also need the new
  per-file acceptance audit.
- [ ] Resolve the failed F16/Q8 numeric gates without weakening the F32 source
  contract; verify every claimed quantization/backend, dataset semantics and
  controlled release-performance baseline. Passing one graph/page is narrower
  than a complete OCR acceptance matrix.
- [x] Add independent original-ONNX execution with exact dequantized GGUF
  constants, so storage error and native graph error remain separate. Require
  exact recognizer/classifier argmax, multiple real inputs, matching source
  roles/dictionary SHA and the existing storage-specific numeric budgets.
  Save explicit mixed-precision policies and reproduce all 24 candidate SHA
  values (`quant-policy-replay-v7.json`). CPU/CUDA/Vulkan each pass 22/24
  multi-input quantized graph gates and 33/36 including orientation workflows.
  The classifier now includes five upright plus five 180-degree inputs;
  its F16/Q8 candidates pass on all three backends. Tiny recognizer F16/Q8_0
  remain failures, not supported releases. Mixed precision is reported by
  actual parameter counts: the classifier Q8 candidate is only 4.8% Q8_0.
- [x] Add SHA-pinned official 50-image scene validation and 50-page-per-language
  XFUND recognition to the common runner. Report raw detection Hmean,
  fixed-GT CER/WER and complete-pipeline CER/WER independently. XFUND field
  labels support recognition only, not detector Hmean or automatic KIE.
  F32 CUDA development matrix passes 42/42. This is not a controlled release
  performance or all-backend/all-quantization acceptance result.
- [x] Add 50 complete scanned receipt pages and their 2,779 original annotations
  from a pinned SROIE contributor corpus. The common input catalog verifies
  all 100 image/CSV SHA values and downloads them without private probe logic.
  Register all six presets x three quantizations as 18 additional scenarios;
  retain commas in transcriptions and bind label filenames/content into the
  directory digest. Dataset provenance and evaluation limits are recorded in
  `core/AICore/tests/ocr/DATA_CARD.md`; this subset is not official held-out
  competition certification.
- [ ] Finish all labelled-page quantization/backend runs and resolve every
  failed metric. v7 v4 Q8 Chinese/Japanese scene Hmean, v5 F16 complete-pipeline
  CER/WER, and Tiny F16/Q8 scene metrics fail despite several passing graph
  gates. Historical v7 results must not be assigned to later candidate SHA.
- [x] Independently revalidate v8 v4-det Q8 on the corrected production library:
  source probability gates pass on CPU/CUDA/Vulkan, Chinese/Japanese scene
  pages pass on CPU/Vulkan, and both scene and receipt combinations pass on
  CUDA (`filter-v8-*`). Conversion replay matches the tested file SHA.
- [x] Add the v5 real-page confidence/CTC regression crop to all three storage
  source scenarios and require five inputs. Old Q8 fails the new crop on all
  three backends. New v12 F16/v14 Q8 candidates pass five-input source gates
  on CPU/CUDA/Vulkan and CUDA scene/receipt pages; conversion replay is
  byte-identical. Record actual mixed storage: v14 is 2.10% Q8_0, 19.13% F16,
  78.77% F32, not pure Q8. These are not release/performance passes.
- [ ] Resolve Tiny quantization completely. The v15 recognizer F16 passes
  four-input graph gates on all three backends and CUDA orientation/receipts,
  but the original F16 detector changes a real scene contour: the combination
  loses 0.005540166 Hmean and adds 0.009803922 complete-pipeline WER. A separate
  v16 detector retaining conv2d_112 in F32 now passes all three graph backends
  and CUDA scene/receipt gates with v15 recognizer. Replay matches both SHA;
  other-backend pages and XFUND are still under test. Tiny Q8 source gates
  remain unresolved.
- [x] Correct source batch-resize rounding and preserve zero recognition
  confidence in the production C ABI. Native affected regressions pass 35/35,
  with zero skips. Add fast validation-runner, OCR metric/probability and GPU
  measurement-audit CTest gates; all four registered groups pass.
- [x] Apply source TextDetector's additional vertex ordering, width-1/height-1
  clipping and <=3 edge rejection only to source ONNX detector recipes.
  Preserve first-index ties even when they repeat a corner, and match OpenCV's
  singular-warp top-left sampling. A real scene exposed the previous phantom
  text; 13 source-generated filter fixtures and the singular-crop regression
  pass alongside the existing 400+ DB and 720+ crop goldens. All 12 affected
  OCR/table/plugin/validation CTests pass with zero skips.
- [ ] Revalidate complete page cohorts after the source detector-filter fix.
  Unfinished old CPU/receipt/v9 cohorts were stopped and recorded in
  `interrupted-before-source-detector-filter.json`; completed old reports are
  retained as historical evidence. The new `filter-v7-cuda` cohort records the
  library/probe SHA and progress separately from final metric JSON. Do not
  merge old and new production-library page rows into a release PASS.
- [x] Add strict GPU-activity auditing, source CUDA-provider graph benchmarks,
  explicit thread counts and allocation holds outside timed forwards. Reject
  missing telemetry and busy GPU measurements. Report sampled process-memory
  high-water values without claiming an exact allocator peak.
- [x] Retain each GPU audit under its unique ID, including rejected idle waits.
  Repeated common-runner attempts keep distinct raw records instead of
  overwriting previous telemetry. The four fast gate groups pass (runner 96,
  GT 7, probability 11, audit 8).
- [x] Record validation execution identities before inference and reject changed,
  removed or added command/runtime artifacts afterward. Reports include SHA
  values for selected executable/Python files, build AICore/ggml libraries,
  CMake options and the exact consumed manifest bytes. This is a bounded build
  identity record, not a transitive dynamic-loader trace. Runner tests now
  pass 98/98; all four fast CTest groups and an actual classifier common-runner
  CPU scenario (`artifact-provenance-cpu`) pass with unchanged artifacts.
- [x] Prevent reduced detector input sizes from qualifying as release speed:
  require source-default max-side 960 in the release probe, preserve that
  size for every additional probability input, and explicitly use 960 in all
  15 detector graph manifest rows. Default-size guards pass; historical
  max-side-320 graph results remain scoped to 320.
- [ ] Complete `default-size-accuracy-*` probability/stored-weight/backend
  gates at 960. Existing 960 page tests do not replace full probability maps;
  previous 320 graph PASS values cannot close the new default-size gate.
- [ ] Revalidate the new v17 Small-detector F16 SHA on every affected backend
  and page cohort. Default-size CPU initially passes 14/15 detector files;
  Small F16 source mean error is 0.000101013, above the unchanged 0.0001 gate.
  Retaining only conv2d_122 in F32 reduces it to 0.000082143 on CPU, and replay
  is byte-identical. This result must not inherit the old file's page results.
  The new file's graph, 50 scene pages and 50 receipt pages now pass 3/3 on
  CPU (`filter-v17-cpu`); CUDA and Vulkan scene/receipt checks also pass 2/2
  each (`filter-v17-pages-{cuda,vulkan}`). Remaining page and performance gates stay
  open. The default-size CPU/CUDA/Vulkan 36-row cohorts each retain the same
  old Tiny Q8 recognizer failure (35/36), which is not assigned to v19.
- [ ] Complete all page cohorts and release gates for the new v19 Tiny Q8 pair.
  Default-size detector probability, four-input recognizer probability,
  50 scene pages and 50 receipt pages pass 4/4 on each of CPU/CUDA/Vulkan
  (`filter-v19-cpu`, `filter-v19-pages-{cuda,vulkan}`), with
  byte-identical conversion replay. Calibration uses 256 pinned training
  crops, separately from the regression fixtures; document selection scope
  in DATA_CARD. Detector/recognizer contain only 0.23903%/0.57945% Q8_0
  parameters and remain explicit mixed-precision experiments. No speedup,
  public model registration or release qualification follows from this PASS.
  The explicit experimental conversion policy is now pinned in
  `ocr_quantization_policies.json` (SHA-256
  `3f8700eb6ab5672f396ff3c93b73c649ca480de4ec59cb0d2c14b3369d22ccc4`).
  Both Tiny files and the Small F16 file replay byte-identically; the shared
  model cache and product defaults remain unchanged.
- [x] Recheck the existing 12 published OCR files through an empty cache and
  offline preflight. `public-cold-download-0zhcaspv/report.json` records all
  pinned hashes and both preflight passes. Current graph/layout developer
  matrices pass 15/15 on each backend (`public-current-{cpu,cuda,vulkan}`).
  Original Paddle detector/Chinese/English recognizer parameters match their
  GGUF source identities; fresh Paddle probability fixture replay passes 3/3.
  These checks do not close the new real-page and controlled-performance gates.
- [ ] Finish release-profile repeated same-host A/B on an idle GPU. The three
  baseline and three candidate classifier attempts were all rejected for
  external GPU activity and are retained as failures. The source CUDA
  I/O-binding smoke check ran successfully (max CPU/CUDA error 0.000016362),
  but shared-GPU smoke timing is not controlled performance evidence. No
  throughput/memory release PASS or new upload is claimed.
  A fresh cohort on the corrected library also fails: some native samples
  pass, while source measurements detect desktop or external RelateAnything
  activity. Its finally block resumes the paused page scheduler; the record
  is `controlled-cls-f32-current-cuda/matrix.json`.
- [x] Add independent original-Paddle graph consumers for all nine published
  detector/Chinese-recognizer/English-recognizer quantizations. The original
  program/parameters and dictionary YAML are SHA-pinned runner inputs;
  `public-paddle-graphs-v4-{cpu,cuda,vulkan}/matrix.json` each pass 9/9,
  including direct native GPU/CPU comparisons. Source crashes,
  changed argmax, nonfinite output and failed GPU audits fail closed.
- [ ] Finish the new original-Paddle public page and GPU performance gates.
  Six hard page/crop fixtures pass; current Chinese/English F32 and F16 street
  cohorts pass. The public Q8_0 Chinese/English combinations fail complete
  pipeline CER deltas (0.009708738 / 0.005295675, gate <=0.005). Remaining
  cohorts are still running. Paddle GPU 3.0.0 exit-time heap corruption was traced to
  inherited GLOG_log_dir, and the isolated source child now removes application
  logging configuration. Three source GPU graphs finish normally. Replacing
  unrecognized TF32 flags with NVIDIA_TF32_OVERRIDE=0 also restores strict CPU
  agreement (det/ch-rec/en-rec maxima 0.000017583/0.000006557/0.000000238).
  These shared-GPU smoke results are not controlled performance evidence.
  Layout original-source results are recorded below, including its failures.
- [x] Add the eight-head original PicoDet graph consumer and native head
  export/nonfinite rejection. All source/config/fixture files participate in
  automatic download and offline verification. F32 passes CPU/CUDA/Vulkan;
  a real native NaN fixture is rejected. Public Paddle probe tests pass 27/27.
- [ ] Repair published layout F16/Q8_0 source argmax mismatch. In
  `public-layout-original-zfzv7xec/{cpu,cuda,vulkan}-v2`, both quantizations
  fail the new exact argmax gate on all backends (worst-head CPU agreement
  0.992307692 / 0.969230769). Native backend-to-CPU agreement passes; the
  existing probability-only budgets cannot close this source-output gap.
- [x] Share Paddle source-process isolation and add verified page-reference
  replay. Require successful source teardown, pinned export SHA, unchanged
  framework library/dependencies/code/config/GT/pages and the same language,
  evaluation mode and threads. Native inference remains per quant/backend.
  The old v2 page cohort was stopped for CPU source exit-time corruption;
  keep its failure record. New v3 Chinese/English street source exports each
  finish 50 pages normally; remaining native/source page checks are pending.
- [x] Attribute the public Q8_0 street failure one stage at a time, with the
  unchanged complete 50-page GT. Q8 detector + F32 recognizer fails both
  languages (e2e CER delta 0.011473963); F32 detector + Q8 recognizer passes
  both. Evidence: `public-q8-stage-attribution-194al707/completed.json`.
- [ ] Repair the public Q8 detector under the original full-page defaults.
  Keep the failed published SHA as evidence; candidate files are isolated
  experiments and must not replace public assets before all release gates.
- [x] Fix graph-reference dependency identity: Paddle imports vendored package
  metadata, so last-write-wins enumeration recorded typing_extensions 4.12.2
  instead of selected 4.16.0. Record normalized selected versions and all
  candidate versions, hash the helper, and reject changes during execution.
  The active v3 page cohort retains its old identity implementation; it cannot
  be claimed to have the stronger dependency identity check retroactively.
- [ ] Retry controlled performance when the measured GPU meets the idle gate.
  `controlled-classifiers-current-v2-cuda` fails baseline idle acquisition after
  draining this session's work: gnome-shell remains about 27% SM. Raw rejected
  audits and interrupted.json are retained; the cohort was interrupted and
  its finally block restored all own page queues. No complete matrix or
  Vulkan performance result is assigned to this attempt.
- [x] Recheck published SLANet graph consumers on the current build:
  `current-table-graphs-aenvh0zm/{cpu,cuda,vulkan}/matrix.json` each has three
  PASS rows, bound to unchanged native/library artifacts. This is one-repeat
  developer numeric evidence; table TEDS and controlled release performance
  are separate, unfinished gates.
- [ ] Finish the per-file release ledger for all 51 files (48 text/layout/
  classification and three SLANet table files). The read-only evidence index
  keeps historical matching-weight graph results separate from current
  before/after build-identity evidence. Fresh 36-model graph rows are recorded below;
  no absent row or missing page dataset is assigned PASS.
- [x] Finish the fresh 36-candidate developer graph evidence: CPU has 36
  PASS rows, CUDA's first/retry reports together contain 36 distinct PASS
  rows, and the Vulkan retry matrix passes all 36. Preserve the first
  INCOMPLETE reports and their VRAM skips; this is not a release matrix.
- [x] Apply selected-version/all-candidate dependency identity to the Paddle
  page exporter too. Reject old reference receipts and changed helper/code
  or dependencies during export. The reference tests pass 31/31. Own v3
  page/repaired-CPU queues were interrupted with an explicit receipt; old
  results remain available and do not inherit this stronger identity.
- [ ] Run the newly pinned, independent SROIE receipts 050–099 through all
  six public Chinese/English quantization workflows. They are disjoint from
  the policy-selection/regression receipts 000–049; 100 source files and the
  combined label-directory SHA are mandatory common-runner inputs. Keep the
  repaired SHA identities separate, and do not tune on this acceptance set.
- [x] Check DFL regression argmax per independent eight-bin edge as well as
  the existing full-channel argmax; export the corresponding softmax error.
  A near-tie test proves that an unchanged global argmax cannot hide a changed
  second edge. The reference probe suite is now 28/28 PASS.
- [ ] Repair public Q8_0 against both scenes and receipts. Four of fourteen
  detector fallbacks pass both languages on the existing scene regression;
  two also pass Chinese SROIE. English SROIE still fails fixed-region WER
  (delta 0.006833713) independently of the detector, so an English recognizer
  repair is also required. Candidate files stay isolated and unpublished.
  The converter now keeps detector `conv2d_397.w_0` and English recognizer
  `conv2d_161.w_0` F16. The second fallback passes the 50-page English SROIE
  CUDA and scene CPU regressions with the detector candidate. Repeated
  conversion of both files is byte-identical; pending hashes are `cab74270…`
  (detector) and `695bdcbb…` (English rec). Actual Q8_0 parameter proportions
  are 66.069% / 43.398%. Current public catalog/digests stay on the published
  files until the pending files finish every gate. A separate local-only
  manifest runs their source graphs and complete page workflows through the
  common runner while verifying all original published dependencies.
- [ ] All OCR-window visuals. The six active
  source OCR preset checkpoints are now available as local F32 recipes;
  unpublished artifacts are not public catalog/download support.
- [x] Match upstream orthographic point visibility and crop/expand controls:
  share the main viewport's filtered point indices, default depth cropping on,
  crop Side/Front only with the 0.1 m margin, skip crop in Top/create mode, and
  hide the perspective view while expanded and restore the three-view splitter
  sizes afterward. Side views also use
  the active semantic/instance/RGB/intensity point colors and match upstream's
  device-pixel point-size rule. Native raster tests cover crop boundaries,
  per-point color/size and window state transitions.
- [ ] Complete point-cloud handle-icon and panel/dialog visual parity,
  remaining keyboard/gesture/overlap
  states and full-window screenshots.
  Current camera navigation/zoom and numeric projection checks cover only
  their recorded paths. Chunked point-cloud streaming remains an optional
  optimization: the inspected source loads whole frames asynchronously.
- [ ] Native Windows/macOS builds, installation/relocation and full Linux
  package inspection. New OCR dictionaries live inside GGUF; existing RCC
  resources remain in the plugin. This design does not replace native package
  verification. Chat/VQA advanced workflows, local VLM/remaining models and
  training/quality tools stay open in the preceding ledger.


### Dedicated PaddleOCR document workspace (partial, 2026-10-19)

- [x] Open an independent workspace from the upstream `Ctrl+4` action. Keep
  local typed AICore OCR annotation inference in the model picker.
- [x] Add the official async-jobs transport with the upstream model/default
  payload, streamed multipart image/PDF upload, polling, JSON/JSONL results,
  deadlines, cancellation and request-generation isolation. No model asset
  ownership or local inference enters the transport.
- [x] Add upstream-compatible on-disk records, favorites/search/status,
  document/page/block selection, zoom, duplicate block keys, corrections and
  JSON/Markdown export. Preserve unknown fields and use atomic JSON saves.
- [x] Export document ZIPs with upstream Markdown order/names, corrected text,
  cropped image assets and per-page layout previews. Reuse the streaming ZIP
  writer for OCR and video datasets; preserve atomic replacement and video ZIP
  root-directory behavior. An independent Python decoder checks CRC, JPEG
  dimensions and content; the two-page Markdown fixture matches upstream.
- [x] Apply persisted API URLs, image/PDF type filters and newest/oldest sort.
  Exercise settings through the actual modal dialog and a local Qt HTTP
  service; changing the selected record must not redirect an in-flight result.
- [x] Isolated Qt 5 window/service tests and the real plugin menu/window probe
  pass. Keep their evidence separate from release OCR model qualification.
- [x] Remote service discovery/model refresh and generation-scoped remote
  result-image caching; test service settings and selection changes through
  actual Qt modal dialogs.
- [x] Specialized OCR table token/HTML/TSV editing, fonts, spans,
  merge/split, row/column insert/delete, default 3x3 creation, clipboard,
  and bounded undo/redo. Native document tests verify Save/Cancel, atomic
  persistence and preservation of unknown block fields.
- [ ] Native PDF rendering, formula/rich-text editors, complete table and
  full-window visual/gesture states, translation coverage and clean-package
  installation checks.
- [ ] Advanced VQA/Chat and remaining local model catalogs are still open.


### 2026-10-19 current worktree evidence

- [x] Preserve all 38 OCR release candidates in one SHA-verified staging
  directory; distinguish the 36 new files from the two public Q8 repairs.
- [x] Complete current-library source graphs for the two public Q8 repairs
  on CPU as well as CUDA/Vulkan. Their seven composed page scenarios pass
  on all three backends; English rec is consumed by three of those scenarios.
- [x] Compile the complete HTML5 entity mapping into the OCR table editor,
  with a reproducible generator and no installed-resource dependency.
  Native/source comparison passes 5,407 cases; document Save/Cancel
  preserves unknown fields. Compare both parses for source NBSP behavior.
- [x] Match table default creation and paste as two undo steps, extra Python
  splitlines separators, formula-dispatch priority and context action labels.
- [x] Match table column/row geometry, format-button selection state and
  right-click multi-selection restoration. Six native geometry cases execute
  the pinned source methods with the same runtime font metrics.
- [x] Match the table icon toolbar, top Cancel/Save order, body margins and
  light/dark QSS; explicitly inherit the workspace palette. Two theme cases
  compare the actual source stylesheet methods. Preserve standard format,
  save, undo and redo shortcuts without ambiguous duplicate bindings.
- [x] Replace the modal table dialog with a block-local QFrame editor and
  source viewport-limited height. Six height cases execute the actual source
  methods. Save/Cancel, unknown JSON fields and a queued stale Save after
  document selection changes pass through the real document window.
- [x] Verify Chat/VQA shared provider persistence, deferred-window destruction,
  Anthropic model-list headers, 45 vision capability cases and 1,060 model
  search cases against actual source methods. Five upstream comparison
  scripts and four actual plugin/document/video regressions pass.
- [x] Finish the current-library 54 new-model source page scenarios on
  CUDA/Vulkan (54/54 each) and SLANet semantic pages (12/12 GPU rows). CPU
  pages are retained as diagnostics and no longer gate GPU release. These
  developer results do not complete release repeat/performance qualification.
- [ ] Finish release repeat/stability, controlled CUDA/Vulkan same-host
  throughput/VRAM A/B, layout accuracy repair and independent GT, and
  upload/catalog/digest/manifest/cold-download closure. No newly uploaded
  model is qualified yet.
- [ ] Finish remaining table/full-window visual states,
  formula/rich-text and PDF, advanced Chat rendering/model refresh and full
  visuals, remaining exact local models, and Linux clean-package.


### 2026-10-20 remote synchronization and GPU-only acceptance

- [x] Fetch all remotes/tags and merge both `origin/users/dahailu/v3.9.5`
  (`5d1486089`) and `origin/main` (`7bc6520bd`) into the current branch.
  Merge commits are `1688a2896` and `b9a739ce1`. Resolve the command-map and
  validation-manifest conflicts; preserve all 241 original staged paths and
  the safety stash `7a8d0cd5`. No unmerged index entries remain. Both remote
  tips are ancestors of HEAD; the manifest retains 289 unique scenario ids.
  A later external pull advanced the branch remote to `5a6a40c06` and HEAD
  to `8db589b2d`; both current remote tips remain ancestors and there are no
  unmerged files. The additional six upstream files do not change AICore or
  AnyLabeling. ACloudViewer, QANYLABELING_PLUGIN and QOCR_PLUGIN rebuilds pass.
- [x] Remove native CPU comparison metrics from the 48 OCR graph release
  gates. Keep original-framework/stored-weight truth, GPU precision checks
  and strict CUDA/Vulkan performance audits. GPU comparisons consume the
  actual original CUDA output when its interpreter is supplied, including
  accuracy-only developer runs. Tests: 31 probe tests and 100 runner tests.
- [x] Finish the old OCR page and restored SLANet queues with unchanged
  library/probe identities. Preserve every failure and incomplete old attempt.
- [x] Prune four obsolete reference environments/wheel caches only after
  checking live command lines, mappings and open files. Free 15.21 GiB;
  preserve current environments, candidate GGUFs and validation evidence.
  Cleanup plan, removed-file inventory and receipts are under
  `build_app/Testing/ocr-storage-reference/cleanup-20261020/`.
- [x] After the final container preflight, retain its complete file/SHA and
  symlink inventory and command receipts, remove the disposable container and
  its generated payload copies, freeing another 1.65 GiB (16.87 GiB total).
  Preserve source builds, the rebuilt OCR plugin, models and test outputs.
- [x] Add an inline rich-text block editor with Markdown/HTML round-trip,
  H1-H6, bold/italic/strike, heading/body Enter behavior and Save/Cancel.
  Thirteen serialization and six heading-metric cases execute actual upstream
  methods; native window tests cover atomic persistence, unknown fields and
  stale Save after changing documents. Actual plugin/document/video tests and
  the same plugin's Chat/VQA regression pass. Qt-version-sensitive APIs go
  through QtCompat; the GPL source adaptation is attributed explicitly.
- [x] Match twelve clipboard cases against actual upstream MIME sanitization,
  block insertion and heading decisions. Ignore local image URLs, strip inline
  images without losing text, restore pasted headings, and reuse an existing
  blank block on Enter after an image.
- [x] Display image-block crop previews from `_ppocr_meta.block_image_paths`,
  preserving duplicate-key fallback and excluding text blocks. Three preview
  sizes and six viewport/image-height cases match actual upstream methods;
  the actual document window also loads the preview and cancels the editor.
- [x] Match the rich editor's container, body, scrollbar, format buttons,
  heading active/inactive and Save/Cancel QSS against the actual upstream
  functions in both light/dark themes. Stop inheriting the list selection's
  blue background; preserve source toolbar/body margins and spacing.
- [x] Match the custom six-row heading popup, selected/hovered states,
  Escape/button dismissal and source-drawn heading/bold/italic/strike icons.
  Both theme cases execute the upstream selection methods. Match embedded
  image click selection and its padded blue frame; two cases verify viewport
  geometry against the source before/after scrolling, cursor/text preservation,
  ordinary click dismissal and stale-frame removal after deleting the image.
  Latest native source comparison: `rich-dropdown-reference-s8zurz7a/` under
  the storage-reference directory. These mini-probe results precede the final
  merged-branch plugin rebuild.
- [ ] Complete full-window comparisons, formula rendering and native PDF
  rendering. Current Qt5 tests do not certify Qt6.
- [ ] Qualify controlled performance: the latest CUDA/Vulkan release repeats
  were rejected before timing by real desktop GPU SM activity, so no speed
  or VRAM qualification was produced. Keep the audit thresholds unchanged.
- [x] Complete original CUDA truth reruns: new ONNX models 36/36 on CUDA
  and 36/36 on Vulkan; public Paddle graphs 10 PASS / 2 FAIL per backend.
  Layout F16/Q8 remain strict argmax precision failures on both GPUs. Fix
  the virtualenv interpreter symlink bug and eight-head probe export before
  rerunning; retain the earlier tool failures separately.
- [x] Repeat all 96 graph rows after rebuilding the merged runtime
  (SHA-256 `9f2548ef28f6edadbcdbb0658d6fbf67e18dbd73ca2afb28183c2b2e9af59f21`):
  each GPU has 36/36 ONNX PASS and 10 PASS / 2 FAIL Paddle. Only layout
  F16/Q8 fail. Receipt: `merged-gpu-truth-20261020-awcjlpee/finished.json`.
- [x] Finish the fresh 120-scenario labelled-page/SLANet queue against this
  merged runtime: Vulkan 60/60 PASS and CUDA 59/60 PASS initially; the sole
  CUDA identity-only failure passes its separate unchanged-condition retry.
  These page comparisons retain the original CPU source and are developer
  evidence, not the pending original-CUDA page cohort. Old 54+6 per-GPU page results belong to runtime
  `154a0b62460edbe0273deeb17f766bd0d949aefc63a4782b0d9c3251665105cf`
  and remain historical evidence; do not inherit them across library SHAs.
  Preserve the Japanese XFUND F32 cache-identity failure: its CER/WER deltas
  are zero, but CMake reconfigured during the row. The original identity-only
  failure receipt is retained. Formal SLANet source-graph consumers also pass
  3/3 on each GPU, with unchanged build identity. Controlled release sentinels
  remain separate. Follow-up receipts use
  `merged-gpu-followup-path.txt`; a failed receipt is never overwritten.
- [ ] Finish layout repairs/independent GT, release repeat stability and
  eligible uploads, then verify catalog SHA,
  manifest ownership and automatic cold downloads. Local VLM and Linux
  clean-package remain open; the stale build CPack config is a TBB package
  and must not be presented as an ACloudViewer installer.

- [x] Reassess the packaging claim against actual GitHub focal jobs and logs.
  Ubuntu run 39273926284 generated and installed the focal artifact; headless
  tests have 126 PASS / 1 skip and JSON-RPC has 138 PASS. CUDA focal run
  39273926321 also passes build/package/install checks. The local Oct 14
  zero-byte installation is stale local evidence, not a remote CI failure.
  The later Ubuntu run 39372646030 at `9952ac8d` also completes successfully:
  all three distributions build/package/install/test; focal again records
  126 headless PASS / 1 skip and 138 JSON-RPC PASS. Preserve its actual logs
  separately from the earlier run and current uncommitted worktree.
- [x] Restore all packaging-related worktree/index changes to synchronized
  HEAD, including CMake/PostInstall, bundlers, launchers/installer scripts,
  CI plugin switches and build documentation. Remove the added staging helper
  and four packaging test files after preserving exact backups and patches.
  Receipt: `packaging-reassessment-20261020/restoration.json` under the
  storage-reference directory. Earlier 12-test/local-payload results are
  historical investigation and do not certify changes that were rolled back.
- [x] Cancel the local installer and dependent clean-container queue using
  the withdrawn scripts. Continue the formal plugin/cache tests and OCR GPU
  queue independently; do not infer model accuracy from installer CI success.
- [ ] Validate future submitted plugin/model changes through the existing
  unchanged CI build/install workflow. Historical successful artifacts do not
  include the current uncommitted worktree; do not reuse the stale local
  installation as a package acceptance result.
- [x] Replace the three common `table-structure-*` fixture-only consumers with
  independent original Paddle graph comparisons, retaining 289 unique scenarios.
  Pin the official SLANet archive and all three source members. Normalize its
  leading `./` member paths in the shared extractor while rejecting ambiguous
  names, links and path escapes; 102 runner tests and the actual shared download/
  extraction pass. Table/Paddle probe suites: 27 tests pass.
- [x] Run those source-graph consumers on CUDA/Vulkan for F32/F16/Q8_0 using
  the uniformly compiled isolated probe: all six scenarios pass, three inputs
  per scenario and exact EOS-inclusive argmax. Receipts are under
  `slanet-common-gpu-path.txt`. The formal CMake-target reruns subsequently
  pass all six consumers in `merged-gpu-followup-path.txt`; controlled release
  performance and original-CUDA page references remain separate pending gates.
- [x] Verify one existing published OCR F32 file through an empty, space-containing
  shared data root: OCR/table C ABI cache paths agree; cold download and SHA,
  offline reuse and corrupt-offline rejection pass. Do not extend this evidence
  to unpublished candidates or plugin UI download interaction. Receipt:
  `ocr-shared-cache-cold-path.txt`.
- [x] Exercise the actual shared Qt downloader on the same published OCR file
  with a fresh space-containing data root. Preserve the reproduced stale-ledger
  failure; fix the shared integrity service to keep rejecting changed pinned
  files across repeated cheap/empty-pin queries and failed deep checks. qOCR
  now uses that service for model presence. Cold download/SHA/ledger, warm reuse,
  corruption rejection and restored deep verification pass with unchanged
  executable/library identities. Formal shared-service/TRELLIS compatibility
  and seven AnyLabeling regressions pass (9/9); AICore/OCR probe SHA values are
  unchanged. The installer queue was canceled after packaging restoration;
  existing CI must validate future submitted plugin changes.
- [x] Align Chat/VQA model-discovery triggers and per-provider refresh decisions
  with the pinned source: initial window, provider change and menu open;
  Ollama always queries; cached other providers use the source call tracker;
  explicit refresh bypasses it. The source's first-interval reset keeps timer=0,
  so later calls refresh again; preserve and test that actual behavior rather
  than claiming a rolling five-minute cooldown. Fifteen controlled-clock cases
  match execution of `get_models_data` and `EventTracker`.
- [x] Keep model discovery asynchronous and independent of completion requests,
  using two instances of the existing transport and the existing models.json
  owner. Real plugin HTTP tests cover cache hits, forced refresh, Ollama,
  provider switching, first empty-menu population and concurrent completion/discovery; native workflow and
  source comparison pass, followed by 2/2 formal Chat/plugin regressions.
  Receipts: `chat-model-refresh-path.txt`. Chat Markdown now uses QtCompat;
  hover dispatch uses the shared Qt5/Qt6 event API. Qt6 visuals remain unverified.

- [x] Match model-popup placement to the actual source method: centered on
  the selector, resized to the settings panel width and the space above it.
  A real plugin window exports its geometry and the comparison executes
  `ChatbotDialog.show_model_dropdown`; light/dark native workflows and 2/2
  formal Chat/plugin tests pass. Receipt: `chat-model-position-path.txt`.
  This closes the positioning rule only, not the complete conversation layout.

- [x] Pin all 1,000 official CDLA validation pages and their 8,711 human-labelled
  regions, including archive/member SHA values and source converter versions.
  Execute the actual official converter/rasterizer and confirm identical
  classes, raster-mask areas and bounding boxes. Use positive COCO annotation
  IDs to avoid its zero/unmatched sentinel. No training-set tuning is involved.
- [x] Add independent original-Paddle and public-C-ABI layout region exporters
  plus class-aware COCO page evaluation. Seventeen manual unit tests pass;
  the original CPU export completes all 1,000 pages and three native CPU pages
  verify the wiring only. CPU results are diagnostic, never GPU release gates.
- [x] Verify the shared cold archive download, all 2,000 pinned-member checks
  and zero-download offline reuse through the existing common downloader.
- [x] Complete the original CUDA reference and all six native
  CUDA/Vulkan × F32/F16/Q8_0 exports on the same full 1,000-page selection;
  every process exits normally. Original CUDA mAP50:95 is 0.8546702421.
  Native CUDA/Vulkan F32 deltas are -0.0000003951/-0.0000004017;
  F16 deltas are +0.0000631982 on both, Q8_0 -0.0009133202 on both.
  Receipt: `layout-cdla-source-path.txt` (`gpu-finished.json`). Formal
  `test_ocr_layout_gt_eval` passes after reconfiguration. These measured page
  deltas do not remove strict raw-output failures or qualify performance.
- [ ] Finish independently reviewed layout page-gate tolerances and common
  manifest consumers; preserve the existing strict raw-output failures.


- [x] Reproduce the remaining source-CPU gate in GPU graph validation: the
  native probe consumed CPU reference arrays before the original CUDA check,
  so a later CUDA match could not clear the earlier CPU failure. Prepare an
  isolated correction for original Paddle/ONNX, stored-weight ONNX and the
  direction-classifier/CTC source path. Preserve all SHA, probability and exact
  argmax requirements. Thirty-seven isolated probe tests pass, including
  CPU/CUDA near-tie disagreement and provider fallback rejection; CPU original/
  stored-weight and orientation/CTC exports complete only as wiring checks.
- [x] Apply the reviewed CUDA-reference correction after the frozen graph/table
  queues. All seven production probe/runner suites pass (184 tests including
  the 17 CDLA evaluator tests); no native runtime/probe binary changes.
  Receipt: `gpu-truth-selection-path.txt`.
- [ ] Rerun the common CUDA/Vulkan graph, orientation and page consumers
  with original CUDA truth for every accuracy input. Graph/orientation is now
  complete (128/132 pass); pages remain running. The formal CDLA CTest passes.
  Production unit tests are not GPU acceptance; layout F16/Q8 strict CUDA
  failures remain unresolved.

- [x] Extend the isolated CUDA-reference correction to the labelled-page
  probes: source ONNX uses the supplied CUDA interpreter, verifies provider
  selection and normal subprocess exit; public Paddle uses the exact original
  CUDA package and rejects replayed CPU predictions for GPU rows. Keep human
  GT and the existing Hmean/CER/WER thresholds. The 66 ONNX-page, 21 Paddle-page
  and 18 orientation manifest consumers receive explicit CUDA reference paths.
  All 167 isolated probe/runner tests pass. Download/verify the 100 additional
  held-out SROIE files through the existing common cache before queued reruns.
- [ ] Validate the new page reference execution on actual CUDA/Vulkan; the
  older completed page queue uses its original CPU source reference and remains
  a separate historical/developer cohort. Its original-CUDA-reference replacement
  is now running. Complete all page runs before claiming GPU-only acceptance.

- [x] Prepare exact-SHA GPU reruns for the two unpublished public Q8 repairs
  using the current common manifest: two original graph rows and seven human
  labelled/held-out page rows per backend. Offline shared-cache preflight passes
  with original Paddle CUDA references; no replayed CPU predictions or changed
  thresholds. Receipt: `public-q8-gpu-truth-path.txt`.
- [ ] Finish the queued CUDA/Vulkan repair rows after the complete current-model
  GPU queue. Keep pending model SHA values separate from published originals;
  successful preparation does not qualify a new file for upload.


### 2026-10-20 Chat source panels and input workflow

- [x] Replace the single-column Chat settings layout with the source's three
  columns: 200..250-pixel provider panel, expanding conversation and fixed
  360-pixel preview/settings panel. Reuse the same provider configuration and
  transport. Match the Backend/Generation tabs, 0..20 temperature slider,
  model-popup anchor, image navigation/open actions and source constraints.
- [x] Embed 21 original light/dark QSS pairs as generated compile-time data;
  record all three source SHA values and reproduce the file with
  `tests/generate_chat_styles.py`. There is no installed Python or source-path
  dependency. Verify actual workspace theme actions, panel background and
  provider foreground colors against source theme constants. Defer scoped
  overrides until workspace stylesheet propagation completes.
- [x] Match the plain-text QTextEdit input frame, 20..300-pixel dynamic height,
  scrollbars, first @image highlight, Enter send/Shift+Enter newline and
  settings-field Enter suppression. Empty input disables send; generation
  uses the same button to stop, disables the input, and preserves cancellation
  and stale-source guards. Clear conversation defaults to No and preserves
  all other sidecar fields after Yes.
- [x] Actual plugin workflow and nine source layout/theme comparisons pass.
  Six formal Chat/plugin/OCR-document/table/rich/shared-integrity CTests pass
  after integrating the exact tested objects and plugin. Frozen OCR execution
  artifacts remain byte-identical. Receipt: `chat-three-columns-path.txt`.
- [ ] Finish conversation message/code/math rendering and full-window visual
  states, typography/DPI and Qt6/native-platform checks. These source-structure
  and selected-color checks do not certify complete visual parity.


### 2026-10-20 Chat message presentation and GPU failure attribution

- [x] Embed 12 additional original light/dark message style pairs (33 total),
  including bubbles, assistant role header, actions and editor controls. The
  generator verifies the pinned source; no runtime Python dependency is added.
- [x] Match user/assistant context-menu action order and source 1,000-ms copy
  feedback. Menus call existing message operations and honor disabled actions;
  owned menus and timers survive source switching without stale mutation.
- [x] Actual plugin regression verifies source theme surface/text colors,
  role icons, menu copy/edit, timer reset and deleting the open-menu message
  during image navigation. Twelve independent source layout/theme/menu checks
  and all six formal shared-integrity/Chat/OCR-window/plugin CTests pass.
  All 28 frozen OCR execution identities remain unchanged. Receipt:
  `chat-message-source-path.txt`.
- [x] Complete the original-CUDA graph/orientation cohorts: 128/132 pass;
  only layout F16/Q8_0 fail (two per backend). Read-only retained-tensor
  comparison finds zero CUDA/Vulkan argmax differences on all eight heads.
  Both backends have the exact same original-source disagreement masks:
  F16 22 class + 40 edge flips; Q8_0 253 + 419. Maximum backend tensor
  difference is below 0.000021. Receipt: `layout-backend-attribution.json`.
  This argues against a backend-specific defect, without replacing the
  strict failed gate or qualifying a quantization repair.
- [ ] Finish all original-CUDA-reference page cohorts, queued public Q8
  repair rows, strict layout repair and controlled performance. PDF/formula,
  Markdown syntax highlighting/math, local VLM and full-window/platform
  acceptance remain open; no model is qualified or uploaded.


### 2026-10-20 VQA navigation, clear and refresh

- [x] Match 1-based page input, first/last navigation availability, empty-input
  restoration, maximum clamping and Enter jump without triggering another
  dialog button. Match 68×32 page input, 32×32 navigation and 28×28 sidebar
  controls. Preserve collapsed image-panel state across page switches.
- [x] Add the source's default-No Clear All confirmation for the current image.
  Reset all four configured component types through the existing value writer;
  cancel pending answers and retain unknown VQA/sidecar fields. Execute the
  original clear/value/save methods independently against the native fixture.
- [x] Add both source refresh controls; show the right-side refresh/sidebar
  controls while the image panel is collapsed. Reuse the owning workspace's
  current-file load path for disk refresh and source-generation invalidation.
  A real external JSON edit is reloaded without rewriting unrelated fields.
- [x] Native workflow and independent source page/default/clear comparisons
  pass; all six formal plugin/OCR-window/Chat/shared-cache CTests pass after
  integration. All 28 active OCR execution identities remain unchanged.
  Receipts: `vqa-navigation-source-path.txt`, `vqa-refresh-source-path.txt`.
- [x] Prepare a serial controlled GPU retry using the unchanged common runner,
  original CUDA references and preserved same-host baseline. All eight release
  command preflights pass, including the two exact-SHA public Q8 repairs;
  actual measurement waits for the active page and
  public Q8 repair queues. Receipt: `gpu-controlled-retry-path.txt`.
- [ ] Complete the full VQA layout, component/dialog styles, fonts/DPI and
  Qt6/platform comparisons. Performance preparation is not a completed gate;
  strict layout repair, all page cohorts, model publication/download closure,
  native PDF/formula and local VLM remain open.


### 2026-10-20 VQA source styles and dark-to-light repair

- [x] Embed 14 additional original VQA light/dark QSS pairs (47 Chat/VQA
  pairs total). Replace duplicate prompt/loading/button styles and apply
  source filename, image-container/preview, page, splitter, panel/navigation
  and action styles. Match the source image-container eight-pixel margins.
- [x] Fix dark-to-light switching in the owning workspace: clearing QSS after
  setting the new palette restored stale dark colors. A minimal Qt5 reproducer
  distinguishes the faulty order from QSS-before-palette; child dialogs use
  the same corrected order and Chat reads its owner's theme.
- [x] Independently execute original VQA style functions and compare all 14
  actual native widget styles in both modes. Verify rendered container/export
  colors and filename text after live workspace switches. Native workflow,
  source configuration/layout/VQA comparisons and six formal CTests pass;
  all 28 frozen OCR execution identities remain unchanged. Receipt:
  `vqa-styles-source-path.txt`.
- [x] Extend read-only layout diagnosis: on the retained source input,
  F16/Q8_0 raw argmax flips occur below score .5, but respectively one/two
  anchors cross that score threshold. This does not remove strict raw-output
  failures or establish page equivalence. Receipt:
  `layout-decision-confidence.json`.
- [ ] Full VQA component/layout/dialog, font/DPI/Qt6/platform visuals,
  Markdown/formula/PDF, local VLM, remaining GPU page/precision/performance
  gates and qualified publication/download closure remain open.


### 2026-10-20 Native llama.cpp / embedded ggml migration audit

- [x] Audit current native Chat/VQA HTTP transport and VLM-R1 annotation I/O.
  Neither embeds a local language/vision model. Pin audited llama.cpp master
  `707616feb5f379e99ff7442680a55b4a1704c7c9` and its archive SHA; it embeds
  ggml 0.27.0 versus the application's patched 0.21.0.
- [x] Replay the existing 40 patches on the old pristine baseline; inventory
  the new three-way merge: 119 modified files, 24 conflicted files / 78 blocks,
  and one upstream-deleted Metal shader. Resolve three common/CPU files and
  register a standalone CPU migration patch under `3rdparty/llama/experimental`.
- [x] Recompile 239 AICore units and link an isolated candidate to llama, mtmd
  and one embedded ggml. Rebuild 33 contract executables / 72 units; all
  36 public-contract CTests pass. Verify actual loader paths after correcting
  the experimental executable RPATHs. Repeat all 36 against the clean
  ExternalProject install and verify its library paths. This does not certify GPU operators or
  local language/vision model inference.
- [x] Inventory all 22 catalog tasks: 335 explicit F16/Q8 rows, plus eight
  unmarked mixed/F32 bundles; 343 conservative rows / 323 unique files. Identify
  the common runner's dot-separated quantization bug; a focused repair passes
  103 runner tests in the isolated tree and is retained as a pending patch
  while the production OCR execution identities remain frozen.
- [ ] Finish CUDA/Vulkan/Metal operator rebasing (the source-owner CMake
  entry is verified below),
  preserving SAM3D strict math, RAM precision and MapAnything attention paths.
  Complete every selected model's upstream accuracy, backend parity, stability
  and controlled baseline/candidate performance before production promotion.
- [ ] Add the private local LLM/VLM AICore task, typed C ABI, model/projector
  catalog/cache and AnyLabeling worker/UI. The audited llama.cpp has no
  Florence-2 implementation; its 14 upstream tasks and beam-search defaults
  require separate implementation/acceptance, not a generic VLM substitution.
  Scope, reproducible CPU experiment, evidence and per-task model table:
  `3rdparty/llama/README.md`, `migration-evidence.json`.

### 2026-10-20 OCR release follow-up and shared llama CMake entry

- [x] Add independent stored-GGUF-weight Paddle parameter reconstruction with
  exact SHA/name/shape/order checks. Six tests pass; F32 is byte-identical to
  the original. All eight stored-weight Paddle CPU heads agree in argmax with
  existing native CUDA/Vulkan outputs. This attributes rounding only; original
  CUDA truth and strict layout F16/Q8 failures remain mandatory.
- [x] Attempt unchanged controlled CUDA/Vulkan sentinel A/B. Desktop GPU audit
  rejects measurement; preserve rejection/cancellation receipts, restore page
  dispatchers and schedule an exact-condition retry for the page timeout
  caused by the scheduling pause. No new model is upload-qualified.
- [x] Create `3rdparty/llama/llama_cpp.cmake`; share backend flags, ordered patch
  replay/invalidation and install/export logic with the ggml source owner.
  Build/install the pinned parent archive, replay the CPU patch twice and
  verify installed llama/mtmd/ggml loading. Runtime and five configure contract
  checks pass. Production ggml configure exports/signature/arguments remain
  identical to HEAD. Production source ownership has not switched.
- [ ] Finish all current original-CUDA page cohorts, repaired public Q8 files,
  strict layout precision repair and controlled GPU release gates; publish only
  qualified exact SHAs, then complete catalog/cache/cold-download acceptance.
  User explicitly retained 100% argmax agreement for every original head,
  including low-confidence classes and DFL intermediates; stored-weight graph
  agreement and CDLA page mAP cannot replace this gate. The old table probe
  path is a symlink to the candidate, so SLANet still needs an independent
  baseline rather than treating that path as a preserved old executable.
- [x] Add repeated public table C ABI timing with exact structure tokens and
  independently preserved old/current libraries. Final harness loader/stability
  wiring, seven failure-case tests and both offline runner preflights pass.
  Three lossless image inputs reproduce the pinned original NCHW arrays exactly.
- [ ] Finish the queued CUDA/Vulkan SLANet F32/F16/Q8_0 release-profile A/B
  against those libraries and original Paddle CUDA, retaining independent raw
  probability/location and labelled GT gates. CPU wiring and queued commands
  are not GPU performance evidence. Receipt: `table-capi-baseline-path.txt`.
- [ ] Rebase and validate remaining llama embedded CUDA/Vulkan/Metal operators,
  promote its source owner only after the full required task regression, and
  implement local LLM/VLM inference through AICore's public contracts.

### 2026-10-20 Source ownership and layout rounding clarification

- [x] Trace `llama_cpp.cmake -> llama_runtime.cmake -> ExternalProject_Add(ext_llama)`.
  The entry already uses ExternalProject; the active application still uses
  `ext_ggml`. Final removal of `3rdparty/ggml` must also migrate the shared
  build helper, patch runner and SAM3D dependencies and update all callers.
- [x] Expand the migration audit to report full-catalog consumer gaps: all
  556 rows / 526 unique files / 22 tasks, including F32, have static consumers.
  Four audit tests pass, including detection of an uncovered F32 row outside
  the 343-row conservative F16/Q8 subset. Full model execution remains open.
- [x] Reverify 27 layout input/output identities and all eight heads from
  retained tensors. F16 and Q8_0 fail original Paddle CUDA argmax on both GPUs;
  unchanged Paddle graphs populated with stored weights reproduce the flips.
  Fixed CPU-only storage interventions isolate the actual Q8_0 tensors
  (59 tensors: 244 class / 411 DFL flips) and F16 fallback tensors (427:
  22 / 48). Both can independently flip decisions. These are attribution
  experiments; they do not replace original CUDA truth or grant release PASS.
  See `OCR_RELEASE_GATES.md` and `layout-storage-factorial-path.txt`.
- [x] Register the embedded-ggml CUDA source port as ordered migration patches;
  resolve 11 conflicted CUDA files / 35 blocks and adapt the new matrix/vector
  scheduling API while preserving owner-specific precision. Replay all four
  patches on pristine pinned ggml twice; final source SHA matches the compiled
  tree. Linux CUDA 11.8 / SM86 ggml CUDA/CPU modules, llama and mtmd build;
  all four dynamic relocation checks pass. Preserve the initial unresolved
  attention helper and intermediate ADL failure logs; the final registered
  patch fixes both. Receipt: `3rdparty/llama/validation/cuda-rebase-evidence.json`.
  This is build/linkage evidence only, with no GPU model or performance claim.
- [ ] Finish four conflicted Vulkan files / 30 blocks, six Metal files / eight
  blocks and the removed Metal shader; complete original-framework accuracy,
  stability and controlled baseline/candidate performance for the entire
  556-row catalog plus unpublished OCR. Do not promote/delete the old owner
  based on the CPU contract tier or the CUDA compilation result.

### 2026-10-20 Quantization acceptance review and dependency collision audit

- [x] Review whether raw-head disagreement can coexist with useful quantized
  task accuracy. Recompute all six retained GPU layout comparisons and their
  decision margins against original Paddle CUDA; preserve per-head and DFL
  semantics. F16/Q8_0 also cross the 0.5 audit threshold at 1/2 candidates on
  this fixture. Record a conditional, per-model acceptance design in
  `OCR_RELEASE_GATES.md`; existing strict gates remain active, with no release
  qualification from this analysis.
- [x] Audit the complete `ggml.cmake` / `llama_cpp.cmake` source, target, link,
  install and runtime chain. Both entries share one ExternalProject definition;
  production still uses only the old source. Record the final removal closure
  and open model/backend/install work in `3rdparty/llama/README.md`.
- [x] Reproduce both cross-configure source switches being accepted. Bind build
  and dependency install directories to source target plus archive SHA, reject
  conflicting shared prefixes and unidentified candidate installations before
  fetch/overwrite. Fresh CPU build/install, installed runtime/relocation checks
  and nine configure tests pass. Evidence:
  `3rdparty/llama/validation/source-owner-guard-evidence.json`. This is not a GPU model or
  complete migration regression result.
- [x] Track the llama JSON pin as an automatic CMake regeneration dependency.
  A negative control without tracking silently ignores a changed archive SHA;
  the final test rejects it during a normal build without explicit configure
  or download. Reconfigure/build and both CTests pass; five installed library
  SHAs are unchanged, with initial/final input fingerprints retained separately.
- [ ] Optimize independent mixed-precision candidates and validate on held-out
  original-framework/page inputs with CUDA/Vulkan. A relaxed quantization
  acceptance policy has not replaced the user's retained 100% original-head
  gate; neither an optimization proposal nor source-owner isolation grants
  model upload eligibility.

### 2026-10-21 llama source-owner resource and caller migration

- [x] Move the shared ExternalProject implementation, ordered original ggml
  patches/runner and all SAM3D support sources into `3rdparty/llama`; retain
  only the 14-line old archive entry as the production regression baseline.
  Record the pre-move 247-file identity inventory and pristine legacy replay
  twice. Replayed sources match the previous patched baseline exactly.
- [x] Candidate creates only `ext_llama` / `3rdparty_llama`, with one embedded
  ggml. AICore links/test dependencies, Python package dependencies and RAM/YOLO
  build-flag discovery consume source-owner exports. Add the new path to the
  existing CI change filter and update the engineering patch locations.
- [x] Select the rebased CUDA operator manifest through `llama_cpp.cmake` and
  resolve the CUDA architecture helper from its repository location. Validate
  the real entry in a fresh prefix with the complete SAM3D source closure.
  Build support and source relocation do not certify model inference.
- [x] CPU relocated entry build/install and two CTests pass; ten configure
  contract tests cover unique ownership, CUDA manifest selection, unported
  backend rejection, stale-source and shared-install collision cases.
- [x] Fresh CUDA parent build/install with full SAM3D closure and all three
  CTests pass. Isolated rebuilt OCR/table probes execute 158 input checks over
  53 exact model SHAs; every output is element-wise identical to its retained
  pre-migration native output. Keep original-source layout F16/Q8 failures and
  missing controlled performance/page coverage explicit. Receipt:
  `3rdparty/llama/validation/resource-migration-evidence.json`.
- [x] Rebase the 33-file Vulkan operator/shader delta and register standalone
  Vulkan plus combined CUDA/Vulkan manifests under the llama source owner.
  Fresh parent build/install and three CTests (eleven configure contracts)
  pass; replay is idempotent and all 225 Vulkan files match the reviewed port.
  Installed Vulkan executes the same 53 model SHAs / 158 OCR inputs with no
  new applicable head argmax flips against retained native outputs. Probability
  and SLANet location deltas, retained original-source failures and initial
  corrected build failures are explicit in
  `3rdparty/llama/validation/vulkan-rebase-evidence.json`. No release/performance claim.
- [x] Build/install the real parent entry with CPU/CUDA/Vulkan enabled together
  and the full SAM3D source closure. Four entry CTests and 36 retained public
  contracts pass with no skips; the monolithic AICore is relinked from retained
  candidate objects. All 158 OCR inputs per GPU execute, CUDA bitwise equal and
  Vulkan with no new applicable head decisions. Actual loader paths, the initial
  incorrect local CUDA runtime selection, corrected rerun and precise scope
  limits are recorded in `3rdparty/llama/validation/dual-gpu-evidence.json`.
- [ ] Finish Metal porting and every required task/model original-source
  numeric and controlled performance regression before switching the production
  archive and deleting the remaining independent entry. Local LLM/VLM task,
  projector/model catalog and template/image/output acceptance remain open.

### 2026-10-21 OCR follow-up

- [x] Verify the two repaired public Q8 SHA identities against completed original
  CUDA-source reports: CUDA/Vulkan each 9/9 PASS (two graphs plus seven combined
  page scenarios). Keep their evidence separate from old published files.
- [x] Diagnose two role-based mixed-precision layout policies in the independent
  Paddle CPU graph. F32 norm/bias leaves 13 class / 16 DFL flips; adding final
  heads leaves 10 / 15. Neither is an original-source GPU gate PASS.
- [x] Recover the SLANet queue from its dead predecessor's missing terminal
  receipt; verify frozen identities before execution. Four 120-second CUDA
  idle-audit attempts fail before measurement. Stop redundant own retries and
  preserve INCOMPLETE; do not promote missing samples or unexecuted Vulkan.
- [ ] Resolve remaining strict layout accuracy, controlled CUDA/Vulkan A/B and
  exact-file release/catalog/manifest/cold-download gates. Qualified new uploads
  remain zero. See `OCR_RELEASE_GATES.md` and its 2026-10-21 evidence link.

### 2026-10-21 full regression execution and calibration experiments

- [x] Freshly compile the isolated candidate's 239 AICore units and 81 probe
  units; link 36 probes against the llama-owned runtime. Preserve production
  `build_app` binaries. Start the full-tier available-cache CUDA matrix with
  346 expanded scenarios; missing catalog assets remain incomplete coverage.
- [x] Correct the isolated RF-DETR harness's retained legacy static archive and
  executable-local backend staging. Freshly compile 178 static test units;
  nano Q8 CUDA/Vulkan smoke both exit 0. Full RF-DETR acceptance remains open.
- [x] Fix MapAnything's remaining parity-patch resource path after relocation
  to `3rdparty/llama/ggml-patches`; its six existing tests pass.
- [x] Restore the independent MapAnything reference from its pinned repository
  and ggml commits. Its ordered upstream patches reproduce the manifest's exact
  `8e87111a...` diff SHA; repeat application and the vendored attention-source
  comparison pass. See `map-anything-reference-prepare/source-contract.json`
  under the isolated build. GPU reference binaries and the explicit reference
  argument in a fresh full rerun remain pending; source recovery is not inference.
- [x] Run independent offline F16 error-feedback, Q8 feedback, F32 exceptions
  and bias/affine correction. Repeat using 16 pinned upstream CDLA training
  pages separate from acceptance. Serialize six real GGUF candidates per
  experiment and round-trip all 486 parameters before native GPU inference.
  Both GPUs match stored-weight Paddle argmax; original unquantized heads
  still fail. Preserve failures and do not publish these experiments.
- [ ] Finish the full CUDA/Vulkan matrix and the queued 22-task reruns with
  catalog-verified missing downloads and the corrected probes. Include the
  repaired SAM3 F16 candidate in that full rerun. Keep missing sources,
  VRAM skips and incomplete rows visible. See
  `3rdparty/llama/validation/full-model-regression-evidence.json` and the live state files
  it references; none is a completed release or controlled performance gate.
- [ ] Finish the remaining source/backend, strict OCR and controlled A/B gates
  before deleting the preserved production `3rdparty/ggml/ggml.cmake` entry.

### 2026-10-21 full-matrix CUDA stride diagnosis

- [x] Reproduce the SAM3 F16 migration abort and fix native GEMM's output
  leading dimension in the ordered llama CUDA manifests. The six-case scalar
  test fails with the old module and passes exactly with the fix; two corrected
  real-model candidate repeats pass with baseline-identical output hashes.
  Preserve the invalid first focused rerun that loaded old modules through
  executable symlinks/RPATHs. See `3rdparty/llama/validation/cuda-native-stride-evidence.json`.
- [x] Try final-projection block reconstruction on 16 pinned CDLA training
  pages, without fitting acceptance outputs. F16 still has 9 class / 16 DFL
  flips; Q8_0 has 105 / 174 in the independent source CPU diagnostic. Serialize
  both real GGUFs and verify all 486 parameter round-trips. These mixed models
  retain four explicit F32 projection kernels and are not release qualified.
- [x] Complete both projection-reconstructed GGUFs' CUDA/Vulkan checks with
  original Paddle CUDA rerun. F32 control passes, each GPU matches stored-weight
  Paddle argmax, but original-weight flips remain 9 / 16 and 105 / 174.
  SAM3 F16's complete focused old/new repeated report is PASS.
- [ ] Complete the corrected full 22-task CUDA/Vulkan queue. Keep the
  obsolete matrix explicitly incomplete, retain
  per-attempt outputs, and resolve every remaining failure before promotion.
- [x] Add offline real-storage F16/Q8_0 QAT graph plumbing and isolate imported
  Paddle variable names from newly created loss expressions. Independent GGUF
  storage checks pass (4 tests); a real original layout graph has 102 finite
  kernel gradients and all eight F32 outputs exactly match its unmodified
  source on one pinned, disjoint training page. This is training plumbing,
  not a qualified model or a replacement for independent source validation.
- [x] Evaluate the fixed 16-train-page, four-epoch storage-aware QAT pilot after
  exact GGUF round trips on all 486 parameters. Keep original raw-head argmax
  at 100%; preserve every failure and run both native GPUs against fresh
  original-source CUDA outputs. Training loss and stored-weight parity cannot
  qualify the unchanged source, real-page or controlled performance gates.
  All 486 parameters round-trip exactly and both GPUs match stored-weight
  Paddle, but original F32 head flips remain: F16 13 class / 11 DFL, Q8_0
  205 / 347. The strict verdict is FAIL; neither candidate qualifies for release.
- [ ] Resolve remaining original-head ranking failures without changing the
  100% criterion. The optional teacher-derived ranking loss has five passing
  actual Paddle graph tests (including four independent DFL groups); a new
  fixed 16-page/four-epoch experiment with weight 64 and margin 1e-6 completed
  in `ocr-calibration/qat-ranking-pilot/`: both GPUs still fail (F16 11 class /
  12 DFL, Q8_0 258 / 405), with zero stored-reference flips. Original and
  stored-reference diagnostics remain separate; the unchanged strict gate is open.
- [ ] Resolve RF-DETR CUDA performance failures in the corrected full matrix:
  all 44 model rows execute successfully, but large F32/Q8_0, nano Q4_K/Q8_0
  and segmentation-medium Q4_K trigger timing comparisons (5.10% to 20.31%).
  The two-repeat developer samples do not yet isolate the cause or qualify
  controlled throughput. Preserve the overall FAIL and existing thresholds;
  Vulkan's 44 models also execute, but 29 trigger timing failures. The probe
  mixed cold setup into two measured forwards; identical new probes with
  explicit warmups, cold latency and raw samples are built against both
  archives in isolated directories. Full GPU A/B is running after the corrected
  real-page CUDA/Vulkan smoke (3 process repeats / 3 warmups / 10 measured forwards). Keep failures
  open pending results.
- [ ] Resolve the warmup-corrected RF-DETR CUDA result: all 44 numeric/repeated
  fingerprints match the baseline, but six models trigger 14 timing metrics
  (+5.94% to +28.48%). Vulkan subsequently completes 44 numeric rows with matching
  baseline fingerprints but 22 models / 45 timing metrics fail (+5.02% to +18.04%).
  Both overall verdicts remain FAIL. Preserve
  this distinct report and identify sustained runtime regressions versus external
  GPU interference with controlled repeats before accepting the migration.
- [x] Repair the private full-matrix input overlay's missing sibling archive
  links. Reuse and verify existing shared downloads; preserve the original
  OCR/table/relation preflight failures and retry them. Replace the original
  ONNX CUDA reference environment, which lacks `onnx` with user-site imports
  disabled, with the existing isolated dependency-complete environment.
  Classifier F32 and v5 mobile det/rec F32 repeat successfully afterward;
  this does not complete all OCR source/page/performance acceptance.
- [x] Resolve the common runner's `threads=0` before ONNX/Paddle page and
  SLANet graph consumers configure either engine; preserve explicit counts and
  reject negatives. The old script fails the new regression check; the repaired
  OCR script suite passes 66 tests and SLANet passes 9 tests. Preserve the
  interrupted partial matrix as incomplete and rerun against immutable scripts.
- [x] Build an ONNX CUDA reference venv from pinned requirements without system
  site packages. Isolated imports include original TextSystem polygon/image
  dependencies. PP-OCRv5 mobile F32 real-page comparisons pass on CUDA/Vulkan
  with both frozen baseline and candidate repeated twice; full OCR acceptance
  remains open. Current serial state:
  `build_llama_dual_gpu_3nl01j4j/ocr-thread-protocol-fix/state.json`.
- [x] Match the experimental ExternalProject driver's hidden symbol visibility
  to the production dependency flags. The actual generated external configure
  command fails the new check before the fix; all 12 CMake contract tests pass
  afterward. Existing measured binaries remain unchanged. This configuration
  repair has not yet been shown to resolve the RF-DETR latency failures.
- [ ] Finish the queued independent MapAnything CUDA/Vulkan reference builds
  and full matrix rerun with the explicit reference argument. The new follower
  waits for both current serial queues to finish before building; exact source
  pins, foreign-loader rejection and private-cache verification stay enforced.
  `map-anything-reference-prepare/serial-followup-state.json` currently records
  waiting, not completed build or inference evidence.
- [x] Refactor the dependency configuration into `llama_cpp.cmake` (pin and
  owner), `llama_runtime.cmake` (ExternalProject/install/exports), and backend
  and patch helpers. Replace the source-pin JSON with CMake constants; archive
  the twelve historical JSON receipts without changing their SHA-256 values.
  Four generated configure commands/source identities/patch signatures remain
  identical. A clean isolated CPU build/install, two CTests (fourteen configure
  contracts), five relation closure tests and installed loader trace pass.
  Requested Vulkan dependencies now fail explicitly for the candidate; preserve
  the baseline's existing behavior. Evidence: `build_llama_config_refactor_glj5ibw7/`.
- [x] Audit the pinned pristine embedded ggml against all five custom operation
  enum values and seventeen additional helper APIs. Record official alternatives,
  tensor/precision differences and required gates in
  `3rdparty/llama/KERNEL_AUDIT.md`. Retain required custom semantics; already reuse
  official native CUDA GEMM with the verified F16 output-stride patch. The
  combined manifest replays twice unchanged and all 1503 files match the active
  candidate. No unmeasured official-kernel speedup or all-model pass is claimed.
- [ ] Complete the immutable OCR CUDA/Vulkan queue and resolve every remaining
  original-head, page, controlled source-speed and migration A/B failure.
  The read-only 11:25 UTC snapshot completes 61/168 CUDA scenarios: baseline and
  candidate each pass 122 structured accuracy attempts. Candidate source-speed
  comparisons still fail ten attempts across seven scenarios; all 56 recorded
  candidate source-performance qualifications and GPU audits are zero. This
  developer snapshot is INCOMPLETE, cannot qualify a release, and does not
  supersede the failed layout F16/Q8 original-head gates. See
  `OCR_RELEASE_GATES.md` and `build_llama_config_refactor_glj5ibw7/Testing/ocr-cuda-progress.json`.

### 2026-10-21 QAT target and migration A/B audit

- [x] Correct the layout QAT ranking target: enforce the unchanged teacher
  winner with a configured positive margin, without reconstructing its full
  top-1/top-2 probability gap; retain energy-normalized head MSE and four
  independent DFL distributions. The six actual Paddle graph/gradient tests
  pass through the default runner, including clean process exit.
- [ ] Retrain F16 and Q8_0 with the corrected ranking loss using only the
  pinned 16 CDLA train pages. The prior MSE/ranking pilots remain failures and
  cannot serve as evidence for this objective. Re-serialize every parameter,
  then require original F32 versus stored model raw argmax 100%, full page
  metrics, CUDA/Vulkan parity and controlled performance before release. The
  serialized experiment is deferred until the active GPU comparison queue
  yields; its latest log is at OCR CUDA scenario 91/168.
- [x] Match future isolated llama validation builds to the frozen baseline's
  `GGML_CPU_ALL_VARIANTS=ON`; verify the value reaches the nested configure
  command in the 14-test CMake contract suite.
- [ ] Rebuild both RF-DETR A/B owners with matching CPU ISA variants and hidden
  visibility after the active GPU queue reaches a child-free boundary. The
  prior warmup report remains FAIL (CUDA: 14 failed metrics / 6 models;
  Vulkan: 45 / 22) despite matching numeric fingerprints. RF-DETR's custom
  deformable-attention callback runs on CPU, making the module difference a
  plausible contributor, not a proven cause. The production
  `AICore_CPU_ALL_VARIANTS` default and existing binaries were not changed.
- [ ] Port the Metal patch to llama's split `ggml-metal/kernels/*.metal`
  layout and validate on macOS. The old patch changes 11 files, including a
  monolithic shader absent from the pinned source; the migration inventory has
  six conflicted Objective-C/C++ files and eight blocks. Upstream 0.27 already
  has newer command-buffer error handling, so the old lifecycle hunks need
  semantic review. Linux cannot provide the required Metal compile, numerical
  regression or performance gate. Keep Metal disabled for the candidate until
  those checks pass. Source inspection confirms `kernels/conv.metal` already
  indexes the batch dimension and uses `nb3`; drop the old v0.21 MapAnything
  batch-index backport during rebase. The separate parallel-channel tree
  reduction remains an unmeasured Metal optimization.

### 2026-10-22 verified corrections and remaining acceptance

- [x] Reject CPU-only Paddle before creating a CUDA-teacher experiment, and
  verify the inference configuration actually enables GPU. Preserve the earlier
  CPU-teacher run as invalid CUDA evidence. Fix the default zero-ranking Q8
  training feed; seven real forward/gradient cases now pass in the isolated
  GPU-wheel environment with normal process exit. See `OCR_RELEASE_GATES.md`.
- [ ] Finish the replacement 256-page, four-epoch fixed training experiment
  `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-cdla256-gpu-teacher-v2-20261022/`.
  It uses a real CUDA teacher and disjoint pinned train pages, then serializes
  every parameter and independently compares all original class/DFL winners on
  CUDA/Vulkan. The prior 16-page pilots and mislabeled CPU-teacher run do not
  qualify the replacement. Keep the 100% original-head gate unchanged.
- [x] Build the isolated dual-GPU candidate with matching CPU variants; loader
  evidence confirms Haswell/AVX2 selection for both owners. Complete RF-DETR
  44-model numeric fingerprints match on CUDA/Vulkan. Timing still fails (14
  CUDA metrics, 44 Vulkan metrics); matching ISA does not repair the regression.
- [ ] Resolve all full-tier task numerical/stability/performance failures;
  the immutable serial matrix currently runs SAM3 Vulkan. Earlier Vulkan
  private-cache audit ordering failures and the two SAM3 CUDA artifact-change
  failures remain invalid evidence requiring a fresh frozen rerun. Never edit
  active probes or sources to turn an existing report into a pass.
- [x] Port Metal changes to the pinned split-kernel layout as an ordered patch;
  replay and idempotence checks, four host C++ syntax checks, 16 exact CPU
  transpose-convolution cases and 17 CMake contracts pass. Drop the redundant
  old batch-index backport. See `3rdparty/llama/KERNEL_AUDIT.md`.
- [ ] Compile MSL and execute Metal accuracy/performance gates on macOS through
  the isolated opt-in; Linux source/CPU checks do not satisfy this requirement.
  Keep production source ownership and the final `3rdparty/ggml/ggml.cmake`
  baseline entry until the complete regression passes.

### 2026-10-22 serialized OCR truth and frozen llama follow-up

- [x] Complete the replacement 256-page, eight-epoch CUDA-teacher and CUDA-
  optimizer layout pilot using log-probability class ranking and four separate
  DFL distributions. Independently reload all 486 serialized parameters and
  verify their values against the Paddle storage graph. On the held-out input,
  CUDA and Vulkan each match the stored-weight Paddle argmax exactly. Original
  F32 truth still fails: F16 has 13 class / 18 DFL flips; Q8_0 has 140 / 232.
  These candidates are unqualified and have not been uploaded. The earlier
  unchecked training items remain historical; see `OCR_RELEASE_GATES.md` for
  artifacts, SHA-256 values and the strict original-head gate.
- [x] Diagnose the remaining serialized-layout flips with original winner
  margins: no original ties; quantization perturbations cross small, nonzero
  margins. Verify STE forwarding reproduces actual storage values for all
  1,794,568 trainable parameters without cancellation discrepancies. This
  excludes backend disagreement and STE cancellation for the checked pilot;
  it does not establish a solution to the original F32 argmax requirement.
- [x] Fix the SCRFD manifest glob so the published unsuffixed default model has
  exactly one inference consumer alongside F32/Q8_0. All 108 validator tests
  pass. Preserve the active Gaussian CUDA report invalidated by this edit;
  its isolated retry with an explicit frozen manifest passes all nine rows.
- [x] Verify the Vulkan dependency-cache optimization with 3,062,201 original
  versus cached predicate comparisons, zero mismatches, and pristine replay /
  idempotence of all ten ordered patches. Audit actual loaded module paths:
  the executable directory takes precedence over `LD_LIBRARY_PATH`. Mark the
  earlier two patch-timing experiments invalid where the loader used the old
  module; retain their independent source and semantic checks.
- [x] Repeat the two-model Vulkan A/B with copied executables and verified
  module hashes, four interleaved processes and 5 warmups / 30 forwards. RF-DETR
  base F32 forward regression changes from +3.79% to +1.32%; base Q8_0 from
  +8.19% to +3.73%. All output hashes and candidate optimizer node orders match.
  This developer-profile result is not controlled release qualification.
- [ ] Complete the fixed-module full 44-model RF-DETR Vulkan retry, followed
  by AnyCalib and ALIKED, in
  `stride-fix/vulkan-fixed-full-validation-9aqib7e8/`. Keep both probe-directory
  module links audited and the original candidate immutable. Other full-tier
  failures and required independent source-reference bindings remain open.
  Update: RF-DETR Vulkan now passes 44/44 and AnyCalib Vulkan 3/3 under the
  unchanged developer-profile gates. ALIKED stopped before inference on a
  missing F32 cache file; the existing catalog/SHA downloader has restored it,
  and its replacement inference run is still required.
- [ ] Resolve actual Depth tensor differences; do not suppress the fingerprint
  failures. With the original 224 resize protocol, base F16 maximum absolute
  depth differences are 0.000495 on CUDA and 0.008865 on Vulkan. Camera outputs
  and dimensions match in these two GPU checks. The separate 504-size Vulkan
  run is bit-exact and cannot substitute for this failed input profile.
- [ ] Resolve LightGlue Vulkan fingerprints with typed match-index / score
  diagnostics and retry the isolated GKD CUDA preprocessing timing failure.
  Neither has been accepted by changing the fingerprint policy or threshold.
- [ ] Finish remaining task, independent upstream truth, controlled performance
  and macOS Metal gates before removing the independent ggml baseline entry.

### 2026-10-22 teacher math policy and migration causal controls

- [x] Reproduce the original Paddle CUDA teacher with and without TF32 in
  separate processes on four pinned training pages. Each untrained F32 QAT
  graph is bit-exact to its own teacher, but the two teacher policies disagree
  on 39 class / 81 DFL winners. Set `NVIDIA_TF32_OVERRIDE=0` before importing
  CUDA dependencies in the offline trainer and record it in the recipe.
- [x] Complete the replacement 256-page/eight-epoch F16/Q8_0 run with the
  corrected math policy and independent serialized-weight checks. Held-out
  original-source flips remain F16 10 / 14 on both GPUs, Q8_0 140 / 243 on
  CUDA and 140 / 244 on Vulkan. The latter also has one stored-source DFL
  disagreement at a top-two gap of `6.556511e-7`; backend exactness from the
  earlier pilot does not qualify this new checkpoint. No candidate is released.
- [x] Confirm Depth Vulkan's first changed operation by one-variable control:
  restoring the old short-M split-K guard restores all five final public arrays
  and 619 captured intermediate tensors for base F16 at target 224. Keep the
  global guard as an experiment; a DPT-only candidate patch still needs device
  positive/negative controls, task coverage and performance evidence.
- [x] Replace the invalid copied LoMa matcher probe with one identical portable
  executable and verify both AICore library identities. All six selected CUDA
  rows pass; Vulkan retains one published G-model process-wall regression
  (+8.77%). Do not substitute the earlier mixed-runtime failures for this result.
- [ ] Finish the running RF-DETR CUDA 44-row longer retry, DPT-only patch
  qualification, LightGlue first-operation diagnosis and reference-bound OCR
  release measurements. The immutable full-task matrix and its remaining
  fingerprint/performance failures remain open. Keep the baseline ggml entry.
  RF-DETR CUDA has now completed: 44/44 pass with three process repeats,
  five warmups and ten forwards under unchanged developer thresholds. Together
  with the separate fixed-module Vulkan 44/44, this closes that task's current
  developer retry; it does not qualify controlled release performance or the
  remaining task families.

### 2026-10-22 remaining gate attribution and current-source queue

- [x] Check the new Q8_0 checkpoint against its independent stored-weight
  Paddle CUDA truth on both old/new runtimes: CUDA 0 class / 0 DFL flips,
  Vulkan 0 / 1 in each runtime. CUDA's eight old/new heads are bit-exact.
  The Vulkan stored-source flip predates the upgrade and remains unresolved;
  original-source quantization failures remain separate release blockers.
- [x] Complete DPT-only patch device controls: five final arrays and 619
  captured nodes restored; unrelated names and non-im2col negative controls
  retain new upstream results. Actual loaded module identities verified.
  Keep the patch unregistered until full model/performance qualification.
- [x] Locate LightGlue's first difference to the new one-row GEMV dispatch,
  with identical actual operands and independently verified final-output
  invariance. Its F64 mathematical error is substantially lower than the old
  operation's. Exact fingerprints still fail; upstream task truth remains
  required. Do not call the new, more accurate operation a numeric regression.
- [x] Finish both source-bound OCR direction-classifier release sentinels.
  Both reports FAIL GPU activity audits; most attempts execute only nine of
  ten numeric fixtures after the timed fixture's idle wait fails. No qualified
  speed/VRAM claim or upload follows from these partial measurements.
- [x] Verify the interrupted SAM3 base-plus F32 model independently under the
  original standard gate: PASS. Retain the interrupted process-wall sample as
  invalid and link its separate replacement instead of rewriting that sample.
- [ ] Finish the current full matrix (SAM3 39 models is running), full DPT
  patch coverage, remaining numeric/performance failures and Metal hardware
  validation. Production ggml retirement remains blocked on these results.
- [ ] Complete current-candidate reference-bound OCR/table/tagging/relation
  and MapAnything follow-ups. Replaced the obsolete waiting controller, which
  still pointed at validation-v4, with
  `build_llama_dual_gpu_3nl01j4j/current-reference-followup-buca6r5k/run.py`.
  It waits for the current complete matrix, binds explicit source references,
  then builds the pinned MapAnything CUDA/Vulkan references outside inference
  timing and runs the full task without `--allow-incomplete`. Queued work is
  not completed evidence; it cannot qualify a release or remove the baseline.

### 2026-10-22 serialized-update and independent framework checks

- [x] Verify all 486 QAT deployment parameters again. Actual F16/Q8_0 values
  equal freshly serialized masters; F16 changes 446,690 stored values and Q8_0
  changes 25,866 integer codes plus 7,561 block scales. Updates are retained,
  but the original-source strict argmax failures remain open.
- [x] Fix host environment removal in both offline Paddle entry points:
  updating a cleaned dictionary alone retained removed logging/PYTHONPATH
  keys. Two real Paddle import/normal-exit checks pass with injected dirty
  host keys. This is not a new model or GPU acceptance result.
- [x] Replay official pinned LightGlue transformer/assignment modules with
  exact stored GGUF weights and the common captured input projection. Both
  runtimes keep all 64 ALIKED/SIFT indices; new score MAE improves slightly,
  maximum error increases slightly. Preserve both metrics and the existing
  fingerprint failure; this diagnostic does not replace original-checkpoint
  or real-image acceptance.
- [ ] Close the remaining full task groups and original-source/controlled
  performance gates. The scope table in `3rdparty/llama/KERNEL_AUDIT.md`
  distinguishes complete retries, six-row LoMa selections, active work and
  queued work. No new OCR model or production runtime is qualified yet.
- [x] Make the independent Paddle graph and legacy layout-reference entries
  reject an actual CPU library or disabled CUDA configuration before creating
  a predictor. Fix the layout entry's pre-import TF32 policy and record its
  actual device configuration. All 23 focused tests pass, including mixed
  package metadata and disabled-GPU negative cases. Four real-wheel entry
  preflights also pass; CUDA checks intercept predictor creation and are not
  new inference or performance acceptance. See the latest OCR gate section.
- [x] Separate requested ONNX inputs from completed probability comparisons
  and passing fixture rows; retain incomplete indices and reject stale success
  reports when a repeated source export fails. The 34 focused tests pass.
  Read-only historical reaggregation confirms requested 10 / completed 9 for
  both retained controlled classification reports; both remain FAIL.
- [x] Add a Q8-only Adam learning-rate multiplier with default 1, keeping F16
  kernels and F32 affine rates unchanged. Eleven real Paddle graph tests pass,
  including actual parameter-update ratios; this does not qualify a model.
- [x] Finish the predeclared Q8 step sweep and F16 small-step training, using
  training-only checkpoint selection, actual serialization, independent source
  CUDA and native CUDA/Vulkan raw heads. Preserve the original 100% class and
  per-edge DFL argmax gates, all failures and exact loaded-module identities.
  Both new Q8 step multipliers fail to improve training and are not exported.
  F16 training improves 2,621 to 580 total flips, but its actual serialized
  model still has 7 class / 12 DFL flips on each GPU against original Paddle
  CUDA. Both GPUs have 0 / 0 against the actual stored-weight source. This
  completes those experiments, not the original-source or release gates;
  retain the unresolved checkbox and numeric evidence in OCR_RELEASE_GATES.md.

### 2026-10-22 maintenance and user-approved regression scope

- [x] Move the migration source/catalog audit and its tests from the root
  `scripts/` directory to `3rdparty/llama/scripts/`. Update the documented
  invocation; all four audit tests pass. This source inventory is separate
  from the shared inference gate and never certifies model compatibility.
- [x] Translate Chinese development prose under `plugins/` to English,
  including both OCR/parity audit ledgers. Preserve all SHA-256 identities,
  per-file evidence rows and historical evidence locators. Keep localized
  manuals/translations, Chinese RAM tags, language names, prompt dictionaries
  and Unicode/OCR test fixtures in their original languages.
- [x] Review the Ubuntu workflow diff against HEAD: its sole addition is
  `3rdparty/llama/**` in the AICore path filter. Changes to the new source owner
  must trigger the existing CI job; no packaging step was changed.
- [x] Apply the user's waiver to expensive large-model migration gates:
  SAM3D/TRELLIS/MapAnything/LingBot full reconstruction, SAM3 large variants,
  RAM-family F32, and large YOLO/ReID/relation variants. Keep their catalog
  entries; record compatibility as user-accepted rather than measured.
- [ ] Finish explicit smaller-model CUDA/Vulkan selections through the common
  validation runner and diagnose every remaining numerical/performance failure.
  OCR remains subject to real-page quality, CUDA/Vulkan source-speed, stability,
  provenance and release-download gates. Text-detector/recognizer quantization
  probability/argmax differences are diagnostic under the later user acceptance;
  the explicitly retained layout raw-class/per-edge DFL argmax 100% gate remains
  mandatory.
  Current Q8 graph/page results and the desktop GPU audit rule are recorded in
  `OCR_RELEASE_GATES.md` (2026-10-23); no OCR model is release-qualified yet.
- [x] Reject the earlier RMBG IM2COL-first attribution: its interposer marked
  retained tensors after the explicit allocation reserve. Independent old/new
  host controls prove that later nodes overwrite those captures despite final
  output invariance. Corrected before-reserve capture is queued separately;
  the actual matte differences and regression failures remain open.
- [x] Check constant DFL head-bias feasibility for the best F16 candidate
  without changing any parameters. Four head/edge groups contain contradictory
  zero-margin constraints; constant bias cannot recover all original winners
  even on this fixture. Preserve the negative-cycle evidence and continue to
  require every original raw-head comparison; this is not a release pass.

### 2026-10-23 continuation: selection preflight and current verification

- [x] Fix the common validation runner's explicit `--models` catalog filter.
  It now prefetches the selected model and manifest-declared fixed dependencies,
  rather than every catalog row matched by the task-wide globs. A unit test
  covers shared cross-task dependencies and excludes sibling/unrelated model
  files. All 109 `test_validate_all.py` tests pass. An offline YOLO preflight
  narrowed from 20 catalog rows to five; it then stopped on the baseline
  catalog's SHA mismatch for the cached current-candidate OBB file. The
  migrated candidate catalog is not present on disk, so no candidate inference
  is claimed from that check.
- [x] Re-run the current qAnyLabeling CTest set: 36/36 pass, including plugin
  entry, Chat, OCR document/table/rich-text, video, point-cloud UI/camera/3D
  cuboid and LightGlue bridge tests. The ten GPU audit unit tests also pass.
- [ ] Repeat and retain current-candidate selected CUDA/Vulkan results. The
  prior short diagnostic batch recorded ALIKED n16rot Q8 accuracy agreement
  (median keypoint delta 0 px; descriptor cosine 1.000), RF-DETR nano Q8
  release-profile runner PASS, and selected DeepLSD/LoMa process PASS rows.
  LightGlue Vulkan still fails the exact output fingerprint despite stable
  candidate runs and preserved match indices. These receipts were in the
  temporary `build_llama_dual_gpu_3nl01j4j/` tree, which is absent now; treat
  the rows as historical diagnostics and rerun them into persistent reports
  before any migration/release conclusion.
- [ ] Resolve English Q8's 50-page SROIE recognition gate, then rerun and
  archive the complete per-file page/source-speed/stability matrix. Chinese Q8
  passed that page-quality cohort; English Q8 exceeded the declared word-error
  delta. See the dated OCR gate entry for exact metrics and the evidence limit.
- [ ] Complete the still-open full-window visual comparison, local VLM and
  advanced Chat/VQA, native PDF/formula/rich-text workflows, remaining model
  families, full point-cloud camera interaction, Metal device validation and
  Linux clean-package acceptance. The feature checklists above remain the
  authoritative work inventory; this continuation does not close them.
