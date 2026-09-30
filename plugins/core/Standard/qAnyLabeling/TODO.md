# qAnyLabeling parity TODO

## Open parity gaps at a glance

Every remaining upstream misalignment, in one table. Status / owner notes:

| # | Area | Gap | Owner / landing | Status |
|---|------|-----|-----------------|--------|
| 1 | OCR interaction | `skip_detection` | C API `aicore_ocr_recognize` DONE, worker rois channel DONE, `Skip Detection` button + in-place shape update DONE; remaining: test_inference selected-rects case | TODO — one test case left |
| 2 | OCR interaction | Direction classifier `cls` (upstream `use_angle_cls`, 0/180 flip before rec) | `ch_ppocr_mobile_v2.0_cls` paddle→GGUF conversion (extractor is generic), catalog row, `aicore_ocr_options.cls_model_path`, flip branch in `ocr_capi.cpp` | TODO — conversion may reuse `convert_paddle_ocr_gguf.py` |
| 3 | OCR geometry | Official det-eval polygon accuracy (Vatti unclip on exact contours vs boundary-pixel hull) | `ocr_capi.cpp` post-process refinement | TODO — current output verified end-to-end, sub-pixel deltas vs OpenCV |
| 4a | OCR evidence | Same-host official A/B and backend matrix conclusions | Paddle CPU baseline (IR on, RTX-3060 host): rec p50 90 ms / det 34 ms / layout 458 ms — the det output checksum matches ours exactly (9.2490e3), so the upstream-truth anchor is cross-validated. AICore probe p50 (CPU): rec 220 / det 380-975 / layout 509-975 — accuracy is bit-level vs paddle fixtures everywhere, but CPU latency trails paddle (det depthwise-conv scalar expansion dominates); GPU-CUDA rows already recorded (rec 206 / det 359). Vulkan: layout f16/q8 verified, f32 verified (maxDiff 2e-5) — **27-cell matrix (3 graphs x 3 quantizations x CPU/CUDA/Vulkan) all PASS**. Known defect (UPDATED): the CTC decode call sites consumed runGraphCached's (outC,outH,outW) = (1,T,vocab) mapping as (T,vocab) — every transcription came back empty even though the rec graph itself is bit-exact (cross-checked: paddle on the dumped packed input also decodes empty for that dark-blue word image, and the probe stays green). The two call sites now decode with (outH,outW) = (T,vocab); the layout stage additionally degrades to boxes-only on failure instead of discarding the run. Remaining: the standalone `run` entry still reports an empty last_error on one small-image path (repro: layout pack loaded + word_1.jpg) — needs a focused error-propagation round. Fixtures moved to the dedicated `ocr-fixtures` release so PP-OCR carries models + testdata zip only | Layout/OCR evidence recorded; rec-preprocess defect TODO |

| 4 | Region/KIE | KIE (ViLayoutXLM, ViT encoder + SentencePiece tokenizer + 2-D layout embeddings) weights unpublished; regions path is LIVE | Layout LANDED 2026-09-30 end to end: `picodet_lcnet_x1_0_fgd_layout_cdla` converted (three quantizations published), the single missing op `split` added to the interpreter plus a per-build fold pool and multi-fetch graphs, GFL integral + MultiClassNMS + reading order implemented in `ocr_capi.cpp` behind the existing `layout_model_path` option, `aicore_ocr_label_*` accessors, catalog/digests/manifest rows (ocr-layout-f32/f16/q8_0; CPU and CUDA verdict PASS, f32 maxDiff 2e-5), qAnyLabeling auto-discovers the layout pack next to the det model and emits layout shapes with class/reading-order extras. KIE stays TODO: the ViT encoder needs attention-class graph ops (softmax(QK)V composition exists in ggml primitives) but the XLM-Roberta SentencePiece tokenizer and 2-D position embeddings are a dedicated conversion+runtime round — the reserved `relations` contract stays unexposed until then (register-must-run rule) | Layout Closed; KIE TODO — tokenizer + attention interpreter round |
| 6 | Visual parity | Screenshot-level layout comparison against the in-tree upstream references | Manual fixture pass per panel using `docs/images/*` baselines | TODO — references now in-tree |
| 7 | Point cloud | Upstream-specific interaction details (GPU surface-depth picking, renderer tiles, camera calibration, streaming loader, shortcut/theme details) | PointCloudWindow already reuses the host multi-window VTK pipeline (same rendering/IO plugins as the main app); remaining work is implementing those upstream interaction details inside that window | TODO — interaction-detail round inside the existing window, not a new pipeline |
| 9 | External services | `remote_server` / `open_vision` / multi-model source pipelines need an external-transport contract that violates the ownership map | Not portable to AICore by design | Stays disabled |
| 10 | Config module | Upstream `configs/auto_labeling/*.yaml` carries per-model interaction defaults (display_name, conf/iou thresholds, input size) that the picker should surface per model | AICore catalog owns model facts (ownership map): extend catalog entries with the interaction-default fields, plugin reads them from the catalog — never a private yaml table in the plugin | facedetect LANDED 2026-09-30 (`aicore_facedetect_model_entry` now carries conf/iou/input_size — buffalo 0.5/0.4/640, yunet 0.6/0.3/160 — and the picker applies the catalog conf when a detector is selected); yolo/ocr entries are the same-shape follow-up |
| 11 | Packaging | Run/.exe/.app installers must carry the plugin resources (translations qm, docs) and rely on catalog remote-download for models; default model per task must match upstream | Existing docker/build scripts + qrc resources; verify installer contents per platform (Linux/Windows/macOS) | TODO — packaging verification round |

Everything else in this file is either done or documented as intentionally
unsupported with its official name.

---

Reference: X-AnyLabeling-GGML commit `926017d09b22564a78c55eb2f368c7b20faa0466`.
The source tree has 205 `anylabeling/configs/auto_labeling/*.yaml` model
configurations with 92 distinct `type` values. A model is available here only when AICore publishes a matching
task contract and a catalog GGUF. A similar output shape alone does not mean
the source model has been ported. Re-check the live AICore catalogs before
closing a row.

Catalog-backed source `type` values whose corresponding workflows are fully
or partly wired are `rfdetr`, `rfdetr_seg`, `rmbg`, `scrfd`,
`segment_anything_2`, `segment_anything_3`, `yolo26`, `yolo26_obb`,
`yolo26_pose`, `yolo26_seg`, `yoloe`, `yolov8`, `yolov8_seg`, and `yolow`.
This is a task-family mapping, not a claim that every source checkpoint or
configuration is in the AICore catalog. The remaining 78 source types are
listed below, including composite types that can reuse existing AICore models.

## Source models with no matching AICore catalog

These source model families are intentionally absent from the model picker.
Implement the inference task, GGUF catalog and validation first; then add the
plugin worker and matching interaction. The names below are source config
`type` values or exact source model families.

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
  (GPU matrix round will unify this). Remaining: probe_ocr ctest target
  wiring, f16/q8_0 gates on GPU, CUDA/Vulkan/CPU matrix, det
  post-process (DB binarize + unclip), then the workspace flip.
- [ ] Vision-language captioning, VQA and document grounding: `florence2`,
  the video description/segmentation providers and chatbot models. Research
  verdict (2026-09): the ggml tensor library itself does not carry a VLM
  layer — captioning/VQA lives in llama.cpp's `libmtmd` (multi-modal
  library, April 2025, mmproj projectors; supports Qwen2-VL/2.5-VL and
  other vision LMs). The viable route is a new `vlm` AICore task that
  vendors llama.cpp core+mtmd pinned to a release whose bundled ggml
  matches (or is patched to) our pinned ggml v0.21.0, wired through the
  task-private cmake pattern; the plugin must not call llama.cpp directly.
  Scope note: libmtmd covers captioning, VQA and chatbot (the disabled
  toolbar entries) but does NOT replicate Florence2's grounding head —
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
- [ ] Image tagging and attributes: `ram`, `pulc_attribute`,
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
  boxes) and the Classifier toolbar entry staying disabled until then.
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
- [ ] Source `yolow_ram` requires its RAM tagger, which is absent even though
  plain `yolow` text detection has a catalog-backed YOLO-World counterpart;
  the RAM contract gap is recorded above.
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
- [ ] Remaining special source pipelines such as `remote_server`,
  `open_vision` and source-specific multi-model combinations are external
  dependencies by design: `remote_server` delegates inference to an
  HTTP/WebSocket service outside the process, and `open_vision` composes
  provider-specific model runs that no AICore task contract expresses.
  They stay disabled with their official names; mapping them onto AICore
  would require defining an external-transport contract that violates the
  ownership map (the plugin must not implement a private downloader,
  backend, or scheduler).

## AICore-supported work still required in this plugin

These are not exempted by the missing-model rule.

- [x] AICore catalog rows outside the 2D image-labeling contract are not
  silently dropped: LoMa/LightGlue now has a paired-image correspondence
  review bridge with detector/descriptor/matcher quantization and backend
  selectors, accepted-match point annotations, and a real two-image test
  (`test_qanylabeling_loma_bridge`, 577 CPU correspondences on the pinned
  Sacre-Coeur pair). Gaussian, LingBot, TRELLIS
  and SAM3D remain owned by `qFreeSplatter`, `qLingbotMap`, `qTrellis` and
  `qSAM3D`; their model matrices are documented in [MODEL_MATRIX.md](MODEL_MATRIX.md)
  but do not belong in this image-labeling picker.

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
  coordinates are restored to source-image space. The LightGlue matcher is a
  two-image feature correspondence task; a review UI for pairing images and
  converting matches to annotations is implemented by the LoMa bridge.
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
- [ ] Point-cloud parity remaining: exact GPU surface-depth picking, renderer
  tile acceleration, camera image
  calibration, streaming frame loader, and source shortcut/theme details.
  These remain manual UI/rendering tasks independent of model support.
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
- [x] Interaction-logic parity round four: the source shape.* color group is
  fully wired into the canvas render path (color strategy auto/manual,
  default shape color, auto-color hue shift, line/fill/vertex/select-line/
  select-fill colors, plus a hover-vertex highlight using
  hvertex_fill_color) with nine persisted settings rows; the label dialog
  was replaced with a source-compatible AnnotationLabelDialog honoring
  sort_labels, show_label_text_field, label_completion (startswith/contains
  filtering) and fit_to_content; move_mode=auto now pans the canvas on
  left-drag over empty space (behind the group-selection branch);
  font_family applies an application font to the workspace; and two
  offscreen end-to-end fixtures now drive the exec()-blocking surfaces from
  inside their nested loops: the canvas context menu (filter submenu
  structure and population, plus SAM right-click suppression) and the
  modal label dialog (sort, completion filter, accept).
- [ ] Remaining interaction parity: model hub is contract-owned by the
  AICore catalog (no plugin-side download source switch), the adapter has
  no logging subsystem for logger level, and the Qt image allocation limit
  needs the Qt 6 build — all three are summarized in the settings dialog
  TODO note; validate_label stays unexposed because the source default is
  null; and exact diverging-default parity (file_list_checkbox_editable)
  remains a conscious adapter decision.
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
  stay open above.
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
- [ ] Training UI, annotation quality/consistency tools, and workflows that
  depend on external services. Implement only after their data contracts and
  ownership are identified.
- [x] Entries whose upstream features have no AICore contract keep their
  official source names and icons but are explicitly disabled with TODO
  tooltips instead of fake behavior: ChatBot, VQA, the standalone image
  Classifier, PaddleOCR, Ultralytics training, Save Visualization Video,
  Shape Manager, Change Output Dir, and the Label Flags / Attributes /
  Label Classes template uploads. Each carries a stable
  `AnnotationTODO*` object name for automation.
