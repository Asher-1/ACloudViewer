# P1 task-contract designs for the three confirmed-missing families

These are the contract designs required by the AICore engineering contract
before any new task lands ("Before a new task, write its input ownership,
typed output/release path, options, errors, timing boundaries, backend
matrix, assets, and numeric gate"). They are designs, not coverage: no
picker entry, catalog row, or accuracy claim exists until the corresponding
P0 converter and P2 gates produce passing evidence on this repository.

Shared preconditions for all three families:

- **P0 converter**: first-party, deterministic, emits GGUFs whose metadata
  carries the source identity (upstream repo, revision, config hash) exactly
  like `convert_paddle_ocr_gguf.py` embeds `ocr.source.name/sha256`.
- **Deformable attention** (`grounding_dino`, `dfine`): a new ggml operator
  via the ordered llama-patches chain. The patch must add the op to CPU plus
  the CUDA and Vulkan dispatch tables in one migration manifest (the
  CPU-first pattern used by the OpenPCDet sparse extension), because the
  P2 matrix runs all three backends on the same graph.
- **One monolithic `libAICore`**: each family lands as a task under
  `core/AICore/src/tasks/<task>/` behind the existing
  `AICore_ENABLED=ON` build; no per-task shared library and no public
  `AICore_USE_<TASK>` build switch.

---

## 1. `grounding_dino` — text-prompted open-vocabulary detection

- **Input ownership**: caller-owned decoded image crossing the ABI as a
  borrowed, stride-aware `aicore_image_view` (RGB8); the text prompt is a
  caller-owned NUL-terminated UTF-8 string copied into the context at
  option time. Nothing in the task writes to caller buffers except the
  typed result.
- **P0 source**: `IDEA-Research/grounding-dino-base` (or `swin-ogc`)
  HuggingFace checkpoint; converter walks the Swin backbone, BERT text
  tower, feature-enhancer (six deformable layers) and contrastive head.
  The text tower uses the same BERT weight layout the embedding models
  already use, so the converter reuses the existing BERT-to-GGUF mapping
  where shapes allow.
- **GGUF layout**: single file `grounding-dino-base-f32.gguf` with
  `gdino.source.name/sha256`, the tokenizer vocabulary embedded as
  `gdino.tokens`, and the per-layer deformable sampling grids as constants.
  F16/Q8_0 are produced afterwards by `aicore_gguf_quantize`, never by the
  converter.
- **P1 public ABI** (`core/AICore/include/aicore/gdino_capi.h`, C only):
  - `aicore_gdino_options` (extensible, `struct_size` + `abi_version`):
    `model_path`, `text_prompt`, `box_threshold` (default 0.25),
    `text_threshold` (default 0.20), `max_boxes`, `n_threads`, `device`.
  - `aicore_gdino_context_create/destroy/ready`, `last_error`,
    `context_device`.
  - `aicore_gdino_run(ctx, view, prompt)` → `aicore_gdino_result` owning
    typed boxes: `aicore_gdino_box { float quad[8]; float score; char
    phrase[128]; }` plus per-phrase token span indices; released by
    `aicore_gdino_result_free`.
  - `aicore_pipeline_timings` on every result (preprocess / encode /
    cross-modality / head / e2e; only truthful stages set `valid_fields`).
  - Shutdown delegates to `aicore_runtime_shutdown()`; contexts are
    created and destroyed on the owning worker thread.
- **Errors**: no model file, digest mismatch, empty/over-length prompt,
  prompt exceeding the token budget, image smaller than the 800-side
  window — all surfaced through `last_error` with a stable prefix
  (`gdino: ...`).
- **Backend matrix**: CPU (reference), CUDA, Vulkan on the same pinned
  fixtures; the deformable-attention op must exist in all three dispatch
  tables before P2 starts. Metal follows the standalone single-backend
  rule.
- **Numeric gate (P2)**: against the HF `transformers` reference on pinned
  fixtures (COCO val2017 subset chosen by the converter manifest): per-box
  best-IoU ≥ 0.99 against the reference detections at the same thresholds,
  per-box score |Δ| ≤ 0.01, phrase argmax identity, plus the standard
  repeat/recreate determinism and stability fingerprints. Quantized rows
  add the stored-weight diff budget like every other task.
- **P3 picker**: qAnyLabeling gains a text-prompt detection entry only
  after G2; the annotation result maps to rectangles with the matched
  phrase as the label, exactly like the YOLO-World flow it supplements.

## 2. `dfine` / `dfine_seg` — closed-set DETR-family detection

- **Input ownership**: identical borrowed `aicore_image_view` rules; no
  text tower (closed-set COCO classes embedded in the GGUF).
- **P0 source**: `ustc-community/dfine-medium_coco` (and `_seg` when the
  segmentation head ports); converter walks the HGNetv2 backbone, CCFM
  fusion and the D-FINE decoder (FDR + GTG distributions instead of
  ordinary regression heads).
- **GGUF layout**: `dfine-medium-coco-f32.gguf` with `dfine.source.*`,
  class names as `dfine.classes`, segmentation prototypes as a separate
  tensor section when `_seg`.
- **P1 public ABI** (`aicore/dfine_capi.h`): mirrors the RF-DETR result
  shape so the plugin worker can share its mapping code —
  `aicore_dfine_box { float quad[8]; float score; int32_t class_id; }`
  (plus `mask` prototype indices for the `_seg` variant),
  `aicore_dfine_options { model_path, score_threshold, n_threads, device,
  struct_size, abi_version }`, create/destroy/ready/run/last_error/
  result_free/timings exactly like `gdino`.
- **Backend matrix**: CPU/CUDA/Vulkan; D-FINE uses scaled dot-product
  attention and standard conv/linear ops plus the shared deformable
  operator, so the operator work is the same patch set as family 1.
- **Numeric gate (P2)**: HF `transformers` reference on the pinned COCO
  subset: best-IoU ≥ 0.99 per box, class argmax identity, score |Δ| ≤
  0.01, segmentation mask IoU ≥ 0.99 for the `_seg` rows; determinism and
  stability fingerprints identical to the other tasks.
- **P3 picker**: closed-set detection/segmentation entries follow the
  RF-DETR worker mapping (rectangles/masks with model labels).

## 3. `florence2` — local vision-language tasks (caption / OCR / OD / grounded caption)

- **Input ownership**: borrowed `aicore_image_view`; the task prompt is one
  of the fixed Florence-2 task tokens (`<CAPTION>`, `<OCR>`, `<OD>`,
  `<CAPTION_TO_PHRASE_GROUNDING>`, …) selected through an option enum, not
  a free string, so the decoder prompt is byte-reproducible.
- **P0 source**: `microsoft/Florence-2-base` (0.23B). The DaViT vision
  encoder plus the BERT-family language decoder are converted into a GGUF
  pair (`florence2-base-f32.gguf` vision + decoder in one file; the
  projector-less design differs from the llm catalog's mmproj pattern and
  is declared in the metadata as `florence2.layout = "encdec"`).
- **P1 public ABI** (`aicore/florence2_capi.h`):
  - `aicore_florence2_task` enum (CAPTION, CAPTION_MORE_DETAILED, OCR,
    OD, DENSE_REGION_CAPTION, GROUNDED_CAPTION).
  - `aicore_florence2_options { model_path, task, max_new_tokens,
    n_threads, device, struct_size, abi_version }`.
  - `aicore_florence2_run(ctx, view)` → `aicore_florence2_result`
    owning `text` (caption/OCR) and typed `regions` (boxes + phrase for
    OD / grounded rows), plus common timings; `result_free`,
    `last_error`, readiness, shutdown semantics identical to `llm`.
- **Interpreter route**: the architecture is outside libmtmd, so the task
  carries its own topology-driven interpreter following the qOCR
  interpreter precedent (op vocabulary declared in the GGUF, parity
  fixtures generated from the HF reference before any GGUF is published).
- **Backend matrix**: CPU first (interpreter reference), then CUDA/Vulkan
  through ggml graph scheduling; the attention ops are standard ggml ops,
  so no new operator is required — this family's cost is the interpreter
  and the token/region decoders.
- **Numeric gate (P2)**: HF reference on pinned fixtures — greedy text
  equality for caption/OCR (single-thread contract like the llm matrix),
  box IoU ≥ 0.99 with reference regions for OD rows, phrase identity for
  grounded rows; quantized rows follow the stored-weight budgets.
- **P3 picker**: one "Florence-2" picker family with a task selector; OCR
  rows write text shapes exactly like PP-OCR, OD rows write rectangles
  with phrase labels, captions land in the image-level `description`
  (the AI Describe sink), so all outputs reuse existing annotation flows.

---

## Sequencing

1. `florence2` first: no new ggml operator, the interpreter pattern is
   proven by qOCR, and it directly upgrades the annotation features
   (local OCR/OD/captioning beyond SmolVLM).
2. `dfine` second: closed-set only, reuses the detector result contract,
   and its converter validates the shared deformable-attention patch.
3. `grounding_dino` last: the text tower adds the prompt-token machinery
   on top of the same operator work.

Each step publishes through the standard ladder (catalog + digests +
manifest scenarios + runner rows) and none of it may claim speed or
accuracy before the P2 evidence exists on all three backends.
