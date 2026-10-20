# qAnyLabeling AICore model matrix

The model picker is driven directly by the public AICore catalogs at runtime.
Unpublished OCR pipelines are described separately by public task-owned recipes;
their local-file support does not create downloadable catalog entries.
For supported task roles it exposes every published user-selectable weight and
quantization, plus the paired text/metric dependency when required. Every
backend returned by `aicore_device_count/at` is available. There is no second
plugin model list to drift from AICore. A dependency-only catalog row such as
`depth-anything-nested-metric.gguf` is intentionally not a separate model
family: the nested AnyView workflow resolves and verifies it automatically.

| AICore catalog family | qAnyLabeling workflow | Result | Prompt/dependency |
| --- | --- | --- | --- |
| DA3 depth / nested AnyView | Depth image and metric depth sidecar | Typed depth map | Nested metric GGUF when selected |
| DeepLSD | Wireframe and line proposals | Editable line/rectangle shapes | None |
| ALIKED | Local keypoint proposals | Editable point shapes | None |
| YOLO (detect, segment, semantic, pose, OBB, classify) | Detection, instance/semantic masks, pose, OBB and image flags | Typed boxes, masks, points and flags | None |
| YOLO classify in Image Classifier | Local model Run/AutoRun over document flags | Trained category vocabulary, one active category; preserves existing shapes | Shared classification model/quantization/backend selection |
| YOLO-World | Text prompted detection | Typed boxes | CLIP text encoder |
| YOLO-World + RAM/RAM++ | Per-object tags after integer detection crops | Replaced rectangles with descriptions; preserves image tags | Catalog detector/text encoder and resident tagging model; shared language/filter options |
| YOLOE | Text, prompt-free and SAVPE visual prompts | Typed masks | Text encoder except prompt-free; SAVPE weights for visual mode |
| RF-DETR | Detection and instance segmentation | Typed boxes and masks | None |
| SAM 2.1 / SAM3 | Point, box and text interactive segmentation | Editable masks and polygons | SAM3 text encoder for text mode |
| RMBG-2.0 | Foreground extraction | Editable foreground mask/polygon | None |
| GKD | Text, top-down ROI and one-shot visual keypoints | Grouped editable points | Optional support image/text |
| FaceDetect + landmark | Face boxes and dense landmarks | Grouped boxes and points | Paired landmark GGUF |
| RAM / RAM++ / Tag2Text | Image tags and optional caption | Image-level tags/description | Public tagging catalog; no object-box substitution |
| PP-OCRv4 Chinese/English + CDLA layout | Detect/recognize or selected-shape recognition; explicit optional layout | Text descriptions and real layout classes/reading order | Paired detector; optional catalog layout; optional local direction classifier, enabled by default |
| Local source Chinese PP-OCRv4 / Japanese PP-OCRv3 / PP-OCRv5 / PP-OCRv6 Tiny, Small, Medium | Exact source detector/recognizer recipes with shared 0/180 text classifier | Editable text quads/descriptions; typed orientation results | Twelve converted F32 GGUFs, source defaults and embedded recognizer dictionaries; publication pending |
| RelateAnything | Relationships between selected/detected rectangles | Relations referencing existing objects | Public relation catalog and optional catalog-backed detector |

The table describes the task roles consumed by qAnyLabeling. It excludes
catalog roles without a matching annotation workflow (for example ReID
embeddings). The following are validation states, not claims that every
published file has passed a real-data accuracy gate:

- **Picker coverage:** supported task roles returned by the catalog APIs are
  enumerated; quantization and backend are selected per entry.
- **Real smoke coverage:** see the integration list in [README.md](README.md);
  it contains two-forward checks for representative CPU, CUDA and Vulkan rows.
  The 2026-10-17 RAM/OCR follow-up also records 90 passing developer-profile
  rows across these three backends: all 12 current OCR catalog weights and
  nine tagging weights, including six RAM/RAM++ OpenCV-resize scenarios.
  This selected scope does not close the full task/dataset release gate.
- **Release gate:** the repository report contains 122 real CUDA catalog rows
  and the qAnyLabeling rows in that report pass. The expanded CPU/Vulkan
  cross-product was started against all 161 candidate rows; completed rows
  returned `rc=0`, but the run was intentionally interrupted while the slow
  SAM3/YOLO rows were still executing, so no `qany_*_matrix.json` is claimed
  as a complete report. The retained per-row probes under
  `build_app/Testing/probes/` are useful for resuming, but are not a release
  verdict. Unavailable backend/model rows must be recorded as unavailable,
  never treated as a silent pass. A source-app latency/memory comparison is
  still required before claiming a speedup.

The 2026-10-17 OCR extension verifies twelve F32 graphs against independent
ONNX Runtime outputs, ten upright/180-degree real word crops per recognizer,
and the actual pinned source TextSystem on one real page. Source output counts
are 74 for Chinese v4, 77 for Japanese v3, 81 for v5 and 73 for each v6 recipe; exact text/quad gates are recorded
by the common runner, separately from graph gates. F16/Q8 candidates with failed
numerical gates are withheld from the six new recipe selectors. A one-page
source comparison does not establish dataset accuracy or release performance.
PP-OCR KIE entity/link import/export is an annotation format workflow, rather
than a new inference model; active source model registration contains no SER/RE
pipeline. Selected polygons use source integer/minimum-area normalization,
reading-order batches and input-index ownership, with six independent ten-region
source comparisons in the common manifest. Composite picker configurations and
complete OCR-window visuals stay in TODO.

## AICore catalogs outside this 2D labeling workflow

These AICore catalogs are included in the repository-wide model inventory, but
their result contracts are not image annotations and therefore are not shown in
the qAnyLabeling picker. Their model matrices remain owned by the corresponding
plugin or by a future annotation workflow:

| AICore catalog | Current owner or status | Why it is not a qAnyLabeling model |
| --- | --- | --- |
| LoMa | qAnyLabeling paired-image bridge | Detector/descriptor/matcher correspondences; accepted matches become point annotations |
| LightGlue matcher | qAnyLabeling paired-image bridge | ALIKED proposals are matched through the separate LightGlue worker; SIFT remains intentionally unexposed |
| Gaussian | `qFreeSplatter` | Produces 3D Gaussian scene data |
| LingBot | `qLingbotMap` | Streams RGB-D reconstruction and map state |
| TRELLIS | `qTrellis` | Generates 3D assets and meshes |
| SAM3D | `qSAM3D` | Produces 3D masks/assets from images |

The qAnyLabeling paired-image bridge provides the two-image pairing view,
correspondence review, model/quantization/backend selectors, and explicit
conversion from accepted matches to annotation geometry. LoMa and LightGlue
have separate bridge workers and real fixture tests; both cover CPU/CUDA/Vulkan
where their model contract permits. Its LoMa fixture test is
`test_qanylabeling_loma_bridge`: the pinned shared Sacre-Coeur pair and F16
detector/DeDoDe-G/B-matcher pass on CPU/CUDA/Vulkan, with 578/581/578 matches.
This verifies bridge configuration, not every quantization or an independent
upstream numeric match. The LightGlue test is
`test_qanylabeling_lightglue_bridge`; it verifies the ALIKED-to-LightGlue path,
while SIFT remains intentionally outside the picker.

Remote ChatBot, VQA and arbitrary-category Image Classifier use a shared
user-configured OpenAI-compatible HTTP service. Their models come from that
service's `/models` endpoint or a user-entered ID; they do not add local AICore
model/catalog entries. Image Classifier AI Assistant/AutoRun use the source
category prompt and JSON flags, while its Local model menu retains the YOLO
classification path above. This does not supply a local Florence2/VLM graph
or establish parity with any particular remote model's outputs.

Source configurations without a matching local AICore contract are intentionally
not added to this table. Their exact families, including local VQA/VLM and
Grounding, are tracked in [TODO.md](TODO.md).
