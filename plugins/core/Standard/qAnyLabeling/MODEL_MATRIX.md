# qAnyLabeling AICore model matrix

The model picker is driven directly by the public AICore catalogs at runtime.
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
| YOLO-World | Text prompted detection | Typed boxes | CLIP text encoder |
| YOLOE | Text, prompt-free and SAVPE visual prompts | Typed masks | Text encoder except prompt-free; SAVPE weights for visual mode |
| RF-DETR | Detection and instance segmentation | Typed boxes and masks | None |
| SAM 2.1 / SAM3 | Point, box and text interactive segmentation | Editable masks and polygons | SAM3 text encoder for text mode |
| RMBG-2.0 | Foreground extraction | Editable foreground mask/polygon | None |
| GKD | Text, top-down ROI and one-shot visual keypoints | Grouped editable points | Optional support image/text |
| FaceDetect + landmark | Face boxes and dense landmarks | Grouped boxes and points | Paired landmark GGUF |

The table describes the task roles consumed by qAnyLabeling. It excludes
catalog roles without a matching annotation workflow (for example ReID
embeddings). The following are validation states, not claims that every
published file has passed a real-data accuracy gate:

- **Picker coverage:** supported task roles returned by the catalog APIs are
  enumerated; quantization and backend are selected per entry.
- **Real smoke coverage:** see the integration list in [README.md](README.md);
  it contains two-forward checks for representative CPU, CUDA and Vulkan rows.
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

## AICore catalogs outside this 2D labeling workflow

These AICore catalogs are included in the repository-wide model inventory, but
their result contracts are not image annotations and therefore are not shown in
the qAnyLabeling picker. Their model matrices remain owned by the corresponding
plugin or by a future annotation workflow:

| AICore catalog | Current owner or status | Why it is not a qAnyLabeling model |
| --- | --- | --- |
| LoMa / LightGlue | qAnyLabeling paired-image bridge | Produces feature descriptors and image-to-image correspondences; accepted matches become point annotations |
| Gaussian | `qFreeSplatter` | Produces 3D Gaussian scene data |
| LingBot | `qLingbotMap` | Streams RGB-D reconstruction and map state |
| TRELLIS | `qTrellis` | Generates 3D assets and meshes |
| SAM3D | `qSAM3D` | Produces 3D masks/assets from images |

The qAnyLabeling LoMa bridge provides the two-image pairing view, correspondence
review, model/quantization/backend selectors, and explicit conversion from
accepted matches to annotation geometry. Its real CPU fixture test is
`test_qanylabeling_loma_bridge`.

Source configurations without a matching AICore contract are intentionally not
added to this table. Their exact families, including VQA, Grounding and RAM,
are tracked in [TODO.md](TODO.md).
