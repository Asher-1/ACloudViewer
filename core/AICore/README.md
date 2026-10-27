# AICore

AICore is ACloudViewer's monolithic inference runtime. Plugins, reconstruction
code, tools, and bindings link one shared library and consume a stable C ABI;
ggml and its CPU/GPU backends remain private implementation details.

## Documentation Ownership

| Need | Source of truth |
|---|---|
| Runtime topology, ownership, and data flow | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Model, parity, and performance tests | [`tests/TESTING.md`](tests/TESTING.md) |
| Build switches | [`../../BUILD.md`](../../BUILD.md), [`../../cmake/AICoreOptions.cmake`](../../cmake/AICoreOptions.cmake) |
| ggml version and archive digest | [`../../3rdparty/llama/llama_cpp.cmake`](../../3rdparty/llama/llama_cpp.cmake) |
| Ordered downstream ggml changes | [`../../3rdparty/llama/llama-patches/manifest.yaml`](../../3rdparty/llama/llama-patches/manifest.yaml) |
| Agent engineering contract | [`../../.agents/skills/acloudviewer-aicore-plugin/SKILL.md`](../../.agents/skills/acloudviewer-aicore-plugin/SKILL.md) |

Do not duplicate current version numbers, patch counts, or benchmark snapshots
outside their owning source. Historical reports do not override current code or
test output.

## Dependency Boundary

Consumers link only `AICore` and include only `include/aicore/`:

```cmake
target_link_libraries(my_app_or_plugin PRIVATE AICore)
```

Do not include ggml headers from plugins, COLMAP, Python bindings, or public
tools. Exceptions, STL containers, Qt/OpenCV types, backend handles, and ggml
types must not cross the C ABI.

```text
Qt plugin / reconstruction / tool / binding
                    |
             public aicore C ABI
                    |
            one libAICore shared library
                    |
      task sessions + common runtime services
                    |
       private ggml core and backend modules
```

`AICore_capi_link` exists only to close direct C-API executable links when
dynamic backend loading leaves backend symbols unresolved. It is not a public
replacement for `AICore`.

## Common Contracts

The preferred decoded-image path is:

```text
borrowed pixels + width/height + row stride + channel format
  -> aicore_image_view
  -> task preprocess/upload
  -> typed result
```

- [`image_view.h`](include/aicore/image_view.h) supports RGB8, RGBA8, GRAY8,
  BGR8, and BGRA8 without requiring tightly packed rows.
- [`pipeline_timing.h`](include/aicore/pipeline_timing.h) defines common
  preprocess, inference, postprocess, optional serialization, and end-to-end
  fields. Consumers must inspect `valid_fields`.
- [`runtime_capi.h`](include/aicore/runtime_capi.h) provides caller-owned
  cancellation, per-device scheduling, and idempotent
  `aicore_runtime_shutdown()`.
- Task-specific typed results own their allocation contract. Compatibility
  path/RGB/JSON functions are not the preferred plugin inference path.

## Public Modules

| Header | Responsibility |
|---|---|
| `backend_capi.h` | Device discovery, resolution, warmup, capabilities |
| `runtime_capi.h` | Cancellation, device queues, process cleanup |
| `model_catalog_capi.h` | Shared read-only model metadata (role, URL, digest) for plugin/UI consumers |
| `image_view.h` | Borrowed decoded-image input |
| `pipeline_timing.h` | Common timing ABI |
| `export.h` | `AICORE_CAPI` / `AICORE_CXX_API` export macros |
| `runtime_raii.h` | Header-only C++ RAII helpers (`DeviceTaskLock`, `CancelScope`); Qt-free |
| `depth_image.h` | Qt convenience layer for depth (the single documented Qt exception) |
| `inference_log.h` | CVLog logging helpers; intentionally not umbrella-included (depends on `CVLog.h`) |
| `asset_digests.h` | Generated SHA-256 source table; plugins consume digests through exported catalogs/shared integrity helpers |
| `aliked_capi.h` | ALIKED feature extraction |
| `deeplsd_capi.h` | DeepLSD line extraction |
| `depth_capi.h` | Depth, pose, reconstruction, and export |
| `facedetect_capi.h` | Detection, landmarks, analysis, embedding, verification |
| `gaussian_capi.h` | FreeSplatter Gaussian reconstruction |
| `gkd_capi.h` | GKDT general keypoint detection (text / visual / multimodal prompts) |
| `lightglue_capi.h` | Sparse feature matching |
| `lingbot_capi.h` | LingBot-Map (GCT) streaming 3D reconstruction: depth, pose, native sky masking |
| `llm_capi.h` | Local chat LLM / VLM generation over the pinned libllama + libmtmd (optional mmproj, streaming, cancel, stop sequences); contract in [`docs/llm_vlm_task_design.md`](docs/llm_vlm_task_design.md) |
| `loma_capi.h` | LoMa feature detection, description, and matching (COLMAP-aligned) |
| `rfdetr_capi.h` | RF-DETR detection and segmentation |
| `rmbg_capi.h` | RMBG alpha matte and RGBA composition |
| `sam3_capi.h` | SAM3 segmentation and tracking |
| `sam3d_capi.h` | SAM 3D Objects image-to-3D generation (Gaussian PLY + FlexiCubes mesh) |
| `tagging_capi.h` | RAM/RAM++ tags, logits and scores; Tag2Text captions |
| `attribute_capi.h` | 本地 PULC person/vehicle 多属性：只读 stride-aware 图像、完整 Sigmoid 概率、独立持有的 typed 属性与全图几何、公共 timing/cancel/shutdown；本地 F32 recipe/cache 不等于已发布 catalog 或 release 资格 |
| `relation_capi.h` | RelateAnything typed edges, raw predicate/pair logits and native CLIP text embeddings | `aicore_image_view` plus borrowed object regions |
| `trellis_capi.h` | TRELLIS image-to-3D generation |
| `yolo_capi.h` | YOLO detection, segment, pose, OBB, semantic, classify, depth, and prompts |

The individual task header is authoritative. The umbrella `aicore.h` is a
complete convenience include for public task contracts and must be updated
when a task is added. New code should still include only the headers it uses.
`inference_log.h` is deliberately excluded because it depends on `CVLog.h`.

Tagging keeps the original `aicore_tagging_request` layout and by-value default
request ABI frozen. The original inference entry retains official Pillow
bilinear/antialias preprocessing. Consumers requiring X-AnyLabeling's uint8
OpenCV INTER_LINEAR resize use `aicore_tagging_default_request_v2()` and
`aicore_tagging_infer_image_v2()` with `AICORE_TAGGING_RESIZE_OPENCV`.

## Source Layout

```text
core/AICore/
  include/aicore/       public ABI only
  src/common/           runtime, backend registry, logging, shared validation
  src/tasks/<task>/     task-owned loader, graph, preprocess, postprocess, C API
  tests/common/         shared ABI/image/timing/runtime contracts
  tests/<task>/         task contracts, parity, model and performance tests
  scripts/              complete validation runner and manifest
```

A new task is added to the existing `AICore` target in
`core/AICore/CMakeLists.txt`. Do not create a task-specific shared library.

`common/paddle_graph` 是 OCR、table 与 attribute 共享的 Qt-free 图解释器 owner；
OCR 的 CTC 与专属预处理仍在 OCR task，属性 threshold/argmax 和模型 source 身份
仍在 attribute context。公共 `aicore_attribute_preset_at` 只描述本地 F32 recipe，
`aicore_attribute_model_cache_dir` 沿共享 `CLOUDVIEWER_DATA_ROOT` 返回
`extract/attribute_models`，不创建目录或下载文件。未发布文件不进入下载 catalog。
真实 C ABI 测试接受 `AICORE_TEST_ATTRIBUTE_{PERSON,VEHICLE}_{GGUF,REFERENCE}`、
`AICORE_TEST_ATTRIBUTE_IMAGE` 和 `AICORE_TEST_DEVICE`；参照是 build-local 生成的
pinned source 输出，不提交模型 logits/真值到源目录。缺少模型或参照明确失败。
三后端冻结 11 图公共留出仍各 48/66、FAIL，数值/属性门与 Q8 失败完整保留；
当前仅隔离候选验收，正式构建和完整 release 状态见
[llama 验证收据](../../3rdparty/llama/README.md#current-gate-status)。

## Runtime Artifacts

| Artifact | Policy |
|---|---|
| `libAICore` | The only inference library linked by consumers |
| ggml core/base | Private, version-locked AICore runtime |
| ggml CPU backend | Required CPU fallback |
| Vulkan backend | Linux/Windows default when configured; not supported on macOS |
| Metal backend | macOS default |
| CUDA backend | Optional; dynamically loaded without a mandatory `libAICore` CUDA dependency |
| SYCL/OpenCL | Explicit opt-in development backends |

Backend modules are loaded from the AICore runtime directory. Do not copy
`libggml*.so` into `bin/plugins/`; that directory is reserved for application
plugin modules.

Auto device order is CUDA -> Vulkan -> CPU on Linux/Windows when CUDA is built,
Vulkan -> CPU otherwise, and Metal -> CPU on macOS. Device discovery alone does
not prove that a model graph is accurate on that backend; parity tests remain
authoritative.

## Build

```bash
cmake -S . -B build_app \
  -DAICore_ENABLED=ON \
  -DAICore_BUILD_TESTS=ON
cmake --build build_app --target AICore aicore-contract-tests -j4
```

Use only `AICore_*` public options. Internal `GGML_*` cache entries are derived
by `cmake/AICoreOptions.cmake`; do not pass them as user configuration.
统一库编译所有公共 task 契约。task-specific plugin 开关仅控制 Qt 消费者，
不从 AICore 删除 task 契约。

ggml is an ExternalProject. Persistent modifications are ordered unified diff
patches in `3rdparty/llama/llama-patches/manifest.yaml`. Extracted sources under any
`build*/ggml/` tree are disposable and must not be committed.

## Test Entry Points

```bash
# Fast public ABI/runtime contracts, no model assets
cmake --build build_app --target aicore-contract-tests -j4

# All C API tests selected by their CTest label
ctest --test-dir build_app -L capi --output-on-failure -j1

# Real-asset matrix for one backend (default light tier; add --full for the
# complete matrix including all sam3/trellis models)
python3 core/AICore/scripts/validate_all.py \
  --build build_app --backend cuda \
  --output build_app/Testing/aicore_validation.json
```

Missing assets use exit 77 only in tests configured to skip. A complete
validation claim must reject missing task/model/quantization/backend rows. See
[`tests/TESTING.md`](tests/TESTING.md) for the evidence and benchmark protocol.
