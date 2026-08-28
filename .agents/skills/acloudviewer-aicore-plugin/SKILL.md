---
name: acloudviewer-aicore-plugin
description: ACloudViewer AICore plugin integration guide. Complete specification for adding a new task's C API, implementing ggml inference, wiring up the CMake build system, and writing contract tests. Use when creating a new AICore inference task or integrating a new plugin.
---

# ACloudViewer AICore Plugin Integration Guide

This skill aggregates the full specification for wiring `core/AICore/` into the ACloudViewer plugin ecosystem. When adding a new inference task to AICore or integrating a new AICore-dependent plugin, follow the sections below in order.

Underlying rules: `.agents/rules/acloudviewer-ggml-aicore.mdc` (ggml build), `.agents/rules/acloudviewer-plugin-dev.mdc` (plugin architecture).

## 0. Global Principle: Conclusions Must Be Based on Facts (Mandatory)

Every conclusion and recommendation must cite verifiable evidence: code file paths, line numbers, function signatures, CMake variable names, version numbers, and actual output of `rg`/`ctest`/`grep`/`python3` commands. Guessing from experience or deriving conclusions by subjective estimation is forbidden.

- Verify evidence first (read code, run commands), then conclude
- Anything that cannot be verified must be explicitly marked "unverified" — never stated as fact
- When a document contradicts the code, **the actual code wins**, and the discrepancy must be called out in the conclusion
- Every "precedent" cited in this document can be reproduced with a command (e.g. `rg -n` to locate a function/constant)

---

## 1. AICore C-API Design Specification

### Parameter Encapsulation

Functions with more than 6 input parameters must be encapsulated into a struct. Two landed precedents in the repo (signatures verifiable via grep):

| Precedent file | Encapsulated real signature (verify: `rg -n "func\(" core/AICore/include/aicore/*.h`) |
|---|---|
| `core/AICore/include/aicore/depth_capi.h:226` | `int aicore_depth_depth_dense(aicore_depth_ctx* ctx, const char* image_path, aicore_depth_dense_result* out)` — outputs (depth/conf/sky/ext/intr/is_metric) go into `aicore_depth_dense_result` (depth_capi.h:113-122), freed via `aicore_depth_dense_result_free` (depth_capi.h:126) |
| `core/AICore/include/aicore/gaussian_capi.h:251` | `int aicore_gaussian_tree_overlap(const float** pairs, int n_pairs, const aicore_gaussian_geometry* geom, float opacity_threshold, const aicore_gaussian_merge_options* merge_opts, aicore_gaussian_point** out, size_t* n_out, int* n_nodes_out)` — block/overlap/max_levels/spacing/cap folded into `aicore_gaussian_merge_options` (gaussian_capi.h:171-175) |

Rules:
- Outputs go into a result struct with a `_result_free` function (`_result_free` internally calls `aicore_<task>_free_buffer`; see depth_capi.h:126)
- Config/options go into an options struct set via builder pattern; every setter must be a no-op on NULL (each yolo_capi.h setter comment states this; covered by contract tests)
- Headers expose only opaque handles (`struct aicore_<task>_ctx`); implementation details stay in `.cpp` files
- Verification: `python3 core/AICore/tests/check_capi_coverage.py` scans all headers with the regex `\b(aicore_[a-z0-9_]+)\s*\(` (script L40)

### ABI Version Management

Every task must define `aicore_<task>_abi_version(void)`:

```c
AICORE_CAPI int aicore_<task>_abi_version(void);  // bump on breaking ABI change
```

Breaking changes (signature changes, struct field changes, function removal) must bump the version. The return value is asserted in contract tests.

### Memory Ownership Convention

Single release entry point:

```c
AICORE_CAPI void aicore_<task>_free_buffer(void* p);  // the only free function
```

Exporting multiple differently-named free functions (e.g. `free_string`, `free_floats`, `free_bytes`) is forbidden. Every API that returns malloc'd memory must note "free with aicore_<task>_free_buffer" in its doc comment.

### Error Handling

```c
// error message stored in ctx
AICORE_CAPI const char* aicore_<task>_last_error(const aicore_<task>_ctx* ctx);
// C-API returns int: 0=success, -1=error
AICORE_CAPI int aicore_<task>_do_something(aicore_<task>_ctx* ctx, ...);
```

Implementation: `ctx->last_error = "reason";` and return -1. Direct `fprintf`/`printf` of errors to stderr is forbidden (see logging rules).

### Header Function Name Uniqueness

`check_capi_coverage.py` matches all public APIs with the regex `\b(aicore_[a-z0-9_]+)\s*\(`. Naming style: `aicore_<task>_<verb>_<noun>`.

---

## 2. Explicit Configuration Rule (No getenv/setenv Logic Control)

**All flow-control switches must be passed explicitly through options structs. Reading environment variables with getenv/setenv to control pipeline logic is forbidden.**

### Why

Environment variables are global implicit state:

- Opaque flow: callers cannot see what switches affect the inference chain (debug switches hidden in the environment)
- Non-reproducible tests: the same binary behaves differently across shells
- Concurrency-unsafe: one thread's setenv affects all threads
- Migration legacy: upstream repos often hide debug switches in env; porting them directly brings implicit state into AICore

### Existing AICore Precedents

- depth: historical `DA_FUSED` / `DA3_FORCE_JOINT_MV` / `DA_PROFILE` env vars → `aicore_depth_options_set_fused_graph` / `_set_force_joint_multiview` / `_set_profile_logging`. Evidence: the ABI comment in `core/AICore/include/aicore/depth_capi.h:25-36` states "AICore reads no environment variables for logic control"; the setters are declared at depth_capi.h:57-66.
- rmbg: upstream `RMBG_VULKAN_MODE` / `RMBG_STRICT_MATH` / `RMBG_VULKAN_*` → `aicore_rmbg_options_set_math_profile` and other setters. Evidence: `core/AICore/include/aicore/rmbg_capi.h:67-104`, comments annotate each one as "Replaces the ... environment variables of the upstream port" (e.g. RMBG_VK_QKV_LAYOUT → `aicore_rmbg_options_set_vulkan_qkv_layout`, rmbg_capi.h:93-95).

When porting upstream code, any `getenv("XXX")` must become an options field + setter, with a header comment recording "this setter replaces the upstream XXX environment variable".

### Enforced Check (CTest Hard Gate)

`core/AICore/tests/CMakeLists.txt:580-587` registers `test_no_env_getenv`: `bash core/AICore/tests/check_no_env_getenv.sh core/AICore/src` statically scans all .cpp/.c/.hpp/.h files; only two files are whitelisted (script L18-21):

1. `src/common/data_root_util.cpp` — data-root path configuration (`CLOUDVIEWER_DATA_ROOT`)
2. `src/common/ggml_env_bridge.cpp` — the only sanctioned env writer

Any `getenv|secure_getenv|setenv|unsetenv|putenv|_putenv_s` call in task code fails the check (script L24). Local verification:

```bash
bash core/AICore/tests/check_no_env_getenv.sh core/AICore/src   # expected: "reads/writes environment only through the sanctioned bridge"
ctest -R test_no_env_getenv --output-on-failure                  # POSIX only (no .sh launcher on Windows)
```

### Exceptions (exactly two; no third kind for new code)

1. `core/AICore/src/common/ggml_env_bridge.cpp`: ggml snapshots environment variables at init (actually handled: `GGML_VK_DISABLE_F16` / `GGML_VK_DISABLE_COOPMAT` / `GGML_METAL_GRAPH_OPTIMIZE_DISABLE` / `RMBG_VK_SCALAR_DIRECT_CONV` / `NVIDIA_TF32_OVERRIDE` etc., see ggml_env_bridge.cpp:60-79); AICore must translate options into the env ggml expects. This is ggml's hard interface requirement, concentrated in the single bridge file.
2. `core/AICore/src/common/data_root_util.cpp`: the data root shared across plugins (`CLOUDVIEWER_DATA_ROOT`, a path configuration, not a flow switch).

---

## 3. Resource Management Rules

### shutdown Implementation

`aicore_<task>_shutdown()` must perform real cleanup — an empty implementation is forbidden. Four task precedents (all call `aicore::runtime::purge_inactive_backend_leases()`): yolo (`core/AICore/src/tasks/yolo/capi.cpp:837`), rmbg (`rmbg/capi.cpp:483`), sam3 (`sam3_capi.cpp:1040`), trellis (`aicore_trellis_capi.cpp:1484`):

```cpp
AICORE_CAPI void aicore_yolo_shutdown(void) {
    aicore::runtime::purge_inactive_backend_leases();
}
```

`purge_inactive_backend_leases()` is defined at `core/AICore/src/common/ggml_backend_registry.cpp:181` (declared at `ggml_backend_registry.hpp:82`): under a mutex it walks the registry and removes backend leases whose owners are gone (`it->second.expired()`). Verify: `rg -n "purge_inactive_backend_leases" core/AICore/src` should hit 1 definition + 4 shutdown calls.

### Host Weight Copy Management

Design pattern: lazy release + on-demand reload. Real signatures (`core/AICore/include/aicore/yolo_capi.h:139-142`):

```cpp
int aicore_yolo_release_host_weights(aicore_yolo_ctx* ctx);  // drop host copies (device weights untouched); 0 ok / -1 no engine
int aicore_yolo_ensure_host_weights(aicore_yolo_ctx* ctx);   // reload from GGUF by offset; no-op when present
```

Implementation points:
- `HostTensor` records `file_type` / `file_offset` (original type and offset in the GGUF file)
- `prepare_host_weights` is extracted as a re-entrant function (Vulkan Q8→F16, CUDA F32→F16 conversions are idempotent)
- In `build_run_plan`, `if (!s->wbuf)` guarantees weights are uploaded only once; graph rebuilds reuse wbuf without touching host data
- **After inference, host weights can be safely released while device weights stay**, greatly reducing host memory peaks

### Graph Allocator Buffer Control

For real-time scenarios (video frames), provide the `keep_graph_buffers` option. Real signature and semantics (`core/AICore/include/aicore/depth_capi.h:67-73`):

```cpp
void aicore_depth_options_set_keep_graph_buffers(aicore_depth_options* opts, int enabled);
// ON: keep graph buffers reused (high-water VRAM), for repeated same-shape inference like video frames
// OFF (default): release graph buffers after each inference (VRAM peak = single graph), for multi-view/one-shot jobs
```

Companion VRAM release entry: `aicore_depth_release_gpu_working_memory(aicore_depth_ctx* ctx)` (depth_capi.h:252, "Drop ggml graph buffers and (when GPU offloading) device-resident weights").

---

## 4. Dependency Introduction Rules (Reuse Existing Capabilities)

### Core Principle

When integrating upstream code, **check first whether the repo already has an equivalent module; reuse it, never re-introduce a duplicate**. Introducing a new dependency = solving a named gap (not "that's how upstream wrote it").

### Confirmed Reusable Capabilities

| Need | Existing repo capability | Forbidden to introduce |
|---|---|---|
| Image decode (JPEG/PNG etc.) | Qt QImage (built-in codecs) | stb_image / direct libjpeg |
| Image encode/save | QImage / QPainter | stb_image_write |
| Image scale/crop | QImage scaled / copy | stb_image_resize |
| Inference runtime | ggml (3rdparty ExternalProject) | second ggml / ONNX Runtime |
| Linear algebra | Eigen (3rdparty) | hand-rolled matrix library |
| JSON | jsoncpp (3rdparty) | hand-rolled parser |
| Logging | AICORE_LOG_* → CVLog | private logging system |
| Model download | ecvModelDownloader (CVPluginAPI) | private downloader |
| Camera/video | video_base (shared plugin library) | private clock/decode |

### stb Case (Real Lesson)

face-detect upstream (InsightFace) depends on stb_image decoding; the AICore facedetect port switched to Qt QImage (`core/AICore/src/tasks/facedetect/image_io.cpp`, 68 lines). Same precedent: the comment in `core/AICore/src/tasks/yolo/yolo_image.hpp:11-15` states "the upstream stb_image load/save ... live on the plugin side (Qt QImage / QPainter)".

Verification commands (exclude comments and known exceptions):

```bash
# No stb implementation or include in AICore sources (comments don't count) — expect 0 hits:
rg -n 'STB_IMAGE_IMPLEMENTATION|#include [<"](stb_image|stb_image_write|stb_image_resize)' core/AICore/src
# Full plugin scan excluding known exceptions (qSIBR's xatlas 3rdparty and tinygltf's STB define):
rg -n 'STB_IMAGE_IMPLEMENTATION' core/AICore/src plugins --glob '!**/3rdparty/**' --glob '!plugins/core/Standard/qSIBR/**'
```

Current repo baseline: in `core/AICore/src` only the `yolo_image.hpp` comment contains the string "stb_image"; there is no implementation or header dependency. `plugins/core/Standard/qSIBR/3rdparty/xatlas/` and `3rdparty/find_dependencies.cmake:1345` (tinygltf's `STB_IMAGE_IMPLEMENTATION` define) are known exceptions.

### Checklist Before Introducing a New Dependency

1. Does the repo (including `3rdparty/`) already have a functionally equivalent module?
2. What does the existing module lack (format support? performance?) — fill the gap, don't replace the whole stack
3. Is the new dependency's license compatible with ACloudViewer (root project GPL-2.0-or-later, AICore MIT)?
4. Does the new dependency add build burden (CMake config, cross-platform patches)?

---

## 5. Coding Style and Naming (Consistent with AICore)

### Naming

| Category | Style | Examples |
|---|---|---|
| Functions/variables | snake_case | `prepare_host_weights`, `run_dense_impl`, `use_direct_conv` |
| Types | PascalCase | `HostTensor`, `ModelDef`, `EngineOptions` |
| Constants | k-prefix or UPPER_SNAKE | `kQuantCount`, `QK8_0` |
| C API | `aicore_<task>_<verb>_<noun>` | `aicore_yolo_set_detect_thresholds` |
| Macros | AICORE_ prefix | `AICORE_LOG_WARN`, `AICORE_CAPI` |

### Includes Use Absolute Paths (from module root)

Full paths from the `core/AICore/` root. **`../` relative paths and bare filenames are forbidden**:

```cpp
// correct
#include "aicore/depth_capi.h"     // public header: include/ root
#include "common/capi_utils.hpp"   // internal: src/ root
#include "tasks/yolo/yolo_common.hpp"

// wrong
#include "../tasks/yolo/yolo_common.hpp"
#include "yolo_common.hpp"
```

Verification (current repo baseline: 0 hits):

```bash
rg -n '#include "\.\./' core/AICore/src          # no ../ relative includes
rg -n '#include "(aicore|common|tasks)/' core/AICore/src | wc -l   # module-root paths should cover all internal includes
```

### Other

- C++17, RAII, `std::unique_ptr` with custom deleter, non-copyable sessions
- Raw pointers mean only non-owning views or C ABI opaque handles; ownership is explicit in the type/comment
- Never expose exceptions, STL, Qt, OpenCV, or ggml types across the C boundary
- Split loader/graph/postprocess into small task-specific classes; no multi-thousand-line single files
- No unrelated style rewrites of surrounding code; clang-format matches the existing format

---

## 6. Performance and Memory Rules (End-to-End Pipeline)

### Zero-Copy Principles

- Input borrowing: the C ABI receives a read-only stride-aware view, valid for the duration of the call; no ownership transfer
- Output reuse: hot loops avoid temporary vector allocations; reuse session scratch capacity; shrink logical size for small results instead of `shrink_to_fit` every frame
- No intermediate materialization: don't build packed RGB and then do a second CHW conversion; no JSON serialize/parse on the hot path; never write DB images to disk temporarily and re-decode

### Save Memory

- Weights uploaded once (`if (!s->wbuf)` in `build_run_plan` reuses wbuf); after inference the host copy can be released (`session_release_host_weights` / `aicore_<task>_release_host_weights`)
- Large buffers (mask/depth) use immutable result handles + borrowed views; queued-signal deep copies are forbidden
- mask/depth materialized on demand: never generate N × source_width × source_height data per frame

### Save VRAM

- Multi-view/one-shot tasks: `keep_graph_buffers=OFF` (single-graph peak)
- Real-time video: `keep_graph_buffers=ON` (buffer reuse)
- Call `aicore_<task>_release_gpu_working_memory` between views to drop graph buffers when needed

### End-to-End Acceleration

- Real-time video: single job + latest-wins, queue depth ≤ 2 (running + pending), results bound to source frame + generation
- Preprocess writes the persistent CHW staging directly from the stride-aware view, one upload
- detect reads back only the tensors the decoder needs; segment materializes masks only for selected detections
- Logs contain model/task/device/stage, never user-sensitive paths

---

## 7. Logging Integration Rules

All AICore output must go through the `AICORE_LOG_*` macro system, ultimately reaching the ACloudViewer Console via CVLog.

### Log Levels

```c
// level constants defined in core/AICore/src/common/aicore_log.hpp:29-32
#define AICORE_LOG_LEVEL_DEBUG 0
#define AICORE_LOG_LEVEL_INFO  1
#define AICORE_LOG_LEVEL_WARN  2
#define AICORE_LOG_LEVEL_ERROR 3

// usage (macros are unconditional; use aicore_log_at for runtime thresholds)
AICORE_LOG_DEBUG("yolo", "preprocess took %.2f ms", ms);
AICORE_LOG_INFO("depth", "loaded model: %s", name);
AICORE_LOG_WARN("rmbg", "fallback to CPU");
AICORE_LOG_ERROR("gaussian", "OOM during inference");
```

Note: under `AICore_HAS_CVLOG` the `AICORE_LOG_*` macros map directly to `CVLog::Print/PrintDebug/Warning/Error` (aicore_log.hpp:5-11) with no runtime filtering; thresholded logging must go through `aicore_log_at(level, tag, fmt, ...)` (aicore_log.hpp:42).

### Forbidden Behavior

- **No direct `fprintf(stderr, ...)` for errors or status on the inference/business path** — use `AICORE_LOG_*` / `aicore_log_at` (precedent: `ggml_env_bridge.cpp:55` fixed to `AICORE_LOG_WARN`)
- **No private log-level enums inside tasks** — use the shared layer `aicore_set_log_level` / `aicore_log_at`

The 5 allowed `fprintf(stderr)` exceptions (verify: every hit of `rg -n 'fprintf\(stderr' core/AICore/src` must fall into one category):

1. `aicore_log.hpp:15-16` — the `AICORE_LOG_*` macro fallback when `AICore_HAS_CVLOG` is absent (the logging system itself)
2. `ggml_backend_utils.hpp:113-129` — `#ifndef NDEBUG` gated debug-build diagnostics (not compiled in Release)
3. `yolo/backend.cpp:266-270`, `yolo_graph.cpp:838,888` — `[op profile]` / `[gap-prof]`, controlled by the explicit options `aicore_yolo_options_set_profile_ops` / `_set_profile_gaps` (yolo_capi.h:70-75)
4. `lightglue/quantize.cpp:227`, `common/simple_gguf_io.cpp:441` — CLI quantizer (`aicore_gguf_quantize`) output
5. `deeplsd/lsd.cpp:75` — legacy error handling from the upstream LSD library (prints before exit; don't propagate this pattern)

### Thread-Local Log Level

```cpp
// defined in the anonymous namespace of core/AICore/src/common/aicore_log.cpp:21
thread_local int tls_log_level = AICORE_LOG_LEVEL_INFO;
void aicore_set_log_level(int level);   // set the current thread's minimum level (aicore_log.cpp:27)
int aicore_get_log_level(void);         // query the current thread's level (aicore_log.cpp:29)
```

yolo's `logf` already delegates to the shared layer (aicore_log.cpp:19-20 comment: "Default INFO matches the historical yolo::tls_log_level behavior"); new tasks just use the `AICORE_LOG_*` macros.

### Underlying Implementation

`core/AICore/src/common/aicore_log.cpp` uses `CVLog::Print*` to reach the Console under `AICore_HAS_CVLOG` (aicore_log.cpp:41-56, level mapping in the switch); otherwise it falls back to `fprintf(stderr, ...)` (aicore_log.cpp:58). `AICore_HAS_CVLOG` is defined by `core/AICore/CMakeLists.txt:290` when linking CVCoreLib.

Verification (review command; each hit must fall into the 5 exceptions above):

```bash
rg -n 'fprintf\(stderr' core/AICore/src
```

> Tool note: `rg` = ripgrep (use `grep -rnE` as an equivalent if not installed).

---

## 8. ggml Modification Rules

**Directly modifying ggml sources in build directories is absolutely forbidden.** Full rules: `acloudviewer-ggml-aicore.mdc`. Condensed essentials below:

### ggml Version Lock (v0.18.1)

- ACloudViewer pins ggml to **v0.18.1** (`3rdparty/ggml/ggml.cmake:24` `set(GGML_VERSION "0.18.1")`, URL `https://github.com/ggml-org/ggml/archive/refs/tags/v0.18.1.tar.gz`, SHA256 `e9679cc9a8f0480ddc137b0a650df31b7c955e53ac6fdded1967aac36790c5e3`, ggml.cmake:26). **Upgrading or downgrading ggml is forbidden** — all 14 patches in manifest.yaml are generated against this version (git apply hunks anchor to v0.18.1 sources), so version drift breaks every AI plugin at once.
- The minimal patch for a new task must be **semantically de-duplicated** against the existing patch chain (do not re-introduce what rmbg/aliked already provide), split by file/operator into: public ggml API, CPU, CUDA, Vulkan, build dependency.

### Current Patch List (manifest.yaml is the single source of truth: 14 patches)

Verify: `rg -n "file:" 3rdparty/ggml/patches/manifest.yaml`. Current full list (manifest.yaml:9-36):

| Subdirectory | Content | Inert when |
|---|---|---|
| `aliked_merged/0001-vulkan-aliked.patch` | ALIKED Vulkan extraction (compute/DCN/SDDH/DKD/C API/shader registration) | never |
| `msvc_vulkan/0001-msvc-vulkan-hpp-compat.patch` | MSVC `__faststorefence` intrinsic compatibility | not MSVC |
| `cpu_all_variants/0001-cpu-all-variants-compiler-checks.patch` | CPU ALL_VARIANTS compiler gating | `GGML_CPU_ALL_VARIANTS=OFF` |
| `metal_merged/0001-metal-optimizations.patch` | Metal optimizations (conv_transpose/flash-attn/Swin QKV etc.) | Metal OFF |
| `cuda_mmq/0001-cuda-mmq-force-static.patch` | Force MMQ kernels + static cudart | `GGML_CUDA_FORCE_MMQ=OFF` |
| `vulkan_parallel/0001-vulkan-shaders-gen-skip-parallel-trycompile.patch` | Fix MSBuild parallel file-lock race (MSB3491) | not Windows |
| `rmbg_merged/0001-rmbg-custom-ops.patch` | RMBG custom ops (CUDA/Vulkan) | never |
| `rfdetr_merged/0001-ggml-cpu-fold-broadcast-iterations.patch` | RF-DETR CPU perf (llamafile epilogue) | `GGML_LLAMAFILE=OFF` |
| `yolo_merged/0001-yolo-ggml-backend-integration.patch` | YOLO GPU ops | never |
| `sam3_merged/0001-sam3-ggml-custom-ops.patch` | SAM3 custom ops | never |
| `trellis_merged/0001-ggml-cuda-cpy-q8_0.patch` | CUDA Q8_0→Q8_0 direct block copy | not CUDA |
| `igemm_fix/0001-igemm-plan-rebuild-guards.patch` | YOLO igemm plan rebuild guards | igemm path unused |
| `glslc_fconvert/0001-pool-shaders-avoid-redundant-fconvert.patch` | Ubuntu 24.04+ shaderc FConvert fix | `GGML_USE_VULKAN=OFF` |
| `cuda_mul_mat_f16_dst/0001-cuda-mul-mat-f16-dst.patch` | CUDA mul_mat F16 output fix | F16-output path unused |

Note: the manifest comments for `yolo_merged` and `sam3_merged` explicitly say "merged against the aliked/rmbg/vulkan_parallel chain" — new patches must follow the same on-chain merge semantics; never append upstream patches verbatim.

### Patch Compatibility Requirements (no impact on other modules' inference)

1. A new patch must be generated on the tree **after the full existing manifest has been applied** (replay `manifest.yaml` completely, then diff) — never on the pristine tarball.
2. Verify against the real replay semantics (`3rdparty/ggml/patches/apply_ggml_patches.py:198-244`, not a "three-pass replay"):
   - forward replay: the whole chain is applied to a temporary copy (`_try_sequence`, L129-147); on failure, a full-chain reverse is attempted (treated as already applied);
   - if forward passes: `git apply` each patch to the real source, then **verify the whole chain by full reverse on a temporary copy** (L215-225, ensuring the chain is fully present);
   - if both fail: `_try_recover_partial` un-applies from the tail one by one and re-applies forward (L150-195); an unrecoverable tree fails with an error.
   Manual rebuild commands (ggml.cmake's patch-signature mechanism auto-discards the old tree when patch contents change):
   ```bash
   rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
   cmake --build build_app --target ext_ggml -j4
   ```
3. Regression verification: after applying the patch, **all other AI modules' contract tests must stay green**. The contract suite is defined at `core/AICore/tests/CMakeLists.txt:589-610` (`_aicore_contract_targets`: test_runtime_capi_contract / test_depth_capi_contract / test_gaussian_capi_contract / test_lightglue_capi_contract / test_aliked_capi_contract / test_deeplsd_capi_contract / test_facedetect_capi_contract / test_rfdetr_capi_contract / test_yolo_capi_contract / test_rmbg_capi_contract / test_sam3_capi_contract / test_trellis_capi_contract):
   ```bash
   cmake -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ..
   cmake --build build_app --target aicore-contract-tests -j4   # runs ctest -L capi -LE "model|gpu|e2e"
   ```
4. Precision constraint: new operators must not change the numerical path of existing modules — only add new ops or new backend branches, never alter the default behavior of existing ops.
5. Never append upstream patches verbatim (e.g. ultralytics-ggml's `0001-yolo-ggml-backend-integration.patch`) — the two patch sets have evolved independently and must be merged semantically (yolo_merged/sam3_merged in the manifest are the merged products).

### Modification Flow

```
1. Temporarily modify and verify in build_app/ggml/... (experimentation only)
2. diff -ruN orig/ modified/ > 3rdparty/ggml/patches/<subdir>/0001-description.patch
3. Register in 3rdparty/ggml/patches/manifest.yaml (order matters)
4. rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
   cmake --build build_app --target ext_ggml -j4
5. Commit only the patch + manifest.yaml + glue code; never commit sources under build*/ggml/
```

### Precision Constraint

ggml changes **must not affect inference numerical precision**. Contract tests must include inference-result verification (not just ABI). Existing numerical gate precedent in the repo (`core/AICore/tests/aliked/test_aliked_capi_parity.cpp:17,39-40`):

```cpp
// Gates (vs CPU ref): kpt median <= 0.005 px, desc cosine median >= 0.9996
constexpr float kKptMedianTolPx = 0.005f;      // keypoint median error (pixels)
constexpr float kDescCosMedianTol = 0.9996f;   // descriptor cosine median lower bound
```

GPU (Vulkan/CUDA/Metal) vs CPU reference comparison tests run against these gates (`test_aliked_capi_parity`, registered at tests/CMakeLists.txt:370-374, LABELS "capi;model;gpu"). New tasks follow the same pattern: compare GPU output against a CPU F32 reference with deterministic numerical gates (pixel-level/cosine-level metrics) instead of non-reproducible descriptions like "threshold per upstream repo".

---

## 9. CMake Integration Checklist

### AICore Build Switches

```bash
-DAICore_ENABLED=ON              # master switch; auto-enables GGML_ENABLED (cmake/AICoreOptions.cmake:36-41)
-DAICore_USE_VULKAN=ON/OFF       # Vulkan (Linux/Windows default ON, macOS default OFF; AICoreOptions.cmake:62-64)
-DAICore_USE_METAL=ON/OFF        # Metal (Apple default ON; AICoreOptions.cmake:59-61)
-DAICore_USE_CUDA=ON             # CUDA (developer opt-in; AICoreOptions.cmake:65-67)
-DAICore_BUILD_TESTS=ON          # build contract tests (AICoreOptions.cmake:24-26)
-DAICore_BUILD_WHITEBOX_TESTS=ON # whitebox tests (requires AICore_BUILD_TESTS=ON, else FATAL_ERROR; AICoreOptions.cmake:31-34)
```

Note: all `GGML_*` variables are internal (synced from `AICore_*` by `aicore_sync_options_to_ggml()`, AICoreOptions.cmake:120-174) — **do not pass `-DGGML_*` on the command line**; stale cache entries are cleared with a warning (AICoreOptions.cmake:103-118). Read-only result variables: `AICore_VULKAN_ENABLED` / `AICore_CUDA_ENABLED` / `AICore_METAL_ENABLED` etc. (`aicore_sync_results_from_ggml()`, AICoreOptions.cmake:185-199).

### Plugin Build Switches

```bash
-DPLUGIN_STANDARD_Q<task>=ON     # standard plugin switch (uppercase + _PLUGIN suffix)
```

CMake target naming: `Q<task>_PLUGIN` all uppercase (e.g. `QDA3_PLUGIN`, `QYOLO_PLUGIN`).

### Common Linking

Plugin CMakeLists.txt template (`AddPlugin(NAME ...)` defined in `plugins/cmake/Plugins.cmake`):

```cmake
AddPlugin(NAME Q<NAME> ...)
target_link_libraries(Q<NAME>_PLUGIN PRIVATE
    CVPluginAPI
    CVPluginStub
    CVCoreLib
)
```

### Registering a New AICore Task in CMake (the actual pattern)

`core/AICore/CMakeLists.txt` does **not** use `add_subdirectory` — a new task registers in 4 steps (following the existing yolo/sam3 pattern):

1. Sources go to `core/AICore/src/tasks/<task>/`, the header to `core/AICore/include/aicore/<task>_capi.h`;
2. In `core/AICore/CMakeLists.txt` add `set(AICORE_<TASK>_SRC_DIR ...)` (pattern at L36-49) + `file(GLOB AICORE_<TASK>_SOURCES ...)` (L69-88) + an entry-file existence check (`FATAL_ERROR`, pattern at L91-117);
3. Add `AICORE_<TASK>_SOURCES` to the `add_library(${PROJECT_NAME} SHARED ...)` source list (L119-133) and the task directory to `target_include_directories(... PRIVATE ...)` (L167-193);
4. For private implementation tests: add `aicore_add_<task>_capi_test(test_<task>_capi_contract)` in `core/AICore/tests/CMakeLists.txt` (see `aicore_add_yolo_capi_test`, L408-423) and add it to `_aicore_contract_targets` (L589-610) and `_aicore_fast_targets` (L659-682).

ggml linking is always through the `3rdparty_ggml` interface target (`target_link_libraries(${PROJECT_NAME} PRIVATE 3rdparty_ggml)`, L197); tasks never link ggml directly.

---

## 10. Testing Specification

### Contract Tests (mandatory)

Every C-API function needs at least a contract test verifying:
- ABI version return value
- NULL argument safety (no crash)
- Full lifecycle: create (load/options_new) → use → free

Test file naming: `tests/<task>/test_<task>_capi_contract.cpp` (e.g. `tests/yolo/test_yolo_capi_contract.cpp`, `tests/aliked/test_aliked_capi_contract.cpp`). Real template (`tests/yolo/test_yolo_capi_contract.cpp:24-80`):

```cpp
#include "aicore/yolo_capi.h"
#include "tests/common/test_macros.hpp"   // AICORE_CHECK assertion macro

int main() {
    AICORE_CHECK(aicore_yolo_abi_version() >= 1);
    // NULL-safe teardown / lifecycle
    aicore_yolo_free(nullptr);
    aicore_yolo_options_free(nullptr);
    aicore_yolo_free_buffer(nullptr);
    // every setter's NULL no-op + getter round-trip (options lifecycle)
    // loading a nonexistent model must fail cleanly and report last_error
    // default-value assertions: aicore_yolo_options_get_conf_thres(nullptr) == 0.25f etc.
}
```

Registration: `aicore_add_<task>_capi_test(test_<task>_capi_contract)` in `core/AICore/tests/CMakeLists.txt` (L408-423 is yolo's full function body, including the WIN32 dirent link branch), with `LABELS "capi"`. Run:

```bash
cmake -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ..
cmake --build build_app --target test_<task>_capi_contract -j4
./build_app/bin/aicore_tests/test_<task>_capi_contract   # no GGUF assets; pure ABI/NULL/lifecycle
```

### Coverage Requirement

`python3 core/AICore/tests/check_capi_coverage.py` enforces coverage >= 95% (script L24 `COVERAGE_TARGET = 95`, exit code 0=PASS / 1=FAIL).

The script scans all `aicore_*` functions in `include/aicore/*_capi.h` (regex `\b(aicore_[a-z0-9_]+)\s*\(`, L40) and checks whether they are called by consumers (`core/AICore/tests`, `core/AICore/tools`, `plugins/core/Standard`, `libs`, L27-32). A new API must be referenced by at least one contract test.

### Inference Precision Verification

Contract tests without GGUF assets cover ABI/NULL/lifecycle; **numerical verification lives in asset/GPU parity tests**. Real precedent in the repo (`core/AICore/tests/aliked/test_aliked_capi_parity.cpp:17,39-40`, CTest LABELS "capi;model;gpu"):

```cpp
// Fixed input (sacre_coeur1.jpg) → pointwise comparison of GPU output vs CPU reference
// Gates (vs CPU ref): kpt median <= 0.005 px, desc cosine median >= 0.9996
constexpr float kKptMedianTolPx = 0.005f;
constexpr float kDescCosMedianTol = 0.9996f;
```

Test assets are injected as environment variables by `aicore_configure_model_test_assets()` (tests/CMakeLists.txt:457-516: `AICORE_TEST_<TASK>_GGUF` / `AICORE_TEST_<TASK>_IMAGE`); tests without assets skip with exit code 77 (`SKIP_RETURN_CODE 77`, L66).

---

## 11. Plugin Integration Code Templates

### C-API Header Template

```c
// include/aicore/<task>_capi.h
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aicore_<task>_ctx aicore_<task>_ctx;
typedef struct aicore_<task>_options aicore_<task>_options;

AICORE_CAPI int aicore_<task>_abi_version(void);

// Options builder
AICORE_CAPI aicore_<task>_options* aicore_<task>_options_new(void);
AICORE_CAPI void aicore_<task>_options_free(aicore_<task>_options* opts);
AICORE_CAPI void aicore_<task>_options_set_device(opts, const char* device);
AICORE_CAPI void aicore_<task>_options_set_threads(opts, int n_threads);

// Lifecycle
AICORE_CAPI aicore_<task>_ctx* aicore_<task>_load_opts(const char* gguf, const opts*);
AICORE_CAPI void aicore_<task>_free(aicore_<task>_ctx* ctx);
AICORE_CAPI int aicore_<task>_is_ready(const aicore_<task>_ctx* ctx);
AICORE_CAPI const char* aicore_<task>_last_error(const aicore_<task>_ctx* ctx);
AICORE_CAPI void aicore_<task>_free_buffer(void* p);

// Inference entry points (use struct parameters for complexity > 6 params)
// ...

#ifdef __cplusplus
}
#endif
```

### Worker Class Template

```cpp
// plugins/core/Standard/q<Worker>/src/q<Worker>.cpp
#include "aicore/<task>_capi.h"

class Q<Worker> : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    ...
private:
    aicore_<task>_ctx* m_ctx = nullptr;
    
    bool loadModel(const QString& gguf_path) {
        auto* opts = aicore_<task>_options_new();
        aicore_<task>_options_set_device(opts, "auto");
        m_ctx = aicore_<task>_load_opts(gguf_path.toUtf8().data(), opts);
        aicore_<task>_options_free(opts);
        return m_ctx != nullptr;
    }
    
    void runInference() {
        // ... call inference API ...
        // resource release
        aicore_<task>_free_buffer(some_output);
        aicore_<task>_free(m_ctx);
    }
};
```

---

## 12. Test Data Integration Specification (use test data button)

New plugins must provide a "Try sample data" one-click entry, **reusing the shared component `ecvTestDataRepository` (libs/CVPluginAPI); hand-rolled downloaders are forbidden**.

### Dataset Selection

The `ecvTestDataRepository::Dataset` enum (`libs/CVPluginAPI/include/ecvTestDataRepository.h:41-46`) has 4 datasets:

| Plugin type | Dataset | Content |
|---|---|---|
| AI inference (qYOLO/qRFDetr/qRMBG/qDeepLSD) | `ObjectsDetection` | shared images/videos |
| Reconstruction (qDA3) | `Monstree` | multi-view images |
| Face (qFaceDetect) | `FriendsFaces` | face video |
| Image-to-3D (qTrellis etc.) | `Image2Mesh` | single-image-to-3D samples (`examples_images/`, ecvTestDataRepository.h:154-158) |

### Button Specification (consistent look across plugins)

```cpp
m_useTestDataBtn =
        new QPushButton(QStringLiteral("\U0001f9ea  Try sample data"));
m_useTestDataBtn->setToolTip(
        "Load sample images for inference.\n"
        "Downloads on first use, then cached locally.");
m_useTestDataBtn->setStyleSheet(
        "QPushButton { background: #00897b; color: white; font-weight: bold;"
        " border: none; border-radius: 4px; padding: 5px 12px; }"
        "QPushButton:hover { background: #00796b; }"
        "QPushButton:pressed { background: #00695c; }"
        "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
```

The teal theme (`#00897b`) is the unified look of all AICore plugins (qDA3/qDeepLSD/qFaceDetect/qFreeSplatter/qLightGlue/qRFDetr/qRMBG/qYOLO/qSAM3/qTrellis); custom colors are forbidden. The stylesheet lives in `makeSampleDataBtn()` at `libs/CVPluginAPI/include/ecvAICoreUiHelper.h:132-143`, verbatim identical to the button spec above.

### Click Flow (onUseTestData three states)

1. **Already extracted**: extract dir exists and `findDatasetFile` / `getMonstreeImages` hit → fill components and return
2. **Zip cached**: `verifyZipIntegrity(zipPath, expectedMd5, expectedSize)` passes → progress bar + `extractDataset` → fill after extraction
3. **No cache**: `startDownload(kind)` + signal chain (`downloadFinished` → `extractDataset` → fill)

```cpp
void onUseTestData() {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    const TestDataset kind = TestDataset::ObjectsDetection;

    // 1. already extracted: fill directly
    const QString path = ecvTestDataRepository::findDatasetFile(kind, kTestImage);
    if (!path.isEmpty()) { fillComponents(path); return; }

    // 2. zip cached: extract
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    if (ecvTestDataRepository::verifyZipIntegrity(
            ecvTestDataRepository::zipPath(kind), info.expectedMd5,
            info.expectedSize)) {
        setTestDataControlsEnabled(false);
        m_progress->setVisible(true);
        repo.extractDataset(kind);
        return;
    }

    // 3. no cache: download (fill in the signal-chain callback)
    m_progress->setRange(0, 100);
    m_progress->setVisible(true);
    repo.startDownload(kind);
}
```

### Filling Components

```cpp
// image list → path input (semicolon separated)
m_inputPath->setText(images.join(";"));
// video → video input source
m_liveWidget->setInputSource(YOLOLiveWidget::InputSource::VideoFile);
m_liveWidget->setVideoFilePath(path, false);
```

### Cache and Integrity

- Directories: `~/cloudViewer_data/download/` (zip) + `~/cloudViewer_data/extract/` (extracted)
- Download integrity: MD5 + size check (`verifyZipIntegrity`); cache hits skip re-download
- While downloading/extracting: disable the button + keep progress bar/status label visible; on failure restore the button and log the reason

---

## 13. Plugin UI Design Specification (mandatory)

All AICore plugin dialogs **must** reuse the shared UI helper `ecvAICoreUiHelper.h` (`libs/CVPluginAPI/include/`, namespace `ecvAICoreUi`); local duplicate helpers / stylesheets / pixel constants are forbidden. This is the unified standard of the 10 existing plugins (qDA3/qDeepLSD/qFaceDetect/qLightGlue/qFreeSplatter/qRFDetr/qRMBG/qYOLO/qSAM3/qTrellis) — the header comment (ecvAICoreUiHelper.h:8-9) names 8, with qSAM3 (`SAM3Dialog.cpp:16`) and qTrellis (`TrellisDialog.cpp:24`) already migrated. New plugins follow it directly to reach production-grade appearance.

Verify: `rg -l "ecvAICoreUiHelper.h" plugins/core/Standard/q*/src` should cover every AICore plugin dialog.

### 13.1 Shared Helper Quick Reference

| Capability | API | Defined at | Description |
|---|---|---|---|
| DPI scaling | `ecvAICoreUi::dpiScaled(px)` | ecvAICoreUiHelper.h:47-51 | 96-dpi nominal pixels → actual pixels at current screen DPI; **every hardcoded pixel must go through it** |
| Spacing/margins | `tabMargins()`, `vSpacing()`, `hSpacing()`, `tightVSpacing()` | ecvAICoreUiHelper.h:63-79 | unified compact spacing (margin 4px / spacing 4/6/2px); no magic numbers |
| Size constants | `previewSize()` (96), `slotPreviewSize()` (88), `dbListMaxHeight()` (140), `filePoolMaxHeight()` (120) | ecvAICoreUiHelper.h:88-97 | thumbnail / list heights, DPI-aware |
| Label factories | `makeLabel(text)`, `makeHintLabel(text)` | ecvAICoreUiHelper.h:104-124 | form labels (left-aligned) + grey hint text |
| Button factories | `makeSampleDataBtn()`, `makeBrowseBtn(text)` | ecvAICoreUiHelper.h:132-152 | teal sample button (`#00897b`) + fixed-width browse button (`browseBtnWidth()`=dpiScaled(72)) |
| SpinBox | `setCompactDoubleSpin()`, `setCompactSpin()` | ecvAICoreUiHelper.h:158-168 | compact fixed-width spinboxes (`compactSpinWidth()`=dpiScaled(72)) |
| Layout tools | `setupTabLayout()`, `setupFormGrid()`, `tightenGroupBox()`, `styleTabWidget()` | ecvAICoreUiHelper.h:175-220 | page / form grid / group box / tab unified styling |
| Section builders | `makeRuntimeRow(device, threads)`, `makeDbSection()`, `connectDbToggle()`, `setupProgressSection()`, `makeActionRow()` | ecvAICoreUiHelper.h:251-340 | Device/Threads row, DB collapse section, progress section, action row |

Basic template (`setupUi()` start):

```cpp
#include "ecvAICoreUiHelper.h"

void MyDialog::setupUi() {
    auto* root = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(root);
    root->setSizeConstraint(QLayout::SetNoConstraint);  // see 13.4
    auto* tabs = new QTabWidget(this);
    ecvAICoreUi::styleTabWidget(tabs);
    // form grid: label column width 92 (two-column label|field structure)
    auto* grid = new QGridLayout;
    ecvAICoreUi::setupFormGrid(grid, 92);
    // runtime params: Device/Threads in one row
    root->addWidget(ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads));
    // DB collapse section:
    auto* dbToggle = ecvAICoreUi::makeDbSection(nullptr);
    ecvAICoreUi::connectDbToggle(dbToggle, m_dbContentWidget);
    // progress section (label + progress, hidden by default):
    ecvAICoreUi::setupProgressSection(root, m_downloadLabel, m_progress);
    // action row:
    auto* row = ecvAICoreUi::makeActionRow(m_runBtn, m_cancelBtn);
}
```

### 13.2 Layout Guidelines

1. **DPI-aware**: `setMinimumSize` / thumbnail sizes / list heights / button widths all use `ecvAICoreUi::dpiScaled()`; bare pixels forbidden.
2. **Compact**: page layouts use `setupTabLayout()` (margin 4px / spacing 4px); group boxes use `tightenGroupBox()` (QSizePolicy::Maximum + compact margins) so boxes hug their content.
3. **Forms**: every QGridLayout uses `setupFormGrid()` (fixed label column width, field column stretch=1), labels via `makeLabel()` for left-aligned vertical centering.
4. **Unified look**: sample-data buttons must use `makeSampleDataBtn()` (teal `#00897b`); browse buttons must use `makeBrowseBtn()`; spinboxes must use `setCompactSpin*()`. Custom palettes forbidden.
5. **DB input area**: use `makeDbSection()` + `connectDbToggle()` for the collapse section; list height clamped to `[dpiScaled(60), dbListMaxHeight()]` with internal scrolling; **expanding must not grow the dialog** (see 13.4).
6. **Progress section**: use `setupProgressSection()`; note its progress bar is **hidden by default** — you must explicitly call `m_progress->setVisible(true)` where download/inference starts (the existing qFaceDetect/qLightGlue pattern), otherwise the bar never shows.

### 13.3 Input Preview and Click-to-Enlarge (mandatory)

- Preview widgets use `ecvClickableImageLabel`; **call `setPreviewImage(img, size)` (or `setPreviewPixmap`) instead of bare `setPixmap`** — click-to-enlarge depends on the internal `m_fullImage`; bare `setPixmap` makes clicks dead.
- **DB entity input (`db://EntityName`) must support enlarge too**: in `setDbImages()` store the full-resolution image in an item role (`Qt::UserRole + 1`); in `updateImagePreview()` when the path has a `db://` prefix, fetch from the role and call `setPreviewImage`. See the existing qDeepLSD / qLightGlue implementations; qDA3 (QComboBox) stores images via `setItemData(idx, img, role)`.
- Directory / multi-file inputs: use the first image for the preview.

```cpp
// in setDbImages:
item->setData(Qt::UserRole, e.name);
item->setData(kDbFullImageRole, e.preview);  // full-resolution image

// in updateImagePreview:
if (path.startsWith(QLatin1String("db://"))) {
    // fetch from the list item's kDbFullImageRole → setPreviewImage
} else {
    img = QImage(path);
}
```

### 13.4 Dialog Sizing and Self-Adaptation (no growth / no looseness)

1. **Main layout uses `QLayout::SetNoConstraint`**: QDialog's default minimum-size constraint makes the window follow content minimumSizeHint and grow automatically — expanding the DB section, switching tabs, or status-text changes all inflate the dialog. With NoConstraint the window size is decided once by the first sizeHint and content changes no longer grow it.
2. **Fix the size on first show**: in `showEvent` (guarded by an `m_firstShow` flag) call `adjustSize()` synchronously — Qt has finished layout before the Show event, so the sizeHint is clean at that point.
3. **Tab height management** (dialogs with multiple tabs and video/long forms, see qFaceDetect / qFreeSplatter):
   - On first `showEvent`, measure `m_baseChrome = height() - tabWidget->height()` **synchronously** (minimumSizeHint deltas or delayed singleShot both miscompute / lose to X11 window mapping);
   - On tab switch: `resize(width, qBound(min, baseChrome + tabContentSizeHint, available-20))`, targetHeight = tabBar + content sizeHint;
   - Re-measure chrome on `ScreenChangeInternal` (cross-screen DPI);
   - Never mix minimum-based and sizeHint-based numbers in the formulas (historical lesson: each tab switch inflated the dialog by 230~600px).
4. **Video previews must have a height cap** (see 13.5 pitfall 1), otherwise "self-adaptation" becomes "infinite growth".

### 13.5 Common Pitfalls (real bug sediment)

1. **Uncapped video preview → UI grows unboundedly (qFreeSplatter regression)**:
   - Feedback loop: per-frame `setPixmap` → QLabel `sizeHint` = pixmap size → `QScrollArea(widgetResizable)` grows the widget per sizeHint → preview gets taller → next frame's pixmap scales to the bigger label → loop. Internally convergent, but **any external window perturbation (WM tweaks/DPI/remote desktop) that enlarges it is retained 1:1 forever** — "the whole UI keeps growing while playing video".
   - Fix: `updatePreviewHeightCap()` (`video_base/src/VideoPlaybackWidget.cpp:1291-1316`) adaptive branch must `m_previewLabel->setMaximumHeight(ecvAICoreUi::dpiScaled(560))` (L1314); `m_faceCaptureScroll->setMaximumHeight(dpiScaled(560))`; `adaptTabWidgetHeight()` contentHeight additionally `min(..., dpiScaled(560))`. The cap must be > 16:9 minimum to leave stretch room, otherwise the "big blank" returns (see pitfall 3).
   - Verify: `rg -n "dpiScaled\(560\)" plugins/core/Standard/video_base plugins/core/Standard/qFaceDetect` should hit the 3 places above.
2. **QBoxLayout compression + setGeometry clamp → overlapping widgets**: when an outer container height is locked by `setFixedHeight`, content growth (e.g. videoControlsRow going from hidden to visible) makes the layout compress widgets proportionally; a widget with an explicit minimumHeight is clamped back by `setGeometry`, shifting later widgets' y and overlapping them. Debug: check whether "layout-computed geometry" and "post-setGeometry actual geometry" agree; fix: replace `setFixedHeight` with `setMinimumHeight` + Expanding and re-measure the host when content visibility changes.
3. **maximumHeight on the preview → big blank**: QVBoxLayout hands the space clipped from the capped widget to other Preferred widgets (input row/statusLabel get stretched), producing a "big blank". The single stretch preview must own the remaining space exclusively: don't cap it (or cap very high), keep other widgets at natural height.
4. **DB components inflate the first tab**: list height unbounded or the dialog follows minimumSizeHint. Fix: clamp the list height + `SetNoConstraint` on the main layout (see 13.4).
5. **Preview click does not enlarge**: see 13.3 — bare `setPixmap` or DB input without the stored full image.
6. **Signal emitted during the QWidget destruction cascade → SIGSEGV in `QFunctorSlotObject` (qSAM3 exit crash, v3.9.5)**:
   - Crash chain (stack + code double-confirmed): `~MainWindow` → `~SAM3Dialog` → QWidget `deleteChildren` → `~QTabWidget` → `~VideoTab` → `releaseModel()` → `emit backendChanged("none")` (VideoTab.cpp:181) → `setupUi()` lambda connected with `this` context (SAM3Dialog.cpp:738) → dereferences `m_backendLabel`/`m_tabs` (SAM3Dialog.cpp:483/494, created **before** `m_videoTab` at :728) → jump to near-NULL offset 0x21.
   - Root cause — three conditions, all required: (a) a child widget **emits a signal directly or indirectly from its destructor chain**; (b) the receiver was connected with an ancestor dialog as context — during `deleteChildren` the ancestor's QObject is still alive so Qt does **not** auto-disconnect; (c) the slot/lambda touches sibling child widgets or members whose owning objects were already destroyed earlier in the cascade. The receivers' context dies only at `~QObject`, which runs **after** the whole `deleteChildren` walk.
   - Why normal testing misses it: the emit fires only when a stream/model is **active while the window closes**; headless lifecycle tests never enter the destructor cascade.
   - Fix (landed, SAM3Dialog.cpp:458-466): `disconnect(m_videoTab, &VideoTab::backendChanged, this, nullptr)` at the top of `~SAM3Dialog` — before the `deleteChildren` cascade. Single-point cut of the only destruction-time emit path; zero runtime behavior change (label refresh on live backend switches stays connected).

   #### Full-plugin audit result (v3.9.5, every `~Class()` in `plugins/core/Standard` inspected)

   | Emitter in destruction chain | Receivers with ancestor context | Status |
   |---|---|---|
   | `VideoTab::~VideoTab` → `releaseModel` → `emit backendChanged` (VideoTab.cpp:142-149, 181) | SAM3Dialog.cpp:738 | **crashed, fixed** |
   | `~VideoPlaybackWidget` → `stopStream()` → `emit streamStopped` (VideoPlaybackWidget.cpp:770-774) | YOLODialog.cpp:346, RFDetrDialog.cpp:340, RMBGDialog.cpp:293, FaceDetectDialog.cpp:201, FreeSplatterDialog.cpp:535 (via FaceCaptureWidget.cpp:127 forwarding) | **order-luck safe → root-fixed** (destructor disconnect guards, see below) |
   | `RFDetrLiveInferWorker::~{releaseModel();}` (:48), `YOLOLiveInferWorker` (:48), `RMBGLiveInferWorker` (:45) | — | safe: their `releaseModel()` frees the ctx only, emits nothing |
   | All 9 `QThread` workers (`~SAM3Worker`, `~TrellisWorker`, `~DA3Worker`, …) | — | safe: cancel/wait/free only, no emit |
   | `onStreamStopping`/`onStreamReset` overrides (YOLOLiveWidget.cpp:733, RFDetrLiveWidget.cpp:601, RMBGLiveWidget.cpp:488, FaceLiveDetectWidget.cpp:987, FaceCaptureWidget.cpp:983) | — | safe: reset state only, no emit |

   **"Order-luck safe" and the root fix (landed v3.9.5)**: those 5 dialogs connect `streamStopped`→lambda that touches `m_liveStartBtn` etc.; they survived only because the emitter child was created **before** the buttons (YOLODialog.cpp:301 vs 315, RFDetrDialog.cpp:291 vs 305, RMBGDialog.cpp:240 vs 252, FaceDetectDialog.cpp:159 vs 176, FreeSplatterDialog.cpp:436 vs 475) — `deleteChildren` destroys children in creation order, so when the live widget emitted, the buttons were still alive. Reordering `setupUi()` would reintroduce the exact qSAM3 crash with zero compiler/test signal; trigger condition: closing the window while the stream is active (`if (m_streamActive)` guard, VideoPlaybackWidget.cpp:770).

   #### Mandatory rules for new AICore plugin dialogs

   1. **Never emit signals from a destructor chain** (destructor body or anything it calls). If teardown must run logic that would normally emit (release model, stop stream), disconnect that signal first, or suppress the emit with a `m_destroying` flag.
   2. **Emitter-side root fix (landed v3.9.5)**: `disconnect(this, nullptr, nullptr, nullptr);` as the first statement of every destructor that can reach a teardown emit. **C++ ordering trap**: a subclass destructor body runs BEFORE `~VideoPlaybackWidget`, so the base-class guard alone cannot stop a subclass-destructor `stopStream()` — the landed fix puts the guard in `~VideoPlaybackWidget` (baseline for direct use / future subclasses, VideoPlaybackWidget.cpp:99) AND in all 5 subclasses that call `stopStream()` from their own destructors: YOLOLiveWidget.cpp:70, RFDetrLiveWidget.cpp:69, RMBGLiveWidget.cpp:67, FaceLiveDetectWidget.cpp:109, FaceCaptureWidget.cpp:249 (its `stopCamera()` is `stopStream()`, FaceCaptureWidget.h:73). Dropping outgoing connections at destruction time is safe: receiver-side connections die with `~QObject` anyway, and internal `this→this` wiring (timers, the FaceCaptureWidget forwarding) is exactly what must stop. The receiver-side `disconnect` (SAM3 style) remains the fallback when the emitter is not under your control.
   3. **Ordering contract**: if a destructor-chain emit cannot be removed, the emitting child must be created before every widget its receivers touch — and add a comment at both creation sites; this is a fragile last resort, not a design.
   4. **Related teardown-trap family (already handled, do not regress)**: `QThread::finished → deleteLater` is delivered as a DIRECT call during thread teardown while the event loop is half-dead; drop the connection before `quit()` and own the worker lifetime manually (VideoPlaybackWidget.cpp:109-111, FaceLiveDetectWidget.cpp:119-127).

   Verification:

   ```bash
   # no emit inside any destructor body (direct pattern) — expect 0 hits:
   rg -U --glob '*.cpp' '~\w+\s*\([^;{}]*\)\s*\{[^{}]*\bemit\s' plugins/core/Standard
   # locate destruction-chain emitters manually for the indirect pattern (e.g. stopStream/releaseModel)
   # qSAM3 fix present:
   rg -n "disconnect\(m_videoTab, &VideoTab::backendChanged" plugins/core/Standard/qSAM3/src/SAM3Dialog.cpp
   # streamStopped root fix present (6 guards: video_base baseline + 5 subclasses):
   rg -n "disconnect\(this, nullptr, nullptr, nullptr\);" \
     plugins/core/Standard/video_base/src/VideoPlaybackWidget.cpp \
     plugins/core/Standard/qYOLO/src/YOLOLiveWidget.cpp \
     plugins/core/Standard/qRFDetr/src/RFDetrLiveWidget.cpp \
     plugins/core/Standard/qRMBG/src/RMBGLiveWidget.cpp \
     plugins/core/Standard/qFaceDetect/src/FaceLiveDetectWidget.cpp \
     plugins/core/Standard/qFreeSplatter/src/FaceCaptureWidget.cpp
   ```
7. **Never `memcpy` into a `QImage` as one contiguous block — use the explicit-stride constructor (zero-copy) or per-row copies (qRF-DETR seg masks rendered as diagonal-stripe garbage, v3.9.5)**:
   - Symptom: detection boxes correct; segmentation tint appears as diagonal stripes/blobs unrelated to the objects. Same artifact in the still-image (DB export) and live render paths.
   - Root cause: `QImage` scanlines are **32-bit aligned** — for `Format_Grayscale8` with `width = 78` (the RF-DETR mask-head resolution, `image_size / mask_downsample_ratio`), `bytesPerLine = 80 > 78`. One contiguous `memcpy(img.bits(), src, w * h)` fills row `y` starting at offset `y * 78` while the image reads row `y` at `y * 80` — a cumulative 2 px/row horizontal shear = diagonal stripes. Nothing upstream was wrong (verified: the C-API masks were byte-identical to the correct per-query planes; model, postprocess and plugin data all agreed).
   - **Decision table (use the fastest pattern that fits the access pattern)** — measured on 78x78 Grayscale8, 200k iters, Qt 5.15, -O2, including the `QImage` allocation:

     | Pattern | µs/op | Correct? | Use when |
     |---|---|---|---|
     | `QImage(const uchar*, w, h, w, fmt)` **zero-copy wrap** | **0.035** | yes | source is read-only while the QImage lives (mask tint/blit sources, PNG-encode-once metadata). No copy at all — faster than even the buggy contiguous memcpy |
     | per-row `memcpy(img.scanLine(y), src + y*w, w)` | 0.389 | yes | source must be **copied and mutated** afterwards (Gaussian blur, `{0,1}`→`{0,255}` rescale) — the write target must be an owned, Qt-aligned buffer |
     | contiguous `memcpy(img.bits(), src, w*h)` | 0.094 | **NO (shears)** | forbidden unless `w % 4 == 0` is guaranteed AND asserted |

     ```cpp
     // READ-ONLY consumption (landed in RFDetrModelCatalog.cpp drawDetections):
     const QImage mask(reinterpret_cast<const uchar*>(d.maskRaw.constData()),
                       d.maskWidth, d.maskHeight, d.maskWidth,
                       QImage::Format_Grayscale8);   // stride passed = no shear, no copy
     // WRITE path (landed in RFDetrLiveWidget.cpp: blur mutates the buffer,
     // so an owned Qt-aligned copy is required — fill it row by row):
     QImage mask(w, h, QImage::Format_Grayscale8);
     for (int y = 0; y < h; ++y)
         std::memcpy(mask.scanLine(y), src + (size_t)y * w, (size_t)w);
     ```

   - Wrap lifetime rule: the ctor does **not** copy; the source bytes must outlive the QImage and must not be resized/mutated through the original container while the view is alive. For a needed owned copy prefer `view.copy()` (deep, stride preserved) over a new contiguous memcpy.
   - **Qt version compatibility (CI-verified)**: both the explicit-stride ctor and `constScanLine` exist since Qt 5.12 — verified against the actual `qtbase5-dev 5.12.8` headers on Ubuntu 20.04 (`/usr/include/x86_64-linux-gnu/qt5/QtGui/qimage.h` declares `QImage(const uchar*, int, int, int bytesPerLine, Format)`), and the parameter type widening to `qsizetype` in Qt 6 accepts the same int call sites. CI matrix coverage: apt Qt 5.12.8 (focal docker), conda `qt=5.15.*` (all platforms), conda `qt6-main>=6.4`. No `QT_VERSION` branching needed for this fix.
   - Same trap in other formats: **any** format whose bytes-per-pixel makes `w * bpp % 4 != 0` pads scanlines — `Format_RGB888` (3 B/px) shears for `w % 4 != 0`, `Format_Grayscale8` (1 B/px) shears for `w % 4 != 0`. `Format_ARGB32` (4 B/px) is immune. The rule is about `bytesPerLine`, never about pixel semantics.
   - Landed: RFDetrModelCatalog.cpp `drawDetections` (zero-copy wrap, still-image path), RFDetrLiveWidget.cpp (per-row, write path with blur), qYOLO.cpp / YOLOLiveWidget.cpp / YOLOModelCatalog.cpp (per-row, write paths with `{0,1}`→`{0,255}` rescale / blur; latent only — YOLO's 160-wide masks satisfy `w % 4 == 0`).
   - Audit baseline (safe patterns already in tree): qSAM3 (SAM3Worker/VideoWorker) and qDA3 copy via `scanLine(y)` per row; qRFDetr.cpp DB-mask metadata uses the explicit-stride ctor; LightGlue/DeepLSD use `convertToFormat`; VtkMultiTextureRenderer, rfdetr `image_io.cpp`, `depth_export.cpp` are row-wise or explicit-stride.
   - Verify:

     ```bash
     # no contiguous memcpy into QImage::bits() left in plugin sources — expect 0 hits:
     rg -U --glob '*.cpp' 'memcpy\(\w+\.bits\(\),' plugins/core/Standard
     ```

     (Debug tip when a rendered buffer looks wrong: dump the **raw source bytes** with a minimal viewer first and compare against the QImage render — that separates "data is wrong" from "stride is wrong" in minutes. In this incident the raw 78x78 planes were perfect cat silhouettes while every QImage-based render showed stripes.)

---

## 14. Cross-Platform Compilation Compatibility (mandatory)

Compiling on Linux does **not** mean all three platforms pass. Compiler behavior differences (all from real CI incidents):

| Platform | Compiler | Key behavior | Incident |
|---|---|---|---|
| macOS | AppleClang | GNU extensions are **errors by default** (no -Werror needed), e.g. void* pointer arithmetic | qSAM3's `mask.data + offset` failed to compile |
| Windows | MSVC | `__attribute__` unknown (C3646 cascade); POSIX functions missing (C3861); /W4 /WX- warnings don't break | one attribute in rfdetr → 8 files failed; 4 strncasecmp uses in sam3 |
| Linux | GCC/Clang | most permissive; GNU extensions only warn | —— |

### 14.1 void* Pointer Arithmetic (breaks on macOS)

`+`/`-`/`[]` on `const void*` (e.g. `aicore_<task>_plane_view::data`) is a GNU extension: Linux only warns, **AppleClang errors by default** with `arithmetic on a pointer to void`.

```cpp
// wrong: mask.data is const void*
memcpy(m.scanLine(y), mask.data + y * mask.row_stride_bytes, w);

// correct: cast first, size_t offsets, explicit #include <cstdint>
memcpy(m.scanLine(y),
       static_cast<const uint8_t*>(mask.data) +
               static_cast<size_t>(y) * mask.row_stride_bytes,
       static_cast<size_t>(mask.width));
```

Precedents: `plugins/core/Standard/qYOLO/src/YOLOWorker.cpp` (casts then feeds QByteArray), the tracker internals of `core/AICore/src/tasks/sam3/sam3_capi.cpp` (casts to uint8_t* then memcpy row by row).

### 14.2 Bare `__attribute__` (breaks on Windows)

All `__attribute__((...))` must have a compiler guard, otherwise MSVC reports `C3646: '__attribute__': unknown override specifier` plus a C2059/C2143 cascade (one header included by N source files = N error dumps).

```cpp
void rfdetr_logf(rfdetr_log_level lvl, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((format(printf, 2, 3)))
#endif
        ;
```

- Precedent: the `AICORE_LEGACY_API` three-branch macro in `core/AICore/include/aicore/runtime_capi.h` (`_MSC_VER` → `__declspec` / `__GNUC__||__clang__` → `__attribute__` / else empty)
- Function multiversioning `__attribute__((target("avx512...")))` additionally needs ISA macro gating (see the `FD_WINO_AVX512_TARGET` pattern in `facedetect/winograd.cpp` / `directconv.cpp`: `FACEDETECT_*_AVX512 && __AVX2__ && (__GNUC__ || __clang__)`)
- `__declspec` likewise: only inside `#ifdef _WIN32` branches

### 14.3 POSIX Functions and Headers (breaks on Windows)

The project CMake defines `_CRT_NONSTDC_NO_DEPRECATE` globally for WIN32 targets (`cmake/CloudViewerSetGlobalProperties.cmake`) → **MSVC headers do not declare any underscore-less POSIX names**; direct use fails with C3861. Common replacement table:

| POSIX | MSVC equivalent | Repo precedent (verifiable) |
|---|---|---|
| `strncasecmp` / `strcasecmp` | `_strnicmp` / `_stricmp` | `core/AICore/src/tasks/sam3/sam3.cpp:64-69` (`#ifdef _WIN32` macro mapping, defined after all includes) |
| `strdup` | `_strdup` | the `dupString()` wrappers in 4 `model_catalog.cpp` files (yolo/rmbg/rfdetr/trellis, all `#ifdef _MSC_VER` three-branch) |
| `usleep(us)` | `::Sleep(ms)` (note µs→ms) | app/ecvContourExtractorDlg.cpp:144-146, app/ecvDeepSemanticSegmentationTool.cpp:383-385, app/ecvPoissonReconDlg.cpp:251-253 (`#if defined(CV_WINDOWS)` / `#else` branches) |
| `localtime_r` / `gmtime_r` | `localtime_s` / `gmtime_s` (different argument order) | `core/src/Helper.cpp:288-293` |
| `getpid` | `_getpid` | `app/ecvConsole.cpp:46-48` (`#define getpid _getpid`) |
| `mkdir(path, mode)` | `_mkdir(path)` | `core/AICore/src/tasks/sam3/sam3.cpp:33-39` (`#define mkdir(path, mode) _mkdir(path)`) |
| `fdopen(fd, ...)` | `std::fopen` directly | the `#else` branch of `core/AICore/src/tasks/depth/ply_export.cpp:23-31` |

POSIX headers with no Windows equivalent must be platform-branched: `unistd.h`, `dirent.h`, `strings.h`, `pthread.h`, `sys/time.h`, `sys/sysinfo.h`, `sys/sysctl.h`, `dlfcn.h`, `mach/mach.h` — `#ifdef _WIN32` / `__APPLE__` / `__linux__` three-way, following the dlfcn/windows.h branch in `core/AICore/src/common/ggml_backend_utils.hpp`.

### 14.4 Don't Re-`#define` Command-Line Macros (Windows C4005)

`_USE_MATH_DEFINES`, `__STDC_LIMIT_MACROS`, `NOMINMAX`, `_CRT_SECURE_NO_WARNINGS` are provided by global compile definitions (`cmake/CMakeSetCompilerOptions.cmake` / `cmake/CloudViewerSetGlobalProperties.cmake`); **do not** write `#define _USE_MATH_DEFINES` in sources (triggers C4005 macro redefinition). When a local define is genuinely needed:

```cpp
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
```

Math constants like `M_PI` are guaranteed by the global `_USE_MATH_DEFINES`; use them directly, never redefine.

### 14.5 Qt Version Compatibility (Qt5/Qt6, mandatory)

The project supports both Qt 5.12+ and Qt 6.2+. **Any API that behaves differently between Qt5 and Qt6 must go through the compatibility layer `core/include/QtCompat.h`** (CVCoreLib public header, `#include <QtCompat.h>`); calling divergent APIs directly or writing per-plugin `#if QT_VERSION` branches is forbidden.

Common replacements (QtCompat.h covers 14 categories):

| Qt5-only (removed/deprecated in Qt6) | Unified compatible form |
|---|---|
| `QRegExp` / `QString::split(QRegExp)` | `QtCompatRegExp` / `qtCompatSplit` / `qtCompatSplitRegex` / `qtCompatReplace` / `QtCompatRegExpWrapper` |
| `QString::SkipEmptyParts` | `QtCompat::SkipEmptyParts` (`QtCompat::KeepEmptyParts`) |
| `QStringRef` / `midRef` / `splitRef` | `QtCompatStringRef` / `qtCompatStringRef*` / `qtCompatSplitRef*` |
| `QTextCodec::codecForLocale()` | `qtCompatCodecForLocale()` (type `QtCompatQTextCodec`) |
| `QTextStream::endl` | `QtCompat::endl` / `QTCOMPAT_ENDL` |
| `QFontMetrics::width(text)` | `QTCOMPAT_FONTMETRICS_WIDTH(fm, text)` |
| `QWheelEvent::delta()` / `pos()` | `qtCompatWheelEventDelta` / `qtCompatWheelEventPos` |
| `QMouseEvent::pos()` / `globalPos()` | `qtCompatMouseEventPos*` / `qtCompatMouseEventGlobalPos*` |
| `QDropEvent::pos()` | `qtCompatDropEventPos*` |
| `QMap::insertMulti()` / `unite()` | `qtCompatMapInsertMulti` / `qtCompatMapUnite` |
| `QVariant::type()` / `var.type() == QVariant::String` | `qtCompatVariantType` / `qtCompatVariantIsString` etc. |
| `QPlainTextEdit::setTabStopWidth()` | `qtCompatSetTabStopWidth` |
| `QSet<T>(begin,end)` / `QVector<T>(begin,end)` | `qtCompatQSetFromVector` / `qtCompatQVectorFromSet` |
| `QAtomicInteger::load()/store()` | `qtCompatLoadRelaxed` / `qtCompatStoreRelaxed` |

Rules:

1. **Extend QtCompat.h rather than local `#if`**: when a new Qt5/Qt6 divergent API appears, add a `qtCompat*` helper to `QtCompat.h` incrementally (one wrapper, globally reused); writing separate `#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)` branches in multiple plugins is forbidden.
2. **The mouse/wheel/drop/plaintextedit parts of QtCompat.h depend on QtWidgets**: the AICore core (`core/AICore`, linking only Qt::Core + Qt::Gui) uses only the Core parts (regex/split/stringref/endl/variant/map); the plugin layer (linking QtWidgets) may use everything.
3. APIs **not covered by QtCompat.h** may use a local `#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)` branch (see the existing pattern in `app/ecvUIManager.cpp`), with a comment explaining the difference.
4. Implicitly-shared containers (`QString`/`QList`) whose Qt6 API changed (e.g. `QVector` → `QList` alias) should be wrapped via QtCompat rather than bare container migration.

### 14.6 Pre-Commit Cross-Platform Self-Check Commands

```bash
# macOS breaker: void* arithmetic/subscript (should only hit already-cast safe uses)
rg -n '\.data\s*[+\-]|->data\s*[+\-]' core/AICore/src plugins/core/Standard/q<task>/ | rg -v 'static_cast<const uint8_t|static_cast<const char'

# Windows breaker: bare __attribute__ (should all sit next to #if/#else/#elif guards)
rg -n '__attribute__' core/AICore/src plugins/core/Standard/q<task>/ | rg -v '#if|#else|#elif|#endif'

# Windows breaker: POSIX functions without _WIN32/_MSC_VER branches
rg -n '\b(strncasecmp|strcasecmp|strdup|usleep|localtime_r|gmtime_r|gettimeofday|mkstemp)\s*\(' core/AICore/src plugins/core/Standard/q<task>/

# redefining command-line macros
rg -n '^\s*#define\s+(_USE_MATH_DEFINES|__STDC_LIMIT_MACROS|NOMINMAX|_CRT_SECURE_NO_WARNINGS)' core/AICore/src plugins/core/Standard/q<task>/

# Qt5-only API direct calls (should go through QtCompat.h)
rg -n 'QRegExp|QString::SkipEmptyParts|QStringRef|QTextCodec|QTextStream::endl|QFontMetrics.*\.width\(|QWheelEvent.*->delta\(\)|QMouseEvent.*->pos\(\)|insertMulti\(' core/AICore/src plugins/core/Standard/q<task>/
```

### 14.7 Troubleshooting Notes

- **A flood of Windows errors is usually one root cause cascading** (one header × N translation units reporting the same line): locate the first root-cause file/line; don't fix entries one by one
- After a fix, IDE (clangd) errors may persist (deterministic case): when `compile_commands.json` was configured for a different platform than the current machine (e.g. build_app configured for Windows, local machine macOS), clangd necessarily misresolves headers. Verify: compare `rg -n '"platform"' build_app/compile_commands.json | head -1` with the current system; reconfigure locally to clear it.
- All three platform CIs must run: Linux is the most permissive and **cannot** be the only pass criterion

---

## 15. Checklist (for onboarding a new task)

When adding a new AICore task with a companion plugin, confirm each item:

- [ ] `include/aicore/<task>_capi.h` defines `aicore_<task>_abi_version`
- [ ] all output memory is freed via the single `aicore_<task>_free_buffer`
- [ ] long parameter lists (>6 inputs) are encapsulated in structs
- [ ] options use the builder pattern; every setter is a NULL no-op
- [ ] shutdown performs real cleanup (calls `purge_inactive_backend_leases`)
- [ ] inference/business-path logging uses `AICORE_LOG_*` macros (`fprintf(stderr)` only in the 5 exception categories of section 7: macro fallback / NDEBUG diagnostics / options-gated profile output / CLI tools / upstream legacy)
- [ ] ggml stays at v0.18.1; patches generated after replaying the existing manifest; other modules' contract tests all green
- [ ] "Try sample data" button (teal style) reuses ecvTestDataRepository (ObjectsDetection/Monstree/FriendsFaces/Image2Mesh, see section 12)
- [ ] **UI reuses `ecvAICoreUiHelper.h`** (setupTabLayout/setupFormGrid/makeLabel/makeSampleDataBtn/makeBrowseBtn/makeRuntimeRow/makeDbSection/setupProgressSection); no local duplicate helpers or magic pixels
- [ ] **main dialog layout is `SetNoConstraint` + first-show adjustSize** (DB expand/tab switch must not grow the window)
- [ ] **preview uses `ecvClickableImageLabel` + `setPreviewImage`**; DB input (`db://`) stores the full image in an item role and supports click-to-enlarge
- [ ] **video preview has a height cap** (`updatePreviewHeightCap` adaptive-branch max + scroll max + tab height clamp, preventing the feedback-loop infinite growth)
- [ ] all pixel sizes go through `ecvAICoreUi::dpiScaled()`; no bare hardcoding
- [ ] no getenv/setenv logic control (only `data_root_util.cpp` / `ggml_env_bridge.cpp`; verify: `bash core/AICore/tests/check_no_env_getenv.sh core/AICore/src` or `ctest -R test_no_env_getenv`)
- [ ] no third-party modules covering capabilities the repo already has (stb check: `rg -n 'STB_IMAGE_IMPLEMENTATION|#include [<"](stb_image|stb_image_write|stb_image_resize)' core/AICore/src` zero hits; comments don't count)
- [ ] includes use module-root absolute paths (no `../`, no bare filenames)
- [ ] naming consistent with AICore (snake_case functions, PascalCase types)
- [ ] zero-copy design: borrowed inputs, buffer reuse, no intermediate materialization
- [ ] weights uploaded once; host copies releasable; `keep_graph_buffers` set per real-time need
- [ ] CMakeLists.txt registers the task per the 4 steps of section 9 (`AICORE_<TASK>_SRC_DIR` + `file(GLOB ...)` + existence `FATAL_ERROR` + `add_library` source list + PRIVATE include dirs; no `add_subdirectory`)
- [ ] `tests/<task>/test_<task>_capi_contract.cpp` covers all public APIs
- [ ] `python3 core/AICore/tests/check_capi_coverage.py` >= 95%
- [ ] if ggml sources are modified, the patch goes into `3rdparty/ggml/patches/<subdir>/` and is registered in `manifest.yaml` (currently 14 patches, see section 8)
- [ ] non-ABI-compatible changes bump `aicore_<task>_abi_version`
- [ ] **cross-platform (section 14)**: no void* arithmetic/subscript (cast plane_view.data to `uint8_t*` first, size_t offsets)
- [ ] **cross-platform**: no bare `__attribute__` / `__declspec` (`#if defined(__GNUC__) || defined(__clang__)` / `#ifdef _WIN32` guards)
- [ ] **cross-platform**: POSIX functions and headers have platform branches (strncasecmp/strdup/usleep/localtime_r/getpid/mkdir/fdopen etc., see 14.3 replacement table)
- [ ] **cross-platform**: no re-`#define` of command-line macros (`_USE_MATH_DEFINES`/`__STDC_LIMIT_MACROS`/`NOMINMAX` etc.)
- [ ] **Qt compat (14.5)**: Qt5/Qt6 divergent APIs go through `QtCompat.h` (`QtCompatRegExp`/`QtCompat::SkipEmptyParts`/`qtCompatCodecForLocale`/`QtCompat::endl` etc.); no bare `#if QT_VERSION` branches in plugins
- [ ] **Qt compat**: newly discovered divergent APIs are incrementally added to `core/include/QtCompat.h` rather than handled locally
- [ ] **Qt compat**: avoid `QSet<T>(begin, end)` iterator-range construction (not available in Qt 5.12); use `QSet<T> s; for(auto& v : src) s.insert(v);`
- [ ] **Qt compat**: AICore core does not use the QtWidgets-dependent parts of QtCompat.h (mouse/wheel/drop/plaintextedit)
- [ ] **cross-platform**: 14.6 self-check commands run and all three platform CIs green (Linux passing is not completion)
- [ ] **cross-platform**: lambdas have explicit capture lists (MSVC rejects implicit capture of `constexpr` locals; use `[A]` or `[=]`)
- [ ] **cross-platform**: `/openmp` + `/openmp:experimental` together trigger MSVC D9025; strip the standard `/openmp` from the upstream INTERFACE (see the MSVC branch pattern in `AICore/CMakeLists.txt`)
- [ ] **plugin docs**: `plugins/core/Standard/<Plugin>/models/MODEL_CARD.md` created, listing every supported model (filename, size, quantization, recommended use)
- [ ] **plugin docs**: `README.md` references `models/MODEL_CARD.md` with the full model catalog
- [ ] **test registration**: new tests outside the AICore build tree must have a `LABELS` property and be excluded from `aicore-fast-tests` (`-LE "model|gpu|e2e|cvpluginapi"`)
