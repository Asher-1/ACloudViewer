# AICore

Unified ggml inference core for CloudViewer. **One public include tree, one naming scheme.**

## Dependency boundary

Downstream code links only `AICore` and includes only `include/aicore/`.
`ggml` headers, targets, compile definitions, backend handles, and platform
branches are private implementation details.

### CMake linking (consumer)

```cmake
target_link_libraries(my_app_or_plugin PRIVATE AICore)
```

The `AICore` target is built with **`3rdparty_ggml` PRIVATE** — consumers do
**not** receive `GGML_INCLUDE_DIRS` or the `3rdparty_ggml` interface target.

For **link closure** only, `AICore` propagates on **`INTERFACE`**:

| Propagated | Purpose |
|------------|---------|
| `GGML_CORE_LINK_LIBS` | File paths to `libggml` + `libggml-base` (shared packaging) or core static archives (dev static ggml) — **link line only, no headers** |

**Direct C API executables** (AICore tests, `aicore_gguf_quantize`, `free_splatter-cli`)
also link **`AICore_capi_link`** when `GGML_BACKEND_DL` is enabled on Linux. That
INTERFACE target adds `-Wl,--allow-shlib-undefined` so `libAICore.so` can leave
backend symbols unresolved until `ggml_backend_load_all_from_path()`. It is **not**
propagated through `AICore` itself — doing so breaks Colmap/VTK executables that
link DSOs needing `crc32` while static `libz.a` is hidden via `--exclude-libs`.

Do **not** `#include <ggml.h>` from plugins, COLMAP, pybind, or CLIs. The stable
surface is `include/aicore/*.h` (C API) plus optional Qt helpers such as
`depth_image.h`.

Executables installed outside `bin/` (e.g. `bin/plugins/free_splatter-cli`) should
set **`INSTALL_RPATH` / `BUILD_RPATH` to `$ORIGIN/..`** so they find
`libAICore.so` and the ggml runtime bundle copied next to it under `bin/`.
Backend MODULE `.so` files are loaded from **libAICore's directory**, not from
the executable's directory. **Never copy `libggml*.so` into `bin/plugins/`** —
the app only loads `*_PLUGIN.{so,dll,dylib}` from that folder at startup.

```
COLMAP / pybind / qDA3 / qFreeSplatter / ccImage
                         |
           target_link_libraries(... AICore)
                         |
                  AICore public ABI (headers)
                         |
          PRIVATE 3rdparty_ggml (build + ggml headers)
          INTERFACE GGML_CORE_LINK_LIBS + DL link opts
                         |
          private ggml core + runtime backend MODULEs
```

The C ABI is the stable boundary. `depth_image.h` is a narrow Qt adapter kept
for the existing `ccImage` integration; new cross-language features belong in a
`*_capi.h` header. Exceptions, STL containers, ggml types, and backend-specific
types must not cross the C ABI.

## Runtime artifacts

| Artifact | Required | Policy |
|----------|----------|--------|
| `AICore` shared library | yes | the only library linked by consumers |
| `ggml` and `ggml-base` | yes | private, version-locked AICore runtime |
| `ggml-cpu` backend | yes | packaging fails if absent; guarantees CPU fallback |
| Vulkan backend | Linux/Windows default | cross-vendor; shaders are embedded at build time; **not built on macOS** |
| Metal backend | macOS default | Apple GPU path; CPU remains available |
| SYCL backend | opt-in | Intel GPU; requires a validated matching oneAPI runtime bundle |
| CUDA/OpenCL backend | developer opt-in | never required for a distributed AICore runtime |

ggml's `GGML_BACKEND_DL` requires shared ggml core libraries. This is why the
distribution contains private ggml runtime files instead of embedding a static
ggml copy into every consumer. The loader uses local scope and validates the
backend API version before registration. See the upstream
[ggml build configuration](https://github.com/ggml-org/ggml/blob/v0.17.0/CMakeLists.txt)
and [backend loader](https://github.com/ggml-org/ggml/blob/v0.17.0/src/ggml-backend-reg.cpp).

### Platform policy

- Linux/Windows: default Auto is **Vulkan → CPU**. With `-DAICore_USE_CUDA=ON`
  (CUDA backend built), Auto becomes **CUDA → Vulkan → CPU**. Vulkan build still
  requires `glslc`/headers; end users need a Vulkan-capable display driver, not
  the Vulkan SDK.
- macOS: Auto uses native **Metal → CPU** only. Vulkan is **unsupported** on
  macOS due to MoltenVK SPIR-V translation limitations that cause inference
  crashes; `-DAICore_USE_VULKAN=OFF` is the default.
- SYCL remains explicit-only (never in Auto).
- OpenCL: disabled by default. Upstream targets recent Adreno GPUs and its
  desktop operation coverage is too limited for AICore's default distribution.
- CUDA: `libAICore` never has a CUDA `DT_NEEDED`/import dependency. The optional
  CUDA backend uses the target system's CUDA runtime. A backend built against
  CUDA 11 cannot consume CUDA 12 runtime libraries by filename/ABI; publish a
  backend or wheel variant per supported CUDA major when both majors are needed.
  Release builds include native cubins plus PTX for the highest configured ISA
  so newer GPUs can JIT when the installed driver supports that PTX version.

NVIDIA documents minor compatibility within a CUDA major, driver backward
compatibility, and the PTX limitation separately in the
[CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
PyTorch follows the same major-runtime constraint by publishing distinct CPU,
CUDA 11.8, CUDA 12.x, and newer CUDA wheel variants; it is not one CUDA binary
that links against every locally installed toolkit. See its official
[binary matrix](https://docs.pytorch.org/get-started/previous-versions/).

### GGUF portability on macOS

GGUF stores typed tensors and metadata, not backend-specific executable code.
The same F16/Q8_0/Q4_K GGUF can be produced on Linux or Windows and loaded by
Metal on macOS when that tensor type and graph operation are supported. No
macOS-specific quantization/export is required; unsupported Metal operations
are scheduled to CPU. A separate macOS GGUF TODO is therefore intentionally not
created.

## Public API (`include/aicore/`)

| Header | Module | C API | C++ namespace |
|--------|--------|-------|---------------|
| `aicore/aicore.h` | Umbrella + extension contract | — | — |
| `aicore/export.h` | Export macros | `AICORE_CAPI` / `AICORE_CXX_API` | — |
| `aicore/backend_capi.h` | Runtime devices and backend warmup | `aicore_backend_*`, `aicore_device_*` | — |
| `aicore/depth_capi.h` | Multi-view depth | `aicore_depth_*` | `aicore::depth` |
| `aicore/gaussian_capi.h` | Image → 3D Gaussians | `aicore_gaussian_*` | `aicore::gaussian` |
| `aicore/lightglue_capi.h` | Sparse feature matching | `aicore_lightglue_*` | `aicore::lightglue` |
| `aicore/depth_image.h` | Qt `QImage` helpers | — | `aicore::depth::ImageDepth` |

No legacy aliases (`da_capi_*`, `fs_capi_*`, `DA3ImageDepth`, …).

## Layout

```
core/AICore/
  include/aicore/     ← sole public headers
  src/common/         aicore::locate_data_root, extract_model_dir
  src/depth/          aicore::depth
  src/gaussian/       aicore::gaussian
  src/lightglue/      aicore::lightglue
  tests/common/       shared data-path tests
  tests/depth/
  tests/gaussian/
```

## Tests

```bash
cmake -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ...
cmake --build build --target aicore-contract-tests
ctest -LE model    # fast tier (no GGUF)
ctest -L model     # needs AICORE_TEST_DEPTH_GGUF / AICORE_TEST_GAUSSIAN_GGUF
```

Set `AICore_BUILD_WHITEBOX_TESTS=ON` only for private implementation tests. CI
and packaging use the lighter public contract suite by default.

### Vulkan CI

Hosted Ubuntu 20.04/22.04/24.04 builds install Vulkan through
`util/install_deps_ubuntu.sh`; Mesa lavapipe supplies the deterministic software
ICD smoke test. Windows builders use the fixed/checksummed LunarG SDK
installers under `util/`. Linux/Windows CI packages must contain `ggml-vulkan`;
macOS packages use Metal only (no Vulkan backend). `libAICore` itself must have
no hard Vulkan loader dependency.

**ALIKED Vulkan (qLightGlue):** custom ggml-vulkan shaders are not edited in the
fetched ggml tree. They live in
`3rdparty/ggml/patches/aliked/0001-vulkan-aliked-custom-compute.patch` (plus
`0002` header install, `0003` `get_proc_address`) and apply at `ext_ggml`
configure time. With `AICore_USE_VULKAN=ON`, AICore defines `AICORE_VULKAN_ALIKED`
and resolves `ggml_vulkan_aliked_*` at runtime via
`src/aliked/vulkan_aliked_dispatch.cpp` (`ggml_backend_reg_get_proc_address` /
`dlsym`) — no link-time dependency on `libggml-vulkan.so`. Regenerate the main
patch from LightGlue-GGML after ggml changes:
`3rdparty/ggml/patches/export_aliked_patch.sh [path/to/third_party/ggml]`.

Real model performance runs are manual through
`.github/workflows/aicore-vulkan-hardware.yml`. Linux runners use the labels
`aicore-vulkan` plus `intel`, `amd`, or `nvidia`. Each runner provides these
file-path environment variables:

```text
AICORE_TEST_DEPTH_GGUF
AICORE_TEST_DEPTH_IMAGE
AICORE_TEST_DEPTH_BASELINE
AICORE_TEST_DEPTH_BASELINE_MV
AICORE_TEST_DEPTH_BASELINE_MV4
AICORE_TEST_DEPTH_BASELINE_NESTED
AICORE_TEST_DEPTH_BASELINE_GIANT
AICORE_TEST_DEPTH_BASELINE_DA2
AICORE_TEST_DEPTH_BASELINE_MONO
AICORE_TEST_DEPTH_BASELINE_NATIVE
AICORE_TEST_DEPTH_BASELINE_PREPROC
AICORE_TEST_DEPTH_BASELINE_RAY_POSE
AICORE_TEST_DEPTH_BASELINE_RAYS
AICORE_TEST_DEPTH_GGUF_DA2
AICORE_TEST_DEPTH_GGUF_MONO
AICORE_TEST_DEPTH_GGUF_GIANT
AICORE_TEST_DEPTH_GGUF_METRIC
AICORE_TEST_DEPTH_GGUF_ANYVIEW
AICORE_TEST_DEPTH_GGUF_AUX
AICORE_TEST_DEPTH_MONO_IMAGE
AICORE_TEST_DEPTH_PREPROC_IMAGE
AICORE_TEST_GAUSSIAN_GGUF
AICORE_TEST_GAUSSIAN_IMAGE_0
AICORE_TEST_GAUSSIAN_IMAGE_1
```

Legacy `DA_TEST_*` names are still accepted by `depth/whitebox/fixtures.hpp` helpers
when running old harness scripts locally.

### Coverage matrix

| Area | Test | Tier |
|------|------|------|
| `data_root_util` | `test_data_root` | fast |
| depth GGUF keys | `test_gguf_keys` | fast |
| depth path / compute | `test_path_util`, `test_compute_mode` | fast |
| depth whitebox (blocks/engine) | `core/AICore/tests/depth/whitebox/test_*` | whitebox / model |
| depth C API contract | `test_depth_capi_contract` | capi |
| depth Qt helper | `test_image_depth` | capi |
| depth load + JSON | `test_depth_capi_load` | model |
| gaussian loader | `test_loader` | fast |
| gaussian ingest / pose / graph | `test_image`, `test_pose`, `test_graph_blocks` | fast |
| gaussian path | `test_gaussian_path_util` | fast |
| gaussian C API contract | `test_gaussian_capi_contract` | capi |
| gaussian load + JSON | `test_gaussian_capi_load` | model |
| gaussian parity | `test_parity` | model |
| lightglue C API contract | `test_lightglue_capi_contract` | capi |
| no legacy exports | `test_no_legacy_symbols` | capi |

## CMake options

Public build switches use the **`AICore_*`** prefix. **All** of them — including
`AICore_ENABLED`, test toggles, backends, and packaging — are defined in
[`cmake/AICoreOptions.cmake`](../../cmake/AICoreOptions.cmake) and forwarded to
ggml's internal `GGML_*` names during configure.

| Option | Default | Role |
|--------|---------|------|
| `AICore_ENABLED` | OFF | Build `libAICore.so` (auto-enables `GGML_ENABLED`) |
| `AICore_BUILD_TESTS` | OFF | Public ABI/runtime contract tests (`ctest -L capi`) |
| `AICore_BUILD_WHITEBOX_TESTS` | OFF | Private implementation tests (requires `AICore_BUILD_TESTS`) |
| `AICore_USE_METAL` | Apple: ON, else OFF | Build Metal backend (macOS Auto default) |
| `AICore_USE_VULKAN` | Linux/Win: ON, macOS: OFF | Build Vulkan backend (configure fails if SDK/glslc missing) |
| `AICore_USE_CUDA` | OFF | Developer CUDA backend (not `BUILD_CUDA_MODULE`) |
| `AICore_USE_SYCL` | OFF | Developer Intel SYCL backend |
| `AICore_SYCL_USE_DNN` | ON | oneDNN kernels in SYCL backend (requires `AICore_USE_SYCL=ON`) |
| `AICore_USE_OPENCL` | OFF | Developer OpenCL backend (legacy/Adreno) |
| `AICore_OPENCL_TARGET_VERSION` | 200 | OpenCL API target: 120, 200, or 300 |
| `AICore_BUNDLE_CUDA_RUNTIME` | OFF | Redist CUDA runtime into installer (`lib/cuda-runtime/`) |
| `AICore_CPU_ALL_VARIANTS` | OFF | Build all ggml CPU ISA variants (`libggml-cpu-*.so`; compiler-adaptive; CI release/wheel default ON) |
| `AICore_*_ENABLED` | (auto) | Read-only: which backends were actually built |

**Adding a new option:** edit `cmake/AICoreOptions.cmake` (option + sync functions),
then update `BUILD.md`, `util/ci_utils.{sh,ps1}`, compile guides, and this section.

Direct `-DGGML_USE_*` / `-DGGML_ENABLED` are **ignored** (cleared from cache with a warning); use `AICore_*` only. After upgrading, run once: `cmake -U GGML_USE_METAL -U GGML_USE_VULKAN ...` or use a fresh build directory.

Compile-time macros (set by CMake, not user flags):

| Macro | When set | Role |
|-------|----------|------|
| `AICORE_BACKEND_DL` | Dynamic ggml backend modules | Default packaging |
| `AICORE_CUDA_ALIKED` | `AICore_CUDA_ENABLED` | ALIKED CUDA custom kernels (DCN, DKD, SDDH) |
| `AICORE_VULKAN_ALIKED` | `AICore_VULKAN_ENABLED` | ALIKED Vulkan custom compute via runtime dispatch |
| `AICORE_CUDA_STATIC_LINKED` | CUDA statically linked into libAICore | Non-DL dev builds |
| `AICORE_AUTO_INCLUDE_CUDA` | `AICore_CUDA_ENABLED` | CUDA in Auto fallback |
| `AICore_ENABLED` | Public | Consumer knows AICore is linked |

## Extending

See `include/aicore/aicore.h`. New capability = `src/<cap>/` + `include/aicore/<cap>_capi.h` + `tests/<cap>/`.

Regenerate depth GGUF keys:

```bash
python3 plugins/core/Standard/qDA3/scripts/gen_gguf_keys_header.py
```
