# AICore

Unified ggml inference core for CloudViewer. **One public include tree, one naming scheme.**

## Dependency boundary

Downstream code links only `AICore` and includes only `include/aicore/`.
`ggml` headers, targets, compile definitions, backend handles, and platform
branches are private implementation details.

```
COLMAP / pybind / qDA3 / qFreeSplatter / ccImage
                         |
                  AICore public ABI
                         |
          private ggml core + runtime backends
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
| Vulkan backend | Linux/Windows default | cross-vendor; shaders are embedded at build time |
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

- Linux/Windows: Auto uses Vulkan, then the pure ggml CPU backend. Vulkan only
  requires `glslc`/headers while building;
  users need a Vulkan-capable display driver, not the Vulkan SDK.
- macOS: Auto uses native Metal and CPU. CI also builds Vulkan against
  MoltenVK for explicit compatibility testing; Vulkan does not replace Metal
  in the automatic order. The partial-graph ggml BLAS backend stays disabled.
- SYCL and CUDA remain explicit developer devices; they do not participate in
  Auto selection or define the distributed runtime contract.
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
ICD smoke test. Windows and macOS builders use the fixed/checksummed LunarG SDK
installers under `util/`. Every CI package must contain `ggml-vulkan`, while
`libAICore` itself must have no hard Vulkan loader dependency.

Real model performance runs are manual through
`.github/workflows/aicore-vulkan-hardware.yml`. Linux runners use the labels
`aicore-vulkan` plus `intel`, `amd`, or `nvidia`; the macOS ARM64 runner uses
`aicore-vulkan`. Each runner provides these file-path environment variables:

```text
AICORE_TEST_DEPTH_GGUF
AICORE_TEST_DEPTH_IMAGE
AICORE_TEST_GAUSSIAN_GGUF
AICORE_TEST_GAUSSIAN_IMAGE_0
AICORE_TEST_GAUSSIAN_IMAGE_1
```

### Coverage matrix

| Area | Test | Tier |
|------|------|------|
| `data_root_util` | `test_data_root` | fast |
| depth GGUF keys | `test_gguf_keys` | fast |
| depth path / compute | `test_path_util`, `test_compute_mode` | fast |
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

## Extending

See `include/aicore/aicore.h`. New capability = `src/<cap>/` + `include/aicore/<cap>_capi.h` + `tests/<cap>/`.

Regenerate depth GGUF keys:

```bash
python3 plugins/core/Standard/qDA3/scripts/gen_gguf_keys_header.py
```
