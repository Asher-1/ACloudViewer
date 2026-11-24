# qDA3 tests (legacy)

Depth whitebox tests have moved to **`core/AICore/tests/depth/whitebox/`** (enable
`AICore_BUILD_WHITEBOX_TESTS=ON`).

| Tier | CMake flag | Location |
|------|------------|----------|
| Public C API contract | `AICore_BUILD_TESTS=ON` | `core/AICore/tests/depth/test_depth_capi_*.cpp` |
| Fast unit (no GGUF) | same | `test_gguf_keys`, `test_path_util`, … |
| Depth whitebox | `AICore_BUILD_WHITEBOX_TESTS=ON` | `core/AICore/tests/depth/whitebox/` |
| Model / parity | `ctest -L model` | whitebox + `test_depth_capi_load`, gaussian parity, … |

The former `test_capi`, `test_capi_dense`, and `test_capi_da2` harnesses
are consolidated into `test_depth_capi_model_contract` under AICore. GoogleTest
now owns the DualDPT/mono/nested/DA2 output assertions and successful GLB/COLMAP
exports; model-free ABI/info checks remain in the existing C API suites.
Missing assets produce explicit GoogleTest skips (exit 77 when all cases skip).

The diagnostic `glb_parity_dump` / `colmap_parity_dump` sources now live in
`core/AICore/tests/depth/whitebox/` and build with the whitebox library. Their
stable executable names preserve the Python parity scripts. No plugin-side
private AICore include path or second CMake test entry remains here.

Model inputs: `AICORE_TEST_DEPTH_GGUF` (relative DualDPT),
`AICORE_TEST_DEPTH_GGUF_MONO`, `AICORE_TEST_DEPTH_GGUF_METRIC`,
`AICORE_TEST_DEPTH_GGUF_DA2`, and `AICORE_TEST_DEPTH_IMAGE`. Each optional
family is a separate GoogleTest case; a supplied but invalid asset fails.

Do not add new depth tests here — use AICore tests instead.

**Scripts:** `scripts/validate_gpu.sh` targets the upstream depth-anything.cpp repo layout
(`-DDA_GGML_CUDA`); for ACloudViewer use:

```bash
cmake .. -DAICore_ENABLED=ON -DAICore_USE_CUDA=ON \
         -DAICore_BUILD_TESTS=ON -DAICore_BUILD_WHITEBOX_TESTS=ON
ctest -LE model -L whitebox
```

Whitebox model fixtures use `AICORE_TEST_DEPTH_*` (see `core/AICore/README.md`).

## Unified Execution

Enabled suites register through the root
[tests module](../../../../../tests/README.md). From the repository root,
`cmake -DBUILD_DIR=build_app -DJOBS=4 -P tests/Run.cmake` builds prerequisites
and runs the common headless selection. Direct CTest examples above are
focused diagnostics after building their prerequisites.
