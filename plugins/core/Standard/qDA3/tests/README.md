# qDA3 tests (legacy)

Depth whitebox tests have moved to **`core/AICore/tests/depth/whitebox/`** (enable
`AICore_BUILD_WHITEBOX_TESTS=ON`).

| Tier | CMake flag | Location |
|------|------------|----------|
| Public C API contract | `AICore_BUILD_TESTS=ON` | `core/AICore/tests/depth/test_depth_capi_*.cpp` |
| Fast unit (no GGUF) | same | `test_gguf_keys`, `test_path_util`, … |
| Depth whitebox | `AICore_BUILD_WHITEBOX_TESTS=ON` | `core/AICore/tests/depth/whitebox/` |
| Model / parity | `ctest -L model` | whitebox + `test_depth_capi_load`, gaussian parity, … |

**Remaining here (legacy plugin C API harnesses, not wired by default):**

- `test_capi.cpp`
- `test_capi_dense.cpp`
- `test_capi_da2.cpp`
- `glb_parity_dump` / `colmap_parity_dump` (parity dump tools)

Do not add new depth tests here — use AICore tests instead.

**Scripts:** `scripts/validate_gpu.sh` targets the upstream depth-anything.cpp repo layout
(`-DDA_GGML_CUDA`); for ACloudViewer use:

```bash
cmake .. -DAICore_ENABLED=ON -DAICore_USE_CUDA=ON \
         -DAICore_BUILD_TESTS=ON -DAICore_BUILD_WHITEBOX_TESTS=ON
ctest -LE model -L whitebox
```

Whitebox model fixtures use `AICORE_TEST_DEPTH_*` (see `core/AICore/README.md`).
